//! The run-time state a tracing driver reads this VM through.
//!
//! The build lowered `run_frame` to jitcode and recorded which of the merge
//! point's operands are green and which red, with the kind each one's operand
//! actually had. This is the other half: the types a `JitDriver` needs to walk
//! that jitcode -- how the reds are handed to it, how it mints a symbolic value
//! for each, and which of them the closing jump carries.
//!
//! Nothing here restates the driver's shape. Every count, name and kind is read
//! back out of [`jitcodes::driver`], because the build already checked its
//! declaration against the portal graph's own operands; a second declaration
//! here would agree with that one only until one of the two was edited.
//!
//! The embedded jitcodes still carry function addresses the build could not
//! bind, and majit's walker invokes a residual call to obtain a concrete
//! shadow rather than only recording it. What keeps those out of compiled code
//! is the walk itself: it refuses such a call before making it and abandons
//! the trace, so no body that reaches one is ever compiled.

use majit_ir::{GcRef, OpRef, Type, Value};
use majit_metainterp::{
    GuardResumeFrame, JitCodeRuntime, JitCodeSym, JitDriverStaticData, JitState, TraceAction,
    TraceCtx, seed_bridge_virtualizable_boxes, trace_jitcode_at_resume_framestack,
};

/// The six register lists of a `jit_merge_point` op: green I/R/F then red I/R/F.
///
/// Same payload `setup_frame_from_merge_point` decodes. Grain's two reds are
/// both Ref, so the live frame is red-R[0] and the live Vm is red-R[1].
fn merge_point_slot_regs(
    jitcode: &majit_metainterp::JitCode,
    header_pc: usize,
) -> Option<[Vec<usize>; 6]> {
    let code = &jitcode.code;
    let mut cur = header_pc.checked_add(2)?;
    let mut slots: [Vec<usize>; 6] = std::array::from_fn(|_| Vec::new());
    for regs in slots.iter_mut() {
        let len = usize::from(*code.get(cur)?);
        cur = cur.checked_add(1)?;
        for _ in 0..len {
            regs.push(usize::from(*code.get(cur)?));
            cur = cur.checked_add(1)?;
        }
    }
    Some(slots)
}

/// Replace every Ref register that still holds `stale` with `live`.
#[cfg(test)]
fn forward_ref_bits(frames: &mut [GuardResumeFrame], stale: i64, live: i64) {
    if stale == live {
        return;
    }
    for frame in frames {
        for reg in &mut frame.regs {
            if reg.bank == Type::Ref && reg.value == stale {
                reg.value = live;
            }
        }
    }
}

use super::jitcodes;

/// What the grain driver did on this thread since [`reset_stats`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GrainJitStats {
    /// Dispatch-loop markers that reached the runtime consultation.
    pub merge_points_consulted: usize,
    /// Entry-door calls that kept interpreting without starting a trace.
    pub entry_door_interpret: usize,
    /// Entry-door calls declined because a trace was already active.
    pub entry_door_already_tracing: usize,
    /// Warm-loop decisions that opened a trace.
    pub traces_started: usize,
    /// Traces the metainterpreter abandoned before compilation.
    pub traces_aborted: usize,
    /// Loop traces successfully compiled.
    pub loops_compiled: usize,
    /// Operations appended by portal walks, before optimization.
    pub ops_recorded: usize,
    /// Largest active trace observed after one portal walk.
    pub max_trace_ops: usize,
    /// Calls that reached majit's immediately-before-backend-entry hook.
    pub compiled_entries: usize,
    /// Compiled artifacts rejected by this state's compatibility gate.
    pub compiled_entries_refused: usize,
    /// Walks stopped before an unbound symbolic residual callee was invoked.
    pub symbolic_residual_aborts: usize,
    /// Consultations answered without asking the driver, because `pc` had
    /// advanced on its own and no trace was open. The door is what upstream
    /// reaches from `can_enter_jit`; these are the steps that are not back
    /// edges, and each one costs a compare rather than four allocations.
    pub forward_steps_skipped: usize,
    /// Nested consultations skipped instead of panicking on a TLS borrow.
    pub reentrant_consultations_declined: usize,
    /// Consultations skipped because another thread owns the process's driver.
    pub non_owner_consultations_declined: usize,
    /// Majit's abort-reason counter deltas for the observation window.
    pub abort_reasons: String,
}

/// Snapshot this thread's grain-JIT counters.
pub fn stats() -> GrainJitStats {
    super::jit::stats()
}

/// Start a fresh observation window for this thread's grain-JIT counters.
pub fn reset_stats() {
    super::jit::reset_stats();
}

/// Majit's own warm-up census, as `label=count` pairs, non-zero slots only.
///
/// Process-global and cumulative -- the counters live in majit, not here, so
/// [`reset_stats`] does not scope them. What it answers is which of the
/// driver's doors refused: `mst_entered` beside `mst_sync_before_false` and
/// `mst_live_values_mismatch` says whether a trace was turned away before it
/// could start, which no counter on this side can distinguish from a warm
/// threshold that was never crossed.
#[must_use]
pub fn majit_diag_summary() -> String {
    majit_metainterp::mc_diag_summary()
        .split_whitespace()
        .filter(|pair| !pair.ends_with("=0"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The compiled driver, held for the life of the process.
///
/// [`JitDriverStaticData::new`] borrows the green and red names, so they have
/// to outlive the descriptor rather than the call that builds it.
fn driver() -> &'static majit_translate::CompiledJitDriver {
    static DRIVER: std::sync::OnceLock<majit_translate::CompiledJitDriver> =
        std::sync::OnceLock::new();
    DRIVER.get_or_init(|| {
        jitcodes::driver().expect(
            "the build lowered no JIT driver; a state describing one has nothing to describe",
        )
    })
}

/// The kind of each red, in the order the merge point passes them.
///
/// This is the list [`JitDriver::live_values_match_descriptor`] checks the live
/// values against. Its default in [`JitState`] types every red `Int`, which for
/// this VM would misdescribe both reference reds and make the driver decline to
/// trace without saying so.
pub fn red_kinds() -> &'static [Type] {
    &driver().red_args_types
}

/// The portal descriptor, built from what the lowering recorded.
///
/// `greens[0]` is the pc: `back_edge_structured`'s resume arm reads
/// `args.green_int.first()`, so a declaration that put anything else first
/// would resume at the wrong instruction.
pub fn grain_driver_descriptor() -> JitDriverStaticData {
    let driver = driver();
    assert_eq!(
        driver.greens.first().map(String::as_str),
        Some("pc"),
        "the first green is the one a back edge resumes at",
    );
    let pair = |names: &'static [String], kinds: &'static [Type]| -> Vec<(&'static str, Type)> {
        assert_eq!(
            names.len(),
            kinds.len(),
            "every merge-point operand is named and kinded, or neither",
        );
        names
            .iter()
            .map(String::as_str)
            .zip(kinds.iter().copied())
            .collect()
    };
    JitDriverStaticData::new(
        pair(&driver.greens, &driver.green_args_spec),
        pair(&driver.reds, &driver.red_args_types),
    )
}

/// The shape of the code being traced, captured when tracing starts.
///
/// Records the compile-time red shape. Live values are republished on
/// [`GrainJitState`] for every consultation; a reused artifact must not read
/// the frame/VM addresses captured when its meta was built.
#[derive(Clone)]
pub struct GrainMeta {
    /// The merge point's own offset in the portal body.
    pub header_pc: usize,
    /// One raw word per red, in declaration order.
    pub reds: Vec<i64>,
}

/// One symbolic value per red, in the order `create_sym` numbers them.
pub struct GrainSym {
    /// The merge point's offset in the portal body, which the tracer closes
    /// the loop back to.
    pub header_pc: usize,
    /// One input arg per red, minted in declaration order and typed from the
    /// build's own record of the operand kinds.
    pub reds: Vec<OpRef>,
}

impl JitCodeSym for GrainSym {
    fn total_slots(&self) -> usize {
        // The reds are merge-point operands, not slots of a state field bank.
        0
    }

    fn loop_header_pc(&self) -> usize {
        self.header_pc
    }

    fn loop_carried_boxes(&self, _vable_boxes: &[(OpRef, Type)]) -> Option<Vec<(OpRef, Type)>> {
        Some(
            self.reds
                .iter()
                .copied()
                .zip(red_kinds().iter().copied())
                .collect(),
        )
    }
}

/// This VM, as a tracing driver reads it.
#[derive(Default)]
pub struct GrainJitState {
    /// What the last [`JitState::restore`] was handed.
    ///
    /// The merge point drains this and applies it to the live frame while it
    /// still holds the frame's mutable borrow, which is the only place the
    /// borrow exists; a deoptimization path reaches `restore` with no way to
    /// reach the frame itself.
    pub last_restored: Vec<i64>,
    /// The red image of the consultation now in progress.
    ///
    /// A `Meta` describes the SHAPE a compiled artifact was built against and
    /// is stored with that artifact, so its own copy of the reds is as old as
    /// the compile. The values an entry passes have to be this consultation's,
    /// and [`JitState::extract_live`] is handed only `&self` and that stored
    /// meta -- so the live image lives here, republished by the merge point
    /// before it consults the door.
    reds: Vec<i64>,
    /// The live frame's own fields, in the order the virtualizable
    /// declaration below names them.
    ///
    /// Published beside the reds for the same reason, and read off the frame
    /// itself rather than through the declared byte offsets: this module
    /// forbids `unsafe`, and the frame is in hand where the merge point
    /// publishes.
    vable_statics: Vec<i64>,
}

/// Bind vinfo so `GETFIELD_VABLE` has field descrs when the resume stream
/// declined. Empty boxes: the getfield then loads through the live frame
/// rather than aborting the walk for a missing descr.
fn seed_grain_vable_info_only(
    ctx: &mut TraceCtx,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
) {
    ctx.set_virtualizable_boxes_with_info(Vec::new(), Vec::new(), info, &[]);
}

impl GrainJitState {
    pub(super) fn take_restored(&mut self) -> Vec<i64> {
        core::mem::take(&mut self.last_restored)
    }

    /// Publish the red image and the frame fields the merge point was handed.
    ///
    /// Overwrites in place: this runs on every consultation, and both lengths
    /// are fixed by the driver declaration, which does not change.
    pub(super) fn publish_live(&mut self, env: &[i64], vable_statics: &[i64]) {
        self.reds.clear();
        self.reds.extend_from_slice(env);
        self.vable_statics.clear();
        self.vable_statics.extend_from_slice(vable_statics);
    }

    /// The live frame, as the virtualizable identity red names it.
    fn live_frame_ptr(&self) -> Option<*mut u8> {
        self.reds.first().map(|frame| *frame as usize as *mut u8)
    }
}

#[cfg(all(test, rhai_grain_jit_tables))]
mod tests {
    use super::*;

    #[test]
    fn live_values_come_from_this_entry_not_the_compiled_artifacts_meta() {
        let mut state = GrainJitState::default();
        let first = [0x1000, 0x2000];
        state.publish_live(&first, &[]);
        let meta = state.build_meta(
            jitcodes::portal_merge_point_offset().expect("the lowered portal has a merge point"),
            &first,
        );
        assert_eq!(state.extract_live(&meta), first);

        let next = [0x3000, 0x4000];
        state.publish_live(&next, &[]);
        assert_eq!(state.extract_live(&meta), next);
        assert_eq!(meta.reds, first);
        let values = state.extract_live_values(&meta);
        assert_eq!(
            values.iter().map(Value::get_type).collect::<Vec<_>>(),
            red_kinds()
        );
    }

    #[test]
    fn merge_point_red_r_names_two_ref_reds() {
        let Some(header_pc) = jitcodes::portal_merge_point_offset() else {
            return;
        };
        let Some(portal_index) = jitcodes::portal_index() else {
            return;
        };
        let Some(portal) = jitcodes::all().into_iter().nth(portal_index) else {
            return;
        };
        let Some(slots) = merge_point_slot_regs(&portal, header_pc) else {
            panic!("the portal merge point must decode");
        };
        assert!(
            slots[4].len() >= 2,
            "grain's two reds are both Ref: {:?}",
            slots[4]
        );
    }

    #[test]
    fn forward_ref_bits_replaces_only_matching_shadows() {
        use majit_metainterp::GuardResumeReg;

        let Some(header_pc) = jitcodes::portal_merge_point_offset() else {
            return;
        };
        let Some(portal_index) = jitcodes::portal_index() else {
            return;
        };
        let Some(portal) = jitcodes::all().into_iter().nth(portal_index) else {
            return;
        };
        let mk = |index, value| GuardResumeReg {
            bank: Type::Ref,
            index,
            opref: OpRef::input_arg_typed(index, Type::Ref),
            value,
        };
        let stale = 0xDEAD_0000;
        let live = 0x2222_0000;
        let unrelated = 0xBEEF_0000;
        let mut frames = vec![GuardResumeFrame {
            jitcode: portal,
            pc: header_pc,
            regs: vec![mk(0, stale), mk(1, unrelated)],
            result_slot: None,
            sub_idx: None,
        }];
        forward_ref_bits(&mut frames, stale, live);
        assert_eq!(frames[0].regs[0].value, live);
        assert_eq!(frames[0].regs[1].value, unrelated);
    }

    #[test]
    fn rebind_bridge_reds_rewrites_only_the_reserved_vm_slot() {
        use majit_metainterp::GuardResumeReg;

        let Some(header_pc) = jitcodes::portal_merge_point_offset() else {
            return;
        };
        let Some(portal_index) = jitcodes::portal_index() else {
            return;
        };
        let Some(portal) = jitcodes::all().into_iter().nth(portal_index) else {
            return;
        };
        let Some(slots) = merge_point_slot_regs(&portal, header_pc) else {
            panic!("the portal merge point must decode");
        };
        let vm_reg = slots[4][1] as u32;
        let mut state = GrainJitState::default();
        state.publish_live(&[0x1111_0000, 0x2222_0000], &[]);
        let stale = 0xDEAD_0000;
        let other = 0xBEEF_0000;
        let mk = |index, value| GuardResumeReg {
            bank: Type::Ref,
            index,
            opref: OpRef::input_arg_typed(index, Type::Ref),
            value,
        };
        let mut frames = vec![GuardResumeFrame {
            jitcode: portal,
            pc: header_pc,
            regs: vec![
                mk(vm_reg, stale),
                mk(vm_reg.wrapping_add(3), stale),
                mk(7, other),
            ],
            result_slot: None,
            sub_idx: None,
        }];
        state.rebind_bridge_reds(&mut frames);
        assert_eq!(frames[0].regs[0].value, 0x2222_0000);
        assert_eq!(
            frames[0].regs[0].opref,
            OpRef::ConstPtr(GcRef(0x2222_0000)),
            "without a live-Vm failarg the reserved slot is a Const of the live bits",
        );
        assert_eq!(
            frames[0].regs[1].value, stale,
            "only the reserved vm index is rewritten, not every copy of its bits",
        );
        assert_eq!(frames[0].regs[2].value, other);
    }

    #[test]
    fn rebind_bridge_reds_keeps_the_live_vm_inputarg() {
        use majit_metainterp::GuardResumeReg;

        let Some(header_pc) = jitcodes::portal_merge_point_offset() else {
            return;
        };
        let Some(portal_index) = jitcodes::portal_index() else {
            return;
        };
        let Some(portal) = jitcodes::all().into_iter().nth(portal_index) else {
            return;
        };
        let Some(slots) = merge_point_slot_regs(&portal, header_pc) else {
            panic!("the portal merge point must decode");
        };
        let vm_reg = slots[4][1] as u32;
        let mut state = GrainJitState::default();
        let live = 0x2222_0000;
        state.publish_live(&[0x1111_0000, live], &[]);
        let stale = 0xDEAD_0000;
        let vm_box = OpRef::input_arg_typed(3, Type::Ref);
        let mk = |index, opref, value| GuardResumeReg {
            bank: Type::Ref,
            index,
            opref,
            value,
        };
        let mut frames = vec![GuardResumeFrame {
            jitcode: portal,
            pc: header_pc,
            regs: vec![
                mk(vm_reg, OpRef::input_arg_typed(vm_reg, Type::Ref), stale),
                mk(4, vm_box, live),
            ],
            result_slot: None,
            sub_idx: None,
        }];
        state.rebind_bridge_reds(&mut frames);
        assert_eq!(frames[0].regs[0].value, live);
        assert_eq!(
            frames[0].regs[0].opref, vm_box,
            "the reserved slot takes the failarg that already names the live Vm",
        );
    }
}

impl JitState for GrainJitState {
    type Meta = GrainMeta;
    type Sym = GrainSym;
    /// The red image, straight: one raw word per red in declaration order.
    type Env = [i64];

    fn build_meta(&self, header_pc: usize, env: &Self::Env) -> Self::Meta {
        assert_eq!(
            env.len(),
            red_kinds().len(),
            "the merge point passes one value per declared red",
        );
        GrainMeta {
            header_pc,
            reds: env.to_vec(),
        }
    }

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64> {
        let _ = meta;
        self.reds.clone()
    }

    fn live_value_types(&self, _meta: &Self::Meta) -> Vec<Type> {
        red_kinds().to_vec()
    }

    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym {
        let _ = meta;
        GrainSym {
            header_pc,
            reds: red_kinds()
                .iter()
                .enumerate()
                .map(|(k, tp)| OpRef::input_arg_typed(k as u32, *tp))
                .collect(),
        }
    }

    /// Rebuild the guard's frame sections through the same resume-numbering
    /// decoder as generated `#[jit_interp]` states. Each section is split by
    /// its owning jitcode's liveness rather than treating the remainder of the
    /// stream as one frame.
    fn rebuild_from_resumedata(
        _meta: &mut Self::Meta,
        fail_arg_types: &[Type],
        storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    ) -> Option<majit_metainterp::ResumeDataResult> {
        let storage = storage?;
        let frame_value_count = majit_ir::resumedata::get_frame_value_count_fn();
        let frame_value_count_ref: Option<&dyn Fn(i32, i32) -> usize> = frame_value_count
            .as_ref()
            .map(|callback| callback as &dyn Fn(i32, i32) -> usize);
        let (num_failargs, virtualizable_values, virtualref_values, frames) =
            majit_ir::resumedata::rebuild_from_numbering(
                storage.rd_numb.as_ref(),
                storage.rd_consts(),
                fail_arg_types,
                frame_value_count_ref,
                storage.rd_virtuals.len(),
            );
        if frames.is_empty() {
            return None;
        }
        Some(majit_metainterp::ResumeDataResult {
            frames,
            virtualizable_values,
            virtualref_values,
            storage: Some(storage.clone()),
            num_failargs,
            fail_arg_types: fail_arg_types.to_vec(),
        })
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // The reds are the live frame and VM, and the merge point passes both
        // afresh on every consultation, so no property of them can go stale
        // between the compile and an entry. What the artifact does fix is how
        // many live values its entry expects, and a build whose driver
        // declaration changed under an artifact compiled against the previous
        // one would enter it with the wrong count.
        //
        // A symbolic residual target the build left unbound cannot appear in a
        // compiled body: the walk that recorded it refuses such a call before
        // making it, which abandons the trace, so a trace that closed and
        // compiled reached none.
        let compatible = meta.reds.len() == red_kinds().len();
        if !compatible {
            super::jit::record_compiled_entry_refusal();
        }
        compatible
    }

    fn virtualizable_heap_ptr(
        &self,
        meta: &Self::Meta,
        virtualizable: &str,
        _info: &majit_metainterp::virtualizable::VirtualizableInfo,
    ) -> Option<*mut u8> {
        let _ = meta;
        (virtualizable == "frame")
            .then(|| self.live_frame_ptr())
            .flatten()
    }

    /// The frame's own fields, read out of the live frame for a warm entry.
    ///
    /// `warmstate.py:482-511` hands the assembler the virtualizable itself and
    /// lets the compiled code read through it; majit's entry passes the fields
    /// as scalar inputargs instead, so the compiled loop's arity is the reds
    /// plus these. The trace carries them across its back edge
    /// ([`JitState::collect_jump_args_with_boxes`]), so an entry that could not
    /// supply them would be entering with fewer arguments than the artifact
    /// declares -- which is what the driver declines on.
    ///
    /// No array fields are declared, so `arrays` is emptied rather than
    /// resized.
    fn export_virtualizable_boxes_into(
        &self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: &majit_metainterp::virtualizable::VirtualizableInfo,
        statics: &mut Vec<i64>,
        arrays: &mut Vec<Vec<i64>>,
    ) -> bool {
        let _ = meta;
        if virtualizable != "frame" || self.vable_statics.len() != info.static_fields.len() {
            return false;
        }
        statics.clear();
        statics.extend_from_slice(&self.vable_statics);
        arrays.clear();
        true
    }

    #[allow(non_snake_case)]
    fn __build_virtualizable_info()
    -> Option<std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>> {
        use super::GrainFrame;
        use majit_metainterp::virtualizable::VirtualizableInfo;

        // `GrainFrame` is a stack-resident interpreter frame, so it has no
        // heap force token.  Its identity is red #0 and ref-bank input #0.
        let mut info = VirtualizableInfo::without_vable_token();
        info.name = "frame".to_string();
        info.identity_live_index = Some(0);
        info.identity_ref_bank_index = Some(0);
        info.add_field(
            "scope",
            Type::Ref,
            core::mem::offset_of!(GrainFrame<'static, 'static>, scope),
        );
        info.add_field(
            "base",
            Type::Int,
            core::mem::offset_of!(GrainFrame<'static, 'static>, base),
        );
        info.add_field(
            "reached",
            Type::Int,
            core::mem::offset_of!(GrainFrame<'static, 'static>, reached),
        );
        info.add_field(
            "stack_base",
            Type::Int,
            core::mem::offset_of!(GrainFrame<'static, 'static>, stack_base),
        );
        info.add_field(
            "jit_resume_pc_plus_one",
            Type::Int,
            core::mem::offset_of!(GrainFrame<'static, 'static>, jit_resume_pc_plus_one),
        );
        Some(
            info.finalize_arc(majit_ir::descr::make_size_descr(core::mem::size_of::<
                GrainFrame<'static, 'static>,
            >())),
        )
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        let _ = meta;
        self.last_restored = values.to_vec();
    }

    fn restore_values(&mut self, meta: &Self::Meta, values: &[Value]) {
        let _ = meta;
        self.last_restored = values
            .iter()
            .map(|value| match value {
                Value::Int(value) => *value,
                Value::Ref(value) => value.as_usize() as i64,
                Value::Float(value) => value.to_bits() as i64,
                Value::Void => 0,
            })
            .collect();
    }

    /// Grain has no exception object to re-raise. Returning `None` makes
    /// the driver set `single_pass_finish` and treat the portal as done.
    /// `usize::MAX` is the same LeaveFrame sentinel the exhaust path
    /// already handles: no replay of aborted tails, and later merge
    /// points can still enter compiled code.
    fn deliver_blackhole_exception(&mut self, _exc: GcRef) -> Option<usize> {
        Some(usize::MAX)
    }

    /// Bind the frame virtualizable onto a guard-resume bridge.
    ///
    /// The generated `#[jit_interp]` `setup_bridge_sym` only calls
    /// `seed_bridge_virtualizable_boxes` when the state declares a virt
    /// array. Grain's vable is five static fields and no arrays, so that
    /// arm is empty and `GETFIELD_VABLE` aborts with no `VirtualizableInfo`
    /// on the bridge ctx. The main loop already ran
    /// `initialize_virtualizable`; a bridge has to re-bind the same
    /// shape or every mid-loop branch deopts.
    fn setup_bridge_sym(
        _sym: &mut Self::Sym,
        ctx: &mut TraceCtx,
        resume_data: &majit_metainterp::ResumeDataResult,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        fail_values: &[i64],
        fail_types: &[Type],
        executing: Option<&dyn majit_metainterp::resume::BlackholeAllocator>,
    ) {
        let Some(info) = Self::__build_virtualizable_info() else {
            return;
        };
        let bridge_virtual_count = rd_virtuals.map_or(0, |v| v.len());
        let mut cache = match executing {
            Some(allocator) => majit_metainterp::BridgeVirtualCache::executing(
                bridge_virtual_count,
                majit_metainterp::default_bridge_array_descr,
                allocator,
                fail_values,
                fail_types,
            ),
            None => majit_metainterp::BridgeVirtualCache::new(
                bridge_virtual_count,
                majit_metainterp::default_bridge_array_descr,
            ),
        };
        if !majit_metainterp::replay_pending_fields(ctx, resume_data, rd_virtuals, &mut cache) {
            ctx.mark_bridge_replay_incomplete();
        }
        let seeded = seed_bridge_virtualizable_boxes(
            ctx,
            &info,
            rd_virtuals,
            resume_data,
            &mut cache,
            fail_values,
        );
        if std::env::var_os("MAJIT_BRIDGE_DEBUG").is_some() {
            eprintln!(
                "[bridgeB] grain vable seed={seeded} stream={}",
                resume_data.virtualizable_values.len(),
            );
        }
        if seeded {
            return;
        }
        seed_grain_vable_info_only(ctx, &info);
    }

    /// Generated `setup_bridge_sym` rebinds a missing identity register
    /// from the live state. The LLBC portal now reserves merge-point
    /// reds the way `ref_identity_base` reserves `#[jit_interp]`
    /// identity slots, so red-R[1] is the Vm at every guard. Rewrite
    /// only that reserved index — not every copy of its bits, which
    /// is what SIGSEGV'd when the slot was still reusable.
    fn rebind_bridge_reds(&self, frames: &mut [GuardResumeFrame]) {
        let Some(live_vm) = self
            .reds
            .iter()
            .zip(red_kinds())
            .filter(|(_, kind)| **kind == Type::Ref)
            .map(|(bits, _)| *bits)
            .nth(1)
        else {
            return;
        };
        let Some(root) = frames.first() else {
            return;
        };
        let Some(header_pc) = jitcodes::portal_merge_point_offset() else {
            return;
        };
        let Some(slots) = merge_point_slot_regs(&root.jitcode, header_pc) else {
            return;
        };
        let Some(&vm_reg) = slots[4].get(1) else {
            return;
        };
        if std::env::var_os("MAJIT_BRIDGE_DEBUG").is_some() {
            eprintln!(
                "[bridgeB] merge slots gI={:?} gR={:?} gF={:?} rI={:?} rR={:?} rF={:?} \
                 live_reds={:x?} vm_reg={vm_reg}",
                slots[0], slots[1], slots[2], slots[3], slots[4], slots[5], self.reds,
            );
        }
        let Some(slot) = root
            .regs
            .iter()
            .find(|resume| resume.bank == Type::Ref && resume.index as usize == vm_reg)
        else {
            return;
        };
        let stale = slot.value;
        if stale == live_vm && !slot.opref.is_constant() {
            return;
        }
        if self.reds.first() == Some(&stale) {
            return;
        }
        if std::env::var_os("MAJIT_BRIDGE_DEBUG").is_some() {
            eprint!("[bridgeB] grain rebind vm reg={vm_reg} {stale:#x} -> {live_vm:#x} regs");
            for resume in &root.regs {
                eprint!(" {:?}[{}]={:#x}", resume.bank, resume.index, resume.value);
            }
            eprintln!();
        }
        // The reserved slot may hold a reused colour (a Scope failarg
        // at PC 6103) whose bits are not the Vm. Folding the live
        // address to `ConstPtr` compiles `Call*(this_eval's stack)` —
        // dead on the next `Vm::new`. Prefer a failarg that already
        // names the live Vm (the loop's red-R[1] InputArg).
        let vm_opref = root
            .regs
            .iter()
            .find_map(|resume| {
                (resume.bank == Type::Ref && resume.value == live_vm && !resume.opref.is_constant())
                    .then_some(resume.opref)
            })
            .unwrap_or(OpRef::ConstPtr(GcRef(live_vm as usize)));
        if std::env::var_os("MAJIT_BRIDGE_DEBUG").is_some() {
            eprintln!(
                "[bridgeB] rebind-opref vm_reg={vm_reg} stale={stale:#x} live={live_vm:#x} \
                 opref={vm_opref:?} const={}",
                vm_opref.is_constant(),
            );
        }
        for frame in frames {
            for reg in &mut frame.regs {
                if reg.bank == Type::Ref && reg.index as usize == vm_reg {
                    reg.value = live_vm;
                    reg.opref = vm_opref;
                }
            }
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.reds.clone()
    }

    fn collect_jump_args_with_boxes(sym: &Self::Sym, boxes: &[(OpRef, Type)]) -> Vec<OpRef> {
        // `pyjitpl.py reached_loop_header`: `live_arg_boxes +=
        // self.virtualizable_boxes` then `.pop()`. The frame's own fields are
        // carried across the back edge like any other loop-carried value; the
        // identity the list ends with is not, because the frame red already
        // names it.
        let mut args = Self::collect_jump_args(sym);
        if let Some((_identity, fields)) = boxes.split_last() {
            args.extend(fields.iter().map(|(opref, _)| *opref));
        }
        args
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = meta;
        sym.reds.len() == red_kinds().len()
    }

    /// `pyjitpl.py handle_guard_failure` → `rebuild_state_after_failure`
    /// → `setup_resume_at_op` then `interpret()`. Walk from the failed
    /// guard so the other arm is recorded. The default `None` makes the
    /// driver abort and resume at the loop header.
    fn trace_from_guard_resume_position<R: JitCodeRuntime>(
        ctx: &mut TraceCtx,
        sym: &mut Self::Sym,
        frames: &[GuardResumeFrame],
        outer_program_pc: usize,
        runtime: &R,
    ) -> Option<TraceAction> {
        majit_metainterp::set_bridge_walking(true);
        let action =
            trace_jitcode_at_resume_framestack(ctx, sym, frames, outer_program_pc, runtime);
        majit_metainterp::set_bridge_walking(false);
        Some(action)
    }
}
