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

use majit_ir::{OpRef, Type, Value};
use majit_metainterp::{JitCodeSym, JitDriverStaticData, JitState};

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
/// Carries the red image itself because [`JitState::extract_live`] is handed
/// only the meta -- the live values have to survive from `build_meta` to there.
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

    fn fail_args(&self) -> Option<Vec<OpRef>> {
        Some(self.reds.clone())
    }

    fn fail_args_types(&self) -> Option<Vec<Type>> {
        Some(red_kinds().to_vec())
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
    fn __build_virtualizable_info(
    ) -> Option<std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>> {
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
}
