//! The tracing-JIT merge point.
//!
//! A meta-tracing JIT does not read this VM's instructions; it traces the
//! *interpreter* running them, and the merge point is where the interpreter
//! tells the tracer that one iteration of its dispatch loop has begun and what
//! the machine state is at that moment. Everything the tracer needs to decide
//! "have I been here before" is in the arguments.
//!
//! The arguments split two ways. `pc`, `program_identity` and `program` are
//! *green*: constant for any one trace, so the tracer specialises on their
//! values and a loop that comes back to the same instruction of the same
//! program is the same loop.  The identity prevents a dropped stack-local
//! program's address from aliasing the next program allocated in that slot.
//! The rest are *red*: they vary per iteration and the trace carries them as
//! live values. Neither the split nor the order is inferable from the source,
//! so a consumer that lowers this VM declares both, and the declaration and
//! this signature have to agree position for position.
//!
//! The untranslated body also consults majit's warm state. The lowering still
//! replaces the call itself with the merge-point opcode, which is why the type
//! name, signature and `inline(never)` remain part of the build contract.

use std::cell::{Cell, RefCell};
use std::sync::{Arc, OnceLock};

use majit_ir::{GreenKey, GreenType};
use majit_metainterp::{JitDriver, TraceAction};

use super::{jit_state, jitcodes};
use super::{GrainFrame, Vm};
use crate::grain::bytecode::AssignOp;
use crate::grain::Program;
use crate::types::dynamic::Union;
use crate::{Dynamic, Position, RhaiResultOf, Scope};

/// Write the scalar payload of an existing integer `Dynamic`.
///
/// The MAJIT frontend lowers this named boundary to a `setfield_gc_i` on
/// `Union::Int::__pos_0`.  Keeping the target variant in place preserves its
/// tag/access fields and avoids representing Rust's `&mut INT` as a JIT ref.
#[inline(always)]
pub(super) fn dynamic_store_int(target: &mut Dynamic, value: crate::INT) {
    let Union::Int(held, ..) = &mut target.0 else {
        unreachable!("dynamic_store_int target changed variant")
    };
    *held = value;
}

/// Float twin of [`dynamic_store_int`].
#[cfg(not(feature = "no_float"))]
#[inline(always)]
pub(super) fn dynamic_store_float(target: &mut Dynamic, value: crate::FLOAT) {
    let Union::Float(held, ..) = &mut target.0 else {
        unreachable!("dynamic_store_float target changed variant")
    };
    **held = value;
}

fn position_bits(position: Position) -> i64 {
    let line = position.line().unwrap_or(0) as u16;
    let column = position.position().unwrap_or(0) as u16;
    i64::from((u32::from(line) << 16) | u32::from(column))
}

fn position_from_bits(bits: i64) -> Position {
    let bits = bits as u32;
    let line = (bits >> 16) as u16;
    if line == 0 {
        Position::NONE
    } else {
        Position::new(line, bits as u16)
    }
}

/// Read one bytecode byte across the residual-call ABI.
///
/// `Program::code()` returns a Rust slice, whose `(data, len)` fat pointer
/// cannot be carried in one majit register.  Keep that representation inside
/// ordinary Rust and expose only scalar results to the trace, like PyPy's
/// elidable bytecode accessors.  Returning `-1` spells `None` without putting
/// `Option<u8>`'s enum layout on the ABI.
#[majit_macros::elidable_cannot_raise]
pub(super) extern "C" fn code_byte(program: &Program<'_>, at: usize) -> i64 {
    program.code().get(at).map_or(-1, |byte| i64::from(*byte))
}

/// Return the verified instruction width, or `-1` for an invalid/truncated
/// instruction.  See [`code_byte`] for why this is an opaque scalar helper.
#[majit_macros::elidable_cannot_raise]
pub(super) extern "C" fn code_width(program: &Program<'_>, at: usize) -> i64 {
    crate::grain::bytecode::code::width(program.code(), at).map_or(-1, |n| n as i64)
}

/// Read a little-endian `u16` operand, or `-1` when it is truncated.
#[majit_macros::elidable_cannot_raise]
pub(super) extern "C" fn code_u16(program: &Program<'_>, at: usize) -> i64 {
    crate::grain::bytecode::code::u16_at(program.code(), at).map_or(-1, i64::from)
}

/// Read a little-endian `u32` operand, or `-1` when it is truncated.
#[majit_macros::elidable_cannot_raise]
pub(super) extern "C" fn code_u32(program: &Program<'_>, at: usize) -> i64 {
    crate::grain::bytecode::code::u32_at(program.code(), at).map_or(-1, i64::from)
}

/// Return `Position` as two packed `u16` scalars (`line << 16 | column`).
/// `Position::NONE` is zero.  The aggregate itself stays out of the residual
/// ABI, while the wrapper below reconstructs the exact public value.
#[majit_macros::elidable_cannot_raise]
pub(super) extern "C" fn code_position_bits(program: &Program<'_>, at: usize) -> i64 {
    position_bits(program.position(at))
}

/// Resolve an immutable constant-pool entry without exposing the backing
/// slice's fat pointer to generated code.
#[majit_macros::elidable_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn program_constant<'a>(
    program: &'a Program<'_>,
    index: u32,
) -> Option<&'a Dynamic> {
    program.constant(index)
}

/// Resolve an immutable assignment-operator descriptor without exposing its
/// backing `Vec` and `slice::get` implementation to the trace.
#[majit_macros::elidable_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn program_assign_op<'a>(
    program: &'a Program<'_>,
    index: u32,
) -> Option<&'a AssignOp> {
    program.assign_op(index)
}

#[inline]
pub(super) fn code_position(program: &Program<'_>, at: usize) -> crate::Position {
    position_from_bits(code_position_bits(program, at))
}

/// Execute Rhai's operation-limit/progress hook outside the trace.
///
/// `Engine` owns callbacks and nested containers whose concrete Rust layout is
/// not a translated GC object.  `Option<Box<_>>` is the same nullable pointer
/// word the Ref result bank carries; the wrapper restores the exact
/// `RhaiResultOf<()>` seen by the interpreter without raw-pointer ownership
/// tricks.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn track_operation_abi(
    vm: &mut Vm<'_>,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    vm.engine
        .track_operation(&mut vm.global, position_from_bits(position))
        .err()
}

/// Read the engine's immutable fast-operator option without tracing through
/// the `bitflags` implementation used by `LangOptions`.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn fast_operators(vm: &Vm<'_>) -> i64 {
    i64::from(vm.engine.fast_operators())
}

#[inline]
pub(super) fn track_operation(vm: &mut Vm<'_>, position: Position) -> RhaiResultOf<()> {
    match track_operation_abi(vm, position_bits(position)) {
        None => Ok(()),
        Some(error) => Err(error),
    }
}

/// Read the length of the frame-local array without exposing `ThinVec`'s
/// allocator-specific header to generated code.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn scope_len(scope: &Scope<'_>) -> i64 {
    scope.len() as i64
}

/// Resolve one pointer-stable frame-local slot across the residual ABI.
///
/// The caller checks the index against [`scope_len`] first. Under
/// `grain-jit`, each `Dynamic` is boxed, so growing the surrounding `ThinVec`
/// cannot invalidate the reference carried by the trace.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn scope_entry<'a>(
    scope: &'a mut Scope<'_>,
    index: usize,
) -> &'a mut Dynamic {
    scope.get_mut_by_index(index)
}

/// Read the operand-stack allocation length without exposing `Vec`'s
/// allocator-specific header to generated code. Unlike immutable `Program`
/// accessors, this is execution state owned by the red `Vm`, so it remains an
/// opaque, non-elidable residual call.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn operand_stack_len(vm: &Vm<'_>) -> i64 {
    vm.stack.len() as i64
}

/// Resolve one pointer-stable operand slot across the residual ABI.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn operand_stack_entry<'a>(vm: &'a Vm<'_>, index: usize) -> &'a Dynamic {
    super::operand_ref(&vm.stack[index])
}

/// Resolve one pointer-stable mutable operand slot across the residual ABI.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn operand_stack_entry_mut<'a>(
    vm: &'a mut Vm<'_>,
    index: usize,
) -> &'a mut Dynamic {
    super::operand_mut(&mut vm.stack[index])
}

/// Move one value out of a pointer-stable operand slot.
///
/// Keep `mem::take::<Dynamic>` behind this named ABI just as stores are kept
/// behind [`operand_stack_store`].  A generic standard-library path has no
/// stable, monomorphisation-specific symbolic name for the embedded table to
/// bind, while this wrapper has exactly one concrete Rust ABI. `Dynamic` is
/// an aggregate in the native Rust ABI, so use an out parameter instead of
/// pretending its translated Ref result is a native pointer return.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn operand_stack_take(vm: &mut Vm<'_>, index: usize, value: &mut Dynamic) {
    *value = core::mem::take(super::operand_mut(&mut vm.stack[index]));
}

/// Move one `Dynamic` into a pointer-stable operand slot.
///
/// A whole-value `*slot = value` currently reaches the MIR frontend as the
/// synthetic `__deref_write` marker, for which no executable host address can
/// exist. Keep that Rust aggregate store inside the opaque container ABI. The
/// mutable source makes this a move (`mem::take`), not an observable clone.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn operand_stack_store(vm: &mut Vm<'_>, index: usize, value: &mut Dynamic) {
    *super::operand_mut(&mut vm.stack[index]) = core::mem::take(value);
}

/// Drop every operand above `depth` and update the red VM's stack depth.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn truncate_stack(vm: &mut Vm<'_>, depth: usize) {
    vm.truncate_stack(depth);
}

/// RPython's default warm-loop threshold (`warmstate.rs` uses the same value).
const THRESHOLD: u32 = 1039;

struct Runtime {
    driver: JitDriver<jit_state::GrainJitState>,
    state: jit_state::GrainJitState,
    portal: Arc<majit_metainterp::JitCode>,
    portal_merge_point: usize,
    green_types: Vec<GreenType>,
}

impl Runtime {
    fn new() -> Option<Self> {
        if jitcodes::count() == 0 {
            return None;
        }

        // The portal backpointer is process-global and write-once, while
        // JitDriver itself is thread-affine. Until majit can let several TLS
        // drivers adopt one portal, the first consulting thread owns tracing;
        // other threads decline instead of attempting a second registration.
        static OWNER: OnceLock<std::thread::ThreadId> = OnceLock::new();
        let current = std::thread::current().id();
        if OWNER.get_or_init(|| current) != &current {
            bump_stats(|stats| stats.non_owner_consultations_declined += 1);
            return None;
        }

        // The build tables, liveness and dispatch identity have to be complete
        // before a trace clones staticdata. Keep them beside the state in one
        // thread-local owner because JitDriver is not a shared runtime object.
        jitcodes::install();
        let descriptor = jit_state::grain_driver_descriptor();
        let green_types = descriptor.green_args_spec();
        let mut driver = JitDriver::with_descriptor(THRESHOLD, descriptor);
        driver.ensure_descriptor_registered();
        let (insns, all_liveness) = jitcodes::liveness_parts();
        driver
            .meta_interp_mut()
            .install_liveness_from_build_parts(&insns, all_liveness);

        let table = jitcodes::all();
        let portal_index = jitcodes::portal_index().expect("the driver names its portal");
        let portal = table
            .get(portal_index)
            .expect("the driver portal is present in the embedded table")
            .clone();
        driver.register_dispatch_jitcode_shared(&portal);
        let registered = driver
            .dispatch_jitcode()
            .expect("registration names the Grain portal");
        assert!(
            Arc::ptr_eq(registered, &portal),
            "the driver must adopt the embedded table's portal identity",
        );
        assert_eq!(
            driver.meta_interp_mut().jitcodes().len(),
            table.len(),
            "registration must publish the complete embedded table",
        );

        driver.set_on_trace_abort(|_, _| bump_stats(|stats| stats.traces_aborted += 1));
        driver.set_on_compile_loop(|_, _, _, _| bump_stats(|stats| stats.loops_compiled += 1));
        driver.set_on_compiled_entry(|_, _| bump_stats(|stats| stats.compiled_entries += 1));

        Some(Self {
            driver,
            state: jit_state::GrainJitState::default(),
            portal,
            portal_merge_point: jitcodes::portal_merge_point_offset()
                .expect("the registered portal names its merge point"),
            green_types,
        })
    }
}

#[derive(Default)]
struct Counters {
    merge_points_consulted: usize,
    entry_door_interpret: usize,
    entry_door_already_tracing: usize,
    traces_started: usize,
    traces_aborted: usize,
    loops_compiled: usize,
    ops_recorded: usize,
    max_trace_ops: usize,
    compiled_entries: usize,
    compiled_entries_refused: usize,
    reentrant_consultations_declined: usize,
    non_owner_consultations_declined: usize,
    abort_reasons_before: Vec<(&'static str, u64)>,
    symbolic_residual_aborts_before: u64,
}

/// What the cheap half of the merge point reads, before the driver is asked.
///
/// One group rather than three thread-locals: this is read once per dispatched
/// instruction and each `thread_local!` is an access of its own.
struct Step {
    /// The `pc` the previous consultation carried.
    prev_pc: Cell<usize>,
    /// Whether the driver was tracing when the door last ran. A mirror, not a
    /// second source: tracing can only start or stop inside the door.
    tracing: Cell<bool>,
    /// Consultations answered without opening the door.
    skipped: Cell<usize>,
}

thread_local! {
    // JitDriver contains thread-affine tracing state. A trace can call back
    // into interpreter code, so `try_borrow_mut` declines a nested
    // consultation instead of panicking while the outer trace remains active.
    static RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };
    static COUNTERS: RefCell<Counters> = RefCell::new(Counters::default());
    static STEP: Step = const {
        Step {
            prev_pc: Cell::new(usize::MAX),
            tracing: Cell::new(false),
            skipped: Cell::new(0),
        }
    };
}

fn bump_stats(f: impl FnOnce(&mut Counters)) {
    COUNTERS.with(|cell| f(&mut cell.borrow_mut()));
}

pub(super) fn record_compiled_entry_refusal() {
    bump_stats(|stats| stats.compiled_entries_refused += 1);
}

pub(super) fn reset_stats() {
    STEP.with(|step| {
        step.skipped.set(0);
        step.prev_pc.set(usize::MAX);
    });
    COUNTERS.with(|cell| {
        *cell.borrow_mut() = Counters {
            abort_reasons_before: majit_metainterp::embed::abort_reasons(),
            symbolic_residual_aborts_before: majit_metainterp::symbolic_residual_trace_aborts(),
            ..Counters::default()
        };
    });
}

pub(super) fn stats() -> jit_state::GrainJitStats {
    COUNTERS.with(|cell| {
        let stats = cell.borrow();
        let mut abort_reasons = majit_metainterp::embed::render_abort_delta(
            &stats.abort_reasons_before,
            &majit_metainterp::embed::abort_reasons(),
        );
        let symbolic_residual_aborts = majit_metainterp::symbolic_residual_trace_aborts()
            .saturating_sub(stats.symbolic_residual_aborts_before)
            as usize;
        if symbolic_residual_aborts > 0 {
            if !abort_reasons.is_empty() {
                abort_reasons.push(' ');
            }
            abort_reasons.push_str(&format!(
                "unbound_symbolic_residual={}",
                symbolic_residual_aborts
            ));
        }
        jit_state::GrainJitStats {
            forward_steps_skipped: STEP.with(|step| step.skipped.get()),
            merge_points_consulted: stats.merge_points_consulted,
            entry_door_interpret: stats.entry_door_interpret,
            entry_door_already_tracing: stats.entry_door_already_tracing,
            traces_started: stats.traces_started,
            traces_aborted: stats.traces_aborted,
            loops_compiled: stats.loops_compiled,
            ops_recorded: stats.ops_recorded,
            max_trace_ops: stats.max_trace_ops,
            compiled_entries: stats.compiled_entries,
            compiled_entries_refused: stats.compiled_entries_refused,
            symbolic_residual_aborts,
            reentrant_consultations_declined: stats.reentrant_consultations_declined,
            non_owner_consultations_declined: stats.non_owner_consultations_declined,
            abort_reasons,
        }
    })
}

fn report_if_enabled(event: &str) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var_os("RHAI_GRAIN_JIT_STATS").is_some()) {
        eprintln!("[grain-jit] {event}: {:?}", stats());
    }
}

/// The driver `run_frame`'s loop reports to.
///
/// Zero-sized: the live driver/state are thread-local because a consumer
/// recognises this type's name as the marker receiver. Renaming it silently
/// disconnects the loop from the JIT rather than failing to build.
pub struct GrainJitDriver;

impl GrainJitDriver {
    /// A control-flow transfer is about to enter a loop header.
    ///
    /// The untranslated marker is inert, as RPython's `JitDriver` hint is.
    /// The translator recognises this receiver and lowers the call to
    /// `loop_header`; the following [`Self::jit_merge_point`] performs the
    /// native warm-state consultation with the target state.
    #[inline(never)]
    pub fn can_enter_jit(
        &self,
        pc: usize,
        program_identity: u64,
        program: &Program,
        frame: &mut GrainFrame<'_, '_>,
        vm: &Vm<'_>,
    ) {
        let _ = (pc, program_identity, program, frame, vm);
    }

    /// One iteration of the dispatch loop is about to run.
    ///
    /// Greens `(pc, program_identity, program)` first, then reds
    /// `(frame, vm)`.
    #[inline(never)]
    pub fn jit_merge_point(
        &self,
        pc: usize,
        program_identity: u64,
        program: &Program,
        frame: &mut GrainFrame<'_, '_>,
        vm: &Vm<'_>,
    ) {
        // The door below resolves a celltable cell, builds a driver descriptor
        // and extracts the live values before it decides to interpret -- four
        // heap blocks per call, on a path taken once per *dispatched
        // instruction*. Upstream is not asked that often. `jit_merge_point`
        // sits at the top of the dispatch loop there too, but what ticks the
        // counter is `can_enter_jit`, and `pyopcode.py jump_absolute` calls
        // that only on a backward jump.
        //
        // There is no separate `can_enter_jit` on this driver, so the back edge
        // is recognised instead of being told: `pc` advances on its own every
        // instruction and only a jump can fail to advance it, so a jump that
        // did not advance it went backwards. Entering or returning from a frame
        // can also fail to advance it, which costs one door call that decides
        // nothing. While a trace is open the door runs unconditionally -- the
        // tracer records every instruction, not every loop.
        let opened = STEP.with(|step| {
            if pc > step.prev_pc.replace(pc) && !step.tracing.get() {
                step.skipped.set(step.skipped.get() + 1);
                return false;
            }
            true
        });
        if !opened {
            return;
        }

        bump_stats(|stats| stats.merge_points_consulted += 1);

        let env = [
            frame as *const GrainFrame<'_, '_> as usize as i64,
            vm as *const Vm<'_> as usize as i64,
        ];
        assert_eq!(
            env.len(),
            jit_state::red_kinds().len(),
            "the merge point passes one raw word per declared red",
        );

        let mut report_event = None;
        let mut restored = Vec::new();
        RUNTIME.with(|cell| {
            let Ok(mut slot) = cell.try_borrow_mut() else {
                bump_stats(|stats| stats.reentrant_consultations_declined += 1);
                report_event = Some("reentrant consultation declined");
                return;
            };
            if slot.is_none() {
                *slot = Runtime::new();
            }
            let Some(runtime) = slot.as_mut() else {
                return;
            };

            // On the stack: this runs once per back edge, and the door it
            // feeds is already the allocating part. `green_key_hash_typed`
            // takes a slice, and the owned `GreenKey` is built only inside the
            // factory below, which the door calls only when it needs one.
            let green_values = [
                pc as i64,
                program_identity as i64,
                program as *const Program as usize as i64,
            ];
            assert_eq!(
                green_values.len(),
                runtime.green_types.len(),
                "the merge point passes one value per declared green",
            );
            let green_hash =
                majit_metainterp::green_key_hash_typed(&green_values, &runtime.green_types);
            let green_key = || GreenKey {
                values: green_values.to_vec(),
                types: runtime.green_types.clone(),
            };

            let was_tracing = runtime.driver.is_tracing();
            let resume = runtime.driver.back_edge_structured(
                green_hash,
                green_key,
                runtime.portal_merge_point,
                &mut runtime.state,
                &env,
                || {},
            );
            let started = !was_tracing && runtime.driver.is_tracing();
            assert!(
                resume.is_none() || started,
                "a non-tracing resume would mean compiled code bypassed GrainJitState's refusal",
            );
            bump_stats(|stats| {
                if started {
                    stats.traces_started += 1;
                } else if was_tracing {
                    stats.entry_door_already_tracing += 1;
                } else {
                    stats.entry_door_interpret += 1;
                }
            });

            if runtime.driver.is_tracing() {
                let portal = Arc::clone(&runtime.portal);
                let header_pc = runtime.portal_merge_point;
                let green_types = runtime.green_types.clone();
                runtime.driver.merge_point(|meta, sym| {
                    let ctx = meta.trace_ctx().expect("an active trace owns a context");
                    let before = ctx.num_ops();
                    assert_eq!(
                        sym.reds.len(),
                        jit_state::red_kinds().len(),
                        "the symbolic state carries every declared red",
                    );

                    // A warm-loop trace begins at the marker, not at
                    // `run_frame`'s function entry.  Seed the registers named
                    // by that marker from its typed greens/reds; ordinary
                    // `setup_call` would reset the cursor to zero and replay
                    // the frame-allocation prologue on every hot iteration.
                    let green_kinds: Vec<_> = green_types
                        .iter()
                        .copied()
                        .map(majit_ir::green_type_to_ir)
                        .map(|tp| {
                            majit_metainterp::JitArgKind::from_type(tp)
                                .expect("a green portal argument is not void")
                        })
                        .collect();
                    let green_args = [
                        (green_kinds[0], pc as i64),
                        (green_kinds[1], program_identity as i64),
                        (green_kinds[2], program as *const Program as usize as i64),
                    ];
                    let red_args: Vec<_> = jit_state::red_kinds()
                        .iter()
                        .copied()
                        .zip(sym.reds.iter().copied())
                        .zip(env.iter().copied())
                        .map(|((tp, opref), value)| {
                            let kind = majit_metainterp::JitArgKind::from_type(tp)
                                .expect("a red portal argument is not void");
                            (kind, opref, value)
                        })
                        .collect();

                    let frame = majit_metainterp::setup_frame_from_merge_point(
                        ctx,
                        Arc::clone(&portal),
                        header_pc,
                        &green_args,
                        &red_args,
                    );
                    let mut stack = majit_metainterp::StandaloneFrameStack::new();
                    stack.frames.push(frame);
                    let trace_runtime = majit_metainterp::ClosureRuntime::new(|label| label);
                    majit_metainterp::JitCodeSym::begin_portal_op(sym, header_pc);

                    let action = loop {
                        let step = {
                            let mut machine = majit_metainterp::JitCodeMachine::<
                                jit_state::GrainSym,
                                _,
                            >::with_framestack(
                                &mut stack.frames, &[], &[]
                            );
                            machine.run_one_step(ctx, sym, &trace_runtime)
                        };
                        match step {
                            TraceAction::Continue => {}
                            other => {
                                if std::env::var_os("RHAI_GRAIN_JIT_TRACE").is_some() {
                                    let current = stack.frames.frames.last();
                                    eprintln!(
                                        "[grain-jit-trace] depth={} stack={:?} after={:?} action={other:?}",
                                        stack.frames.len(),
                                        stack
                                            .frames
                                            .frames
                                            .iter()
                                            .map(|frame| (frame.jitcode.name(), frame.code_cursor))
                                            .collect::<Vec<_>>(),
                                        current.map(|frame| (
                                            frame.jitcode.name(),
                                            frame.pc,
                                            frame.code_cursor,
                                            frame.jitcode.code.get(frame.last_opcode_position).copied(),
                                            frame
                                                .jitcode
                                                .code
                                                .get(frame.last_opcode_position)
                                                .and_then(|opcode| jitcodes::insn_name(*opcode)),
                                        )),
                                    );
                                }
                                break other;
                            }
                        }
                    };
                    let after = ctx.num_ops();
                    let recorded = after.saturating_sub(before);
                    bump_stats(|stats| {
                        stats.ops_recorded += recorded;
                        stats.max_trace_ops = stats.max_trace_ops.max(recorded);
                    });
                    action
                });
            }

            // The mirror the gate above reads. Set from the driver here, the
            // one place tracing can have started or stopped.
            let tracing = runtime.driver.is_tracing();
            STEP.with(|step| step.tracing.set(tracing));

            if started {
                report_event = Some(if tracing {
                    "trace started"
                } else {
                    "trace decision"
                });
            }

            restored = runtime.state.take_restored();
        });

        if !restored.is_empty() {
            frame.restore_from_jit(&restored, vm);
        }

        if let Some(event) = report_event {
            report_if_enabled(event);
        }
    }
}
