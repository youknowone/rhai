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
use majit_metainterp::{JitDriver, JitState};

use super::{jit_state, jitcodes};
use super::{GrainFrame, Vm};
use crate::grain::bytecode::AssignOp;
use crate::grain::Program;
use crate::types::dynamic::Union;
use crate::{Dynamic, Position, Scope};

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

pub(super) fn position_bits(position: Position) -> i64 {
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

/// Return what the operator-and-call arm does with the instruction tagged
/// `tag`.
///
/// The `FORMS` table this reads is a `static`, and a static read the lowering
/// walks has no host address to bind: it falls through the whole global
/// folding chain to a nullary residual call naming the table itself.  Behind
/// this boundary the read is invisible, the way `WIDTHS` already is behind
/// [`code_width`].  The result is a bit set that no caller compares against a
/// sentinel, so widening it loses nothing and needs no `-1`.
#[majit_macros::elidable_cannot_raise]
pub(super) extern "C" fn code_form(tag: u8) -> i64 {
    i64::from(crate::grain::bytecode::code::form(tag))
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

/// Resolve an immutable program-table entry without exposing the backing
/// container.
///
/// Each of these is `slice::get` over a table the artifact froze, and `get` is
/// a callee the lowering leaves as an unbound symbolic residual -- one of those
/// anywhere the portal can reach refuses every trace.  `Option<&T>` is the
/// nullable pointer word the Ref result bank already carries, so the caller
/// keeps the exact value the interpreter had.  Elidable, as the sibling
/// [`program_constant`] is: the table does not change for the life of the
/// program.
#[majit_macros::elidable_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn program_token<'a>(
    program: &'a Program<'_>,
    index: u32,
) -> Option<&'a crate::tokenizer::Token> {
    program.token(index)
}

/// [`program_token`]'s sibling over the chain table.
#[majit_macros::elidable_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn program_chain<'a>(
    program: &'a Program<'_>,
    index: u32,
) -> Option<&'a crate::grain::bytecode::Chain> {
    program.chain(index)
}

/// A chain's tail, as the one scalar [`super::chain_tail`] defines.
///
/// `Tail` is an inline enum inside `Chain`, so reading the field is an
/// address-of rather than a load. A lowering that has no opcode for
/// `base + offset` cannot say that, and this boundary is what keeps the
/// distinction out of the trace: what crosses is the scalar, and the match
/// stays behind it the way `WIDTHS` stays behind [`code_width`].
///
/// Opaque rather than elidable: an elidable body whose every operation the
/// lowering can spell is inlined, and inlining this one puts the field read
/// back in the caller.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn chain_tail(chain: &crate::grain::bytecode::Chain) -> i64 {
    super::chain_tail_plain(chain)
}

/// An op-assignment's operator, as the one scalar [`super::assign_op_kind`]
/// defines.
///
/// `Option<BinOpKind>` is an inline enum inside `AssignOp`, and the same
/// address-of-rather-than-load distinction [`chain_tail`] crosses applies to
/// it. Opaque rather than elidable for the reason given there.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn assign_op_kind(op: &crate::grain::bytecode::AssignOp) -> i64 {
    super::assign_op_kind_plain(op)
}

/// [`program_token`]'s sibling over the switch table.
#[majit_macros::elidable_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn program_switch<'a>(
    program: &'a Program<'_>,
    index: u32,
) -> Option<&'a crate::grain::bytecode::Switch> {
    program.switch(index)
}

/// [`program_token`]'s sibling over the residual-expression table.
#[majit_macros::elidable_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn program_residual<'a>(
    program: &'a Program<'_>,
    index: u32,
) -> Option<&'a crate::ast::Expr> {
    program.residual(index)
}

/// [`array_entry`]'s write-side sibling.
///
/// The caller has already tested the index against the length, so this panics
/// for the same inputs the bare index expression would.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn array_entry_mut<'a>(
    array: &'a mut crate::Array,
    index: usize,
) -> &'a mut Dynamic {
    &mut array[index]
}

/// Whether the built-in op-assignment table answers for this operand pair.
///
/// The table picks its callee out of a chain of closures, and the build can
/// name an address for none of them, so each reaches the lowering as a residual
/// call whose target stays unbound.  One such target anywhere the portal can
/// reach refuses every trace, arm taken or not, which is what puts the
/// resolution behind this boundary at all.
///
/// Asked separately from [`store_builtin_apply`] because the answer is a third
/// state, and a residual call returns one word.  Spelling it as an out
/// parameter does not work: the argument banks carry ABI words, and the address
/// of a lowered local is not one, so the callee is handed a null to write
/// through.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn store_builtin_available(
    op: &AssignOp,
    target: &Dynamic,
    rhs: &Dynamic,
) -> i64 {
    i64::from(
        crate::func::builtin::get_builtin_op_assignment_fn(&op.op_assign, target, rhs).is_some(),
    )
}

/// Run the built-in op-assignment the table answers for this operand pair.
///
/// Total, because [`store_builtin_available`] has already answered `Some` for
/// the same triple and the table it reads is pure — nothing between the two
/// calls touches the operator token or either operand's type.  The result is
/// the same nullable exception pointer [`track_operation_abi`] returns.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn store_builtin_apply(
    vm: &mut Vm<'_>,
    op: &AssignOp,
    target: &mut Dynamic,
    rhs: &mut Dynamic,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    let position = position_from_bits(position);
    let outcome = vm.store_builtin_resolved(op, target, rhs, position);
    if outcome.handled {
        return outcome.error;
    }
    // Unreachable while the two reads agree. Raised rather than passed over in
    // silence, so that a table which stopped being pure loses the assignment
    // loudly instead of leaving the target at its old value.
    Some(Box::new(crate::EvalAltResult::ErrorRuntime(
        "the built-in op-assignment table answered twice and disagreed".into(),
        position,
    )))
}

/// Read the engine's immutable fast-operator option without tracing through
/// the `bitflags` implementation used by `LangOptions`.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn fast_operators(vm: &Vm<'_>) -> i64 {
    i64::from(vm.engine.fast_operators())
}

#[inline]
pub(super) fn track_operation_error(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    at: usize,
) -> Option<Box<crate::EvalAltResult>> {
    track_operation_abi(vm, code_position_bits(program, at))
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

/// Resolve one array element across the residual ABI.
///
/// `Vec`'s indexing is a callee the lowering leaves as an unbound symbolic
/// residual, and one of those anywhere the portal reaches refuses every
/// trace. The caller has already tested the index against the length, so this
/// panics for the same inputs the bare index expression would.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn array_entry<'a>(array: &'a crate::Array, index: usize) -> &'a Dynamic {
    &array[index]
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
    dynamic_store(super::operand_mut(&mut vm.stack[index]), value);
}

/// Move one whole [`Dynamic`] through an already-resolved mutable place.
///
/// Scope cells and operand slots resolve to the same final `&mut Dynamic`, but
/// the MIR frontend otherwise represents their assignments with the shared
/// synthetic `__deref_write` marker.  Its single annotator stub cannot retain
/// the distinct pointer owners, and it has no executable host address after
/// residualization.  This concrete ABI keeps the aggregate move and the old
/// value's drop together while allowing each caller to preserve how it found
/// the destination (including a shared cell's write guard).
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn dynamic_store(target: &mut Dynamic, value: &mut Dynamic) {
    *target = core::mem::take(value);
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
        let mut resume_pc = None;
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
            //
            // The key is the JUMP TARGET's position followed by the declared
            // greens, with the declared position green reading the target
            // rather than the back edge — the shape `jit_interp`'s
            // `green_key_expr` / `subst_target_for_pc` build for a marker
            // interpreter, and the shape `TraceCtx::merge_point_green_key`
            // reconstructs when `compile_loop` files the loop. A key that
            // omits the leading target is a key the close never derives, and
            // the loop is then stored where no back edge looks for it.
            let green_values = [
                pc as i64,
                pc as i64,
                program_identity as i64,
                program as *const Program as usize as i64,
            ];
            let green_types: Vec<GreenType> = std::iter::once(GreenType::Int)
                .chain(runtime.green_types.iter().copied())
                .collect();
            assert_eq!(
                green_values.len(),
                green_types.len(),
                "the merge point passes the target and one value per declared green",
            );
            let green_hash = majit_metainterp::green_key_hash_typed(&green_values, &green_types);
            let green_key = || GreenKey {
                values: green_values.to_vec(),
                types: green_types.clone(),
            };

            // The driver reads the live values off the state, and the state
            // is the one object that outlives a single consultation, so the
            // image this merge point was handed has to be published there
            // before the door is asked anything.
            runtime.state.publish_live(&env, &frame.jit_vable_words());

            let was_tracing = runtime.driver.is_tracing();
            // The source pc, not the merge point's offset in the portal body.
            // This is what `TraceCtx::header_pc` becomes, and the closing
            // visit compares its own `pc` green against that field: handing it
            // a jitcode offset makes the comparison one no source pc can
            // satisfy, so the walk records the whole remaining loop instead of
            // closing at the back edge. The offset is still what seeds the
            // walk's first frame below, because that one names a position in
            // the lowered body.
            let resume = runtime.driver.back_edge_structured(
                green_hash,
                green_key,
                pc,
                &mut runtime.state,
                &env,
                || {},
            );
            let started = !was_tracing && runtime.driver.is_tracing();
            assert!(
                runtime.driver.take_back_edge_finish().is_none(),
                "a compiled run reached `run_frame`'s own return; this loop has no path \
                 that carries a `VmResult` back out of the portal",
            );
            // A compiled run answers with the position the interpreter takes
            // over at, and nothing else in this call produces one when the run
            // happened -- tracing cannot have started in the same call.
            resume_pc = resume;
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

                    // `run_to_end` is the walk every other majit entry point
                    // uses: it opens and closes the portal op around the walk,
                    // catches a panic raised inside one step, and stops a walk
                    // that steps without recording. A hand-rolled step loop
                    // has none of those and differs from the shared one only
                    // by lacking them.
                    let action = {
                        let mut machine = majit_metainterp::JitCodeMachine::<
                            jit_state::GrainSym,
                            _,
                        >::with_framestack(
                            &mut stack.frames, &[], &[]
                        );
                        machine.set_outer_program_pc(pc);
                        machine.run_to_end(ctx, sym, &trace_runtime)
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

            // The jitcode walk above is the concrete execution of the traced
            // portal interval.  RPython returns from it through
            // `raise_continue_running_normally`; take majit's equivalent
            // single-pass handoff so `run_frame` resumes at the closing merge
            // point instead of replaying those side effects natively.
            let mut outcome = runtime.driver.take_single_pass_outcome();
            // `pyjitpl.py run_blackhole_interp_to_cancel_tracing`: an abort
            // can stop in the middle of one source opcode. Finish that opcode
            // and the rest of the portal interval in the blackhole, then use
            // the next merge point it reports. This must outrank the walk's
            // source-PC snapshot or the native loop replays the committed
            // prefix and applies its heap effects twice.
            if let Some(pc) = runtime
                .driver
                .run_pending_abort_blackhole(&mut runtime.state, &env)
            {
                outcome = Some((pc, Vec::new()));
            }
            assert!(
                !runtime.driver.take_single_pass_finish(),
                "the walk ran `run_frame` to its own return; this loop has no path that \
                 carries a `VmResult` back out of the portal",
            );
            if let Some((pc, reds)) = outcome {
                runtime
                    .driver
                    .writeback_scalar_state_fields(&mut runtime.state);
                runtime
                    .driver
                    .writeback_ref_scalar_state_fields(&mut runtime.state);
                runtime
                    .driver
                    .writeback_virt_array_state_fields(&mut runtime.state);
                debug_assert!(
                    reds.is_empty(),
                    "Grain's red operands are live frame/VM references, not a scalar state bank",
                );
                runtime.state.recover_after_compiled_run();
                runtime
                    .driver
                    .arm_single_pass_label_entry_on_next_back_edge(&runtime.state);
                runtime.driver.discard_single_pass_resume();
                resume_pc = Some(pc);
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
        if let Some(pc) = resume_pc {
            frame.jit_resume_pc_plus_one = pc + 1;
        }

        if let Some(event) = report_event {
            report_if_enabled(event);
        }
    }
}
