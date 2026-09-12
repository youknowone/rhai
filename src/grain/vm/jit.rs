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

use super::{GrainFrame, Vm};
use super::{jit_state, jitcodes};
use crate::grain::Program;
use crate::grain::bytecode::{AssignOp, Chain, Switch, code};
use crate::grain::program::Function;
use crate::types::dynamic::{AccessMode, Union};
use crate::{Dynamic, Position, Scope};

thread_local! {
    static INT_MODULO_RESULT: Cell<crate::INT> = const { Cell::new(0) };
    static ITER_NEXT_PRODUCED: Cell<i64> = const { Cell::new(0) };
    static SWITCH_TARGET: Cell<i64> = const { Cell::new(0) };
    static SWITCH_KIND: Cell<i64> = const { Cell::new(0) };
    static SWITCH_HASH: Cell<i64> = const { Cell::new(0) };
    static FAST_INT: Cell<crate::INT> = const { Cell::new(0) };
    static FAST_BOOL: Cell<i64> = const { Cell::new(0) };
    #[cfg(not(feature = "no_float"))]
    static FAST_FLOAT: Cell<crate::FLOAT> = const { Cell::new(0.0) };
    static PROGRAM_NAME: Cell<Option<&'static str>> = const { Cell::new(None) };
    static OPERATOR_BUILTIN_HANDLED: Cell<i64> = const { Cell::new(0) };
    static UNARY_BUILTIN_HANDLED: Cell<i64> = const { Cell::new(0) };
    static PLAIN_ADD_HANDLED: Cell<i64> = const { Cell::new(0) };
    static PREPARED_CHUNK_ENTRY: Cell<i64> = const { Cell::new(0) };
    static RESIDUAL_NEST: Cell<u32> = const { Cell::new(0) };
    static PLAIN_ADD_CACHE: RefCell<Vec<(usize, u32, bool)>> = const { RefCell::new(Vec::new()) };
}

/// Compiled code holds `RUNTIME` for the residual it called. Nested
/// `run_frame` merge points cannot consult, so skip them before `STEP`.
struct ResidualNest;

impl ResidualNest {
    fn enter() -> Option<Self> {
        let compiled = RUNTIME
            .try_with(|cell| cell.try_borrow().is_err())
            .unwrap_or(false);
        if !compiled {
            return None;
        }
        RESIDUAL_NEST.with(|nest| nest.set(nest.get().saturating_add(1)));
        bump_stats(|stats| stats.reentrant_consultations_declined += 1);
        Some(Self)
    }
}

impl Drop for ResidualNest {
    fn drop(&mut self) {
        RESIDUAL_NEST.with(|nest| nest.set(nest.get().saturating_sub(1)));
    }
}

enum PlainFast {
    Did(Option<Box<crate::EvalAltResult>>),
    Miss,
}

fn try_plain_add_ref(
    vm: &mut Vm<'_>,
    scope: &mut Scope<'_>,
    base: usize,
    slot: u16,
    first: usize,
) -> PlainFast {
    let at = base.saturating_add(slot as usize);
    if at >= scope.len() || first >= vm.depth {
        return PlainFast::Miss;
    }
    match super::apply_binary(
        crate::grain::bytecode::BinOpKind::Add,
        scope.get_mut_by_index(at),
        super::stack_ref(vm, first),
    ) {
        Ok(Some(value)) => {
            vm.truncate_stack(first);
            vm.push_fast(value);
            PlainFast::Did(None)
        }
        Ok(None) => PlainFast::Miss,
        Err(err) => PlainFast::Did(Some(err)),
    }
}

fn try_plain_add_stack(vm: &mut Vm<'_>, first: usize) -> PlainFast {
    if first + 1 >= vm.depth {
        return PlainFast::Miss;
    }
    for slot in first..first + 2 {
        #[cfg(not(feature = "no_closure"))]
        if super::stack_ref(vm, slot).is_shared() {
            let held = core::mem::replace(super::stack_mut(vm, slot), super::unit_value());
            super::store_value(super::stack_mut(vm, slot), held.flatten());
        }
    }
    match super::apply_binary(
        crate::grain::bytecode::BinOpKind::Add,
        super::stack_ref(vm, first),
        super::stack_ref(vm, first + 1),
    ) {
        Ok(Some(value)) => {
            vm.truncate_stack(first);
            vm.push_fast(value);
            PlainFast::Did(None)
        }
        Ok(None) => PlainFast::Miss,
        Err(err) => PlainFast::Did(Some(err)),
    }
}

fn try_plain_abs_ref(
    vm: &mut Vm<'_>,
    scope: &mut Scope<'_>,
    base: usize,
    slot: u16,
    position: i64,
) -> PlainFast {
    let at = base.saturating_add(slot as usize);
    if at >= scope.len() {
        return PlainFast::Miss;
    }
    let Ok(x) = scope.get_mut_by_index(at).as_int() else {
        return PlainFast::Miss;
    };
    let value = if cfg!(not(feature = "unchecked")) {
        match x.checked_abs() {
            Some(abs) => Dynamic::from(abs),
            None => {
                return PlainFast::Did(Some(Box::new(crate::EvalAltResult::ErrorArithmetic(
                    format!("Negation overflow: -{x}"),
                    position_from_bits(position),
                ))));
            }
        }
    } else {
        Dynamic::from(x.abs())
    };
    vm.push(value);
    PlainFast::Did(None)
}

fn try_plain_add_named(
    vm: &mut Vm<'_>,
    scope: &mut Scope<'_>,
    name: &str,
    first: usize,
) -> PlainFast {
    if first >= vm.depth {
        return PlainFast::Miss;
    }
    let Some(lhs) = scope.get_mut(name) else {
        return PlainFast::Miss;
    };
    match super::apply_binary(
        crate::grain::bytecode::BinOpKind::Add,
        lhs,
        super::stack_ref(vm, first),
    ) {
        Ok(Some(value)) => {
            vm.truncate_stack(first);
            vm.push_fast(value);
            PlainFast::Did(None)
        }
        Ok(None) => PlainFast::Miss,
        Err(err) => PlainFast::Did(Some(err)),
    }
}

fn try_plain_abs_named(
    vm: &mut Vm<'_>,
    scope: &mut Scope<'_>,
    name: &str,
    position: i64,
) -> PlainFast {
    let Some(entry) = scope.get_mut(name) else {
        return PlainFast::Miss;
    };
    let Ok(x) = entry.as_int() else {
        return PlainFast::Miss;
    };
    let value = if cfg!(not(feature = "unchecked")) {
        match x.checked_abs() {
            Some(abs) => Dynamic::from(abs),
            None => {
                return PlainFast::Did(Some(Box::new(crate::EvalAltResult::ErrorArithmetic(
                    format!("Negation overflow: -{x}"),
                    position_from_bits(position),
                ))));
            }
        }
    } else {
        Dynamic::from(x.abs())
    };
    vm.push(value);
    PlainFast::Did(None)
}

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

/// Write `Items::IntRange.next` without a portal `&mut INT` borrow.
///
/// A portal `*next += 1` is `__deref_write`. Behind this boundary the
/// write is a real field store on the live iterator, not a walk-local.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn store_int_range_next(items: &mut super::Items, value: crate::INT) {
    let super::Items::IntRange { next, .. } = items else {
        return;
    };
    *next = value;
}

/// Write `Iteration.count` without a `&mut INT` borrow.
///
/// Twin of [`store_int_range_next`] for the overflow counter.
#[inline(always)]
pub(super) fn store_iteration_count(iteration: &mut super::Iteration, value: crate::INT) {
    iteration.count = value;
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
/// not a translated GC object.  A void residual is required: an `Option<Box<_>>`
/// or `i64` result was guarded after the call, and a later `setfield` reused
/// that result register, so the compiled loop deopted onto the error arm
/// with a leftover value.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn track_operation_abi(vm: &mut Vm<'_>, position: i64) {
    if !live_vm_ptr(vm) {
        majit_metainterp::request_walk_abort();
        return;
    }
    let _ = vm
        .engine
        .track_operation(&mut vm.global, position_from_bits(position));
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

/// [`program_token`]'s sibling over the name table.
///
/// `Strings::get` is three `slice::get`s, and one of those anywhere the portal
/// can reach refuses every trace. `Option<&str>` is a fat pointer, so this
/// residual is the presence bit; the slice is interned and read back with
/// [`program_name_held`].
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn program_name(program: &Program<'_>, index: u32) -> i64 {
    match program.name_plain(index) {
        Some(name) => {
            PROGRAM_NAME.with(|cell| cell.set(Some(intern_program_name(name))));
            1
        }
        None => {
            PROGRAM_NAME.with(|cell| cell.set(None));
            0
        }
    }
}

/// The slice [`program_name`] interned. Not a residual: a TLS load so
/// `Strings::get` stays behind the one-word boundary.
///
/// Must stay a function (not an inline `LocalKey::with` in the portal): the
/// static is not a code address. Must not be `dont_look_inside` either: the
/// fat-pointer return is not a residual word and binding it panics in the GC.
#[inline(always)]
pub(super) fn program_name_held() -> Option<&'static str> {
    PROGRAM_NAME.with(Cell::get)
}

fn intern_program_name(name: &str) -> &'static str {
    static INTERN: std::sync::OnceLock<std::sync::Mutex<Vec<&'static str>>> =
        std::sync::OnceLock::new();
    let intern = INTERN.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut names = intern
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(&held) = names.iter().find(|&&held| held == name) {
        return held;
    }
    let held: &'static str = Box::leak(name.to_string().into_boxed_str());
    names.push(held);
    held
}

/// [`program_name`]'s sibling over the compiled-function table.
#[majit_macros::elidable_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn program_function<'a>(
    program: &'a Program<'_>,
    name: u32,
    argc: usize,
) -> Option<&'a Function> {
    program.function_plain(name, argc)
}

/// Walk one property/index chain without exposing its internals to the trace.
///
/// `Vm::run_chain` is a residual the build cannot address, and one such
/// target anywhere the portal reaches refuses every trace. The result is
/// pushed onto the operand stack so the walk is not handed the address of a
/// lowered local.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn run_chain_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    chain: &Chain,
    index: u32,
    scope: &mut Scope<'_>,
    base: usize,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    match vm.run_chain_inner(
        program,
        chain,
        index,
        scope,
        base,
        position_from_bits(position),
    ) {
        Ok(result) => {
            vm.push(result);
            None
        }
        Err(err) => Some(err),
    }
}

/// Integer remainder without exposing `checked_rem` to the portal.
///
/// The quotient is a second one-word residual ([`int_modulo_result`]) rather
/// than an out-parameter: a lowered local's address is not an ABI word, so
/// writing through one is a null store.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn int_modulo(
    lhs: crate::INT,
    rhs: crate::INT,
) -> Option<Box<crate::EvalAltResult>> {
    match crate::packages::arithmetic::arith_basic::INT::functions::modulo(lhs, rhs) {
        Ok(value) => {
            INT_MODULO_RESULT.with(|cell| cell.set(value));
            None
        }
        Err(err) => Some(err),
    }
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn int_modulo_result() -> crate::INT {
    INT_MODULO_RESULT.with(std::cell::Cell::get)
}

/// `Ord::max` for two machine words. The translator leaves
/// `core::cmp::impls::<Impl>::max` as a symbolic residual; jumping to that
/// hash is a fault, and refusing it aborts the cold-arm bridge.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn int_max(a: i64, b: i64) -> i64 {
    if a >= b { a } else { b }
}

/// Advance a `for` loop that stores into a scope slot.
///
/// The produced/exhausted bit is a second one-word residual
/// ([`iter_next_produced`]) so this can use the nullable exception word.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn iter_next_store(
    vm: &mut Vm<'_>,
    scope: &mut Scope<'_>,
    index: usize,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) || !live_scope_ptr(scope) || !live_count(index) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    match vm.iter_next_store(scope, index, position_from_bits(position)) {
        Ok(produced) => {
            // The iterator was advanced or popped. Replay of this
            // opcode is no longer sound.
            majit_metainterp::note_residual_committed();
            ITER_NEXT_PRODUCED.with(|cell| cell.set(i64::from(produced)));
            None
        }
        Err(err) => Some(err),
    }
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn iter_next_produced() -> i64 {
    ITER_NEXT_PRODUCED.with(Cell::get)
}

/// Write an integer into a `for` loop slot without a walk-local `Dynamic`.
///
/// A whole-value `*slot = Dynamic` is the synthetic `__deref_write` marker.
/// The payload write on an existing `Union::Int` is a field store; replacing
/// a non-int slot (the first turn's `Unit`, or a shared cell) stays behind
/// this boundary and goes through [`store_shared`] so a captured cell is
/// written rather than replaced.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn store_scope_int(
    scope: &mut Scope<'_>,
    index: usize,
    value: crate::INT,
    position: i64,
) {
    if !live_scope_ptr(scope) || !live_count(index) {
        majit_metainterp::request_walk_abort();
        return;
    }
    majit_metainterp::note_residual_committed();
    let entry = scope.get_mut_by_index(index);
    if let crate::types::dynamic::Union::Int(held, ..) = &mut entry.0 {
        *held = value;
    } else {
        let _ = super::store_shared(
            entry,
            crate::Dynamic::from_int(value),
            position_from_bits(position),
        );
    }
}

/// Pop the stack top into a `for` loop slot, including a shared cell.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn store_shared_from_stack(
    vm: &mut Vm<'_>,
    scope: &mut Scope<'_>,
    index: usize,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) || !live_scope_ptr(scope) || !live_count(index) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    match vm.pop() {
        Ok(value) => {
            majit_metainterp::note_residual_committed();
            super::store_shared(
                &mut scope.get_mut_by_index(index),
                value.flatten(),
                position_from_bits(position),
            )
            .err()
        }
        Err(err) => Some(err),
    }
}

/// The innermost running `for` loop, without exposing `slice::last_mut`.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn iterator_last_mut<'a>(
    vm: &'a mut Vm<'_>,
) -> Option<&'a mut super::Iteration> {
    vm.iterators_last_mut()
}

/// Pop one `switch` subject and classify it without exposing the hasher seed.
///
/// The subject is popped inside this boundary: a walk local's address is not
/// an ABI word, so handing `&Dynamic` out of the portal stores through null.
///
/// Kind `1` is an integer: [`fast_int`] holds the value so the portal can
/// compare it against recovered case keys (and integer ranges) as in-loop
/// guards. Hashing stays behind this boundary. Any other kind stores the
/// table's target in [`switch_target`].
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_pop_subject(
    vm: &mut Vm<'_>,
    table: &Switch,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    match vm.pop() {
        Ok(subject) => {
            match subject.as_int() {
                Ok(value) => {
                    FAST_INT.with(|cell| cell.set(value));
                    SWITCH_KIND.with(|cell| cell.set(1));
                }
                Err(_) => {
                    SWITCH_KIND.with(|cell| cell.set(0));
                    SWITCH_TARGET.with(|cell| cell.set(i64::from(table.dispatch(&subject))));
                }
            }
            None
        }
        Err(err) => Some(err),
    }
}

/// Hash-dispatch one `switch` without exposing the hasher's seed static.
///
/// Kept for subjects that are not integers. The subject is popped inside this
/// boundary; the arm is a second one-word residual ([`switch_target`]).
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_dispatch(
    vm: &mut Vm<'_>,
    table: &Switch,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    match vm.pop() {
        Ok(subject) => {
            SWITCH_TARGET.with(|cell| cell.set(i64::from(table.dispatch(&subject))));
            None
        }
        Err(err) => Some(err),
    }
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn switch_target() -> i64 {
    SWITCH_TARGET.with(Cell::get)
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn switch_subject_kind() -> i64 {
    SWITCH_KIND.with(Cell::get)
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn switch_subject_hash() -> i64 {
    SWITCH_HASH.with(Cell::get)
}

/// Case-table scalars. Elidable: the table is green, so a specialized `index`
/// folds to a constant and the compiled loop's arm compares stay guards
/// rather than residual calls. `inline(never)` keeps `slice::get` out of the
/// portal the way [`program_switch`] does.
#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_case_count(table: &Switch) -> i64 {
    table.cases().len() as i64
}

#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_case_hash(table: &Switch, index: i64) -> i64 {
    table
        .cases()
        .get(index as usize)
        .map(|case| case.hash as i64)
        .unwrap_or(0)
}

#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_case_has_int(table: &Switch, index: i64) -> i64 {
    i64::from(table.int_key(index as usize).is_some())
}

#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_case_int_key(table: &Switch, index: i64) -> crate::INT {
    table.int_key(index as usize).unwrap_or(0)
}

#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_case_target(table: &Switch, index: i64) -> i64 {
    table
        .cases()
        .get(index as usize)
        .map(|case| i64::from(case.target))
        .unwrap_or(0)
}

#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_default(table: &Switch) -> i64 {
    i64::from(table.default)
}

#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_range_count(table: &Switch) -> i64 {
    table.ranges.len() as i64
}

#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_range_from(table: &Switch, index: i64) -> crate::INT {
    table
        .ranges
        .get(index as usize)
        .map(|range| range.from)
        .unwrap_or(0)
}

#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_range_to(table: &Switch, index: i64) -> crate::INT {
    table
        .ranges
        .get(index as usize)
        .map(|range| range.to)
        .unwrap_or(0)
}

#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_range_inclusive(table: &Switch, index: i64) -> i64 {
    table
        .ranges
        .get(index as usize)
        .map(|range| i64::from(range.inclusive))
        .unwrap_or(0)
}

#[majit_macros::elidable_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn switch_range_target(table: &Switch, index: i64) -> i64 {
    table
        .ranges
        .get(index as usize)
        .map(|range| i64::from(range.target))
        .unwrap_or(0)
}

/// Copy a scalar out of a resident `Dynamic` without exposing `Union`.
///
/// `1` int, `2` bool, `3` float, `4` unit, `0` anything else. The payload is
/// a second one-word residual (`fast_int` / `fast_bool` / `fast_float`).
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn dynamic_as_fast(value: &Dynamic) -> i64 {
    if !live_dynamic_ptr(value) {
        return 0;
    }
    match &value.0 {
        Union::Int(held, ..) => {
            FAST_INT.with(|cell| cell.set(*held));
            1
        }
        Union::Bool(held, ..) => {
            FAST_BOOL.with(|cell| cell.set(i64::from(*held)));
            2
        }
        #[cfg(not(feature = "no_float"))]
        Union::Float(held, ..) => {
            FAST_FLOAT.with(|cell| cell.set(**held));
            3
        }
        Union::Unit(..) => 4,
        Union::Shared(cell, ..) => match crate::func::locked_read(cell) {
            Some(inner) => dynamic_as_fast(&inner),
            None => 0,
        },
        _ => 0,
    }
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn fast_int() -> crate::INT {
    FAST_INT.with(Cell::get)
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn fast_bool() -> i64 {
    FAST_BOOL.with(Cell::get)
}

#[cfg(not(feature = "no_float"))]
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn fast_float() -> crate::FLOAT {
    FAST_FLOAT.with(Cell::get)
}

/// The live operand-stack depth, as one ABI word.
///
/// A residual that `push`es has already updated the real field. The walk still
/// holds the pre-call depth, so it must read the real one back rather than
/// adding one to a stale local.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn vm_depth(vm: &Vm<'_>) -> i64 {
    vm.depth() as i64
}

/// Array length without exposing `Vec::len` as a synthetic `__len` residual.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn array_len(array: &crate::Array) -> i64 {
    array.len() as i64
}

/// Remember a successful portal return so native `run_frame` can leave with it.
///
/// A tracing walk that reaches `RETURN` finishes the portal. Rust cannot
/// raise that result across the merge-point marker, so the live frame carries
/// the value the same way it carries the resume pc.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn stash_ok_result(frame: &mut GrainFrame<'_, '_>, value: &Dynamic) {
    frame.jit_finished = Some(Box::new(Ok(super::value::clone_value(value))));
    frame.jit_return_kind = 1;
}

/// Bind a compiled script call into [`Vm::prepared_scope`].
///
/// The portal then calls `run_frame` on that scope so the tracer can
/// `can_inline` the body (`Function.call_args` → `execute_frame`).
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn prepare_compiled_call_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    name_index: u32,
    argc: usize,
    first: usize,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    match vm.prepare_compiled_call(
        program,
        name_index,
        argc,
        first,
        position_from_bits(position),
    ) {
        Ok(entry) => {
            PREPARED_CHUNK_ENTRY.with(|cell| cell.set(entry as i64));
            None
        }
        Err(err) => Some(err),
    }
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn prepared_chunk_entry() -> i64 {
    PREPARED_CHUNK_ENTRY.with(Cell::get)
}

/// Rewind the prepared callee scope and give it back to the pool.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn finish_compiled_call_abi(vm: &mut Vm<'_>) {
    vm.finish_compiled_call();
}

/// Run a stacked or syntactic call without exposing its internals to the trace.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn call_syntactic_or_stacked_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    name_index: u32,
    argc: usize,
    first: usize,
    scope: &mut Scope<'_>,
    capture: i64,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if capture == 0
        && argc == 2
        && vm.engine.fast_operators()
        && cached_plain_add(program, name_index, argc)
    {
        match try_plain_add_stack(vm, first) {
            PlainFast::Did(err) => return err,
            PlainFast::Miss => {}
        }
    }
    let _nest = ResidualNest::enter();
    let Some(name) = program.name_plain(name_index) else {
        return Some(Box::new(crate::EvalAltResult::ErrorRuntime(
            format!("malformed chunk: no name {name_index}").into(),
            position_from_bits(position),
        )));
    };
    match vm.call_syntactic_or_stacked(
        program,
        name_index,
        name,
        argc,
        first,
        scope,
        capture != 0,
        position_from_bits(position),
    ) {
        Ok(result) => {
            vm.push(result);
            None
        }
        Err(err) => Some(err),
    }
}

/// Run a by-reference call without exposing its internals to the trace.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn call_by_reference_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    name_index: u32,
    argc: usize,
    receiver_kind: i64,
    receiver_payload: u32,
    scope: &mut Scope<'_>,
    base: usize,
    capture: i64,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if capture == 0 && vm.engine.fast_operators() {
        // CallRef `add(s, i)` reports argc=2 (arity) with the receiver
        // in a local/name and only `i` on the stack.
        if (argc == 1 || argc == 2) && cached_plain_add(program, name_index, 2) {
            if let Some(first) = vm.depth.checked_sub(1) {
                let hit = match receiver_kind {
                    0 => try_plain_add_ref(vm, scope, base, receiver_payload as u16, first),
                    1 => match program.name_plain(receiver_payload) {
                        Some(name) => try_plain_add_named(vm, scope, name, first),
                        None => PlainFast::Miss,
                    },
                    _ => PlainFast::Miss,
                };
                match hit {
                    PlainFast::Did(err) => return err,
                    PlainFast::Miss => {}
                }
            }
        }
        // CallRef `abs(a)` reports argc=1 (arity) with the receiver in a
        // local/name and nothing on the stack.
        if (argc == 0 || (argc == 1 && vm.depth == 0))
            && program.name_plain(name_index) == Some("abs")
        {
            let hit = match receiver_kind {
                0 => try_plain_abs_ref(vm, scope, base, receiver_payload as u16, position),
                1 => match program.name_plain(receiver_payload) {
                    Some(name) => try_plain_abs_named(vm, scope, name, position),
                    None => PlainFast::Miss,
                },
                _ => PlainFast::Miss,
            };
            match hit {
                PlainFast::Did(err) => return err,
                PlainFast::Miss => {}
            }
        }
    }
    let _nest = ResidualNest::enter();
    let Some(name) = program.name_plain(name_index) else {
        return Some(Box::new(crate::EvalAltResult::ErrorRuntime(
            format!("malformed chunk: no name {name_index}").into(),
            position_from_bits(position),
        )));
    };
    let receiver = match receiver_kind {
        0 => crate::grain::bytecode::Receiver::Local(receiver_payload as u16),
        1 => crate::grain::bytecode::Receiver::Named(receiver_payload),
        _ => crate::grain::bytecode::Receiver::This,
    };
    match vm.call_by_reference(
        program,
        name_index,
        name,
        argc,
        receiver,
        scope,
        base,
        capture != 0,
        position_from_bits(position),
    ) {
        Ok(result) => {
            vm.push(result);
            None
        }
        Err(err) => Some(err),
    }
}

/// Store into a scope slot, including the access-mode stamp and shared-cell
/// write lock. Those two writes otherwise reach the portal as `__deref_write`.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn store_scope_slot(
    vm: &mut Vm<'_>,
    entry: &mut Dynamic,
    read_only: i64,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_dynamic_ptr(entry) {
        return None;
    }
    let mut value = match vm.pop() {
        Ok(value) => value,
        Err(err) => return Some(err),
    };
    value.set_access_mode(if read_only != 0 {
        crate::types::dynamic::AccessMode::ReadOnly
    } else {
        crate::types::dynamic::AccessMode::ReadWrite
    });
    match super::place(entry, "", position_from_bits(position)) {
        Ok(mut target) => {
            super::store_value(&mut target, value);
            None
        }
        Err(err) => Some(err),
    }
}

/// Flatten the RHS and write it through a local, including a shared cell.
///
/// `flatten_clone_value` returns a `Dynamic` and the shared-cell arm reads the
/// name as a fat pointer. Neither is a residual word, so both stay behind this
/// boundary; the portal sees only the nullable exception.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn assign_local_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    base: usize,
    slot: usize,
    var_name: u32,
    from_kind: i64,
    from_index: u32,
    op: Option<&AssignOp>,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) || !live_scope_ptr(scope) || !live_count(base) || !live_count(slot) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    let from = match from_kind {
        1 => Some(crate::grain::bytecode::BinOperand::Local(from_index as u16)),
        2 => Some(crate::grain::bytecode::BinOperand::Const(from_index)),
        _ => None,
    };
    match vm.assign_local(
        program,
        scope,
        base,
        slot,
        var_name,
        from,
        op,
        position_from_bits(position),
    ) {
        Ok(()) => None,
        Err(err) => Some(err),
    }
}

/// Load a named variable. The name stays behind this boundary.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn load_named_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    name_index: u32,
    flatten: i64,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) || !live_scope_ptr(scope) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    let Some(name) = program.name_plain(name_index) else {
        return Some(Box::new(crate::EvalAltResult::ErrorRuntime(
            format!("malformed chunk: no name {name_index}").into(),
            position_from_bits(position),
        )));
    };
    match vm.load_named(name, scope, flatten != 0, position_from_bits(position)) {
        Ok(value) => {
            vm.push(value);
            None
        }
        Err(err) => Some(err),
    }
}

/// Assign to a named variable. The name stays behind this boundary.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn assign_named_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    name_index: u32,
    op: Option<&AssignOp>,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) || !live_scope_ptr(scope) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    let Some(name) = program.name_plain(name_index) else {
        return Some(Box::new(crate::EvalAltResult::ErrorRuntime(
            format!("malformed chunk: no name {name_index}").into(),
            position_from_bits(position),
        )));
    };
    let rhs = match vm.pop() {
        Ok(value) => value.flatten(),
        Err(err) => return Some(err),
    };
    vm.assign_named(program, op, name, rhs, scope, position_from_bits(position))
        .err()
}

/// Declare a local or constant. The name stays behind this boundary.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn declare_local_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    name_index: u32,
    constant: i64,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) || !live_scope_ptr(scope) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    let Some(name) = program.name_plain(name_index) else {
        return Some(Box::new(crate::EvalAltResult::ErrorRuntime(
            format!("malformed chunk: no name {name_index}").into(),
            position_from_bits(position),
        )));
    };
    let value = match vm.pop() {
        Ok(value) => value.flatten(),
        Err(err) => return Some(err),
    };
    if constant != 0 {
        scope.push_constant_dynamic(name, value);
    } else {
        scope.push_dynamic(name, value);
    }
    None
}

/// Bind a catch variable. The name stays behind this boundary; the value
/// is popped from the operand stack.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn catch_bind_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    name_index: u32,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) || !live_scope_ptr(scope) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    let Some(name) = program.name_plain(name_index) else {
        return Some(Box::new(crate::EvalAltResult::ErrorRuntime(
            format!("malformed chunk: no name {name_index}").into(),
            position_from_bits(position),
        )));
    };
    let value = match vm.pop() {
        Ok(value) => value,
        Err(err) => return Some(err),
    };
    scope.push_dynamic(name, value);
    None
}

/// Built-in binary operator after the typed/fast arms declined.
///
/// Flattening a shared stack cell is not a one-word residual. `1` means the
/// built-in ran and pushed; `0` means fall through to the syntactic helper.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn operator_builtin_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    token: &crate::tokenizer::Token,
    name_index: u32,
    first: usize,
    generation: u64,
    pc: usize,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    OPERATOR_BUILTIN_HANDLED.with(|cell| cell.set(0));
    if !live_vm_ptr(vm) || !live_count(first) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    let name = program.name_plain(name_index).unwrap_or("");
    let pos = position_from_bits(position);
    if !vm.engine.fast_operators() {
        return None;
    }
    for slot in first..vm.depth {
        #[cfg(not(feature = "no_closure"))]
        if super::stack_ref(vm, slot).is_shared() {
            let held = core::mem::replace(super::stack_mut(vm, slot), super::unit_value());
            super::store_value(super::stack_mut(vm, slot), held.flatten());
        }
    }
    let memo = &mut vm.operator_memo;
    let top = vm.depth;
    let (lhs, rhs) = vm.stack[..top].split_at_mut(first + 1);
    let lhs = &mut lhs[first];
    let rhs = &mut rhs[0];
    if lhs.is_variant() || rhs.is_variant() {
        return None;
    }
    let Some((func, need_context)) = super::resolve_operator(memo, generation, pc, token, lhs, rhs)
    else {
        return None;
    };
    let context = need_context.then(|| (vm.engine, name, None, &vm.global, pos).into());
    match func(context, &mut [lhs, rhs]) {
        Ok(value) => {
            vm.push(value);
            OPERATOR_BUILTIN_HANDLED.with(|cell| cell.set(1));
            None
        }
        Err(err) => Some(err),
    }
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn operator_builtin_handled() -> i64 {
    OPERATOR_BUILTIN_HANDLED.with(Cell::get)
}

/// Whether a compiled script function is the body `a + b`.
///
/// The tracer records the typed `+` already lowered for `BIN_OP`; a
/// residual `CALL` of that body cannot. Program and name are green.
#[majit_macros::elidable_cannot_raise]
pub(super) extern "C" fn compiled_fn_is_plain_add(
    program: &Program<'_>,
    name_index: u32,
    argc: usize,
) -> i64 {
    i64::from(argc == 2 && function_body_is_plain_add(program, name_index, argc))
}

fn cached_plain_add(program: &Program<'_>, name_index: u32, argc: usize) -> bool {
    let key = program as *const Program<'_> as usize;
    PLAIN_ADD_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((_, _, hit)) = cache.iter().find(|(p, n, _)| *p == key && *n == name_index) {
            return *hit;
        }
        let hit = argc == 2 && function_body_is_plain_add(program, name_index, argc);
        cache.push((key, name_index, hit));
        hit
    })
}

fn function_body_is_plain_add(program: &Program<'_>, name_index: u32, argc: usize) -> bool {
    use crate::grain::bytecode::code::disassemble;
    use crate::grain::bytecode::{BinOpKind, BinOperand, Op};
    let Some(function) = program.function_plain(name_index, argc) else {
        return false;
    };
    let start = function.chunk.entry() as usize;
    let end = function.chunk.end() as usize;
    let Some(bytes) = program.code().get(start..end) else {
        return false;
    };
    let ops: Vec<Op> = disassemble(bytes)
        .map(|(_, op)| op)
        .filter(|op| !matches!(op, Op::Tick | Op::Checkpoint))
        .collect();
    match ops.as_slice() {
        [
            Op::BinOpFrom {
                kind: BinOpKind::Add,
                lhs: 0,
                rhs: BinOperand::Local(1),
                branch: None,
                ..
            },
            Op::Return,
        ]
        | [
            Op::BinOpFrom {
                kind: BinOpKind::Add,
                lhs: 0,
                rhs: BinOperand::Local(1),
                branch: None,
                ..
            },
        ]
        | [
            Op::LoadLocal(0),
            Op::LoadLocal(1),
            Op::BinOp {
                kind: BinOpKind::Add,
                rhs: None,
                branch: None,
                ..
            },
            Op::Return,
        ]
        | [
            Op::LoadLocal(0),
            Op::LoadLocal(1),
            Op::Call { argc: 2, .. },
            Op::Return,
        ]
        | [Op::LoadLocal(0), Op::LoadLocal(1), Op::Call { argc: 2, .. }] => true,
        _ => false,
    }
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn plain_add_handled() -> i64 {
    PLAIN_ADD_HANDLED.with(Cell::get)
}

/// Two stack ints through the same `+` the portal already runs for `BIN_OP`.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn call_plain_add_abi(
    vm: &mut Vm<'_>,
    first: usize,
) -> Option<Box<crate::EvalAltResult>> {
    PLAIN_ADD_HANDLED.with(|cell| cell.set(0));
    if !live_vm_ptr(vm) || first + 1 >= vm.depth {
        return None;
    }
    match super::apply_binary(
        crate::grain::bytecode::BinOpKind::Add,
        super::stack_ref(vm, first),
        super::stack_ref(vm, first + 1),
    ) {
        Ok(Some(value)) => {
            vm.truncate_stack(first);
            vm.push_fast(value);
            PLAIN_ADD_HANDLED.with(|cell| cell.set(1));
            None
        }
        Ok(None) => None,
        Err(err) => Some(err),
    }
}

/// `add(s, i)` as CallRef: local receiver plus one stack arg.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn call_plain_add_ref_abi(
    vm: &mut Vm<'_>,
    scope: &mut Scope<'_>,
    base: usize,
    slot: u16,
    first: usize,
) -> Option<Box<crate::EvalAltResult>> {
    PLAIN_ADD_HANDLED.with(|cell| cell.set(0));
    if !live_vm_ptr(vm) || !live_scope_ptr(scope) {
        return None;
    }
    let at = base.saturating_add(slot as usize);
    if at >= scope.len() || first >= vm.depth {
        return None;
    }
    match super::apply_binary(
        crate::grain::bytecode::BinOpKind::Add,
        scope.get_mut_by_index(at),
        super::stack_ref(vm, first),
    ) {
        Ok(Some(value)) => {
            vm.truncate_stack(first);
            vm.push_fast(value);
            PLAIN_ADD_HANDLED.with(|cell| cell.set(1));
            None
        }
        Ok(None) => None,
        Err(err) => Some(err),
    }
}

/// `abs(x)` as CallRef: the receiver is the sole operand.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn call_plain_abs_ref_abi(
    vm: &mut Vm<'_>,
    scope: &mut Scope<'_>,
    base: usize,
    slot: u16,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    UNARY_BUILTIN_HANDLED.with(|cell| cell.set(0));
    if !live_vm_ptr(vm) || !live_scope_ptr(scope) {
        return None;
    }
    let at = base.saturating_add(slot as usize);
    if at >= scope.len() {
        return None;
    }
    let Ok(x) = scope.get_mut_by_index(at).as_int() else {
        return None;
    };
    let value = if cfg!(not(feature = "unchecked")) {
        match x.checked_abs() {
            Some(abs) => Dynamic::from(abs),
            None => {
                return Some(Box::new(crate::EvalAltResult::ErrorArithmetic(
                    format!("Negation overflow: -{x}"),
                    position_from_bits(position),
                )));
            }
        }
    } else {
        Dynamic::from(x.abs())
    };
    vm.push(value);
    UNARY_BUILTIN_HANDLED.with(|cell| cell.set(1));
    None
}

/// Whether a pooled name is the unary `abs` builtin, as one word.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn name_is_abs(program: &Program<'_>, name_index: u32) -> i64 {
    i64::from(program.name_plain(name_index) == Some("abs"))
}

/// Unary built-in (`abs` on an integer) after the typed/fast arms declined.
///
/// The full syntactic/stacked residual walks the engine's dispatch on
/// every call. A one-word residual for the primitive case is what the
/// binary operator path already does through [`operator_builtin_abi`].
/// `1` means it ran and pushed; `0` means fall through.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn unary_builtin_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    name_index: u32,
    first: usize,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    UNARY_BUILTIN_HANDLED.with(|cell| cell.set(0));
    if !live_vm_ptr(vm) || !live_count(first) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    if !vm.engine.fast_operators() {
        return None;
    }
    let name = program.name_plain(name_index).unwrap_or("");
    if name != "abs" {
        return None;
    }
    let Ok(x) = super::stack_ref(vm, first).as_int() else {
        return None;
    };
    let value = if cfg!(not(feature = "unchecked")) {
        match x.checked_abs() {
            Some(abs) => Dynamic::from(abs),
            None => {
                return Some(Box::new(crate::EvalAltResult::ErrorArithmetic(
                    format!("Negation overflow: -{x}"),
                    position_from_bits(position),
                )));
            }
        }
    } else {
        Dynamic::from(x.abs())
    };
    vm.push(value);
    UNARY_BUILTIN_HANDLED.with(|cell| cell.set(1));
    None
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn unary_builtin_handled() -> i64 {
    UNARY_BUILTIN_HANDLED.with(Cell::get)
}

/// Build a function pointer from a pooled name.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn make_closure_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    name_index: u32,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    match vm.make_closure_fnptr(program, name_index, position_from_bits(position)) {
        Ok(()) => None,
        Err(err) => Some(err),
    }
}

/// Share a named scope entry for a closure capture.
#[cfg(not(feature = "no_closure"))]
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn share_named_abi(
    vm: &mut Vm<'_>,
    program: &Program<'_>,
    scope: &mut Scope<'_>,
    name_index: u32,
    position: i64,
) -> Option<Box<crate::EvalAltResult>> {
    if !live_vm_ptr(vm) || !live_scope_ptr(scope) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    vm.share_named(program, scope, name_index, position_from_bits(position))
        .err()
}

/// Take a portal return stashed by [`stash_ok_result`].
///
/// Kept behind this boundary so `Option::take` never appears in `run_frame`:
/// that callee is a residual the build cannot address, and one such target
/// anywhere the portal reaches refuses every trace.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn take_finished_result(
    frame: &mut GrainFrame<'_, '_>,
    value: &mut Dynamic,
) -> Option<Box<crate::EvalAltResult>> {
    frame.jit_return_kind = 0;
    match frame.jit_finished.take() {
        Some(result) => match *result {
            Ok(result) => {
                *value = result;
                None
            }
            Err(err) => Some(err),
        },
        None => None,
    }
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
/// the same nullable exception pointer the builtin-apply residual returns.
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
    if !live_vm_ptr(vm) {
        majit_metainterp::request_walk_abort();
        return 0;
    }
    i64::from(vm.engine.fast_operators())
}

fn live_vm_ptr(vm: &Vm<'_>) -> bool {
    if (vm as *const Vm as usize) <= 0x1000 {
        return false;
    }
    (vm.engine as *const _ as usize) > 0x1000
}

fn live_scope_ptr(scope: &Scope<'_>) -> bool {
    (scope as *const Scope as usize) > 0x1000
}

fn live_count(n: usize) -> bool {
    n < 0x1_0000
}

fn live_stack_store(vm: &Vm<'_>, index: usize) -> bool {
    live_vm_ptr(vm) && index < vm.stack.len()
}

/// The slot exists and its `Box<Dynamic>` is a heap cell, not a walk-local
/// small integer. Assigning through the latter is `drop_glue` at `0x2`.
fn live_operand_box(vm: &Vm<'_>, index: usize) -> bool {
    live_stack_store(vm, index) && live_dynamic_ptr(super::operand_ref(&vm.stack[index]))
}

/// Grow if needed and refuse a walk-local destination. The caller writes the
/// slot and advances [`Vm::depth`] only when this returns `Some`.
fn prepare_fast_push(vm: &mut Vm<'_>) -> Option<usize> {
    if !live_vm_ptr(vm) || !live_count(vm.depth) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    let depth = vm.depth;
    if depth == vm.stack.len() {
        vm.grow_stack(1);
    }
    if !live_operand_box(vm, depth) {
        majit_metainterp::request_walk_abort();
        return None;
    }
    Some(depth)
}

fn dummy_unit() -> &'static Dynamic {
    thread_local! {
        static UNIT: std::cell::Cell<Option<&'static Dynamic>> =
            const { std::cell::Cell::new(None) };
    }
    UNIT.with(|cell| {
        if cell.get().is_none() {
            cell.set(Some(Box::leak(Box::new(Dynamic(Union::Unit(
                (),
                0,
                AccessMode::ReadWrite,
            ))))));
        }
        cell.get().expect("dummy unit is initialized")
    })
}

#[inline]
pub(super) fn track_operation_error(vm: &mut Vm<'_>, program: &Program<'_>, at: usize) {
    track_operation_abi(vm, code_position_bits(program, at));
}

/// The frame's scope, with the Vm red kept live at the load.
///
/// A plain `let scope = &mut *frame.scope` is a GETFIELD of the frame
/// alone. Once the merge-point Vm copy dies, the colourer reuses that
/// register for `scope`, and a mid-opcode guard snapshots the Scope
/// under the reserved vm index. Taking both reds here makes the three
/// refs interfere, so they keep distinct colours.
#[majit_macros::dont_look_inside_cannot_raise]
#[inline(never)]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn scope_from_frame<'a, 'f, 's>(
    frame: &'a mut super::GrainFrame<'f, 's>,
    vm: &Vm<'_>,
) -> &'a mut Scope<'s> {
    // The field read keeps `vm` in MIR. An unused `_vm` is dropped
    // before Charon sees the call, and the portal then has no
    // interference between the merge-point Vm and `scope`.
    let _ = vm.generation;
    &mut *frame.scope
}

/// Keep the merge-point Vm live past [`scope_from_frame`] so the two
/// refs interfere in the portal. An unused field read of `vm` inside
/// that helper is dropped before Charon sees the call.
#[majit_macros::dont_look_inside_cannot_raise]
#[inline(never)]
pub(super) extern "C" fn pin_scope_with_vm(vm: &Vm<'_>, scope: &Scope<'_>) -> i64 {
    i64::from(vm.generation != 0) | scope.len() as i64
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

/// Native entry for the original `Vm::grow_stack` jitcode.
///
/// `blackhole.py::BlackholeInterpreter.bhimpl_inline_call_ir_v` calls the
/// callee's native `fnaddr` even if tracing inlined its body. The cold growth
/// branch can be reached only after a guard fails, so binding just the calls
/// observed while recording the hot path is insufficient. This adapter keeps
/// the existing helper's allocation and initialization together and exposes
/// its exact `(Vm reference, extra count) -> void` signature as a C ABI.
#[majit_macros::dont_look_inside_cannot_raise]
#[inline(never)]
pub(super) extern "C" fn grow_stack_abi(vm: &mut Vm<'_>, extra: usize) {
    if !live_vm_ptr(vm) || !live_count(extra) {
        majit_metainterp::request_walk_abort();
        return;
    }
    majit_metainterp::note_residual_committed();
    vm.grow_stack(extra);
}

/// Push one integer without exposing a walk-local `Box<Dynamic>`.
///
/// The portal must not inline `operand_stack_store_int` plus a depth bump:
/// a walk-local destination drops `Union` at `0x2`, and incrementing depth
/// after a refused store corrupts the live stack.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn push_fast_int(vm: &mut Vm<'_>, value: crate::INT) {
    let Some(depth) = prepare_fast_push(vm) else {
        return;
    };
    *super::operand_mut(&mut vm.stack[depth]) =
        Dynamic(Union::Int(value, 0, AccessMode::ReadWrite));
    vm.depth = depth + 1;
}

/// Bool twin of [`push_fast_int`].
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn push_fast_bool(vm: &mut Vm<'_>, value: i64) {
    let Some(depth) = prepare_fast_push(vm) else {
        return;
    };
    *super::operand_mut(&mut vm.stack[depth]) =
        Dynamic(Union::Bool(value != 0, 0, AccessMode::ReadWrite));
    vm.depth = depth + 1;
}

/// Float twin of [`push_fast_int`].
#[cfg(not(feature = "no_float"))]
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn push_fast_float(vm: &mut Vm<'_>, value: crate::FLOAT) {
    let Some(depth) = prepare_fast_push(vm) else {
        return;
    };
    *super::operand_mut(&mut vm.stack[depth]) = Dynamic::from(value);
    vm.depth = depth + 1;
}

/// Unit twin of [`push_fast_int`].
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn push_fast_unit(vm: &mut Vm<'_>) {
    let Some(depth) = prepare_fast_push(vm) else {
        return;
    };
    *super::operand_mut(&mut vm.stack[depth]) = Dynamic(Union::Unit((), 0, AccessMode::ReadWrite));
    vm.depth = depth + 1;
}

/// Push a flatten-clone of a live cell. The cell is a scope or constant
/// slot, never a walk local.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn push_from_cell(vm: &mut Vm<'_>, cell: &Dynamic) {
    if !live_vm_ptr(vm) || !live_dynamic_ptr(cell) {
        majit_metainterp::request_walk_abort();
        return;
    }
    vm.push(super::value::flatten_clone_value(cell));
}

/// A walk-local `Dynamic` is not an ABI word. The walker still residual-calls
/// a bound helper with whatever is in the register, which for a lowered local
/// is a small integer (often a FastValue tag or a payload). Dropping that as
/// `Union` is `EXC_BAD_ACCESS` at that address. Refuse rather than dereference.
fn live_dynamic_ptr(value: &Dynamic) -> bool {
    let addr = value as *const Dynamic as usize;
    addr > 0x1000
}

/// `blackhole.py::bhimpl_inline_call_r_i` calls the original helper while
/// finishing a guard exit up to the next merge point, including its greens.
#[inline(never)]
pub(super) extern "C" fn program_jit_identity_abi(program: &Program) -> u64 {
    program.jit_identity()
}

/// Resolve one pointer-stable operand slot across the residual ABI.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn operand_stack_entry<'a>(vm: &'a Vm<'_>, index: usize) -> &'a Dynamic {
    if !live_stack_store(vm, index) {
        majit_metainterp::request_walk_abort();
        return dummy_unit();
    }
    super::operand_ref(&vm.stack[index])
}

/// Resolve one pointer-stable mutable operand slot across the residual ABI.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn operand_stack_entry_mut<'a>(
    vm: &'a mut Vm<'_>,
    index: usize,
) -> &'a mut Dynamic {
    // A walk-local index cannot be turned into a dummy `&mut` without
    // unsafe. The caller already tested the index; a refuse here would
    // still have to return a place. Keep the bound path only for a live
    // slot; the walker aborts `ri` stores via `request_walk_abort` on
    // the take/store siblings.
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
    if index >= vm.stack.len() || !live_dynamic_ptr(value) {
        majit_metainterp::request_walk_abort();
        return;
    }
    *value = core::mem::take(super::operand_mut(&mut vm.stack[index]));
}

/// Move one `Dynamic` into a pointer-stable operand slot.
///
/// A whole-value `*slot = value` currently reaches the MIR frontend as the
/// synthetic `__deref_write` marker, for which no executable host address can
/// exist. Keep that Rust aggregate store inside the opaque container ABI. The
/// mutable source makes this a move (`mem::take`), not an observable clone.
///
/// A walk local's address is not an ABI word, so compiled code that names a
/// lowered `Dynamic` as this source stores through null. Scalar siblings
/// below take one-word payloads instead.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn operand_stack_store(vm: &mut Vm<'_>, index: usize, value: &mut Dynamic) {
    if !live_stack_store(vm, index) || !live_dynamic_ptr(value) {
        // A walk-local index is a pointer-sized word, not a depth. A
        // walk-local `Dynamic` is a small integer, not a cell. Refuse
        // rather than dropping `Union` at that address.
        majit_metainterp::request_walk_abort();
        return;
    }
    dynamic_store(super::operand_mut(&mut vm.stack[index]), value);
}

/// Store an integer into a pointer-stable operand slot.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn operand_stack_store_int(vm: &mut Vm<'_>, index: usize, value: crate::INT) {
    if !live_operand_box(vm, index) {
        majit_metainterp::request_walk_abort();
        return;
    }
    *super::operand_mut(&mut vm.stack[index]) =
        Dynamic(Union::Int(value, 0, AccessMode::ReadWrite));
}

/// Store a bool into a pointer-stable operand slot.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn operand_stack_store_bool(vm: &mut Vm<'_>, index: usize, value: i64) {
    if !live_operand_box(vm, index) {
        majit_metainterp::request_walk_abort();
        return;
    }
    *super::operand_mut(&mut vm.stack[index]) =
        Dynamic(Union::Bool(value != 0, 0, AccessMode::ReadWrite));
}

/// Store a float into a pointer-stable operand slot.
#[cfg(not(feature = "no_float"))]
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn operand_stack_store_float(
    vm: &mut Vm<'_>,
    index: usize,
    value: crate::FLOAT,
) {
    if !live_operand_box(vm, index) {
        majit_metainterp::request_walk_abort();
        return;
    }
    *super::operand_mut(&mut vm.stack[index]) = Dynamic::from(value);
}

/// Store unit into a pointer-stable operand slot.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn operand_stack_store_unit(vm: &mut Vm<'_>, index: usize) {
    if !live_operand_box(vm, index) {
        majit_metainterp::request_walk_abort();
        return;
    }
    *super::operand_mut(&mut vm.stack[index]) = Dynamic(Union::Unit((), 0, AccessMode::ReadWrite));
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
    if !live_dynamic_ptr(target) || !live_dynamic_ptr(value) {
        majit_metainterp::request_walk_abort();
        return;
    }
    *target = core::mem::take(value);
}

/// Drop every operand above `depth` and update the red VM's stack depth.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn truncate_stack(vm: &mut Vm<'_>, depth: usize) {
    if !live_vm_ptr(vm) || !live_count(depth) {
        majit_metainterp::request_walk_abort();
        return;
    }
    majit_metainterp::note_residual_committed();
    vm.truncate_stack(depth);
}

/// `Vec::len` / `ThinVec::len` on the running `for` stack, as one ABI word.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn iterators_len(vm: &Vm<'_>) -> i64 {
    vm.iterators_len() as i64
}

/// Drop the innermost `for` iterator without exposing `Vec::pop`.
///
/// Exhaust of a compiled range loop blackholes through this path to the
/// next merge point. A walk-abort here becomes `LeaveFrame` (`usize::MAX`)
/// and the native loop re-decodes the body, adding the last item twice.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn iterators_pop(vm: &mut Vm<'_>) {
    if !live_vm_ptr(vm) {
        return;
    }
    let len = vm.iterators_len();
    if len == 0 {
        return;
    }
    majit_metainterp::note_residual_committed();
    vm.iterators_truncate(len - 1);
}

/// `Vec::truncate` on the running `for` stack.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn iterators_truncate(vm: &mut Vm<'_>, len: usize) {
    if !live_vm_ptr(vm) || !live_count(len) {
        majit_metainterp::request_walk_abort();
        return;
    }
    majit_metainterp::note_residual_committed();
    vm.iterators_truncate(len);
}

/// Handler-stack twins of [`iterators_len`].
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn handlers_len(vm: &Vm<'_>) -> i64 {
    vm.handlers_len() as i64
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn handlers_truncate(vm: &mut Vm<'_>, len: usize) {
    if !live_vm_ptr(vm) || !live_count(len) {
        majit_metainterp::request_walk_abort();
        return;
    }
    majit_metainterp::note_residual_committed();
    vm.handlers_truncate(len);
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn sizes_len(vm: &Vm<'_>) -> i64 {
    vm.sizes_len() as i64
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn sizes_truncate(vm: &mut Vm<'_>, len: usize) {
    if !live_vm_ptr(vm) || !live_count(len) {
        majit_metainterp::request_walk_abort();
        return;
    }
    majit_metainterp::note_residual_committed();
    vm.sizes_truncate(len);
}

/// `Scope::rewind` without exposing `ThinVec::truncate`.
#[majit_macros::dont_look_inside_cannot_raise]
#[allow(improper_ctypes_definitions)]
pub(super) extern "C" fn scope_rewind(scope: &mut Scope<'_>, size: usize) {
    if !live_scope_ptr(scope) || !live_count(size) {
        majit_metainterp::request_walk_abort();
        return;
    }
    majit_metainterp::note_residual_committed();
    scope.rewind(size);
}

/// The lowering still emits a synthetic `__len` for some containers.
/// Calling it with a walk-local is a null GETFIELD. Abort the walk.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn refuse_synthetic_len(_obj: i64) -> i64 {
    majit_metainterp::request_walk_abort();
    0
}

/// Twin of [`refuse_synthetic_len`] for `ThinVec::truncate`.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn refuse_thinvec_truncate(_obj: i64, _len: i64) {
    majit_metainterp::request_walk_abort();
}

/// Synthetic write through a walk-local. Abort rather than store through null.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn refuse_deref_write(_obj: i64, _value: i64) {
    majit_metainterp::request_walk_abort();
}

/// The walk-abort hook, as a bound one-word residual.
///
/// Calling `request_walk_abort` from the portal leaves it as an unbound
/// symbolic residual and aborts setup. Behind this boundary the hook is a
/// real address.
#[majit_macros::dont_look_inside_cannot_raise]
pub(super) extern "C" fn request_walk_abort_abi() {
    majit_metainterp::request_walk_abort();
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
        let mut descriptor = jit_state::grain_driver_descriptor();
        // pypyjitdriver.is_recursive: script-fn CALL's recursive
        // `run_frame` is a portal call `can_inline` may look inside.
        descriptor.is_recursive = true;
        let green_types = descriptor.green_args_spec();
        let mut driver = JitDriver::with_descriptor(THRESHOLD, descriptor);
        driver.set_is_recursive(true);
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
        driver.register_blackhole_allocator(majit_metainterp::resume::LlmodelBlackholeAllocator);
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
        if RESIDUAL_NEST.with(Cell::get) != 0 {
            return;
        }
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

        // Compiled code holds `RUNTIME` for the whole residual it called.
        // A nested `run_frame` (script-fn CALL) cannot consult, so decline
        // before building the door's green key. `add` has no back edge of
        // its own; the consult cannot compile it.
        if RUNTIME
            .try_with(|cell| cell.try_borrow_mut().is_err())
            .unwrap_or(true)
        {
            bump_stats(|stats| stats.reentrant_consultations_declined += 1);
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
                values: green_values.to_vec().into(),
                types: green_types.clone().into(),
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
            let mut resume = runtime.driver.back_edge_structured(
                green_hash,
                green_key,
                pc,
                &mut runtime.state,
                &env,
                || {},
            );
            let started = !was_tracing && runtime.driver.is_tracing();
            if runtime.driver.take_back_edge_finish().is_some() {
                // Compiled code finished the portal. The walk stashed the
                // return on the live frame; native `run_frame` takes it.
                resume_pc = None;
            } else {
                // `warmspot.py` `EnterJitAssembler`: after a guard failure
                // blackholes to this merge-point PC, or after a bridge is
                // compiled and patched, re-enter the compiled loop here
                // instead of handing the interpreter the rest of the run.
                // Each re-entry sees state the blackhole already advanced,
                // so a still-unpatched arm makes forward progress rather
                // than spinning.
                if !started && !runtime.driver.is_tracing() {
                    let mut hops = 0u32;
                    while hops < 1_000_000 {
                        let Some(next) = resume else { break };
                        if next == usize::MAX || next != pc {
                            break;
                        }
                        if !runtime.driver.has_compiled_loop(green_hash) {
                            break;
                        }
                        hops += 1;
                        runtime.state.publish_live(&env, &frame.jit_vable_words());
                        resume = runtime.driver.back_edge_structured(
                            green_hash,
                            green_key,
                            next,
                            &mut runtime.state,
                            &env,
                            || {},
                        );
                        if runtime.driver.take_back_edge_finish().is_some() {
                            resume = None;
                            break;
                        }
                    }
                }
                resume_pc = resume;
            }
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

                    let mut stack = majit_metainterp::StandaloneFrameStack::new();
                    let frame = majit_metainterp::setup_frame_from_merge_point(
                        ctx,
                        &mut stack.frames,
                        Arc::clone(&portal),
                        header_pc,
                        &green_args,
                        &red_args,
                    );
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
                    // Same handoff `trace_jitcode_with_args_and_runtime`
                    // publishes: an abort after a bound residual must not
                    // replay the merge-point opcode.
                    majit_metainterp::publish_walk_abort_handoff(ctx, &action, &mut stack);
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
            let finished = runtime.driver.take_single_pass_finish();
            if finished || outcome.is_some() {
                runtime
                    .driver
                    .writeback_scalar_state_fields(&mut runtime.state);
                runtime
                    .driver
                    .writeback_ref_scalar_state_fields(&mut runtime.state);
                runtime
                    .driver
                    .writeback_virt_array_state_fields(&mut runtime.state);
                runtime.state.recover_after_compiled_run();
                runtime
                    .driver
                    .arm_single_pass_label_entry_on_next_back_edge(&runtime.state);
                runtime.driver.discard_single_pass_resume();
            }
            if finished {
                // The walk finished the portal. `stash_ok_result` wrote the
                // return onto the live frame; do not resume at a source pc.
                resume_pc = None;
            } else if let Some((pc, reds)) = outcome {
                debug_assert!(
                    reds.is_empty(),
                    "Grain's red operands are live frame/VM references, not a scalar state bank",
                );
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
        if let Some(resume_at) = resume_pc {
            // usize::MAX is LeaveFrame from a blackhole that could not
            // finish (`convert_and_run_from_pyjitpl` had no merge point).
            // Staying here re-decodes this pc. For a compiled `for` that
            // is the body, so the last `s += i` runs again. Advance past
            // this instruction instead, the way a failed `JUMP_IF` resumes
            // after the test. `while` reports a real post-loop pc and
            // never takes this arm.
            if resume_at == usize::MAX {
                if range_exhausted(vm) {
                    if let Some(header) = next_iter_header(program.code(), pc) {
                        frame.jit_resume_pc_plus_one = header + 1;
                    }
                }
            } else if let Some(resume) = resume_at.checked_add(1) {
                frame.jit_resume_pc_plus_one = resume;
            }
        }

        if let Some(event) = report_event {
            report_if_enabled(event);
        }
    }
}

/// Bytecode address of the next `ITER_NEXT*` after `from`, inclusive.
///
/// A compiled rotated `for` resumes at the body. When the blackhole cannot
/// name a merge point, the native loop must run the header's exhaust, not
/// the first body opcode.
fn range_exhausted(vm: &Vm<'_>) -> bool {
    match vm.iterators_last().map(|iteration| &iteration.items) {
        Some(super::Items::IntRange { next, end }) => *next >= *end,
        _ => false,
    }
}

fn next_iter_header(code: &[u8], from: usize) -> Option<usize> {
    let mut at = from;
    for _ in 0..64 {
        let tag = *code.get(at)?;
        if matches!(
            tag,
            code::tag::ITER_NEXT | code::tag::ITER_NEXT_INDEXED | code::tag::ITER_NEXT_STORE
        ) {
            return Some(at);
        }
        at += crate::grain::bytecode::code::width(code, at)?;
    }
    None
}
