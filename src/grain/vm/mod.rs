use core::any::TypeId;
use core::mem;
#[cfg(feature = "no_std")]
use std::prelude::v1::*;

// Indexing survives either feature alone — a map is indexed by string, an
// array by number — and only goes when both do. `eval/chaining.rs` is gated on
// exactly this, and takes the getter and setter names with it.
// Measuring a value is measuring what an array, a map or a string holds, so
// the same pair of features takes it away — see the note above.
#[cfg(not(feature = "unchecked"))]
#[cfg(not(all(feature = "no_index", feature = "no_object")))]
use crate::eval::calc_data_sizes;
use crate::eval::{Caches, FnResolutionCacheEntry, GlobalRuntimeState};
use crate::func::native::FnBuiltin;
use crate::func::{
    get_builtin_binary_op_fn, get_builtin_op_assignment_fn, is_syntactic_fn_name, CallSite,
};
use crate::packages::string_basic::print_with_func;
use crate::tokenizer::Token;
use crate::types::dynamic::{AccessMode, DynamicWriteLock, Union};
use crate::types::fn_ptr::FnPtrType;
use crate::types::StringsInterner;
// `Variant` is only re-exported from the crate root under `internals`, so it
// comes from where it is defined.
use crate::ast::{Expr, FnCallHashes};
#[cfg(not(feature = "no_index"))]
use crate::Array;
#[cfg(not(feature = "no_object"))]
use crate::Map;
#[cfg(not(feature = "no_function"))]
use crate::{types::dynamic::Variant, CallFnOptions};
use crate::{
    Dynamic, Engine, EvalAltResult, EvalContext, FnArgsVec, FnPtr, ImmutableString,
    NativeCallContext, Position, RhaiResultOf, Scope, FUNC_TO_STRING, INT,
};

mod arith;
mod callback;
#[cfg(feature = "grain-jit")]
mod jit;
#[cfg(feature = "grain-jit")]
pub mod jit_state;
#[cfg(feature = "grain-jit")]
pub mod jitcodes;
mod value;

use value::{clone_value, flatten_clone_value, overwrite, release};

use crate::grain::bytecode::{
    code, AssignOp, BinOpKind, BinOperand, Chain, Chunk, Receiver, Root, Step, StepFlags, Tail,
    UnOpKind,
};
use crate::grain::program::{Program, SharedModule, SharedProgram};

pub(crate) use callback::CrossingPool;

/// The pool the run a crossing arrived from is lending, if it lends one.
///
/// Reached through the state rather than through the closure that was called,
/// because that state is what a crossing inherits and a closure is not: see
/// [`callback::Crossings`].
#[must_use]
fn crossing_pool<'a>(context: &'a NativeCallContext<'_>) -> Option<&'a CrossingPool> {
    context.global_runtime_state().grain_crossings.as_ref()
}

/// Rhai's own `RhaiResult`, which it does not re-export.
pub type VmResult = Result<Dynamic, Box<EvalAltResult>>;

/// Whether a value is a shared cell.
///
/// Sharing is how closures capture, so under `no_closure` there are no cells,
/// `Dynamic` has no `is_shared` to call, and the answer is a constant. The
/// opcodes that create and read cells are compiled out with it.
#[cfg(not(feature = "no_closure"))]
macro_rules! is_shared {
    ($value:expr) => {
        $value.is_shared()
    };
}
#[cfg(feature = "no_closure")]
macro_rules! is_shared {
    ($value:expr) => {{
        let _ = &$value;
        false
    }};
}

/// Take an `Option`'s value, or leave through the error the second argument
/// builds.
///
/// The check open-coded rather than written through `Option::ok_or_else`:
/// `core`'s combinator is opaque to anything reading this loop from outside,
/// the same reason the operand decoders in `execute` are macros and not
/// closures. `pyopcode.py:868` spells the same guard as a test and a raise.
/// The error expression is still evaluated only on the miss, so a `format!`
/// on the failure path costs nothing on the way through.
macro_rules! or_raise {
    ($option:expr, $error:expr) => {
        match $option {
            Some(value) => value,
            None => return Err($error),
        }
    };
}

/// Make a function call into Rhai using [`dispatch_fn`](crate::eval::dispatch_fn).
///
/// `dispatch_fn` rather than `_call_fn_raw` because this is an evaluator making
/// a call, not a native reentering one: the extra level `_call_fn_raw` takes is
/// the boundary a native crosses, and there is no boundary here. Taking it
/// would spend `max_call_levels` at twice the walker's rate.
///
/// `site` is what the call site already worked out about itself, for a site
/// that has run before and kept it. See [`CallMemo`].
#[inline(always)]
fn call_engine(
    engine: &Engine,
    global: &mut GlobalRuntimeState,
    caches: &mut Caches,
    scope: &mut Scope,
    fn_name: &str,
    args: &mut [&mut Dynamic],
    is_ref_mut: bool,
    is_method_call: bool,
    pos: Position,
    site: Option<&CallSite>,
) -> VmResult {
    // A site is only ever made for a name a script function could carry, so a
    // site that exists has already answered this. Asserted rather than assumed,
    // because widening [`memoisable_name`] would otherwise send a call down the
    // other of the dispatch's two branches without anything saying so.
    debug_assert!(
        site.is_none() || is_identifier(fn_name),
        "`{fn_name}` has a call site but is a name only a native can carry"
    );
    let native_only = site.is_none() && !is_identifier(fn_name);

    crate::eval::dispatch_fn(
        engine,
        global,
        caches,
        scope,
        fn_name,
        args,
        native_only,
        is_ref_mut,
        is_method_call,
        pos,
        site,
    )
}

/// Whether Rhai would look for a script function of this name, rather than
/// treating the name as one only a native can carry.
///
/// `func/native.rs:475`, which is where the dispatch decides it.
#[inline]
#[must_use]
fn is_identifier(fn_name: &str) -> bool {
    let identifier = crate::tokenizer::is_valid_identifier(fn_name);
    #[cfg(not(feature = "no_function"))]
    let identifier = identifier || crate::parser::is_anonymous_fn(fn_name);
    identifier
}

/// Stamp the call site on an error that passes through a function boundary unwrapped,
/// as Rhai does for exits and system exceptions (`func/script.rs:134`).
fn reposition(mut err: Box<EvalAltResult>, pos: Position) -> Box<EvalAltResult> {
    err.set_position(pos);
    err
}

/// Take back off the scope whatever a debugger callback left on it.
///
/// Slots are positions. An entry a callback declares sits underneath every
/// declaration the chunk has yet to make, so the next `let` would land above it
/// and every slot from there on would name the variable before the one it meant
/// — silently, and in a program that has nothing to do with the debugger. Rhai
/// answers this by searching by name from then on (`eval/debugger.rs:436`); a
/// chunk has no names to search, so the scope goes back to the shape it was
/// compiled against instead, and what a callback declares does not outlive the
/// stop it declared it at.
#[cfg(feature = "debugging")]
fn rewind_after_stop(scope: &mut Scope, before: usize) {
    if scope.len() > before {
        scope.rewind(before);
    }
}

/// Stamp the site on an error that arrived without one.
///
/// Unlike [`reposition`] this never overwrites a position the callee already set.
fn positioned(err: Box<EvalAltResult>, pos: Position) -> Box<EvalAltResult> {
    if err.position().is_none() {
        reposition(err, pos)
    } else {
        err
    }
}

/// Stamp the call site on anything a dispatched call came back with.
///
/// This is `fill_position`, which Rhai applies to the whole of
/// `exec_native_fn_call` — the callee not being found, the call being refused,
/// and the error a native *returned* alike (`func/call.rs:365`, `:406`, `:413`).
///
/// What looks like a counter-example is not one: `1 / 0` reports
/// `ErrorArithmetic` with no position at all, because under `fast_operators` a
/// binary operator returns the built-in's error without going through here
/// (`func/call.rs:1798`). The VM's own fast path skips it for the same reason.
fn dispatch_failure(err: Box<EvalAltResult>, pos: Position) -> Box<EvalAltResult> {
    positioned(err, pos)
}

/// A/B gate — a measurement scaffold, deleted with the measurement it serves.
///
/// A candidate whose two spellings produce the same bytecode cannot be graded
/// by building two programs, and grading it by building two binaries compares
/// two code layouts rather than two spellings (`examples/grain_nsiter.rs`).
/// So both spellings are compiled into this binary and one relaxed load picks
/// between them. Both legs pay that load, so what is compared is the two
/// spellings and not one leg's branch against the other's absence of one.
#[doc(hidden)]
pub mod ab_gate {
    use core::sync::atomic::{AtomicBool, Ordering};

    static ON: AtomicBool = AtomicBool::new(false);

    /// Pick the leg the gated sites take from here on.
    #[inline]
    pub fn set(on: bool) {
        ON.store(on, Ordering::Relaxed);
    }

    /// Which leg a gated site should take.
    #[must_use]
    #[inline(always)]
    pub fn on() -> bool {
        ON.load(Ordering::Relaxed)
    }
}

/// How many operator sites the memo below holds at once.
///
/// Direct-mapped and small on purpose. A memo is worth having because a site
/// that ran once will almost certainly run again with the same operand types,
/// not because a chunk's operators are numerous — and a table big enough to
/// hold all of them would cost more to clear per frame than it saves.
const OPERATOR_MEMO_SLOTS: usize = 16;

/// What an operator site resolved to, the last time it ran.
///
/// The point is [`get_builtin_binary_op_fn`], which walks `(&x.0, &y.0, op)`
/// and then the operator token to hand back a function pointer. That walk
/// answers the same thing every time a site sees the same pair of operand
/// types, and a monomorphic site is the common one — so it is done once and
/// remembered against the instruction's own address.
///
/// [`Op::BinOp`](crate::grain::bytecode::Op::BinOp) is what removed the
/// *integer* sites from this population: they never reach a resolution at all.
/// What is left is strings, characters, booleans, float comparisons and the
/// mixed pairs, which is exactly what this exists for.
#[derive(Clone, Copy)]
struct OperatorMemo {
    /// Which frame wrote it. See [`Vm::operator_generation`].
    generation: u64,
    /// The instruction's own address, within the program that frame is running.
    at: u32,
    /// The operand pair it was resolved against, as [`type_code`] answers.
    /// Never zero in a live entry: a zero byte is what says "do not memoise".
    operands: u16,
    /// What the resolution said — including that there was no built-in, which
    /// is worth remembering for the same reason the positive answer is.
    resolved: Option<FnBuiltin>,
}

impl OperatorMemo {
    /// An entry no lookup can hit: generation zero is what a frame never has,
    /// because [`Vm::operator_generation`] is incremented before it is read.
    const EMPTY: Self = Self {
        generation: 0,
        at: 0,
        operands: 0,
        resolved: None,
    };
}

/// Which of `Union`'s arms a value is, or zero for the two that do not decide
/// a resolution by themselves.
///
/// [`get_builtin_binary_op_fn`] dispatches on the arm for every pair it names,
/// and on `Dynamic::type_id` for the rest. Those agree — the arm fixes the
/// type — for every arm but two: a trait object's type is the boxed value's,
/// and a shared cell's is whatever is inside the lock. Neither may be
/// memoised against a discriminant, so neither gets a code.
#[inline]
#[must_use]
fn type_code(value: &Dynamic) -> u8 {
    match value.0 {
        Union::Unit(..) => 1,
        Union::Bool(..) => 2,
        Union::Str(..) => 3,
        Union::Char(..) => 4,
        Union::Int(..) => 5,
        #[cfg(not(feature = "no_float"))]
        Union::Float(..) => 6,
        #[cfg(feature = "decimal")]
        Union::Decimal(..) => 7,
        #[cfg(not(feature = "no_index"))]
        Union::Array(..) => 8,
        #[cfg(not(feature = "no_index"))]
        Union::Blob(..) => 9,
        #[cfg(not(feature = "no_object"))]
        Union::Map(..) => 10,
        Union::FnPtr(..) => 11,
        #[cfg(not(feature = "no_time"))]
        Union::TimeStamp(..) => 12,
        _ => 0,
    }
}

/// The built-in for this operator and this operand pair, from the memo if it
/// is there and from Rhai if it is not.
///
/// A free function rather than a method because the operands are borrowed out
/// of `Vm::stack` and the memo is a different field of the same `Vm`.
#[inline]
fn resolve_operator(
    memo: &mut [OperatorMemo; OPERATOR_MEMO_SLOTS],
    generation: u64,
    at: usize,
    token: &Token,
    lhs: &Dynamic,
    rhs: &Dynamic,
) -> Option<FnBuiltin> {
    let operands = (u16::from(type_code(lhs)) << 8) | u16::from(type_code(rhs));

    // One of the two arms whose answer is not its discriminant. Resolve it and
    // remember nothing, which is what keeps the memo from ever being consulted
    // for a value whose type it cannot see.
    if operands & 0x00ff == 0 || operands & 0xff00 == 0 {
        return get_builtin_binary_op_fn(token, lhs, rhs);
    }

    let slot = &mut memo[at & (OPERATOR_MEMO_SLOTS - 1)];
    if slot.generation == generation && slot.at as usize == at && slot.operands == operands {
        return slot.resolved;
    }

    let resolved = get_builtin_binary_op_fn(token, lhs, rhs);
    *slot = OperatorMemo {
        generation,
        at: at as u32,
        operands,
        resolved,
    };
    resolved
}

/// How many call sites the memo below holds at once.
///
/// Direct-mapped and small for [`OPERATOR_MEMO_SLOTS`]'s reasons, and keyed on
/// the name rather than the address so that two sites calling the same
/// function of the same arity share one entry.
const CALL_MEMO_SLOTS: usize = 16;

/// The table itself, which a [`Vm`] holds only once it has memoised a call.
///
/// It is 1.4KB, and a crossing builds a whole `Vm` per element a native calls
/// back over ([`callback::invoke`](crate::grain::vm::callback)) — so for a
/// chunk that calls nothing, which is what a `map` or a `filter` body usually
/// is, this was the largest thing a crossing zeroed and then walked again to
/// drop. Made where the first site is memoised instead. See [`call_site`].
type CallMemoTable = [Option<CallMemo>; CALL_MEMO_SLOTS];

/// What a call site resolved to, the last time it ran.
///
/// Rhai answers a call by name: it hashes the name with the argument count, and
/// then the argument types on top of that, and looks the result up in the
/// resolution cache the run carries (`func/call.rs` `resolve_fn`). The walker
/// pays only the second half, because its parser worked out the first and left
/// it in the AST node (`FnCallExpr::hashes`). An instruction has nowhere to
/// leave it, so without this a site in a loop hashes its own name once a turn
/// and asks the same three questions again every time.
///
/// This is where the answers are left instead. What an entry replaces is a
/// hash of the name, a lookup for a script function of that name, the check for
/// a syntactic call, and the resolution itself — all of which Rhai skips when
/// handed a [`CallSite`], and none of which it skips otherwise.
///
/// Four things make an entry unreadable rather than wrong, and every one of
/// them is checked before it is used:
///
/// * **Another frame.** `generation` is the frame that made it. Two frames can
///   be running two programs, where equal name indices mean different names —
///   and a frame is also the largest span over which the modules a resolution
///   searches cannot change. See [`Vm::generation`] and [`memoisable_state`].
/// * **Other argument types.** `args` is the types it resolved against; a value
///   whose type its discriminant does not fix stops an entry being made at all.
///   See [`arg_codes`].
/// * **Another site in the same slot.** The table is direct-mapped, so `name`
///   and `argc` say whether what is in a slot belongs to this site.
/// * **A site the memo may not answer for**, which is remembered as an entry
///   that says so — otherwise deciding it would cost every call what it saves.
struct CallMemo {
    /// Which frame made it. See [`Vm::generation`].
    generation: u64,
    /// The name pool index of the site it belongs to.
    name: u32,
    /// That site's argument count.
    argc: u8,
    /// The argument types it resolved against, four bits each and the first
    /// argument lowest. See [`arg_codes`].
    args: u32,
    /// How many modules were imported when it was made. See [`num_imports`].
    imports: usize,
    /// What Rhai's resolution answered, with the hashes the site would have
    /// computed to ask it — or [`None`] for a site that must go the ordinary
    /// way, which is worth remembering for the same reason the answer is.
    resolved: Option<(FnCallHashes, FnResolutionCacheEntry)>,
}

impl CallMemo {
    /// Whether this entry is the one a lookup is asking for.
    #[inline]
    #[must_use]
    fn answers(&self, generation: u64, name: u32, argc: usize, args: u32, imports: usize) -> bool {
        self.generation == generation
            && self.name == name
            && usize::from(self.argc) == argc
            && self.args == args
            && self.imports == imports
    }

    /// What this entry says to hand Rhai, if it says anything.
    #[inline]
    #[must_use]
    fn site(&self) -> Option<CallSite<'_>> {
        self.resolved.as_ref().map(|(hashes, entry)| CallSite {
            // A site is only made for a name no operator is spelled with, so
            // this is what the dispatch would have looked up. See
            // [`memoisable_name`].
            op_token: None,
            hashes: *hashes,
            resolved: Some(entry),
        })
    }
}

/// Which slot a site's entry lives in.
#[inline]
#[must_use]
fn call_memo_slot(name: u32, argc: usize) -> usize {
    (name as usize ^ argc) & (CALL_MEMO_SLOTS - 1)
}

/// The argument types packed four bits each, or [`None`] for a call no entry
/// may be keyed on.
///
/// [`None`] for a call with more arguments than fit, and for one carrying a
/// value whose type its discriminant does not fix ([`type_code`]) — a trait
/// object's is the boxed value's and a shared cell's is whatever is inside the
/// lock, and Rhai keys a resolution on the type rather than on the arm.
#[inline]
#[must_use]
fn arg_codes(args: &[&mut Dynamic]) -> Option<u32> {
    if args.len() > 8 {
        return None;
    }
    let mut packed = 0;
    for (index, arg) in args.iter().enumerate() {
        match type_code(arg) {
            0 => return None,
            code => packed |= u32::from(code) << (index * 4),
        }
    }
    Some(packed)
}

/// How many modules the run has imported, or zero where there are no modules.
///
/// A resolution searches the function libraries, the global modules and the
/// imported ones (`func/call.rs` `resolve_fn`). The first two cannot change
/// under a running frame — a library is stacked around a whole run
/// ([`Vm::with_environment`]) and registering with the engine wants it by
/// mutable reference, which a run holds by shared one — and neither can the
/// third, because the two statements that would declare one both defeat the
/// lowering outright (`compile/mod.rs`, the `Stmt::Import` and `KEYWORD_EVAL`
/// arms), so a compiled frame has none to run.
///
/// The frame a `Vm` was reentered from may still have imported some, so this is
/// part of an entry's key rather than a condition on making one: a count is
/// what says the search this entry answered is the search being made.
#[inline]
#[must_use]
fn num_imports(global: &GlobalRuntimeState) -> usize {
    #[cfg(not(feature = "no_module"))]
    return global.num_imports();
    #[cfg(feature = "no_module")]
    {
        let _ = global;
        0
    }
}

/// Whether a call by this name resolves the way an entry assumes.
///
/// Two names do not: one Rhai answers itself
/// ([`is_syntactic_fn_name`](crate::func::is_syntactic_fn_name)), and one no
/// script function could carry, which the dispatch sends down a path of its own
/// ([`is_identifier`]). A site is made only for the names that are neither.
#[inline]
#[must_use]
fn memoisable_name(name: &str) -> bool {
    is_identifier(name) && !is_syntactic_fn_name(name)
}

/// What Rhai's own dispatch would find for this site, for a site an entry may
/// be made for at all.
///
/// This is the work an entry exists to move: the hashes the dispatch computes
/// from the name and the argument count, the lookups that rule out a script
/// function of that name, and the resolution itself — reached through Rhai's
/// own `resolve_fn`, with the same hash, the same arguments and the same
/// `allow_dynamic`, so what is kept is what the dispatch would have used.
///
/// [`None`] where a site may not be answered from an entry, which is a name
/// Rhai answers itself, a name only a native can carry, a script function that
/// would win, or a call that resolves to nothing at all.
fn resolve_site(
    engine: &Engine,
    global: &GlobalRuntimeState,
    caches: &mut Caches,
    name: &str,
    args: &mut [&mut Dynamic],
) -> Option<(FnCallHashes, FnResolutionCacheEntry)> {
    if !memoisable_name(name) {
        return None;
    }

    // The hashes the dispatch computes for a call that is not method-style,
    // which the two callers of this both are: passing a variable by reference
    // is Rhai's rewrite for reaching a `&mut` first argument and does not
    // change how the name is keyed (`eval/eval_context.rs` `_call_fn_raw`).
    let hashes = FnCallHashes::from_hash(crate::calc_fn_hash(None, name, args.len()));

    // A script function of this name is reached before any native, and which of
    // the two a site finds is exactly what an entry may not be wrong about.
    #[cfg(not(feature = "no_function"))]
    if engine.has_script_fn(global, caches, hashes.script()) {
        return None;
    }

    let local_entry = &mut None;
    engine
        .resolve_fn(
            global,
            caches,
            local_entry,
            None,
            hashes.native(),
            Some(args),
            true,
        )
        .filter(|entry| entry.func.is_native())
        .cloned()
        .map(|entry| (hashes, entry))
}

/// The site to hand Rhai for this call, or [`None`] for a call that has to go
/// the ordinary way.
///
/// Works the answer out and keeps it when there is nothing to read back, so a
/// site costs a resolution once per frame rather than once per call. See
/// [`CallMemo`].
///
/// A free function rather than a method because the arguments are borrowed out
/// of the operand stack and the scope, and the memo is a different field of the
/// same `Vm`.
#[allow(clippy::too_many_arguments)]
fn call_site<'m>(
    memo: &'m mut Option<Box<CallMemoTable>>,
    engine: &Engine,
    global: &GlobalRuntimeState,
    caches: &mut Caches,
    generation: u64,
    name_index: u32,
    name: &str,
    args: &mut [&mut Dynamic],
) -> Option<&'m CallMemo> {
    let argc = args.len();
    let codes = arg_codes(args)?;
    let imports = num_imports(global);
    let slot = call_memo_slot(name_index, argc);

    // After the two questions that can refuse a site, so a call that may not be
    // memoised at all does not leave a table behind.
    let memo = memo.get_or_insert_with(|| Box::new(core::array::from_fn(|_| None)));

    if memo[slot].as_ref().map_or(true, |entry| {
        !entry.answers(generation, name_index, argc, codes, imports)
    }) {
        memo[slot] = Some(CallMemo {
            generation,
            name: name_index,
            // Bounded by [`arg_codes`], which refuses anything wider.
            argc: argc as u8,
            args: codes,
            imports,
            resolved: resolve_site(engine, global, caches, name, args),
        });
    }

    memo[slot].as_ref()
}

/// The operator applied to the operands themselves, for the pairs the dispatch
/// loop runs without resolving anything.
///
/// `None` is not a refusal — it is "this pair is not one of them", and the
/// caller goes on to do exactly what it would have done without this. So the
/// arms here decide speed and the arms missing from here decide nothing.
///
/// The integer pair is tested first because it has to be: the float rules
/// cover float/int and int/float but never int/int, and widening two integers
/// would answer a question Rhai answers with integer arithmetic.
///
/// `inline(always)`, because the dispatch loop asks this from two places — the
/// operands where they are named and the operands on the stack — and two call
/// sites are enough for the inliner to give up on a body this size and leave a
/// call in the arm every operator runs through. The arm is issue-bound rather
/// than fetch-bound, so the copy is cheaper than the call.
#[inline(always)]
fn apply_binary(kind: BinOpKind, lhs: &Dynamic, rhs: &Dynamic) -> RhaiResultOf<Option<Dynamic>> {
    if let (Union::Int(x, ..), Union::Int(y, ..)) = (&lhs.0, &rhs.0) {
        return arith::int_binary(kind, *x, *y);
    }

    #[cfg(not(feature = "no_float"))]
    if let (Some((x, x_is_float)), Some((y, y_is_float))) =
        (arith::as_float_operand(lhs), arith::as_float_operand(rhs))
    {
        if x_is_float || y_is_float {
            return Ok(arith::float_binary(kind, x, y));
        }
    }

    Ok(None)
}

/// The same for `op x`.
///
/// The coverage here is not a built-in table's — there is no unary one — it is
/// the walker's. `eval_fn_call_expr` short-circuits exactly one unary operator
/// under `fast_operators`, `!` on a `Union::Bool`, and resolves a function for
/// every other one. Answering for a second pair would answer a host-registered
/// unary operator differently from the tree this VM replaces.
///
/// A shared cell is not flattened first, matching [`apply_binary`]: it answers
/// `None` and the caller dispatches, which reaches the same function the
/// walker reaches for it.
#[inline]
fn apply_unary(kind: UnOpKind, operand: &Dynamic) -> Option<Dynamic> {
    match (kind, &operand.0) {
        (UnOpKind::Not, Union::Bool(b, ..)) => Some((!*b).into()),
        _ => None,
    }
}

/// The same for `x op= y`, applied in place.
///
/// The pairs are the op-assignment table's rather than the operator table's,
/// and they are not the same set: `f += 1` is there and `i += 1.5` is not —
/// an integer target with a float operand has no built-in op-assignment and
/// expands into `i = i + 1.5`, which is a different answer and Rhai's.
///
/// `target` is untouched unless the answer is `Ok(Some(()))`: an operator that
/// declines has written nothing, and one that fails computes its result before
/// there is anywhere to put it. A caller that answers a failure by handing the
/// same operands to the walk rests on this, and would apply the operator twice
/// without it.
#[inline]
fn apply_assign(kind: BinOpKind, target: &mut Dynamic, rhs: &Dynamic) -> RhaiResultOf<Option<()>> {
    match (&mut target.0, &rhs.0) {
        (Union::Int(x, ..), Union::Int(y, ..)) => {
            let (x, y) = (*x, *y);
            match arith::int_assign(kind, x, y)? {
                Some(value) => {
                    #[cfg(feature = "grain-jit")]
                    jit::dynamic_store_int(target, value);
                    #[cfg(not(feature = "grain-jit"))]
                    if let Union::Int(held, ..) = &mut target.0 {
                        *held = value;
                    }
                    Ok(Some(()))
                }
                None => Ok(None),
            }
        }
        #[cfg(not(feature = "no_float"))]
        (Union::Float(x, ..), Union::Float(y, ..)) => {
            let (x, y) = (**x, **y);
            Ok(arith::float_assign(kind, x, y).map(|value| {
                #[cfg(feature = "grain-jit")]
                jit::dynamic_store_float(target, value);
                #[cfg(not(feature = "grain-jit"))]
                if let Union::Float(held, ..) = &mut target.0 {
                    **held = value;
                }
            }))
        }
        #[cfg(not(feature = "no_float"))]
        #[allow(clippy::cast_precision_loss)]
        (Union::Float(x, ..), Union::Int(y, ..)) => {
            let (x, y) = (**x, *y as crate::FLOAT);
            Ok(arith::float_assign(kind, x, y).map(|value| {
                #[cfg(feature = "grain-jit")]
                jit::dynamic_store_float(target, value);
                #[cfg(not(feature = "grain-jit"))]
                if let Union::Float(held, ..) = &mut target.0 {
                    **held = value;
                }
            }))
        }
        _ => Ok(None),
    }
}

/// A scope entry, addressed the way whatever wants it was written.
///
/// A slot always names one and a name may name nothing, which is the whole of
/// the difference between the two at run time — for a [`Receiver`] and for a
/// [`Root`] alike.
#[derive(Clone, Copy)]
enum Site<'a> {
    Slot(usize),
    Name(&'a str),
}

/// What a chain turned out to be rooted at, and the value to walk.
///
/// Rhai draws this line in `search_namespace`, which hands back a `Target`: a
/// scope entry becomes a reference to write through, and a resolver's answer or
/// a module's constant becomes a read-only temporary (`eval/expr.rs:120-155`).
/// Which one a [`Root::Named`] is cannot be known until it is looked up.
///
/// Whichever it is, the value keeps the access mode it was found with, and that
/// is not decoration: it is the only thing standing between a `const` and a
/// method that mutates it — `exec_native_fn_call` refuses a non-pure function
/// whose first argument is read-only (`func/call.rs:405`).
enum RootValue<'s> {
    /// A scope entry, walked where it lives.
    ///
    /// Nothing is written back afterwards because nothing was copied: a
    /// mutation partway down the chain landed in the entry itself, which is
    /// what Rhai's `Target::RefMut` does (`eval/chaining.rs:517-563`).
    Entry(&'s mut Dynamic),

    /// A value with a name but no entry behind it — a resolver's answer, or a
    /// module's constant. Nowhere to write back to.
    Detached(Dynamic),

    /// The frame's receiver, moved out of the register for the walk.
    ///
    /// A register cannot lend a `&mut` across the `&mut self` the walk needs,
    /// so this is the one root that still travels: moved out here and moved
    /// back by [`Vm::run_chain`], the same trade [`bind_this`] makes.
    This(Dynamic),

    /// A value taken off the operand stack: `[1, 2].len()`, `f().x`.
    ///
    /// Nothing can be assigned to one, because Rhai's parser refuses it
    /// outright. Moved rather than borrowed for the same reason [`Self::This`]
    /// is — the stack is part of `self`.
    Temporary(Dynamic),
}

/// Outcome of the built-in assignment fast path.
///
/// Keep the handled bit separate from the nullable error pointer.  Rust's
/// `Option<Result<(), Box<_>>>` otherwise lowers a successful `?` through a
/// transient `ControlFlow<Result<..>>` shell; RPython's exception-edge graph
/// has no corresponding object.  This spelling carries the same three states
/// while leaving success on ordinary control flow and failure as one nullable
/// exception pointer.
struct StoreBuiltinOutcome {
    handled: bool,
    error: Option<Box<EvalAltResult>>,
}

impl RootValue<'_> {
    /// The value to walk.
    fn as_mut(&mut self) -> &mut Dynamic {
        match self {
            Self::Entry(value) => value,
            Self::Detached(value) | Self::This(value) | Self::Temporary(value) => value,
        }
    }
}

/// A chain's root, looked up.
struct ChainRoot<'s> {
    value: RootValue<'s>,
    /// Where to blame a refusal: the variable for a name, the chain otherwise.
    pos: Position,
}

/// What one indexing step managed.
#[cfg(not(all(feature = "no_index", feature = "no_object")))]
enum Indexed {
    /// Taken through a reference: the value, and whether anything wrote.
    Done(Dynamic, bool),
    /// There was no reference to take. Carries the value back out, because the
    /// caller cannot touch the container until this borrow has ended.
    NoReference(Dynamic),
}

/// What a chain's root is called, for the two errors that name it.
///
/// `None` for a temporary, which has no name to give — and neither error can
/// reach one: nothing assigns through a temporary, and flattening it on the way
/// in means there is no cell left to contend for.
///
/// `this` *can* reach both and has no name either, so it answers with the empty
/// one rather than with nothing. That is Rhai's own answer: `Expr::ThisPtr`
/// carries no name, so assigning through a read-only receiver is
/// `ErrorAssignmentToConstant("")` (`eval/stmt.rs:118-122`). Answering `None`
/// here would report a malformed chunk instead.
fn root_name<'p>(program: &'p Program, chain: &Chain) -> Option<&'p str> {
    match chain.root {
        Root::Local { name, .. } | Root::Named { name, .. } => program.name(name),
        Root::This { .. } => Some(""),
        Root::Temporary => None,
    }
}

/// Where chain `index`'s slots start in the sidecar's flat stream.
///
/// Walked rather than cached: only a chain that already failed asks.
fn chain_slot_base(program: &Program, index: u32) -> u32 {
    program
        .chains()
        .iter()
        .take(index as usize)
        .map(Chain::position_slots)
        .sum()
}

/// The op-assignment a chain ends with, resolved out of the pool.
fn chain_op<'p>(
    program: &'p Program,
    chain: &Chain,
) -> Result<Option<&'p AssignOp>, Box<EvalAltResult>> {
    let Tail::Assign { op: Some(op) } = &chain.tail else {
        return Ok(None);
    };
    Ok(Some(or_raise!(
        program.assign_op(*op),
        malformed(format!("no op-assignment {op}"))
    )))
}

/// One `for` loop in progress.
///
/// The count is here rather than in a local because Rhai keeps it outside the
/// scope too, and checks it for overflow before writing it — a loop long
/// enough to wrap the counter is an error rather than a wrap
/// (`eval/stmt.rs:729`).
struct Iteration {
    items: Items,
    /// The index of the item last handed out, starting one below the first.
    count: INT,
}

/// What a running `for` loop pulls its items from.
///
/// Almost always the iterator the registry handed over, which is a
/// `Box<dyn Iterator>`: one heap allocation and a registry search to build, and
/// one indirect call per turn. `for i in 0..n` is common enough for the
/// exclusive integer range to walk itself instead — but only when nothing has
/// registered its own meaning for the type. See [`Vm::iter_init`].
enum Items {
    /// An exclusive integer range, walked in place.
    IntRange {
        /// The next value to hand out; at or past `end` when exhausted.
        next: INT,
        end: INT,
    },
    /// Whatever the registry built.
    Boxed(Box<dyn Iterator<Item = VmResult>>),
}

impl Items {
    #[inline]
    fn next(&mut self) -> Option<VmResult> {
        match self {
            // `next < end` before the increment, so the increment cannot
            // overflow however close `end` is to `INT::MAX`.
            Self::IntRange { next, end } => {
                if *next >= *end {
                    return None;
                }
                let value = *next;
                *next += 1;
                Some(Ok(Dynamic::from_int(value)))
            }
            Self::Boxed(items) => items.next(),
        }
    }
}

/// A scope entry as a place to write, seeing through a shared cell.
///
/// Rhai reaches a variable through a `Target`, whose shared arm hands over the
/// cell's guard rather than the cell (`eval/target.rs:409-422`), so an
/// assignment lands where every closure holding that cell can see it. Writing
/// the slot itself would replace the cell and quietly sever them — the value
/// would be right and the aliasing dead.
///
/// For an ordinary value `write_lock` is a downcast to itself, so the common
/// case pays nothing. Rhai's own for-loop does this with `.unwrap()` and
/// panics on a contended cell; a VM that promises errors instead of panics
/// reports `ErrorDataRace`, as `Target` does.
fn place<'a>(
    entry: &'a mut Dynamic,
    name: &str,
    pos: Position,
) -> Result<DynamicWriteLock<'a, Dynamic>, Box<EvalAltResult>> {
    Ok(or_raise!(
        entry.write_lock::<Dynamic>(),
        Box::new(EvalAltResult::ErrorDataRace(name.to_string(), pos))
    ))
}

/// Write a `for` loop variable into its slot, through a shared cell if one is
/// there.
///
/// A closure made in an earlier iteration shares this slot, and Rhai writes
/// into it rather than replacing it (`eval/stmt.rs:752`) so that every closure
/// holding the cell sees the last value.
///
/// The guard is only needed for a cell a closure captured, so the check for one
/// is a discriminant test rather than the downcast chain `write_lock` walks —
/// the same shape [`Op::AssignLocal`](crate::grain::bytecode::Op::AssignLocal)
/// uses, and for the same reason: this runs on every turn of every `for` loop.
/// It also means the position is resolved only on the branch that can report
/// one.
#[inline]
fn store_shared(
    entry: &mut Dynamic,
    value: Dynamic,
    pos: impl Fn() -> Position,
) -> Result<(), Box<EvalAltResult>> {
    if is_shared!(entry) {
        *place(entry, "", pos())? = value;
    } else {
        overwrite(entry, value);
    }
    Ok(())
}

/// Turn the two control-flow errors back into the value they carry.
///
/// Rhai unwinds `return` and `exit` as errors rather than returning them;
/// `eval_global_statements` is where they turn back into values, and anything
/// entering a program from outside has to do the same.
fn unwind_exit(result: VmResult) -> VmResult {
    result.or_else(|err| match *err {
        EvalAltResult::Return(out, ..) | EvalAltResult::Exit(out, ..) => Ok(out),
        _ => Err(err),
    })
}

fn missing(name: &str, pos: Position) -> Box<EvalAltResult> {
    Box::new(EvalAltResult::ErrorVariableNotFound(name.to_string(), pos))
}

fn malformed(detail: String) -> Box<EvalAltResult> {
    Box::new(EvalAltResult::ErrorRuntime(
        format!("malformed chunk: {detail}").into(),
        Position::NONE,
    ))
}

/// One frame of a failed run, as an address rather than a position.
///
/// What a stripped program reports instead. Chunks share one instruction
/// buffer, so an address names an instruction whichever chunk it is in.
///
/// Travels from the device that failed to the host holding the
/// [`Sidecar`](crate::grain::Sidecar), by whatever the link uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fault {
    /// Byte offset of the instruction this frame stopped at.
    pub address: usize,
    /// Which chain slot raised, for an
    /// [`Op::Chain`](crate::grain::bytecode::Op::Chain).
    ///
    /// One instruction walks every step of `a.b[i].c` and gets one address
    /// between them, so the slot is what separates them.
    pub slot: Option<u32>,
}

/// One operand-stack cell in the interpreter build selected for this crate.
///
/// RPython's value stack is an array of object references.  Rust's `Dynamic`
/// is instead a 16-byte inline enum, so exposing `Vec<Dynamic>` to majit makes
/// `getarrayitem_gc_r` load only the first word of each value.  The JIT build
/// gives every preallocated stack cell a stable box and mutates its contents;
/// the vector then contains genuine pointer-width elements without allocating
/// on push/pop.  The non-JIT Grain interpreter keeps its original inline
/// representation and cost model.
#[cfg(feature = "grain-jit")]
type OperandSlot = Box<Dynamic>;
#[cfg(not(feature = "grain-jit"))]
type OperandSlot = Dynamic;

#[inline(always)]
fn operand_slot(value: Dynamic) -> OperandSlot {
    #[cfg(feature = "grain-jit")]
    {
        Box::new(value)
    }
    #[cfg(not(feature = "grain-jit"))]
    {
        value
    }
}

#[inline(always)]
fn operand_ref(slot: &OperandSlot) -> &Dynamic {
    #[cfg(feature = "grain-jit")]
    {
        slot.as_ref()
    }
    #[cfg(not(feature = "grain-jit"))]
    {
        slot
    }
}

#[inline(always)]
fn operand_mut(slot: &mut OperandSlot) -> &mut Dynamic {
    #[cfg(feature = "grain-jit")]
    {
        slot.as_mut()
    }
    #[cfg(not(feature = "grain-jit"))]
    {
        slot
    }
}

#[inline(always)]
fn clone_operand(slot: &OperandSlot) -> Dynamic {
    clone_value(operand_ref(slot))
}

#[inline(always)]
fn set_operand(slot: &mut OperandSlot, value: Dynamic) {
    overwrite(operand_mut(slot), value);
}

#[inline(always)]
fn stack_ref<'a>(vm: &'a Vm<'_>, index: usize) -> &'a Dynamic {
    #[cfg(feature = "grain-jit")]
    {
        jit::operand_stack_entry(vm, index)
    }
    #[cfg(not(feature = "grain-jit"))]
    {
        operand_ref(&vm.stack[index])
    }
}

#[inline(always)]
fn stack_mut<'a>(vm: &'a mut Vm<'_>, index: usize) -> &'a mut Dynamic {
    #[cfg(feature = "grain-jit")]
    {
        jit::operand_stack_entry_mut(vm, index)
    }
    #[cfg(not(feature = "grain-jit"))]
    {
        operand_mut(&mut vm.stack[index])
    }
}

#[inline(always)]
fn stack_take(vm: &mut Vm<'_>, index: usize) -> Dynamic {
    #[cfg(feature = "grain-jit")]
    {
        // Spell the aggregate here instead of loading `Dynamic::UNIT` through
        // its associated-const shim. The latter has no executable address in
        // Charon's symbolic call table; the enum construction is ordinary
        // translated allocation plus field stores.
        let mut value = Dynamic(Union::Unit((), 0, AccessMode::ReadWrite));
        jit::operand_stack_take(vm, index, &mut value);
        value
    }
    #[cfg(not(feature = "grain-jit"))]
    {
        mem::take(stack_mut(vm, index))
    }
}

/// The live state belonging to one interpreted call frame.
///
/// PyPy threads one `PyFrame` red through every invocation of its dispatch
/// loop.  Keeping these values as unrelated portal reds aliases an inlined
/// callee back onto its caller's `Vm`/`Scope`; a frame object gives every
/// recursive [`Vm::execute`] its own identity and its own locals base, fault
/// position and operand floor.
#[repr(C)]
pub(super) struct GrainFrame<'a, 'scope> {
    scope: &'a mut Scope<'scope>,
    base: usize,
    reached: usize,
    stack_base: usize,
    /// Source-PC handoff from the synchronous meta-interpreter walk.
    ///
    /// RPython's `raise_continue_running_normally` unwinds out of the tracing
    /// portal after the walk has already performed its concrete effects. Rust
    /// cannot raise across this marker call, so the live red frame carries the
    /// equivalent one-shot coordinate back to the dispatch loop. Zero means
    /// that this merge point only observed the native iteration; a real
    /// source PC is stored plus one so PC zero remains representable without
    /// materialising Rust's associated `usize::MAX` constant in jitcode.
    jit_resume_pc_plus_one: usize,
}

#[cfg(feature = "grain-jit")]
impl GrainFrame<'_, '_> {
    /// Apply the canonical compiled-loop image while the merge point still
    /// owns the safe mutable borrow of this stack-resident frame.
    fn restore_from_jit(&mut self, values: &[i64], vm: &Vm<'_>) {
        let [frame, restored_vm, scope, base, reached, stack_base, ..] = values else {
            return;
        };
        let frame_addr = self as *mut Self as usize;
        let scope_addr = self.scope as *mut Scope<'_> as usize;
        let vm_addr = vm as *const Vm<'_> as usize;
        if *frame as usize != frame_addr
            || *restored_vm as usize != vm_addr
            || *scope as usize != scope_addr
        {
            return;
        }
        self.base = *base as usize;
        self.reached = *reached as usize;
        self.stack_base = *stack_base as usize;
    }
}

/// What a resolution cache was filled against, and so what it may answer for.
///
/// `resolve_fn` keys its cache on a hash of the name and the argument types
/// alone (`func/call.rs:221-223`), and then searches the function libraries,
/// the engine's modules and the imported ones (`func/call.rs:238-262`) — so
/// what the cache means depends on a set the key does not name. Rhai carries
/// one `Caches` across a whole evaluation and reaches for the same three
/// answers under every library it stacks, clearing or layering the cache only
/// where an `import` brings global functions (`eval/stmt.rs:88-100`). A pool
/// spans no more than that evaluation, so it could rest on the same argument.
///
/// It does not. The set is recorded and compared instead, because the one
/// thing that must never happen is a cache filled against one engine being
/// consulted by another — which the engine address alone settles, and which
/// the libraries beside it settle for a run whose crossings do not all arrive
/// through the same environment.
///
/// Identities rather than counts: [`num_imports`] keys an entry that cannot
/// outlive a frame, and a frame is a span across which none of this can
/// change; a pool is not.
#[derive(Default, PartialEq, Eq)]
struct Searched {
    /// The engine whose registrations the search reached, as an address.
    ///
    /// A run holds the engine by shared reference and registering wants it by
    /// mutable one, so an engine cannot gain a function while a pool that
    /// searched it is alive — the pool belongs to the run
    /// ([`Vm::eval_with_callbacks`]) and dies with it.
    engine: usize,
    /// The function libraries stacked around the call, each as the address of
    /// the module it shares.
    #[cfg(not(feature = "no_function"))]
    lib: crate::StaticVec<usize>,
    /// The modules imported at the call, the same way.
    #[cfg(not(feature = "no_module"))]
    imports: crate::StaticVec<usize>,
}

impl Searched {
    /// What a crossing running against this state searches.
    ///
    /// Recorded once, as a crossing ends, rather than on both sides of the
    /// pool: the state a cache was last filled under is the state the crossing
    /// finished in, and [`Searched::matches`] answers the other half without
    /// building anything.
    fn of(engine: &Engine, global: &GlobalRuntimeState) -> Self {
        Self {
            engine: engine as *const Engine as usize,
            #[cfg(not(feature = "no_function"))]
            lib: global.lib.iter().map(module_address).collect(),
            #[cfg(not(feature = "no_module"))]
            imports: global
                .scan_imports_raw()
                .map(|(_, module)| module_address(module))
                .collect(),
        }
    }

    /// Whether a crossing about to run against this state searches the same.
    fn matches(&self, engine: &Engine, global: &GlobalRuntimeState) -> bool {
        if self.engine != engine as *const Engine as usize {
            return false;
        }
        #[cfg(not(feature = "no_function"))]
        if self.lib.len() != global.lib.len()
            || self
                .lib
                .iter()
                .zip(global.lib.iter())
                .any(|(&was, module)| was != module_address(module))
        {
            return false;
        }
        #[cfg(not(feature = "no_module"))]
        if self.imports.len() != global.num_imports()
            || self
                .imports
                .iter()
                .zip(global.scan_imports_raw())
                .any(|(&was, (_, module))| was != module_address(module))
        {
            return false;
        }
        true
    }
}

/// A shared module named by where it is, which is the only way to ask whether
/// two libraries hold the same one.
#[cfg(any(not(feature = "no_function"), not(feature = "no_module")))]
#[inline]
#[must_use]
fn module_address(module: &SharedModule) -> usize {
    crate::Shared::as_ptr(module) as usize
}

/// The parts of a [`Vm`] that one crossing may hand to the next.
///
/// A crossing builds a whole `Vm` per element a native calls back over
/// ([`callback::invoke`]), and the module doc for [`mod@callback`] names what
/// that costs: a second VM whose resolution cache starts empty. Everything
/// here is what a later crossing may inherit so that it does not.
///
/// Split out as a value rather than reached through a flag, so that
/// [`Vm::reentrant`] *is* the pooled constructor called with a cold
/// [`Warm::new`]. A reused `Vm` is then indistinguishable from a fresh one by
/// construction: a field is either named here, and carried, or seeded from the
/// [`NativeCallContext`] on every crossing exactly as it was.
///
/// What may be carried is what answers a question the context cannot change:
///
/// * `caches` is Rhai's function-resolution cache. It is the whole cost this
///   exists to remove, and the only part whose answer depends on anything
///   outside the `Vm` — hence [`Searched`] beside it.
/// * `strings_interner` maps text to an `ImmutableString` and nothing else.
/// * `stack`, `scopes`, `iterators`, `handlers`, `sizes` and `pending_steps`
///   are storage. Each is empty when a crossing ends — the operand stack
///   because every slot at or above the depth holds unit (see [`Vm::stack`])
///   and the crossing's frame truncated the depth to zero, the others because
///   [`Vm::execute`] floors them to what it found — so what carries over is an
///   allocation and never a value. [`Warm::reset`] states that rather than
///   assuming it.
/// * `operator_memo`, `call_memo` and `last_generation` **travel together or
///   not at all**. An entry is readable only by the generation that stamped
///   it, and [`Vm::execute`] mints one by incrementing `last_generation`.
///   Carrying the tables while restarting the counter would hand a frame a
///   number an entry of an earlier crossing already holds — a hit on a memo of
///   another program, which is a wrong answer rather than a slow one. Carried
///   together, a generation is still never handed out twice and every entry
///   from an earlier crossing is unreadable, which is exactly the guarantee
///   two frames of one `Vm` have. See [`Vm::generation`].
///
/// What is not named here is not carried. `global` is cloned from the context
/// on every crossing and is what holds the call level, the source, the
/// function library and the operation count; `depth`, `generation`, `this`,
/// `owns_trace`, `chain_step`, `pending_slot` and `unwind_floor` belong to the
/// frame and start where a fresh `Vm` starts them; `callbacks` is the share of
/// the program this crossing came out of, and is dropped rather than lent so
/// that a pool never keeps a program alive.
pub(super) struct Warm {
    caches: Caches,
    searched: Searched,
    strings_interner: StringsInterner,
    stack: Vec<OperandSlot>,
    scopes: crate::StaticVec<Scope<'static>>,
    iterators: Vec<Iteration>,
    handlers: Vec<Handler>,
    sizes: Vec<(usize, usize, usize)>,
    operator_memo: [OperatorMemo; OPERATOR_MEMO_SLOTS],
    call_memo: Option<Box<CallMemoTable>>,
    last_generation: u64,
    #[cfg(feature = "debugging")]
    pending_steps: Vec<(usize, u16, crate::eval::DebuggerStatus)>,
}

impl Warm {
    /// Nothing carried, which is what a crossing that inherits nothing gets.
    ///
    /// Every value here is the one [`Vm::new`] uses, so a `Vm` built on this is
    /// the `Vm` a crossing built before there was a pool.
    #[must_use]
    pub(super) fn new() -> Self {
        Self {
            caches: Caches::new(),
            searched: Searched::default(),
            strings_interner: StringsInterner::new(256),
            stack: Vec::new(),
            scopes: crate::StaticVec::new(),
            iterators: Vec::new(),
            handlers: Vec::new(),
            sizes: Vec::new(),
            operator_memo: [OperatorMemo::EMPTY; OPERATOR_MEMO_SLOTS],
            call_memo: None,
            last_generation: 0,
            #[cfg(feature = "debugging")]
            pending_steps: Vec::new(),
        }
    }

    /// Make this fit to be lent to a crossing arriving with this state.
    ///
    /// The resolution cache is the one part that can be wrong rather than
    /// merely cold, so it is thrown away exactly when what it was filled
    /// against is no longer what is being searched. Everything else is storage,
    /// or is keyed on a generation this carries, and neither depends on the
    /// libraries. See [`Searched`].
    fn fit(&mut self, engine: &Engine, global: &GlobalRuntimeState) {
        if !self.searched.matches(engine, global) {
            self.caches = Caches::new();
        }
    }

    /// Empty this out, the way [`Vm::give_scope`] empties a scope it takes
    /// back.
    ///
    /// Everything here is already empty — the field list above says why — and
    /// clearing regardless is what makes that a property of this type rather
    /// than of every path that reaches it. A `clear` on an empty container is
    /// a length store, and the capacity is what was worth keeping.
    ///
    /// The cache stack is rewound to its bottom layer on the same terms. That
    /// layer is the one a crossing of this run filled; anything above it was
    /// pushed by a nesting inside the crossing that has since ended, and Rhai
    /// rewinds those itself (`func/script.rs:94,206`).
    fn reset(&mut self) {
        if self.caches.fn_resolution_caches_len() > 1 {
            self.caches.rewind_fn_resolution_caches(1);
        }
        self.iterators.clear();
        self.handlers.clear();
        self.sizes.clear();
        #[cfg(feature = "debugging")]
        self.pending_steps.clear();
    }
}

/// Executes a [`Program`] against an `Engine`.
///
/// Holds one `GlobalRuntimeState` and one `Caches` for its whole lifetime, so
/// the function-resolution cache survives across calls. That matters: the
/// reentrant helpers Rhai exposes to native functions build a fresh
/// `Caches::new()` per call (`func/native.rs:519`), which would throw away
/// resolution work on every dispatch.
pub struct Vm<'e> {
    engine: &'e Engine,
    global: GlobalRuntimeState,
    caches: Caches,
    /// The operand stack: one allocation shared by every frame, with
    /// [`Vm::depth`] naming the top rather than the vector's own length.
    ///
    /// `pyframe.py:110-112` allocates `[None] * size` and calls
    /// `make_sure_not_resized` on it, so a push is a store and an increment
    /// and never a length change. Every slot from `depth` up holds unit; a
    /// value leaving the stack is cleared out of its slot rather than
    /// abandoned, because a `Dynamic` holds a reference and nothing else
    /// would drop it.
    stack: Vec<OperandSlot>,
    /// One past the top operand — `pyframe.py:88 valuestackdepth`.
    depth: usize,
    /// Emptied `Scope`s that finished calls gave back. See [`Vm::take_scope`].
    ///
    /// Inline storage rather than a `Vec`, because a crossing builds a whole
    /// `Vm` per element a native calls back over
    /// ([`callback::invoke`](crate::grain::vm::callback)) and a heap-allocated
    /// pool would be one allocation per crossing for the first call inside it
    /// — moving the cost rather than removing it. Inline, a pool that is never
    /// used costs nothing to build and nothing to walk.
    ///
    /// It grows to the deepest the call stack has been, which
    /// `max_call_levels` bounds wherever `unchecked` is off; the operand stack
    /// beside it is retained on the same terms.
    scopes: crate::StaticVec<Scope<'static>>,
    #[cfg_attr(any(feature = "no_index", feature = "no_object"), allow(unused))]
    strings_interner: StringsInterner,
    /// One entry per `for` loop currently running.
    ///
    /// Not on the operand stack, because an iterator is not a `Dynamic`. A
    /// frame truncates this to what it found on entry, so a `return` or an
    /// escaping error drops whatever its loops were holding without the
    /// compiler emitting anything.
    iterators: Vec<Iteration>,
    /// One entry per `try` region currently armed or catching. Frame-floored
    /// the same way the iterators are, so an error in a called function can
    /// never find its caller's handler and jump into another chunk.
    handlers: Vec<Handler>,
    /// The running data-size total of each literal currently being built,
    /// innermost last.
    ///
    /// One entry per array or map literal under construction, so
    /// `[a, [b, c], d]` keeps the inner total separate from the outer.
    /// Truncated per frame, as the iterators are, so an error part way through
    /// a literal leaves nothing behind.
    sizes: Vec<(usize, usize, usize)>,
    /// Where the scope goes back to if an error escapes the running frame.
    ///
    /// Set by [`Op::Checkpoint`](crate::bytecode::Op::Checkpoint) at each
    /// top-level statement of a chunk, saved and restored per frame. See
    /// [`Vm::execute`].
    unwind_floor: usize,
    /// The receiver bound to the frame currently running.
    ///
    /// Owned rather than borrowed: the value a binder has to hand over always
    /// lives in `stack` or in a caller's `Scope`, and neither can lend a `&mut`
    /// across the `&mut self` call that runs the callee. So the binder moves it
    /// in, the frame owns it, and [`Vm::call_compiled_with_this`] hands it back
    /// for the binder to put where it came from.
    ///
    /// Saved and restored per *call* rather than per frame, which is what makes
    /// `this` never inherited: a plain nested call passes `None` and gets the
    /// caller's back on the way out, reproducing `func/call.rs:669` without a
    /// conditional anywhere.
    this: Option<Dynamic>,
    /// Whether this `Vm` started the run, and so may clear its trace.
    ///
    /// False for a [`Vm::reentrant`]: a callback clearing the trace would
    /// discard frames the run around it already recorded.
    owns_trace: bool,
    /// Which step the innermost chain walk has reached.
    ///
    /// Saved and restored per chain, because a method step can run a body
    /// holding another chain. See [`Vm::run_chain`].
    chain_step: usize,
    /// A failing chain's slot, waiting for the frame it happened in.
    ///
    /// Set by [`Vm::run_chain`], taken by the next [`Vm::record_fault`].
    pending_slot: Option<u32>,
    /// What each operator site last resolved to. See [`OperatorMemo`].
    operator_memo: [OperatorMemo; OPERATOR_MEMO_SLOTS],
    /// What each call site last resolved to, once there is one. See
    /// [`CallMemo`] and [`CallMemoTable`].
    call_memo: Option<Box<CallMemoTable>>,
    /// The program a pointer this run creates carries with it, when there is
    /// one to carry.
    ///
    /// `Some` exactly where the wrappers are registered — see
    /// [`Vm::eval_with_callbacks`] and the crossings it leads to — because
    /// both answer the same question, which is whether a pointer this run
    /// hands out has anywhere to be called from. A `Vm` reached any other way
    /// holds the program by reference for the length of a call and has no
    /// share of it to give away. See [`callback::pointer`].
    callbacks: Option<SharedProgram>,
    /// A number identifying the frame currently running.
    ///
    /// Handed out on entry to [`Vm::execute`] and put back on the way out, so
    /// an entry either memo holds belongs to exactly one frame and no other.
    /// That is what makes a site safe to key on: two frames can be at the same
    /// address, or hold the same name index, in two different programs, and a
    /// nested call can evict a slot its caller wrote — but neither can be
    /// mistaken for a hit, because a generation is never handed out twice.
    ///
    /// Clearing the tables would do the same job and would cost every frame the
    /// memset. A counter costs an increment.
    generation: u64,
    /// The last generation handed out, so that the next one is one nothing
    /// carries. Never read back into an entry.
    last_generation: u64,
    /// Steps waiting for the statement that asked for them to end.
    ///
    /// Rhai keeps this in a `defer` per AST node (`eval/stmt.rs:271`): a `next`
    /// runs the statement it was asked at with the debugger quiet, and re-arms
    /// once that statement is over. A marker is a point rather than a scope, so
    /// the call level and statement depth it was asked at are recorded here
    /// instead, and the next marker at or outside them is where the statement
    /// has ended. Innermost last.
    #[cfg(feature = "debugging")]
    pending_steps: Vec<(usize, u16, crate::eval::DebuggerStatus)>,
}

/// Take a receiver for a callee to own, and say whether it goes back.
///
/// The rule is `chain_root`'s `walkable` (see [`Vm::chain_root`]) plus the
/// question of ownership. A read-only receiver is *cloned*, because `take`
/// would leave `UNIT` behind and strip the const-ness the caller's slot is
/// carrying; nothing is written back, since Rhai refuses the mutation rather
/// than making it. Anything else is taken, as `call_compiled_body` already
/// takes a call's arguments — a receiver is not shared with anything the callee
/// can reach, and taking it saves a deep clone of an array or a map per call.
///
/// A shared receiver is taken like any other: the taken value *is* the cell, so
/// `is_shared(this)` stays true inside the body and a write lands where every
/// other holder can see it. It still goes back, because taking it left `UNIT`
/// where the cell was.
fn bind_this(receiver: &mut Dynamic) -> (Dynamic, bool) {
    if receiver.is_read_only() {
        (clone_value(receiver).into_read_only(), false)
    } else {
        (mem::take(receiver), true)
    }
}

/// Put a receiver back where [`bind_this`] took it from.
///
/// Unconditional on how the call ended. Rhai reaches `this` through a pointer
/// into the caller's storage, so a body that mutates and then raises has
/// already written; a binder that only restored on success would discard
/// exactly that.
fn unbind_this(receiver: &mut Dynamic, taken: Option<Dynamic>, write_back: bool) {
    if let (true, Some(value)) = (write_back, taken) {
        *receiver = value;
    }
}

/// A `try` region.
struct Handler {
    target: usize,
    catch_var: Option<u32>,
    /// Where the three stacks were when the region was entered. An error can
    /// be raised at any depth of all three, and the catch block has to begin
    /// where the `try` did.
    operands: usize,
    scope_len: usize,
    iters: usize,
    /// Set once the catch block is running, holding the error it caught. That
    /// is what a bare `throw;` in the catch block re-raises, and its presence
    /// is what tells an escaping error it is leaving a catch rather than
    /// entering one.
    caught: Option<Box<EvalAltResult>>,
}

impl<'e> Vm<'e> {
    /// A VM that dispatches through `engine`.
    #[must_use]
    pub fn new(engine: &'e Engine) -> Self {
        let mut global = engine.new_global_runtime_state();

        // Eagerly, because a callback gets a *clone* of this state and a clone
        // of `None` is a separate `None`. Allocating on first fault would leave
        // every callback recording into a cell of its own.
        global.grain_faults = Some(crate::Shared::new(crate::Locked::new(Vec::new())));

        Self {
            engine,
            global,
            caches: Caches::new(),
            strings_interner: StringsInterner::new(256),
            stack: Vec::new(),
            depth: 0,
            scopes: crate::StaticVec::new(),
            iterators: Vec::new(),
            handlers: Vec::new(),
            sizes: Vec::new(),
            unwind_floor: 0,
            this: None,
            owns_trace: true,
            chain_step: 0,
            pending_slot: None,
            operator_memo: [OperatorMemo::EMPTY; OPERATOR_MEMO_SLOTS],
            call_memo: None,
            callbacks: None,
            generation: 0,
            last_generation: 0,
            #[cfg(feature = "debugging")]
            pending_steps: Vec::new(),
        }
    }

    /// A `Vm` for a call arriving from inside a native function.
    ///
    /// Reproduces what Rhai does at every reentrant boundary
    /// (`func/native.rs:516-519`, `types/fn_ptr.rs:451-454`): the caller's
    /// runtime state is *cloned* rather than shared, and the resolution cache
    /// starts empty. The clone is what carries the imported modules, the source
    /// name and — the part that matters here — the function library holding the
    /// wrappers, so a closure reached from a native can hand out a pointer of
    /// its own.
    ///
    /// The empty `Caches` is the cost, and it is the one thing a `Vm` normally
    /// exists to avoid. The outer `Vm` cannot lend its own: it is borrowed by
    /// the frame still running beneath this one. What one crossing can lend the
    /// next is the subject of `grain::vm::callback`, and this constructor is
    /// that one called with nothing lent.
    ///
    /// Operation counting has the same shape and the same reason as the clone:
    /// increments inside the callback land on it and are lost when it drops, as
    /// they are for any reentrant call Rhai makes.
    #[must_use]
    pub fn reentrant(context: &'e NativeCallContext<'_>) -> Self {
        Self::reentrant_from(context, Warm::new())
    }

    /// The same, with the parts an earlier crossing finished with.
    ///
    /// Every field the context decides is seeded here on every crossing, and
    /// every field [`Warm`] names is taken from it — so the two are exhaustive
    /// between them and a reused `Vm` differs from a fresh one only in what
    /// [`Warm`] argues may differ.
    #[must_use]
    pub(super) fn reentrant_from(context: &'e NativeCallContext<'_>, mut warm: Warm) -> Self {
        let engine = context.engine();
        let global = context.global_runtime_state().clone();
        warm.fit(engine, &global);

        Self {
            engine,
            global,
            caches: warm.caches,
            strings_interner: warm.strings_interner,
            stack: warm.stack,
            depth: 0,
            scopes: warm.scopes,
            iterators: warm.iterators,
            handlers: warm.handlers,
            sizes: warm.sizes,
            unwind_floor: 0,
            // A crossing carries no receiver: Rhai binds one only where it
            // dispatches a method, and this arrives through `call_fn_raw`.
            this: None,
            // `global` is a clone, so the trace is shared with the run that
            // called in and is not this call's to clear.
            owns_trace: false,
            chain_step: 0,
            pending_slot: None,
            // A memo names a site in a frame, and the counter that names frames
            // comes with the tables. See [`Warm`].
            operator_memo: warm.operator_memo,
            call_memo: warm.call_memo,
            // Set by [`callback::invoke`], which has the share of the program
            // this crossing came out of. A crossing reached any other way has
            // none, and hands out the pointers it can rather than none at all.
            callbacks: None,
            generation: 0,
            last_generation: warm.last_generation,
            // A step belongs to the statement that asked for it, and that
            // statement is running in the `Vm` this crossing came from.
            #[cfg(feature = "debugging")]
            pending_steps: warm.pending_steps,
        }
    }

    /// The parts of a finished crossing, for the next one to start from.
    ///
    /// Consumes the `Vm`, so every borrow of what is handed over has ended —
    /// the same thing that makes [`Vm::give_scope`] safe. What is not taken is
    /// dropped here, which is where the share of the program the crossing
    /// carried goes.
    #[must_use]
    pub(super) fn cool(mut self) -> Warm {
        // The frame the crossing ran left the depth where it found it, so this
        // is already zero; doing it anyway is what lets [`Warm`] say the stack
        // it lends holds nothing, rather than assuming the caller's path.
        self.truncate_stack(0);

        // Before anything moves: what the cache in hand was last filled
        // against is the state the crossing finished in.
        let searched = Searched::of(self.engine, &self.global);

        let mut warm = Warm {
            caches: self.caches,
            searched,
            strings_interner: self.strings_interner,
            stack: self.stack,
            scopes: self.scopes,
            iterators: self.iterators,
            handlers: self.handlers,
            sizes: self.sizes,
            operator_memo: self.operator_memo,
            call_memo: self.call_memo,
            last_generation: self.last_generation,
            #[cfg(feature = "debugging")]
            pending_steps: self.pending_steps,
        };
        warm.reset();
        warm
    }

    /// Where the last run failed, innermost frame first.
    ///
    /// What a stripped program reports instead of a position;
    /// [`restore`](crate::grain::restore::restore) turns it back into one.
    ///
    /// Cleared at the start of a run and whenever a `catch` handles an error,
    /// so it describes the failure actually being reported.
    #[must_use]
    pub fn fault_trace(&self) -> Vec<Fault> {
        self.global
            .grain_faults
            .as_ref()
            .and_then(|faults| crate::func::native::locked_read(faults))
            .map(|faults| faults.clone())
            .unwrap_or_default()
    }

    /// Which instruction the last run failed at, if it failed.
    ///
    /// The innermost frame of [`Vm::fault_trace`].
    #[must_use]
    pub fn fault_pc(&self) -> Option<usize> {
        self.fault_trace().first().map(|fault| fault.address)
    }

    /// Note that a frame stopped at `address`, as an error leaves it.
    fn record_fault(&mut self, address: usize) {
        let fault = Fault {
            address,
            slot: self.pending_slot.take(),
        };

        // Normally already there, from `Vm::new` or from a reentrant call's
        // clone. The fallback covers a `Vm` reached some other way.
        let faults = self
            .global
            .grain_faults
            .get_or_insert_with(|| crate::Shared::new(crate::Locked::new(Vec::new())));

        if let Some(mut faults) = crate::func::native::locked_write(faults) {
            faults.push(fault);
        }
    }

    /// Forget where a run failed, because it did not or has not yet.
    ///
    /// A no-op on a reentrant `Vm`: the trace belongs to the run that called it.
    fn clear_faults(&mut self) {
        if !self.owns_trace {
            return;
        }

        self.pending_slot = None;
        if let Some(faults) = self.global.grain_faults.as_ref() {
            if let Some(mut faults) = crate::func::native::locked_write(faults) {
                faults.clear();
            }
        }
    }

    /// Run a program, returning its value.
    ///
    /// Mirrors `Engine::eval_ast_with_scope_raw`: the program's function
    /// library, module resolver and source name are installed for the duration
    /// and restored afterwards, so a `Vm` reused across programs does not leak
    /// one program's definitions into the next.
    /// Call one compiled function by name, with arguments already evaluated.
    ///
    /// The entry point a native needs. `Op::Call` reaches a chunk through the
    /// name *index* it shares with the call site, which a caller from outside
    /// does not have — a `FnPtr` carries a string, and so does Rhai when it
    /// dispatches. This is the same call by the other key.
    ///
    /// `level` is the caller's call depth, so `max_call_levels` still counts
    /// across a boundary that leaves this VM and comes back. Left unthreaded,
    /// a closure calling itself through `map` would recurse until the stack
    /// went rather than until the limit did.
    ///
    /// # Errors
    ///
    /// `ErrorFunctionNotFound` if no compiled function has that name and
    /// arity, and whatever the function itself raises otherwise.
    pub fn call_function(
        &mut self,
        program: &Program,
        name: &str,
        args: FnArgsVec<Dynamic>,
        level: usize,
        pos: Position,
    ) -> VmResult {
        let scope = &mut Scope::new();
        self.call_function_with_this(program, name, None, args, level, scope, true, pos, None)
            .0
    }

    /// The same, reaching the callee by its name-pool index.
    ///
    /// A crossing already knows which function it is carrying, and a name is
    /// how a caller that does not asks. The name stays for what it is still
    /// needed for, which is naming the callee if the arity does not match.
    pub(super) fn call_function_at(
        &mut self,
        program: &Program,
        index: u32,
        name: &str,
        args: FnArgsVec<Dynamic>,
        level: usize,
        pos: Position,
    ) -> VmResult {
        // Lent rather than built, which the entry point above cannot do: the
        // scope a call binds into is this `Vm`'s to hand out, and the two
        // arrays a fresh one reserves on its first push are two allocations
        // per crossing. See [`Vm::take_scope`].
        let mut detached = self.take_scope();
        let result = self
            .call_function_with_this(
                program,
                name,
                Some(index),
                args,
                level,
                &mut detached,
                true,
                pos,
                None,
            )
            .0;
        self.give_scope(detached);
        result
    }

    /// The same, against a receiver the callee owns for the duration.
    ///
    /// The receiver comes back however the call ended, for the caller to put
    /// where it took it from — see [`bind_this`] and [`unbind_this`].
    fn call_function_with_this(
        &mut self,
        program: &Program,
        name: &str,
        index: Option<u32>,
        args: FnArgsVec<Dynamic>,
        level: usize,
        scope: &mut Scope,
        rewind_scope: bool,
        pos: Position,
        this: Option<Dynamic>,
    ) -> (VmResult, Option<Dynamic>) {
        // An entry point in its own right so the trace starts here rather than at `run_main`.
        self.clear_faults();
        // A step a previous run left waiting is not this one's to honour.
        #[cfg(feature = "debugging")]
        self.pending_steps.clear();

        // By index where the caller has one: the name pool gives equal names
        // equal indices, so this is the same question asked with an integer
        // comparison instead of a string one.
        let found = match index {
            Some(index) => program.function(index, args.len()),
            None => program.function_named(name, args.len()),
        };
        let Some(function) = found else {
            return (
                Err(Box::new(EvalAltResult::ErrorFunctionNotFound(
                    format!("{name} ({} args)", args.len()),
                    pos,
                ))),
                this,
            );
        };
        let (params, chunk) = (&*function.param_names, function.chunk);

        // `call_compiled` takes its arguments off the operand stack, where a
        // compiled call site would already have put them.
        //
        // Reserved for the arguments and the body at once. A crossing arrives
        // on an empty stack, and taking both in one step is one growth rather
        // than the two that pushing and then entering the frame would each ask
        // for.
        let first = self.depth;
        self.reserve_stack(args.len() + program.max_stack() as usize);
        self.push_all(args.into_iter());

        let restore = mem::replace(&mut self.global.level, level);
        let (result, this) = self.call_compiled_with_this(
            program,
            name,
            params,
            chunk,
            first,
            scope,
            rewind_scope,
            pos,
            this,
        );
        self.global.level = restore;

        self.truncate_stack(first);
        (result, this)
    }

    /// Call one compiled function by name, instead of running the whole
    /// program.
    ///
    /// Mirrors [`Engine::call_fn`](crate::Engine::call_fn), including that the
    /// program's body runs first — a function usually needs what the top level
    /// declared. [`call_fn_with_options`](Self::call_fn_with_options) turns
    /// that off.
    ///
    /// Not available under `no_function`.
    ///
    /// # Errors
    ///
    /// `ErrorFunctionNotFound` if no compiled function has that name and
    /// arity, `ErrorMismatchOutputType` if the result is not a `T`, and
    /// whatever the function itself raises.
    #[cfg(not(feature = "no_function"))]
    pub fn call_fn<T: Variant + Clone>(
        &mut self,
        scope: &mut Scope,
        program: &Program,
        name: impl AsRef<str>,
        args: impl crate::FuncArgs,
    ) -> Result<T, Box<EvalAltResult>> {
        self.call_fn_with_options(CallFnOptions::new(), scope, program, name, args)
    }

    /// The same, with Rhai's [`CallFnOptions`](crate::CallFnOptions).
    ///
    /// Three of the five options mean something here:
    ///
    /// * `eval_ast` runs the program's main chunk before the call, so what the
    ///   top level declares is in scope for it. On by default, as in Rhai.
    /// * `rewind_scope` truncates the scope back afterwards. On by default.
    /// * `tag` sets the evaluation's custom state.
    ///
    /// `this_ptr` binds the callee's receiver, as it does in Rhai. A write
    /// through `this` lands in the pointer's own `Dynamic` — including when the
    /// call goes on to fail, because Rhai reaches `this` through the caller's
    /// storage and a body that mutates and then raises has already written.
    ///
    /// `in_all_namespaces` is ignored: this looks only in the program's own
    /// compiled functions.
    ///
    /// Not available under `no_function`.
    ///
    /// # Errors
    ///
    /// As [`call_fn`](Self::call_fn).
    #[cfg(not(feature = "no_function"))]
    pub fn call_fn_with_options<T: Variant + Clone>(
        &mut self,
        options: CallFnOptions,
        scope: &mut Scope,
        program: &Program,
        name: impl AsRef<str>,
        args: impl crate::FuncArgs,
    ) -> Result<T, Box<EvalAltResult>> {
        let name = name.as_ref();

        let mut arg_values = FnArgsVec::new();
        args.parse(&mut arg_values);

        let orig_scope_len = scope.len();
        let mut this_ptr = options.this_ptr;
        if let Some(tag) = options.tag {
            self.global.tag = tag;
        }

        // The pointer is the host's and outlives the call, so unlike every
        // other binder this one can hold it across the whole thing.
        let bound = this_ptr.as_deref_mut().map(bind_this);
        let (this, write_back) = bound.map_or((None, false), |(v, w)| (Some(v), w));

        // The program's environment stays installed for the *call*, not only for
        // the main chunk that may precede it: the function being called can
        // reach whatever the compiler left Rhai to interpret, and Rhai looks for
        // it in `global.lib`.
        let mut evaluated = Ok(());
        let (result, returned) = self.with_environment(program, None, |vm| {
            if options.eval_ast {
                // Run for the scope it leaves behind; the body's own value is
                // not what the caller asked for. The main chunk gets no
                // receiver, as `eval_global_statements` does not either.
                evaluated = unwind_exit(vm.run_main(program, scope)).map(|_| ());
            }

            let (result, returned) = if evaluated.is_ok() {
                vm.call_function_with_this(
                    program,
                    name,
                    None,
                    arg_values,
                    0,
                    scope,
                    options.rewind_scope,
                    Position::NONE,
                    this,
                )
            } else {
                (Ok(Dynamic::UNIT), this)
            };

            // However it went, and whether the call happened at all: Rhai
            // reports the end of a `call_fn` unconditionally
            // (`api/call_fn.rs:302-308`).
            #[cfg(feature = "debugging")]
            let result = vm.at_end(scope).and(result);

            (result, returned)
        });

        // Before the errors below, both of them: Rhai's mutation through `this`
        // survives a failed call, and survives one whose result is the wrong
        // type just as much.
        if let Some(slot) = this_ptr {
            unbind_this(slot, returned, write_back);
        }
        evaluated?;

        if options.rewind_scope {
            scope.rewind(orig_scope_len);
        }

        result?.try_cast_result().map_err(|value| {
            Box::new(EvalAltResult::ErrorMismatchOutputType(
                self.engine
                    .map_type_name(core::any::type_name::<T>())
                    .into(),
                self.engine.map_type_name(value.type_name()).into(),
                Position::NONE,
            ))
        })
    }

    /// Evaluate a program's main chunk against `scope`, yielding its value.
    ///
    /// The scope is the caller's, as it is for
    /// [`Engine::eval_ast_with_scope`](crate::Engine::eval_ast_with_scope):
    /// what the program declares at the top level is left in it, and what the
    /// caller put there beforehand is visible to the program.
    ///
    /// # Errors
    ///
    /// Whatever the program raises, and `ErrorRuntime` for a malformed one.
    pub fn eval_with_scope(&mut self, scope: &mut Scope, program: &Program) -> VmResult {
        self.run_with(program, scope, None)
    }

    /// The same against a scope of its own, for a program that needs none.
    ///
    /// # Errors
    ///
    /// As [`eval_with_scope`](Self::eval_with_scope).
    pub fn eval(&mut self, program: &Program) -> VmResult {
        self.eval_with_scope(&mut Scope::new(), program)
    }

    /// Evaluate a program against `scope` for its effects, discarding its value.
    ///
    /// # Errors
    ///
    /// As [`eval_with_scope`](Self::eval_with_scope).
    pub fn run_with_scope(
        &mut self,
        scope: &mut Scope,
        program: &Program,
    ) -> Result<(), Box<EvalAltResult>> {
        self.eval_with_scope(scope, program).map(|_| ())
    }

    /// The same against a scope of its own.
    ///
    /// # Errors
    ///
    /// As [`eval_with_scope`](Self::eval_with_scope).
    pub fn run(&mut self, program: &Program) -> Result<(), Box<EvalAltResult>> {
        self.run_with_scope(&mut Scope::new(), program)
    }

    /// Evaluate a program that hands function pointers to native functions.
    ///
    /// The same run, plus one native wrapper per compiled function registered
    /// for its duration, so a pointer this program creates resolves when Rhai
    /// dispatches it — `let a = [1, 2]; a.map(|x| x * 2)` is `map` calling us
    /// back, and `map` looks the pointer up its own way. See the `callback`
    /// module.
    ///
    /// Only worth the owned program when [`Program::makes_fn_pointers`] says a
    /// pointer can escape; [`eval_with_scope`](Self::eval_with_scope) is
    /// otherwise identical and copies nothing. A program that needs this and
    /// does not get it still runs — the pointer simply fails to resolve, as
    /// `ErrorFunctionNotFound`, at the point the native tries to call it.
    ///
    /// Read the `callback` module before relying on it: a crossing is slower than the
    /// walker, and a *capturing* closure handed to a native that binds `this`
    /// arrives with its arguments rotated.
    ///
    /// Named `eval_` rather than `run_` because it yields the program's value;
    /// Rhai has no `Engine` method to mirror here, so the crate's own rule is
    /// the one that applies.
    ///
    /// # Errors
    ///
    /// As [`eval_with_scope`](Self::eval_with_scope).
    pub fn eval_with_callbacks(&mut self, scope: &mut Scope, program: &SharedProgram) -> VmResult {
        let wrappers =
            (!program.functions().is_empty()).then(|| callback::wrappers(program).into());
        // Put back rather than cleared: a `Vm` reentered from a callback is
        // running the program its own field already names.
        let outer = mem::replace(&mut self.callbacks, Some(program.clone()));
        // The same for the pool the crossings of this run share, and on the
        // same condition as the wrappers: a program with no compiled functions
        // has no pointer to hand out and so nothing can cross back into it.
        // Installed here and nowhere else, so that a pool holds the engine this
        // run holds for exactly as long as this run does — see
        // `callback::Crossings`.
        let pool = wrappers.is_some().then(callback::pool);
        let outer_pool = mem::replace(&mut self.global.grain_crossings, pool);
        let result = self.run_with(program, scope, wrappers);
        self.global.grain_crossings = outer_pool;
        self.callbacks = outer;
        result
    }

    fn run_with(
        &mut self,
        program: &Program,
        scope: &mut Scope,
        wrappers: Option<SharedModule>,
    ) -> VmResult {
        self.with_environment(program, wrappers, |vm| {
            // `exit` and a top-level `return` become the run's value before the
            // debugger hears that it is over, because Rhai maps them inside
            // `eval_global_statements` (`eval/stmt.rs:1046-1048`) and reports
            // the end after it.
            let value = unwind_exit(vm.run_main(program, scope))?;
            #[cfg(feature = "debugging")]
            vm.at_end(scope)?;
            Ok(value)
        })
    }

    /// What a program contributes to `global`, in place for the duration of `f`.
    ///
    /// Everything entering a program from outside needs this, not only its main
    /// chunk: a function reached through [`call_fn`](Self::call_fn) can call
    /// whatever the compiler left Rhai to interpret, and Rhai looks for it in
    /// `global.lib`.
    fn with_environment<T>(
        &mut self,
        program: &Program,
        wrappers: Option<SharedModule>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let orig_source = mem::replace(&mut self.global.source, program.source().cloned());
        #[cfg(not(feature = "no_function"))]
        let orig_lib_len = self.global.lib.len();
        #[cfg(not(feature = "no_function"))]
        {
            if let Some(lib) = program.lib() {
                self.global.lib.push(lib.clone());
            }
            // Last, so the search — which runs in reverse — reaches a compiled
            // function before whatever the compiler left Rhai to interpret.
            if let Some(wrappers) = wrappers {
                self.global.lib.push(wrappers);
            }
        }
        // Without script functions there is no library to stack them in, and
        // `wrappers` is `None` for want of anything to wrap.
        #[cfg(feature = "no_function")]
        let _ = wrappers;
        #[cfg(not(feature = "no_module"))]
        let orig_resolver = mem::replace(
            &mut self.global.embedded_module_resolver,
            program.resolver().cloned(),
        );

        let result = f(self);

        #[cfg(not(feature = "no_module"))]
        {
            self.global.embedded_module_resolver = orig_resolver;
        }
        #[cfg(not(feature = "no_function"))]
        self.global.lib.truncate(orig_lib_len);
        self.global.source = orig_source;

        result
    }

    /// The main chunk, against an environment the caller has already installed.
    fn run_main(&mut self, program: &Program, scope: &mut Scope) -> VmResult {
        self.clear_faults();
        // A step a previous run left waiting is not this one's to honour.
        #[cfg(feature = "debugging")]
        self.pending_steps.clear();
        let mut pc = program.main().entry() as usize;
        // Slots are indices into the caller's scope, so a caller that arrives
        // with variables already in it shifts every one of them.
        let base = scope.len();
        let result = self.execute(program, scope, *program.main(), base, &mut pc);
        if result.is_err() {
            // Whatever failed deeper already recorded itself,
            // and this is the outermost frame of the same failure.
            self.record_fault(pc);
        }
        result
    }

    /// Slots for `extra` more operands above the top.
    ///
    /// Called once per frame entry with the deepest chunk's need, so the
    /// store below it never has to test — and marked cold because a
    /// well-formed program reaches it once and a growing one is already
    /// paying for an allocation.
    #[cold]
    #[inline(never)]
    fn grow_stack(&mut self, extra: usize) {
        let want = (self.depth + extra.max(1)).next_power_of_two();
        self.stack.resize_with(want, || operand_slot(Dynamic::UNIT));
    }

    #[inline]
    fn reserve_stack(&mut self, extra: usize) {
        #[cfg(feature = "grain-jit")]
        let stack_len = jit::operand_stack_len(self) as usize;
        #[cfg(not(feature = "grain-jit"))]
        let stack_len = self.stack.len();
        if self.depth + extra > stack_len {
            self.grow_stack(extra);
        }
    }

    /// `pyframe.py:390 pushvalue` — a store into the slot the depth names,
    /// and the depth moved on.
    #[inline]
    #[allow(unused_mut)]
    fn push(&mut self, mut value: Dynamic) {
        let depth = self.depth;
        #[cfg(feature = "grain-jit")]
        let stack_len = jit::operand_stack_len(self) as usize;
        #[cfg(not(feature = "grain-jit"))]
        let stack_len = self.stack.len();
        if depth == stack_len {
            self.grow_stack(1);
        }
        #[cfg(feature = "grain-jit")]
        {
            jit::operand_stack_store(self, depth, &mut value);
        }
        #[cfg(not(feature = "grain-jit"))]
        {
            // Every slot at or above the depth holds unit — `Vm::stack` says
            // so, and `Vm::truncate_stack` and `stack_take` are what keep it
            // true — so what this replaces owns nothing, and the read-back
            // that would decide whether to drop it decides nothing. Asserted
            // rather than assumed: a slot that held something would leak it.
            let slot = stack_mut(self, depth);
            debug_assert!(
                slot.is_unit(),
                "a slot at or above the operand stack's top holds unit"
            );
            mem::forget(mem::replace(slot, value));
        }
        self.depth = depth + 1;
    }

    fn push_all(&mut self, values: impl ExactSizeIterator<Item = Dynamic>) {
        self.reserve_stack(values.len());
        for value in values {
            self.push(value);
        }
    }

    /// The operands, without the slots above the top — those hold unit and
    /// belong to nobody.
    #[inline]
    fn values(&self) -> &[OperandSlot] {
        &self.stack[..self.depth]
    }

    #[inline]
    fn values_mut(&mut self) -> &mut [OperandSlot] {
        &mut self.stack[..self.depth]
    }

    /// `pyframe.py:427 dropvalues` — the abandoned slots are cleared rather
    /// than left holding the reference the value carried.
    fn truncate_stack(&mut self, depth: usize) {
        // `Vec::truncate` was a no-op for a length it already sat below, and a
        // caller that has popped past its own floor relies on that: taking the
        // depth as given here would hand it slots it had already given up.
        if depth >= self.depth {
            return;
        }
        let mut slot = depth;
        while slot < self.depth {
            overwrite(stack_mut(self, slot), Dynamic::UNIT);
            slot += 1;
        }
        self.depth = depth;
    }

    /// Move the operands from `first` upwards off the stack.
    ///
    /// What `Vec::drain` was: it took the values out and shortened the
    /// vector, and this takes them out and lowers the depth.
    fn take_values_from(&mut self, first: usize) -> FnArgsVec<Dynamic> {
        if first >= self.depth {
            return FnArgsVec::new();
        }
        let mut taken = FnArgsVec::new();
        let mut slot = first;
        while slot < self.depth {
            taken.push(stack_take(self, slot));
            slot += 1;
        }
        self.depth = first;
        taken
    }

    /// Open `count` slots at `at`, moving whatever is above them up.
    fn open_slots(&mut self, at: usize, count: usize) {
        self.reserve_stack(count);
        let mut slot = self.depth;
        while slot > at {
            slot -= 1;
            let value = stack_take(self, slot);
            *stack_mut(self, slot + count) = value;
        }
        self.depth += count;
    }

    /// The top operand, or unit where there is none.
    ///
    /// The two sites that read the stack without a depth to check — a
    /// `Return` with nothing pushed, and the fast assignment that has
    /// already established its two operands — share this rather than one
    /// carrying `unwrap_or` and the other an `expect`.
    fn pop_or_unit(&mut self) -> Dynamic {
        match self.depth.checked_sub(1) {
            Some(depth) => {
                self.depth = depth;
                stack_take(self, depth)
            }
            None => Dynamic::UNIT,
        }
    }

    fn pop(&mut self) -> Result<Dynamic, Box<EvalAltResult>> {
        let depth = or_raise!(
            self.depth.checked_sub(1),
            malformed("operand stack underflow".to_string())
        );
        self.depth = depth;
        Ok(stack_take(self, depth))
    }

    fn inspect(&mut self) -> Result<&Dynamic, Box<EvalAltResult>> {
        Ok(or_raise!(
            self.values().last().map(operand_ref),
            malformed("operand stack underflow".to_string())
        ))
    }

    /// `reached` tracks the instruction being executed, so a failure can be
    /// attributed to one. It is what a stripped program reports in place of a
    /// position.
    /// Call a chunk this compiler produced, reproducing `call_script_fn`
    /// (`func/script.rs:24`) step for step.
    ///
    /// The parts that are not obvious, and that the differential corpus is
    /// what proves: arguments are *taken* out of the caller's stack slots
    /// rather than cloned, the depth check happens after the level is
    /// incremented, and only errors that are neither a `return` nor a system
    /// exception get wrapped in `ErrorInFunctionCall`.
    /// Walk `a.b[i].c`, reading it or assigning to it.
    ///
    /// The reason this is one instruction and a recursion rather than a
    /// sequence: every level holds a `&mut` into the level above, exactly as
    /// Rhai does (`eval/chaining.rs:659`). For a map or an array that borrow is
    /// the whole story — the mutation lands in the container and no write-back
    /// is needed. Doing it on the operand stack instead would mutate a copy.
    ///
    /// That starts at the root: a scope entry is walked where it lives, not
    /// copied out and put back. Copying it would make a chain cost the size of
    /// what it is rooted at rather than the length of the chain, which for
    /// `a[i] = x` in a loop is the difference between linear and quadratic.
    ///
    /// The exceptions are the two roots a `&mut` cannot be had for, because
    /// they live in `self` and the walk needs `&mut self`: the `this` register
    /// and the operand stack. Both are *moved* rather than cloned, and `this`
    /// is moved back.
    ///
    /// Write-back is only for the levels where a borrow was not possible:
    /// a getter on a host type hands back a value, and Rhai calls the setter
    /// afterwards if the sub-chain was a method call. `changed` reproduces
    /// that, and it is deliberately coarse in the same way Rhai's is — Rhai's
    /// flag is `func.is_method()`, "does the resolved function take its
    /// receiver by reference", not "did it actually write".
    fn run_chain(
        &mut self,
        program: &Program,
        chain: &Chain,
        index: u32,
        scope: &mut Scope,
        base: usize,
        pos: Position,
    ) -> VmResult {
        // A method step can run a body holding a chain of its own, which writes
        // the same field. Saved so the step reported is this chain's.
        let outer_step = mem::replace(&mut self.chain_step, 0);
        let result = self.run_chain_steps(program, chain, scope, base, pos);
        let reached = mem::replace(&mut self.chain_step, outer_step);

        if result.is_err() {
            // Not if something deeper already claimed it: the slot belongs to
            // the innermost frame's address.
            if self.pending_slot.is_none() {
                self.pending_slot = chain
                    .step_slot(reached)
                    .map(|slot| chain_slot_base(program, index) + slot);
            }
        }

        result
    }

    /// [`Vm::run_chain`] without the bookkeeping around it.
    fn run_chain_steps(
        &mut self,
        program: &Program,
        chain: &Chain,
        scope: &mut Scope,
        base: usize,
        pos: Position,
    ) -> VmResult {
        // Step operands were pushed first, then the root if it is one that has
        // to be evaluated, then the value being assigned.
        let operands_at = or_raise!(
            self.depth.checked_sub(chain.consumes()),
            malformed("chain with too few operands".to_string())
        );

        let ChainRoot {
            value: mut root,
            pos: root_pos,
        } = self.chain_root(program, chain, scope, base, operands_at, pos)?;

        // Nothing between here and the restore below may use `?`: a `this` root
        // was moved out of the register, and an early return would drop it on
        // the floor. Everything that can fail travels in `result` instead.
        let read_only = root.as_mut().is_read_only();

        // Read-only is what refuses an assignment, not the absence of a place:
        // a module's constant is neither, so Rhai assigns into the copy and
        // discards it. Only a `const` and a resolver's answer are refused, and
        // both are read-only for that reason.
        //
        // A temporary is separate again — Rhai's parser refuses `f().x = 1`
        // outright, so one reaches here only from a chunk this compiler did
        // not build.
        let value = match (chain.assigns(), &root) {
            (false, _) => Ok(None),
            (true, RootValue::Temporary(..)) => {
                Err(malformed("chain assigns through a temporary root".into()))
            }
            (true, _) if read_only => Err(match root_name(program, chain) {
                Some(name) => Box::new(EvalAltResult::ErrorAssignmentToConstant(
                    name.to_string(),
                    root_pos,
                )),
                None => malformed("no chain root name".to_string()),
            }),
            // Rhai flattens the right-hand side before assigning, so a shared
            // cell is copied out rather than aliased in.
            (true, _) => Ok(Some(clone_operand(&self.stack[self.depth - 1]).flatten())),
        };

        let result = value.and_then(|value| {
            let mut operands: FnArgsVec<Dynamic> = self.values()
                [operands_at..operands_at + chain.operands as usize]
                .iter()
                .map(clone_operand)
                .collect();

            // A shared cell cannot be walked directly. `get_indexed_mut`
            // refuses one outright — `unreachable!("cannot handle shared
            // values")`, `eval/chaining.rs:461` — because Rhai always reaches a
            // root through a `Target`, whose shared arm hands over the guard
            // rather than the cell. Walking the cell would take the host down,
            // so this is a panic-safety fix and not only a correctness one.
            if is_shared!(*root.as_mut()) {
                let mut guard = or_raise!(root.as_mut().write_lock::<Dynamic>(), {
                    let name = root_name(program, chain).unwrap_or_default();
                    Box::new(EvalAltResult::ErrorDataRace(name.to_string(), pos))
                });
                self.walk_chain(
                    program,
                    chain,
                    &chain.steps,
                    &mut guard,
                    &mut operands,
                    value,
                    pos,
                )
            } else {
                self.walk_chain(
                    program,
                    chain,
                    &chain.steps,
                    root.as_mut(),
                    &mut operands,
                    value,
                    pos,
                )
            }
        });

        // The receiver is the one root that travels rather than being walked
        // where it lives, so it goes back — and it goes back however the walk
        // ended. Rhai reaches `this` through a pointer into the caller's
        // storage, so a body that mutated and then raised has already written;
        // restoring only on success would discard exactly that.
        if let RootValue::This(value) = root {
            self.this = Some(value);
        }

        let (out, _) = result?;
        self.truncate_stack(operands_at);
        Ok(out)
    }

    /// Find what a chain is rooted at, resolving a name if that is what it is.
    ///
    /// The search is `load_named`'s and the order is observable: a resolver
    /// registered with `Engine::on_var` sees the name before the scope does,
    /// and a name in no scope is looked for among the global modules before it
    /// is reported missing. It runs exactly once — a chain is one instruction,
    /// so unlike [`Op::CallRef`] there is nothing to resolve twice.
    ///
    /// `ErrorVariableNotFound` is reported against the *variable*, which is why
    /// [`Root::Named`] carries a position of its own.
    fn chain_root<'s>(
        &mut self,
        program: &Program,
        chain: &Chain,
        scope: &'s mut Scope,
        base: usize,
        operands_at: usize,
        pos: Position,
    ) -> Result<ChainRoot<'s>, Box<EvalAltResult>> {
        match chain.root {
            Root::Local { slot, .. } => {
                let index = base + slot as usize;
                if index >= scope.len() {
                    return Err(malformed(format!(
                        "chain root slot {index} is out of scope"
                    )));
                }
                Ok(ChainRoot {
                    value: RootValue::Entry(scope.get_mut_by_index(index)),
                    pos,
                })
            }

            // The `this` position wins over the chain's for the same reason a
            // name's does: this lookup can fail, and Rhai blames the `this`
            // rather than the `.` after it (`eval/chaining.rs:519-527`).
            Root::This { pos: this_pos } => {
                let value = or_raise!(
                    self.this.take(),
                    Box::new(EvalAltResult::ErrorUnboundThis(this_pos))
                );
                Ok(ChainRoot {
                    value: RootValue::This(value),
                    pos: this_pos,
                })
            }

            // A name has a position of its own, and it wins: the lookup below
            // can fail, and Rhai blames the variable rather than the chain.
            Root::Named { name, pos: var_pos } => {
                let name = or_raise!(program.name(name), malformed(format!("no name {name}")));

                // A resolver hands back a value rather than a place, which is
                // what makes writing through it an error.
                if let Some(value) = self.resolve_var(name, scope, var_pos)? {
                    return Ok(ChainRoot {
                        value: RootValue::Detached(value),
                        pos: var_pos,
                    });
                }
                // By index rather than through `Scope::get_mut`, which refuses
                // a constant outright. A constant is walked here rather than
                // refused: the read-only mode the entry carries is what turns
                // away an assignment or a non-pure method, and it is the same
                // mode Rhai's `Target` into the entry would have carried.
                if let Some(index) = scope.search(name) {
                    return Ok(ChainRoot {
                        value: RootValue::Entry(scope.get_mut_by_index(index)),
                        pos: var_pos,
                    });
                }
                // A constant a host published with `Module::set_var`. Not
                // marked read-only, because Rhai does not mark it either
                // (`eval/expr.rs:151` against `:122`) — so a chain assigns
                // into the copy and discards it, where writing to the name
                // directly is refused.
                let value = self
                    .engine
                    .global_modules
                    .iter()
                    .find_map(|module| module.get_var(name));
                Ok(ChainRoot {
                    value: RootValue::Detached(or_raise!(value, missing(name, var_pos))),
                    pos: var_pos,
                })
            }

            // Moved off the operand stack rather than copied. The slot it
            // leaves behind is unit, and nothing reads it again: `run_chain`
            // truncates past it, and an error unwinds the stack to the frame
            // floor.
            Root::Temporary => Ok(ChainRoot {
                value: RootValue::Temporary(
                    stack_take(self, operands_at + chain.operands as usize).flatten(),
                ),
                pos,
            }),
        }
    }

    /// One level of the walk. Returns the value and whether anything below may
    /// have written.
    #[allow(clippy::too_many_arguments)]
    fn walk_chain(
        &mut self,
        program: &Program,
        chain: &Chain,
        steps: &[Step],
        target: &mut Dynamic,
        operands: &mut [Dynamic],
        value: Option<Dynamic>,
        pos: Position,
    ) -> Result<(Dynamic, bool), Box<EvalAltResult>> {
        let Some((step, rest)) = steps.split_first() else {
            // The end of the chain, reached with nothing to do: a bare `a` is
            // not a chain, so this only happens for an empty step list.
            return Ok((clone_value(target), false));
        };
        let last = rest.is_empty();
        let coalescing = match step {
            Step::Index { flags, .. }
            | Step::Property { flags, .. }
            | Step::Method { flags, .. } => flags.contains(StepFlags::SKIP_IF_UNIT),
        };
        if coalescing && target.is_unit() {
            return Ok((Dynamic::UNIT, false));
        }

        // Which step a failure gets blamed on. The walk only descends, so the
        // last one written is the one that raised.
        self.chain_step = chain.steps.len() - steps.len();

        match step {
            // Without indexing of either kind there is no `[..]` to compile,
            // so a step that says otherwise came from a corrupted artifact.
            #[cfg(all(feature = "no_index", feature = "no_object"))]
            Step::Index { .. } => Err(malformed(
                "an index step, in a build with no indexing".to_string(),
            )),

            #[cfg(not(all(feature = "no_index", feature = "no_object")))]
            Step::Index {
                operand,
                flags: _,
                pos: idx_pos,
                bracket,
            } => {
                let idx = or_raise!(
                    operands.get_mut(*operand as usize),
                    malformed("chain index operand missing".to_string())
                );
                let mut idx = clone_value(idx);
                // Rhai reports an out-of-bounds index against the index and a
                // value that cannot be indexed at all against this step's `[`.
                // Both belong to the step, and neither is the chain's.
                let idx_pos = *idx_pos;
                let bracket = *bracket;

                // Split out so the borrow `get_indexed_mut` takes of `target`
                // ends when it returns: the fallback below needs `target`
                // again, and a `Target` in scope would still be holding it.
                match self.index_by_reference(
                    program, chain, rest, target, &mut idx, idx_pos, operands, value, last,
                    bracket, pos,
                )? {
                    Indexed::Done(out, changed) => Ok((out, changed)),
                    Indexed::NoReference(value) => {
                        self.assign_through_indexer(
                            program, chain, target, &mut idx, value, bracket,
                        )?;
                        Ok((Dynamic::UNIT, true))
                    }
                }
            }

            Step::Property {
                name,
                getter,
                setter,
                flags: _,
                pos: step_pos,
            } => self.walk_property(
                program, chain, rest, target, operands, value, pos, *step_pos, *name, *getter,
                *setter,
            ),

            Step::Method {
                name,
                argc,
                operand,
                flags: _,
                pos: step_pos,
            } => {
                let step_pos = *step_pos;
                let name_index = *name;
                let name = or_raise!(
                    program.name(name_index),
                    malformed(format!("no name {name_index}"))
                );
                let first = *operand as usize;
                let argc = *argc as usize;
                if first + argc > operands.len() {
                    return Err(malformed("chain method arguments missing".to_string()));
                }

                // A method call is where Rhai tries the receiver's type before
                // the plain name (`func/call.rs:614-629`), and the only place it
                // does. `argc` already excludes the receiver, which is the arity
                // the script side is keyed on (`parser.rs:2128-2145`).
                //
                // Consulting our own table first is safe because nothing can get
                // between: `import` pushes onto `global.modules`, not
                // `global.lib` (`eval/stmt.rs:947`), and `global.lib` is only
                // ever pushed where an AST is being run.
                let type_name = target.type_name();
                let compiled = {
                    let typed = self.engine.map_type_name(type_name);
                    program
                        .method(name_index, argc, typed)
                        .map(|f| (&*f.param_names, f.chunk))
                };

                let mut args: FnArgsVec<Dynamic> =
                    operands[first..first + argc].iter().cloned().collect();
                let out = if let Some((params, chunk)) = compiled {
                    // The receiver is moved into the frame and moved back, so a
                    // write through `this` lands in the level above — which for
                    // a chain rooted at a local is the scope entry itself.
                    let (bound, write_back) = bind_this(target);
                    let at = self.depth;
                    self.push_all(args.into_iter());
                    // A chained call always starts an empty scope.
                    let mut new_scope = self.take_scope();
                    let (result, returned) = self.call_compiled_with_this(
                        program,
                        name,
                        params,
                        chunk,
                        at,
                        &mut new_scope,
                        true,
                        step_pos,
                        Some(bound),
                    );
                    self.give_scope(new_scope);
                    self.truncate_stack(at);
                    // Before `?`: a body that mutated and then raised has
                    // already written, as it would through Rhai's pointer.
                    unbind_this(target, returned, write_back);
                    result?
                } else {
                    let mut call_args: FnArgsVec<&mut Dynamic> = core::iter::once(&mut *target)
                        .chain(args.iter_mut())
                        .collect();
                    call_engine(
                        self.engine,
                        &mut self.global,
                        &mut self.caches,
                        &mut Scope::new(),
                        name,
                        &mut call_args,
                        true,
                        true,
                        step_pos,
                        None,
                    )?
                };

                if last {
                    match value {
                        // `a.f() = x` is not something Rhai parses.
                        Some(_) => Err(malformed("assignment to a method call".to_string())),
                        None => Ok((out, true)),
                    }
                } else {
                    let mut inner = out;
                    let (out, _) =
                        self.walk_chain(program, chain, rest, &mut inner, operands, value, pos)?;
                    // Whatever the sub-chain did, it did to the method's
                    // return value, which nothing owns.
                    Ok((out, true))
                }
            }
        }
    }

    /// One `[i]` step, taken through a reference into the container.
    ///
    /// Returns [`Indexed::NoReference`] when there is no reference to be had —
    /// a custom indexer being assigned through — handing the value back so the
    /// caller can take the long way round once this borrow has ended.
    #[allow(clippy::too_many_arguments)]
    #[cfg(not(all(feature = "no_index", feature = "no_object")))]
    fn index_by_reference(
        &mut self,
        program: &Program,
        chain: &Chain,
        rest: &[Step],
        target: &mut Dynamic,
        idx: &mut Dynamic,
        idx_pos: Position,
        operands: &mut [Dynamic],
        value: Option<Dynamic>,
        last: bool,
        bracket: Position,
        pos: Position,
    ) -> Result<Indexed, Box<EvalAltResult>> {
        let assigning = last && value.is_some();
        let mut detached = Scope::new();

        // Cloned *before* the call, as Rhai clones it (`eval/chaining.rs:706`).
        // `get_indexed_mut` may reach a custom indexer, and a native's by-value
        // parameter is bound by `take` (`func/register.rs:69`) — so afterwards
        // there is nothing left to address the setter with. Only the paths that
        // can write need it; a read returns below without ever looking.
        let index_for_setter = (!last || value.is_some()).then(|| clone_value(idx));

        let mut item = match self.engine.get_indexed_mut(
            &mut self.global,
            &mut self.caches,
            &mut detached,
            None,
            target,
            idx,
            idx_pos,
            bracket,
            // Auto-vivify a missing map key only when writing, as Rhai does
            // for the assignment case (`eval/chaining.rs:791`).
            assigning,
            // And do not reach for a custom indexer when writing: a value it
            // handed back could not be assigned through. Rhai asks the same
            // way and takes the error as its signal.
            !assigning,
        ) {
            Ok(item) => item,
            Err(err) if assigning && matches!(*err, EvalAltResult::ErrorIndexingType(..)) => {
                return Ok(Indexed::NoReference(value.expect("assigning")));
            }
            Err(err) => return Err(err),
        };

        // A read changes nothing, so it consumes the target and there is
        // nothing to put back.
        if last && value.is_none() {
            return Ok(Indexed::Done(item.take_or_clone(), false));
        }

        let temp = item.is_temp_value();
        let (out, changed) = if last {
            let value = value.expect("checked above");
            self.store(
                program,
                chain_op(program, chain)?,
                item.as_mut(),
                value,
                pos,
            )?;
            (Dynamic::UNIT, true)
        } else {
            // Straight through the borrow: for an array, a map or a blob this
            // *is* the container's element, so a mutation below lands where
            // Rhai's would.
            self.walk_chain(program, chain, rest, item.as_mut(), operands, value, pos)?
        };

        // Bit-fields, string characters and blob bytes cannot be pointed at
        // directly, so `Target` carries a copy and this is what puts it back
        // (`eval/target.rs:282`).
        item.propagate_changed_value(pos)?;

        if temp && changed {
            // The element was a temporary — a custom indexer's — so the setter
            // is the only way back (`eval/chaining.rs:744`).
            let mut updated = item.take_or_clone();
            let mut index = index_for_setter.expect("a read returns before here");
            self.call_indexer_set(target, &mut index, &mut updated, bracket)?;
        }

        Ok(Indexed::Done(out, changed))
    }

    /// Assign through a custom indexer, which cannot hand out a reference.
    ///
    /// An op-assignment has to read the current value back through the getter
    /// first, and Rhai *ignores* a getter that fails here — a write-only
    /// indexer takes the new value as-is (`eval/chaining.rs:812`).
    #[cfg(not(all(feature = "no_index", feature = "no_object")))]
    fn assign_through_indexer(
        &mut self,
        program: &Program,
        chain: &Chain,
        target: &mut Dynamic,
        index: &mut Dynamic,
        value: Dynamic,
        pos: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let mut new_val = value;

        if matches!(chain.tail, Tail::Assign { op: Some(_) }) {
            let mut probe = clone_value(index);
            if let Ok(mut current) = self.engine.call_indexer_get(
                &mut self.global,
                &mut self.caches,
                target,
                &mut probe,
                pos,
            ) {
                self.store(
                    program,
                    chain_op(program, chain)?,
                    &mut current,
                    new_val,
                    pos,
                )?;
                new_val = current;
            }
        }

        self.call_indexer_set(target, index, &mut new_val, pos)
    }

    /// Put an element back into a container that had no reference to give.
    ///
    /// A custom indexer returns a value, so a mutation below it landed in a
    /// temporary; this is the replay Rhai does at `eval/chaining.rs:744`,
    /// including swallowing "this type cannot be indexed" the way it does.
    #[cfg(not(all(feature = "no_index", feature = "no_object")))]
    #[inline(always)]
    fn call_indexer_set(
        &mut self,
        target: &mut Dynamic,
        index: &mut Dynamic,
        value: &mut Dynamic,
        pos: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let result = self.engine.call_indexer_set(
            &mut self.global,
            &mut self.caches,
            target,
            index,
            value,
            true,
            pos,
        );
        match result.map(|_| ()) {
            Ok(()) => Ok(()),
            Err(err) if matches!(*err, EvalAltResult::ErrorIndexingType(..)) => Ok(()),
            Err(mut err) => {
                if err.position().is_none() {
                    err.set_position(pos);
                }
                Err(err)
            }
        }
    }

    // Try to get a property through an indexer.
    //
    // This requires `no_index` and `no_object` to be off,
    // otherwise it just passes the error through.
    fn try_index_get(
        &mut self,
        target: &mut Dynamic,
        key: &str,
        err: Box<EvalAltResult>,
        pos: Position,
    ) -> VmResult {
        #[cfg(not(any(feature = "no_index", feature = "no_object")))]
        return match *err {
            EvalAltResult::ErrorDotExpr(..) => {
                let mut index = self.strings_interner.get(key).into();
                self.engine
                    .call_indexer_get(&mut self.global, &mut self.caches, target, &mut index, pos)
                    .map_err(|err2| match *err2 {
                        EvalAltResult::ErrorIndexingType(..) => err,
                        _ => positioned(err2, pos),
                    })
            }
            _ => Err(err),
        };
        #[cfg(any(feature = "no_index", feature = "no_object"))]
        {
            let _ = (target, key, pos);
            return Err(err);
        }
    }

    // Try to set a property through an index setter.
    //
    // This requires `no_index` and `no_object` to be off,
    // otherwise it just passes the error through.
    fn try_index_set(
        &mut self,
        target: &mut Dynamic,
        key: &str,
        value: &mut Dynamic,
        fail_silently: bool,
        err: Box<EvalAltResult>,
        pos: Position,
    ) -> VmResult {
        #[cfg(not(any(feature = "no_index", feature = "no_object")))]
        return match *err {
            EvalAltResult::ErrorDotExpr(..) => {
                let mut index = self.strings_interner.get(key).into();
                match self
                    .engine
                    .call_indexer_set(
                        &mut self.global,
                        &mut self.caches,
                        target,
                        &mut index,
                        value,
                        true,
                        pos,
                    )
                    .map(|_| ())
                {
                    Ok(()) => Ok(Dynamic::UNIT),
                    Err(err2) if matches!(*err2, EvalAltResult::ErrorIndexingType(..)) => {
                        if fail_silently {
                            Ok(Dynamic::UNIT)
                        } else {
                            Err(err)
                        }
                    }
                    Err(err2) => Err(positioned(err2, pos)),
                }
            }
            _ => Err(err),
        };
        #[cfg(any(feature = "no_index", feature = "no_object"))]
        {
            let _ = (target, key, value, fail_silently, pos);
            return Err(err);
        }
    }

    /// `.name`, which is a key on a map and a getter call on anything else.
    ///
    /// The distinction is Rhai's and it is made at runtime, not at parse time
    /// (`eval/chaining.rs:898`). It matters for more than speed: a map hands
    /// back a reference, so a mutation below lands in the map, while a getter
    /// hands back a value that has to be given to the setter afterwards.
    #[allow(clippy::too_many_arguments)]
    fn walk_property(
        &mut self,
        program: &Program,
        chain: &Chain,
        rest: &[Step],
        target: &mut Dynamic,
        operands: &mut [Dynamic],
        value: Option<Dynamic>,
        pos: Position,
        // The property's own position, which is where Rhai blames a getter or
        // setter that does not exist (`eval/chaining.rs:1039`). `pos` is the
        // chain's, and stays that for everything else.
        step_pos: Position,
        name: u32,
        getter: u32,
        setter: u32,
    ) -> Result<(Dynamic, bool), Box<EvalAltResult>> {
        let last = rest.is_empty();

        // The name is a map key for maps, and the same string is what a host
        // type's fallback string indexer is addressed with.
        let key = or_raise!(program.name(name), malformed(format!("no name {name}")));

        // A map is the one property holder that is not a host type, and
        // `no_object` removes both it and the syntax that would reach one.
        #[cfg(not(feature = "no_object"))]
        if target.is_map() {
            let mut map = or_raise!(
                target.write_lock::<Map>(),
                malformed("a map that is not a map".to_string())
            );

            // Only a write creates a key. Rhai passes `add_if_not_found` for
            // an assignment (`eval/chaining.rs:930`) and withholds it for a
            // read (`:959`) and for a step on the way through (`:1086`), so
            // reading `m.absent` must leave `m` alone — otherwise a closure
            // holding the map sees a key nobody wrote.
            if last {
                if let Some(value) = value {
                    let entry = map.entry(key.into()).or_insert(Dynamic::UNIT);
                    self.store(program, chain_op(program, chain)?, entry, value, pos)?;
                    return Ok((Dynamic::UNIT, true));
                }
                return match map.get(key) {
                    Some(entry) => Ok((clone_value(entry), false)),
                    None => self.absent_key(key, step_pos).map(|unit| (unit, false)),
                };
            }

            return match map.get_mut(key) {
                Some(entry) => self.walk_chain(program, chain, rest, entry, operands, value, pos),
                // Rhai walks on into a detached unit, so whatever the rest of
                // the chain does to it is discarded (`eval/chaining.rs:211`).
                None => {
                    let mut absent = self.absent_key(key, step_pos)?;
                    drop(map);
                    self.walk_chain(program, chain, rest, &mut absent, operands, value, pos)
                }
            };
        }

        // A host type: getter in, setter out.
        let call = |vm: &mut Self, fn_name: u32, args: &mut [&mut Dynamic]| -> VmResult {
            let fn_name = or_raise!(
                program.name(fn_name),
                malformed(format!("no name {fn_name}"))
            );
            call_engine(
                vm.engine,
                &mut vm.global,
                &mut vm.caches,
                &mut Scope::new(),
                fn_name,
                args,
                true,
                true,
                step_pos,
                None,
            )
        };

        if last {
            if let Some(value) = value {
                // `x.p += 1` has to read `p` back through the getter before it
                // can add to it — the setter takes a finished value.
                let mut new_val = if matches!(chain.tail, Tail::Assign { op: Some(_) }) {
                    let mut current = call(self, getter, &mut [target])
                        .or_else(|err| self.try_index_get(target, key, err, step_pos))?;
                    self.store(program, chain_op(program, chain)?, &mut current, value, pos)?;
                    current
                } else {
                    value
                };
                // A setter's return value is thrown away, as in Rhai.
                let _ = call(self, setter, &mut [target, &mut new_val]).or_else(|err| {
                    self.try_index_set(target, key, &mut new_val, false, err, step_pos)
                })?;
                return Ok((Dynamic::UNIT, true));
            }
            let out = call(self, getter, &mut [target])
                .or_else(|err| self.try_index_get(target, key, err, step_pos))?;
            return Ok((out, false));
        }

        // A getter returns a value, so the rest of the chain works on a
        // temporary. Rhai puts it back through the setter when the sub-chain
        // was a method call, and skips the setter otherwise.
        let mut temp = call(self, getter, &mut [target])
            .or_else(|err| self.try_index_get(target, key, err, step_pos))?;
        let (out, changed) =
            self.walk_chain(program, chain, rest, &mut temp, operands, value, pos)?;
        if changed {
            let _ = call(self, setter, &mut [target, &mut temp])
                .or_else(|err| self.try_index_set(target, key, &mut temp, true, err, step_pos))?;
        }
        Ok((out, changed))
    }

    /// Store into a slot the walk arrived at, through an operator if there is
    /// one.
    ///
    /// Same resolution order as a plain local assignment, and for the same
    /// reason: `x += y` is not `x = x + y` unless nothing implements `+=`.
    /// The built-in op-assignment for these operands, if Rhai has one.
    ///
    /// Inlined deliberately: `x += 1` in a loop is entirely this, and routing
    /// it through the out-of-line resolution below measured 10% on the
    /// tight-loop benchmark. Inlining the *whole* of `store` instead costs
    /// more than it saves — it took `branch heavy` from 1.59x to 1.41x — so
    /// the split is where the two paths part.
    #[inline]
    fn store_builtin(
        &mut self,
        op: &AssignOp,
        target: &mut Dynamic,
        rhs: &mut Dynamic,
        pos: impl Fn() -> Position,
    ) -> StoreBuiltinOutcome {
        #[cfg(feature = "grain-jit")]
        let fast_operators = jit::fast_operators(self) != 0;
        #[cfg(not(feature = "grain-jit"))]
        let fast_operators = self.engine.fast_operators();
        if !fast_operators {
            return StoreBuiltinOutcome {
                handled: false,
                error: None,
            };
        }

        // `AssignOp::kind` is the `+=` token already decoded, pooled beside the
        // token itself. An integer target and an integer operand reach the
        // integer add from here with no resolution at all — no token match, no
        // walk over the operand pair, no indirect call over
        // `&mut [&mut Dynamic]`, and no downcast on the far side of it.
        //
        // A pair with no entry answers `None` and falls through to the
        // resolution below, which answers what it always did. The set of pairs
        // is the op-assignment table's, not the operator table's — see
        // `apply_assign`.
        if let Some(kind) = op.kind {
            match apply_assign(kind, target, rhs) {
                Ok(Some(())) => {
                    return StoreBuiltinOutcome {
                        handled: true,
                        error: None,
                    };
                }
                Ok(None) => {}
                Err(mut err) => {
                    if err.position().is_none() {
                        err.set_position(pos());
                    }
                    return StoreBuiltinOutcome {
                        handled: true,
                        error: Some(err),
                    };
                }
            }
        }

        let Some((func, need_context)) = get_builtin_op_assignment_fn(&op.op_assign, target, rhs)
        else {
            return StoreBuiltinOutcome {
                handled: false,
                error: None,
            };
        };
        let context = need_context.then(|| (self.engine, "", None, &self.global, pos()).into());
        match func(context, &mut [target, rhs]) {
            Ok(_) => StoreBuiltinOutcome {
                handled: true,
                error: None,
            },
            Err(mut err) => {
                if err.position().is_none() {
                    err.set_position(pos());
                }
                StoreBuiltinOutcome {
                    handled: true,
                    error: Some(err),
                }
            }
        }
    }

    fn store(
        &mut self,
        program: &Program,
        op: Option<&AssignOp>,
        target: &mut Dynamic,
        mut rhs: Dynamic,
        pos: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let Some(op) = op else {
            *target = rhs;
            return Ok(());
        };

        let done = self.store_builtin(op, target, &mut rhs, || pos);
        if done.handled {
            return match done.error {
                Some(error) => Err(error),
                None => Ok(()),
            };
        }

        let op_assign_name = or_raise!(
            program.name(op.op_assign_name),
            malformed(format!("no op-assign name {}", op.op_assign_name))
        );
        let op_name = or_raise!(
            program.name(op.op_name),
            malformed(format!("no operator name {}", op.op_name))
        );

        // The real scope may be borrowed by the target, and dispatch does not
        // read it anyway — operators resolve against the engine.
        let result = call_engine(
            self.engine,
            &mut self.global,
            &mut self.caches,
            &mut Scope::new(),
            op_assign_name,
            &mut [target, &mut rhs],
            true,
            false,
            pos,
            None,
        );
        match result {
            Ok(_) => Ok(()),
            Err(err)
                if matches!(&*err,
                    EvalAltResult::ErrorFunctionNotFound(name, ..)
                        if name.starts_with(op_assign_name)) =>
            {
                let value = call_engine(
                    self.engine,
                    &mut self.global,
                    &mut self.caches,
                    &mut Scope::new(),
                    op_name,
                    &mut [target, &mut rhs],
                    true,
                    false,
                    pos,
                    None,
                )?;
                *target = value;
                Ok(())
            }
            Err(err) => Err(dispatch_failure(err, pos)),
        }
    }

    /// Read a variable no slot names, the way Rhai's `search_scope_only` does
    /// (`eval/expr.rs:107-155`).
    ///
    /// Three places in a fixed order, and the order is observable: a resolver
    /// registered with `Engine::on_var` sees the name before the scope does,
    /// and a name in no scope is looked for among the global modules before it
    /// is reported missing.
    ///
    /// `flatten` is what the two reads differ by, and only for a scope entry:
    /// a value position wants what a shared cell contains, and a capture wants
    /// the cell. The other two places can only ever produce a value.
    ///
    /// Kept out of the dispatch loop for the reason [`Vm::call_compiled`] is.
    fn load_named(
        &mut self,
        name: &str,
        scope: &mut Scope,
        flatten: bool,
        pos: Position,
    ) -> VmResult {
        // A resolver hands back a value, not a place, so it is read-only —
        // which is what makes assigning to one an error.
        if let Some(value) = self.resolve_var(name, scope, pos)? {
            return Ok(value);
        }

        if let Some(value) = scope.get(name) {
            return Ok(if flatten {
                flatten_clone_value(value)
            } else {
                clone_value(value)
            });
        }

        // A constant a host published with `Module::set_var`.
        if let Some(value) = self
            .engine
            .global_modules
            .iter()
            .find_map(|module| module.get_var(name))
        {
            return Ok(value);
        }

        Err(missing(name, pos))
    }

    /// Ask the resolver a host registered with `Engine::on_var`, if there is
    /// one.
    ///
    /// `Ok(None)` covers both "no resolver" and "the resolver declined", which
    /// are the same thing to every caller.
    fn resolve_var(
        &mut self,
        name: &str,
        scope: &mut Scope,
        pos: Position,
    ) -> Result<Option<Dynamic>, Box<EvalAltResult>> {
        // Copied out so the borrow is of the engine rather than of `self`,
        // which the context below needs mutably.
        let engine = self.engine;
        let Some(resolver) = &engine.resolve_var else {
            return Ok(None);
        };

        let before = scope.len();
        let context = EvalContext::new(engine, &mut self.global, &mut self.caches, scope, None);
        // Index zero: Rhai passes the slot its parser resolved, and a name
        // that reached here had none.
        let resolved = resolver(name, 0, context);

        // A resolver that pushed onto the scope has moved every entry a
        // parse-time index named, so Rhai stops trusting those from here on.
        // Nothing this compiler emits depends on them — its slots are counted
        // from a base taken before the run — but a fragment's do.
        if scope.len() != before {
            self.global.always_search_scope = true;
        }

        match resolved {
            Ok(Some(value)) => Ok(Some(value.into_read_only())),
            Ok(None) => Ok(None),
            Err(err) => Err(dispatch_failure(err, pos)),
        }
    }

    /// Assign to a variable no slot names.
    ///
    /// Rhai reaches the target through the same search and then refuses
    /// anything that is not a reference it can write through: a value the
    /// resolver produced, a module's constant, a `const` entry
    /// (`eval/stmt.rs:330-344` and `eval/stmt.rs:118-122`). All three are
    /// `ErrorAssignmentToConstant`, so the distinction never reaches a script.
    fn assign_named(
        &mut self,
        program: &Program,
        op: Option<&AssignOp>,
        name: &str,
        rhs: Dynamic,
        scope: &mut Scope,
        pos: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let constant = || {
            Box::new(EvalAltResult::ErrorAssignmentToConstant(
                name.to_string(),
                pos,
            ))
        };

        // Try the variable resolver first.
        if let Some(mut value) = self.resolve_var(name, scope, pos)? {
            if value.is_read_only() {
                return Err(constant());
            }
            let mut target = place(&mut value, name, pos)?;
            return self.store(program, op, &mut target, rhs, pos);
        }

        // Search the scope.
        match scope.get_mut_raw(name) {
            Some(None) => Err(constant()),
            Some(Some(entry)) => {
                let mut target = place(entry, name, pos)?;
                self.store(program, op, &mut target, rhs, pos)
            }
            // Not a variable at all. A module's is a value rather than a
            // place, so writing to one is the same refusal as writing to a
            // `const`.
            None => Err(
                if self
                    .engine
                    .global_modules
                    .iter()
                    .any(|module| module.get_var(name).is_some())
                {
                    constant()
                } else {
                    missing(name, pos)
                },
            ),
        }
    }

    /// Call a function pointer, preferring a chunk we compiled.
    ///
    /// The pointer sits under its arguments. Rhai's own dispatch would work
    /// for all of this, but it cannot reach our chunks — the compiled function
    /// table is keyed on names from the pool, and a pointer carries a string —
    /// so the name is matched against it first and only the miss goes to
    /// `call_raw`.
    #[allow(clippy::too_many_arguments)]
    fn call_fn_ptr(
        &mut self,
        program: &Program,
        argc: usize,
        method: bool,
        receiver: Option<Receiver>,
        scope: &mut Scope,
        frame_base: usize,
        pos: Position,
    ) -> VmResult {
        let base = or_raise!(
            self.depth.checked_sub(argc + 1),
            malformed("function pointer call is missing its target".into())
        );
        let mut at = base;

        // In method position a target that is not a pointer means the *first
        // argument* is one and the target is the receiver — `obj.call(f, ..)`
        // is how a closure is called against a `this`. Rhai reports the
        // mismatch against that argument, not against the target, which is why
        // the position moves with it.
        let mut receiver_at = None;
        if method && !operand_ref(&self.stack[at]).is::<FnPtr>() {
            receiver_at = Some(at);
            at += 1;
            if at >= self.depth {
                return Err(
                    self.mismatch::<FnPtr>(operand_ref(&self.stack[at - 1]).type_name(), pos)
                );
            }
        }

        let pointer = or_raise!(
            clone_operand(&self.stack[at]).try_cast::<FnPtr>(),
            self.mismatch::<FnPtr>(operand_ref(&self.stack[at]).type_name(), pos)
        );

        let taken = self.depth - at - 1;
        let curried = pointer.curry().len();
        // A receiver does not change which function a pointer names, only what
        // the callee's `this` is: Rhai keys its own script pointers on the
        // declared parameter count alone (`types/fn_ptr.rs:422`), and binds the
        // receiver alongside them.
        let function = program
            .function_named(pointer.fn_name(), curried + taken)
            .map(|f| (&*f.param_names, f.chunk));

        // Bound once, whichever path takes it. Curried values are spliced in
        // above `at`, so the receiver's index is unaffected either way.
        let (mut bound, write_back) = match receiver_at {
            Some(index) => {
                let (value, write_back) = bind_this(operand_mut(&mut self.stack[index]));
                (Some(value), write_back)
            }
            None => (None, false),
        };

        let outcome = if let Some((params, chunk)) = function {
            // Curried arguments go in front of the call's own, which is what
            // currying means and where the callee's parameters expect them.
            let first = at + 1;
            let curried = pointer.curry();
            self.open_slots(first, curried.len());
            for (offset, value) in curried.iter().enumerate() {
                set_operand(&mut self.stack[first + offset], clone_value(value));
            }

            // A function pointer call always starts with an empty scope.
            let mut new_scope = self.take_scope();

            let (result, returned) = self.call_compiled_with_this(
                program,
                pointer.fn_name(),
                params,
                chunk,
                first,
                &mut new_scope,
                true,
                pos,
                bound.take(),
            );
            self.give_scope(new_scope);
            bound = returned;
            result
        } else {
            // Anything else is Rhai's: a native function, a name registered
            // elsewhere, or a pointer it built itself.
            let mut args = self.take_values_from(at + 1);
            let context = (self.engine, pointer.fn_name(), None, &self.global, pos).into();
            pointer
                .call_raw(&context, bound.as_mut(), &mut args)
                .map_err(|mut err| {
                    if err.position().is_none() {
                        err.set_position(pos);
                    }
                    err
                })
        };

        // Before `?`, as everywhere else: Rhai binds the receiver by reference,
        // so a closure that writes and then raises has already written.
        if let Some(index) = receiver_at {
            unbind_this(operand_mut(&mut self.stack[index]), bound, write_back);
            if write_back {
                let updated = clone_operand(&self.stack[index]);
                self.return_receiver(program, receiver, updated, scope, frame_base)?;
            }
        }

        let value = outcome?;
        self.truncate_stack(base);
        Ok(value)
    }

    /// Carry a write through `obj.call(f)`'s `this` back to `obj` itself.
    ///
    /// Rhai binds the receiver by reference (`func/call.rs:862`), so the write
    /// lands in the variable. The operand stack only ever held a copy of it,
    /// and this is what puts the copy back where it came from.
    ///
    /// Nothing to do for a shared receiver: it arrived *as* the cell, so the
    /// write already landed where every holder can see it — `run_chain`'s rule
    /// for a chain root, and for the same reason.
    fn return_receiver(
        &mut self,
        program: &Program,
        receiver: Option<Receiver>,
        value: Dynamic,
        scope: &mut Scope,
        frame_base: usize,
    ) -> Result<(), Box<EvalAltResult>> {
        if is_shared!(value) || value.is_read_only() {
            return Ok(());
        }

        match receiver {
            Some(Receiver::Local(slot)) => {
                let index = frame_base + slot as usize;
                if index >= scope.len() {
                    return Err(malformed(format!("local slot {slot} is out of scope")));
                }
                *scope.get_mut_by_index(index) = value;
            }
            Some(Receiver::Named(var)) => {
                let name = or_raise!(program.name(var), malformed(format!("no name {var}")));
                // A resolver's answer or a module constant has no entry behind
                // it, and Rhai could not have written through one either.
                if let Some(entry) = scope.get_mut(name) {
                    *entry = value;
                }
            }
            Some(Receiver::This) => {
                if let Some(entry) = self.this.as_mut() {
                    *entry = value;
                }
            }
            // A temporary, which Rhai mutates a copy of too.
            None => {}
        }
        Ok(())
    }

    /// Concatenate the segments of an interpolated string, reproducing
    /// `eval/expr.rs:280-304`.
    ///
    /// Every step of it is load-bearing. A **string** segment is written
    /// straight out and never reaches dispatch, so a host's `to_string` for
    /// strings is not consulted here even though `+` would consult it.
    /// Anything else goes through Rhai's own rendering, which calls **native**
    /// functions only — a script `fn to_string` is invisible to it — and
    /// substitutes the mapped type name when the call returns a non-string.
    /// The size limit is checked after every segment against the running
    /// total, not once at the end.
    fn append_segment(
        &mut self,
        segment: Dynamic,
        pos: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        use core::fmt::Write;

        let mut item = segment.flatten();
        let mut rendered = None;

        // A string is written straight out and never reaches dispatch, so a
        // host's `to_string` for strings is not consulted here even though `+`
        // would consult it.
        if !item.is_string() {
            let context = (self.engine, FUNC_TO_STRING, None, &self.global, pos).into();
            rendered = Some(print_with_func(FUNC_TO_STRING, &context, &mut item));
        }

        let mut buffer = or_raise!(
            self.values_mut()
                .last_mut()
                .and_then(|value| value.write_lock::<ImmutableString>()),
            malformed("interpolation lost its buffer".into())
        );

        // `make_mut` is in place while the buffer is uniquely held, which on
        // the operand stack it is — so this is one growing allocation rather
        // than one per segment.
        match rendered {
            Some(text) => write!(buffer.make_mut(), "{text}"),
            None => write!(buffer.make_mut(), "{item}"),
        }
        .expect("writing to a string cannot fail");
        let len = buffer.len();
        drop(buffer);

        // After every segment, against the running total — a script must not
        // be able to build a string past `max_string_size` and hand it over
        // whole.
        #[cfg(not(feature = "unchecked"))]
        {
            self.engine.throw_on_size((0, 0, len)).map_err(|mut err| {
                if err.position().is_none() {
                    err.set_position(pos);
                }
                err
            })
        }
        // `unchecked` removes the limits, and with them the only reason to have
        // measured.
        #[cfg(feature = "unchecked")]
        {
            let _ = (len, pos);
            Ok(())
        }
    }

    /// Start iterating a value, the way Rhai's `for` does
    /// (`eval/stmt.rs:680-703`).
    ///
    /// Three places are searched by `TypeId`, in order, and the order is
    /// Rhai's: the modules in the global namespace, then the imports, then the
    /// statically registered sub-modules. Nothing matching is `ErrorFor`.
    ///
    /// The iterable is flattened first — so iterating a captured array walks a
    /// snapshot rather than the shared cell — and is consumed by value, which
    /// is why the iterator is built once and held for the life of the loop.
    fn iter_init(&mut self, iterable: Dynamic, pos: Position) -> Result<(), Box<EvalAltResult>> {
        let iterable = iterable.flatten();
        let type_id = iterable.type_id();

        // An exclusive integer range walks itself, but only if the entry the
        // search below would have found is the type's own iteration — anything
        // registered through `Module::set_iter` decides what the type means and
        // has to be called. Asked once per loop, and only for a range.
        if type_id == TypeId::of::<crate::ExclusiveRange>() && self.int_range_is_natural() {
            let range = iterable
                .try_cast::<crate::ExclusiveRange>()
                .expect("the type id says it is an exclusive range");
            self.iterators.push(Iteration {
                items: Items::IntRange {
                    next: range.start,
                    end: range.end,
                },
                count: -1,
            });
            return Ok(());
        }

        let func = self
            .engine
            .global_modules
            .iter()
            .find_map(|module| module.get_iter(type_id));

        // Imported and sub-modules can register iterators too, but neither
        // exists to be searched under `no_module`.
        #[cfg(not(feature = "no_module"))]
        let func = func.or_else(|| self.global.get_iter(type_id)).or_else(|| {
            self.engine
                .global_sub_modules
                .values()
                .find_map(|module| module.get_qualified_iter(type_id))
        });

        let func = or_raise!(func, Box::new(EvalAltResult::ErrorFor(pos)));

        self.iterators.push(Iteration {
            items: Items::Boxed(func(iterable)),
            count: -1,
        });
        Ok(())
    }

    /// Does the exclusive integer range still iterate the way its own
    /// [`Iterator`] does?
    ///
    /// The same three places [`Vm::iter_init`] searches, in the same order,
    /// stopping at the first that has an entry rather than the first that has a
    /// natural one — a registration shadowing the built-in one is exactly the
    /// case this has to say no to.
    fn int_range_is_natural(&self) -> bool {
        let type_id = TypeId::of::<crate::ExclusiveRange>();

        let natural = self
            .engine
            .global_modules
            .iter()
            .find_map(|module| module.iter_is_natural(type_id));

        #[cfg(not(feature = "no_module"))]
        let natural = natural
            .or_else(|| self.global.iter_is_natural(type_id))
            .or_else(|| {
                self.engine
                    .global_sub_modules
                    .values()
                    .find_map(|module| module.qualified_iter_is_natural(type_id))
            });

        natural == Some(true)
    }

    /// Called only by [`call_syntactic_or_stacked`] after it has checked for a
    /// syntactic call.
    ///
    /// Some syntactic calls can be self-implemented or short-circuited.
    fn call_syntactic(
        &mut self,
        #[cfg_attr(feature = "no_function", allow(unused))] program: &Program,
        name: &str,
        argc: usize,
        first: usize,
        scope: &mut Scope,
        pos: Position,
    ) -> Result<Option<Dynamic>, Box<EvalAltResult>> {
        match name {
            crate::engine::KEYWORD_IS_DEF_VAR => {
                if argc != 1 {
                    return Err(EvalAltResult::ErrorFunctionNotFound(name.to_string(), pos).into());
                }
                let var_name = self.stack[first].as_immutable_string_ref().map_err(|typ| {
                    self.engine
                        .make_type_mismatch_err::<ImmutableString>(typ, pos)
                })?;
                return Ok(Some(scope.contains(&var_name).into()));
            }
            #[cfg(not(feature = "no_function"))]
            crate::engine::KEYWORD_IS_DEF_FN => {
                let (this_type, fn_name, arity) = match argc {
                    2 => {
                        let var_name = self.stack[first]
                            .as_immutable_string_ref()
                            .as_deref()
                            .cloned()
                            .map_err(|typ| {
                                self.engine
                                    .make_type_mismatch_err::<ImmutableString>(typ, pos)
                            })?;
                        let arity = self.stack[first + 1]
                            .as_int()
                            .map_err(|typ| self.engine.make_type_mismatch_err::<INT>(typ, pos))?;
                        (None, var_name, arity as usize)
                    }
                    3 => {
                        let this_type = self.stack[first]
                            .as_immutable_string_ref()
                            .as_deref()
                            .cloned()
                            .map_err(|typ| {
                                self.engine
                                    .make_type_mismatch_err::<ImmutableString>(typ, pos)
                            })?;
                        let var_name = self.stack[first + 1]
                            .as_immutable_string_ref()
                            .as_deref()
                            .cloned()
                            .map_err(|typ| {
                                self.engine
                                    .make_type_mismatch_err::<ImmutableString>(typ, pos)
                            })?;
                        let arity = self.stack[first + 2]
                            .as_int()
                            .map_err(|typ| self.engine.make_type_mismatch_err::<INT>(typ, pos))?;
                        (Some(this_type), var_name, arity as usize)
                    }
                    _ => {
                        return Err(
                            EvalAltResult::ErrorFunctionNotFound(name.to_string(), pos).into()
                        )
                    }
                };

                // Check if there is a compiled function.
                for f in program.functions() {
                    let local_name = or_raise!(
                        program.name(f.name),
                        malformed(format!("no name {}", f.name))
                    );

                    if local_name == fn_name.as_str() && f.params.len() == arity {
                        if let Some(ref this_type) = this_type {
                            if let Some(local_this_type_index) = f.this_type {
                                let local_this_type_name =
                                    or_raise!(program.name(local_this_type_index), {
                                        malformed(format!("no name {local_this_type_index}"))
                                    });

                                if local_this_type_name == this_type {
                                    return Ok(Some(Dynamic::TRUE));
                                }
                            }
                        } else if f.this_type.is_none() {
                            return Ok(Some(Dynamic::TRUE));
                        }
                    }
                }

                // Call into Rhai.
                let top = self.depth;
                let mut args = self.stack[first..top]
                    .iter_mut()
                    .map(operand_mut)
                    .collect::<FnArgsVec<_>>();

                let answer = self
                    .engine
                    .exec_syntactic_fn_call(
                        &mut self.global,
                        &mut self.caches,
                        name,
                        &mut args,
                        pos,
                    )
                    .map_err(|err| dispatch_failure(err, pos))?;
                return Ok(Some(or_raise!(
                    answer,
                    EvalAltResult::ErrorFunctionNotFound(name.to_string(), pos).into()
                )));
            }
            _ => {}
        }

        Ok(None)
    }

    /// Called only by [`call_syntactic_or_stacked`] after it has checked for a
    /// syntactic call.
    ///
    /// Call function with `argc` arguments sitting contiguously from `first` up.
    ///
    /// A function this compiler lowered is called directly, with no hash and no
    /// module walk: the call site's name index and the function's come from the
    /// same pool, so equal names have equal indices. Everything else goes to
    /// Rhai's dispatch, and resolves exactly as it would in the walker.
    fn call_stacked(
        &mut self,
        program: &Program,
        name_index: u32,
        name: &str,
        argc: usize,
        first: usize,
        scope: &mut Scope,
        pos: Position,
    ) -> VmResult {
        // Run compiled function if available.
        if let Some(function) = program.function(name_index, argc) {
            return self.call_compiled(
                program,
                name,
                &function.param_names,
                function.chunk,
                first,
                scope,
                pos,
            );
        }

        self.call_dispatched(name_index, name, first, scope, pos)
    }

    /// The half of [`Vm::call_stacked`] that is Rhai's: a call this compiler
    /// did not lower, resolved and run by the engine's own dispatch.
    ///
    /// Split out because the scope a call needs is decided by which half it
    /// lands in, and a caller that already knows can go straight to one.
    fn call_dispatched(
        &mut self,
        name_index: u32,
        name: &str,
        first: usize,
        scope: &mut Scope,
        pos: Position,
    ) -> VmResult {
        // Arguments are already contiguous at the top of the operand stack,
        // which is exactly the shape Rhai's ABI wants (`func/call.rs:36`). It
        // consumes them, replacing each with unit, so the caller truncates
        // afterwards rather than reusing them.
        let top = self.depth;
        let mut args: FnArgsVec<&mut Dynamic> =
            self.stack[first..top].iter_mut().map(operand_mut).collect();

        // What this site resolved to when it last ran, which is what Rhai has
        // to work out again for a site with nothing to read back. See
        // [`CallMemo`].
        let memo = call_site(
            &mut self.call_memo,
            self.engine,
            &self.global,
            &mut self.caches,
            self.generation,
            name_index,
            name,
            &mut args,
        );
        let site = memo.and_then(CallMemo::site);

        call_engine(
            self.engine,
            &mut self.global,
            &mut self.caches,
            scope,
            name,
            &mut args,
            false,
            false,
            pos,
            site.as_ref(),
        )
    }

    /// An empty `Scope` for a call that starts one of its own.
    ///
    /// A call binds its parameters into a scope that holds nothing else, and
    /// the first push into an empty one reserves `MIN_SCOPE_ENTRIES` on both
    /// of its arrays (`types/scope.rs:414-420`) — two allocations per call,
    /// and two frees when it drops. A scope lent back instead comes with those
    /// arrays still at capacity and only their lengths taken to zero, so the
    /// reserve finds the room already there and asks the allocator for
    /// nothing.
    ///
    /// Nothing here is `Scope`'s to know: it is lent by the `Vm`, to callers
    /// that were building one anyway.
    fn take_scope(&mut self) -> Scope<'static> {
        self.scopes.pop().unwrap_or_default()
    }

    /// Take a finished call's scope back, for [`Vm::take_scope`] to lend again.
    ///
    /// Emptied first, and unconditionally: the scope belongs to the call that
    /// has just ended, so whatever is still in it is exactly what dropping it
    /// would have discarded — including anything Rhai's own dispatch left
    /// behind on the way through (`func/call.rs:801-806`).
    ///
    /// That is also what makes lending safe where a closure captured out of the
    /// scope. Capturing shares the *value*: the entry is turned into a cell and
    /// the closure holds a clone of the cell, so clearing drops this end of the
    /// share and leaves the closure's alone. No later call can be handed a cell
    /// an earlier closure holds, because what is lent is the array and never an
    /// entry. Nor can anything still point into the array — the scope is moved
    /// in here, so every borrow of it has ended.
    fn give_scope(&mut self, mut scope: Scope<'static>) {
        scope.clear();
        self.scopes.push(scope);
    }

    /// This is the main entry-point for function calls.
    ///
    /// First check whether the call is a syntactic one (e.g. `is_def_fn`)
    /// which are self-implemented or directly called into the
    /// corresponding Rhai function.
    ///
    /// If the call is not to a syntactic one, it calls the function
    /// normally, with arguments pushed onto the stack.
    fn call_syntactic_or_stacked(
        &mut self,
        program: &Program,
        name_index: u32,
        name: &str,
        argc: usize,
        first: usize,
        scope: &mut Scope,
        capture: bool,
        pos: Position,
    ) -> VmResult {
        // Check if it is a built-in syntactic function.
        if let Some(value) = self.call_syntactic(program, name, argc, first, scope, pos)? {
            return Ok(value);
        }

        // A capturing call runs in the parent's scope, which is not this VM's
        // to lend or to take back.
        if capture {
            return self.call_stacked(program, name_index, name, argc, first, scope, pos);
        }

        // Only a compiled function binds anything into the scope it is given.
        // Rhai's dispatch fills one only for a function it evaluates itself,
        // and a native never touches it at all — so the lending is worth its
        // bookkeeping on one of the two halves and not on the other.
        //
        // Written as its own arm rather than as a scope chosen between the two:
        // a lent scope is a `Scope<'static>`, and `&mut Scope<'static>` is not
        // a `&mut Scope<'_>`, since `&mut` is invariant in the lifetime the
        // caller's scope carries.
        let Some(function) = program.function(name_index, argc) else {
            return self.call_dispatched(name_index, name, first, &mut Scope::new(), pos);
        };
        let mut detached = self.take_scope();
        let result = self.call_compiled(
            program,
            name,
            &function.param_names,
            function.chunk,
            first,
            &mut detached,
            pos,
        );
        self.give_scope(detached);
        result
    }

    /// The same call, with a variable as its first argument and Rhai's
    /// method-call rewrite applied to it (`func/call.rs:1434-1460`).
    ///
    /// The other arguments are already on the operand stack and were evaluated
    /// before the receiver was reached, which is the order Rhai uses and is
    /// observable whenever one of them writes to the receiver.
    #[allow(clippy::too_many_arguments)]
    fn call_by_reference(
        &mut self,
        program: &Program,
        name_index: u32,
        name: &str,
        argc: usize,
        receiver: Receiver,
        scope: &mut Scope,
        base: usize,
        capture: bool,
        pos: Position,
    ) -> VmResult {
        // Every argument count here includes the receiver, so zero of them
        // names no receiver at all and the instruction is nonsense. Only an
        // artifact can say it; the compiler emits one of these for a call that
        // has a first argument.
        if argc == 0 {
            return Err(malformed(
                "a call by reference with no receiver".to_string(),
            ));
        }

        // The register is not a scope entry, so it takes a path of its own
        // rather than a third [`Site`].
        if let Receiver::This = receiver {
            return self.call_by_this(program, name_index, name, argc, scope, capture, pos);
        }

        // A named receiver's value is already argument zero — [`Op::LoadNamed`]
        // put it there. A local's is not on the stack at all.
        let (at, on_stack) = match receiver {
            Receiver::Local(slot) => {
                let index = base + slot as usize;
                if index >= scope.len() {
                    return Err(malformed(format!("local slot {slot} is out of scope")));
                }
                (Site::Slot(index), argc - 1)
            }
            Receiver::Named(var) => {
                let name = or_raise!(program.name(var), malformed(format!("no name {var}")));
                (Site::Name(name), argc)
            }
            Receiver::This => unreachable!("taken above"),
        };
        let first = or_raise!(
            self.depth.checked_sub(on_stack),
            malformed("call with too few arguments".to_string())
        );

        let place = match at {
            Site::Slot(index) => Some(scope.get_mut_by_index(index)),
            // A resolver's answer shadows the scope, and `load_named` marks one
            // read-only precisely because it is a value and not a place. Asking
            // the resolver again to find that out would run it twice, which a
            // host can see.
            Site::Name(..) if self.stack[first].is_read_only() => None,
            Site::Name(name) => scope.get_mut(name),
        };

        // Three things rule out a reference, and Rhai rules out the same three:
        // it hands one out for neither a shared cell nor a constant
        // (`func/call.rs:1449-1454`), and a function this compiler lowered
        // copies its first argument whatever it is handed, exactly as Rhai
        // copies it before running a script function (`func/call.rs:661`).
        let by_reference = place.map_or(false, |value| !is_shared!(value) && !value.is_read_only())
            && program.function(name_index, argc).is_none();

        // All three want the ordinary shape, with every argument on the stack.
        if !by_reference {
            // A local's value has not been pushed. A name's already is: it is
            // what carried the lookup's position (see [`Receiver::Named`]), and
            // it is exactly the value Rhai would pass.
            if let Site::Slot(index) = at {
                let value = flatten_clone_value(scope.get_mut_by_index(index));
                self.open_slots(first, 1);
                set_operand(&mut self.stack[first], value);
            }
            let value = self.call_syntactic_or_stacked(
                program, name_index, name, argc, first, scope, capture, pos,
            )?;
            self.truncate_stack(first);
            return Ok(value);
        }

        let value = {
            let top = self.depth;
            let (entry, rest) = match at {
                Site::Slot(index) => (scope.get_mut_by_index(index), first),
                // Argument zero is dead weight now that there is an entry to
                // reach, and it is the price of having resolved the name where
                // its position was.
                Site::Name(name) => (
                    or_raise!(
                        scope.get_mut(name),
                        malformed(format!("`{name}` stopped being writable"))
                    ),
                    first + 1,
                ),
            };
            let mut args: FnArgsVec<&mut Dynamic> = core::iter::once(entry)
                .chain(self.stack[rest..top].iter_mut().map(operand_mut))
                .collect();
            // Keyed the same way [`Vm::call_stacked`] keys it, because it is
            // the same call: passing the receiver by reference is how Rhai
            // reaches a `&mut` first argument and is not a different lookup.
            let memo = call_site(
                &mut self.call_memo,
                self.engine,
                &self.global,
                &mut self.caches,
                self.generation,
                name_index,
                name,
                &mut args,
            );
            let site = memo.and_then(CallMemo::site);
            // The scope a dispatched script function runs in, which is never
            // this frame's — see [`Vm::call_stacked`], which has to build one
            // for the same reason and cannot borrow this one because the
            // receiver is holding it.
            call_engine(
                self.engine,
                &mut self.global,
                &mut self.caches,
                &mut Scope::new(),
                name,
                &mut args,
                true,
                false,
                pos,
                site.as_ref(),
            )
        };

        self.truncate_stack(first);
        value
    }

    /// The same again, with `this` as the first argument.
    ///
    /// [`Op::LoadThis`] has already pushed a flattened snapshot as argument
    /// zero — *before* the other arguments, unlike either of the other two
    /// receivers, because the path a shared or unbound receiver takes reads
    /// `this` first (`func/call.rs:1462`) where the by-reference path takes a
    /// pointer to it afterwards (`:1417`). Reading first is what makes an
    /// unbound `f(this, no_such)` report `ErrorUnboundThis` rather than the
    /// argument's failure.
    ///
    /// The snapshot is what gets passed when the register cannot be lent out,
    /// and dead weight when it can — the trade [`Receiver::Named`] makes too.
    fn call_by_this(
        &mut self,
        program: &Program,
        name_index: u32,
        name: &str,
        argc: usize,
        scope: &mut Scope,
        capture: bool,
        pos: Position,
    ) -> VmResult {
        let first = or_raise!(
            self.depth.checked_sub(argc),
            malformed("call with too few arguments".to_string())
        );

        // Rhai turns `f(this, ..)` into `this.f(..)` for a receiver that is not
        // shared, and read-only is *not* part of that test — unlike the variable
        // arm, which copies a constant before deciding (`func/call.rs:1449`).
        // A function this compiler lowered copies its first argument whatever it
        // is handed, exactly as Rhai copies one before running a script function
        // (`func/call.rs:661`), so a compiled callee rules a reference out too.
        let by_reference = self.this.as_ref().map_or(false, |value| !is_shared!(value))
            && program.function(name_index, argc).is_none();

        if !by_reference {
            let value = self.call_syntactic_or_stacked(
                program, name_index, name, argc, first, scope, capture, pos,
            )?;
            self.truncate_stack(first);
            return Ok(value);
        }

        let value = {
            let top = self.depth;
            let entry = or_raise!(
                self.this.as_mut(),
                malformed("`this` stopped being bound".to_string())
            );
            // Argument zero is the snapshot, dead now that there is a register
            // to reach through.
            let mut args: FnArgsVec<&mut Dynamic> = core::iter::once(entry)
                .chain(self.stack[first + 1..top].iter_mut().map(operand_mut))
                .collect();
            let memo = call_site(
                &mut self.call_memo,
                self.engine,
                &self.global,
                &mut self.caches,
                self.generation,
                name_index,
                name,
                &mut args,
            );
            let site = memo.and_then(CallMemo::site);
            call_engine(
                self.engine,
                &mut self.global,
                &mut self.caches,
                &mut Scope::new(),
                name,
                &mut args,
                true,
                false,
                pos,
                site.as_ref(),
            )
        };

        self.truncate_stack(first);
        value
    }

    fn call_compiled(
        &mut self,
        program: &Program,
        name: &str,
        params: &[ImmutableString],
        chunk: Chunk,
        first: usize,
        scope: &mut Scope,
        pos: Position,
    ) -> VmResult {
        self.call_compiled_with_this(program, name, params, chunk, first, scope, true, pos, None)
            .0
    }

    /// The same, against a receiver the callee owns for the duration.
    ///
    /// Hands the receiver back however the call ended, so a body that mutated
    /// `this` and then raised still gives its binder something to write back —
    /// which is what Rhai's pointer into the caller's storage does for free.
    ///
    /// Every compiled call comes through here, and [`Vm::call_compiled`] is this
    /// with no receiver. That is what makes `this` per-call rather than
    /// inherited: an ordinary call installs `None` and gives the caller's back
    /// on the way out, so a callee can never read the receiver of the frame that
    /// called it (`func/call.rs:669`).
    ///
    /// Kept out of the dispatch loop. Inlined, it is enough extra code to change
    /// register allocation across every other instruction — measured as a
    /// uniform slowdown on benchmarks that call no functions at all.
    fn call_compiled_with_this(
        &mut self,
        program: &Program,
        name: &str,
        params: &[ImmutableString],
        chunk: Chunk,
        first: usize,
        scope: &mut Scope,
        rewind_scope: bool,
        pos: Position,
        this: Option<Dynamic>,
    ) -> (VmResult, Option<Dynamic>) {
        let saved = mem::replace(&mut self.this, this);

        let result = match self.engine.track_operation(&mut self.global, pos) {
            Ok(()) => {
                self.global.level += 1;
                let result = self.call_compiled_body(
                    program,
                    name,
                    params,
                    chunk,
                    first,
                    scope,
                    rewind_scope,
                    pos,
                );
                self.global.level -= 1;
                result
            }
            Err(err) => Err(err),
        };

        (result, mem::replace(&mut self.this, saved))
    }

    fn call_compiled_body(
        &mut self,
        program: &Program,
        name: &str,
        params: &[ImmutableString],
        chunk: Chunk,
        first: usize,
        scope: &mut Scope,
        rewind_scope: bool,
        pos: Position,
    ) -> VmResult {
        #[cfg(not(feature = "unchecked"))]
        {
            // The limit exists only where recursion does — `no_function` leaves
            // nothing to call, so Rhai drops the setting with the functions.
            #[cfg(not(feature = "no_function"))]
            if self.global.level > self.engine.max_call_levels() {
                return Err(Box::new(EvalAltResult::ErrorStackOverflow(pos)));
            }
            if params.len() > self.engine.max_variables() {
                return Err(Box::new(EvalAltResult::ErrorTooManyVariables(pos)));
            }
        }

        let scope_start_len = scope.len();

        #[cfg(feature = "debugging")]
        let orig_call_stack_len = self
            .global
            .debugger
            .as_ref()
            .map_or(0, |dbg| dbg.call_stack().len());

        for (param, slot) in params.iter().zip(first..) {
            // Taken, not cloned — Rhai consumes the caller's argument slots
            // (`func/script.rs:75`), and the caller truncates them away after.
            let value = or_raise!(
                self.values_mut().get_mut(slot),
                malformed("call with too few arguments".to_string())
            )
            .take();
            scope.push_entry(param.clone(), value.access_mode(), value);
        }
        let scope_end_len = scope.len();

        // A frame for `back_trace` to see, pushed once the arguments are in the
        // scope so it reports the values the body will run with — the moment
        // Rhai picks (`func/script.rs:78`).
        #[cfg(feature = "debugging")]
        if self.engine.is_debugger_registered() {
            let args = scope
                .iter_inner()
                .skip(scope_start_len)
                .map(|(.., v)| v.flatten_clone());
            let source = self.global.source.clone();

            self.global
                .debugger_mut()
                .push_call_stack_frame(name.into(), args, source, pos);
        }

        // A function's parameters are its first locals, sitting at 0 upwards in
        // a scope that holds nothing else — so slot 0 is index 0.
        let mut reached = chunk.entry() as usize;
        let outcome = self.execute(program, scope, chunk, scope_start_len, &mut reached);
        // Read before the mapping below, which turns the `Return` that carries a
        // body's value into a success.
        let failed = outcome.is_err();

        let result = outcome.or_else(|err| match *err {
            // A `return` inside the body is the body's value.
            EvalAltResult::Return(value, ..) => Ok(value),
            // Exits and system errors pass straight through, positioned at the
            // call rather than at whatever raised them.
            EvalAltResult::Exit(..) => Err(reposition(err, pos)),
            _ if err.is_system_exception() => Err(reposition(err, pos)),
            // Everything else is attributed to the call.
            _ => Err(Box::new(EvalAltResult::ErrorInFunctionCall(
                name.to_string(),
                self.global.source().unwrap_or("").to_string(),
                err,
                pos,
            ))),
        });

        // Mapped first, then reported: Rhai tells the debugger what the *caller*
        // will see (`func/script.rs:157-186`), with the body's locals still in
        // scope for the callback to read.
        #[cfg(feature = "debugging")]
        let result = self.at_function_exit(scope, result, orig_call_stack_len, pos);

        // Rewind scope.
        if rewind_scope {
            scope.rewind(scope_start_len);
        } else if scope_end_len != scope_start_len {
            // Remove arguments only, leaving new variables in the scope
            scope.remove_range(scope_start_len, scope_end_len - scope_start_len);
        }

        if failed {
            self.record_fault(reached);
        }

        result
    }

    /// Report how a body ended and drop its call stack frame.
    ///
    /// The frame goes however the call ended — an error escaping one must not
    /// leave the stack deep, or every later trace is wrong.
    #[cfg(feature = "debugging")]
    fn at_function_exit(
        &mut self,
        scope: &mut Scope,
        result: VmResult,
        orig_call_stack_len: usize,
        pos: Position,
    ) -> VmResult {
        if !self.engine.is_debugger_registered() {
            return result;
        }

        // Only where something is waiting for it: a `FunctionExit` asked for at
        // this level or outside it, or a step that has to stop somewhere and
        // this is where the body ran out (`func/script.rs:159-163`).
        let trigger = match self.global.debugger().status {
            crate::eval::DebuggerStatus::FunctionExit(n) => n >= self.global.level,
            crate::eval::DebuggerStatus::Next(.., true) => true,
            _ => false,
        };

        let result = if trigger {
            // The call site, where Rhai has the body's closing brace: a chunk
            // keeps no position for the end of itself, and the call is the place
            // Rhai falls back to when the body has none either.
            let node = crate::ast::Stmt::Noop(pos);
            let event = match result {
                Ok(ref value) => crate::eval::DebuggerEvent::FunctionExitWithValue(value),
                Err(ref err) => crate::eval::DebuggerEvent::FunctionExitWithError(err),
            };

            let before = scope.len();
            let reported = self.engine.dbg_raw(
                &mut self.global,
                &mut self.caches,
                scope,
                self.this.as_mut(),
                (&node).into(),
                event,
            );
            rewind_after_stop(scope, before);

            match reported {
                Ok(..) => result,
                Err(err) => Err(err),
            }
        } else {
            result
        };

        if let Some(dbg) = self.global.debugger.as_mut() {
            dbg.rewind_call_stack(orig_call_stack_len);
        }

        result
    }

    /// Stop at a statement boundary, as Rhai stops at a statement node
    /// (`eval/stmt.rs:269`).
    ///
    /// `depth` is the marker's, and says which of the steps waiting in
    /// [`Vm::pending_steps`] belong to statements that have now ended.
    #[cfg(feature = "debugging")]
    fn at_statement(
        &mut self,
        scope: &mut Scope,
        depth: u16,
        pos: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        if !self.engine.is_debugger_registered() {
            return Ok(());
        }

        let level = self.global.level;

        // A marker inside the statement a step was asked at is one that step
        // must not stop for — that is what stepping *over* a call or a block
        // means. Anything at the same depth or outside it, or in a frame further
        // out, is past the end of that statement.
        while let Some(&(at_level, at_depth, status)) = self.pending_steps.last() {
            if level > at_level || (level == at_level && depth > at_depth) {
                break;
            }
            self.pending_steps.pop();
            self.global.debugger_mut().reset_status(status);
        }

        let node = crate::ast::Stmt::Noop(pos);
        let before = scope.len();
        let resumed = self.engine.dbg_reset(
            &mut self.global,
            &mut self.caches,
            scope,
            self.this.as_mut(),
            &node,
        );
        rewind_after_stop(scope, before);

        if let Some(status) = resumed? {
            self.pending_steps.push((level, depth, status));
        }

        Ok(())
    }

    /// Tell the debugger the run is over, as Rhai does at the end of an `eval`
    /// (`api/eval.rs:255-260`).
    ///
    /// Only where the run got there: the walker's `?` on the statements means a
    /// failed script never reaches this.
    #[cfg(feature = "debugging")]
    fn at_end(&mut self, scope: &mut Scope) -> Result<(), Box<EvalAltResult>> {
        if !self.engine.is_debugger_registered() {
            return Ok(());
        }

        self.global.debugger_mut().status = crate::eval::DebuggerStatus::Terminate;
        let node = crate::ast::Stmt::Noop(Position::NONE);

        self.engine
            .dbg(&mut self.global, &mut self.caches, scope, None, &node)
    }

    /// Run one frame, cleaning up after it however it leaves.
    ///
    /// Whatever the frame's loops are holding goes when the frame does — a
    /// `return` out of a `for`, or an error escaping one, both skip the
    /// `IterNext` that would have dropped the iterator. Doing it here rather
    /// than at each exit means there is one place to be right.
    fn execute(
        &mut self,
        program: &Program,
        scope: &mut Scope,
        chunk: Chunk,
        base: usize,
        reached: &mut usize,
    ) -> VmResult {
        let iter_base = self.iterators.len();
        let handler_base = self.handlers.len();
        let size_base = self.sizes.len();
        // Each frame's floor is its own. A checkpoint inside a function this
        // one calls must not become what this one unwinds to.
        let outer_floor = mem::replace(&mut self.unwind_floor, base);
        // The same for what this frame's memo entries are stamped with, except
        // that the stamp is new rather than saved: a frame's entries must not
        // be readable from the one it returns to. See [`Vm::generation`].
        self.last_generation = self.last_generation.wrapping_add(1);
        let outer_generation = mem::replace(&mut self.generation, self.last_generation);

        // One red frame per invocation, including non-portal callees inlined by
        // the meta-tracer.  Its address remains stable until this execute call
        // has completed or unwound.
        let mut frame = GrainFrame {
            scope,
            base,
            reached: chunk.entry() as usize,
            stack_base: self.depth,
            jit_resume_pc_plus_one: 0,
        };

        // The dispatch loop uses `?` throughout, so an error leaves it rather
        // than being examined inside it. Catching therefore happens out here:
        // the loop stops, a handler this frame armed gets the error, and the
        // loop restarts at the catch block. `run_frame` keeps `pc` in a
        // register and the fault address arrives through `reached`, which is
        // written every instruction anyway, so none of this costs the common
        // path anything.
        let mut start = chunk.entry() as usize;
        let result = loop {
            match self.run_frame(program, &mut frame, start) {
                Ok(value) => break Ok(value),
                Err(err) => match self.catch(program, err, handler_base, frame.scope) {
                    // Metered like a backward jump, and for the same reason:
                    // a catch block that sits before the throw is a cycle the
                    // dispatch loop never sees as one, because control got
                    // there through the error path rather than through a jump.
                    Ok(resume) => {
                        // Handled, so the frames it unwound past are not where
                        // this run failed. Left behind, they would head the
                        // next error's trace.
                        self.clear_faults();
                        self.engine
                            .track_operation(&mut self.global, program.position(resume))?;
                        start = resume;
                    }
                    Err(err) => break Err(err),
                },
            }
        };

        self.iterators.truncate(iter_base);
        self.handlers.truncate(handler_base);
        self.sizes.truncate(size_base);

        if result.is_err() {
            self.unwind_after_error(frame.scope);
        }
        *reached = frame.reached;
        self.unwind_floor = outer_floor;
        self.generation = outer_generation;
        result
    }

    /// Rhai's `Engine::make_type_mismatch_err` (`api/formatting.rs:246`).
    ///
    /// The asymmetry is Rhai's and is easy to get wrong in either direction:
    /// the *expected* type goes through the engine's registered names and the
    /// *actual* one does not. So `if 0..1 {}` reports
    /// `core::ops::range::Range<i64>` rather than the `range` the same engine
    /// would print anywhere else. Mapping both — which reads like the obvious
    /// thing — makes every one of these differ from the walker.
    fn mismatch<T>(&self, actual: &str, pos: Position) -> Box<EvalAltResult> {
        Box::new(EvalAltResult::ErrorMismatchDataType(
            self.engine
                .map_type_name(core::any::type_name::<T>())
                .into(),
            actual.into(),
            pos,
        ))
    }

    /// Whether a guard holds, for an operator that is also the branch reading
    /// its result.
    ///
    /// Rhai requires a boolean guard and reports the mismatch at the guard's
    /// own position (`eval/stmt.rs:487-490`), which is the fused instruction's
    /// -- the operator that computed the guard *is* the guard expression.
    ///
    /// Borrowed rather than taken: reading a guard is `Dynamic::as_bool`, which
    /// reads through a reference, so the caller keeps the value and can release
    /// it itself rather than leaving it to the drop a move would have implied.
    ///
    /// The position arrives as a closure for the reason
    /// [`Engine::track_operation_at`](crate::Engine) takes one: it is a table
    /// lookup keyed on the address, and a guard that holds — which is every
    /// turn of every loop but the last — never reports one.
    fn guard_holds(
        &self,
        value: &Dynamic,
        pos: impl FnOnce() -> Position,
    ) -> Result<bool, Box<EvalAltResult>> {
        value
            .as_bool()
            .map_err(|actual| self.mismatch::<bool>(actual, pos()))
    }

    /// What reading a key a map does not have produces.
    ///
    /// Unit, unless the host asked for the strict reading — which is a whole
    /// engine option (`fail_on_invalid_map_property`) rather than anything the
    /// script says, so it has to be consulted rather than assumed.
    #[cfg(not(feature = "no_object"))]
    fn absent_key(&self, key: &str, pos: Position) -> VmResult {
        if self.engine.fail_on_invalid_map_property() {
            Err(Box::new(EvalAltResult::ErrorPropertyNotFound(
                key.to_string(),
                pos,
            )))
        } else {
            Ok(Dynamic::UNIT)
        }
    }

    /// Fold the element on top of the stack into the literal's running total,
    /// and refuse it if that puts the literal over a configured limit.
    ///
    /// Reproduces `eval/expr.rs:318-329` for an array and `:349-359` for a map.
    /// The two differ in one place — an array element adds one to the array
    /// count, a map entry adds one to the map count — and in nothing else, so
    /// they share this.
    ///
    /// Worth being exact about what is *not* counted: a map's total starts at
    /// zero and only the entries with computed values are added to it, because
    /// Rhai's loop runs over those alone and the constant ones are already
    /// sitting in the template. A literal that is entirely constant never
    /// reaches here at all — the optimizer folded it long before.
    ///
    /// Out of line: it is a handful of instructions in the common case and a
    /// call to Rhai in the rare one, and the dispatch loop is measurably
    /// sensitive to what shares its registers.
    fn check_size(
        &mut self,
        index: u16,
        map: bool,
        pos: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        if index == 0 {
            self.sizes.push((0, 0, 0));
        }

        // `unchecked` removes every limit this could reject against, so the
        // running total is never read and measuring it is pure cost. The push
        // above still happens: the stack is frame-floored either way, and the
        // instruction that drops it does not know which build it is in.
        //
        // Losing both `no_index` and `no_object` is the other way to get here:
        // with no literal to build there is nothing to measure, and the
        // measurement itself goes with the containers.
        #[cfg(any(
            feature = "unchecked",
            all(feature = "no_index", feature = "no_object")
        ))]
        {
            let _ = (map, pos);
            Ok(())
        }

        #[cfg(not(any(
            feature = "unchecked",
            all(feature = "no_index", feature = "no_object")
        )))]
        {
            // Rhai skips the whole measurement when no limit could reject it,
            // and measuring is a walk of the value — so this is the difference
            // between free and proportional to what the literal holds.
            if self.engine.max_string_size() == 0
                && self.engine.max_array_size() == 0
                && self.engine.max_map_size() == 0
            {
                return Ok(());
            }

            let value = or_raise!(
                self.values().last(),
                malformed("size check with no element".to_string())
            );
            let delta = calc_data_sizes(value, true);

            let total = or_raise!(
                self.sizes.last_mut(),
                malformed("size check outside a literal".to_string())
            );
            *total = (
                total.0 + delta.0 + usize::from(!map),
                total.1 + delta.1 + usize::from(map),
                total.2 + delta.2,
            );

            self.engine
                .throw_on_size(*total)
                .map_err(|err| dispatch_failure(err, pos))
        }
    }

    /// Drop what an escaping error skipped the unwind for.
    ///
    /// An error leaves a block by jumping over the [`Op::UnwindTo`] that would
    /// have dropped what it declared, so those locals are still in the scope.
    /// Rhai rewinds a block whether it is left normally or by a throw, and
    /// rewinds nothing at a chunk's top level — which is what the floor is: the
    /// last top-level statement boundary. Anything above it belongs to a block
    /// that did not get to finish.
    ///
    /// Guarded rather than unconditional because [`Op::Return`] has already
    /// unwound to `base`, which is below the floor.
    ///
    /// Out of line for the reason [`Vm::catch`] is: it sits on the error edge
    /// of the frame, where nothing is hot and everything competes with the
    /// dispatch loop for the same registers.
    #[cold]
    fn unwind_after_error(&self, scope: &mut Scope) {
        if scope.len() > self.unwind_floor {
            scope.rewind(self.unwind_floor);
        }
    }

    /// Hand an error to the innermost handler this frame armed, if any.
    ///
    /// `Ok` is the address the catch block starts at. `Err` means nothing here
    /// wanted it and it should keep going up.
    ///
    /// Kept out of line for the reason [`Vm::call_compiled`] is: it sits on
    /// the dispatch loop's error edge, and letting it inline there costs every
    /// instruction that never fails.
    #[cold]
    fn catch(
        &mut self,
        program: &Program,
        err: Box<EvalAltResult>,
        handler_base: usize,
        scope: &mut Scope,
    ) -> Result<usize, Box<EvalAltResult>> {
        // Only handlers this frame armed. A callee must never resume into its
        // caller's catch block — that is a jump into another chunk, which the
        // verifier forbids and nothing would catch at run time. The callee's
        // error propagates normally instead, and `ErrorInFunctionCall` is
        // catchable, so the caller's own frame still sees it.
        // Walking outwards, because leaving one region can land the error in
        // the next: `try { try { throw 1 } catch { throw; } } catch (e) { .. }`
        // re-raises from the inner catch and the outer `try` still has to see
        // it.
        let mut err = err;
        let handler = loop {
            if self.handlers.len() <= handler_base {
                return Err(err);
            }
            let handler = self.handlers.last_mut().expect("checked");

            // Leaving a catch block rather than entering one. A bare `throw;`
            // there — an `ErrorRuntime` carrying unit — means "re-raise what
            // was caught, from here" (`eval/stmt.rs:866`).
            let Some(original) = handler.caught.take() else {
                break handler;
            };
            self.handlers.pop();
            let rethrown =
                matches!(&*err, EvalAltResult::ErrorRuntime(value, ..) if value.is_unit());
            if rethrown {
                let pos = err.position();
                err = original;
                err.set_position(pos);
            }
        };

        // `return`, `break`, `continue`, `exit` and the system exceptions
        // unwind as errors and are not exceptions a script may catch.
        if !err.is_catchable() {
            return Err(err);
        }

        let (target, catch_var) = (handler.target, handler.catch_var);
        let (operands, scope_len, iters) = (handler.operands, handler.scope_len, handler.iters);

        let mut err = err;
        let value = self.catch_value(&mut err, catch_var.is_some());

        // Back to where the `try` began, at all three depths.
        self.truncate_stack(operands);
        self.iterators.truncate(iters);
        scope.rewind(scope_len);

        if let Some(index) = catch_var {
            let name = or_raise!(program.name(index), malformed(format!("no name {index}")));
            #[cfg(not(feature = "unchecked"))]
            if scope.len() >= self.engine.max_variables() {
                return Err(Box::new(EvalAltResult::ErrorTooManyVariables(
                    program.position(target),
                )));
            }
            scope.push_dynamic(name, value);
        }

        self.handlers.last_mut().expect("checked").caught = Some(err);
        Ok(target)
    }

    /// What the catch variable is bound to (`eval/stmt.rs:809-845`).
    ///
    /// Three shapes: nothing at all without a variable, the raw thrown value
    /// for a `throw`, and a map of the error's parts for anything else. The
    /// unwrapping matters — a `throw` inside a called function arrives wrapped
    /// in `ErrorInFunctionCall`, and Rhai still binds the bare value.
    ///
    /// Under `no_object` there is no map to build one in, and Rhai binds the
    /// message alone (`eval/stmt.rs:815`).
    fn catch_value(&self, err: &mut Box<EvalAltResult>, wanted: bool) -> Dynamic {
        if !wanted {
            return Dynamic::UNIT;
        }
        if let EvalAltResult::ErrorRuntime(value, ..) = err.unwrap_inner() {
            return value.clone();
        }

        // Read *and cleared*, as Rhai does, so the message below carries no
        // trailing position and a re-raise starts from the catch site.
        #[cfg(feature = "no_object")]
        {
            let _ = err.take_position();
            err.to_string().into()
        }

        #[cfg(not(feature = "no_object"))]
        {
            let mut map = Map::new();
            let pos = err.take_position();

            map.insert("message".into(), err.to_string().into());
            if let Some(source) = &self.global.source {
                map.insert("source".into(), source.into());
            }
            if !pos.is_none() {
                let line = pos.line().unwrap_or(0) as INT;
                map.insert("line".into(), line.into());
                let column = pos.position().unwrap_or(0) as INT;
                map.insert("position".into(), column.into());
            }
            err.dump_fields(&mut map);
            map.into()
        }
    }

    /// The dispatch loop. `start` is the chunk's entry, or a catch block's
    /// address when [`Vm::execute`] resumes one after an error.
    ///
    /// Inlined into its one caller: splitting the loop out so errors could be
    /// caught outside it cost 1.55x to 1.40x on the tight-loop benchmark until
    /// this was here.
    #[inline(always)]
    fn run_frame(
        &mut self,
        program: &Program,
        frame: &mut GrainFrame<'_, '_>,
        start: usize,
    ) -> VmResult {
        self.reserve_stack(program.max_stack() as usize);

        // A residual's `Expr::Variable` nodes carry offsets Rhai's parser
        // computed against its own scope discipline, not against ours. Forcing
        // name lookup inside them costs a reverse scan but cannot be wrong.
        // Only programs that still have residuals pay it, which is the point of
        // driving the count to zero.
        if program.residual_count() > 0 {
            self.global.always_search_scope = true;
        }

        // The chunk's entry the first time round, a catch block's address when
        // resumed after one.
        let mut pc = start;

        // The marker receiver is a zero-sized declaration object, like
        // RPython's JitDriver.  Construct it in the frame prologue so a trace
        // that starts at the merge point never encounters Rust's synthetic
        // unit-struct constructor on the closing back edge.
        #[cfg(feature = "grain-jit")]
        let jit_driver = jit::GrainJitDriver;

        loop {
            // Nothing inside an iteration moves `pc` except a jump, and a jump
            // only happens after the instruction succeeded — so recording it
            // here names whichever instruction fails.
            frame.reached = pc;

            // Before anything is decoded, so the state the tracer records is
            // the state an instruction starts from and a loop that jumps back
            // to `pc` arrives here holding exactly what it held last time.
            #[cfg(feature = "grain-jit")]
            {
                frame.jit_resume_pc_plus_one = 0;
                jit_driver.jit_merge_point(pc, program.jit_identity(), program, frame, self);
                if frame.jit_resume_pc_plus_one != 0 {
                    pc = frame.jit_resume_pc_plus_one - 1;
                    continue;
                }
            }

            // Re-read every loop-carried value from the marker's green program
            // or red frame/VM after the marker.  Keeping one of these in a
            // prologue local makes the lowered body read an unseeded register
            // when tracing correctly starts at the merge point instead of
            // replaying the function entry.
            let base = frame.base;
            let stack_base = frame.stack_base;
            // What this frame's memo entries are stamped with. Nothing else
            // can carry it, so nothing else can read them back. See
            // [`Vm::generation`].
            let generation = self.generation;

            // Borrow the locals only after the marker has seen the whole
            // frame; the borrow ends at the iteration boundary before the next
            // marker.
            let scope = &mut *frame.scope;
            macro_rules! scope_len {
                () => {{
                    #[cfg(feature = "grain-jit")]
                    {
                        jit::scope_len(scope) as usize
                    }
                    #[cfg(not(feature = "grain-jit"))]
                    {
                        scope.len()
                    }
                }};
            }
            macro_rules! scope_entry {
                ($index:expr) => {{
                    #[cfg(feature = "grain-jit")]
                    {
                        jit::scope_entry(scope, $index)
                    }
                    #[cfg(not(feature = "grain-jit"))]
                    {
                        scope.get_mut_by_index($index)
                    }
                }};
            }
            macro_rules! truncate_stack {
                ($depth:expr) => {{
                    #[cfg(feature = "grain-jit")]
                    {
                        jit::truncate_stack(self, $depth);
                    }
                    #[cfg(not(feature = "grain-jit"))]
                    {
                        self.truncate_stack($depth);
                    }
                }};
            }

            // No check against the chunk's end. Verification proves execution
            // cannot leave it — every path reaches a `Return`, no jump goes
            // outside, nothing falls off — so a comparison here would cost
            // every instruction to restate something already established.
            // No address in the message: `reached` is written every iteration,
            // so the fault trace already names this instruction. Read it from
            // `Vm::fault_pc`, corrupt artifact or not.
            #[cfg(feature = "grain-jit")]
            let tag = match jit::code_byte(program, pc) {
                -1 => {
                    return Err(Box::new(EvalAltResult::ErrorRuntime(
                        "ran off the end of a chunk".into(),
                        Position::NONE,
                    )))
                }
                byte => byte as u8,
            };
            #[cfg(not(feature = "grain-jit"))]
            let tag = *or_raise!(program.code().get(pc), {
                Box::new(EvalAltResult::ErrorRuntime(
                    "ran off the end of a chunk".into(),
                    Position::NONE,
                ))
            });

            // Every instruction's operands sit at a fixed offset from its tag,
            // so dispatch is a match and a couple of loads with nothing decoded
            // and nothing allocated. The bounds checks are what let this run
            // straight off an artifact without trusting it; the verifier has
            // already made them unreachable for anything that loaded.
            #[cfg(feature = "grain-jit")]
            let width = match jit::code_width(program, pc) {
                -1 => return Err(malformed("undecodable instruction".to_string())),
                width => width as usize,
            };
            #[cfg(not(feature = "grain-jit"))]
            let width = or_raise!(
                code::width_of(tag, program.code(), pc),
                malformed("undecodable instruction".to_string())
            );
            // Macros rather than closures, for the reason the position lookup
            // below is one and one more. Both were rebuilt every iteration
            // because they capture `pc`, which moves; and a closure is opaque
            // to anything reading this loop from outside, which arrives at
            // `Fn::call` on an anonymous type it has no body for. The 45
            // operand reads in the dispatch below were 45 indirect calls, each
            // paying a second one for the lazy error.
            //
            // The bounds check stays. It is what lets this run straight off an
            // artifact without trusting it, and `mutated_artifacts_load_or_fail            //_but_never_misbehave` is the claim it carries.
            macro_rules! small {
                ($offset:expr) => {{
                    #[cfg(feature = "grain-jit")]
                    {
                        match jit::code_u16(program, pc + $offset) {
                            -1 => return Err(malformed("truncated operand".to_string())),
                            operand => operand as u16,
                        }
                    }
                    #[cfg(not(feature = "grain-jit"))]
                    {
                        match code::u16_at(program.code(), pc + $offset) {
                            Some(operand) => operand,
                            None => return Err(malformed("truncated operand".to_string())),
                        }
                    }
                }};
            }
            macro_rules! wide {
                ($offset:expr) => {{
                    #[cfg(feature = "grain-jit")]
                    {
                        match jit::code_u32(program, pc + $offset) {
                            -1 => return Err(malformed("truncated operand".to_string())),
                            operand => operand as u32,
                        }
                    }
                    #[cfg(not(feature = "grain-jit"))]
                    {
                        match code::u32_at(program.code(), pc + $offset) {
                            Some(operand) => operand,
                            None => return Err(malformed("truncated operand".to_string())),
                        }
                    }
                }};
            }

            // Instructions carry no position; the table does, keyed on the
            // address. A stripped program answers `NONE` for every one of
            // these, which is what a device runs — the address travels back
            // with the error instead, and the host resolves it.
            //
            // A macro rather than a value: most instructions never ask, and the
            // ones that do mostly ask only on the way to an error. A macro
            // rather than a closure because a closure whose address is taken
            // even once is materialised on the stack, and the dispatch loop was
            // paying four instructions per instruction to build its environment
            // whether or not the instruction ever asked for a position. The one
            // site that needs a callable builds one there.
            macro_rules! pos {
                () => {{
                    #[cfg(feature = "grain-jit")]
                    {
                        jit::code_position(program, pc)
                    }
                    #[cfg(not(feature = "grain-jit"))]
                    {
                        program.position(pc)
                    }
                }};
            }
            macro_rules! byte {
                ($offset:expr) => {{
                    #[cfg(feature = "grain-jit")]
                    {
                        match jit::code_byte(program, pc + $offset) {
                            -1 => return Err(malformed("truncated operand".to_string())),
                            operand => operand as u8,
                        }
                    }
                    #[cfg(not(feature = "grain-jit"))]
                    {
                        match program.code().get(pc + $offset) {
                            Some(operand) => *operand,
                            None => return Err(malformed("truncated operand".to_string())),
                        }
                    }
                }};
            }
            macro_rules! fast_operators {
                () => {{
                    #[cfg(feature = "grain-jit")]
                    {
                        jit::fast_operators(self) != 0
                    }
                    #[cfg(not(feature = "grain-jit"))]
                    {
                        self.engine.fast_operators()
                    }
                }};
            }

            // Whether an `INDEX_GET`/`INDEX_SET` instruction names its index in
            // a slot. Its own macro because the answer is wanted twice: once
            // to read the index and once to push it back for a walk that has
            // to find it on the stack.
            macro_rules! index_is_local {
                () => {{
                    matches!(
                        tag,
                        code::tag::INDEX_GET_FROM_LOCAL
                            | code::tag::INDEX_SET_FROM_LOCAL
                            | code::tag::INDEX_SET_FROM_LOCAL_VALUE_CONST
                            | code::tag::INDEX_SET_OP_FROM_LOCAL
                            | code::tag::INDEX_SET_OP_FROM_LOCAL_VALUE_CONST
                    )
                }};
            }

            // The index itself: the operand word at `$offset` for the naming
            // tags, or the stack entry `$under` for the plain ones, which name
            // it in neither place. `None` is the fast path declining, which
            // the walk below then answers.
            macro_rules! indexed_int {
                ($offset:expr, $under:expr) => {{
                    let mut found = None;
                    if index_is_local!() {
                        let at = base + small!($offset) as usize;
                        if at < scope_len!() {
                            if let Union::Int(i, ..) = scope_entry!(at).0 {
                                found = Some(i);
                            }
                        }
                    } else if matches!(
                        tag,
                        code::tag::INDEX_GET_FROM_CONST
                            | code::tag::INDEX_SET_FROM_CONST
                            | code::tag::INDEX_SET_FROM_CONST_VALUE_CONST
                            | code::tag::INDEX_SET_OP_FROM_CONST
                            | code::tag::INDEX_SET_OP_FROM_CONST_VALUE_CONST
                    ) {
                        let index = u32::from(small!($offset));
                        #[cfg(feature = "grain-jit")]
                        let constant = jit::program_constant(program, index);
                        #[cfg(not(feature = "grain-jit"))]
                        let constant = program.constant(index);
                        if let Some(Union::Int(i, ..)) = constant.map(|value| &value.0) {
                            found = Some(*i);
                        }
                    } else if let Some(under) = $under {
                        if let Union::Int(i, ..) = stack_ref(self, under).0 {
                            found = Some(i);
                        }
                    }
                    found
                }};
            }

            // A named operand as a value, for the walk that answers a declined
            // instruction: it reads its operands off the stack, and a named
            // operand has never been there. The tag is what says whether the
            // number names a slot or a constant, and which tags mean which
            // depends on the operand, so the caller decides that much.
            macro_rules! operand_value {
                ($offset:expr, $is_local:expr) => {{
                    if $is_local {
                        let slot = small!($offset);
                        let at = base + slot as usize;
                        if at >= scope_len!() {
                            return Err(malformed(format!("local slot {slot} is out of scope")));
                        }
                        flatten_clone_value(scope_entry!(at))
                    } else {
                        let index = u32::from(small!($offset));
                        #[cfg(feature = "grain-jit")]
                        let constant = jit::program_constant(program, index);
                        #[cfg(not(feature = "grain-jit"))]
                        let constant = program.constant(index);
                        clone_value(or_raise!(
                            constant,
                            malformed(format!("no constant {index}"))
                        ))
                    }
                }};
            }

            // The index a naming tag carries.
            macro_rules! indexed_value {
                ($offset:expr) => {{
                    operand_value!($offset, index_is_local!())
                }};
            }

            // The constant an assigning naming tag carries. The value is
            // written after the index and before the operator, so it sits at
            // seven however the form spells the other two.
            macro_rules! assigned_value {
                () => {{
                    operand_value!(7, false)
                }};
            }

            // Every transfer of control goes through this, and a backward one
            // is charged an operation.
            //
            // A cycle in a chunk always contains a backward edge, so this is
            // the whole of `max_operations` and the `on_progress` interrupt —
            // for the loops this compiler writes and for a chunk it did not.
            // A corrupt artifact carries no metering instruction and would
            // otherwise spin forever inside a loader that had already accepted
            // it. Found by `mutated_artifacts_load_or_fail_but_never_misbehave`,
            // whose whole claim is that this cannot happen.
            //
            // The back edge a loop is lowered with names the body's position,
            // so what a stopped loop reports is the place Rhai reports.
            //
            // A macro rather than four open-coded checks because the failure
            // mode of missing one is silent, and because it costs nothing on
            // the straight-line path: only a jump pays the comparison.
            macro_rules! transfer {
                ($target:expr) => {{
                    let target: usize = $target;
                    if target <= pc {
                        #[cfg(feature = "grain-jit")]
                        {
                            if let Some(error) = jit::track_operation_error(self, program, pc) {
                                return Err(error);
                            }
                            jit_driver.can_enter_jit(
                                target,
                                program.jit_identity(),
                                program,
                                frame,
                                self,
                            );
                        }
                        // The position is a table lookup keyed on the
                        // address, and metering only reports one when it
                        // stops the run — so it is asked for behind the
                        // check rather than in front of it.
                        #[cfg(not(feature = "grain-jit"))]
                        self.engine
                            .track_operation_at(&mut self.global, || pos!())?;
                    }
                    pc = target;
                }};
            }

            match tag {
                code::tag::CONST => {
                    let index = u32::from(small!(1));
                    #[cfg(feature = "grain-jit")]
                    let constant = jit::program_constant(program, index);
                    #[cfg(not(feature = "grain-jit"))]
                    let constant = program.constant(index);
                    let value = or_raise!(constant, malformed(format!("no constant {index}")));
                    self.push(clone_value(value));
                }

                code::tag::UNIT => self.push(Dynamic::UNIT),
                code::tag::FALSE => self.push(Dynamic::from(false)),
                code::tag::TRUE => self.push(Dynamic::from(true)),

                code::tag::LOAD_LOCAL => {
                    let slot = small!(1);
                    let index = base + slot as usize;
                    if index >= scope_len!() {
                        return Err(malformed(format!("local slot {slot} is out of scope")));
                    }
                    // Reads clone out, matching how Rhai's own variable reads
                    // leave the scope entry alone (`eval/expr.rs:276-278`), and
                    // flattening any shared cell the way a read should.
                    self.push(flatten_clone_value(scope_entry!(index)));
                }

                code::tag::STORE_LOCAL | code::tag::STORE_CONST => {
                    let slot = small!(1);
                    let index = base + slot as usize;
                    if index >= scope_len!() {
                        return Err(malformed(format!("local slot {slot} is out of scope")));
                    }
                    let mut value = self.pop()?;
                    value.set_access_mode(if tag == code::tag::STORE_CONST {
                        AccessMode::ReadOnly
                    } else {
                        AccessMode::ReadWrite
                    });
                    // Through the cell, not over it — see `place`.
                    *place(scope_entry!(index), "", pos!())? = value;
                }

                code::tag::LOAD_NAMED | code::tag::LOAD_SHARED_NAMED => {
                    let index = u32::from(small!(1));
                    let name =
                        or_raise!(program.name(index), malformed(format!("no name {index}")));
                    let flatten = tag == code::tag::LOAD_NAMED;
                    let value = self.load_named(name, scope, flatten, pos!())?;
                    self.push(value);
                }

                code::tag::ASSIGN_NAMED | code::tag::ASSIGN_NAMED_OP => {
                    let index = u32::from(small!(1));
                    let name =
                        or_raise!(program.name(index), malformed(format!("no name {index}")));
                    let op = if tag == code::tag::ASSIGN_NAMED_OP {
                        let index = u32::from(small!(3));
                        #[cfg(feature = "grain-jit")]
                        let assign_op = jit::program_assign_op(program, index);
                        #[cfg(not(feature = "grain-jit"))]
                        let assign_op = program.assign_op(index);
                        Some(or_raise!(
                            assign_op,
                            malformed(format!("no op-assignment {index}"))
                        ))
                    } else {
                        None
                    };

                    // Flattened before assigning, as Rhai does, so a shared
                    // cell is copied out rather than aliased into the target.
                    let rhs = self.pop()?.flatten();
                    self.assign_named(program, op, name, rhs, scope, pos!())?;
                }

                code::tag::DECLARE_LOCAL | code::tag::DECLARE_CONST => {
                    let index = u32::from(small!(1));
                    // A `Scope` entry is keyed by an `ImmutableString`, which
                    // is a refcounted `SmartString`, so building one from the
                    // pool's `&str` copies the bytes and boxes them. A
                    // declaration runs once per `let` rather than per call,
                    // which is why the name is taken from the pool here and
                    // interned ahead of time for parameters only.
                    let name =
                        or_raise!(program.name(index), malformed(format!("no name {index}")));
                    // Flattened, as Rhai flattens a declaration's initializer
                    // (`eval/stmt.rs:438`). A native can hand back a cell that is
                    // already shared, and sharing must stop at the `let` rather
                    // than becoming a property of the new local.
                    let value = self.pop()?.flatten();
                    if tag == code::tag::DECLARE_CONST {
                        scope.push_constant_dynamic(name, value);
                    } else {
                        scope.push_dynamic(name, value);
                    }
                }

                code::tag::ASSIGN_LOCAL
                | code::tag::ASSIGN_LOCAL_OP
                | code::tag::ASSIGN_LOCAL_FROM
                | code::tag::ASSIGN_LOCAL_FROM_OP
                | code::tag::ASSIGN_LOCAL_FROM_CONST
                | code::tag::ASSIGN_LOCAL_FROM_CONST_OP => {
                    let slot = small!(1);
                    let var_name = u32::from(small!(3));
                    // The fused forms carry the source operand where the plain
                    // ones end, so the operator index moves along by it.
                    let from = match tag {
                        code::tag::ASSIGN_LOCAL_FROM
                        | code::tag::ASSIGN_LOCAL_FROM_OP
                        | code::tag::ASSIGN_LOCAL_FROM_CONST
                        | code::tag::ASSIGN_LOCAL_FROM_CONST_OP => {
                            Some(code::assign_source(tag, small!(5)))
                        }
                        _ => None,
                    };
                    let op = match tag {
                        code::tag::ASSIGN_LOCAL_OP => Some(u32::from(small!(5))),
                        code::tag::ASSIGN_LOCAL_FROM_OP | code::tag::ASSIGN_LOCAL_FROM_CONST_OP => {
                            Some(u32::from(small!(7)))
                        }
                        _ => None,
                    };
                    let op = match op {
                        Some(index) => {
                            #[cfg(feature = "grain-jit")]
                            let assign_op = jit::program_assign_op(program, index);
                            #[cfg(not(feature = "grain-jit"))]
                            let assign_op = program.assign_op(index);
                            Some(or_raise!(
                                assign_op,
                                malformed(format!("no op-assignment {index}"))
                            ))
                        }
                        None => None,
                    };

                    // Rhai flattens the right-hand side before assigning
                    // (`eval/stmt.rs:324`), so a shared cell is copied out
                    // rather than aliased into the target. A fused form reads
                    // it out of the slot the `Op::LoadLocal` it swallowed would
                    // have pushed from, which flattens for the same reason.
                    //
                    // Before the target's own bounds check rather than after,
                    // which is the order the two instructions ran in. Both of
                    // those errors describe an artifact the verifier has
                    // already refused, so only their order could differ.
                    let rhs = match from {
                        Some(BinOperand::Local(src)) => {
                            let index = base + src as usize;
                            if index >= scope_len!() {
                                return Err(malformed(format!("local slot {src} is out of scope")));
                            }
                            flatten_clone_value(scope_entry!(index))
                        }
                        // A constant is read exactly as the `Op::Const` this
                        // form swallowed read it, then flattened as the pop it
                        // replaced was. A pool entry is never shared, so the
                        // flatten is the identity — it is spelled to keep the
                        // two paths one rule rather than two.
                        Some(BinOperand::Const(index)) => {
                            #[cfg(feature = "grain-jit")]
                            let constant = jit::program_constant(program, index);
                            #[cfg(not(feature = "grain-jit"))]
                            let constant = program.constant(index);
                            let value =
                                or_raise!(constant, malformed(format!("no constant {index}")));
                            flatten_clone_value(value)
                        }
                        None => self.pop()?.flatten(),
                    };

                    let index = base + slot as usize;
                    if index >= scope_len!() {
                        return Err(malformed(format!("local slot {slot} is out of scope")));
                    }

                    if scope_entry!(index).is_read_only() {
                        let name = or_raise!(
                            program.name(var_name),
                            malformed(format!("no name {var_name}"))
                        );
                        return Err(Box::new(EvalAltResult::ErrorAssignmentToConstant(
                            name.to_string(),
                            pos!(),
                        )));
                    }

                    // Written through rather than over: a slot a closure
                    // captured is a shared cell, and replacing it would sever
                    // every holder. `store` is the same path a chain's tail
                    // and a named assignment take, so `x op= y` resolves
                    // identically wherever the target lives.
                    //
                    // The guard is only needed for a cell a closure captured,
                    // and `x += 1` in a loop is the hot path — so the check
                    // for one is a discriminant test rather than the downcast
                    // chain `write_lock` walks, and the built-in operator is
                    // reached without leaving the dispatch loop or resolving
                    // the position.
                    //
                    // It is not free even so: the tight-loop benchmark went
                    // 1.63x to 1.55x when locals stopped being written over
                    // and started being written through. That is the price of
                    // a shared cell surviving an assignment, and of a chain
                    // over one not taking the host down.
                    let entry = scope_entry!(index);
                    if !is_shared!(entry) {
                        let mut rhs = rhs;
                        let done = match op {
                            Some(op) => self
                                .store_builtin(op, entry, &mut rhs, move || program.position(pc)),
                            None => StoreBuiltinOutcome {
                                handled: false,
                                error: None,
                            },
                        };
                        if done.handled {
                            // The built-in read the right-hand side in place
                            // and left it, so this is where the copy taken
                            // above goes — the only owner of it left.
                            release(rhs);
                            if let Some(error) = done.error {
                                return Err(error);
                            }
                            pc += width;
                            continue;
                        }
                        self.store(program, op, entry, rhs, pos!())?;
                        pc += width;
                        continue;
                    }

                    // The name is only ever read from here: `place` puts it in
                    // the `ErrorDataRace` a contended cell raises. Resolving it
                    // above would put a pool read in front of every `x += 1`
                    // for the sake of a branch almost nothing takes, and the
                    // verifier has already bounded the index (`check_indices`),
                    // so nothing is being checked later that was checked before.
                    let name = or_raise!(
                        program.name(var_name),
                        malformed(format!("no name {var_name}"))
                    );
                    let mut target = place(entry, name, pos!())?;
                    self.store(program, op, &mut target, rhs, pos!())?;
                }

                code::tag::LOAD_THIS | code::tag::LOAD_THIS_SHARED => {
                    let value = or_raise!(
                        self.this.as_ref(),
                        Box::new(EvalAltResult::ErrorUnboundThis(pos!()))
                    );
                    // Rhai's read is `this_ptr.cloned()` and does not flatten
                    // (`eval/expr.rs:272`); its consumers do. Which tag this is
                    // is which consumer asked.
                    self.push(if tag == code::tag::LOAD_THIS {
                        flatten_clone_value(value)
                    } else {
                        clone_value(value)
                    });
                }

                code::tag::REQUIRE_THIS => {
                    if self.this.is_none() {
                        return Err(Box::new(EvalAltResult::ErrorUnboundThis(pos!())));
                    }
                }

                code::tag::ASSIGN_THIS | code::tag::ASSIGN_THIS_OP => {
                    let op = if tag == code::tag::ASSIGN_THIS_OP {
                        let index = u32::from(small!(1));
                        #[cfg(feature = "grain-jit")]
                        let assign_op = jit::program_assign_op(program, index);
                        #[cfg(not(feature = "grain-jit"))]
                        let assign_op = program.assign_op(index);
                        Some(or_raise!(
                            assign_op,
                            malformed(format!("no op-assignment {index}"))
                        ))
                    } else {
                        None
                    };

                    // Flattened before assigning, as everywhere else.
                    let rhs = self.pop()?.flatten();

                    // Taken out of the register rather than borrowed from it:
                    // `store` wants the whole `Vm`, and a write lock into the
                    // field could not outlive that borrow. Put back on both
                    // paths — Rhai's mutation survives an error, and a frame
                    // that lost its receiver would answer `ErrorUnboundThis` to
                    // every read after this one.
                    let mut this = or_raise!(
                        self.this.take(),
                        Box::new(EvalAltResult::ErrorUnboundThis(pos!()))
                    );

                    let outcome = if this.is_read_only() {
                        // Named for an expression that has no name, which is
                        // what Rhai reports too (`eval/stmt.rs:118-122`).
                        Err(Box::new(EvalAltResult::ErrorAssignmentToConstant(
                            String::new(),
                            pos!(),
                        )))
                    } else {
                        // Written through, not over: a shared receiver has to
                        // keep its cell, as a captured local does.
                        match place(&mut this, "", pos!()) {
                            Ok(mut target) => self.store(program, op, &mut target, rhs, pos!()),
                            Err(err) => Err(err),
                        }
                    };

                    self.this = Some(this);
                    outcome?;
                }

                code::tag::POP => {
                    release(self.pop()?);
                }

                code::tag::EVAL_AST | code::tag::EVAL_AST_KEEP => {
                    let index = u32::from(small!(1));
                    let expr = or_raise!(
                        program.residual(index),
                        malformed(format!("no residual {index}"))
                    );
                    let rewind_scope = tag == code::tag::EVAL_AST;

                    // Straight to the walker's own entry points rather than
                    // through `EvalContext::eval_expression_tree_raw`, which
                    // is the same two calls behind a shim that only exists
                    // under `custom_syntax`. Total language coverage rests on
                    // this, so it must not depend on a feature.
                    //
                    // The frame's receiver goes with it, by reference. A body
                    // that uses `this` can still hold a fragment — `this?.x`,
                    // or a `this` body containing an `import` — and the walker
                    // has to read and write the same receiver the surrounding
                    // instructions do. The engine is copied out first so the
                    // four borrows below are of disjoint fields.
                    let engine = self.engine;
                    let value = match expr {
                        Expr::Stmt(block) => engine.eval_stmt_block(
                            &mut self.global,
                            &mut self.caches,
                            scope,
                            self.this.as_mut(),
                            block.statements(),
                            rewind_scope,
                        ),
                        expr => engine.eval_expr(
                            &mut self.global,
                            &mut self.caches,
                            scope,
                            self.this.as_mut(),
                            expr,
                        ),
                    }?;

                    self.push(value);
                }

                code::tag::JUMP => {
                    transfer!(wide!(1) as usize);
                    continue;
                }

                code::tag::JUMP_IF_FALSE | code::tag::JUMP_IF_TRUE => {
                    let target = wide!(1) as usize;
                    let condition = self.pop()?;
                    // Rhai requires a boolean guard and reports the mismatch at
                    // the guard's own position (`eval/stmt.rs:487-490`).
                    let holds = match condition.as_bool() {
                        Ok(holds) => holds,
                        Err(actual) => return Err(self.mismatch::<bool>(actual, pos!())),
                    };
                    release(condition);
                    if holds == (tag == code::tag::JUMP_IF_TRUE) {
                        transfer!(target);
                        continue;
                    }
                }

                code::tag::SKIP_IF_NOT_UNIT => {
                    let target = wide!(1) as usize;
                    let condition = self.inspect()?;
                    if !condition.is_unit() {
                        transfer!(target);
                        continue;
                    }
                }

                code::tag::CALL
                | code::tag::CALL_CAPTURE
                | code::tag::CALL_OP
                | code::tag::BIN_OP
                | code::tag::BIN_OP_FROM_LOCAL
                | code::tag::BIN_OP_FROM_CONST
                | code::tag::BIN_OP_RHS_LOCAL
                | code::tag::BIN_OP_RHS_CONST
                | code::tag::BIN_OP_JF
                | code::tag::BIN_OP_FROM_LOCAL_JF
                | code::tag::BIN_OP_FROM_CONST_JF
                | code::tag::BIN_OP_FROM_LOCAL_JT
                | code::tag::BIN_OP_FROM_CONST_JT
                | code::tag::BIN_OP_RHS_LOCAL_JF
                | code::tag::BIN_OP_RHS_CONST_JF
                | code::tag::UN_OP
                | code::tag::UN_OP_JF => {
                    // What this instruction names, what its result is for
                    // and where its operator comes from, in one table read.
                    // Every operator and every call a program runs arrives
                    // here and asks all three, so asking by comparison walks
                    // a tag list that lengthens with each fused spelling the
                    // compiler learns to emit.
                    let form = code::form(tag);
                    let named_is_local = form & code::form::NAMED_IS_LOCAL != 0;
                    let typed = form & code::form::TYPED != 0;

                    // Whether this instruction is also the branch that reads
                    // its result, and which result takes it. A fused form
                    // carries the target as its last operand, so it sits four
                    // bytes from the end and needs no offset of its own.
                    //
                    // Read before the operands are anywhere, because the exit
                    // below is shared by the path that never puts them on the
                    // stack at all.
                    let branching = form & code::form::BRANCHES != 0;
                    let taken_when = form & code::form::TAKEN_WHEN != 0;
                    // What the operator's result is for: pushed, or read by
                    // the branch the instruction swallowed and not pushed at
                    // all. Every way out of this arm goes through here, so an
                    // operand the typed arms decline reaches the same branch by
                    // the same rule the pair of instructions reached it by.
                    //
                    // The reading itself is a call rather than more of this
                    // macro, because the macro stands at three exits and the
                    // arm every instruction sharing it dispatches through is
                    // the arm this lives in.
                    macro_rules! deliver {
                        ($floor:expr, $value:expr) => {{
                            let value = $value;
                            truncate_stack!($floor);
                            if !branching {
                                self.push(value);
                            } else {
                                // Read and then released here rather than left
                                // to fall out of scope: a swallowed branch is
                                // the one exit that keeps nothing, and it runs
                                // once per turn of every loop with a condition.
                                let holds = self.guard_holds(&value, || pos!())?;
                                release(value);
                                if holds == taken_when {
                                    transfer!(wide!(width - 4) as usize);
                                    continue;
                                }
                            }
                        }};
                    }

                    // Whether the typed arm below has already been asked about
                    // this instruction's operands and declined. Set only by
                    // the direct path, which asks the same question of the
                    // same values: what it hands `apply_binary` is what the
                    // copies below would have held, so a second ask has the
                    // same answer and only the cost is new.
                    #[allow(unused_mut)]
                    let mut typed_asked = false;

                    // The operator applied to its named operands where they
                    // live. The instruction below this one names its operands
                    // rather than taking them off the stack, and the arm that
                    // can answer for them reads through a `&Dynamic` — so
                    // between the two there is a copy onto the stack, a read
                    // straight back off it, and a truncation that throws both
                    // slots away. None of that is the operator.
                    //
                    // Compiled out under `grain-jit`: there each scope entry
                    // is boxed behind a residual accessor that hands back
                    // `&mut Dynamic`, and two of those cannot be alive at once
                    // — so a named pair has no by-reference spelling there and
                    // the copy below is the only path.
                    #[cfg(not(feature = "grain-jit"))]
                    {
                        // A named operand where it lives. What `operand_value!`
                        // reads, without the copy: the same slot, the same
                        // constant, and the same two errors in the same order
                        // for a slot that is out of scope and a constant that
                        // is not there.
                        macro_rules! named_ref {
                            ($offset:expr, $is_local:expr) => {{
                                if $is_local {
                                    let slot = small!($offset);
                                    let at = base + slot as usize;
                                    if at >= scope_len!() {
                                        return Err(malformed(format!(
                                            "local slot {slot} is out of scope"
                                        )));
                                    }
                                    scope.get_by_index(at)
                                } else {
                                    let index = u32::from(small!($offset));
                                    or_raise!(
                                        program.constant(index),
                                        malformed(format!("no constant {index}"))
                                    )
                                }
                            }};
                        }

                        let names_from = form & code::form::NAMES_FROM != 0;
                        if typed
                            && (names_from || form & code::form::NAMES_RHS != 0)
                            && fast_operators!()
                        {
                            // Where the result belongs. A named pair leaves the
                            // stack untouched, so the floor is the depth
                            // itself and `deliver!` truncates nothing; a named
                            // right-hand operand's left one is on the stack,
                            // and the result takes its slot. The check is the
                            // one the typed arm below makes after its pushes,
                            // and reports the same corrupt artifact.
                            let floor = if names_from {
                                self.depth
                            } else {
                                or_raise!(self.depth.checked_sub(1), {
                                    malformed("operator with too few operands".to_string())
                                })
                            };
                            // The left operand of a `NAMES_FROM` instruction is
                            // a slot in both spellings of that shape; only the
                            // right one is spelled either way.
                            let (lhs, rhs) = if names_from {
                                (named_ref!(6, true), named_ref!(8, named_is_local))
                            } else {
                                (stack_ref(self, floor), named_ref!(6, named_is_local))
                            };

                            // A shared cell declines. The copy below reaches a
                            // local *through* one — `flatten_clone` reads the
                            // cell's value out (`types/dynamic.rs:1713-1723`)
                            // — and a reference cannot flatten, so the value
                            // this would apply the operator to is the cell and
                            // not what the copy would have held. Leaving
                            // `typed_asked` alone is what sends it to the copy
                            // and to the typed arm below, which is where it was
                            // answered before.
                            if !is_shared!(lhs) && !is_shared!(rhs) {
                                typed_asked = true;
                                // An unknown kind byte is not an error here, as
                                // in the typed arm below: the dispatch further
                                // down reads the operator out of the pool and
                                // answers whatever it answers.
                                if let Some(kind) = BinOpKind::from_byte(byte!(3)) {
                                    if let Some(value) = apply_binary(kind, lhs, rhs)? {
                                        deliver!(floor, value);
                                        pc += width;
                                        continue;
                                    }
                                }
                            }
                        }
                    }

                    if form & code::form::NAMES_FROM != 0 {
                        // A fused operator names its operands instead of
                        // taking them off the stack; copied here so that
                        // everything below — the typed arms and the dispatch
                        // they fall through to — finds them where it always
                        // did. The two instructions this replaces did exactly
                        // this and cost two more trips round the dispatch loop
                        // for it. The left operand is a slot in both
                        // spellings of this shape; only the right one is
                        // spelled either way.
                        let lhs = operand_value!(6, true);
                        let rhs = operand_value!(8, named_is_local);
                        self.push(lhs);
                        self.push(rhs);
                    } else if form & code::form::NAMES_RHS != 0 {
                        // Only the right operand is named here; the left is
                        // already on the stack, where the expression that
                        // computed it left it.
                        let rhs = operand_value!(6, named_is_local);
                        self.push(rhs);
                    }

                    // The typed operator, ahead of every pool read: an
                    // instruction that runs here touches its own bytes, the top
                    // two operands and nothing else.
                    //
                    // Gated on `fast_operators()` for the reason the built-in
                    // short-circuit below is, and it is the same gate: with it
                    // off, Rhai dispatches a binary operator on a primitive
                    // and so must this (`func/call.rs:1775-1799`).
                    //
                    // A pair this cannot run — a string, a custom type, a
                    // shared cell — falls through into the dispatch below and
                    // is answered by it. See
                    // [`Op::BinOp`](crate::grain::bytecode::Op::BinOp).
                    if typed && !typed_asked && fast_operators!() {
                        let top = self.depth;
                        let under = or_raise!(top.checked_sub(2), {
                            malformed("operator with too few operands".to_string())
                        });
                        // An unknown kind byte is not an error here: the
                        // dispatch below reads the operator out of the pool
                        // and answers whatever it answers, which is what a
                        // verified program's byte can never make it do.
                        if let Some(kind) = BinOpKind::from_byte(byte!(3)) {
                            if let Some(value) =
                                apply_binary(kind, stack_ref(self, under), stack_ref(self, top - 1))?
                            {
                                // No position stamped on the way out: under
                                // `fast_operators` Rhai returns a built-in's
                                // error untouched, which is why `1 / 0` has
                                // none. See `dispatch_failure`.
                                deliver!(under, value);
                                pc += width;
                                continue;
                            }
                        }
                    }

                    // The same for one operand, and the same gate for the
                    // same reason: the walker short-circuits `!` on a `bool`
                    // under `fast_operators` and resolves a function for it
                    // without (`func/call.rs` `eval_fn_call_expr`).
                    //
                    // One value in and one out, so the operand's slot is where
                    // the result belongs and the stack does not change depth --
                    // unless the instruction is the branch that reads it, which
                    // leaves nothing behind at all.
                    let unary = form & code::form::UNARY != 0;
                    if unary && fast_operators!() {
                        let under = or_raise!(self.depth.checked_sub(1), {
                            malformed("operator with too few operands".to_string())
                        });
                        // An unknown kind byte is not an error here, as above:
                        // the dispatch below answers whatever it answers.
                        if let Some(kind) = UnOpKind::from_byte(byte!(3)) {
                            if let Some(value) = apply_unary(kind, stack_ref(self, under)) {
                                if branching {
                                    deliver!(under, value);
                                } else {
                                    overwrite(stack_mut(self, under), value);
                                }
                                pc += width;
                                continue;
                            }
                        }
                    }

                    let name_index = u32::from(small!(1));
                    let name = or_raise!(
                        program.name(name_index),
                        malformed(format!("no name {name_index}"))
                    );
                    let capture = form & code::form::CAPTURES != 0;
                    // The kind byte sits where an argument count would, because
                    // an operator's count is always two — or, for `UN_OP`, one.
                    let argc = if typed {
                        2
                    } else if unary {
                        1
                    } else {
                        byte!(3) as usize
                    };
                    let op = if form & code::form::POOLED_OP != 0 {
                        let index = u32::from(small!(4));
                        Some(or_raise!(
                            program.token(index),
                            malformed(format!("no operator {index}"))
                        ))
                    } else {
                        None
                    };

                    let first = or_raise!(
                        self.depth.checked_sub(argc),
                        malformed("call with too few arguments".to_string())
                    );

                    // Reach the same built-in the walker reaches. Gated on
                    // Rhai's own `fast_operators()` rather than a guard of our
                    // own, so an engine that turns it off gets the dispatch
                    // path on both sides, and one that leaves it on gets the
                    // same answer — including for a user-registered operator
                    // on a primitive, which Rhai's fast path also bypasses
                    // (`func/call.rs:1775-1799`).
                    if let (Some(token), 2, true) = (op, argc, fast_operators!()) {
                        // Flattened first, as Rhai flattens both operands
                        // before it looks for a built-in
                        // (`func/call.rs:1890-1898`). A shared cell reaches
                        // `get_builtin_binary_op_fn` as `Union::Shared`, which
                        // its variant match has no arm for; the type-id
                        // comparison after that match reads through the cell,
                        // finds two equal numeric types and answers "same type
                        // but no built-in" — so the operator resolves to
                        // nothing and the dispatch reports a function that was
                        // never missing. An operand the instruction named
                        // arrives flattened already, because `operand_value!`
                        // reads it out with `flatten_clone`; this is the one
                        // that came off the stack, where a native's return
                        // value is the only thing that puts a cell.
                        for slot in first..self.depth {
                            if is_shared!(*stack_ref(self, slot)) {
                                let held = mem::replace(stack_mut(self, slot), Dynamic::UNIT);
                                *stack_mut(self, slot) = held.flatten();
                            }
                        }
                        let memo = &mut self.operator_memo;
                        let top = self.depth;
                        let (lhs, rhs) = self.stack[..top].split_at_mut(first + 1);
                        let lhs = &mut lhs[first];
                        let rhs = &mut rhs[0];

                        // Custom types go to dispatch first, so a registered
                        // function still wins for them.
                        let builtin = (!lhs.is_variant() && !rhs.is_variant())
                            .then(|| resolve_operator(memo, generation, pc, token, lhs, rhs))
                            .flatten();
                        if let Some((func, need_context)) = builtin {
                            let context = need_context
                                .then(|| (self.engine, name, None, &self.global, pos!()).into());
                            let value = func(context, &mut [lhs, rhs])?;
                            deliver!(first, value);
                            pc += width;
                            continue;
                        }
                    }

                    // Check if it is a built-in syntactic function.
                    let value = self.call_syntactic_or_stacked(
                        program,
                        name_index,
                        name,
                        argc,
                        first,
                        scope,
                        capture,
                        pos!(),
                    )?;
                    deliver!(first, value);
                }

                code::tag::CALL_LOCAL_REF
                | code::tag::CALL_LOCAL_REF_CAPTURE
                | code::tag::CALL_NAMED_REF
                | code::tag::CALL_NAMED_REF_CAPTURE
                | code::tag::CALL_THIS_REF
                | code::tag::CALL_THIS_REF_CAPTURE => {
                    let name_index = u32::from(small!(1));
                    let name = or_raise!(
                        program.name(name_index),
                        malformed(format!("no name {name_index}"))
                    );
                    let argc = byte!(3) as usize;
                    // `this` is a register, so this one carries no operand for
                    // the receiver and is two bytes shorter.
                    let receiver = match tag {
                        code::tag::CALL_LOCAL_REF | code::tag::CALL_LOCAL_REF_CAPTURE => {
                            Receiver::Local(small!(4))
                        }
                        code::tag::CALL_NAMED_REF | code::tag::CALL_NAMED_REF_CAPTURE => {
                            Receiver::Named(u32::from(small!(4)))
                        }
                        code::tag::CALL_THIS_REF | code::tag::CALL_THIS_REF_CAPTURE => {
                            Receiver::This
                        }
                        _ => unreachable!(),
                    };
                    let capture = matches!(
                        tag,
                        code::tag::CALL_LOCAL_REF_CAPTURE
                            | code::tag::CALL_NAMED_REF_CAPTURE
                            | code::tag::CALL_THIS_REF_CAPTURE
                    );

                    let value = self.call_by_reference(
                        program,
                        name_index,
                        name,
                        argc,
                        receiver,
                        scope,
                        base,
                        capture,
                        pos!(),
                    )?;
                    self.push(value);
                }

                code::tag::ROTATE => {
                    let under = byte!(1) as usize;
                    let top = or_raise!(
                        self.depth.checked_sub(1),
                        malformed("rotate on an empty stack".to_string())
                    );
                    let to = or_raise!(
                        top.checked_sub(under),
                        malformed("rotate past the bottom".to_string())
                    );
                    self.values_mut()[to..].rotate_right(1);
                }

                // Emitted only for a literal, which is not syntax under the
                // feature that removes the type.
                #[cfg(not(feature = "no_index"))]
                code::tag::MAKE_ARRAY => {
                    let len = small!(1) as usize;
                    let first = or_raise!(
                        self.depth.checked_sub(len),
                        malformed("array with too few elements".to_string())
                    );

                    // The running total belongs to this literal and goes with
                    // it. `Op::CheckSize` is what filled it in, one element at
                    // a time, and what raised `ErrorDataTooLarge` against the
                    // element that tipped it over (`eval/expr.rs:307-330`).
                    //
                    // Only if there was one: an empty literal emits no
                    // `CheckSize` and pushed nothing, so popping here would
                    // take the *enclosing* literal's total — `[a, [], b]`.
                    if len > 0 {
                        self.sizes.pop();
                    }

                    // Flattened, as Rhai does, so a shared cell is copied in
                    // rather than aliased.
                    let array: Array = self
                        .take_values_from(first)
                        .into_iter()
                        .map(Dynamic::flatten)
                        .collect();
                    self.push(Dynamic::from_array(array));
                }

                #[cfg(not(feature = "no_object"))]
                code::tag::MAKE_MAP => {
                    let len = small!(1) as usize;
                    let first = or_raise!(
                        self.depth.checked_sub(2 * len + 1),
                        malformed("map with too few operands".to_string())
                    );
                    // As for `MakeArray`: nothing was pushed for a literal
                    // with no computed entries, so nothing may be popped.
                    if len > 0 {
                        self.sizes.pop();
                    }

                    let mut parts = self.take_values_from(first).into_iter();
                    let template = parts.next().expect("checked above");
                    let mut map = or_raise!(
                        template.try_cast::<Map>(),
                        malformed("map literal without a template".to_string())
                    );
                    while let Some(key) = parts.next() {
                        let value = parts.next().expect("pairs, checked above");
                        let key = key.into_immutable_string().map_err(|actual| {
                            malformed(format!("map key is a {actual}, not a string"))
                        })?;
                        // Flattened as Rhai does, so a shared cell is copied
                        // in rather than aliased.
                        map.insert(key.as_str().into(), value.flatten());
                    }
                    drop(parts);
                    self.push(Dynamic::from_map(map));
                }

                code::tag::CHECK_ARRAY_SIZE | code::tag::CHECK_MAP_SIZE => {
                    let index = small!(1);
                    let map = tag == code::tag::CHECK_MAP_SIZE;
                    self.check_size(index, map, pos!())?;
                }

                code::tag::SWITCH => {
                    let index = u32::from(small!(1));
                    let table = or_raise!(
                        program.switch(index),
                        malformed(format!("no switch {index}"))
                    );
                    let subject = self.pop()?;
                    // Always a jump: an arm that matched nothing still has the
                    // default to go to.
                    let target = table.dispatch(&subject) as usize;
                    release(subject);
                    transfer!(target);
                    continue;
                }

                code::tag::LOAD_SHARED => {
                    let slot = small!(1);
                    let index = base + slot as usize;
                    if index >= scope_len!() {
                        return Err(malformed(format!("local slot {slot} is out of scope")));
                    }
                    // Cloned, not flattened: cloning a shared `Dynamic` clones
                    // the `Rc`, which is the capture.
                    self.push(clone_value(scope_entry!(index)));
                }

                // Emitted only for a closure capture, which cannot be parsed
                // under `no_closure`.
                #[cfg(not(feature = "no_closure"))]
                code::tag::SHARE | code::tag::SHARE_NAMED => {
                    let entry = if tag == code::tag::SHARE {
                        let slot = small!(1);
                        let index = base + slot as usize;
                        if index >= scope_len!() {
                            return Err(malformed(format!("local slot {slot} is out of scope")));
                        }
                        Some(index)
                    } else {
                        let name_index = u32::from(small!(1));
                        let name = or_raise!(
                            program.name(name_index),
                            malformed(format!("no name {name_index}"))
                        );
                        // The resolver gets first refusal, and a name it
                        // answers is not shared at all (`eval/stmt.rs:998`).
                        if self.resolve_var(name, scope, pos!())?.is_some() {
                            pc += width;
                            continue;
                        }
                        // `iter_raw` walks the scope from the top down, which is
                        // the order shadowing wants — the first match is the
                        // live one — but it counts from the other end than
                        // `get_mut_by_index` does, so the position has to be
                        // turned back round. Rhai reaches the same entry
                        // through `Scope::search`, which is not public
                        // (`eval/stmt.rs:1009`).
                        let depth = scope_len!();
                        let found = scope
                            .iter_raw()
                            .position(|(entry, ..)| entry == name)
                            .map(|from_top| depth - 1 - from_top);
                        Some(or_raise!(found, missing(name, pos!())))
                    };

                    if let Some(index) = entry {
                        let value = scope_entry!(index);
                        if !value.is_shared() {
                            *value = value.take().into_shared();
                        }
                    }
                }

                code::tag::MAKE_CLOSURE => {
                    let index = u32::from(small!(1));
                    let name =
                        or_raise!(program.name(index), malformed(format!("no name {index}")));
                    // Non-validated, because `anon$…` is not a name a script could
                    // have written and the validating constructors refuse it.
                    // Nothing unsound rides on that check — a name that will
                    // not resolve simply fails when the pointer is called.
                    //
                    // Carrying the body where there is a share of the program
                    // to carry, which is what spares a native the lookup on
                    // every element. See [`callback::pointer`].
                    debug_assert!(
                        self.callbacks
                            .as_ref()
                            .map_or(true, |owned| core::ptr::eq(&**owned, program)),
                        "a pointer would be handed out carrying a program this frame is not running"
                    );
                    let typ = self
                        .callbacks
                        .as_ref()
                        .and_then(|owned| callback::pointer(owned, index, name))
                        .unwrap_or(FnPtrType::Normal);
                    self.push(
                        FnPtr {
                            name: name.into(),
                            curry: Default::default(),
                            #[cfg(not(feature = "no_function"))]
                            env: None,
                            typ,
                        }
                        .into(),
                    );
                }

                #[cfg(not(feature = "no_closure"))]
                code::tag::IS_SHARED => {
                    let value = self.pop()?;
                    self.push(value.is_shared().into());
                }

                code::tag::MAKE_FN_PTR => {
                    let name = self.pop()?;
                    let name = name
                        .into_immutable_string()
                        .map_err(|actual| self.mismatch::<ImmutableString>(actual, pos!()))?;
                    // Validates that the name is an identifier, as Rhai's own
                    // `Fn(..)` does (`func/call.rs:1215`).
                    let pointer = FnPtr::new(name).map_err(|mut err| {
                        if err.position().is_none() {
                            err.set_position(pos!());
                        }
                        err
                    })?;
                    self.push(pointer.into());
                }

                code::tag::CURRY => {
                    let argc = byte!(1) as usize;
                    let at = or_raise!(
                        self.depth.checked_sub(argc + 1),
                        malformed("curry is missing its target".into())
                    );
                    let mut pointer = or_raise!(
                        clone_operand(&self.stack[at]).try_cast::<FnPtr>(),
                        self.mismatch::<FnPtr>(operand_ref(&self.stack[at]).type_name(), pos!())
                    );
                    for value in self.take_values_from(at + 1) {
                        pointer.add_curry(value);
                    }
                    truncate_stack!(at);
                    self.push(pointer.into());
                }

                code::tag::CALL_FN_PTR
                | code::tag::CALL_FN_PTR_METHOD
                | code::tag::CALL_FN_PTR_ON_LOCAL
                | code::tag::CALL_FN_PTR_ON_NAMED
                | code::tag::CALL_FN_PTR_ON_THIS => {
                    let argc = byte!(1) as usize;
                    let method = tag != code::tag::CALL_FN_PTR;
                    let receiver = match tag {
                        code::tag::CALL_FN_PTR_ON_LOCAL => Some(Receiver::Local(small!(2))),
                        code::tag::CALL_FN_PTR_ON_NAMED => {
                            Some(Receiver::Named(u32::from(small!(2))))
                        }
                        code::tag::CALL_FN_PTR_ON_THIS => Some(Receiver::This),
                        _ => None,
                    };
                    let value =
                        self.call_fn_ptr(program, argc, method, receiver, scope, base, pos!())?;
                    self.push(value);
                }

                code::tag::INTERPOLATE_START => {
                    self.push(self.engine.const_empty_string().into());
                }

                code::tag::INTERPOLATE_APPEND => {
                    let segment = self.pop()?;
                    self.append_segment(segment, pos!())?;
                }

                code::tag::INTERPOLATE_END => {
                    let buffer = self.pop()?;
                    let text = buffer
                        .into_immutable_string()
                        .map_err(|_| malformed("interpolation lost its buffer".into()))?;
                    // Interned, as Rhai does: the same rendered string in ten
                    // places is one allocation, which is the whole reason the
                    // engine keeps an interner.
                    let value = self.engine.get_interned_string(text.as_str());
                    self.push(value.into());
                }

                code::tag::CHAIN | code::tag::CHAIN_DISCARD => {
                    let index = u32::from(small!(1));
                    let chain =
                        or_raise!(program.chain(index), malformed(format!("no chain {index}")));
                    // An assignment is not an expression here: what it
                    // evaluates to is the `Op::Unit` beside it, emitted only
                    // where something reads it. A read in statement position
                    // has nothing behind it to take its value off again, and
                    // says so in its tag; the walk itself is the same either
                    // way, so a chain that raises raises identically.
                    let keeps = matches!(chain.tail, Tail::Read) && tag != code::tag::CHAIN_DISCARD;
                    let value = self.run_chain(program, chain, index, scope, base, pos!())?;
                    if keeps {
                        self.push(value);
                    } else {
                        drop(value);
                    }
                }

                code::tag::INDEX_SET
                | code::tag::INDEX_SET_FROM_LOCAL
                | code::tag::INDEX_SET_FROM_CONST
                | code::tag::INDEX_SET_FROM_LOCAL_VALUE_CONST
                | code::tag::INDEX_SET_FROM_CONST_VALUE_CONST
                | code::tag::INDEX_SET_OP
                | code::tag::INDEX_SET_OP_FROM_LOCAL
                | code::tag::INDEX_SET_OP_FROM_CONST
                | code::tag::INDEX_SET_OP_FROM_LOCAL_VALUE_CONST
                | code::tag::INDEX_SET_OP_FROM_CONST_VALUE_CONST => {
                    // The compiler has already decided the shape, so what is
                    // left to test is the types — which it could not know. A
                    // slot holding a writable, unshared `Array` and an index
                    // that is a non-negative integer inside it runs here; a
                    // map, a shared cell, a host type with an indexer or an
                    // out-of-range index breaks out and is answered by the
                    // chain below, reporting exactly what it reports.
                    //
                    // The operands are the chain's own: the value is on top,
                    // the index under it or named by the instruction, and an
                    // assigning chain leaves nothing behind. Only an `Array`
                    // reaches the fast path, and `no_index` takes the arm it
                    // destructures with the rest of indexing — so there is
                    // nothing left here to be fast about, and the compiler
                    // emits no `IndexSet` on that build either.
                    let named = !matches!(tag, code::tag::INDEX_SET | code::tag::INDEX_SET_OP);
                    // A named value was never pushed, so the index it would
                    // have sat on top of is the top of the stack instead.
                    let valued = matches!(
                        tag,
                        code::tag::INDEX_SET_FROM_LOCAL_VALUE_CONST
                            | code::tag::INDEX_SET_FROM_CONST_VALUE_CONST
                            | code::tag::INDEX_SET_OP_FROM_LOCAL_VALUE_CONST
                            | code::tag::INDEX_SET_OP_FROM_CONST_VALUE_CONST
                    );
                    #[cfg_attr(feature = "no_index", allow(unused_mut))]
                    let mut assigned = false;
                    #[cfg(not(feature = "no_index"))]
                    'fast: {
                        let under = self.depth.checked_sub(if valued { 1 } else { 2 });
                        let Some(i) = indexed_int!(5, under) else {
                            break 'fast;
                        };
                        // A negative index counts from the end, which is
                        // `get_indexed_mut`'s rule and not worth restating.
                        let Ok(i) = usize::try_from(i) else {
                            break 'fast;
                        };
                        let at = base + small!(3) as usize;
                        if at >= scope_len!() {
                            break 'fast;
                        }
                        // The operator an op-assignment applies, which is the
                        // kind byte the instruction ends with. Gated on
                        // `fast_operators()` for the reason every typed arm is:
                        // with it off the walk resolves a function, and a
                        // host-registered `+=` on integers wins on both sides.
                        let kind = match tag {
                            code::tag::INDEX_SET_OP
                            | code::tag::INDEX_SET_OP_FROM_LOCAL
                            | code::tag::INDEX_SET_OP_FROM_CONST
                            | code::tag::INDEX_SET_OP_FROM_LOCAL_VALUE_CONST
                            | code::tag::INDEX_SET_OP_FROM_CONST_VALUE_CONST => {
                                if !fast_operators!() {
                                    break 'fast;
                                }
                                // A byte naming no operator is not an error:
                                // the walk reads the operator out of the
                                // chain's pool entry and answers whatever it
                                // answers, which is what a verified program's
                                // byte can never make it do.
                                let Some(kind) = BinOpKind::from_byte(byte!(width - 1)) else {
                                    break 'fast;
                                };
                                Some(kind)
                            }
                            _ => None,
                        };
                        // Read before the array is borrowed out of the same
                        // scope, which a named value may be read from too.
                        let named_value = if valued {
                            Some(assigned_value!())
                        } else {
                            None
                        };
                        let Union::Array(array, _, AccessMode::ReadWrite) = &mut scope_entry!(at).0
                        else {
                            break 'fast;
                        };
                        let Some(cell) = array.get_mut(i) else {
                            break 'fast;
                        };
                        match kind {
                            // The operator applied to the element where it
                            // lies, which is the whole of what this spelling
                            // buys: an element reached as a `&mut Dynamic`
                            // needs no `Target` to read through, no copy of it
                            // to apply to and no second walk to write back.
                            //
                            // Nothing is consumed until it has answered. A
                            // pair the arm has no entry for — a string, a
                            // shared cell, an integer element under a float
                            // operand — and an operator that fails both leave
                            // every operand where the walk expects it, and the
                            // walk is what answers, error included.
                            Some(kind) => {
                                let rhs = match named_value.as_ref() {
                                    Some(value) => value,
                                    None => {
                                        let Some(top) = self.depth.checked_sub(1) else {
                                            break 'fast;
                                        };
                                        stack_ref(self, top)
                                    }
                                };
                                match apply_assign(kind, cell, rhs) {
                                    Ok(Some(())) => {}
                                    Ok(None) | Err(..) => break 'fast,
                                }
                                match named_value {
                                    Some(value) => release(value),
                                    None => release(self.pop_or_unit()),
                                }
                            }
                            None => {
                                let value = match named_value {
                                    Some(value) => value,
                                    None => self.pop_or_unit(),
                                };
                                overwrite(cell, value);
                            }
                        }
                        if !named {
                            // The index operand, done with.
                            release(self.pop_or_unit());
                        }
                        assigned = true;
                    }
                    if assigned {
                        pc += width;
                        continue;
                    }

                    // Declined, so the walk runs — and it reads its operands
                    // off the stack, where the index belongs under the value.
                    if named || valued {
                        let value = if valued {
                            assigned_value!()
                        } else {
                            self.pop_or_unit()
                        };
                        if named {
                            let index = indexed_value!(5);
                            self.push(index);
                        }
                        self.push(value);
                    }

                    let index = u32::from(small!(1));
                    let chain =
                        or_raise!(program.chain(index), malformed(format!("no chain {index}")));
                    // Always an assigning tail — that is what `indexed_slot`
                    // selects on — so the walk leaves nothing here either.
                    drop(self.run_chain(program, chain, index, scope, base, pos!())?);
                }

                code::tag::INDEX_GET
                | code::tag::INDEX_GET_FROM_LOCAL
                | code::tag::INDEX_GET_FROM_CONST => {
                    // `Op::IndexSet`'s speculation, read side: the compiler
                    // decided the shape and what is left to test is the types.
                    // A shared root or a shared element breaks out, because the
                    // walk reaches either through a `Target` that takes a write
                    // lock and this does not.
                    //
                    // The element takes the index's place —
                    // `Target::take_or_clone` clones what it referenced, so a
                    // clone is what the walk produces too. Where the
                    // instruction names its index that place is one above the
                    // stack it arrived on, and the two naming tags reach the
                    // scope entry and the constant themselves rather than
                    // spending an instruction pushing what the next one takes
                    // straight off again.
                    let named = tag != code::tag::INDEX_GET;
                    #[cfg_attr(feature = "no_index", allow(unused_mut))]
                    let mut read = false;
                    #[cfg(not(feature = "no_index"))]
                    'fast: {
                        let Some(i) = indexed_int!(5, self.depth.checked_sub(1)) else {
                            break 'fast;
                        };
                        // A negative index counts from the end, which is
                        // `get_indexed_mut`'s rule and not worth restating.
                        let Ok(i) = usize::try_from(i) else {
                            break 'fast;
                        };
                        let at = base + small!(3) as usize;
                        if at >= scope_len!() {
                            break 'fast;
                        }
                        let Union::Array(array, ..) = &scope_entry!(at).0 else {
                            break 'fast;
                        };
                        let Some(cell) = array.get(i) else {
                            break 'fast;
                        };
                        if is_shared!(*cell) {
                            break 'fast;
                        }
                        let value = clone_value(cell);
                        if named {
                            self.push(value);
                        } else {
                            *stack_mut(self, self.depth - 1) = value;
                        }
                        read = true;
                    }
                    if read {
                        pc += width;
                        continue;
                    }

                    // Declined, so the walk runs — and it reads its one
                    // operand off the stack, where a named index has never
                    // been.
                    if named {
                        let index = indexed_value!(5);
                        self.push(index);
                    }

                    let index = u32::from(small!(1));
                    let chain =
                        or_raise!(program.chain(index), malformed(format!("no chain {index}")));
                    let value = self.run_chain(program, chain, index, scope, base, pos!())?;
                    self.push(value);
                }

                code::tag::UNWIND_TO => {
                    let depth = small!(1);
                    let target = base + depth as usize;
                    if target > scope_len!() {
                        return Err(malformed(format!(
                            "unwind to {target} past a scope of {}",
                            scope_len!()
                        )));
                    }
                    scope.rewind(target);
                }

                code::tag::TICK => {
                    #[cfg(feature = "grain-jit")]
                    if let Some(error) = jit::track_operation_error(self, program, pc) {
                        return Err(error);
                    }
                    #[cfg(not(feature = "grain-jit"))]
                    self.engine.track_operation(&mut self.global, pos!())?;
                }

                code::tag::CHECKPOINT => self.unwind_floor = scope_len!(),

                code::tag::STATEMENT => {
                    // Nothing to stop for without the interface to stop at, and
                    // no marker in the code at all unless it was compiled by a
                    // build that has one.
                    #[cfg(feature = "debugging")]
                    {
                        let depth = small!(1);
                        self.at_statement(scope, depth, pos!())?;
                    }
                }

                code::tag::PUSH_HANDLER | code::tag::PUSH_HANDLER_VAR => {
                    let target = wide!(1) as usize;
                    let catch_var = if tag == code::tag::PUSH_HANDLER_VAR {
                        Some(u32::from(small!(5)))
                    } else {
                        None
                    };
                    self.handlers.push(Handler {
                        target,
                        catch_var,
                        operands: self.depth,
                        scope_len: scope_len!(),
                        iters: self.iterators.len(),
                        caught: None,
                    });
                }

                code::tag::POP_HANDLER => {
                    self.handlers.pop();
                }

                code::tag::ITER_INIT => {
                    let iterable = self.pop()?;
                    self.iter_init(iterable, pos!())?;
                }

                code::tag::ITER_DROP => {
                    self.iterators.pop();
                }

                code::tag::ITER_NEXT
                | code::tag::ITER_NEXT_INDEXED
                | code::tag::ITER_NEXT_STORE => {
                    let body = wide!(1) as usize;
                    let iteration = or_raise!(
                        self.iterators.last_mut(),
                        malformed("no iterator to advance".to_string())
                    );

                    let Some(item) = iteration.items.next() else {
                        // Out of the loop by falling through: this edge is
                        // taken once, and the one back into the body is taken
                        // every turn.
                        self.iterators.pop();
                        pc += width;
                        continue;
                    };

                    // Counted before the item is unwrapped, as Rhai does, so a
                    // loop long enough to wrap the counter is an error rather
                    // than a wrap.
                    iteration.count = or_raise!(iteration.count.checked_add(1), {
                        Box::new(EvalAltResult::ErrorArithmetic(
                            format!("for-loop counter overflow: {}", iteration.count),
                            pos!(),
                        ))
                    });
                    let count = iteration.count;

                    // A fallible iterator's error is positioned at the
                    // iterable, and only if it brought none of its own
                    // (`eval/stmt.rs:749`).
                    let value = item.map_err(|mut err| {
                        if err.position().is_none() {
                            err.set_position(pos!());
                        }
                        err
                    })?;

                    if tag == code::tag::ITER_NEXT_STORE {
                        // The `Op::StoreShared` this swallowed, which is where
                        // the loop variable is written on every turn.
                        let slot = small!(5);
                        let index = base + slot as usize;
                        if index >= scope_len!() {
                            return Err(malformed(format!("local slot {slot} is out of scope")));
                        }
                        store_shared(scope_entry!(index), value.flatten(), move || {
                            program.position(pc)
                        })?;
                    } else {
                        if tag == code::tag::ITER_NEXT_INDEXED {
                            self.push(Dynamic::from(count));
                        }
                        self.push(value.flatten());
                    }

                    // The back edge, so the turn is metered and the JIT driver
                    // is consulted here rather than at a jump of the body's.
                    transfer!(body);
                    continue;
                }

                code::tag::STORE_SHARED => {
                    let slot = small!(1);
                    let index = base + slot as usize;
                    if index >= scope_len!() {
                        return Err(malformed(format!("local slot {slot} is out of scope")));
                    }
                    let value = self.pop()?;
                    store_shared(scope_entry!(index), value, move || program.position(pc))?;
                }

                code::tag::THROW => {
                    // Flattened, as Rhai does, so a shared cell is thrown as
                    // its value rather than as the cell.
                    let value = self.pop()?.flatten();
                    return Err(Box::new(EvalAltResult::ErrorRuntime(value, pos!())));
                }

                code::tag::RETURN => {
                    let value = self.pop_or_unit();
                    // Whatever else this frame left behind goes with it, so a
                    // caller's stack is exactly as it was.
                    truncate_stack!(stack_base);
                    return Ok(value);
                }

                // `code::width` already refused anything it does not know, so
                // this is unreachable — but a wildcard is what stops a new tag
                // from silently falling through to the next instruction.
                _ => return Err(malformed(format!("unknown instruction {tag:#04x}"))),
            }

            pc += width;
        }
    }
}

/// The `this` register, reached by hand-built chunks.
///
/// The compiler does not emit any of these yet — it still refuses a body that
/// mentions `this` — so this is the only thing that executes them until it does.
/// Worth having on its own account regardless: what a hand-made artifact can say
/// is exactly what a verifier-plus-VM has to survive.
///
/// Every case enters through [`Vm::call_fn_with_options`], which is the only
/// way to bind a receiver from outside — and which `no_function` takes away
/// along with Rhai's `CallFnOptions`.
#[cfg(test)]
#[cfg(not(feature = "no_function"))]
mod tests {
    use super::*;
    use crate::grain::bytecode::{assemble, Chain, Chunk, Op, Positions, Step, Strings, Tail};
    use crate::grain::format::Abi;
    use crate::grain::program::{Function, Parts};
    use crate::{CallFnOptions, Engine, Scope, INT};

    /// The meta-tracer's `getarrayitem_gc_r` works on an array of references,
    /// just as PyPy's value stack does.  The boxes are allocated only when the
    /// frame grows; replacing a live operand must keep the slot address stable.
    #[test]
    #[cfg(feature = "grain-jit")]
    fn jit_operand_slots_are_pointer_sized_and_reused() {
        assert_eq!(
            core::mem::size_of::<OperandSlot>(),
            core::mem::size_of::<usize>()
        );

        let mut slot = operand_slot(Dynamic::from(1 as INT));
        let address = operand_ref(&slot) as *const Dynamic;
        set_operand(&mut slot, Dynamic::from(2 as INT));

        assert_eq!(address, operand_ref(&slot) as *const Dynamic);
        assert_eq!(operand_ref(&slot).as_int(), Ok(2));
    }

    #[test]
    #[cfg(feature = "grain-jit")]
    fn jit_restore_updates_only_the_current_frame_image() {
        let engine = Engine::new();
        let vm = Vm::new(&engine);
        let mut scope = Scope::new();
        let mut frame = GrainFrame {
            scope: &mut scope,
            base: 1,
            reached: 2,
            stack_base: 3,
            jit_resume_pc_plus_one: 0,
        };
        let frame_addr = &mut frame as *mut GrainFrame<'_, '_> as usize as i64;
        let vm_addr = &vm as *const Vm<'_> as usize as i64;
        let scope_addr = frame.scope as *mut Scope<'_> as usize as i64;

        frame.restore_from_jit(&[frame_addr, vm_addr, scope_addr, 10, 20, 30], &vm);
        assert_eq!((frame.base, frame.reached, frame.stack_base), (10, 20, 30));

        frame.restore_from_jit(&[frame_addr + 1, vm_addr, scope_addr, 40, 50, 60], &vm);
        assert_eq!(
            (frame.base, frame.reached, frame.stack_base),
            (10, 20, 30),
            "a resume image for another inlined frame must be ignored",
        );
    }

    /// A program of phantom functions, named `f` upwards in the order given.
    ///
    /// The main chunk does nothing: everything here is entered through
    /// [`Vm::call_fn_with_options`], which is the only thing that can bind a
    /// receiver.
    fn program_of(bodies: &[&[Op]], consts: Vec<Dynamic>) -> Program<'static> {
        program_with(bodies, consts, Vec::new(), Vec::new())
    }

    /// The same, for the chain instruction, whose record lives in a pool rather
    /// than in the code.
    fn program_with_chains(
        bodies: &[&[Op]],
        consts: Vec<Dynamic>,
        chains: Vec<Chain>,
    ) -> Program<'static> {
        program_with(bodies, consts, chains, Vec::new())
    }

    /// The general form. Name indices are positions in `NAMES`.
    fn program_with(
        bodies: &[&[Op]],
        consts: Vec<Dynamic>,
        chains: Vec<Chain>,
        residuals: Vec<Expr>,
    ) -> Program<'static> {
        /// `f` and `g` are the functions; the rest are for chain steps to name.
        const NAMES: [&str; 4] = ["f", "g", "push", "len"];

        let mut all = vec![Op::Unit, Op::Return];
        let mut spans = Vec::new();
        for body in bodies {
            let start = all.len();
            all.extend_from_slice(body);
            spans.push(start..all.len());
        }

        // Assembling a prefix gives the byte offset that prefix ends at, which
        // is what a chunk is measured in.
        let end_of = |ops: usize| {
            assemble(&all[..ops])
                .expect("the test ops must assemble")
                .0
                .len() as u32
        };
        let (code, _) = assemble(&all).expect("the test ops must assemble");

        let functions = spans
            .iter()
            .enumerate()
            .map(|(index, span)| Function {
                name: index as u32,
                params: Vec::new(),
                param_names: Vec::new(),
                this_type: None,
                takes_this: false,
                chunk: Chunk::new(end_of(span.start), end_of(span.end), 8),
            })
            .collect();

        Program::new(
            Abi::host().caps,
            code.into(),
            Chunk::new(0, end_of(2), 8),
            functions,
            Parts {
                positions: Positions::default(),
                debug_id: None,
                residuals,
                consts,
                names: Strings::new(NAMES),
                tokens: Vec::new(),
                assign_ops: Vec::new(),
                chains,
                switches: Vec::new(),
                lib: None,
                #[cfg(not(feature = "no_module"))]
                resolver: None,
                source: None,
            },
        )
    }

    /// The one function `f`, for the cases that need no callee.
    fn one(ops: &[Op], consts: Vec<Dynamic>) -> Program<'static> {
        program_of(&[ops], consts)
    }

    fn call(program: &Program, this: Option<&mut Dynamic>) -> Result<Dynamic, Box<EvalAltResult>> {
        let engine = Engine::new();
        let mut options = CallFnOptions::new().eval_ast(false);
        options.this_ptr = this;
        Vm::new(&engine).call_fn_with_options(options, &mut Scope::new(), program, "f", ())
    }

    /// A fused compare-and-branch keeps nothing: the operator's result goes to
    /// the branch that swallowed it, so a turn of the loop has to leave the
    /// operand stack at the depth it found it.
    ///
    /// Drifting *upwards* is invisible in what the script evaluates to — a
    /// `Return` truncates to the frame's floor and every other reader is
    /// relative to the depth — so what says so is the allocation: one slot
    /// left behind per turn grows the stack past anything the chunk declares,
    /// and four hundred turns put it two orders of magnitude out.
    ///
    #[test]
    fn a_loop_turn_leaves_the_operand_stack_where_it_found_it() {
        // Both guard shapes: `i < 400` names both of its operands and so
        // arrives with the stack untouched, while `(i + 1) <= 400` names only
        // the right one and finds the left where the operator before it left
        // it — which is the slot the result has to take.
        const LOOPS: [&str; 2] = [
            "let s = 0; let i = 0; while i < 400 { s = s + i; i = i + 1; } s",
            "let s = 0; let i = 0; while (i + 1) <= 400 { s = s + i; i = i + 1; } s",
        ];

        let engine = Engine::new();
        for source in LOOPS {
            let ast = engine.compile(source).expect("the source parses");
            let program = crate::grain::Compiler::new().compile(&ast);
            let ceiling = (program.max_stack() as usize).next_power_of_two().max(8);

            let mut vm = Vm::new(&engine);
            let value = vm
                .eval_with_scope(&mut Scope::new(), &program)
                .expect("the loop runs");
            assert_eq!(value.as_int(), Ok(79800), "{source}");
            assert!(
                vm.stack.len() <= ceiling,
                "four hundred turns of `{source}` grew the operand stack to {}, \
                 against a chunk that declares {}",
                vm.stack.len(),
                program.max_stack(),
            );
        }
    }

    #[test]
    fn a_bound_receiver_is_what_load_this_pushes() {
        let program = one(&[Op::LoadThis, Op::Return], Vec::new());
        let mut this = Dynamic::from(7 as INT);
        assert_eq!(call(&program, Some(&mut this)).unwrap().as_int(), Ok(7));
    }

    #[test]
    fn reading_an_unbound_receiver_is_an_error() {
        let program = one(&[Op::LoadThis, Op::Return], Vec::new());
        let err = *call(&program, None).unwrap_err();
        assert!(
            matches!(
                &err,
                EvalAltResult::ErrorInFunctionCall(_, _, inner, _)
                    if matches!(**inner, EvalAltResult::ErrorUnboundThis(..))
            ),
            "expected an unbound `this`, got {err:?}"
        );
    }

    /// `this = v` checks binding before evaluating `v`, which is the whole
    /// reason [`Op::RequireThis`] is a separate instruction.
    #[test]
    fn assigning_to_an_unbound_receiver_is_caught_before_the_value_runs() {
        let program = one(&[Op::RequireThis, Op::Unit, Op::Return], Vec::new());
        let err = *call(&program, None).unwrap_err();
        assert!(
            matches!(
                &err,
                EvalAltResult::ErrorInFunctionCall(_, _, inner, _)
                    if matches!(**inner, EvalAltResult::ErrorUnboundThis(..))
            ),
            "expected an unbound `this`, got {err:?}"
        );
    }

    #[test]
    fn a_write_through_this_reaches_the_hosts_value() {
        let program = one(
            &[
                Op::Const(0),
                Op::AssignThis { op: None },
                Op::Unit,
                Op::Return,
            ],
            vec![Dynamic::from(9 as INT)],
        );
        let mut this = Dynamic::from(1 as INT);
        assert!(call(&program, Some(&mut this)).is_ok());
        assert_eq!(this.as_int(), Ok(9));
    }

    /// Rhai reaches `this` through the caller's storage, so a body that mutates
    /// and then raises has already written.
    #[test]
    fn a_write_through_this_survives_a_failure_after_it() {
        let program = one(
            &[
                Op::Const(0),
                Op::AssignThis { op: None },
                Op::Const(1),
                Op::Throw,
            ],
            vec![Dynamic::from(9 as INT), Dynamic::from("boom")],
        );
        let mut this = Dynamic::from(1 as INT);
        assert!(call(&program, Some(&mut this)).is_err());
        assert_eq!(this.as_int(), Ok(9));
    }

    /// A callee gets `None`, whatever its caller was holding
    /// (`func/call.rs:669`). No conditional makes that true — every ordinary
    /// call goes through `call_compiled`, which installs `None`.
    ///
    /// Worth testing at all because the failure is invisible: a register that
    /// leaked would only show up in a callee that reads `this`, and reading a
    /// value that happens to be there looks like success.
    #[test]
    fn a_receiver_is_not_inherited_by_a_callee() {
        // `f` has a receiver and calls `g`, which reads one it was never given.
        let program = program_of(
            &[
                &[
                    Op::Call {
                        name: 1,
                        argc: 0,
                        op: None,
                        capture_parent_scope: false,
                    },
                    Op::Return,
                ],
                &[Op::LoadThis, Op::Return],
            ],
            Vec::new(),
        );

        let mut this = Dynamic::from(7 as INT);
        let err = *call(&program, Some(&mut this)).unwrap_err();
        assert!(
            format!("{err:?}").contains("ErrorUnboundThis"),
            "expected `g` to have no receiver, got {err:?}"
        );
    }

    /// And the caller still has its own afterwards.
    #[test]
    fn a_callee_frame_does_not_disturb_the_callers_receiver() {
        let program = program_of(
            &[
                &[
                    Op::Call {
                        name: 1,
                        argc: 0,
                        op: None,
                        capture_parent_scope: false,
                    },
                    Op::Pop,
                    Op::LoadThis,
                    Op::Return,
                ],
                &[Op::Unit, Op::Return],
            ],
            Vec::new(),
        );

        let mut this = Dynamic::from(7 as INT);
        assert_eq!(call(&program, Some(&mut this)).unwrap().as_int(), Ok(7));
    }

    /// A chain rooted at `this` whose method is a chunk of ours: the receiver
    /// becomes the callee's `this`, which is the binding Rhai does at
    /// `func/call.rs:649-655` and the only place a method call differs from a
    /// plain one.
    #[test]
    fn a_method_step_reaching_a_chunk_binds_the_receiver() {
        let program = program_with_chains(
            // `f` is `this.g()`; `g` is `this`.
            &[
                &[
                    Op::Chain {
                        chain: 0,
                        discards: false,
                    },
                    Op::Return,
                ],
                &[Op::LoadThis, Op::Return],
            ],
            Vec::new(),
            vec![Chain {
                root: Root::This {
                    pos: Position::NONE,
                },
                steps: vec![Step::Method {
                    name: 1, // `g`
                    argc: 0,
                    operand: 0,
                    flags: Default::default(),
                    pos: Position::NONE,
                }],
                tail: Tail::Read,
                operands: 0,
            }],
        );

        let mut this = Dynamic::from(7 as INT);
        assert_eq!(call(&program, Some(&mut this)).unwrap().as_int(), Ok(7));
    }

    /// And a write inside that callee travels back out through both frames: the
    /// callee's register, the chain's root write-back, then the host's pointer.
    #[test]
    fn a_write_inside_a_method_step_reaches_the_host() {
        let program = program_with_chains(
            // `f` is `this.g()`; `g` is `this = 9`.
            &[
                &[
                    Op::Chain {
                        chain: 0,
                        discards: false,
                    },
                    Op::Return,
                ],
                &[
                    Op::Const(0),
                    Op::AssignThis { op: None },
                    Op::Unit,
                    Op::Return,
                ],
            ],
            vec![Dynamic::from(9 as INT)],
            vec![Chain {
                root: Root::This {
                    pos: Position::NONE,
                },
                steps: vec![Step::Method {
                    name: 1, // `g`
                    argc: 0,
                    operand: 0,
                    flags: Default::default(),
                    pos: Position::NONE,
                }],
                tail: Tail::Read,
                operands: 0,
            }],
        );

        let mut this = Dynamic::from(1 as INT);
        assert!(call(&program, Some(&mut this)).is_ok());
        assert_eq!(this.as_int(), Ok(9));
    }

    /// A fragment the compiler could not lower still sees the receiver. Without
    /// this the walker would be handed `None` and report `ErrorUnboundThis` for
    /// a `this` the surrounding instructions can read perfectly well.
    #[test]
    fn a_residual_fragment_reads_the_frames_receiver() {
        let program = program_with(
            &[&[
                Op::EvalAst {
                    residual: 0,
                    rewind_scope: false,
                },
                Op::Return,
            ]],
            Vec::new(),
            Vec::new(),
            vec![Expr::ThisPtr(Position::NONE)],
        );

        let mut this = Dynamic::from(7 as INT);
        assert_eq!(call(&program, Some(&mut this)).unwrap().as_int(), Ok(7));
    }

    /// A chain rooted at `this` mutates the caller's value rather than a copy.
    /// That is the whole reason `Root::This` is not `Root::Temporary`.
    #[test]
    #[cfg(not(feature = "no_index"))]
    fn a_chain_rooted_at_this_mutates_the_hosts_value() {
        let program = program_with_chains(
            &[&[
                Op::Const(0),
                Op::Chain {
                    chain: 0,
                    discards: false,
                },
                Op::Return,
            ]],
            vec![Dynamic::from(2 as INT)],
            vec![Chain {
                root: Root::This {
                    pos: Position::NONE,
                },
                steps: vec![Step::Method {
                    name: 2, // `push`
                    argc: 1,
                    operand: 0,
                    flags: Default::default(),
                    pos: Position::NONE,
                }],
                tail: Tail::Read,
                operands: 1,
            }],
        );

        let mut this = Dynamic::from(vec![Dynamic::from(1 as INT)]);
        assert!(call(&program, Some(&mut this)).is_ok());

        let array = this.into_array().expect("still an array");
        let items: Vec<INT> = array.iter().map(|v| v.as_int().unwrap()).collect();
        assert_eq!(items, vec![1, 2]);
    }

    /// `f(this, ..)` is Rhai's method-call rewrite, so the receiver goes by
    /// reference and a mutating native reaches the caller's value.
    #[test]
    #[cfg(not(feature = "no_index"))]
    fn this_as_a_first_argument_goes_by_reference() {
        let program = one(
            &[
                Op::LoadThis,
                Op::Const(0),
                Op::CallRef {
                    name: 2, // `push`
                    argc: 2,
                    receiver: Receiver::This,
                    capture_parent_scope: false,
                },
                Op::Return,
            ],
            vec![Dynamic::from(2 as INT)],
        );

        let mut this = Dynamic::from(vec![Dynamic::from(1 as INT)]);
        if let Err(err) = call(&program, Some(&mut this)) {
            panic!("expected the push to succeed, got {err:?}");
        }

        let items: Vec<INT> = this
            .into_array()
            .expect("still an array")
            .iter()
            .map(|v| v.as_int().unwrap())
            .collect();
        assert_eq!(items, vec![1, 2]);
    }

    /// The snapshot is pushed before the other arguments, so an unbound
    /// receiver is what fails — not whatever the arguments would have done.
    #[test]
    fn an_unbound_this_beats_a_failing_argument() {
        let program = one(
            &[
                Op::LoadThis,
                Op::LoadNamed(3), // `len`, which is no variable
                Op::CallRef {
                    name: 2,
                    argc: 2,
                    receiver: Receiver::This,
                    capture_parent_scope: false,
                },
                Op::Return,
            ],
            Vec::new(),
        );

        let err = *call(&program, None).unwrap_err();
        assert!(
            format!("{err:?}").contains("ErrorUnboundThis"),
            "expected the receiver to fail first, got {err:?}"
        );
    }

    #[test]
    fn a_chain_rooted_at_an_unbound_this_is_an_error() {
        let program = program_with_chains(
            &[&[
                Op::Chain {
                    chain: 0,
                    discards: false,
                },
                Op::Return,
            ]],
            Vec::new(),
            vec![Chain {
                root: Root::This {
                    pos: Position::NONE,
                },
                steps: vec![Step::Method {
                    name: 3, // `len`
                    argc: 0,
                    operand: 0,
                    flags: Default::default(),
                    pos: Position::NONE,
                }],
                tail: Tail::Read,
                operands: 0,
            }],
        );

        let err = *call(&program, None).unwrap_err();
        assert!(
            format!("{err:?}").contains("ErrorUnboundThis"),
            "expected an unbound `this`, got {err:?}"
        );
    }

    #[test]
    fn a_coalescing_step_short_circuits_on_unit() {
        let program = program_with_chains(
            &[&[
                Op::Unit,
                Op::Chain {
                    chain: 0,
                    discards: false,
                },
                Op::Return,
            ]],
            Vec::new(),
            vec![Chain {
                root: Root::Temporary,
                steps: vec![Step::Method {
                    name: 3, // `len`
                    argc: 0,
                    operand: 0,
                    flags: crate::grain::bytecode::StepFlags::SKIP_IF_UNIT,
                    pos: Position::NONE,
                }],
                tail: Tail::Read,
                operands: 0,
            }],
        );

        let out = call(&program, None).expect("coalescing should skip method dispatch on unit");
        assert!(out.is_unit(), "expected unit, got {out:?}");
    }

    #[test]
    fn a_read_only_receiver_refuses_the_write_and_is_left_alone() {
        let program = one(
            &[
                Op::Const(0),
                Op::AssignThis { op: None },
                Op::Unit,
                Op::Return,
            ],
            vec![Dynamic::from(9 as INT)],
        );
        let mut this = Dynamic::from(1 as INT).into_read_only();
        let err = *call(&program, Some(&mut this)).unwrap_err();
        assert!(
            matches!(
                &err,
                EvalAltResult::ErrorInFunctionCall(_, _, inner, _)
                    // Named for an expression that has no name.
                    if matches!(&**inner, EvalAltResult::ErrorAssignmentToConstant(name, ..) if name.is_empty())
            ),
            "expected a refused write to a constant, got {err:?}"
        );
        assert_eq!(this.as_int(), Ok(1));
    }
}
