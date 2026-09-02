//! Reaching a compiled chunk from inside a native function.
//!
//! `[1, 2, 3].map(|x| x * 2)` leaves this VM in the middle of a call. `map` is
//! Rhai's, and the pointer it calls back is resolved by Rhai's dispatch, which
//! looks in `global.lib` and the engine's modules. Our chunks are in neither:
//! [`Op::Call`](crate::bytecode::Op::Call) finds one by a name *index* that
//! only the compiler and the call site share, and a `FnPtr` carries a string.
//!
//! So a program that hands a pointer out registers one native wrapper per
//! compiled function for the length of the run. Direct dispatch is untouched —
//! this is somewhere for Rhai to look, not somewhere we look.
//!
//! # Two ways in
//!
//! Rhai reaches its own closures through a pointer carrying the body, which is
//! called without being resolved at all, and a pointer that carries only a name
//! is what the wrappers exist to be found by. A closure a program hands out
//! carries its body too ([`pointer`]), so the wrappers are reached only by a
//! pointer that cannot: one built from a runtime string names whatever the
//! string says, and one binding `this` is given its receiver in a place a chunk
//! does not look.
//!
//! # What being a native costs
//!
//! A crossing is a boundary out of the VM and back into a second one, which the
//! walker does not pay — it stays inside itself and reaches the closure body
//! directly. `native callbacks` in `examples/grain_bench.rs` is the one case
//! the VM loses, and a crossing costs 4 call levels against the walker's 2.
//!
//! The second `Vm` is per *element*, because that is how often a native calls
//! back: `[..500 elements].map(|x| x * 2)` crosses five hundred times. Building
//! one is not what that costs — what it costs is that everything a `Vm` exists
//! to accumulate starts empty each time, above all the resolution cache. So the
//! crossings of one run share a pool of the parts a finished crossing can hand
//! to the next: see [`Crossings`] for where it lives and [`Warm`] for what may
//! travel in it and what may not.
//!
//! None of this touches a pointer called directly from compiled code, which is
//! `Op::CallFnPtr` and never comes through here.

use core::any::TypeId;
use core::mem;
#[cfg(feature = "no_std")]
use std::prelude::v1::*;

use crate::types::fn_ptr::FnPtrType;
use crate::{
    func::{FnCallArgs, RhaiFunc},
    Dynamic, FnArgsVec, FuncRegistration, Locked, Module, NativeCallContext, Shared, SmartString,
};

use super::{malformed, Vm, VmResult, Warm};
use crate::grain::program::SharedProgram;

/// The parts finished crossings of one run left behind, for its next crossing.
///
/// A pool rather than a slot, for the reason [`Vm::take_scope`] is one: a
/// crossing can be live while another begins. `a.map(|x| x.map(|y| y))` has two
/// at once, and so does anything a callback calls that calls back again — so
/// what is lent has to be taken out and given back rather than borrowed from a
/// place both could reach. Nothing is ever lent twice, because a take removes
/// it.
///
/// It grows to the deepest crossings have nested and no further, which is what
/// `max_call_levels` bounds wherever `unchecked` is off.
///
/// # Where this lives
///
/// In the run's `GlobalRuntimeState`, beside `grain_faults` and for the same
/// reason: that is the one thing a crossing is handed. `Vm::reentrant` clones
/// it from the [`NativeCallContext`], so every crossing of a run — including
/// one nested inside another, and one reached through a wrapper rather than
/// through a pointer — finds the same pool, and a crossing arriving from a run
/// that has none finds none.
///
/// The alternatives are worse in a way that is about correctness rather than
/// taste. A pool captured by the [`pointer`] closure would outlive its run: a
/// closure handed back to the host is called again later, and by then the
/// engine may have been registered into — `Engine::register_fn` wants the
/// engine by mutable reference, which only holds while no run is using it. A
/// resolution cache that survived across that gap could answer "no such
/// function" for one that now exists. Held by the run, a pool cannot span a
/// registration, because the run holds the engine by shared reference for
/// exactly as long as the pool exists. A thread-local has the same problem and
/// adds one this crate does not otherwise have.
pub(crate) struct Crossings {
    warm: Vec<Warm>,
}

/// What a run hands its crossings, or [`None`] where a run pools nothing.
pub(crate) type CrossingPool = Shared<Locked<Crossings>>;

/// A pool for a run that is about to install the wrappers.
#[must_use]
pub(super) fn pool() -> CrossingPool {
    Shared::new(Locked::new(Crossings { warm: Vec::new() }))
}

/// Whatever an earlier crossing of this run left, or nothing carried at all.
///
/// A pool that cannot be locked lends nothing rather than waiting: what is on
/// the other side is another crossing of this run holding it open, and a cold
/// `Warm` is the same answer a moment earlier would have given.
#[must_use]
fn take(pool: Option<&CrossingPool>) -> Warm {
    pool.and_then(|pool| crate::func::native::locked_write(pool)?.warm.pop())
        .unwrap_or_else(Warm::new)
}

/// Put a finished crossing's parts back for the next one.
fn give(pool: Option<&CrossingPool>, warm: Warm) {
    let Some(pool) = pool else {
        return;
    };
    if let Some(mut pool) = crate::func::native::locked_write(pool) {
        pool.warm.push(warm);
    }
}

/// The most parameters a wrapper is registered for.
///
/// A wrapper takes `Dynamic` throughout, and Rhai only reaches a `Dynamic`
/// parameter by permuting the call's own argument types towards it — a search
/// it caps at `MAX_DYNAMIC_PARAMETERS`, 16 (`func/call.rs:235`). A wider
/// wrapper would be registered and never found, so it is left out rather than
/// silently dead. Direct dispatch has no such bound; this limits only what a
/// native can call back into.
const MAX_PARAMS: usize = 16;

/// A wrapper per compiled function, for Rhai to resolve a pointer against.
///
/// Built once per run rather than cached on the program: the closures hold the
/// program, so anything the program held back would be a cycle.
pub(super) fn wrappers(program: &SharedProgram) -> Module {
    let mut module = Module::new();

    // What Rhai reports as the source of a function it found here. Its own
    // script library is the AST's, so this is the same string by the same
    // route.
    if let Some(source) = program.source() {
        module.set_id(source.clone());
    }

    for function in program.functions() {
        let arity = function.params.len();
        if arity > MAX_PARAMS {
            continue;
        }
        // A `this`-taking chunk cannot be reached this way. A wrapper is
        // registered at one arity, and how many arguments Rhai asks for depends
        // on what the *native* appends beside the receiver — `map` adds an
        // index, `reduce` adds the running result — which the wrapper has no way
        // to know. Rhai's own pointer carries the body and sizes the call from
        // its declared arity (`types/fn_ptr.rs:501-535`); a name-only pointer,
        // which is all a wrapper can be, cannot. So these are left to Rhai,
        // and `Program::needs_walker` is what keeps its copy alive for them.
        if function.takes_this {
            continue;
        }
        let Some(name) = program.name(function.name) else {
            continue;
        };

        let owner = program.clone();
        let name_index = function.name;
        let called: SmartString = name.into();

        // One closure for every arity, rather than the fixed-arity shapes
        // `Module::set_native_fn` generates. `Dynamic` parameters throughout
        // mean the types never have to line up, only the count.
        let wrapper = move |context: Option<NativeCallContext>, args: &mut FnCallArgs| {
            // Registered with `has_context`, so Rhai always supplies one.
            let context = context
                .ok_or_else(|| malformed("a callback wrapper was given no context".into()))?;
            // Taken rather than cloned, as every registered function does: the
            // arguments are the caller's to give away, and it has already
            // copied anything it still needs — the dispatch that resolved this
            // registration copied the first itself, because it reached it as a
            // call by reference (`func/call.rs:439`).
            let values = args.iter_mut().map(|arg| mem::take(*arg)).collect();
            invoke(&owner, name_index, &called, &context, values)
        };

        FuncRegistration::new(name)
            .in_internal_namespace()
            .set_into_module_raw(
                &mut module,
                vec![TypeId::of::<Dynamic>(); arity],
                RhaiFunc::Pure {
                    func: Shared::new(wrapper),
                    has_context: true,
                    is_pure: true,
                    is_volatile: false,
                },
            );
    }

    module
}

/// The pointer a compiled closure is handed out as.
///
/// A pointer that carries its body is called without being resolved at all
/// (`types/fn_ptr.rs:463`), which is what Rhai's own closures get and a name is
/// not: the wrappers below are somewhere for the dispatch to *find* a chunk,
/// and finding is the whole cost this skips. They stay all the same, for the
/// pointers that cannot carry a body — one built from a runtime string
/// (`Op::MakeFnPtr`) names whatever the string says.
///
/// [`None`] for a chunk that binds `this`, and for a name that is not a
/// compiled function at all. Rhai hands a receiver to a body-carrying pointer
/// as its first argument (`types/fn_ptr.rs:480`), which is not where a chunk
/// expecting `this` looks for one, so those are left to the walker's copy
/// exactly as [`wrappers`] leaves them.
pub(super) fn pointer(program: &SharedProgram, index: u32, name: &str) -> Option<FnPtrType> {
    // By index rather than by name, which is the same question asked without a
    // comparison: a function's name *is* an index into the program's pool.
    let mut found = false;
    for function in program.functions() {
        if function.name != index {
            continue;
        }
        if function.takes_this {
            return None;
        }
        found = true;
    }
    if !found {
        return None;
    }

    // What [`wrappers`] registers, reached directly instead of through a
    // lookup. No arity bound here: a bound exists there because Rhai has to
    // *find* the registration, and nothing is being found.
    let owner = program.clone();
    let name_index = index;
    let called: SmartString = name.into();
    Some(FnPtrType::Native(Shared::new(
        move |context: NativeCallContext, args: &mut FnCallArgs| {
            // The first argument is copied rather than taken, which is what
            // the dispatch does for the wrapper above and what nothing does
            // here: a body-carrying pointer is called directly
            // (`types/fn_ptr.rs:490`), so no registration is resolved and no
            // `ArgBackup` stands between the chunk and the caller's own value.
            // `map` hands over an element of the array it is mapping, and a
            // chunk that took it would leave a unit behind in the array.
            //
            // Only the first: the rest are the native's own temporaries, and
            // where there is no receiver at all the first is one too, so the
            // copy costs a `Dynamic` clone and is never wrong.
            let values = args
                .iter_mut()
                .enumerate()
                .map(|(at, arg)| match at {
                    0 => (**arg).clone(),
                    _ => mem::take(*arg),
                })
                .collect();
            invoke(&owner, name_index, &called, &context, values)
        },
    )))
}

/// Run one chunk for a native that called back into us.
///
/// The values are the caller's to build, because how much of them it may take
/// depends on how it was reached: see [`wrappers`] and [`pointer`].
fn invoke(
    program: &SharedProgram,
    index: u32,
    name: &str,
    context: &NativeCallContext,
    values: FnArgsVec<Dynamic>,
) -> VmResult {
    // Held across the call rather than looked up twice: the run this crossing
    // came out of is what owns the pool, and it is still running underneath.
    let pool = super::crossing_pool(context);
    let mut vm = Vm::reentrant_from(context, take(pool));
    // The share this crossing came out of, so a pointer the chunk creates
    // carries its body too. See [`pointer`].
    vm.callbacks = Some(program.clone());
    let result = vm.call_function_at(
        program,
        index,
        name,
        values,
        context.call_level(),
        context.call_position(),
    );
    // However the call ended. A crossing that raised leaves its parts as
    // empty as one that returned — [`Vm::execute`] floors the three stacks on
    // both paths and the frame truncates the operands — and the resolution
    // work it did is worth no less for having been followed by an error.
    give(pool, vm.cool());
    result
}

#[cfg(test)]
#[cfg(not(feature = "no_function"))]
mod tests {
    use super::*;
    use crate::grain::Compiler;
    use crate::{Engine, FnPtr, Position, Scope};

    /// The value a source runs to, with the wrappers installed.
    fn value_of(source: &str) -> Dynamic {
        let engine = Engine::new();
        let ast = engine.compile(source).unwrap();
        let program = Compiler::new().compile(&ast).into_shared();
        assert_eq!(
            program.residual_count(),
            0,
            "`{source}` still fragments, so the pointer is not ours to hand out",
        );
        Vm::new(&engine)
            .eval_with_callbacks(&mut Scope::new(), &program)
            .unwrap()
    }

    fn carries_its_body(value: Dynamic) -> bool {
        let pointer = value.try_cast::<FnPtr>().expect("a function pointer");
        matches!(pointer.typ, FnPtrType::Native(..))
    }

    /// A closure is handed out carrying its body, which is what spares a native
    /// the lookup on every element it calls back over.
    ///
    /// Asserted at the type because nothing a caller can see says which kind of
    /// pointer it got: both spell themselves `Fn`, both answer the same call
    /// with the same value, and the one that does not carry a body still
    /// resolves — against the wrappers this module registers beside it. So a
    /// test written through behaviour would pass either way. See [`pointer`].
    #[test]
    fn a_closure_is_handed_out_carrying_its_body() {
        assert!(carries_its_body(value_of("|x| x * 2")));
    }

    /// One chunk called the way a crossing calls it, over and over, with the
    /// parts of each call handed to the next.
    ///
    /// Reaches [`invoke`]'s two halves without a native in between, which is
    /// what lets a test ask what the parts *are* rather than only what the
    /// script answered.
    fn crossings(source: &str, name: &str, times: usize, carry: bool) -> (Vec<u64>, Warm) {
        let engine = Engine::new();
        let ast = engine.compile(source).unwrap();
        let program = Compiler::new().compile(&ast).into_shared();
        let index = program
            .functions()
            .iter()
            .find(|function| program.name(function.name) == Some(name))
            .expect("the source declares it")
            .name;

        let global = engine.new_global_runtime_state();
        let context: NativeCallContext = (&engine, name, None, &global, Position::NONE).into();

        let mut generations = Vec::new();
        let mut warm = Warm::new();
        for turn in 0..times {
            let mut vm = Vm::reentrant_from(&context, warm);
            vm.callbacks = Some(program.clone());
            let mut values = FnArgsVec::new();
            values.push(Dynamic::from(turn as crate::INT));
            drop(
                vm.call_function_at(&program, index, name, values, 0, Position::NONE)
                    .expect("the chunk runs"),
            );
            generations.push(vm.last_generation);
            warm = vm.cool();
            if !carry {
                warm = Warm::new();
            }
        }
        (generations, warm)
    }

    /// The counter that names frames travels with the tables it stamps.
    ///
    /// This is what makes a lent memo unreadable rather than wrong: a crossing
    /// that inherited the tables must not be handed a number an entry of the
    /// crossing before it already carries. The cold column is the same run with
    /// nothing carried, and its repetition is exactly what would be a hit.
    #[test]
    fn a_carried_memo_never_sees_a_generation_twice() {
        let source = "fn f(x) { x * 2 }";
        let (carried, ..) = crossings(source, "f", 4, true);
        let (cold, ..) = crossings(source, "f", 4, false);

        assert!(carried.windows(2).all(|pair| pair[0] < pair[1]), "{carried:?}");
        assert_eq!(cold, vec![cold[0]; cold.len()], "{cold:?}");
    }

    /// And the resolution cache is what actually carries.
    ///
    /// A chunk that reaches Rhai's dispatch fills one layer; a crossing that
    /// inherits it starts from that layer rather than from nothing, which is
    /// the whole cost the module doc names.
    #[test]
    fn a_crossing_hands_on_the_resolution_cache_it_filled() {
        assert_eq!(Warm::new().caches.fn_resolution_caches_len(), 0);
        let (.., warm) = crossings("fn f(x) { abs(0 - x - 1) }", "f", 2, true);
        assert_eq!(warm.caches.fn_resolution_caches_len(), 1);
    }

    /// A cache is refused to a crossing that does not search what filled it.
    ///
    /// Nothing a script can do reaches this — a pool belongs to one run and a
    /// run holds one engine — so the guard is asserted where it is decided
    /// rather than through a program.
    #[test]
    fn a_cache_filled_against_one_engine_is_not_lent_to_another() {
        let (.., warm) = crossings("fn f(x) { abs(0 - x - 1) }", "f", 2, true);
        assert_eq!(warm.caches.fn_resolution_caches_len(), 1);

        let elsewhere = Engine::new();
        let global = elsewhere.new_global_runtime_state();
        let context: NativeCallContext = (&elsewhere, "f", None, &global, Position::NONE).into();
        let vm = Vm::reentrant_from(&context, warm);
        assert_eq!(vm.caches.fn_resolution_caches_len(), 0);
    }

    /// Nothing a crossing held is still in what it hands on.
    ///
    /// The operand stack keeps its allocation and gives up every slot, and the
    /// four stacks beside it come back empty however the crossing ended. A
    /// value left in any of them would reach the next crossing as its own.
    #[test]
    #[cfg(not(feature = "no_index"))]
    fn a_crossing_hands_on_no_values() {
        let source = "fn f(x) { let a = [x, x + 1]; let t = x; for i in a { t += i; } t }";
        let (.., warm) = crossings(source, "f", 3, true);
        assert!(warm.stack.iter().all(|slot| super::super::operand_ref(slot).is_unit()));
        assert!(warm.iterators.is_empty());
        assert!(warm.handlers.is_empty());
        assert!(warm.sizes.is_empty());
        assert!(warm.scopes.iter().all(Scope::is_empty));
    }

    /// Including one made inside a callback, which is a second `Vm` holding a
    /// second share of the same program. See [`invoke`].
    #[test]
    #[cfg(not(any(feature = "no_index", feature = "no_object")))]
    fn a_closure_made_inside_a_callback_carries_its_body_too() {
        let value = value_of("let a = [1]; a.map(|x| || 7)");
        let mut array = value.try_cast::<crate::Array>().expect("an array");
        assert!(carries_its_body(array.remove(0)));
    }
}
