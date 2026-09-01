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
//! A crossing is a boundary out of the VM and back into a second one whose
//! resolution cache starts empty, which the walker does not pay — it stays
//! inside itself and reaches the closure body directly. `native callbacks` in
//! `examples/grain_bench.rs` measures 0.87x, the one case the VM loses, and a
//! crossing costs 4 call levels against the walker's 2.
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
    Dynamic, FnArgsVec, FuncRegistration, Module, NativeCallContext, Shared, SmartString,
};

use super::{malformed, Vm, VmResult};
use crate::grain::program::SharedProgram;

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
    let mut vm = Vm::reentrant(context);
    // The share this crossing came out of, so a pointer the chunk creates
    // carries its body too. See [`pointer`].
    vm.callbacks = Some(program.clone());
    vm.call_function_at(
        program,
        index,
        name,
        values,
        context.call_level(),
        context.call_position(),
    )
}

#[cfg(test)]
#[cfg(not(feature = "no_function"))]
mod tests {
    use super::*;
    use crate::grain::Compiler;
    use crate::{Engine, FnPtr, Scope};

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
