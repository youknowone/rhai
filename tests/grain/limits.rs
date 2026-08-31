//! Compiled code must still be stoppable.
//!
//! Rhai enforces `max_operations` and the `on_progress` interrupt from
//! `Engine::track_operation`, which the tree walker calls per AST node. A VM
//! that never called it would turn `loop {}` from a script the engine
//! terminates into one that hangs the host — a safety regression, not a
//! performance one, which is why `track_operation` is in the patch.
//!
//! These live outside the differential corpus on purpose. The walker ticks per
//! node and the VM charges one operation per backward transfer, so the
//! operation *counts* differ and always will. What must hold is that the limit
//! fires and the interrupt is honoured, so that is what is asserted — not
//! parity of counts or positions.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rhai::grain::bytecode::Op;
use rhai::grain::{Compiler, Program, Vm};
use rhai::{Dynamic, Engine, EvalAltResult, Scope};

/// A bare infinite loop, which the compiler lowers with nothing left over —
/// asserted below, so this cannot silently become a test of the fallback.
const SPIN: &str = "loop { }";

fn run_vm(engine: &Engine, source: &str) -> Result<Dynamic, Box<EvalAltResult>> {
    let ast = engine.compile(source).expect("must compile");
    let program = Compiler::new().compile(&ast);

    assert_eq!(program.residual_count(), 0, "{source:?} must be fully lowered, or this tests Rhai rather than the VM",);

    // Nothing in the code meters: the charge is on the backward transfer, and
    // without one in the chunk the tests below would hang rather than fail.
    assert!(!program.main().ops(program.code()).any(|(_, op)| op == Op::Tick), "{source:?} lowered with a metering instruction, which this compiler does not emit",);
    assert!(program.main().ops(program.code()).any(|(at, op)| matches!(op, Op::Jump(target) if target as usize <= at)), "{source:?} lowered to a loop with no backward jump to charge",);

    Vm::new(engine).eval_with_scope(&mut Scope::new(), &program)
}

#[test]
fn compiled_loop_hits_the_operation_limit() {
    let mut engine = Engine::new();
    engine.set_max_operations(10_000);

    let err = run_vm(&engine, SPIN).expect_err("an unbounded loop must be stopped");

    assert!(matches!(*err, EvalAltResult::ErrorTooManyOperations(..)), "expected ErrorTooManyOperations, got {err:?}",);
}

#[test]
fn compiled_loop_honours_the_progress_interrupt() {
    let ticks = Arc::new(AtomicU64::new(0));
    let seen = ticks.clone();

    let mut engine = Engine::new();
    engine.on_progress(move |count| {
        seen.store(count, Ordering::SeqCst);
        // Stand-in for a host's abort flag.
        (count >= 500).then(|| Dynamic::from("terminated"))
    });

    let err = run_vm(&engine, SPIN).expect_err("the interrupt must stop the loop");

    assert!(matches!(*err, EvalAltResult::ErrorTerminated(..)), "expected ErrorTerminated, got {err:?}",);
    assert!(ticks.load(Ordering::SeqCst) >= 500, "on_progress should have been called on every back-edge",);
}

/// A chunk that loops with no metering instruction in it must still be stopped.
///
/// An artifact is not required to have come from this compiler. A cycle whose
/// instructions all do ordinary work still verifies — the jump is in range, the
/// stack balances, every path reaches a `Return` — and runs forever, which
/// would make a hostile file unanswerable.
///
/// So the budget cannot depend on the producer having been generous: the VM
/// charges an operation for every *backward* transfer, and a cycle always has
/// one. Written against a hand-built chunk rather than a compiled one because
/// what is under test is the artifact a loader accepts, not the lowering.
/// Found by `mutated_artifacts_load_or_fail_but_never_misbehave`, which hung on
/// a mutation rather than failing.
#[test]
fn a_loop_with_no_metering_instruction_still_hits_the_limit() {
    let mut engine = Engine::new();
    engine.set_max_operations(10_000);

    let ast = engine.compile(SPIN).expect("must compile");
    let program = Compiler::new().compile(&ast);

    // Through the artifact rather than straight from the compiler, so what runs
    // is what a loader would have accepted off disk.
    let bytes = program.write().expect("a lowered program must write");
    let loaded = Program::read(&bytes).expect("a valid artifact");
    assert!(!loaded.main().ops(loaded.code()).any(|(_, op)| op == Op::Tick), "nothing in the chunk may meter, or this tests the instruction rather than the transfer",);

    let err = Vm::new(&engine)
        .eval_with_scope(&mut Scope::new(), &loaded)
        .expect_err("a loop with nothing metering it must still be stopped");
    assert!(matches!(*err, EvalAltResult::ErrorTooManyOperations(..)), "expected ErrorTooManyOperations, got {err:?}",);
}

/// A turn of a loop is charged exactly one operation.
///
/// Two instructions used to meter the same turn: one the lowering put at the
/// loop header and the backward transfer that closes it. A budget a host sets
/// is then spent at twice the rate the loop runs, which stops a script the
/// setting was chosen to allow — so the rate is pinned here rather than left to
/// follow the shape of the lowering.
#[test]
fn a_turn_of_a_loop_is_charged_once() {
    let counted = Arc::new(AtomicU64::new(0));
    let seen = counted.clone();

    let mut engine = Engine::new();
    engine.on_progress(move |count| {
        seen.store(count, Ordering::SeqCst);
        None
    });

    let source = "let s = 0; let i = 0; while i < 100 { s += i; i += 1; } s";
    let ast = engine.compile(source).expect("must compile");
    let program = Compiler::new().compile(&ast);
    assert_eq!(program.residual_count(), 0, "must be lowered, not walked");

    Vm::new(&engine).eval_with_scope(&mut Scope::new(), &program).expect("a bounded loop must finish");

    assert_eq!(counted.load(Ordering::SeqCst), 100, "one hundred turns, one charge each",);
}

/// The walker and the VM must agree that the script *fails*, even though they
/// disagree about after how many operations.
#[test]
fn the_walker_agrees_the_loop_is_stopped() {
    let mut engine = Engine::new();
    engine.set_max_operations(10_000);

    let ast = engine.compile(SPIN).expect("must compile");
    let err = engine.eval_ast_with_scope::<Dynamic>(&mut Scope::new(), &ast).expect_err("rhai must stop it too");

    assert!(matches!(*err, EvalAltResult::ErrorTooManyOperations(..)), "expected ErrorTooManyOperations, got {err:?}",);
}

/// `max_string_size` is a host's defense, and interpolation is the easiest way
/// to walk past it — Rhai checks the running total after *every* segment
/// rather than once at the end, so a script cannot build a huge string and
/// hand it over.
///
/// The position is checked too, because it is the one thing a single
/// instruction might not be able to reproduce: Rhai blames the segment that
/// tipped the total over, and the VM has one position-table entry per
/// instruction.
#[test]
fn interpolation_respects_the_string_limit() {
    let mut engine = Engine::new();
    engine.set_max_string_size(16);

    let source = r#"let a = "0123456789"; `${a}${a}${a}`"#;
    let ast = engine.compile(source).expect("must compile");
    let program = Compiler::new().compile(&ast);
    assert_eq!(program.residual_count(), 0, "must be lowered, not walked");

    let walker = engine.eval_ast_with_scope::<Dynamic>(&mut Scope::new(), &ast).expect_err("the walker must refuse it");
    let vm = Vm::new(&engine).eval_with_scope(&mut Scope::new(), &program).expect_err("and so must the VM");

    assert!(matches!(*vm, EvalAltResult::ErrorDataTooLarge(..)), "got {vm:?}",);
    assert_eq!(format!("{vm:?}"), format!("{walker:?}"), "including the position of the segment that went over",);
}

/// A loop that does terminate must not be killed by the tick itself, and must
/// still produce the value Rhai produces.
#[test]
fn ticking_does_not_disturb_a_bounded_loop() {
    let mut engine = Engine::new();
    engine.set_max_operations(10_000);

    let source = "let i = 0; loop { i += 1; if i > 100 { break i; } }";
    let ast = engine.compile(source).expect("must compile");

    let program = Compiler::new().compile(&ast);
    let vm = Vm::new(&engine).eval_with_scope(&mut Scope::new(), &program).expect("bounded loop must finish");

    let walker = engine.eval_ast_with_scope::<Dynamic>(&mut Scope::new(), &ast).expect("bounded loop must finish under Rhai too");

    assert_eq!(format!("{vm:?}"), format!("{walker:?}"));
}

/// A call costs the same call level here as it does in the walker.
///
/// `max_call_levels` is a budget a host sets against the *walker*, so a VM that
/// spends it faster stops a program the host's setting was chosen to allow.
/// Reaching Rhai's dispatch through `_call_fn_raw` spent two levels for one
/// call: that entry exists for a native reentering the evaluator and takes the
/// crossing as a frame of its own, on top of the one `exec_fn_call` takes for
/// the callee. The walker has no crossing to count and enters at
/// `exec_fn_call` (`func/call.rs` `make_function_call`), so the VM enters the
/// same way, through `dispatch_fn`.
///
/// The callee is a script function in a *registered module* rather than one in
/// the program, because only `call_script_fn` checks the budget
/// (`func/script.rs:42`) and only a callee the compiler did not lower is
/// reached through the dispatch at all — `Op::Call` answers its own functions
/// without leaving the VM, and a native never checks the budget however many
/// levels it spends.
///
/// Measured as the smallest budget each side runs at rather than asserted at a
/// number, for `a_callback_costs_more_call_levels`'s reason: the number belongs
/// to Rhai's dispatch and would drift. What has to hold is that it is one
/// number and not two.
#[test]
#[cfg(not(any(feature = "unchecked", feature = "no_function", feature = "no_module")))]
fn a_dispatched_call_costs_the_walkers_call_levels() {
    // Recursive, so the budget is reached by the callee rather than by
    // anything around it, and computed rather than constant so the optimizer
    // cannot fold the call away.
    const LIBRARY: &str = "fn down(n) { if n <= 0 { 0 } else { down(n - 1) } }";
    const SOURCE: &str = "let n = 3; down(n)";

    let engine_with_library = |levels: usize| {
        let mut engine = Engine::new();
        let ast = engine.compile(LIBRARY).expect("the library parses");
        let module = rhai::Module::eval_ast_as_new(Scope::new(), &ast, &engine).expect("the library builds");
        engine.register_global_module(module.into());
        engine.set_max_call_levels(levels);
        engine
    };

    let cheapest = |run: &dyn Fn(&Engine) -> bool| {
        let budgets = 1..=32;
        budgets.into_iter().find(|levels| run(&engine_with_library(*levels))).unwrap_or(usize::MAX)
    };

    let walker = cheapest(&|engine| {
        let ast = engine.compile(SOURCE).expect("must compile");
        engine.eval_ast::<Dynamic>(&ast).is_ok()
    });
    let vm = cheapest(&|engine| {
        let ast = engine.compile(SOURCE).expect("must compile");
        let program = Compiler::new().compile(&ast);
        assert_eq!(program.residual_count(), 0, "{SOURCE:?} must be fully lowered");
        Vm::new(engine).eval_with_scope(&mut Scope::new(), &program).is_ok()
    });

    assert!(walker > 1 && walker < 32, "the walker must be stopped by the budget somewhere measurable, not at {walker}",);
    assert_eq!(
        vm, walker,
        "a dispatched call runs at {walker} call level(s) in the walker and {vm} here, \
         so a budget a host set against the walker does not buy the same program",
    );
}
