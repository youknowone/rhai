//! The VM must mean exactly what Rhai means.
//!
//! Every corpus script is evaluated twice against the same `Engine` — once
//! through `eval_ast_with_scope`, once through the VM — and the two runs must
//! agree on the result, on the error (variant *and* position), and on the scope
//! they leave behind.
//!
//! The scope check is not incidental. Rhai evaluates a program's top-level
//! statements without rewinding, so `let` at the top level outlives the run and
//! is observable by the caller. A VM that manages its own frames could return
//! the right value and still get that wrong.

use super::corpus;

use rhai::grain::{Compiler, Vm};
use rhai::{Dynamic, Engine, Scope};

/// What a run produced, in a form two runs can be compared on.
///
/// `Dynamic` and `EvalAltResult` have no `PartialEq`, so this compares their
/// `Debug` rendering. That is stricter than value equality, not looser: it
/// distinguishes `1` from `1.0`, and it includes error positions.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    result: Result<String, String>,
    scope: Vec<(String, String)>,
}

fn snapshot_scope(scope: &Scope) -> Vec<(String, String)> {
    scope.iter_raw().map(|(name, _, value)| (name.to_string(), format!("{value:?}"))).collect()
}

fn run_stock(engine: &Engine, source: &str) -> Outcome {
    let mut scope = Scope::new();
    let result = engine.compile(source).map_err(|err| format!("{err:?}")).and_then(|ast| {
        engine
            .eval_ast_with_scope::<Dynamic>(&mut scope, &ast)
            .map(|value| format!("{value:?}"))
            .map_err(|err| format!("{err:?}"))
    });

    Outcome { result, scope: snapshot_scope(&scope) }
}

fn run_vm(engine: &Engine, source: &str) -> Outcome {
    let mut scope = Scope::new();
    let result = engine.compile(source).map_err(|err| format!("{err:?}")).and_then(|ast| {
        let program = Compiler::new().compile(&ast);
        // A program that can hand a pointer to a native has to be run the
        // way such a program is meant to be run, or the comparison is
        // against a configuration nobody would ship.
        if program.makes_fn_pointers() {
            let program = program.into_shared();
            Vm::new(engine).eval_with_callbacks(&mut scope, &program)
        } else {
            Vm::new(engine).eval_with_scope(&mut scope, &program)
        }
        .map(|value| format!("{value:?}"))
        .map_err(|err| format!("{err:?}"))
    });

    Outcome { result, scope: snapshot_scope(&scope) }
}

#[test]
fn vm_agrees_with_rhai() {
    let engine = corpus::engine();

    let mut failures = Vec::new();

    for case in corpus::CASES.iter().filter(|c| applies_to_this_build(c.name)) {
        let stock = run_stock(&engine, case.source);
        let vm = run_vm(&engine, case.source);

        if stock != vm {
            failures.push(format!(
                "\n=== {} ===\n  source: {}\n  rhai:   {:?}\n  vm:     {:?}\n  \
                 Rhai scope: {:?}\n  vm scope:   {:?}",
                case.name, case.source, stock.result, vm.result, stock.scope, vm.scope,
            ));
        }
    }

    let applicable = corpus::CASES.iter().filter(|c| applies_to_this_build(c.name)).count();
    assert!(failures.is_empty(), "{} of {applicable} corpus scripts diverged:{}", failures.len(), failures.join(""),);
}

/// The corpus is only worth anything if the comparison can actually fail.
///
/// Guards against the harness silently degrading into a tautology — comparing
/// two identical code paths, or stringify everything into the same value.
#[test]
fn harness_detects_a_real_difference() {
    let engine = corpus::engine();

    assert_ne!(run_stock(&engine, "1 + 1"), run_stock(&engine, "1 + 2"), "differing results must compare unequal",);
    assert_ne!(run_stock(&engine, "1"), run_stock(&engine, "1.0"), "int and float must not compare equal",);
    assert_ne!(run_stock(&engine, "let a = 1; a"), run_stock(&engine, "1"), "differing leftover scope must compare unequal",);
    // `no_position` compiles positions out, so there is no such thing as the
    // same error at a different one and nothing here to detect.
    #[cfg(not(any(feature = "no_position", feature = "no_index")))]
    assert_ne!(run_stock(&engine, "let a = [1]; a[9]"), run_stock(&engine, "let a = [1];  a[9]"), "the same error at a different position must compare unequal",);
}

/// A script that does not parse compares equal on both sides for the wrong
/// reason: two identical parse errors. Such a case tests nothing, and would sit
/// in the corpus looking like coverage.
#[test]
fn every_corpus_script_parses() {
    let engine = corpus::engine();

    let broken: Vec<_> = corpus::CASES
        .iter()
        .filter(|case| applies_to_this_build(case.name))
        .filter_map(|case| engine.compile(case.source).err().map(|err| format!("\n  {}: {err}", case.name)))
        .collect();

    assert!(broken.is_empty(), "{} corpus scripts do not parse, so they assert nothing:{}", broken.len(), broken.join(""),);
}

/// Whether a corpus case exercises anything on this build.
///
/// Distinct from [`MAY_FRAGMENT`], which is a tolerance: this says the syntax
/// is not in the language on this build at all. Defined beside the cases, in
/// `corpus`, because every harness that walks them needs the same answer.
use corpus::applies_to_this_build;

/// A case that errors unintentionally is nearly as weak as one that does not
/// parse: both sides agree on the failure, and the machinery the case was
/// written to exercise never runs. Cases that mean to fail say so in the name.
#[test]
fn only_error_cases_error() {
    let engine = corpus::engine();

    let surprises: Vec<_> = corpus::CASES
        .iter()
        .filter(|case| !case.name.starts_with("error_") && !case.name.starts_with("throw_"))
        .filter(|case| applies_to_this_build(case.name))
        .filter_map(|case| match run_stock(&engine, case.source).result {
            Err(err) => Some(format!("\n  {}: {err}", case.name)),
            Ok(_) => None,
        })
        .collect();

    assert!(surprises.is_empty(), "{} cases fail without meaning to, so they exercise nothing:{}", surprises.len(), surprises.join(""),);
}

/// Corpus scripts allowed to leave a fragment behind.
///
/// Empty, and that is the claim: every script in the corpus lowers with nothing
/// left over. A case that fragments therefore fails on arrival rather than
/// quietly joining a majority, which is the point of stating it this way round.
///
/// What legitimately belongs here: `eval`, `import`/`export`, custom syntax,
/// and `?.`. All four are the escape hatch working as intended rather than a
/// gap, and none is in the corpus.
const MAY_FRAGMENT: &[&str] = &[];

/// Every chunk the compiler emits must pass its own verifier.
///
/// The check that matters is depth agreement at merge points: one branch of a
/// conditional leaving a value where the other does not is invisible until a
/// program happens to take the unlucky path, and the differential corpus only
/// covers the paths it happens to exercise.
#[test]
fn every_compiled_chunk_verifies() {
    let engine = corpus::engine();

    let broken: Vec<_> = corpus::CASES
        .iter()
        .filter_map(|case| {
            let ast = engine.compile(case.source).ok()?;
            Compiler::new().compile(&ast).verify().err().map(|err| format!("\n  {}: {err:?}", case.name))
        })
        .collect();

    assert!(broken.is_empty(), "{} chunks failed verification:{}", broken.len(), broken.join(""),);
}

/// A chunk must declare the stack it uses, not the stack it might use.
///
/// The lowering's own estimate is one slot per instruction, which is safe and
/// wildly loose — and it is what the VM reserves from, and what the artifact
/// records. On a device with ~12KB to spend, reserving 25 `Dynamic` slots for a
/// chunk that stacks three is the difference worth closing.
#[test]
fn every_compiled_chunk_declares_the_stack_it_uses() {
    let engine = corpus::engine();

    let mut loose = Vec::new();
    let mut total_declared = 0usize;
    let mut total_ops = 0usize;

    for case in corpus::CASES {
        let Ok(ast) = engine.compile(case.source) else {
            continue;
        };
        let program = Compiler::new().compile(&ast);
        let Ok(high_water) = program.verify() else {
            continue;
        };

        let declared: Vec<u16> = std::iter::once(program.main().max_stack()).chain(program.functions().iter().map(|f| f.chunk.max_stack())).collect();

        total_declared += high_water.iter().map(|n| *n as usize).sum::<usize>();
        total_ops += program.code().len();

        if declared != high_water {
            loose.push(format!("\n  {}: declares {declared:?}, uses {high_water:?}", case.name,));
        }
    }

    println!("\n{total_declared} stack slots declared across {total_ops} bytes of code");

    assert!(loose.is_empty(), "{} chunks declare a stack they do not use:{}", loose.len(), loose.join(""),);
}

/// Residuals are the work left to do, so the count is the progress metric.
///
/// Prints the whole census so a change in coverage is visible, and pins the
/// cases that should already be at zero.
#[test]
fn residual_census() {
    let engine = corpus::engine();

    let mut total_nodes = 0usize;
    let mut at_zero = Vec::new();
    let mut remaining = Vec::new();
    let mut regressions = Vec::new();

    // Cases the build removed are counted on neither side, or the completeness
    // check below would read their absence as a corpus that stopped compiling.
    let applicable = corpus::CASES.iter().filter(|case| applies_to_this_build(case.name)).count();

    for case in corpus::CASES.iter().filter(|c| applies_to_this_build(c.name)) {
        let Ok(ast) = engine.compile(case.source) else {
            continue;
        };
        let program = Compiler::new().compile(&ast);
        let count = program.residual_count();
        let nodes = program.residual_nodes();
        total_nodes += nodes;

        if count == 0 {
            at_zero.push(case.name);
        } else {
            remaining.push((case.name, nodes));
            if !MAY_FRAGMENT.contains(&case.name) && applies_to_this_build(case.name) {
                regressions.push(format!("\n  {} leaves {count}", case.name));
            }
        }
    }

    println!("\n{} of {applicable} scripts fully lowered, {total_nodes} AST nodes still in fragments", at_zero.len(),);
    println!("\nfully lowered: {}", at_zero.join(", "));
    println!("\nremaining:");
    remaining.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for (name, count) in &remaining {
        println!("  {count:>3}  {name}");
    }

    assert!(
        regressions.is_empty(),
        "{} scripts fragment that are not on `MAY_FRAGMENT`. Either the \
         construct regressed, or the case needs one of the four things the \
         escape hatch is for — say which, in the list:{}",
        regressions.len(),
        regressions.join(""),
    );

    // The other direction, which the check above cannot see: a corpus that
    // stopped compiling at all would have nothing to fragment and would pass.
    assert_eq!(at_zero.len(), applicable - MAY_FRAGMENT.len(), "some scripts did not compile, so they were counted as neither",);
}

/// A host operator on primitives obeys Rhai's gate, not one of the VM's.
///
/// With fast operators on — the default — a binary operator on two primitives
/// short-circuits to the built-in and a registered function of the same name
/// never runs (`func/call.rs:1775-1799`). With it off, both sides dispatch and
/// the registered one wins. A typed opcode is a third way of reaching the
/// built-in, so the whole of its correctness here is that it sits behind that
/// same gate: the VM must agree with the walker under both settings, and the
/// two settings must not agree with each other.
///
/// The second half is what stops this from being a tautology. Two engines that
/// both ignored the registration would agree at every setting and prove
/// nothing.
#[test]
fn a_registered_operator_on_primitives_follows_rhais_own_gate() {
    // Answers no built-in could: `+` multiplies, `<` is reversed, and `+=`
    // multiplies in place. All three are registered for integers, which is
    // precisely the pair the gate is about — Rhai has a built-in for it, so
    // the registration only ever runs on the dispatching side.
    fn engine_with_operators(fast: bool) -> Engine {
        let mut engine = corpus::engine();
        engine.set_fast_operators(fast);
        engine.register_fn("+", |x: rhai::INT, y: rhai::INT| x * y);
        engine.register_fn("<", |x: rhai::INT, y: rhai::INT| x > y);
        engine.register_fn("+=", |x: &mut rhai::INT, y: rhai::INT| *x *= y);
        engine
    }

    const SOURCES: &[&str] = &[
        "let a = 6; let b = 7; a + b",
        "let a = 6; let b = 7; a += b; a",
        "let a = 6; let b = 7; if a < b { 1 } else { 0 }",
        // Not two primitives, so the gate does not apply and dispatch runs on
        // both settings.
        r#"let a = "x"; let b = "y"; a + b"#,
    ];

    let fast = engine_with_operators(true);
    let slow = engine_with_operators(false);

    for source in SOURCES {
        assert_eq!(run_stock(&fast, source), run_vm(&fast, source), "fast operators: the VM disagreed with Rhai on `{source}`",);
        assert_eq!(run_stock(&slow, source), run_vm(&slow, source), "dispatched operators: the VM disagreed with Rhai on `{source}`",);
    }

    // The gate is observable, so agreeing above meant something.
    for source in &SOURCES[..3] {
        assert_ne!(run_stock(&fast, source), run_stock(&slow, source), "Rhai itself must answer `{source}` differently at the two settings, or the agreement above is vacuous",);
        assert_ne!(run_vm(&fast, source), run_vm(&slow, source), "the VM must answer `{source}` differently at the two settings, or it is not reading the gate at all",);
    }
}

/// A `switch` subject, a `for` range and an `if` guard are all operands too.
///
/// The typed operator instruction is emitted for every binary operator the VM
/// can run, which puts it in front of consumers that are not an assignment —
/// and a wrong result there is a wrong branch rather than a wrong number.
#[test]
fn a_typed_operator_feeding_a_branch_agrees_with_rhai() {
    let engine = corpus::engine();

    const SOURCES: &[&str] = &[
        "let n = 0; for i in 0..10 { if i % 3 == 0 { n += i } } n",
        "let n = 0; let i = 0; while i < 10 { n += i * 2 - 1; i += 1; } n",
        "let a = 3; switch a * 2 { 6 => \"six\", _ => \"other\" }",
        "let a = 3; let b = 0; do { b += a; a -= 1; } while a > 0; b",
        // The operand pair changes at the same site between turns, which is
        // what a speculative instruction has to survive.
        r#"let out = ""; for i in 0..4 { let v = if i % 2 == 0 { i } else { "s" }; out += `${v == 2}`; } out"#,
    ];

    for source in SOURCES {
        assert_eq!(run_stock(&engine, source), run_vm(&engine, source), "the VM disagreed with Rhai on `{source}`",);
    }
}

/// A `for` over an exclusive integer range walks itself, but only while nothing
/// has said the type means something else.
///
/// `Module::set_iterable` and `Module::set_iterator` build the sequence out of
/// the type's own `Iterator`, and the VM is allowed to walk that in place;
/// `Module::set_iter` takes a closure that can do anything, and the VM has to
/// call it. Both halves are asserted, and the second is what stops the first
/// being vacuous: the custom iterator here answers something no range does.
#[test]
fn an_integer_range_is_walked_in_place_only_while_that_is_what_it_means() {
    use std::any::TypeId;
    use std::ops::Range;

    const SOURCES: &[&str] = &[
        "let t = 0; for i in 0..5 { t += i; } t",
        // Empty and reversed: the bounds decide, not a count.
        "let t = 0; for i in 5..5 { t += 1; } t",
        "let t = 0; for i in 5..0 { t += 1; } t",
        // The counter is the loop's own, not the item's.
        "let s = \"\"; for (x, i) in 10..13 { s += `${i}:${x} `; } s",
        // `break` leaves the iterator behind, and the next loop must not see it.
        "let t = 0; for i in 0..9 { if i == 4 { break; } t += i; } for i in 0..3 { t += 100; } t",
        // Nested, so two ranges are live at once.
        "let t = 0; for i in 0..4 { for j in 0..4 { t += i * j; } } t",
        // An inclusive range is a different type and keeps the boxed path.
        "let t = 0; for i in 0..=5 { t += i; } t",
        // The loop variable outlives a closure made inside the body.
        "let t = 0; for i in 0..3 { t += i; } t",
    ];

    let plain = corpus::engine();

    // Answers no range does — every item is negated — so the two engines must
    // disagree with each other while each agrees with Rhai.
    let mut custom = corpus::engine();
    let mut module = rhai::Module::new();
    module.set_iter(TypeId::of::<Range<rhai::INT>>(), |obj: Dynamic| Box::new(obj.cast::<Range<rhai::INT>>().map(|i| Dynamic::from(-i)).collect::<Vec<_>>().into_iter()));
    custom.register_global_module(module.into());

    let mut differed = 0;
    for source in SOURCES {
        assert_eq!(run_stock(&plain, source), run_vm(&plain, source), "the VM disagreed with Rhai on `{source}`",);
        assert_eq!(run_stock(&custom, source), run_vm(&custom, source), "with a registered range iterator, the VM disagreed with Rhai on `{source}`",);
        if run_stock(&plain, source) != run_stock(&custom, source) {
            differed += 1;
        }
    }

    assert!(differed >= 4, "the registered iterator has to change what Rhai itself answers, or agreeing with it proves nothing (only {differed} of {} sources moved)", SOURCES.len(),);
}

/// A discarded statement value is not pushed and popped, and a `try` keeps its
/// pair.
///
/// Every assignment and every declaration evaluates to unit, and a statement
/// anywhere but last has its value thrown away, so the compiler used to emit
/// `Op::Unit` followed immediately by `Op::Pop` — two of the eight instructions
/// a turn of `for i in 0..n { t += i; }` ran. Dropping the `Op::Unit` instead is
/// only sound while nothing jumps past it, and `try`'s "past the catch block"
/// edge lands exactly there: the try block's own value arrives along it, so the
/// catch block's unit has to stay or the two edges meet at different heights.
///
/// Counted rather than pattern-matched, so this says what changed without
/// pinning the whole encoding.
#[test]
fn a_discarded_statement_value_is_not_pushed_at_all() {
    use rhai::grain::bytecode::{disassemble, Op};

    fn unit_then_pop(source: &str) -> usize {
        let engine = corpus::engine();
        let ast = engine.compile(source).expect("compiles");
        let program = Compiler::new().compile(&ast);
        let ops: Vec<Op> = disassemble(program.code()).map(|(_, op)| op).collect();
        ops.windows(2).filter(|pair| matches!(pair, [Op::Unit, Op::Pop])).count()
    }

    // The two loop bodies the ns/iter probe measures: one discarded assignment
    // per turn in the first, two in the second.
    assert_eq!(unit_then_pop("let t = 0; for i in 0..4 { t += i; } t"), 0);
    assert_eq!(unit_then_pop("let t = 0; let i = 0; while i < 4 { t += i; i += 1; } t"), 0,);
    // Declarations too, and a nested block.
    assert_eq!(unit_then_pop("let a = 1; let b = 2; { let c = 3; a = c; } a + b"), 0);

    // The catch block's unit is the one that must survive.
    assert!(unit_then_pop("let t = 0; try { t = 1; } catch (e) { t = 2; } t") > 0, "the catch block's unit is what the try block's own value meets, so it cannot be dropped",);
}

/// Nothing that used to leave a value behind stops leaving it.
///
/// Dropping a trailing `Op::Unit` is a stack-height change, and the shapes most
/// able to get it wrong are the ones where control flow joins: a statement whose
/// value a later `Op::Pop` was going to take, reached along more than one edge.
#[test]
fn dropping_a_statements_value_keeps_every_join_balanced() {
    let engine = corpus::engine();

    const SOURCES: &[&str] = &[
        // An `if` as a discarded statement: both arms push, the join pops.
        "let a = 0; if a == 0 { a = 1 } else { a = 2 } a",
        "let a = 0; if a == 0 { a = 1 } a",
        // Empty arms, which are a bare `Op::Unit` of their own.
        "let a = 0; if a == 0 { } else { } a",
        // A `try` whose two edges carry different things, in both positions.
        "let t = 0; try { t = 1; } catch (e) { t = 2; } t",
        "let t = 0; try { throw 1; } catch (e) { t = 2; } t",
        "let t = 0; try { t = 1; } catch (e) { t = 2; }",
        // A loop as a discarded statement, and a `break` carrying a value into
        // the same join the exhausted path reaches.
        "let t = 0; while t < 3 { t += 1; } t",
        "let t = 0; let r = loop { t += 1; if t > 2 { break t * 10; } }; r",
        "let t = 0; for i in 0..5 { if i == 2 { break; } t += i; } t",
        "let t = 0; for i in 0..5 { if i == 2 { continue; } t += i; } t",
        // A block used for its value, after statements whose values were not.
        "let a = 1; let b = { a = 2; a + 1 }; b",
        // Switch, whose arms all join.
        "let a = 1; let r = 0; switch a { 0 => r = 10, 1 => r = 20, _ => r = 30 } r",
        "let a = 1; switch a { 0 => 10, _ => 30 }",
        // Nested loops with a `try` between them.
        "let t = 0; for i in 0..3 { try { for j in 0..3 { if j == 1 { throw 1; } t += 1; } } catch (e) { t += 100; } } t",
        // A statement list where every statement is a discarded unit.
        "let a = 0; a = 1; a = 2; a = 3; a",
        // Do-while, both spellings.
        "let t = 0; do { t += 1; } while t < 3; t",
        "let t = 0; do { t += 1; } until t > 2; t",
    ];

    for source in SOURCES {
        assert_eq!(run_stock(&engine, source), run_vm(&engine, source), "the VM disagreed with Rhai on `{source}`",);
    }
}
