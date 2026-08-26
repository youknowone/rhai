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

/// The same for the one unary operator that has a typed instruction.
///
/// `Op::UnOp` runs `!` on a `bool` without resolving a function, so it must sit
/// behind the same gate the walker's own unary short-circuit does
/// (`func/call.rs` `eval_fn_call_expr`, which takes it only under
/// `fast_operators` and only for `Token::Bang`). With the gate off both sides
/// dispatch and a registered `!` wins.
///
/// The second half is again what stops this being a tautology, and the third
/// case is the control: `-` has no typed instruction and no short-circuit on
/// either side, so a registration for it must win at *both* settings.
#[test]
fn a_registered_unary_operator_follows_rhais_own_gate() {
    // Answers no built-in could: `!` is the identity on a `bool`, so a `!true`
    // that dispatches is `true` and one that short-circuits is `false`. And
    // `-` on an integer adds one, which no negation does.
    fn engine_with_operators(fast: bool) -> Engine {
        let mut engine = corpus::engine();
        engine.set_fast_operators(fast);
        engine.register_fn("!", |x: bool| x);
        engine.register_fn("-", |x: rhai::INT| x + 1);
        engine
    }

    // The first two witness the gate. `!!b` cannot and is here only for the
    // agreement above: applying an identity twice and negating twice are the
    // same answer, so it reads the same at both settings however it ran.
    const GATED: &[&str] = &["let b = true; !b", "let b = false; if !b { 1 } else { 0 }", "let b = true; !!b"];
    const OBSERVABLE: usize = 2;
    // No short-circuit exists for `-` on either side, so the gate does not
    // apply and the registration runs at both settings.
    const UNGATED: &[&str] = &["let i = 7; -i"];

    let fast = engine_with_operators(true);
    let slow = engine_with_operators(false);

    for source in GATED.iter().chain(UNGATED) {
        assert_eq!(run_stock(&fast, source), run_vm(&fast, source), "fast operators: the VM disagreed with Rhai on `{source}`",);
        assert_eq!(run_stock(&slow, source), run_vm(&slow, source), "dispatched operators: the VM disagreed with Rhai on `{source}`",);
    }

    // The gate is observable, so agreeing above meant something.
    for source in &GATED[..OBSERVABLE] {
        assert_ne!(run_stock(&fast, source), run_stock(&slow, source), "Rhai itself must answer `{source}` differently at the two settings, or the agreement above is vacuous",);
        assert_ne!(run_vm(&fast, source), run_vm(&slow, source), "the VM must answer `{source}` differently at the two settings, or it is not reading the gate at all",);
    }

    // And `-` is not behind it, on either side. A VM that had given every
    // unary operator a typed instruction would fail here rather than above.
    for source in UNGATED {
        assert_eq!(run_stock(&fast, source), run_stock(&slow, source), "Rhai does not gate `-`, so `{source}` must answer the same at both settings",);
        assert_eq!(run_vm(&fast, source), run_vm(&slow, source), "the VM must not gate `-` either",);
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

/// An assigned value that goes straight onto the stack still lands in Rhai's
/// evaluation order.
///
/// Rhai evaluates the assigned value before the lvalue's own index expressions
/// and method arguments, and `Op::Chain` wants it on top of them — so the
/// compiler stashes it in a local and reads it back. It stops doing that when
/// nothing runs in between, or when the value is a single infallible push that
/// cannot tell. These are the shapes where "cannot tell" has to be true:
/// operands with effects, operands that raise, and a value that is not one of
/// those pushes and therefore keeps the stash.
///
/// `no_index` removes `rhai::Array`, which the evaluation log is, and with it
/// every index step there is to order — so the whole test is that build's
/// missing syntax rather than a case it could still make.
#[cfg(not(feature = "no_index"))]
#[test]
fn a_chain_assignment_evaluates_its_value_where_rhai_does() {
    let mut engine = corpus::engine();
    // Records the order things were evaluated in, so the answer carries it.
    engine.register_fn("trace", |log: &mut rhai::Array, tag: rhai::INT| -> rhai::INT {
        log.push(Dynamic::from(tag));
        tag
    });

    const SOURCES: &[&str] = &[
        // Value is a literal, index has an effect: the deferred spelling.
        "let log = []; let a = [0, 0, 0]; a[trace(log, 1)] = 7; [a, log]",
        "let log = []; let a = [0, 0, 0]; a[trace(log, 1)] += 7; [a, log]",
        // Value is not a literal, so the stash stays and has to still be right.
        "let log = []; let a = [0, 0, 0]; let v = 9; a[trace(log, 1)] = v; [a, log]",
        "let log = []; let a = [0, 0, 0]; a[trace(log, 1)] = trace(log, 2); [a, log]",
        "let log = []; let a = [0, 0, 0]; a[trace(log, 1)] += trace(log, 2); [a, log]",
        // Two operands, so their order matters as well as the value's.
        "let log = []; let a = [[0, 0], [0, 0]]; a[trace(log, 1)][trace(log, 2)] = 5; [a, log]",
        "let log = []; let a = [[0, 0], [0, 0]]; a[trace(log, 1)][trace(log, 2)] = trace(log, 3); [a, log]",
        // The operand raises. With the value stashed it was evaluated first, so
        // this pins that a value which cannot raise does not change the answer.
        "let a = [0]; a[99] = 1; a",
        // A property step evaluates nothing at all, which is the other reason
        // the stash goes.
        "let m = #{ x: 0 }; m.x = 7; m",
        "let m = #{ x: 0 }; m.x += 7; m",
        "let m = #{ x: 0 }; let v = 7; m.x = v; m",
        "let m = #{ x: #{ y: 0 } }; m.x.y = 7; m",
        // A method step with arguments does evaluate between.
        "let log = []; let a = [[1, 2, 3]]; a[0].remove(trace(log, 0)); [a, log]",
        // Mixed property and index.
        "let log = []; let m = #{ a: [0, 0] }; m.a[trace(log, 1)] = 4; [m, log]",
        // Every value kind that is treated as a single push.
        r#"let a = [0]; a[0] = "s"; a"#,
        "let a = [0]; a[0] = 'c'; a",
        "let a = [0]; a[0] = true; a",
        "let a = [0]; a[0] = (); a",
        // A constant array is NOT one of them: building it is what the size
        // limits refuse, so it keeps the stash.
        "let a = [0]; a[0] = [1, 2, 3]; a",
        "let a = [0]; a[0] = #{ k: 1 }; a",
        // A string index into a map, and a bit-field, which are the two places
        // the write goes back through a copy.
        r#"let m = #{}; m["k"] = 1; m"#,
        "let n = 0; n[2] = true; n",
    ];

    // Both halves raise, and which one is reported is part of the answer — so
    // this is the group that says a value evaluated later cannot take the blame
    // off the operand. Excluded under `unchecked`, which removes the guard the
    // case is made of and leaves the operator panicking instead.
    #[cfg(not(feature = "unchecked"))]
    const RAISING: &[&str] = &["let a = [0]; a[99] = 1 / 0; a", "let a = [0]; a[0] = 1 / 0; a", "let a = [0]; a[1 / 0] = 1; a"];
    #[cfg(feature = "unchecked")]
    const RAISING: &[&str] = &[];

    for source in SOURCES.iter().chain(RAISING) {
        assert_eq!(run_stock(&engine, source), run_vm(&engine, source), "the VM disagreed with Rhai on `{source}`",);
    }
}

/// Every operator the typed instruction can run sits behind the same gate.
///
/// [`a_registered_operator_on_primitives_follows_rhais_own_gate`] pins three
/// operators. The typed instruction has seventeen kinds and eleven
/// op-assignment forms, and each is a separate arm of `apply_binary` /
/// `apply_assign` — an arm that ran when it should have dispatched would be
/// invisible to a test that only covers `+`, `<` and `+=`.
///
/// The float arms are here for the same reason and are the sharper case: the
/// module runs float *arithmetic* and deliberately leaves float *comparisons*
/// to dispatch, so a registration on `<` for floats must win at both settings
/// while one on `+` must lose at the fast one.
#[test]
fn every_typed_operator_kind_sits_behind_the_same_gate() {
    use rhai::INT;

    // Answers no built-in gives, and distinct per operator so that a wrong arm
    // shows up as a wrong number rather than a coincidence.
    fn engine_with_operators(fast: bool) -> Engine {
        let mut engine = corpus::engine();
        engine.set_fast_operators(fast);

        macro_rules! int_op {
            ($($name:literal => $tag:literal),* $(,)?) => {
                $(engine.register_fn($name, |x: INT, y: INT| x * 1000 + y * 10 + $tag);)*
            };
        }
        int_op! {
            "+" => 1, "-" => 2, "*" => 3, "/" => 4, "%" => 5, "**" => 6,
            "<<" => 7, ">>" => 8, "&" => 9, "|" => 10, "^" => 11,
        }
        // The comparisons answer the opposite of the truth, which no built-in
        // can be talked into.
        engine.register_fn("==", |x: INT, y: INT| x != y);
        engine.register_fn("!=", |x: INT, y: INT| x == y);
        engine.register_fn("<", |x: INT, y: INT| x > y);
        engine.register_fn("<=", |x: INT, y: INT| x >= y);
        engine.register_fn(">", |x: INT, y: INT| x < y);
        engine.register_fn(">=", |x: INT, y: INT| x <= y);

        macro_rules! int_assign {
            ($($name:literal => $tag:literal),* $(,)?) => {
                $(engine.register_fn($name, |x: &mut INT, y: INT| *x = *x * 1000 + y * 10 + $tag);)*
            };
        }
        int_assign! {
            "+=" => 1, "-=" => 2, "*=" => 3, "/=" => 4, "%=" => 5, "**=" => 6,
            "<<=" => 7, ">>=" => 8, "&=" => 9, "|=" => 10, "^=" => 11,
        }

        #[cfg(not(feature = "no_float"))]
        {
            use rhai::FLOAT;
            engine.register_fn("+", |x: FLOAT, y: FLOAT| x * 1000.0 + y);
            engine.register_fn("*", |x: FLOAT, y: FLOAT| x * 1000.0 + y);
            engine.register_fn("<", |x: FLOAT, y: FLOAT| x > y);
            engine.register_fn("==", |x: FLOAT, y: FLOAT| x != y);
            // The mixed pairs, which reach `impl_float!` through a different
            // arm of the same table.
            engine.register_fn("+", |x: FLOAT, y: INT| x * 1000.0 + y as FLOAT);
            engine.register_fn("+", |x: INT, y: FLOAT| x as FLOAT * 1000.0 + y);
            engine.register_fn("<", |x: FLOAT, y: INT| x > y as FLOAT);
            engine.register_fn("+=", |x: &mut FLOAT, y: FLOAT| *x = *x * 1000.0 + y);
            engine.register_fn("+=", |x: &mut FLOAT, y: INT| *x = *x * 1000.0 + y as FLOAT);
        }

        engine
    }

    // One per registration above. Every one of these is a site where the VM
    // emits the typed instruction.
    const GATED: &[&str] = &[
        "let a = 6; let b = 7; a + b",
        "let a = 6; let b = 7; a - b",
        "let a = 6; let b = 7; a * b",
        "let a = 6; let b = 7; a / b",
        "let a = 6; let b = 7; a % b",
        "let a = 6; let b = 7; a ** b",
        "let a = 6; let b = 2; a << b",
        "let a = 6; let b = 2; a >> b",
        "let a = 6; let b = 7; a & b",
        "let a = 6; let b = 7; a | b",
        "let a = 6; let b = 7; a ^ b",
        "let a = 6; let b = 7; a == b",
        "let a = 6; let b = 7; a != b",
        "let a = 6; let b = 7; a < b",
        "let a = 6; let b = 7; a <= b",
        "let a = 6; let b = 7; a > b",
        "let a = 6; let b = 7; a >= b",
        "let a = 6; a += 7; a",
        "let a = 6; a -= 7; a",
        "let a = 6; a *= 7; a",
        "let a = 6; a /= 7; a",
        "let a = 6; a %= 7; a",
        "let a = 6; a **= 2; a",
        "let a = 6; a <<= 2; a",
        "let a = 6; a >>= 2; a",
        "let a = 6; a &= 7; a",
        "let a = 6; a |= 7; a",
        "let a = 6; a ^= 7; a",
        // The branch consumers, where a wrong answer is a wrong path rather
        // than a wrong number.
        "let a = 6; let b = 7; if a < b { \"lt\" } else { \"ge\" }",
        "let n = 0; let i = 0; while i < 3 { n += 1; i = i + 1 } n",
        #[cfg(not(feature = "no_float"))]
        "let f = 1.5; let g = 2.5; f + g",
        #[cfg(not(feature = "no_float"))]
        "let f = 1.5; let g = 2.5; f * g",
        #[cfg(not(feature = "no_float"))]
        "let f = 1.5; let i = 2; f + i",
        #[cfg(not(feature = "no_float"))]
        "let f = 1.5; let i = 2; i + f",
        #[cfg(not(feature = "no_float"))]
        "let f = 1.5; f += 2.5; f",
        #[cfg(not(feature = "no_float"))]
        "let f = 1.5; f += 2; f",
    ];

    // A float comparison is dispatched by the VM at *both* settings, because
    // the checked built-in's relative-epsilon rule is not inlined — so the
    // registration wins on both sides and the two settings agree. That makes
    // these a control rather than a gate case: they must still match the
    // walker, and they are not counted towards the gate being observable.
    #[cfg(not(feature = "no_float"))]
    const UNGATED: &[&str] = &[
        "let f = 1.5; let g = 2.5; f < g",
        "let f = 1.5; let g = 2.5; f == g",
        "let f = 1.5; let i = 2; f < i",
        // Not two primitives at all: dispatch on both settings.
        "let a = \"x\"; let b = \"y\"; a + b",
    ];
    #[cfg(feature = "no_float")]
    const UNGATED: &[&str] = &["let a = \"x\"; let b = \"y\"; a + b"];

    let fast = engine_with_operators(true);
    let slow = engine_with_operators(false);

    let mut failures = Vec::new();
    let mut observable = 0;

    for (source, gated) in GATED.iter().map(|s| (s, true)).chain(UNGATED.iter().map(|s| (s, false))) {
        let (fast_stock, fast_vm) = (run_stock(&fast, source), run_vm(&fast, source));
        let (slow_stock, slow_vm) = (run_stock(&slow, source), run_vm(&slow, source));

        if fast_stock != fast_vm {
            failures.push(format!("\n  fast operators, `{source}`\n    rhai: {fast_stock:?}\n    vm:   {fast_vm:?}"));
        }
        if slow_stock != slow_vm {
            failures.push(format!("\n  dispatched, `{source}`\n    rhai: {slow_stock:?}\n    vm:   {slow_vm:?}"));
        }
        if gated && fast_stock != slow_stock {
            observable += 1;
        }
    }

    assert!(failures.is_empty(), "{} operator sites disagreed with Rhai:{}", failures.len(), failures.join(""),);

    // Agreeing above is worth nothing unless the registrations were reachable
    // at all. Every gated source must answer differently at the two settings.
    assert_eq!(observable, GATED.len(), "{} of {} gated sources answer the same at both settings, so the gate is not being tested there", GATED.len() - observable, GATED.len(),);
}

/// An arithmetic guard reports the same error, at the same position, wherever
/// the operator is written.
///
/// The typed instruction returns the built-in's error with no position of its
/// own, which is what Rhai does under fast operators — but "no position" is
/// only right because *nothing else* stamps one either, and what stamps one
/// differs by syntactic slot: a chain tail, an op-assignment and a bare
/// expression each take a different path out of the dispatch loop.
///
/// Excluded under `unchecked`, which removes the guard these are made of and
/// leaves the operator panicking in a test build instead.
#[cfg(not(feature = "unchecked"))]
#[test]
fn an_arithmetic_guard_reports_the_same_error_and_position_in_every_slot() {
    let engine = corpus::engine();

    // Written without a literal at the type's limit so that `only_i32` and
    // `only_i64` reach the same overflow.
    const OVERFLOW: &str = "let a = 1; let n = 0; while n < 200 { a += a; n += 1 } a";

    let sources: Vec<String> = vec![
        // Bare expression, top level.
        "1 / 0".into(),
        "let z = 0; 7 % z".into(),
        "2 ** -1".into(),
        OVERFLOW.into(),
        // Behind a `let`, which is a different consumer.
        "let x = 1 / 0; x".into(),
        // As an op-assignment, which takes the `store` path rather than the
        // operator one.
        "let a = 7; let z = 0; a /= z; a".into(),
        "let a = 7; let z = 0; a %= z; a".into(),
        "let a = 2; a **= -1; a".into(),
        // Inside a script function, which is another frame.
        "fn f(x) { x / 0 } f(1)".into(),
        "fn f(x) { let y = 0; x %= y; x } f(7)".into(),
        // Inside each kind of branch, so the position table entry is reached
        // from a jump rather than from the instruction before it.
        "if true { 1 / 0 } else { 0 }".into(),
        "let n = 0; while n < 3 { n += 1; 1 / 0 } n".into(),
        "let t = 0; for i in 0..3 { t += 1 / 0 } t".into(),
        "switch 1 { 1 => 1 / 0, _ => 0 }".into(),
        // Under an index, where the chain decides which of two failures wins.
        #[cfg(not(feature = "no_index"))]
        "let a = [1]; a[0] / 0".into(),
        // `a[0] /= 0` is NOT here: it diverges on position, and has since
        // before any of this. See
        // `an_op_assignment_at_a_chain_tail_blames_the_chain_step`.
        // Type mismatches, which the typed instruction must decline rather
        // than answer.
        "let a = 1; let b = (); a + b".into(),
        "let a = 1; a += ()".into(),
        #[cfg(not(feature = "no_object"))]
        "let a = 1; a + #{ b: 1 }".into(),
        r#"let a = 1; let b = "x"; a - b"#.into(),
        #[cfg(not(feature = "no_float"))]
        "let a = 1.0; let b = (); a * b".into(),
    ];

    // Caught rather than raised: the handler sees the error object, whose
    // rendering carries the same position — so these still compare it, and are
    // counted separately because they end in `Ok`.
    let caught: &[&str] = &["try { 1 / 0 } catch (e) { e }", "let a = 1; try { a += () } catch (e) { e }"];

    let mut failures = Vec::new();
    let mut errors = 0;

    for source in sources.iter().map(String::as_str).chain(caught.iter().copied()) {
        let stock = run_stock(&engine, source);
        if stock.result.is_err() {
            errors += 1;
        }
        let vm = run_vm(&engine, source);
        if stock != vm {
            failures.push(format!("\n  `{source}`\n    rhai: {:?}\n    vm:   {:?}", stock.result, vm.result));
        }
    }

    assert!(failures.is_empty(), "{} guard sites disagreed with Rhai on the error or its position:{}", failures.len(), failures.join(""),);
    // The comparison above is on `Debug`, which carries the position — so a
    // case that stopped raising would still compare equal and prove nothing.
    assert_eq!(errors, sources.len(), "{} of {} sources stopped raising, so they no longer test a guard", sources.len() - errors, sources.len(),);
}

/// Dropping a discarded statement's unit keeps every table target honest.
///
/// [`dropping_a_statements_value_keeps_every_join_balanced`] covers the joins
/// `Lowering::patched_max` records. A `switch` records its arm targets in the
/// switch *pool* instead, which `patch_to` never sees — so a switch whose
/// trailing instruction is a `Unit` is the shape where the guard could be
/// looking at the wrong thing. These put one in every position that can end a
/// discarded statement.
#[test]
fn a_switch_whose_tail_is_a_unit_keeps_its_table_targets() {
    let engine = corpus::engine();

    let sources: Vec<String> = vec![
        // No default arm, so the lowering emits its own trailing `Unit` for it.
        "let x = 9; switch x { 1 => 1 }; 42".into(),
        "let x = 1; switch x { 1 => 1 }; 42".into(),
        // Guards that decline, so the default is reached through the chain.
        "let x = 1; switch x { 1 if false => 1 }; 42".into(),
        "let x = 1; switch x { 1 if x > 100 => 1, 2 if false => 2 }; 42".into(),
        // A range table as well as a case table.
        "let x = 5; switch x { 0..3 => 1 }; 42".into(),
        "let x = 5; switch x { 0..3 if false => 1 }; 42".into(),
        // The switch is the last statement, so its value is kept rather than
        // discarded — the other side of the same decision.
        "let x = 9; switch x { 1 => 1 }".into(),
        // Nested, so the inner one's tail is followed by the outer's.
        "let x = 9; switch x { 1 => 1, _ => switch x { 2 => 2 } }; 42".into(),
        // A discarded statement whose tail is a unit inside a loop body.
        "let t = 0; for i in 0..3 { switch i { 9 => 1 }; t += 1 } t".into(),
        // And the two other constructs that end in a value through a join.
        "let x = 1; if x > 0 { } else { }; 42".into(),
        "try { } catch (e) { }; 42".into(),
        "let t = 0; while t < 2 { t += 1 }; 42".into(),
    ];

    let mut failures = Vec::new();
    for source in &sources {
        // The verifier is the instrument that would see a dangling table
        // target, so run it before the answer is compared.
        if let Ok(ast) = engine.compile(source) {
            let program = Compiler::new().compile(&ast);
            if let Err(err) = program.verify() {
                failures.push(format!("\n  `{source}` failed verification: {err:?}"));
                continue;
            }
        }
        let stock = run_stock(&engine, source);
        let vm = run_vm(&engine, source);
        if stock != vm {
            failures.push(format!("\n  `{source}`\n    rhai: {:?}\n    vm:   {:?}", stock.result, vm.result));
        }
    }

    assert!(failures.is_empty(), "{} switch tails are wrong:{}", failures.len(), failures.join(""),);
}

/// An op-assignment at a chain tail is blamed on the chain step, not the
/// operator — which is not what Rhai does.
///
/// A KNOWN DEFECT, and this test exists to make fixing it visible rather than
/// to bless it. Reproduced at `68336226` (upstream, before any of the work in
/// this file) and at `69d90efe`, with and without the optimizer, so it is not
/// something the typed operator instruction or the chain-stash elision
/// introduced.
///
/// The cause is that `Tail::Assign` carries no position of its own: a chain is
/// one instruction, `Step` was given a position because a step can fail, and
/// the tail was not. Giving it one is a format change.
///
/// When this test fails, the defect is fixed: delete it and put
/// `let a = [1]; a[0] /= 0; a` back in
/// `an_arithmetic_guard_reports_the_same_error_and_position_in_every_slot`.
#[cfg(not(any(feature = "unchecked", feature = "no_index", feature = "no_position")))]
#[test]
fn an_op_assignment_at_a_chain_tail_blames_the_chain_step() {
    let engine = corpus::engine();

    for source in ["let a = [1]; a[0] /= 0; a", "let a = [1]; let z = 0; a[0] /= z; a"] {
        let stock = run_stock(&engine, source);
        let vm = run_vm(&engine, source);

        // The same failure, so only the blame has moved.
        let (Err(stock_err), Err(vm_err)) = (&stock.result, &vm.result) else {
            panic!("`{source}` must still raise on both sides: {stock:?} / {vm:?}");
        };
        assert!(stock_err.starts_with("ErrorArithmetic(\"Division by zero"), "rhai stopped raising the guard this is about: {stock_err}",);
        assert!(vm_err.starts_with("ErrorArithmetic(\"Division by zero"), "the VM stopped raising the guard this is about: {vm_err}",);
        assert_ne!(stock_err, vm_err, "`{source}` now agrees on the position, so the defect is fixed — see this test's own comment",);
    }
}

/// One operator site, many operand pairs, in one frame and across nested ones.
///
/// The operator memo is keyed on `(frame generation, instruction address,
/// operand discriminant pair)` and is consulted for every pair the typed
/// instruction declines. Three things can go wrong with a key like that and
/// none of them shows up on a monomorphic site: a pair change at one address
/// inside one frame, a nested frame evicting a slot its caller wrote, and an
/// arm whose type is *not* fixed by its discriminant — a shared cell and a
/// custom type, which is why `type_code` answers zero for both and why an
/// answer for one must never be handed to the other.
///
/// Built out of script functions rather than an array, so `no_index` gets the
/// test too.
#[cfg(not(feature = "no_function"))]
#[test]
fn one_operator_site_seeing_many_operand_pairs_agrees_with_rhai() {
    let mut engine = corpus::engine();
    // A custom type, so the site also sees `Union::Variant` — whose type is
    // the boxed value's rather than the arm's.
    #[derive(Clone)]
    struct Tag(rhai::INT);
    engine.register_type_with_name::<Tag>("Tag");
    engine.register_fn("tag", |n: rhai::INT| Tag(n));
    engine.register_fn("==", |a: Tag, b: Tag| a.0 == b.0);
    engine.register_fn("<", |a: Tag, b: Tag| a.0 < b.0);

    // An operand of every arm, selected without an array so that `no_index`
    // still has the test. Six values through one `==` is thirty-six ordered
    // pairs at one instruction, and most of them have no built-in at all —
    // which is an answer the memo has to remember too.
    const PICK: &str = "fn pick(k) { if k == 0 { 1 } else if k == 1 { \"a\" } else if k == 2 { 'c' } else if k == 3 { true } else if k == 4 { () } else { tag(1) } }";

    let mut sources: Vec<String> = vec![
        format!("{PICK} let out = \"\"; for i in 0..6 {{ for j in 0..6 {{ let x = pick(i); let y = pick(j); out += `${{x == y}},` }} }} out"),
        format!("{PICK} let out = \"\"; for i in 0..6 {{ for j in 0..6 {{ let x = pick(i); let y = pick(j); out += `${{x < y}},` }} }} out"),
        // The same site reached from nested frames, so a callee writes the slot
        // the caller is using — and a recursive one, so the generation has to
        // tell two frames at the same address in the same program apart.
        format!("{PICK} fn eq(a, b) {{ a == b }} let out = \"\"; for i in 0..6 {{ for j in 0..6 {{ out += `${{eq(pick(i), pick(j))}},` }} }} out"),
        format!("{PICK} fn walk(i) {{ if i >= 6 {{ \"\" }} else {{ `${{pick(0) == pick(i)}},` + walk(i + 1) }} }} walk(0)"),
    ];

    // A shared cell holds whatever is inside the lock, so its discriminant
    // says nothing about the resolution — the arm the memo must refuse.
    //
    // Kept inside a function so no `FnPtr` is left in the top-level scope: the
    // two sides render a closure's pointer differently there (`Fn*+(..)`
    // against `Fn(..)`), which predates all of this and would be the only
    // thing this case reported.
    //
    // `no_object` takes method-call syntax with it, so `f.call()` does not
    // parse there and the case goes with it.
    #[cfg(all(not(feature = "no_closure"), not(feature = "no_object")))]
    sources.push(r#"fn probe(a, b) { let f = || a; let g = || b; let out = ""; for i in 0..4 { let x = if i % 2 == 0 { a } else { b }; out += `${x == a},${x == b},` } out + `${f.call()}${g.call()}` } probe(1, "a")"#.into());

    // Floats are a seventh arm, and the one the typed instruction runs for
    // arithmetic and declines for comparison — so the two halves of the same
    // site take different paths.
    #[cfg(not(feature = "no_float"))]
    sources.push(format!("{PICK} let out = \"\"; for i in 0..6 {{ let x = pick(i); out += `${{x == 2.5}},${{x < 2.5}},` }} out"));

    let mut failures = Vec::new();
    for source in &sources {
        let stock = run_stock(&engine, source);
        let vm = run_vm(&engine, source);
        // A source that fails to compile on this build would compare equal for
        // the wrong reason and test nothing.
        assert!(stock.result.is_ok(), "`{source}` must run: {:?}", stock.result);
        if stock != vm {
            failures.push(format!("\n  `{source}`\n    rhai: {:?}\n    vm:   {:?}", stock.result, vm.result));
        }
    }

    assert!(failures.is_empty(), "{} polymorphic operator sites disagreed with Rhai:{}", failures.len(), failures.join(""),);
}

/// One `Vm` running two programs must not answer the second out of the first.
///
/// The operator memo is keyed on the instruction's own address, and an address
/// only names an instruction within one program. `Vm::eval_with_scope` takes
/// the program by reference, so the same `Vm` can be handed a second one — and
/// two programs of the same shape put their operator at the same address. The
/// frame generation is the field that keeps those apart; nothing else in the
/// key does.
///
/// The pair below is chosen so the memo is actually consulted: two strings have
/// no typed arm, so the instruction falls through to the resolution the memo
/// caches. `+` and `==` resolve to different functions for that same pair.
#[test]
fn one_vm_running_two_programs_does_not_answer_the_second_out_of_the_first() {
    let engine = corpus::engine();

    // Byte-identical up to the operator, so the two instructions land at the
    // same address. Asserted below rather than assumed.
    let compile = |source: &str| {
        let ast = engine.compile(source).expect("both sources parse");
        Compiler::new().compile(&ast)
    };
    let concat = compile(r#"let a = "x"; let b = "y"; a + b"#);
    let compare = compile(r#"let a = "x"; let b = "y"; a == b"#);

    assert_eq!(concat.code().len(), compare.code().len(), "the two programs must lay out identically, or they do not collide and this proves nothing",);

    let run = |vm: &mut Vm, program: &rhai::grain::Program| {
        let mut scope = Scope::new();
        vm.eval_with_scope(&mut scope, program).map_or_else(|err| format!("{err:?}"), |value| format!("{value:?}"))
    };

    let mut vm = Vm::new(&engine);
    assert_eq!(run(&mut vm, &concat), "\"xy\"", "the first program is the one that fills the memo");
    assert_eq!(run(&mut vm, &compare), "false", "the second program read the first program's operator out of the memo",);

    // And the same in the other order, so a hit in either direction is caught.
    let mut vm = Vm::new(&engine);
    assert_eq!(run(&mut vm, &compare), "false");
    assert_eq!(run(&mut vm, &concat), "\"xy\"", "the second program read the first program's operator out of the memo",);
}
