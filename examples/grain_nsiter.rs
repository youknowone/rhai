//! Per-iteration cost of the VM against the walker, on the same AST.
//!
//! `grain_bench` answers "is the VM ahead of the tree it replaced", in whole
//! milliseconds and as a ratio. This answers a narrower question: what does one
//! turn of a loop body cost, in nanoseconds. That is the number a fast-path
//! change moves, and the number an interpreter-level target can be written
//! against (`t += i` at or under 20 ns/iter).
//!
//! # Why one process
//!
//! Two separately built binaries measured 38.4 and 61.1 ns/iter on
//! byte-identical trees. Code placement, not code. So every arm here runs in
//! this process, interleaved run by run rather than arm by arm, and the number
//! reported is the fastest sample: noise on a timing is one-sided, so the
//! fastest sample is the least contaminated one and the spread beside it says
//! how contaminated the rest were.
//!
//! The machine's load average is printed before and after, because a ns figure
//! taken on a busy box is not comparable with one taken on an idle box and
//! there is no way to tell them apart after the fact.
//!
//! # Reading it
//!
//! `iters` is the number of loop turns in one sample, so `ns/iter` is the
//! sample divided by it. Cases are written so that everything outside the loop
//! is negligible against the loop; `t = i` is the cheapest body that survives
//! the AST optimizer, so it stands in for the loop frame and the rest can be
//! read net of it.
//!
//! The last line restates the VM's `t += i` against the 20 ns/iter target, so
//! a run answers that question without the reader doing arithmetic.

use std::time::{Duration, Instant};

use rhai::grain::{Compiler, Program, Vm};
use rhai::{Dynamic, Engine, Scope};

const RUNS: usize = 9;

/// The case the 20 ns/iter target is written against.
const GATE_CASE: &str = "t += i (for)";
const TARGET_NS_PER_ITER: f64 = 20.0;

struct Case {
    name: &'static str,
    source: &'static str,
    /// Loop turns per sample. Divided out of the sample to get ns/iter, so it
    /// has to match what the source actually iterates.
    iters: u64,
}

const CASES: &[Case] = &[
    // The gate. A `for` over an exclusive integer range, one op-assignment in
    // the body, both operands locals.
    Case {
        name: "t += i (for)",
        source: "let t = 0; for i in 0..200000 { t += i; } t",
        iters: 200_000,
    },
    // Same arithmetic, hand-rolled induction variable. The difference between
    // this and the one above is the range iterator against an explicit compare
    // and increment, which is two more op-assignments of its own.
    Case {
        name: "t += i (while)",
        source: "let t = 0; let i = 0; while i < 200000 { t += i; i += 1; } t",
        iters: 200_000,
    },
    // The cheapest body that survives, and therefore the floor row: a single
    // local store. An empty body is not usable as a floor — the AST optimizer
    // deletes the whole loop, and both arms compile from the optimized AST, so
    // `for i in 0..200000 { }` is 14 bytes of bytecode and times as zero. The
    // gap between this row and `t += i` is the binary-operator dispatch alone,
    // which is what a typed arithmetic opcode would be removing.
    Case {
        name: "t = i",
        source: "let t = 0; for i in 0..200000 { t = i; } t",
        iters: 200_000,
    },
    // Same shape, constant right-hand side: says whether reading the loop
    // variable as an operand costs anything over reading a literal.
    Case {
        name: "t += 1",
        source: "let t = 0; for i in 0..200000 { t += 1; } t",
        iters: 200_000,
    },
    // The float bank. Same op-assignment, different builtin pair.
    Case {
        name: "f += 1.5",
        source: "let f = 0.0; for i in 0..200000 { f += 1.5; } f",
        iters: 200_000,
    },
    // A comparison feeding a branch, which is the other half of what a hot
    // loop does and resolves through the same builtin path as the arithmetic.
    Case {
        name: "if i > 0 { t += 1 }",
        source: "let t = 0; for i in 0..200000 { if i > 0 { t += 1; } } t",
        iters: 200_000,
    },
    // One level of script-function call per turn, so the per-call frame cost
    // is visible next to the per-op cost.
    Case {
        name: "t = add(t, i)",
        source: "fn add(a, b) { a + b } let t = 0; for i in 0..50000 { t = add(t, i); } t",
        iters: 50_000,
    },
    // One indexed read and one indexed write per turn. Chains leave the VM
    // through the shared indexing path, so this is the case a typed opcode
    // cannot reach.
    Case {
        name: "a[i % 64] += 1",
        source: "let a = []; a.pad(64, 0); for i in 0..50000 { a[i % 64] += 1; } a[0]",
        iters: 50_000,
    },
];

/// One arm's samples, fastest first.
struct Timing {
    fastest: Duration,
    median: Duration,
}

impl Timing {
    fn ns_per_iter(&self, iters: u64) -> f64 {
        self.fastest.as_secs_f64() * 1e9 / iters as f64
    }

    fn spread(&self) -> f64 {
        self.median.as_secs_f64() / self.fastest.as_secs_f64() - 1.0
    }
}

fn summarize(mut samples: Vec<Duration>) -> Timing {
    samples.sort_unstable();
    Timing {
        fastest: samples[0],
        median: samples[samples.len() / 2],
    }
}

/// The three-field load average, as the kernel reports it.
///
/// Shelling out rather than linking `libc`: this runs twice per process, at the
/// ends, so the cost is irrelevant and the dependency would not be.
fn load_average() -> String {
    std::process::Command::new("uptime")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|line| {
            line.split("load average")
                .nth(1)
                .map(|rest| rest.trim_start_matches([':', 's', ' ']).trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn main() {
    let engine = Engine::new();

    // The walker short-circuits binary operators to builtin function pointers
    // when `fast_operators` is on, which is the default. The third arm turns
    // that off, so the column says how much of the walker's lead is that one
    // fast path — which is the same fast path a typed opcode would be buying
    // for the VM.
    let mut slow_engine = Engine::new();
    slow_engine.set_fast_operators(false);

    println!("load average before: {}", load_average());
    println!("runs: {RUNS}, reporting the fastest\n");
    println!(
        "{:<22} {:>10} {:>12} {:>12} {:>12} {:>9} {:>8}",
        "", "iters", "walker ns", "vm ns", "walker-slow", "speedup", "spread"
    );

    let mut gate = None;

    for case in CASES {
        let ast = engine.compile(case.source).expect("must compile");
        let slow_ast = slow_engine.compile(case.source).expect("must compile");
        let program: Program = Compiler::new().compile(&ast);

        let mut run_walker = || {
            engine
                .eval_ast_with_scope::<Dynamic>(&mut Scope::new(), &ast)
                .expect("walker must succeed")
        };
        let mut run_vm = || {
            Vm::new(&engine)
                .eval_with_scope(&mut Scope::new(), &program)
                .expect("vm must succeed")
        };
        let mut run_slow = || {
            slow_engine
                .eval_ast_with_scope::<Dynamic>(&mut Scope::new(), &slow_ast)
                .expect("walker must succeed")
        };

        // Same answer from every arm, or the timings are measuring different
        // programs. Doubles as the warm-up.
        let expected = format!("{:?}", run_walker());
        for (arm, got) in [
            ("vm", format!("{:?}", run_vm())),
            ("walker-slow", format!("{:?}", run_slow())),
        ] {
            assert_eq!(expected, got, "{} disagreed with the walker on {}", arm, case.name);
        }

        // Interleaved by run, not by arm: a thermal or scheduling drift that
        // lands on one half of the process must not land on one arm.
        let mut walker = Vec::with_capacity(RUNS);
        let mut vm = Vec::with_capacity(RUNS);
        let mut slow = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            for (samples, run) in [
                (&mut walker, &mut run_walker as &mut dyn FnMut() -> Dynamic),
                (&mut vm, &mut run_vm),
                (&mut slow, &mut run_slow),
            ] {
                let start = Instant::now();
                let value = run();
                samples.push(start.elapsed());
                let _ = std::hint::black_box(value);
            }
        }

        let walker = summarize(walker);
        let vm = summarize(vm);
        let slow = summarize(slow);

        if case.name == GATE_CASE {
            gate = Some(vm.ns_per_iter(case.iters));
        }

        println!(
            "{:<22} {:>10} {:>11.1} {:>11.1} {:>11.1} {:>8.2}x {:>7.0}%",
            case.name,
            case.iters,
            walker.ns_per_iter(case.iters),
            vm.ns_per_iter(case.iters),
            slow.ns_per_iter(case.iters),
            walker.ns_per_iter(case.iters) / vm.ns_per_iter(case.iters),
            vm.spread() * 100.0,
        );
    }

    println!("\nload average after:  {}", load_average());

    // Restated rather than gated: this example is a probe, not a CI check, and
    // the number it is compared against is a target for the VM to reach rather
    // than a floor it has already cleared.
    if let Some(gate) = gate {
        println!(
            "\n`t += i` on the VM: {gate:.1} ns/iter against a {TARGET_NS_PER_ITER:.0} ns/iter target ({})",
            if gate <= TARGET_NS_PER_ITER { "met" } else { "not met" },
        );
    }
}
