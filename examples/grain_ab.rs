//! One binary, two spellings, alternating legs.
//!
//! Grading a change by building two binaries compares two code layouts as well
//! as two spellings, and on a loaded machine the layout term is the same size
//! as the effect (`examples/grain_nsiter.rs` records two builds of a
//! byte-identical tree reading 38.4 and 61.1 ns/iter). This binary compiles
//! both spellings and picks between them with `grain::ab_gate`, so the only
//! thing that differs between the legs is which spelling runs.
//!
//! A case that never reaches a gated site is a control: both legs run the same
//! machine code down the same path and must read 1.000. If a control does not,
//! the round is noise and the rest of it says nothing.

use std::time::{Duration, Instant};

use rhai::grain::{ab_gate, Compiler, Program, Vm};
use rhai::{Dynamic, Engine, Scope, Shared};

/// One side of the comparison: a program and the entry point its capabilities
/// require.
enum Leg {
    /// Boxed only so the two variants are the same size; nothing reads it per
    /// instruction.
    Owned(Box<Program<'static>>),
    /// What a program whose function pointers can escape into a native needs.
    Shared(Shared<Program<'static>>),
}

impl Leg {
    fn of(program: Program<'static>, callbacks: bool) -> Self {
        if callbacks {
            Self::Shared(program.into_shared())
        } else {
            Self::Owned(Box::new(program))
        }
    }

    fn run(&self, engine: &Engine) -> Dynamic {
        let mut scope = Scope::new();
        match self {
            Self::Owned(program) => Vm::new(engine).eval_with_scope(&mut scope, program),
            Self::Shared(program) => Vm::new(engine).eval_with_callbacks(&mut scope, program),
        }
        .expect("vm must succeed")
    }
}

/// Script runs per leg per round.
const RUNS: usize = 9;

struct Case {
    name: &'static str,
    source: &'static str,
    iterations: usize,
    /// Whether the run needs the callback wrappers installed, which costs an
    /// owned program and a module built per run.
    callbacks: bool,
    /// Whether the source reaches a gated site. A case that does not is a
    /// control and must read 1.000 — so with no candidate gated, every case
    /// is one, and a round where they all do is the harness reporting itself
    /// healthy rather than a result.
    gated: bool,
}

const CASES: &[Case] = &[
    Case { name: "tight integer loop", source: "let s = 0; let i = 0; while i < 20000 { s += i; i += 1; } s", iterations: 20, callbacks: false, gated: false },
    Case { name: "float arithmetic", source: "let x = 0.0; let i = 0; while i < 20000 { x += (i.to_float() * 1.5) / 2.5; i += 1; } x", iterations: 20, callbacks: false, gated: false },
    Case { name: "script fn calls", source: "fn add(a, b) { a + b } let s = 0; for i in 0..5000 { s = add(s, i); } s", iterations: 20, callbacks: false, gated: false },
    Case { name: "recursive fibonacci", source: "fn fib(n) { if n < 2 { n } else { fib(n-1) + fib(n-2) }} fib(28)", iterations: 1, callbacks: false, gated: false },
    Case {
        name: "switch, 4 arms",
        source: "let s = 0; for i in 0..20000 { switch i % 4 { 0 => s += 1, 1 => s += 2, 2 => s += 3, _ => s += 4 } } s",
        iterations: 20,
        callbacks: false,
        gated: false,
    },
    Case {
        name: "switch, 16 arms",
        source: "let s = 0; for i in 0..20000 { switch i % 16 { \
                 0 => s += 1, 1 => s += 2, 2 => s += 3, 3 => s += 4, \
                 4 => s += 5, 5 => s += 6, 6 => s += 7, 7 => s += 8, \
                 8 => s += 9, 9 => s += 10, 10 => s += 11, 11 => s += 12, \
                 12 => s += 13, 13 => s += 14, 14 => s += 15, _ => s += 16 } } s",
        iterations: 20,
        callbacks: false,
        gated: false,
    },
    Case {
        name: "branch heavy",
        source: "let s = 0; for i in 0..20000 { if i % 3 == 0 { s += 1; } else if i % 3 == 1 { s += 2; } else { s -= 1; } } s",
        iterations: 20,
        callbacks: false,
        gated: false,
    },
    Case { name: "native function calls", source: "let a = 42; for i in 0..20000 { a = abs(abs(abs(abs(a)))); } a", iterations: 20, callbacks: false, gated: false },
    Case {
        name: "primes",
        source: "const SIZE = 1_000_000; let prime_mask = []; prime_mask.pad(SIZE + 1, true); \
                 prime_mask[0] = false; prime_mask[1] = false; let total_primes_found = 0; \
                 for p in 2..=SIZE { if !prime_mask[p] { continue; } total_primes_found += 1; \
                 for i in range(2 * p, SIZE + 1, p) { prime_mask[i] = false; } } total_primes_found",
        iterations: 1,
        callbacks: false,
        gated: false,
    },
    // Index reads and nothing else, so the gated site is priced per element
    // here rather than diluted by a sieve's inner write loop.
    Case {
        name: "indexed reads",
        source: "let a = []; a.pad(64, 1); let s = 0; \
                 for i in 0..20000 { s += a[i % 64]; } s",
        iterations: 20,
        callbacks: false,
        gated: false,
    },
    // The one case the VM is expected to lose: every element crosses out of it
    // into a second `Vm` for the closure body, so a call-path change is priced
    // per element here and nowhere else.
    Case {
        name: "native callbacks",
        source: "let a = []; for i in 0..500 { a.push(i); } \
                 let b = a.map(|x| x * 2); b.filter(|x| x % 3 == 0).len",
        iterations: 20,
        callbacks: true,
        gated: false,
    },
    Case {
        name: "while, no operator fold",
        source: "let s = 0; let i = 0; let n = 20000; while `${i}` != `${n}` { s += i; i += 1; } s",
        iterations: 2,
        callbacks: false,
        gated: false,
    },
];

fn once(run: &mut impl FnMut()) -> Duration {
    let start = Instant::now();
    run();
    start.elapsed()
}

/// The least contaminated sample: noise on a duration is one-sided.
fn best(samples: &[Duration]) -> f64 {
    samples
        .iter()
        .min()
        .expect("a leg takes a sample")
        .as_secs_f64()
}

fn median(samples: &[Duration]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2].as_secs_f64()
}

/// Alternate the two legs one script run at a time, swapping which leads.
fn paired(
    samples: usize,
    mut a: impl FnMut(),
    mut b: impl FnMut(),
) -> (Vec<Duration>, Vec<Duration>) {
    let mut left = Vec::with_capacity(samples);
    let mut right = Vec::with_capacity(samples);
    for sample in 0..samples {
        if sample % 2 == 0 {
            left.push(once(&mut a));
            right.push(once(&mut b));
        } else {
            right.push(once(&mut b));
            left.push(once(&mut a));
        }
    }
    (left, right)
}

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
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(5);
    let engine = Engine::new();

    println!("rounds={rounds} runs_per_round={RUNS}");
    println!("load average at start: {}", load_average());

    // Compiled once, outside the timed region: the legs differ in which
    // spelling runs, not in the work of producing the program. One program
    // serves both legs, because the two spellings compile to the same bytes.
    let mut cases = Vec::new();
    for case in CASES {
        let ast = engine.compile(case.source).expect("must compile");
        let leg = Leg::of(Compiler::new().compile(&ast), case.callbacks);

        let expected = engine
            .eval_ast_with_scope::<Dynamic>(&mut Scope::new(), &ast)
            .expect("walker must succeed");
        for (which, on) in [("A", false), ("B", true)] {
            ab_gate::set(on);
            let actual = leg.run(&engine);
            assert_eq!(
                format!("{expected:?}"),
                format!("{actual:?}"),
                "{} disagreed on leg {which}",
                case.name,
            );
        }
        println!(
            "{:<24} {}",
            case.name,
            if case.gated {
                "reaches the gate"
            } else {
                "control"
            }
        );
        cases.push((case, leg));
    }

    println!();
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8}",
        "", "round", "ratio", "spread", "worse?"
    );
    let mut ratios: Vec<(&str, bool, Vec<f64>)> = cases
        .iter()
        .map(|(case, _)| (case.name, case.gated, Vec::new()))
        .collect();

    for round in 0..rounds {
        for (index, (case, leg)) in cases.iter().enumerate() {
            let samples = RUNS * case.iterations;
            // The store lands inside the timed region, and both legs pay one.
            let (a, b) = paired(
                samples,
                || {
                    ab_gate::set(false);
                    drop(leg.run(&engine));
                },
                || {
                    ab_gate::set(true);
                    drop(leg.run(&engine));
                },
            );
            let ratio = best(&b) / best(&a);
            let spread = (median(&b) / best(&b) - 1.0).max(median(&a) / best(&a) - 1.0);
            println!(
                "{:<24} {:>8} {:>8.4} {:>7.0}% {:>8}",
                case.name,
                round,
                ratio,
                spread * 100.0,
                if ratio > 1.0 { "yes" } else { "" }
            );
            ratios[index].2.push(ratio);
        }
        println!();
    }

    println!("load average at end: {}", load_average());
    println!();
    println!(
        "{:<24} {:>8} {:>9} {:>9} {:>16}",
        "", "kind", "median", "best", "rounds B won"
    );
    for (name, differs, samples) in &ratios {
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let won = samples.iter().filter(|ratio| **ratio < 1.0).count();
        println!(
            "{:<24} {:>8} {:>9.4} {:>9.4} {:>13}/{}",
            name,
            if *differs { "effect" } else { "control" },
            sorted[sorted.len() / 2],
            sorted[0],
            won,
            samples.len(),
        );
    }
}
