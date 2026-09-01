//! Rough VM-versus-walker timings, on the same AST.
//!
//! Indicative, not criterion: repeated runs, reporting the fastest of each and
//! the spread around it so a single scheduling hiccup does not read as a
//! result. Run with `--release`; a debug build measures bounds checks more than
//! anything else.
//!
//! # Catching a regression
//!
//! `cargo run --release --example bench -- --check` exits non-zero if any case
//! has fallen below the floor recorded beside it.
//!
//! What is compared is the **ratio**, not the time. Absolute milliseconds say
//! as much about the machine as about the code, and there is no useful way to
//! commit one; the walker and the VM run back to back on the same machine in
//! the same process, so their ratio mostly divides the machine out. Mostly, not
//! entirely — cache size and core count still move it — so a floor sits about
//! 15% under the observed figure. That is wide enough not to cry wolf on a
//! slower runner and tight enough that losing a fast path shows up.

use std::time::{Duration, Instant};

#[cfg(feature = "grain-jit")]
use rhai::grain::{jit_state, jitcodes};
use rhai::grain::{Compiler, Program, Vm};
use rhai::{Dynamic, Engine, Scope};

const RUNS: usize = 9;

struct Case {
    name: &'static str,
    source: &'static str,
    iterations: usize,
    /// Whether the run needs the callback wrappers installed, which costs an
    /// owned program and a module built per run.
    callbacks: bool,
    /// The speedup this case must not drop below.
    ///
    /// Beside the source rather than in a table of its own, so changing one
    /// without the other is visible in the diff.
    floor: f64,
}

/// One leg's samples, one per script run.
///
/// Held as the whole list rather than a summary because the estimate and the
/// account of how trustworthy it is are two different reductions of it.
struct Leg(Vec<Duration>);

impl Leg {
    /// The least contaminated sample, in seconds.
    ///
    /// Noise on a duration is one-sided -- nothing makes a run finish sooner
    /// than it can -- so the fastest sample is the estimate, and a *ratio* of
    /// two such estimates is a better one than the middle of a list of
    /// per-pair ratios: those multiply both legs' noise together, and a run on
    /// 2026-08-30 that graded that way read one unchanged case at 1.90x,
    /// 3.40x and 2.38x.
    fn secs(&self) -> f64 {
        self.0
            .iter()
            .min()
            .expect("a leg takes at least one sample")
            .as_secs_f64()
    }

    /// How far the middle sample sits above the fastest, as a fraction.
    ///
    /// What this says is whether the fastest sample found a quiet slice. Near
    /// zero, every sample agrees and the estimate is the cost; large, most
    /// samples were contended and only the reader knows whether one of them
    /// escaped.
    fn spread(&self) -> f64 {
        let mut sorted = self.0.clone();
        sorted.sort_unstable();
        sorted[sorted.len() / 2].as_secs_f64() / self.secs() - 1.0
    }
}

const CASES: &[Case] = &[
    Case {
        name: "tight integer loop",
        source: "let s = 0; let i = 0; while i < 20000 { s += i; i += 1; } s",
        iterations: 20,
        callbacks: false,
        floor: 1.30,
    },
    Case {
        name: "float arithmetic",
        source: "let x = 0.0; let i = 0; while i < 20000 { x += (i.to_float() * 1.5) / 2.5; i += 1; } x",
        iterations: 20,
        callbacks: false,
        floor: 1.10,
    },
    Case {
        name: "script fn calls",
        source: "fn add(a, b) { a + b } let s = 0; for i in 0..5000 { s = add(s, i); } s",
        iterations: 20,
        callbacks: false,
        floor: 1.55,
    },
    Case {
        name: "recursive fibonacci",
        source: "fn fib(n) { if n < 2 { n } else { fib(n-1) + fib(n-2) }} fib(28)",
        iterations: 1,
        callbacks: false,
        floor: 1.50,
    },
    // Both sides index a hash: the VM masks it into a table of its own case
    // hashes, Rhai looks it up in a map. Two sizes, because the arms a switch
    // has are what the rest of the statement's cost is measured against.
    Case {
        name: "switch, 4 arms",
        source: "let s = 0; for i in 0..20000 { \
                 switch i % 4 { 0 => s += 1, 1 => s += 2, 2 => s += 3, _ => s += 4 } \
                 } s",
        iterations: 20,
        callbacks: false,
        floor: 1.40,
    },
    Case {
        name: "switch, 16 arms",
        source: "let s = 0; for i in 0..20000 { \
                 switch i % 16 { \
                 0 => s += 1, 1 => s += 2, 2 => s += 3, 3 => s += 4, \
                 4 => s += 5, 5 => s += 6, 6 => s += 7, 7 => s += 8, \
                 8 => s += 9, 9 => s += 10, 10 => s += 11, 11 => s += 12, \
                 12 => s += 13, 13 => s += 14, 14 => s += 15, _ => s += 16 } \
                 } s",
        iterations: 20,
        callbacks: false,
        floor: 1.35,
    },
    Case {
        name: "branch heavy",
        source: "let s = 0; for i in 0..20000 { if i % 3 == 0 { s += 1; } else if i % 3 == 1 { s += 2; } else { s -= 1; } } s",
        iterations: 20,
        callbacks: false,
        floor: 1.45,
    },
    Case {
        name: "native function calls",
        source: "let a = 42; for i in 0..20000 { a = abs(abs(abs(abs(a)))); } a",
        iterations: 5,
        callbacks: true,
        floor: 1.50,
    },
    // The one case the VM is expected to lose. Every element is a boundary out
    // of the VM and back into a second `Vm` with an empty resolution cache —
    // where the walker stays inside itself and reaches the closure body
    // directly. 1000 crossings per iteration, and the crossing is what is
    // left: the pointer itself carries its body, so no element is resolved.
    //
    // Also the only case here that indexes: the `a.push(i)` loop is a chain
    // rooted at a local, so it is what says the root is still being walked
    // where it lives rather than copied out and put back.
    Case {
        name: "native callbacks",
        source: "let a = []; for i in 0..500 { a.push(i); } \
                 let b = a.map(|x| x * 2); b.filter(|x| x % 3 == 0).len",
        iterations: 20,
        callbacks: true,
        floor: 0.80,
    },
    Case {
        name: "primes",
        source: r#"
            const SIZE = 1_000_000;

            let prime_mask = [];
            prime_mask.pad(SIZE + 1, true);

            prime_mask[0] = false;
            prime_mask[1] = false;

            let total_primes_found = 0;

            for p in 2..=SIZE {
                if !prime_mask[p] { continue; }

                total_primes_found += 1;

                for i in range(2 * p, SIZE + 1, p) {
                    prime_mask[i] = false;
                }
            }

            total_primes_found
        "#,
        iterations: 1,
        callbacks: false,
        floor: 1.55,
    },
];

/// Which of the two builds this binary is.
///
/// The merge point the tracer is consulted from sits behind
/// `#[cfg(feature = "grain-jit")]`, so one binary cannot measure both sides.
/// A comparison is therefore two processes, and a run that does not say which
/// build it came from cannot be told from the other afterwards.
/// `grain-jit-check.py` reads this line and refuses to compare two runs that
/// report the same build.
#[cfg(not(feature = "grain-jit"))]
const BUILD: &str = "vm=plain jit=absent";
#[cfg(feature = "grain-jit")]
const BUILD: &str = "vm=jit jit=consulted";

/// The build line, with the lowered table count measured rather than assumed.
///
/// `rhai_grain_jit_tables` says only that an artefact was named at build time;
/// how many jitcodes came out of it is a runtime fact, and a `grain-jit` build
/// that loaded none consults a driver that declines immediately. The two
/// measure different things, so the count is on the line.
fn build_line() -> String {
    #[cfg(feature = "grain-jit")]
    let tables = format!(" jitcodes={}", jitcodes::count());
    #[cfg(not(feature = "grain-jit"))]
    let tables = String::new();
    format!("build: {BUILD}{tables}")
}

fn once(run: &mut impl FnMut()) -> Duration {
    let start = Instant::now();
    run();
    start.elapsed()
}

fn time(samples: usize, mut run: impl FnMut()) -> Leg {
    Leg((0..samples).map(|_| once(&mut run)).collect())
}

/// Sample two legs against each other, alternating one script run at a time.
///
/// Measuring one leg to its end and then the other makes the ratio only as
/// good as the machine holding still in between, and on a busy one it does
/// not: three runs on 2026-08-30 read `float arithmetic` -- whose lowering
/// nothing in that diff touched -- at 1.45x, 2.13x and 0.82x, because each
/// leg met a different minute.
///
/// Alternating per *run* rather than per batch is what matters: it gives the
/// two legs the same load to be unlucky in, and it makes each leg's sample
/// short, so the fastest of many has somewhere quiet to land. Which leg leads
/// alternates too, so a burst beginning mid-pair does not always cost the
/// same one.
fn time_paired(samples: usize, mut walker: impl FnMut(), mut vm: impl FnMut()) -> (Leg, Leg) {
    let mut walked = Vec::with_capacity(samples);
    let mut ran = Vec::with_capacity(samples);

    for sample in 0..samples {
        if sample % 2 == 0 {
            walked.push(once(&mut walker));
            ran.push(once(&mut vm));
        } else {
            ran.push(once(&mut vm));
            walked.push(once(&mut walker));
        }
    }

    (Leg(walked), Leg(ran))
}

/// The three-field load average, as the kernel reports it.
///
/// Printed with every run because a ratio taken on a busy machine is not
/// comparable with one taken on an idle machine, and nothing in the numbers
/// below says which one this was. Shelling out rather than linking `libc`:
/// this runs twice per process, at the ends, so the cost is irrelevant and the
/// dependency would not be.
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
    let check = std::env::args().any(|arg| arg == "--check");
    let engine = Engine::new();

    // Rhai's default options include FAST_OPS, which makes the walker
    // short-circuit binary operators and op-assignments straight to builtin
    // function pointers — no hash, no resolution cache
    // (`func/call.rs:1775-1799`, `eval/stmt.rs:131-148`). Turning it off
    // measures how much of the walker's speed comes from that, and therefore
    // how much of the VM's planned "typed fast opcodes" win is already taken.
    let mut slow_engine = Engine::new();
    slow_engine.set_fast_operators(false);

    println!("{}", build_line());
    println!("load average before: {}", load_average());
    println!(
        "{:<22} {:>11} {:>11} {:>9} {:>7} {:>8} {:>11} {:>10}",
        "", "walker", "vm", "speedup", "floor", "spread", "walker-slow", "fragments"
    );

    let mut below_floor = Vec::new();
    // Collected rather than printed inline, so the table above stays a table.
    #[cfg(feature = "grain-jit")]
    let mut jit_rows: Vec<String> = Vec::new();

    for case in CASES {
        let ast = engine.compile(case.source).expect("must compile");
        let program: Program = Compiler::new().compile(&ast);
        // Owned and shared only where a pointer can escape to a native, so the
        // ordinary cases keep measuring the ordinary path.
        let shared = case
            .callbacks
            .then(|| Compiler::new().compile(&ast).into_shared());
        let run_vm = || match &shared {
            Some(shared) => Vm::new(&engine).eval_with_callbacks(&mut Scope::new(), shared),
            None => Vm::new(&engine).eval_with_scope(&mut Scope::new(), &program),
        };

        // Ahead of the correctness run, not just the timed leg. A warm cell
        // that this case's first VM run already opened and aborted a trace on
        // does not open a second one, so a window that starts after that run
        // reports zero traces for a case that traced.
        #[cfg(feature = "grain-jit")]
        jit_state::reset_stats();

        // Same result, or the comparison is meaningless.
        let expected = engine
            .eval_ast_with_scope::<Dynamic>(&mut Scope::new(), &ast)
            .expect("walker must succeed");
        let actual = run_vm().expect("vm must succeed");
        assert_eq!(
            format!("{expected:?}"),
            format!("{actual:?}"),
            "{} disagreed, so its timing means nothing",
            case.name,
        );

        // One script run per sample, so the two legs alternate closely and
        // the fastest of many has somewhere quiet to land. The count is the
        // same total work the batched form did.
        let samples = RUNS * case.iterations;
        let (walker, vm) = time_paired(
            samples,
            || {
                let _ = engine
                    .eval_ast_with_scope::<Dynamic>(&mut Scope::new(), &ast)
                    .unwrap();
            },
            || {
                let _ = run_vm().unwrap();
            },
        );
        #[cfg(feature = "grain-jit")]
        {
            let jit = jit_state::stats();
            jit_rows.push(format!(
                "[jit] {}: consulted={} skipped={} interpret={} already={} traces={} \
                 aborted={} compiled={} ops={} max_ops={} entries={} refused={} \
                 reentrant={} non_owner={} reasons={}",
                case.name,
                jit.merge_points_consulted,
                jit.forward_steps_skipped,
                jit.entry_door_interpret,
                jit.entry_door_already_tracing,
                jit.traces_started,
                jit.traces_aborted,
                jit.loops_compiled,
                jit.ops_recorded,
                jit.max_trace_ops,
                jit.compiled_entries,
                jit.compiled_entries_refused,
                jit.reentrant_consultations_declined,
                jit.non_owner_consultations_declined,
                if jit.abort_reasons.is_empty() {
                    "-"
                } else {
                    jit.abort_reasons.as_str()
                },
            ));
        }

        let slow_ast = slow_engine.compile(case.source).expect("must compile");
        let walker_slow = time(samples, || {
            let _ = slow_engine
                .eval_ast_with_scope::<Dynamic>(&mut Scope::new(), &slow_ast)
                .unwrap();
        });

        // A ratio of two fastest samples rather than the middle of the
        // per-sample ratios: those multiply both legs' noise together, and the
        // alternation above is what makes the two fastests comparable.
        let speedup = walker.secs() / vm.secs();
        let spread = walker.spread().max(vm.spread());
        // The columns still report what `iterations` runs cost, so they stay
        // the size they were when a sample was a batch of that many.
        let batch = case.iterations as f64 * 1000.0;
        println!(
            "{:<22} {:>9.1}ms {:>9.1}ms {:>8.2}x {:>6.2}x {:>7.0}% {:>9.1}ms {:>10}",
            case.name,
            walker.secs() * batch,
            vm.secs() * batch,
            speedup,
            case.floor,
            spread * 100.0,
            walker_slow.secs() * batch,
            program.residual_nodes(),
        );

        if speedup < case.floor {
            below_floor.push(format!(
                "\n  {}: {speedup:.2}x, floor {:.2}x (worst leg spread {:.0}%)",
                case.name,
                case.floor,
                spread * 100.0,
            ));
        }
    }

    #[cfg(feature = "grain-jit")]
    for row in &jit_rows {
        println!("{row}");
    }
    // Whole-process and cumulative, so it sits under the per-case rows rather
    // than beside them. It is the only account of *why* a case that consulted
    // the door millions of times started no trace.
    #[cfg(feature = "grain-jit")]
    println!("[jit] mc_diag: {}", jit_state::majit_diag_summary());

    println!("\nload average after:  {}", load_average());

    if below_floor.is_empty() {
        return;
    }

    // Printed whether or not this is a gated run: a regression is worth seeing
    // even when nobody asked for an exit code.
    eprintln!(
        "\n{} case(s) below their floor:{}",
        below_floor.len(),
        below_floor.join(""),
    );
    eprintln!(
        "\nA wide spread means the machine was busy — rerun before believing it. \
         If the loss is real, either find it or move the floor in the same commit \
         that causes it.",
    );
    if check {
        std::process::exit(1);
    }
}
