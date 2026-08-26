//! Walker-against-VM timing for a script handed in on the command line.
//!
//! `grain_bench` and `grain_nsiter` carry fixed tables, so answering "which
//! construct in this loop is the one the VM is not ahead on" means editing a
//! shared file. This takes the script as an argument instead, which is what
//! lets a decomposition be written as a set of files rather than as a patch.
//!
//! The measurement discipline is `grain_nsiter`'s, for the reasons written
//! there: one process, arms interleaved run by run, fastest sample reported,
//! spread printed beside it, load average printed before and after.
//!
//! ```text
//! grain_probe [--runs=N] [--iters=N] <script.rhai>...
//! ```
//!
//! `--iters` is the number of loop turns one run of the script performs. It
//! divides the timing into a ns/iter column and is otherwise unused; the ratio
//! stands without it.

use std::time::{Duration, Instant};

use rhai::grain::{Compiler, Program, Vm};
use rhai::{Dynamic, Engine, Scope};

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
    let mut runs = 7usize;
    let mut iters: Option<f64> = None;
    let mut paths = Vec::new();
    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("--runs=") {
            runs = v.parse().expect("--runs must be a number");
        } else if let Some(v) = arg.strip_prefix("--iters=") {
            iters = Some(v.parse().expect("--iters must be a number"));
        } else {
            paths.push(arg);
        }
    }
    assert!(
        !paths.is_empty(),
        "usage: grain_probe [--runs=N] [--iters=N] <script.rhai>..."
    );

    let engine = Engine::new();

    println!("load average before: {}", load_average());
    println!("runs: {runs}, reporting the fastest\n");
    let ns_col = if iters.is_some() { "vm ns/iter" } else { "" };
    println!(
        "{:<28} {:>11} {:>11} {:>9} {:>8} {:>12}",
        "", "walker ms", "vm ms", "speedup", "spread", ns_col
    );

    for path in &paths {
        let source =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        let name = std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());

        let ast = engine.compile(&source).expect("must compile");
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

        // Same answer from both arms, or the timings are measuring different
        // programs. Doubles as the warm-up.
        let expected = format!("{:?}", run_walker());
        let got = format!("{:?}", run_vm());
        assert_eq!(expected, got, "vm disagreed with the walker on {name}");

        let mut walker = Vec::with_capacity(runs);
        let mut vm = Vec::with_capacity(runs);
        for _ in 0..runs {
            for (samples, run) in [
                (&mut walker, &mut run_walker as &mut dyn FnMut() -> Dynamic),
                (&mut vm, &mut run_vm),
            ] {
                let start = Instant::now();
                let value = run();
                samples.push(start.elapsed());
                let _ = std::hint::black_box(value);
            }
        }

        let best = |s: &[Duration]| s.iter().copied().min().expect("a sample");
        let spread = |s: &[Duration]| {
            let lo = best(s).as_secs_f64();
            let hi = s.iter().copied().max().expect("a sample").as_secs_f64();
            if lo > 0.0 {
                (hi - lo) / lo * 100.0
            } else {
                0.0
            }
        };
        let w = best(&walker).as_secs_f64();
        let v = best(&vm).as_secs_f64();
        let ns = iters
            .map(|n| format!("{:.2}", v * 1e9 / n))
            .unwrap_or_default();
        println!(
            "{:<28} {:>9.2}ms {:>9.2}ms {:>8.2}x {:>7.0}% {:>12}",
            name,
            w * 1e3,
            v * 1e3,
            w / v,
            spread(&vm),
            ns
        );
    }

    println!("\nload average after:  {}", load_average());
}
