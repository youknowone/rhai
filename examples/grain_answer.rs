//! One script, both evaluators, and the two answers beside each other.
//!
//! `grain_bench` asserts the pair and panics on a disagreement, which reports
//! that there is one but not what either arm did with a different script. This
//! takes the script on the command line and prints both answers, so a
//! disagreement can be narrowed by editing the script rather than the table.

use rhai::grain::{Compiler, Program, Vm};
use rhai::{Dynamic, Engine, Scope};

fn main() {
    let source = std::env::args().nth(1).unwrap_or_else(|| {
        "let s = 0; let i = 0; while i < 20000 { s += i; i += 1; } s".to_string()
    });
    let engine = Engine::new();
    let ast = engine.compile(&source).expect("the script compiles");
    let program: Program = Compiler::new().compile(&ast);

    let walker = engine
        .eval_ast_with_scope::<Dynamic>(&mut Scope::new(), &ast)
        .expect("the walker runs");
    let vm = Vm::new(&engine)
        .eval_with_scope(&mut Scope::new(), &program)
        .expect("the vm runs");

    println!("walker {walker:?}");
    println!("vm     {vm:?}");
    println!("agree  {}", format!("{walker:?}") == format!("{vm:?}"));
    // After the run, not at the moment a trace started: what a reader wants is
    // how the whole script went, and the driver's own report fires on the
    // first decision.
    println!("jit    {:?}", rhai::grain::jit_state::stats());
    println!("majit  {}", rhai::grain::jit_state::majit_diag_summary());
}
