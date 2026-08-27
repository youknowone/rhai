//! Which half of `*slice.get(i)?` puts a GC reference on a `SwitchInt`.
//!
//! `codewriter/flatten.rs` rejects a ref-kinded exitswitch outright
//! (`switch exitswitch must be int`), and rhai's `grain` dispatch loop trips
//! it on `let tag = *code.get(pc).ok_or_else(..)?;`.  That one line stacks two
//! candidate causes: the `Try` desugaring (`Result::branch` leaving a
//! `ControlFlow` aggregate the front end has to project through) and the
//! `&u8` itself (a borrow of a primitive, which pyre's value model has no
//! representation for — `Rvalue::Ref` is the identity on the byte).
//!
//! The four arms in the `switchfix` fixture separate them: A reads through
//! `Index::index`, B through `Option::unwrap` (no `Try`), C through `?` on
//! `Option`, D through `ok_or_else` + `?` on `Result` — rhai's spelling.
//! Run one arm per invocation, because the portal has to resolve to one graph.
//!
//! usage: grain_switch_arm_probe <switchfix.ullbc> <fn-name>
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        2,
        "usage: grain_switch_arm_probe <ullbc> <fn-name>"
    );
    let (llbc, portal_name) = (&args[0], &args[1]);
    assert!(std::path::Path::new(llbc).exists(), "no artefact: {llbc}");
    let joined = std::env::join_paths([llbc]).expect("LLBC path has no separator");
    // SAFETY: single-threaded at this point; no other thread reads the env.
    unsafe { std::env::set_var("MAJIT_MIR_FRONTEND_LLBC", joined) };

    let module_paths: Vec<&str> = Vec::new();
    let vinfo_factory: &majit_translate::VirtualizableInfoFactory<'_> = &|_, _| None;
    let config = majit_translate::AnalyzeConfig {
        pipeline: majit_translate::PipelineConfig {
            transform: majit_translate::GraphTransformConfig {
                ..Default::default()
            },
            register_trait_families: Vec::new(),
            jit_drivers: vec![majit_translate::JitDriverSpec {
                portal: majit_translate::CallPath::from_segments([portal_name.as_str()]),
                greens: vec!["pc".to_string()],
                reds: vec!["code".to_string()],
                green_kinds: vec![majit_ir::Type::Int],
                red_kinds: vec![majit_ir::Type::Ref],
                autoreds: false,
                virtualizables: Vec::new(),
                red_types: Vec::new(),
            }],
        },
    };

    let t0 = std::time::Instant::now();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        majit_translate::analyze_multiple_pipeline_with_modules(
            &module_paths,
            &config,
            None,
            vinfo_factory,
            &[],
            majit_translate::HostStaticAddrs::default(),
        )
    }));
    match outcome {
        Ok(p) => println!(
            "ARM {portal_name}: OK in {:.2}s — functions={} jitcodes={}",
            t0.elapsed().as_secs_f64(),
            p.functions.len(),
            p.jitcodes.len()
        ),
        Err(_) => println!(
            "ARM {portal_name}: PANIC in {:.2}s (see stderr)",
            t0.elapsed().as_secs_f64()
        ),
    }
}
