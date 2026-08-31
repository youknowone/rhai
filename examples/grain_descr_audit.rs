//! Class audit for the Void-numbering divergence, measured on rhai.
//!
//! The Void-numbering panic (`get_fielddescr_index_in declined to number
//! Union::Unit.__pos_0`) was one instance of a class: a descr mint that
//! numbers a struct's fields with a walk whose skip set differs from the walk
//! the census (`heaptracker.py get_fielddescr_index_in` /
//! `all_fielddescrs`) uses.  Fixing some instances and not others converts a
//! loud panic into a silent wrong descr, so the class needs a measurement, not
//! a reading.
//!
//! `GcCache::positional_invariant_census` is that measurement: it walks every
//! populated `_cache_size` entry and reports how many listed fields report an
//! `index_in_parent` other than their own position in the list.  Upstream that
//! is a theorem — `descr.py get_field_descr` numbers with
//! `heaptracker.get_fielddescr_index_in` and `SizeDescr.all_fielddescrs` lists
//! with `heaptracker.all_fielddescrs`, one skip set shared.
//!
//! Run it on rhai because pyre's own structs carry no `()`-payload variant,
//! which is why the class went unexercised until a non-pyre consumer arrived.
//!
//! usage: grain_descr_audit <scoped.ullbc> [more.ullbc ...]
fn main() {
    let llbc_paths: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        !llbc_paths.is_empty(),
        "usage: grain_descr_audit <scoped.ullbc> [more.ullbc ...]"
    );
    for p in &llbc_paths {
        assert!(
            std::path::Path::new(p).exists(),
            "LLBC artefact not found: {p}"
        );
    }
    let joined = std::env::join_paths(llbc_paths.iter()).expect("LLBC path has no separator");
    // SAFETY: single-threaded at this point; no other thread reads the env.
    unsafe { std::env::set_var("MAJIT_MIR_FRONTEND_LLBC", joined) };

    let module_paths: Vec<&str> = Vec::new();
    let vinfo_factory: &majit_translate::VirtualizableInfoFactory<'_> = &|_, _| None;
    // `analyze_multiple_pipeline_with_modules` refuses to run without a
    // driver, so carry `grain_llbc_smoke`'s portal spec verbatim.  Which graph is
    // the portal is irrelevant here: the census reads every descr the
    // codewriter minted on the way, and it aborts at the same graph either
    // way.
    let config = majit_translate::AnalyzeConfig {
        pipeline: majit_translate::PipelineConfig {
            transform: majit_translate::GraphTransformConfig {
                vable_fields: ["scope", "base", "reached", "stack_base"]
                    .into_iter()
                    .enumerate()
                    .map(|(index, name)| {
                        majit_translate::VirtualizableFieldDescriptor::new(
                            name,
                            Some("GrainFrame".to_string()),
                            index,
                        )
                    })
                    .collect(),
                ..Default::default()
            },
            register_trait_families: Vec::new(),
            jit_drivers: vec![majit_translate::JitDriverSpec {
                portal: majit_translate::CallPath::from_segments([
                    "grain",
                    "vm",
                    "Vm",
                    "run_frame",
                ]),
                greens: vec![
                    "pc".to_string(),
                    "program_identity".to_string(),
                    "program".to_string(),
                ],
                reds: vec!["frame".to_string(), "vm".to_string()],
                green_kinds: vec![
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                    majit_ir::Type::Ref,
                ],
                red_kinds: vec![majit_ir::Type::Ref, majit_ir::Type::Ref],
                autoreds: false,
                virtualizables: vec!["frame".to_string()],
                red_types: vec!["GrainFrame".to_string(), "Vm".to_string()],
                portal_runner: None,
                split_portal: false,
            }],
        },
    };
    let static_addrs = majit_translate::HostStaticAddrs::default();

    let t0 = std::time::Instant::now();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        majit_translate::analyze_multiple_pipeline_with_modules(
            &module_paths,
            &config,
            None,
            vinfo_factory,
            &[],
            static_addrs,
        )
    }));
    println!(
        "pipeline {} after {:.2}s",
        if outcome.is_ok() {
            "completed"
        } else {
            "aborted"
        },
        t0.elapsed().as_secs_f64()
    );

    // The census reads the process-global `gc_cache`, which every mint routed
    // through regardless of whether the pipeline finished.
    let gc = majit_ir::descr::gc_cache().lock();
    let [checked, disagreeing] = gc.positional_invariant_census();
    let [compared, conflicting] = gc.identity_collision_census();
    println!("positional_invariant_census: checked={checked} disagreeing={disagreeing}");
    println!("identity_collision_census:   compared={compared} conflicting={conflicting}");
}
