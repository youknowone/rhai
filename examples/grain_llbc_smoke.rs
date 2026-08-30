//! Stage 0 smoke test: run the majit-translate graph pipeline over a
//! Charon-extracted rhai `grain` VM artefact and dump the abort / residual /
//! descr census.
//!
//! This is a throwaway feasibility probe, not a production consumer. It exists
//! to answer three questions on a real non-pyre crate:
//!   1. does `make_jitcodes()` complete at all (wall time, peak RSS),
//!   2. is the `core::result::branch` / `core::ops::Try::from_residual` pair
//!      absorbed by the front-end recognizers or does it survive as opaque,
//!   3. what array descr does the `Vec<Dynamic>` index leg mint.
//!
//! usage:
//!   cargo run --release --example grain_llbc_smoke -- <scoped.ullbc> [more.ullbc ...]
use std::collections::BTreeMap;

/// rhai's fallible-return carrier: `RhaiResultOf<T> = Result<T,
/// Box<EvalAltResult>>` (`rhai::lib` `RhaiError`).  `Box` is peeled because
/// it contributes no representation — `EvalAltResult` is 64 bytes on the
/// heap and the `Box` is the single owned word the raise site stores, so
/// the payload already IS the trace-level exception value and neither
/// materialisation hook is needed.
const RHAI_ERROR_CARRIER: majit_translate::ErrorCarrierSpec<'static> =
    majit_translate::ErrorCarrierSpec {
        carrier_path: "rhai::types::error::EvalAltResult",
        carrier_wrappers: &["alloc::boxed::Box"],
        to_exc_object: None,
        from_exc_object: None,
    };

/// The shipped interpreter's carrier, kept as the control arm for
/// `--carrier=<name>`: `majit-translate` ships no carrier of its own, so a
/// run that wants the other interpreter's lowering has to name it here.
const SHIPPED_ERROR_CARRIER: majit_translate::ErrorCarrierSpec<'static> =
    majit_translate::ErrorCarrierSpec {
        carrier_path: "pyre_interpreter::error::PyError",
        carrier_wrappers: &[],
        to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
        from_exc_object: Some(("PyError", "from_exc_object")),
    };

fn main() {
    let mut llbc_paths: Vec<String> = std::env::args().skip(1).collect();
    // Both arms run from ONE binary so the two censuses cannot differ by a
    // build.  `--carrier=pyre` swaps in `SHIPPED_ERROR_CARRIER` — the
    // shipped pyre spec, which matches no rhai type — and reproduces the
    // pass-effectively-OFF baseline.
    let rhai_carrier = !llbc_paths.iter().any(|a| a == "--carrier=pyre");
    let front_only = llbc_paths.iter().any(|a| a == "--front-only");
    llbc_paths.retain(|a| !a.starts_with("--carrier=") && a != "--front-only");
    let static_addrs = majit_translate::HostStaticAddrs {
        error_carrier: if rhai_carrier {
            RHAI_ERROR_CARRIER
        } else {
            SHIPPED_ERROR_CARRIER
        },
        ..Default::default()
    };
    println!(
        "=== carrier = {} ===",
        static_addrs.error_carrier.carrier_path
    );
    assert!(
        !llbc_paths.is_empty(),
        "usage: grain_llbc_smoke <scoped.ullbc> [more.ullbc ...]"
    );
    for p in &llbc_paths {
        assert!(
            std::path::Path::new(p).exists(),
            "LLBC artefact not found: {p}"
        );
    }

    // `analyze_multiple_pipeline_with_modules` resolves its LLBC set through
    // the `MAJIT_MIR_FRONTEND_LLBC` OS path-list. That list and an explicit
    // analyzer path set are the only two tiers left (lib.rs
    // `build_semantic_program_via_active_frontend`), so setting it here is what
    // puts the scoped artefact in front of the portal resolver.
    let joined = std::env::join_paths(llbc_paths.iter()).expect("LLBC path has no separator");
    // SAFETY: single-threaded at this point; no other thread reads the env.
    unsafe { std::env::set_var("MAJIT_MIR_FRONTEND_LLBC", joined) };

    // `module_paths` MUST stay empty. It feeds only auto-discovery (dead once
    // the env override resolves) and `should_lower_module`, whose non-matching
    // arm is a pyre-specific `module` / `module::*` carve-out that would drop
    // every `rhai::module::*` item, `Module::get_fn` included.
    let module_paths: Vec<&str> = Vec::new();

    // Portal identity: `rhai::grain::vm::{impl Vm<'_>}::run_frame` is an
    // inherent-impl method, so `front/mir.rs::impl_method_owner_for_fundecl`
    // sets `self_ty_root = "grain::vm::Vm"` and `canonical_inherent_methods`
    // registers it at `CallPath::for_impl_method("grain::vm::Vm", "run_frame")`,
    // i.e. the 4 segments below.
    //
    // Signature: `run_frame(&mut self, program, scope, base, reached, start)`.
    // `pc` is a local, not a parameter, so the green list names the two
    // code-identity operands the driver can actually see at the portal.
    // `base` stays RED (promoted, not green): its value at `Engine::eval` is
    // `scope.len()`, which is embedder-controlled and unbounded, so making it
    // green makes the green-key space unbounded.
    let vinfo_factory: &majit_translate::VirtualizableInfoFactory<'_> = &|_, _| None;
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
                // The VM's merge point is a method on rhai's own driver type
                // (`grain::vm::jit::GrainJitDriver`), not on pyre's, so the
                // recogniser is pointed at it. Without this the portal lowers
                // with no merge point at all and every census below describes a
                // loop the tracer can never enter.
                jitdriver_receiver_roots: vec!["GrainJitDriver".to_string()],
                // Kept in step with `build/majit_prepass.rs`, which is the
                // source of truth: this crate has neither helper, and left at
                // their defaults they name the other interpreter's. A census
                // run with the defaults measures a lowering the build does not
                // produce.
                str_concat_helper: String::new(),
                int_str_helper: String::new(),
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
            }],
        },
    };

    // ---- front-end-only census -------------------------------------------
    // Runs before the codewriter so the M2 / M4 questions stay answerable even
    // when `drain_pending_graphs` aborts on a later graph.
    front_census(&llbc_paths, static_addrs);
    if front_only {
        return;
    }

    let t0 = std::time::Instant::now();
    let pipeline = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        majit_translate::analyze_multiple_pipeline_with_modules(
            &module_paths,
            &config,
            None,
            vinfo_factory,
            &[],
            static_addrs,
        )
    })) {
        Ok(p) => p,
        Err(_) => {
            println!(
                "\n=== PIPELINE ABORTED after {:.2}s (panic, see stderr) ===",
                t0.elapsed().as_secs_f64()
            );
            return;
        }
    };
    let elapsed = t0.elapsed();

    println!("=== [1] pipeline completion ===");
    println!("wall_seconds        = {:.2}", elapsed.as_secs_f64());
    println!("functions           = {}", pipeline.functions.len());
    println!("jitcodes            = {}", pipeline.jitcodes.len());
    println!("total_blocks        = {}", pipeline.total_blocks);
    println!("total_ops           = {}", pipeline.total_ops);
    println!("descrs              = {}", pipeline.descrs.len());
    println!(
        "symbolic_fnaddrs    = {}",
        pipeline.symbolic_fnaddr_paths.len()
    );
    println!(
        "indirectcalltargets = {}",
        pipeline.indirectcalltarget_indices.len()
    );
    for d in &pipeline.jit_drivers {
        println!(
            "driver portal       = {:?} -> jitcode #{}",
            d.portal, d.main_jitcode_index
        );
    }

    println!("\n=== [2] transform notes, bucketed ===");
    let mut note_buckets: BTreeMap<String, usize> = BTreeMap::new();
    let mut abort_by_fn: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_notes = 0usize;
    for f in &pipeline.functions {
        for n in &f.transform_notes {
            total_notes += 1;
            *note_buckets.entry(bucket_note(&n.detail)).or_default() += 1;
            if n.detail.starts_with("abort") {
                *abort_by_fn.entry(n.function.clone()).or_default() += 1;
            }
        }
    }
    println!("total notes = {total_notes}");
    let mut rows: Vec<_> = note_buckets.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, v) in rows.iter().take(60) {
        println!("{v:8}  {k}");
    }
    println!("-- abort-bearing graphs (top 30) --");
    let mut ab: Vec<_> = abort_by_fn.into_iter().collect();
    ab.sort_by(|a, b| b.1.cmp(&a.1));
    println!("abort-bearing graph count = {}", ab.len());
    for (k, v) in ab.iter().take(30) {
        println!("{v:8}  {k}");
    }

    println!("\n=== [3] flattened opname histogram ===");
    let mut opnames: BTreeMap<String, usize> = BTreeMap::new();
    for f in &pipeline.functions {
        let text = majit_translate::codewriter::format::format_assembler(&f.flattened);
        for line in text.lines() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            let head = t.split_whitespace().next().unwrap_or("");
            if head.is_empty() {
                continue;
            }
            *opnames.entry(head.to_string()).or_default() += 1;
        }
    }
    let mut ops: Vec<_> = opnames.iter().map(|(k, v)| (k.clone(), *v)).collect();
    ops.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, v) in ops.iter().take(50) {
        println!("{v:8}  {k}");
    }
    let residual: usize = opnames
        .iter()
        .filter(|(k, _)| k.starts_with("residual_call"))
        .map(|(_, v)| *v)
        .sum();
    let may_force: usize = opnames
        .iter()
        .filter(|(k, _)| k.starts_with("call_may_force"))
        .map(|(_, v)| *v)
        .sum();
    let elidable: usize = opnames
        .iter()
        .filter(|(k, _)| k.starts_with("call_elidable") || k.starts_with("call_pure"))
        .map(|(_, v)| *v)
        .sum();
    let inline: usize = opnames
        .iter()
        .filter(|(k, _)| k.starts_with("inline_call"))
        .map(|(_, v)| *v)
        .sum();
    println!(
        "residual_call*={residual} call_may_force*={may_force} call_elidable/pure*={elidable} inline_call*={inline}"
    );

    println!("\n=== [4] symbolic fnaddr callees, ranked ===");
    let mut sym: BTreeMap<String, usize> = BTreeMap::new();
    for (_, path) in &pipeline.symbolic_fnaddr_paths {
        *sym.entry(path.clone()).or_default() += 1;
    }
    let mut symv: Vec<_> = sym.iter().map(|(k, v)| (k.clone(), *v)).collect();
    symv.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("distinct symbolic callees = {}", symv.len());
    for (k, v) in symv.iter().take(60) {
        println!("{v:8}  {k}");
    }

    println!("\n=== [5] M2: core::result::branch / Try::from_residual ===");
    for needle in [
        "branch",
        "from_residual",
        "Try",
        "FromResidual",
        "ControlFlow",
        "EvalAltResult",
        "unwrap_failed",
    ] {
        let n_sym: usize = symv
            .iter()
            .filter(|(k, _)| k.contains(needle))
            .map(|(_, v)| *v)
            .sum();
        let n_fn = pipeline
            .functions
            .iter()
            .filter(|f| f.name.contains(needle))
            .count();
        println!("needle {needle:16} symbolic_fnaddr_hits={n_sym:6}  lowered_graphs={n_fn}");
    }
    println!("-- symbolic callees matching branch/residual --");
    for (k, v) in symv
        .iter()
        .filter(|(k, _)| k.contains("branch") || k.contains("esidual"))
        .take(40)
    {
        println!("{v:8}  {k}");
    }

    println!("\n=== [6] M4: array descrs ===");
    let mut arr_rows: Vec<String> = Vec::new();
    let mut suspicious = 0usize;
    let mut arr_total = 0usize;
    for d in &pipeline.descrs {
        if let majit_translate::codewriter::jitcode::BhDescr::Array {
            base_size,
            itemsize,
            item_type,
            is_array_of_pointers,
            is_array_of_structs,
            array_type_id,
            interior_fields,
            gc_type_id,
            ..
        } = d
        {
            arr_total += 1;
            let bad = *itemsize == 8
                && matches!(item_type, majit_ir::Type::Ref)
                && !*is_array_of_structs
                && interior_fields.is_empty();
            if bad {
                suspicious += 1;
            }
            arr_rows.push(format!(
                "{}base={base_size} itemsize={itemsize} item_ty={item_type:?} ptrs={is_array_of_pointers} structs={is_array_of_structs} interior={} gc_tid={gc_type_id} array_type_id={:?}",
                if bad { "SUSPECT " } else { "        " },
                interior_fields.len(),
                array_type_id
            ));
        }
    }
    arr_rows.sort();
    arr_rows.dedup();
    println!(
        "array descrs = {arr_total} (distinct rows {}), 8-byte-stride Ref-item, no struct path = {suspicious}",
        arr_rows.len()
    );
    for r in arr_rows.iter().take(80) {
        println!("  {r}");
    }

    println!("\n=== [7] Dynamic / Scope / ThinVec presence ===");
    for needle in ["Dynamic", "Scope", "ThinVec", "SmallVec", "Vec<"] {
        let n = pipeline
            .functions
            .iter()
            .filter(|f| f.name.contains(needle))
            .count();
        let a = arr_rows.iter().filter(|r| r.contains(needle)).count();
        println!("needle {needle:12} graphs={n:6} array_descr_rows={a}");
    }

    println!("\n=== [8] portal graph ===");
    for f in &pipeline.functions {
        if f.name.contains("run_frame") {
            println!(
                "graph {} blocks={} annotations={} vable_rewrites={} notes={} flat_ops={}",
                f.name,
                f.original_blocks,
                f.annotations_count,
                f.vable_rewrites,
                f.transform_notes.len(),
                f.flattened.insns.len()
            );
        }
    }
}

/// Collapse a note detail to a bucket key: keep the shape, drop the identity.
fn bucket_note(detail: &str) -> String {
    if let Some(rest) = detail.strip_prefix("rewrite: ") {
        let head = rest.split(['(', ' ']).next().unwrap_or(rest);
        let tail = rest.rsplit("→ ").next().unwrap_or("");
        return format!("rewrite: {head} → {tail}");
    }
    detail
        .split(':')
        .take(2)
        .collect::<Vec<_>>()
        .join(":")
        .chars()
        .take(120)
        .collect()
}

/// Lower the LLBC through the MIR front-end only (no annotator, no rtyper, no
/// codewriter) and census the two ops the Stage-0 kill criteria name:
/// `OpKind::Call` targets (M2: is `core::result::branch` / `Try::from_residual`
/// absorbed?) and `OpKind::ArrayRead` / `ArrayWrite` identities (M4: does the
/// `Vec<Dynamic>` leg carry `array_type_id: None` + a `Ref` item, i.e. the
/// `arraydescrof_concrete` else arm with an 8-byte stride?).
fn front_census(llbc_paths: &[String], static_addrs: majit_translate::HostStaticAddrs<'_>) {
    use majit_translate::OpKind;
    let t = std::time::Instant::now();
    let llbcs: Vec<majit_charon_reader::Llbc> = llbc_paths
        .iter()
        .map(|p| majit_charon_reader::Llbc::load(p).expect("load llbc"))
        .collect();
    let prog =
        majit_translate::front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
            &llbcs,
            static_addrs,
            &[],
        )
        .expect("front-end lowering failed");
    println!("=== [0] front-end only ===");
    println!(
        "front_seconds = {:.2}  lowered_graphs = {}",
        t.elapsed().as_secs_f64(),
        prog.functions.len()
    );

    let mut call_paths: BTreeMap<String, usize> = BTreeMap::new();
    let mut method_calls: BTreeMap<String, usize> = BTreeMap::new();
    let mut arrays: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_ops = 0usize;
    let mut n_calls = 0usize;
    // graphs that mention a Vec<Dynamic>-shaped array op, for the M4 report
    let mut m4_sites: Vec<String> = Vec::new();
    for f in &prog.functions {
        for b in &f.graph.blocks {
            for op in &b.operations {
                total_ops += 1;
                match &op.kind {
                    OpKind::Call { target, .. } => {
                        n_calls += 1;
                        match target {
                            majit_translate::CallTarget::FunctionPath { segments } => {
                                *call_paths.entry(segments.join("::")).or_default() += 1;
                            }
                            majit_translate::CallTarget::Method {
                                name,
                                receiver_root,
                                ..
                            } => {
                                *method_calls
                                    .entry(format!(
                                        "{}::{name}",
                                        receiver_root.as_deref().unwrap_or("<?>")
                                    ))
                                    .or_default() += 1;
                            }
                            other => {
                                *call_paths.entry(format!("<{other:?}>")).or_default() += 1;
                            }
                        }
                    }
                    OpKind::ArrayRead {
                        item_ty,
                        array_type_id,
                        nolength,
                        ..
                    } => {
                        let key = format!(
                            "ArrayRead item_ty={item_ty:?} array_type_id={array_type_id:?} nolength={nolength}"
                        );
                        if array_type_id.is_none()
                            && matches!(item_ty, majit_translate::ValueType::Ref(_))
                        {
                            m4_sites.push(format!("{}  [read] {key}", f.name));
                        }
                        *arrays.entry(key).or_default() += 1;
                    }
                    OpKind::ArrayWrite {
                        item_ty,
                        array_type_id,
                        nolength,
                        ..
                    } => {
                        let key = format!(
                            "ArrayWrite item_ty={item_ty:?} array_type_id={array_type_id:?} nolength={nolength}"
                        );
                        if array_type_id.is_none()
                            && matches!(item_ty, majit_translate::ValueType::Ref(_))
                        {
                            m4_sites.push(format!("{}  [write] {key}", f.name));
                        }
                        *arrays.entry(key).or_default() += 1;
                    }
                    _ => {}
                }
            }
        }
    }
    println!("front total_ops={total_ops} calls={n_calls}");

    println!("\n--- [0a] M2: Try / Result desugar residue in the lowered graphs ---");
    for needle in [
        "branch",
        "from_residual",
        "from_output",
        "Try",
        "FromResidual",
        "ControlFlow",
        "EvalAltResult",
        "unwrap_failed",
        "Result",
    ] {
        let n: usize = call_paths
            .iter()
            .filter(|(k, _)| k.contains(needle))
            .map(|(_, v)| *v)
            .sum();
        let m: usize = method_calls
            .iter()
            .filter(|(k, _)| k.contains(needle))
            .map(|(_, v)| *v)
            .sum();
        println!("needle {needle:16} fn_path_calls={n:6} method_calls={m:6}");
    }
    println!("-- matching call paths --");
    let mut hits: Vec<_> = call_paths
        .iter()
        .chain(method_calls.iter())
        .filter(|(k, _)| k.contains("branch") || k.contains("esidual") || k.contains("ControlFlow"))
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    hits.sort_by(|a, b| b.1.cmp(&a.1));
    if hits.is_empty() {
        println!("  (none — absorbed by the front-end recognizers)");
    }
    for (k, v) in hits.iter().take(40) {
        println!("{v:8}  {k}");
    }

    println!("\n--- [0b] top opaque/unrecognized call targets (function paths) ---");
    let mut cp: Vec<_> = call_paths.iter().map(|(k, v)| (k.clone(), *v)).collect();
    cp.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("distinct fn-path call targets = {}", cp.len());
    for (k, v) in cp.iter().take(60) {
        println!("{v:8}  {k}");
    }
    println!("--- top method-call targets ---");
    let mut mc: Vec<_> = method_calls.iter().map(|(k, v)| (k.clone(), *v)).collect();
    mc.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("distinct method call targets = {}", mc.len());
    for (k, v) in mc.iter().take(40) {
        println!("{v:8}  {k}");
    }

    println!("\n--- [0c] M4: array op identities ---");
    let mut ar: Vec<_> = arrays.iter().map(|(k, v)| (k.clone(), *v)).collect();
    ar.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (k, v) in ar.iter() {
        println!("{v:8}  {k}");
    }
    println!(
        "identity-less Ref-item array sites = {} (these take the arraydescrof_concrete else arm)",
        m4_sites.len()
    );
    let mut by_graph: BTreeMap<String, usize> = BTreeMap::new();
    for s in &m4_sites {
        let g = s.split("  [").next().unwrap_or("").to_string();
        *by_graph.entry(g).or_default() += 1;
    }
    let mut bg: Vec<_> = by_graph.into_iter().collect();
    bg.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, v) in bg.iter().take(40) {
        println!("{v:8}  {k}");
    }

    println!("\n--- [0c2] M2 residue by module (branch / from_residual) ---");
    let mut m2_by_mod: BTreeMap<String, usize> = BTreeMap::new();
    let mut m2_by_graph: BTreeMap<String, usize> = BTreeMap::new();
    for f in &prog.functions {
        let mut n = 0usize;
        for b in &f.graph.blocks {
            for op in &b.operations {
                if let OpKind::Call {
                    target:
                        majit_translate::CallTarget::Method {
                            name,
                            receiver_root,
                            ..
                        },
                    ..
                } = &op.kind
                {
                    // Nested rather than written as a let-chain: this crate is
                    // edition 2021 and the pattern binds the names the guard reads.
                    if (name == "branch" || name == "from_residual")
                        && receiver_root.as_deref() == Some("Result")
                    {
                        n += 1;
                    }
                }
            }
        }
        if n > 0 {
            *m2_by_mod.entry(f.module_path.clone()).or_default() += n;
            *m2_by_graph
                .entry(format!("{}::{}", f.module_path, f.name))
                .or_default() += n;
        }
    }
    let mut mm: Vec<_> = m2_by_mod.into_iter().collect();
    mm.sort_by(|a, b| b.1.cmp(&a.1));
    println!("graphs carrying the pair = {}", m2_by_graph.len());
    for (k, v) in mm.iter().take(25) {
        println!("{v:8}  module {k}");
    }
    let mut mg: Vec<_> = m2_by_graph.into_iter().collect();
    mg.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, v) in mg.iter() {
        println!("{v:8}  graph {k}");
    }
    // A graph the `result_exc` callee/caller rule DECLINES degrades to a
    // residual stub and leaves `prog.functions` entirely, taking its whole
    // `branch`/`from_residual` population out of this census with it.  That
    // is indistinguishable here from the pass absorbing those diamonds, so
    // dump the lowered population itself: the set difference between two
    // arms is exactly the graphs that dropped out.
    if let Some(path) = std::env::var_os("RHAI_SMOKE_GRAPH_LIST") {
        let mut names: Vec<&str> = prog.functions.iter().map(|f| f.name.as_str()).collect();
        names.sort_unstable();
        std::fs::write(path, names.join("\n")).expect("write graph list");
    }

    println!("\n--- [0d] M4: does `[Dynamic]` reach the ArrayFlag::Struct arm? ---");
    // `arraydescrof_concrete` first calls `extract_element_type_from_str` on
    // `array_type_id`, then `is_known_struct(elem)`. True selects
    // `ArrayFlag::Struct` + `compute_struct_size(elem)`; false falls to
    // `get_type_flag(elem)`, and a `None` identity falls to the else arm
    // (target word size for a Ref item = 8 here).
    for name in [
        "Dynamic",
        "Union",
        "Stmt",
        "Expr",
        "ImmutableString",
        "Scope",
        "Tag",
        "AccessMode",
    ] {
        let known = prog.known_struct_names.contains(name);
        let fields = prog.struct_fields.fields.get(name);
        let sid = prog.struct_ids.get(name).copied().flatten();
        let exact = sid.and_then(|id| prog.exact_layouts.get(&id));
        println!(
            "{name:18} known_struct={known:5} fields={:>3} struct_id={:?} exact_size={:?} align={:?}",
            fields.map(|f| f.len()).unwrap_or(0),
            sid.is_some(),
            exact.and_then(|e| e.size),
            exact.and_then(|e| e.align),
        );
        if let Some(f) = fields {
            for (fname, fty) in f.iter().take(8) {
                println!("      .{fname}: {fty}");
            }
        }
    }

    // `compute_struct_size` path 1 is `cc.struct_layout_for(name)`, whose
    // entries lib.rs installs from `provider.get_struct_layout_exact(name,
    // exact.field_offsets, exact.size)`. Reproduce that call with the same
    // default provider the pipeline uses when `layout_provider` is `None`.
    println!("--- [0e] layout the default provider hands `compute_struct_size` ---");
    let immutable = prog.immutable_fields.clone().into_iter().collect();
    let provider = majit_translate::HeuristicLayoutProvider::from_struct_fields(
        &prog.struct_fields.fields,
        &prog.known_struct_names,
        &immutable,
    );
    for name in [
        "Dynamic",
        "Union",
        "ImmutableString",
        "Scope",
        "Stmt",
        "Expr",
    ] {
        let sid = prog.struct_ids.get(name).copied().flatten();
        let exact = sid.and_then(|id| prog.exact_layouts.get(&id));
        let layout = match exact {
            Some(e) => majit_translate::LayoutProvider::get_struct_layout_exact(
                &provider,
                name,
                &e.field_offsets,
                e.size,
            ),
            None => majit_translate::LayoutProvider::get_struct_layout(&provider, name),
        };
        println!(
            "{name:18} exact={:?} -> provider layout size={:?} fields={:?}",
            exact.and_then(|e| e.size),
            layout.as_ref().map(|l| l.size),
            layout.as_ref().map(|l| l.fields.len()),
        );
    }
}
