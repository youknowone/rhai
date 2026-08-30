//! The censuses `grain_llbc_smoke.rs` structurally cannot print.
//!
//! `grain_llbc_smoke`'s sections [2] (abort census), [3] (residual/inline split),
//! [7] and [8] (portal graph) all read `ProgramPipelineResult::functions`.
//! On the production route that field is initialised `Vec::new()` and never
//! written — `lib.rs` says so in a comment directly above the initialiser
//! ("the canonical jitcode emitter below is the production analysis path;
//! `ProgramPipelineResult.functions` / `total_*` remain diagnostic fields").
//! So those four sections print a real header over a zero denominator.
//!
//! Everything below reads the artefacts the production path actually
//! returns: `jitcodes` (each carrying its assembled `_ssarepr`), `descrs`,
//! and `symbolic_fnaddr_paths`.
//!
//! The LLBC must be extracted with `--features grain-jit`, not `--features
//! grain`. The merge-point call is `#[cfg]`d on that feature, so a `grain`-only
//! extraction hands the front end a dispatch loop with no marker in it at all
//! and `[B3]` refuses the run.
use std::collections::BTreeMap;

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

/// `format_assembler` prefixes each line with `'%4d  '` once the SSARepr has
/// been assembled (`insns_pos` set).  Strip that, then take the opname.
fn opname_of(line: &str) -> Option<&str> {
    let t = line.trim();
    if t.is_empty() {
        return None;
    }
    let mut it = t.split_whitespace();
    let first = it.next()?;
    if first.bytes().all(|b| b.is_ascii_digit()) {
        it.next()
    } else {
        Some(first)
    }
}

fn family(op: &str) -> &str {
    op.split('/').next().unwrap_or(op)
}

fn main() {
    let mut llbc_paths: Vec<String> = std::env::args().skip(1).collect();
    let rhai_carrier = !llbc_paths.iter().any(|a| a == "--carrier=pyre");
    llbc_paths.retain(|a| !a.starts_with("--carrier="));
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
        "usage: grain_jitcode_census <scoped.ullbc> ..."
    );
    let joined = std::env::join_paths(llbc_paths.iter()).expect("LLBC path has no separator");
    // SAFETY: single-threaded at this point; no other thread reads the env.
    unsafe { std::env::set_var("MAJIT_MIR_FRONTEND_LLBC", joined) };

    let module_paths: Vec<&str> = Vec::new();
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

    let t0 = std::time::Instant::now();
    let pipeline = majit_translate::analyze_multiple_pipeline_with_modules(
        &module_paths,
        &config,
        None,
        vinfo_factory,
        &[],
        static_addrs,
    );
    println!("pipeline_seconds = {:.2}", t0.elapsed().as_secs_f64());

    // ---- [A] what the production path actually returns ------------------
    println!("\n=== [A] production artefacts ===");
    println!("jitcodes            = {}", pipeline.jitcodes.len());
    println!("descrs              = {}", pipeline.descrs.len());
    println!("insn table entries  = {}", pipeline.insns.len());
    println!(
        "symbolic_fnaddrs    = {}",
        pipeline.symbolic_fnaddr_paths.len()
    );
    println!(
        "indirectcalltargets = {}",
        pipeline.indirectcalltarget_indices.len()
    );
    println!(
        "DIAGNOSTIC-ONLY (never written on this path): functions={} total_ops={} total_blocks={}",
        pipeline.functions.len(),
        pipeline.total_ops,
        pipeline.total_blocks
    );
    let portal_idx = pipeline
        .jit_drivers
        .first()
        .map(|d| d.main_jitcode_index)
        .expect("configured driver");
    println!("portal jitcode idx  = {portal_idx}");

    let with_ssa = pipeline
        .jitcodes
        .iter()
        .filter(|j| j.body()._ssarepr.is_some())
        .count();
    println!(
        "jitcodes carrying an assembled _ssarepr = {with_ssa}/{}",
        pipeline.jitcodes.len()
    );

    // ---- [B] opname histogram over every assembled jitcode --------------
    let mut all: BTreeMap<String, usize> = BTreeMap::new();
    let mut portal: BTreeMap<String, usize> = BTreeMap::new();
    let mut abort_bearers: Vec<(String, usize)> = Vec::new();
    let mut per_jc_ops: Vec<(usize, String, usize)> = Vec::new();
    for (i, jc) in pipeline.jitcodes.iter().enumerate() {
        let Some(ssa) = jc.body()._ssarepr.as_ref() else {
            continue;
        };
        let text = majit_translate::codewriter::format::format_assembler(ssa);
        let mut n_ops = 0usize;
        let mut n_abort = 0usize;
        for line in text.lines() {
            let Some(op) = opname_of(line) else { continue };
            // `---` (block sentinel) and `L<n>:` (label lines) are
            // `format.py` rendering, not emitted ops.
            if op == "---" || (op.starts_with('L') && op.ends_with(':')) {
                continue;
            }
            n_ops += 1;
            *all.entry(op.to_string()).or_default() += 1;
            if i == portal_idx {
                *portal.entry(op.to_string()).or_default() += 1;
            }
            if family(op).starts_with("abort") {
                n_abort += 1;
            }
        }
        per_jc_ops.push((i, jc.name.clone(), n_ops));
        if n_abort > 0 {
            abort_bearers.push((jc.name.clone(), n_abort));
        }
    }
    let total_ops: usize = all.values().sum();
    println!(
        "\n=== [B] assembled-bytecode opname histogram (all {} jitcodes) ===",
        pipeline.jitcodes.len()
    );
    println!("total assembled ops = {total_ops}");
    let mut fam: BTreeMap<String, usize> = BTreeMap::new();
    for (k, v) in &all {
        *fam.entry(family(k).to_string()).or_default() += v;
    }
    let mut fv: Vec<_> = fam.iter().map(|(k, v)| (k.clone(), *v)).collect();
    fv.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (k, v) in fv.iter() {
        println!("{v:8}  {k}");
    }

    println!("\n=== [B2] complete insn table (opname -> byte), the authoritative opname SET ===");
    let mut it: Vec<_> = pipeline
        .insns
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    it.sort();
    for (k, v) in &it {
        println!("{v:5}  {k}");
    }

    println!("\n=== [B3] jitdriver / merge-point probe ===");
    // The insn table and the assembled bodies spell the same opcode two ways —
    // `jit_merge_point/cIRFIRF` in the table, `jitmergepoint` in a body — so a
    // needle matched verbatim reports a present op as absent. Both sides are
    // compared with the separators dropped.
    fn squash(name: &str) -> String {
        name.chars().filter(|c| *c != '_').collect()
    }
    let mut merge_points = 0usize;
    for needle in [
        "jit_merge_point",
        "can_enter_jit",
        "loop_header",
        "jitdriver",
        "promote",
        "guard",
    ] {
        let needle_squashed = squash(needle);
        let hit = |k: &str| squash(k).contains(&needle_squashed);
        let in_table = it.iter().filter(|(k, _)| hit(k)).count();
        let in_bytecode: usize = all.iter().filter(|(k, _)| hit(k)).map(|(_, v)| *v).sum();
        if needle == "jit_merge_point" {
            merge_points = in_bytecode;
        }
        println!("{needle:18} insn_table_keys={in_table:3}  assembled_ops={in_bytecode}");
    }
    let with_jd = pipeline
        .jitcodes
        .iter()
        .filter(|j| j.jitdriver_sd().is_some())
        .count();
    println!(
        "jitcodes with jitdriver_sd set = {with_jd}/{}",
        pipeline.jitcodes.len()
    );
    // Every other section below reads a portal that is a portal only by name.
    // A merge-point-free portal still assembles, still carries a
    // `jitdriver_sd`, and still produces a residual/inline split that looks
    // exactly like a healthy one -- so nothing else printed here can tell the
    // two apart, and every number would be quoted as if the tracer could enter
    // the loop. Refuse rather than report.
    assert_eq!(
        merge_points, 1,
        "the portal assembles no `jit_merge_point`, so the tracer has no loop \
         to enter and every census below describes an unenterable portal. The \
         marker call sits behind `#[cfg(feature = \"grain-jit\")]`; extract the \
         LLBC with `--features grain-jit`, not `--features grain`, or the call \
         is not in the MIR the front end reads."
    );

    // ---- [C] abort census ------------------------------------------------
    println!("\n=== [C] abort census (assembled bytecode) ===");
    let n_abort: usize = fv
        .iter()
        .filter(|(k, _)| k.starts_with("abort"))
        .map(|(_, v)| *v)
        .sum();
    println!("abort-family ops in assembled bytecode = {n_abort}");
    println!("abort-bearing jitcodes = {}", abort_bearers.len());
    abort_bearers.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, v) in abort_bearers.iter().take(30) {
        println!("{v:8}  {k}");
    }
    println!(
        "NOTE: `GraphTransformResult::notes` — the channel carrying \
         'abort: raises to exceptblock' / 'abort placeholder: …' reasons — is \
         `mem::take`n by `Transformer::transform` and then DROPPED: \
         `codewriter.rs` contains no `.notes` reference at all. The reason \
         strings do not survive `make_jitcodes()` on any route."
    );

    // ---- [D] residual / inline split -------------------------------------
    println!("\n=== [D] call-op split ===");
    let pick = |pfx: &str| -> usize {
        all.iter()
            .filter(|(k, _)| family(k).starts_with(pfx))
            .map(|(_, v)| *v)
            .sum()
    };
    println!("residual_call*   = {}", pick("residual_call"));
    println!("inline_call*     = {}", pick("inline_call"));
    println!("call_may_force*  = {}", pick("call_may_force"));
    println!("call_elidable*   = {}", pick("call_elidable"));
    println!("recursive_call*  = {}", pick("recursive_call"));
    println!("indirect_call*   = {}", pick("indirect_call"));
    println!("-- exact opnames in the call families --");
    for (k, v) in all.iter() {
        let f = family(k);
        if f.contains("call") {
            println!("{v:8}  {k}");
        }
    }

    // ---- [E] the portal's own body ---------------------------------------
    println!("\n=== [E] portal jitcode #{portal_idx} ===");
    if let Some(jc) = pipeline.jitcodes.get(portal_idx) {
        println!("name = {}", jc.name);
        println!("code bytes = {}", jc.body().code.len());
        println!(
            "num_regs i/r/f = {}/{}/{}",
            jc.num_regs_i(),
            jc.num_regs_r(),
            jc.num_regs_f()
        );
        let mut pf: BTreeMap<String, usize> = BTreeMap::new();
        for (k, v) in &portal {
            *pf.entry(family(k).to_string()).or_default() += v;
        }
        let mut pv: Vec<_> = pf.iter().map(|(k, v)| (k.clone(), *v)).collect();
        pv.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!("portal assembled ops = {}", portal.values().sum::<usize>());
        for (k, v) in pv.iter() {
            println!("{v:8}  {k}");
        }
    }
    per_jc_ops.sort_by(|a, b| b.2.cmp(&a.2));
    println!("-- 15 biggest assembled jitcodes --");
    for (i, n, o) in per_jc_ops.iter().take(15) {
        println!("{o:8}  #{i} {n}");
    }

    // ---- [F] symbolic fnaddrs, bucketed ----------------------------------
    println!("\n=== [F] symbolic_fnaddr_paths, bucketed by root ===");
    let mut root: BTreeMap<String, usize> = BTreeMap::new();
    for (_, p) in &pipeline.symbolic_fnaddr_paths {
        let r = p.split("::").next().unwrap_or(p);
        let r = r.split('.').next().unwrap_or(r);
        *root.entry(r.to_string()).or_default() += 1;
    }
    let mut rv: Vec<_> = root.iter().map(|(k, v)| (k.clone(), *v)).collect();
    rv.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("distinct roots = {}", rv.len());
    for (k, v) in rv.iter().take(40) {
        println!("{v:8}  {k}");
    }

    // ---- [G] every array descr, verbatim ---------------------------------
    println!("\n=== [G] BhDescr::Array rows, verbatim ===");
    let mut n_arr = 0usize;
    for (i, d) in pipeline.descrs.iter().enumerate() {
        if let majit_translate::codewriter::jitcode::BhDescr::Array { .. } = d {
            n_arr += 1;
            println!("descr[{i}] {d:?}");
        }
    }
    println!(
        "array descrs = {n_arr} / {} total descrs",
        pipeline.descrs.len()
    );
    // ---- [H] M4 REACH ---------------------------------------------------
    // `format_assembler` renders a descr operand ONLY for `switch`
    // (SwitchDictDescr) and for calls (`descriptor.extra_info`); an array
    // op's `array_type_id` is a field on the `OpKind`, never printed.  So
    // the reach question has to be asked of the typed `FlatOp`s, not of the
    // formatted text — a text scan returns 0 for every row and that 0 is
    // the instrument, not the answer.
    println!("\n=== [H] M4 reach: array ops in assembled jitcode bodies ===");
    use majit_translate::codewriter::flatten::FlatOp;
    use majit_translate::OpKind;
    let mut reach: BTreeMap<String, (usize, std::collections::BTreeSet<usize>, usize)> =
        BTreeMap::new();
    let mut owners: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for (i, jc) in pipeline.jitcodes.iter().enumerate() {
        let Some(ssa) = jc.body()._ssarepr.as_ref() else {
            continue;
        };
        for fo in &ssa.insns {
            let FlatOp::Op(op) = fo else { continue };
            let key = match &op.kind {
                OpKind::ArrayRead {
                    array_type_id,
                    item_ty,
                    nolength,
                    ..
                } => {
                    format!("read  {array_type_id:?} item={item_ty:?} nolength={nolength}")
                }
                OpKind::ArrayWrite {
                    array_type_id,
                    item_ty,
                    nolength,
                    ..
                } => {
                    format!("write {array_type_id:?} item={item_ty:?} nolength={nolength}")
                }
                OpKind::ArrayLen {
                    array_type_id,
                    nolength,
                    ..
                } => {
                    format!("len   {array_type_id:?} nolength={nolength}")
                }
                _ => continue,
            };
            *owners
                .entry(key.clone())
                .or_default()
                .entry(format!("#{i} {}", jc.name))
                .or_default() += 1;
            let e = reach.entry(key).or_insert((0, Default::default(), 0));
            e.0 += 1;
            e.1.insert(i);
            if i == portal_idx {
                e.2 += 1;
            }
        }
    }
    println!("{:>6} {:>9} {:>10}  op", "ops", "jitcodes", "in-portal");
    let mut rv2: Vec<_> = reach.iter().collect();
    rv2.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (k, (n, jcs, inp)) in rv2 {
        println!("{n:6} {:9} {inp:10}  {k}", jcs.len());
        if let Some(o) = owners.get(k) {
            let mut ov: Vec<_> = o.iter().collect();
            ov.sort_by(|a, b| b.1.cmp(a.1));
            for (g, c) in ov.iter().take(12) {
                println!("           {c:5}  in {g}");
            }
        }
    }

    // ---- [I] what the portal calls ---------------------------------------
    // [D] splits the whole artefact's calls into residual and inline but names
    // no callee, and [F] names callees over every jitcode at once. Neither says
    // what the ONE loop a tracer runs is unable to see through, which is the
    // only population a portal's residual rate is about.
    println!("\n=== [I] portal call targets, ranked ===");
    let mut targets: BTreeMap<String, usize> = BTreeMap::new();
    // A direct residual call does NOT carry its callee as a `CallTarget`: the
    // fnaddr is materialised as a `ConstInt` into a register and the call takes
    // that register, so `CallFuncPtr::Value` alone cannot tell a direct call
    // from an indirect one. Resolve the register through its defining constant
    // and through `symbolic_fnaddr_paths` before concluding anything.
    let fnaddr_path: BTreeMap<i64, &str> = pipeline
        .symbolic_fnaddr_paths
        .iter()
        .map(|(v, p)| (*v, p.as_str()))
        .collect();
    let mut const_of: BTreeMap<u64, i64> = BTreeMap::new();
    if let Some(ssa) = pipeline.jitcodes[portal_idx].body()._ssarepr.as_ref() {
        for fo in &ssa.insns {
            let FlatOp::Op(op) = fo else { continue };
            if let (OpKind::ConstInt(v), Some(r)) = (&op.kind, op.result.as_ref()) {
                const_of.insert(r.id(), *v);
            }
        }
    }
    let resolve = |f: &majit_translate::model::CallFuncPtr| -> String {
        match f {
            majit_translate::model::CallFuncPtr::Target(t) => format!("{t}"),
            majit_translate::model::CallFuncPtr::Value(v) => match const_of.get(&v.id()) {
                Some(addr) => match fnaddr_path.get(addr) {
                    Some(path) => (*path).to_string(),
                    None => format!("<const {addr:#x}, not a known fnaddr>"),
                },
                None => "<indirect, callee is a runtime value>".to_string(),
            },
        }
    };
    if let Some(ssa) = pipeline.jitcodes[portal_idx].body()._ssarepr.as_ref() {
        for fo in &ssa.insns {
            let FlatOp::Op(op) = fo else { continue };
            // `OpKind::Call` does not survive jtransform: `rewrite_call`
            // replaces it with the effect-classified family, so a probe that
            // matches `Call` reports a portal full of calls as having none.
            let (family, callee) = match &op.kind {
                OpKind::CallResidual { funcptr, .. } => ("residual", resolve(funcptr)),
                OpKind::CallElidable { funcptr, .. } => ("elidable", resolve(funcptr)),
                OpKind::CallMayForce { funcptr, .. } => ("may_force", resolve(funcptr)),
                OpKind::InlineCall { jitcode, .. } => ("inline", jitcode.as_arc().name.clone()),
                _ => continue,
            };
            *targets.entry(format!("{family:9} {callee}")).or_default() += 1;
        }
    }
    let total: usize = targets.values().sum();
    println!(
        "portal Call ops = {total}, distinct callees = {}",
        targets.len()
    );
    // Per family first. The ranked list below is dominated by singletons — 420
    // distinct callees over 547 ops — so a `take(n)` over the merged list shows
    // whichever family happens to have the repeats and hides the other.
    let mut families: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for (k, v) in &targets {
        let f = k.split_once(' ').map_or("?", |(f, _)| f);
        let e = families.entry(f).or_default();
        e.0 += v;
        e.1 += 1;
    }
    for (f, (ops, callees)) in &families {
        println!("{f:10} ops={ops:5}  distinct callees={callees}");
    }
    for family in ["residual", "may_force", "elidable", "inline"] {
        let mut fv: Vec<_> = targets
            .iter()
            .filter(|(k, _)| k.starts_with(family))
            .map(|(k, v)| (k.split_once(' ').map_or(k.as_str(), |(_, r)| r.trim()), *v))
            .collect();
        if fv.is_empty() {
            continue;
        }
        fv.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        println!("-- {family}, top 30 of {} --", fv.len());
        for (k, v) in fv.iter().take(30) {
            println!("{v:6}  {k}");
        }
    }

    println!("-- descr kind histogram --");
    let mut kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    for d in &pipeline.descrs {
        use majit_translate::codewriter::jitcode::BhDescr as B;
        let k = match d {
            B::Array { .. } => "Array",
            B::Field { .. } => "Field",
            B::Size { .. } => "Size",
            B::Call { .. } => "Call",
            B::JitCode { .. } => "JitCode",
            _ => "other",
        };
        *kinds.entry(k).or_default() += 1;
    }
    for (k, v) in kinds {
        println!("{v:8}  {k}");
    }
}
