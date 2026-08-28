//! Lower the `grain` dispatch loop to jitcode tables, at build time.
//!
//! The meta-tracing JIT does not read this VM's instructions; it traces the
//! interpreter running them. What it traces is not the machine code the Rust
//! compiler emitted either — it is a register bytecode ("jitcode") lowered
//! from the interpreter's own MIR, one jitcode per function the tracer may
//! walk into. That lowering is what happens here.
//!
//! The MIR arrives as an LLBC artefact extracted ahead of time by `charon`,
//! named through `MAJIT_MIR_FRONTEND_LLBC`. Extraction takes minutes and
//! needs a toolchain this build cannot assume, so it is not run from here:
//! when the variable is unset every table below is written empty and the
//! runtime finds no portal, which is the state a build that never asked for
//! a JIT should be in.
//!
//! The artefact must be extracted with `--features grain-jit`. The merge
//! point is `#[cfg]`d on that feature, so a `grain`-only extraction produces
//! a dispatch loop with no marker in it, the lowering finds no loop to enter,
//! and every table below is well-formed and useless. [`run_pipeline`] refuses
//! that case rather than emitting it.

/// The `E` of the `Result<T, E>` this interpreter returns.
///
/// `front::result_exc` lowers a `Result` to exception edges — the shape a
/// tracer can guard on — only for the carrier named here. `majit-translate`
/// ships none, deliberately: naming one interpreter's error type inside the
/// lowering layer is what the spec exists to undo.
///
/// `EvalAltResult` is returned boxed everywhere (`Result<T, Box<EvalAltResult>>`),
/// so the wrapper has to be named too, or the carrier matches no ADT.
const RHAI_ERROR_CARRIER: majit_translate::ErrorCarrierSpec<'static> =
    majit_translate::ErrorCarrierSpec {
        carrier_path: "rhai::types::error::EvalAltResult",
        carrier_wrappers: &["alloc::boxed::Box"],
        // Neither direction is bridged yet: the tracer's exception object and
        // `EvalAltResult` are still two representations, so a guard that fails
        // out of compiled code leaves through the interpreter's own return
        // path rather than being reconstructed here.
        to_exc_object: None,
        from_exc_object: None,
    };

/// Where the extracted MIR is read from. Named by `majit-translate`'s front
/// end, not by this build, so a consumer pointing several artefacts at one
/// lowering uses the platform path separator as it would for `PATH`.
const LLBC_ENV: &str = "MAJIT_MIR_FRONTEND_LLBC";
const REQUIRE_TABLES_ENV: &str = "RHAI_GRAIN_JIT_REQUIRE_TABLES";

/// Every file this build script writes into `OUT_DIR`.
///
/// The runtime loader reads all of them or none: an index without its bodies
/// names byte ranges of a file that is not there. Writing the whole set on
/// the empty path too is what lets the loader `include_bytes!` unconditionally
/// instead of carrying a second, absent-table shape.
const OUTPUTS: &[&str] = &[
    "jitcodes.bin",
    "jitcodes_index.bin",
    "jit_drivers.bin",
    "insns.bin",
    "all_liveness.bin",
    "descrs.bin",
    "descrs_index.bin",
    "symbolic_fnaddrs.bin",
];

pub fn main() {
    println!("cargo:rerun-if-env-changed={LLBC_ENV}");
    println!("cargo:rerun-if-env-changed={REQUIRE_TABLES_ENV}");
    println!("cargo:rustc-check-cfg=cfg(rhai_grain_jit_tables)");
    println!("cargo:rustc-check-cfg=cfg(rhai_grain_jit_require_tables)");
    if std::env::var(REQUIRE_TABLES_ENV).as_deref() == Ok("1") {
        println!("cargo:rustc-cfg=rhai_grain_jit_require_tables");
    }
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set");

    let Some(llbc) = std::env::var_os(LLBC_ENV) else {
        write_empty_tables(&out_dir);
        return;
    };
    println!("cargo:rustc-cfg=rhai_grain_jit_tables");
    let paths: Vec<std::path::PathBuf> = std::env::split_paths(&llbc).collect();
    assert!(!paths.is_empty(), "{LLBC_ENV} is set but names no artefact");
    for path in &paths {
        assert!(
            path.exists(),
            "{LLBC_ENV} names {}, which does not exist",
            path.display()
        );
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let pipeline = run_pipeline();
    write_tables(&out_dir, &pipeline);
}

/// The declaration the merge point's signature has to agree with.
///
/// `GrainJitDriver::jit_merge_point(pc, program, vm, scope, base, reached)`
/// spells the same split positionally and nothing checks the two against each
/// other at compile time — the lowering does, at `check_jit_marker_operand_kinds`,
/// and only once both halves are in the same build.
fn jit_driver() -> majit_translate::JitDriverSpec {
    majit_translate::JitDriverSpec {
        portal: majit_translate::CallPath::from_segments(["grain", "vm", "Vm", "run_frame"]),
        greens: vec!["pc".to_string(), "program".to_string()],
        reds: vec![
            "vm".to_string(),
            "scope".to_string(),
            "base".to_string(),
            "reached".to_string(),
        ],
        green_kinds: vec![majit_ir::Type::Int, majit_ir::Type::Ref],
        red_kinds: vec![
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Ref,
        ],
        autoreds: false,
        virtualizables: Vec::new(),
        red_types: Vec::new(),
    }
}

fn run_pipeline() -> majit_translate::ProgramPipelineResult {
    let vinfo_factory: &majit_translate::VirtualizableInfoFactory<'_> = &|_, _| None;
    let config = majit_translate::AnalyzeConfig {
        pipeline: majit_translate::PipelineConfig {
            transform: majit_translate::GraphTransformConfig {
                // The merge point is a method on this VM's own driver type, so
                // the recogniser is pointed at it. Left at its default the
                // marker is not recognised as a marker, the portal lowers with
                // no merge point at all, and the tables below describe a loop
                // the tracer can never enter.
                jitdriver_receiver_roots: vec!["GrainJitDriver".to_string()],
                // This VM has neither helper. Left at their defaults they name
                // the other interpreter's externs, and every symbolic fnaddr on
                // this side is still unbound, so such a call keeps the build's
                // sentinel. Declining them drops 76 of the portal's 218 residual
                // calls -- 35% of its residual population, 139 program-wide --
                // along with the fnaddr `const_int` and the `-live-` each one
                // carried, for -152 portal ops. The two ops then fall through to
                // the `int_str/i>r` and `int_add/rr>r` defaults, which pyjitpl
                // has no opimpl for: neither shape executes here yet. What this
                // buys is that the gap reads as a missing opcode instead of a
                // wired-looking call to an address this VM cannot bind. Naming
                // this VM's own concat/render shims here is what makes them live.
                str_concat_helper: String::new(),
                int_str_helper: String::new(),
                ..Default::default()
            },
            register_trait_families: Vec::new(),
            jit_drivers: vec![jit_driver()],
        },
    };
    let static_addrs = majit_translate::HostStaticAddrs {
        error_carrier: RHAI_ERROR_CARRIER,
        ..Default::default()
    };

    let pipeline = majit_translate::analyze_multiple_pipeline_with_modules(
        &[],
        &config,
        None,
        vinfo_factory,
        &[],
        static_addrs,
    );

    let portal = pipeline
        .jit_drivers
        .first()
        .map(|driver| driver.main_jitcode_index)
        .expect("the configured driver names a portal jitcode");
    let merge_points = pipeline.jitcodes[portal]
        .body()
        ._ssarepr
        .as_ref()
        .map(|ssa| {
            // The insn table and an assembled body spell one opcode two ways
            // — `jit_merge_point/cIRFIRF` there, `jitmergepoint` here — so a
            // name matched verbatim reports a present op as absent. Compare
            // with the argcode suffix and the separators dropped.
            majit_translate::codewriter::format::format_assembler(ssa)
                .lines()
                .filter(|line| {
                    line.split_whitespace().any(|word| {
                        word.split('/')
                            .next()
                            .is_some_and(|op| op.replace('_', "") == "jitmergepoint")
                    })
                })
                .count()
        })
        .unwrap_or(0);
    assert_eq!(
        merge_points, 1,
        "the portal jitcode assembles {merge_points} merge points, not 1, so the \
         tracer has no loop to enter. The marker call sits behind \
         `#[cfg(feature = \"grain-jit\")]`: extract the LLBC with \
         `--features grain-jit`, or the call is not in the MIR at all."
    );
    pipeline
}

fn write_tables(out_dir: &str, pipeline: &majit_translate::ProgramPipelineResult) {
    // Bodies concatenated, boundaries beside them: the runtime resolves one
    // jitcode without paying to decode the rest, which matters because the
    // portal is entered per back-edge and its callees are not.
    let mut jitcodes = Vec::new();
    let mut offsets = vec![0_u32];
    let mut names = Vec::with_capacity(pipeline.jitcodes.len());
    for jitcode in &pipeline.jitcodes {
        names.push(jitcode.name.clone());
        jitcodes.extend(bincode::serialize(&**jitcode).expect("a jitcode serializes"));
        offsets.push(
            u32::try_from(jitcodes.len()).expect("jitcodes.bin outgrew the u32 offset range"),
        );
    }
    write(out_dir, "jitcodes.bin", &jitcodes);
    write(
        out_dir,
        "jitcodes_index.bin",
        &bincode::serialize(&(names, offsets)).unwrap(),
    );

    let mut descrs = Vec::new();
    let mut descr_offsets = vec![0_u32];
    for descr in &pipeline.descrs {
        descrs.extend(bincode::serialize(descr).expect("a descr serializes"));
        descr_offsets
            .push(u32::try_from(descrs.len()).expect("descrs.bin outgrew the u32 offset range"));
    }
    write(out_dir, "descrs.bin", &descrs);
    write(
        out_dir,
        "descrs_index.bin",
        &bincode::serialize(&descr_offsets).unwrap(),
    );

    write(
        out_dir,
        "jit_drivers.bin",
        &bincode::serialize(&pipeline.jit_drivers).unwrap(),
    );
    // A build script cannot take an address in the process that will run. For
    // a callee it could not bind, the lowering stores a stable symbolic value
    // and records the path beside it; the runtime owns the shims for those
    // paths and swaps the real addresses in before anything is published.
    write(
        out_dir,
        "symbolic_fnaddrs.bin",
        &bincode::serialize(&pipeline.symbolic_fnaddr_paths).unwrap(),
    );
    // Through a `BTreeMap` view so the bytes do not depend on hash iteration
    // order: two builds of the same artefact have to produce the same table,
    // or a cached one cannot be reused.
    let insns: std::collections::BTreeMap<&String, &u8> = pipeline.insns.iter().collect();
    write(out_dir, "insns.bin", &bincode::serialize(&insns).unwrap());
    // Raw, not encoded: every `-live-` op in a lowered body carries a baked
    // two-byte offset into this stream, so the file has to be the stream. A
    // length prefix would shift every one of those offsets by its own width.
    write(out_dir, "all_liveness.bin", &pipeline.all_liveness);
}

/// The no-artefact shape of every table.
///
/// Each is the empty sequence rather than an empty file, so the runtime
/// deserializes it as it would a real one and finds nothing, instead of
/// having to tell "no JIT was built" from "the build was interrupted".
/// `bincode` writes an empty sequence as its length alone, so the element
/// type never reaches the bytes and naming the real one here would pull
/// `majit-translate` into a build that asked for no JIT.
fn write_empty_tables(out_dir: &str) {
    let empty: Vec<u8> = Vec::new();
    let names: Vec<String> = Vec::new();
    let offsets: Vec<u32> = vec![0];
    write(out_dir, "jitcodes.bin", &empty);
    write(
        out_dir,
        "jitcodes_index.bin",
        &bincode::serialize(&(&names, &offsets)).unwrap(),
    );
    write(out_dir, "descrs.bin", &empty);
    write(
        out_dir,
        "descrs_index.bin",
        &bincode::serialize(&offsets).unwrap(),
    );
    write(
        out_dir,
        "jit_drivers.bin",
        &bincode::serialize(&empty).unwrap(),
    );
    write(
        out_dir,
        "symbolic_fnaddrs.bin",
        &bincode::serialize(&Vec::<(i64, String)>::new()).unwrap(),
    );
    write(
        out_dir,
        "insns.bin",
        &bincode::serialize(&std::collections::BTreeMap::<String, u8>::new()).unwrap(),
    );
    write(out_dir, "all_liveness.bin", &empty);
}

fn write(out_dir: &str, name: &str, bytes: &[u8]) {
    debug_assert!(OUTPUTS.contains(&name), "{name} is not a declared output");
    std::fs::write(std::path::Path::new(out_dir).join(name), bytes)
        .unwrap_or_else(|e| panic!("cannot write {out_dir}/{name}: {e}"));
}
