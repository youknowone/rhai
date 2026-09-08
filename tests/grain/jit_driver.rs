//! A `JitDriver` is built from the tables this crate ships, and accepts them.
//!
//! The structural checks run on the build artefact, and the hot-loop check
//! reaches the first-thread-owned driver through the real VM merge point. The
//! first question remains whether the metainterp accepts what the build
//! produced. It has its own ways to say no,
//! and most of them are quiet: a schema whose red count disagrees with the
//! portal's `BC_JIT_MERGE_POINT` payload, a liveness stream whose opcode
//! numbering is not the one the bodies were assembled against, live values
//! whose kinds do not match the declared reds. The last of those is the one
//! this VM would hit first, because both reds are Ref and the `JitState`
//! default types every red Int.

use majit_metainterp::{JitDriver, JitState};
use rhai::grain::{Compiler, Vm, jit_state, jitcodes};
use rhai::{Engine, Scope};

/// The unchanged majit/RPython warm-loop threshold used by the Grain runtime.
const THRESHOLD: u32 = 1039;

/// The build lowered no tables, so there is nothing for a driver to accept.
fn no_tables() -> bool {
    let empty = jitcodes::count() == 0;
    if empty {
        assert_ne!(
            std::env::var("RHAI_GRAIN_JIT_REQUIRE_TABLES").as_deref(),
            Ok("1"),
            "RHAI_GRAIN_JIT_REQUIRE_TABLES=1 requires jitcode tables lowered from the \
             MAJIT_MIR_FRONTEND_LLBC artefact, but this build loaded 0 jitcodes"
        );
        eprintln!(
            "no LLBC artefact was named at build time; set MAJIT_MIR_FRONTEND_LLBC to a \
             `--features grain-jit` extraction to exercise the driver."
        );
    }
    empty
}

#[test]
#[cfg_attr(all(not(rhai_grain_jit_tables), not(rhai_grain_jit_require_tables)), ignore = "vacuous: no MAJIT_MIR_FRONTEND_LLBC tables were built")]
fn the_descriptor_is_the_shape_the_build_recorded() {
    if no_tables() {
        return;
    }
    let jd = jit_state::grain_driver_descriptor();

    let greens: Vec<_> = jd.greens().iter().map(|var| var.tp).collect();
    let reds: Vec<_> = jd.reds().iter().map(|var| var.tp).collect();
    assert_eq!(greens, [majit_ir::Type::Int, majit_ir::Type::Int, majit_ir::Type::Ref],);
    assert_eq!(reds, [majit_ir::Type::Ref, majit_ir::Type::Ref]);
    // `warmspot.py:664` derives one history-kind char per red; the descriptor
    // does the same, so this is the same list twice and disagreeing would mean
    // the descriptor's two accounts of its reds had diverged.
    assert_eq!(jd.red_args_types, vec!['r', 'r']);
}

/// The gate that declines quietly.
///
/// `live_values_match_descriptor` refuses when the live-value count differs
/// from the red count, or when any live value's type differs from its red's --
/// and the callers return without tracing and without an error. The default
/// `live_value_types` types every red `Int`, which for this VM is wrong for
/// three of four, so this is what the override buys.
#[test]
#[cfg_attr(all(not(rhai_grain_jit_tables), not(rhai_grain_jit_require_tables)), ignore = "vacuous: no MAJIT_MIR_FRONTEND_LLBC tables were built")]
fn the_live_values_carry_the_kinds_the_descriptor_declares() {
    if no_tables() {
        return;
    }
    let jd = jit_state::grain_driver_descriptor();
    let state = jit_state::GrainJitState::default();

    // One raw word per red, as the merge point would pass them.
    let env: Vec<i64> = vec![0x1000, 0x2000];
    let meta = state.build_meta(jitcodes::portal_merge_point_offset().expect("the portal names its merge point"), &env);

    // This integration test has no live frame to publish. Actual red-value
    // publication and typed extraction are tested inside jit_state; the
    // stored meta must not supply stale values to a later compiled entry.
    let types = state.live_value_types(&meta);
    let declared: Vec<_> = jd.reds().iter().map(|var| var.tp).collect();
    assert_eq!(types, declared, "a live value whose type differs from its red's makes the driver decline silently",);
}

#[test]
fn the_frame_virtualizable_layout_is_registered_by_the_runtime_state() {
    let info = <jit_state::GrainJitState as JitState>::__build_virtualizable_info().expect("the frame red has runtime virtualizable metadata");

    assert_eq!(info.name, "frame");
    assert!(!info.has_vable_token(), "a stack-resident GrainFrame has no force token");
    assert_eq!(info.identity_live_index, Some(0));
    assert_eq!(info.identity_ref_bank_index, Some(0));
    assert_eq!(
        info.static_fields.iter().map(|field| (field.name.as_str(), field.field_type)).collect::<Vec<_>>(),
        [
            ("scope", majit_ir::Type::Ref),
            ("base", majit_ir::Type::Int),
            ("reached", majit_ir::Type::Int),
            ("stack_base", majit_ir::Type::Int),
            ("jit_resume_pc_plus_one", majit_ir::Type::Int),
        ],
    );
}

/// The compatibility gate checks the compiled red schema, not whether some
/// unrelated path in the build still has an unbound residual.
#[test]
#[cfg_attr(all(not(rhai_grain_jit_tables), not(rhai_grain_jit_require_tables)), ignore = "vacuous: no MAJIT_MIR_FRONTEND_LLBC tables were built")]
fn the_state_accepts_matching_reds_and_refuses_a_mismatched_schema() {
    if no_tables() {
        return;
    }
    jit_state::reset_stats();
    let state = jit_state::GrainJitState::default();
    let header_pc = jitcodes::portal_merge_point_offset().expect("the portal names its merge point");
    let meta = state.build_meta(header_pc, &[0x1000, 0x2000][..]);

    assert!(state.is_compatible(&meta));
    assert_eq!(jit_state::stats().compiled_entries_refused, 0);
    for reds in [&[][..], &[0x1000][..], &[0x1000, 0x2000, 0x3000][..]] {
        let mismatched = jit_state::GrainMeta { header_pc, reds: reds.to_vec() };
        assert!(!state.is_compatible(&mismatched), "the compiled entry requires exactly its declared red schema");
    }
    let stats = jit_state::stats();
    assert_eq!(stats.compiled_entries_refused, 3);
    assert_eq!(stats.compiled_entries, 0);
}

/// The symbolic values the trace carries across the back edge.
#[test]
#[cfg_attr(all(not(rhai_grain_jit_tables), not(rhai_grain_jit_require_tables)), ignore = "vacuous: no MAJIT_MIR_FRONTEND_LLBC tables were built")]
fn the_symbolic_reds_are_input_args_in_declaration_order() {
    if no_tables() {
        return;
    }
    let state = jit_state::GrainJitState::default();
    let header_pc = jitcodes::portal_merge_point_offset().expect("the portal names its merge point");
    let meta = state.build_meta(header_pc, &[0x1000, 0x2000][..]);
    let sym = jit_state::GrainJitState::create_sym(&meta, header_pc);

    let expected: Vec<_> = jit_state::red_kinds().iter().enumerate().map(|(k, tp)| majit_ir::OpRef::input_arg_typed(k as u32, *tp)).collect();
    assert_eq!(sym.reds, expected);
    assert_eq!(jit_state::GrainJitState::collect_jump_args(&sym), expected, "the closing jump carries every red, in the order the label numbered them",);
    assert!(jit_state::GrainJitState::validate_close(&sym, &meta));
}

/// The whole bring-up, in the order the steps have to run in.
///
/// `install` publishes the tables as the global build pool, which is the seed
/// `register_dispatch_jitcode_shared` adopts the portal out of.
/// `install_liveness_from_build_parts` and the registration both take the
/// staticdata through `Arc::get_mut` and panic once any trace has cloned it, so
/// they run before anything could have.
#[test]
#[cfg_attr(all(not(rhai_grain_jit_tables), not(rhai_grain_jit_require_tables)), ignore = "vacuous: no MAJIT_MIR_FRONTEND_LLBC tables were built")]
fn the_driver_accepts_the_tables_this_crate_ships() {
    if no_tables() {
        return;
    }
    jitcodes::install();

    let mut driver: JitDriver<jit_state::GrainJitState> = JitDriver::with_descriptor(THRESHOLD, jit_state::grain_driver_descriptor());
    driver.ensure_descriptor_registered();
    assert_eq!(driver.index(), Some(0), "the first registered driver is the one the lowered bodies name index 0",);

    let (insns, all_liveness) = jitcodes::liveness_parts();
    driver.meta_interp_mut().install_liveness_from_build_parts(&insns, all_liveness);

    let table = jitcodes::all();
    let portal = jitcodes::portal_index().expect("a lowered table is named by a driver");
    let portal_jitcode = table.get(portal).expect("the driver names a portal the lowered table reaches").clone();
    // Not `install_jitcodes`: this is the door that cross-checks the portal's
    // `BC_JIT_MERGE_POINT` payload partition against the reds the descriptor
    // declares, and sets both halves of the `call.py grab_initial_jitcodes`
    // back-pointer. Publishing the table alone leaves the portal not answering
    // to `is_main_jitcode`.
    driver.register_dispatch_jitcode_shared(&portal_jitcode);

    let named = driver.dispatch_jitcode().expect("registration names this driver's portal");
    assert!(
        std::sync::Arc::ptr_eq(named, &portal_jitcode),
        "the driver's portal is the table's own entry, not a copy of it -- every `j` operand at \
         that index resolves through this identity",
    );
    assert_eq!(named.exec.jit_merge_point_offset, jitcodes::portal_merge_point_offset(),);
    assert_eq!(driver.meta_interp_mut().jitcodes().len(), table.len(), "registration publishes the whole seeded table, not just the portal",);
    eprintln!("driver 0 accepted {} jitcodes, portal {portal}", table.len());
}

/// A real VM loop must compile, enter machine code and finish with the same
/// answer as the walker. Adjacent trip counts exercise different positions
/// of the terminating condition relative to the compiled interval: the old
/// incomplete guard handoff accidentally passed 20000 but added 4096 twice.
#[test]
#[cfg_attr(all(not(rhai_grain_jit_tables), not(rhai_grain_jit_require_tables)), ignore = "vacuous: no MAJIT_MIR_FRONTEND_LLBC tables were built")]
fn a_hot_grain_loop_compiles_enters_and_resumes_without_replaying_effects() {
    if no_tables() {
        return;
    }

    for limit in [4095, 4096, 4097] {
        check_hot_loop(limit);
    }
}

fn check_hot_loop(limit: usize) {
    jit_state::reset_stats();
    let engine = Engine::new();
    let ast = engine
        .compile(format!("let i = 0; let total = 0; while i < {limit} {{ total += i; i += 1; }} total"))
        .expect("the hot loop parses");
    let program = Compiler::new().compile(&ast);
    let mut scope = Scope::new();
    let result = Vm::new(&engine).eval_with_scope(&mut scope, &program).expect("the hot loop runs");
    let expected = limit * (limit - 1) / 2;
    assert_eq!(format!("{result:?}"), expected.to_string(), "limit {limit}");

    let stats = jit_state::stats();
    eprintln!("hot grain loop JIT stats: {stats:?}");
    assert!(stats.merge_points_consulted > THRESHOLD as usize, "the real loop must cross the unchanged warm threshold: {stats:?}",);
    assert_eq!(
        stats.entry_door_interpret + stats.entry_door_already_tracing + stats.traces_started,
        stats.merge_points_consulted,
        "every consultation must have an observable entry-door decision: {stats:?}",
    );
    assert!(stats.traces_started > 0, "a warm decision must start tracing: {stats:?}",);
    assert!(stats.ops_recorded > 0, "the portal walk must append trace operations: {stats:?}",);
    assert!(stats.loops_compiled > 0, "the integer loop must close and compile: {stats:?}");
    assert_eq!(stats.symbolic_residual_aborts, 0, "the integer path must not stop at an unbound residual: {stats:?}");
    assert!(stats.compiled_entries > 0, "compilation alone does not prove machine code ran: {stats:?}");
    assert_eq!(stats.non_owner_consultations_declined, 0, "the observation run must own its driver: {stats:?}",);
    assert_eq!(stats.reentrant_consultations_declined, 0, "the observation run must not hide nested consultations: {stats:?}",);

    // The same compiled artifact must accept a fresh VM/frame and scope,
    // rather than carrying the first run's live object identities with it.
    let again = Vm::new(&engine).eval_with_scope(&mut Scope::new(), &program).expect("the compiled loop runs again");
    assert_eq!(format!("{again:?}"), expected.to_string(), "reused artifact, limit {limit}");
    assert!(jit_state::stats().compiled_entries > stats.compiled_entries);
}

/// A hot loop whose body takes more than one arm must still compile, match
/// the walker, and survive the non-hot arm. `handle_guard_failure` traces
/// from the failed guard; the blackhole finishes a half-opcode rather than
/// panicking on `usize::MAX`.
#[test]
#[cfg_attr(all(not(rhai_grain_jit_tables), not(rhai_grain_jit_require_tables)), ignore = "vacuous: no MAJIT_MIR_FRONTEND_LLBC tables were built")]
fn a_multi_arm_loop_compiles_and_agrees_on_the_cold_arm() {
    if no_tables() {
        return;
    }
    jit_state::reset_stats();
    const SOURCE: &str = "let s = 0; let i = 0; \
         while i < 1500 { \
             if i % 3 == 0 { s += 1; } else if i % 3 == 1 { s += 2; } else { s -= 1; } \
             i += 1; \
         } s";
    let engine = Engine::new();
    let ast = engine.compile(SOURCE).expect("the multi-arm loop parses");
    let program = Compiler::new().compile(&ast);
    let walker = engine.eval_ast_with_scope::<rhai::Dynamic>(&mut Scope::new(), &ast).expect("the walker runs");
    let vm = Vm::new(&engine).eval_with_scope(&mut Scope::new(), &program).expect("the vm runs the non-hot arm without panicking");
    assert_eq!(format!("{walker:?}"), format!("{vm:?}"), "walker and vm must agree after the non-hot arm");
    let stats = jit_state::stats();
    eprintln!("multi-arm grain JIT stats: {stats:?}");
    eprintln!("multi-arm majit: {}", jit_state::majit_diag_summary());
    assert!(stats.loops_compiled > 0, "the multi-arm loop must compile: {stats:?}");
    assert_eq!(format!("{walker:?}"), "1000", "1500 iterations of +1/+2/-1 cycle to 1000");

    // Same arms through `for` / `ITER_NEXT_STORE`. The other-arm walk used
    // to drop a walk-local `Union` at a small integer address. Keep this
    // in the same test so it shares the process-owned driver.
    jit_state::reset_stats();
    const FOR_SOURCE: &str = "let s = 0; for i in 0..1500 { \
         if i % 3 == 0 { s += 1; } else if i % 3 == 1 { s += 2; } else { s -= 1; } \
     } s";
    let ast = engine.compile(FOR_SOURCE).expect("the multi-arm for-loop parses");
    let program = Compiler::new().compile(&ast);
    let walker = engine.eval_ast_with_scope::<rhai::Dynamic>(&mut Scope::new(), &ast).expect("the walker runs");
    let vm = Vm::new(&engine)
        .eval_with_scope(&mut Scope::new(), &program)
        .expect("the vm runs the for-loop non-hot arm without panicking");
    assert_eq!(format!("{walker:?}"), format!("{vm:?}"), "for-loop walker and vm must agree after the non-hot arm");
    let stats = jit_state::stats();
    eprintln!("for-loop multi-arm grain JIT stats: {stats:?}");
    eprintln!("for-loop multi-arm majit: {}", jit_state::majit_diag_summary());
    assert!(stats.loops_compiled > 0, "the for-loop multi-arm loop must compile: {stats:?}");
    assert_eq!(format!("{walker:?}"), "1000", "1500 for-loop iterations of +1/+2/-1 cycle to 1000");

    // Long enough that `must_compile` fires on the cold arms. Those arms
    // used to residual-call `operand_stack_store` with a walk-local
    // `Dynamic` (`rir`) and abort the bridge.
    jit_state::reset_stats();
    const LONG_FOR: &str = "let s = 0; for i in 0..8000 { \
         if i % 3 == 0 { s += 1; } else if i % 3 == 1 { s += 2; } else { s -= 1; } \
     } s";
    let ast = engine.compile(LONG_FOR).expect("the long for-loop parses");
    let program = Compiler::new().compile(&ast);
    let walker = engine.eval_ast_with_scope::<rhai::Dynamic>(&mut Scope::new(), &ast).expect("the walker runs");
    let vm = Vm::new(&engine).eval_with_scope(&mut Scope::new(), &program).expect("the vm runs the long for-loop without panicking");
    assert_eq!(format!("{walker:?}"), format!("{vm:?}"), "long for-loop walker and vm must agree");
    let stats = jit_state::stats();
    eprintln!("long for-loop multi-arm grain JIT stats: {stats:?}");
    eprintln!("long for-loop multi-arm majit: {}", jit_state::majit_diag_summary());
    assert!(stats.loops_compiled > 0, "the long for-loop must compile: {stats:?}");
}

/// Script-function calls must match the walker even after the merge point
/// warms. A residual abort mid-assignment used to skip remaining additions
/// or underflow the operand stack.
#[test]
#[cfg_attr(all(not(rhai_grain_jit_tables), not(rhai_grain_jit_require_tables)), ignore = "vacuous: no MAJIT_MIR_FRONTEND_LLBC tables were built")]
fn script_function_calls_match_the_walker() {
    if no_tables() {
        return;
    }

    for limit in [16, 1039, 2000, 5000] {
        jit_state::reset_stats();
        let source = format!("fn add(a, b) {{ a + b }} let s = 0; for i in 0..{limit} {{ s = add(s, i); }} s");
        let engine = Engine::new();
        let ast = engine.compile(&source).expect("the script parses");
        let program = Compiler::new().compile(&ast);
        let result = Vm::new(&engine).eval_with_scope(&mut Scope::new(), &program).expect("the script runs");
        let expected = limit * (limit - 1) / 2;
        assert_eq!(format!("{result:?}"), expected.to_string(), "limit {limit}, stats={:?}", jit_state::stats());
    }
}

/// A recursive script function that returns from the portal must not panic.
#[test]
#[cfg_attr(all(not(rhai_grain_jit_tables), not(rhai_grain_jit_require_tables)), ignore = "vacuous: no MAJIT_MIR_FRONTEND_LLBC tables were built")]
fn recursive_fibonacci_returns_from_the_portal() {
    if no_tables() {
        return;
    }

    jit_state::reset_stats();
    let engine = Engine::new();
    let ast = engine.compile("fn fib(n) { if n < 2 { n } else { fib(n-1) + fib(n-2) }} fib(12)").expect("fib parses");
    let program = Compiler::new().compile(&ast);
    let result = Vm::new(&engine).eval_with_scope(&mut Scope::new(), &program).expect("fib runs");
    assert_eq!(format!("{result:?}"), "144", "stats={:?}", jit_state::stats());
}
