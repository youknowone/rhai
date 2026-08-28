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
//! this VM would hit first, because three of its four reds are Ref and the
//! `JitState` default types every red Int.

use majit_metainterp::JitState;
use rhai::grain::{jit_state, jitcodes, Compiler, Vm};
use rhai::{Engine, Scope};

/// The unchanged majit/RPython warm-loop threshold used by the Grain runtime.
const THRESHOLD: u32 = 1039;

/// The build lowered no tables, so there is nothing for a driver to accept.
fn no_tables() -> bool {
    let empty = jitcodes::count() == 0;
    if empty {
        eprintln!(
            "no LLBC artefact was named at build time; set MAJIT_MIR_FRONTEND_LLBC to a \
             `--features grain-jit` extraction to exercise the driver."
        );
    }
    empty
}

#[test]
fn the_descriptor_is_the_shape_the_build_recorded() {
    if no_tables() {
        return;
    }
    let jd = jit_state::grain_driver_descriptor();

    let greens: Vec<_> = jd.greens().iter().map(|var| var.tp).collect();
    let reds: Vec<_> = jd.reds().iter().map(|var| var.tp).collect();
    assert_eq!(greens, [majit_ir::Type::Int, majit_ir::Type::Ref]);
    assert_eq!(reds, [majit_ir::Type::Ref, majit_ir::Type::Ref, majit_ir::Type::Int, majit_ir::Type::Ref,],);
    // `warmspot.py:664` derives one history-kind char per red; the descriptor
    // does the same, so this is the same list twice and disagreeing would mean
    // the descriptor's two accounts of its reds had diverged.
    assert_eq!(jd.red_args_types, vec!['r', 'r', 'i', 'r']);
}

/// The gate that declines quietly.
///
/// `live_values_match_descriptor` refuses when the live-value count differs
/// from the red count, or when any live value's type differs from its red's --
/// and the callers return without tracing and without an error. The default
/// `live_value_types` types every red `Int`, which for this VM is wrong for
/// three of four, so this is what the override buys.
#[test]
fn the_live_values_carry_the_kinds_the_descriptor_declares() {
    if no_tables() {
        return;
    }
    let jd = jit_state::grain_driver_descriptor();
    let state = jit_state::GrainJitState::default();

    // One raw word per red, as the merge point would pass them.
    let env: Vec<i64> = vec![0x1000, 0x2000, 7, 0x3000];
    let meta = state.build_meta(jitcodes::portal_merge_point_offset().expect("the portal names its merge point"), &env);

    assert_eq!(state.extract_live(&meta), env);
    let types: Vec<_> = state.extract_live_values(&meta).iter().map(majit_ir::Value::get_type).collect();
    let declared: Vec<_> = jd.reds().iter().map(|var| var.tp).collect();
    assert_eq!(types, declared, "a live value whose type differs from its red's makes the driver decline silently",);
}

/// The state gate refuses an artifact before majit's backend-entry hook.
#[test]
fn the_state_actively_refuses_compiled_entry() {
    if no_tables() {
        return;
    }
    jit_state::reset_stats();
    let state = jit_state::GrainJitState::default();
    let header_pc = jitcodes::portal_merge_point_offset().expect("the portal names its merge point");
    let meta = state.build_meta(header_pc, &[0x1000, 0x2000, 7, 0x3000][..]);

    assert!(!state.is_compatible(&meta), "unbound residual targets make every compiled artifact incompatible",);
    let stats = jit_state::stats();
    assert_eq!(stats.compiled_entries_refused, 1);
    assert_eq!(stats.compiled_entries, 0);
}

/// The symbolic values the trace carries across the back edge.
#[test]
fn the_symbolic_reds_are_input_args_in_declaration_order() {
    if no_tables() {
        return;
    }
    let state = jit_state::GrainJitState::default();
    let header_pc = jitcodes::portal_merge_point_offset().expect("the portal names its merge point");
    let meta = state.build_meta(header_pc, &[0x1000, 0x2000, 7, 0x3000][..]);
    let sym = jit_state::GrainJitState::create_sym(&meta, header_pc);

    let expected: Vec<_> = jit_state::red_kinds().iter().enumerate().map(|(k, tp)| majit_ir::OpRef::input_arg_typed(k as u32, *tp)).collect();
    assert_eq!(sym.reds, expected);
    assert_eq!(jit_state::GrainJitState::collect_jump_args(&sym), expected, "the closing jump carries every red, in the order the label numbered them",);
    assert!(jit_state::GrainJitState::validate_close(&sym, &meta));
}

/// The build artifact names a complete portal before the live driver adopts it.
#[test]
fn the_driver_accepts_the_tables_this_crate_ships() {
    if no_tables() {
        return;
    }
    jitcodes::install();
    let table = jitcodes::all();
    let portal = jitcodes::portal_index().expect("a lowered table is named by a driver");
    let portal_jitcode = table.get(portal).expect("the driver names a portal the lowered table reaches").clone();
    assert_eq!(portal_jitcode.exec.jit_merge_point_offset, jitcodes::portal_merge_point_offset(),);
    assert_eq!(jitcodes::count(), table.len());
    eprintln!("embedded table holds {} jitcodes, portal {portal}", table.len());
}

/// A real VM loop reaches warmstate, opens a trace and records portal ops.
///
/// Compiled entry is a separate assertion: the embedded table still has no
/// symbolic fnaddr bindings, so `GrainJitState::is_compatible` must keep the
/// backend-entry hook at zero even if this or a later portal walk compiles.
#[test]
fn a_hot_grain_loop_consults_and_records_without_entering_compiled_code() {
    if no_tables() {
        return;
    }

    jit_state::reset_stats();
    let engine = Engine::new();
    let ast = engine
        .compile(
            "let i = 0; let total = 0; \
             while i < 4096 { total += i; i += 1; } total",
        )
        .expect("the hot loop parses");
    let program = Compiler::new().compile(&ast);
    let mut scope = Scope::new();
    let result = Vm::new(&engine).eval_with_scope(&mut scope, &program).expect("the hot loop runs");
    assert_eq!(format!("{result:?}"), "8386560");

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
    assert_eq!(stats.loops_compiled, 0, "the unbound residual boundary is reached before loop compilation: {stats:?}",);
    assert_eq!(stats.traces_aborted, stats.traces_started, "every trace attempt must end at the observed abort boundary: {stats:?}",);
    assert_eq!(stats.symbolic_residual_aborts, stats.traces_started, "the abort reason must be the actively refused symbolic residual: {stats:?}",);
    assert_eq!(stats.max_trace_ops, 5, "the current portal records five prefix ops before the residual boundary: {stats:?}",);
    assert_eq!(stats.ops_recorded, stats.max_trace_ops * stats.traces_started, "each attempt records the same five-op portal prefix: {stats:?}",);
    assert_eq!(stats.compiled_entries, 0, "the immediately-before-backend-entry hook must remain unreachable: {stats:?}",);
    assert_eq!(stats.non_owner_consultations_declined, 0, "the observation run must own its driver: {stats:?}",);
    assert_eq!(stats.reentrant_consultations_declined, 0, "the observation run must not hide nested consultations: {stats:?}",);
    assert!(stats.abort_reasons.contains("unbound_symbolic_residual="), "the report must name the abort boundary: {stats:?}",);
}
