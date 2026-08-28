//! A `JitDriver` is built from the tables this crate ships, and accepts them.
//!
//! Everything here runs on the build artefact alone: `jit_merge_point` is still
//! an empty marker, so no live `Vm` reaches the state and nothing traces. What
//! is being asked is the question that comes first anyway -- whether the
//! metainterp accepts what the build produced. It has its own ways to say no,
//! and most of them are quiet: a schema whose red count disagrees with the
//! portal's `BC_JIT_MERGE_POINT` payload, a liveness stream whose opcode
//! numbering is not the one the bodies were assembled against, live values
//! whose kinds do not match the declared reds. The last of those is the one
//! this VM would hit first, because three of its four reds are Ref and the
//! `JitState` default types every red Int.

use majit_metainterp::{JitDriver, JitState};
use rhai::grain::{jit_state, jitcodes};

/// Any threshold; nothing here runs the loop often enough for it to matter.
const THRESHOLD: u32 = 1029;

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

/// The whole bring-up, in the order the steps have to run in.
///
/// `install` publishes the tables as the global build pool, which is the seed
/// `register_dispatch_jitcode_shared` adopts the portal out of.
/// `install_liveness_from_build_parts` and the registration both take the
/// staticdata through `Arc::get_mut` and panic once any trace has cloned it, so
/// they run before anything could have.
#[test]
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
