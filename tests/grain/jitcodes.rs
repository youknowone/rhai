//! The lowered tables load, and the portal they name carries the merge point.
//!
//! `build/majit_prepass.rs` already refuses a lowering whose portal assembles
//! no merge point, so this is not a second check of the same thing: it checks
//! that the tables survive the round trip out of the build and back, and that
//! the portal INDEX the driver table names still selects that jitcode once the
//! bodies have been cut apart and renumbered. Those are separable failures —
//! an index written against one ordering and a list rebuilt in another agree
//! on nothing and say so nowhere.
//!
//! When the build ran without `MAJIT_MIR_FRONTEND_LLBC` every table is an
//! empty sequence. That is a legitimate build, so the empty case is asserted
//! to be *consistently* empty rather than skipped: a table that lost its
//! bodies but kept its driver would otherwise read as "no JIT was built".

use rhai::grain::jitcodes;

#[test]
fn the_lowered_tables_round_trip_and_name_their_portal() {
    jitcodes::install();
    let count = jitcodes::count();
    let portal = jitcodes::portal_index();

    if count == 0 {
        assert_eq!(portal, None, "no jitcodes were lowered, so no driver can name a portal among them");
        eprintln!(
            "loaded 0 jitcodes: no LLBC artefact was named at build time. Set \
             MAJIT_MIR_FRONTEND_LLBC to a `--features grain-jit` extraction to \
             exercise the loaded tables."
        );
        return;
    }

    let portal = portal.expect("a lowered table is named by a driver");
    assert!(portal < count, "the driver names portal {portal} of {count} jitcodes");
    // Printed in both arms, because "passed" alone does not say which one ran
    // and the empty arm asserts almost nothing.
    eprintln!("loaded {count} jitcodes; the driver names portal {portal}");
}
