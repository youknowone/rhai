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

use majit_translate::codewriter::insns;
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

    // The offset is what `register_dispatch_jitcode` refuses a portal for
    // lacking, and it cannot be recovered downstream: an operand byte may
    // equal the opcode byte, so a scan can land on a position that is not an
    // instruction start. It has to survive the same round trip as the bodies.
    let offset = jitcodes::portal_merge_point_offset().expect("the portal carries its own merge offset");
    let opcode = jitcodes::body_byte(portal, offset).expect("the offset is inside the portal's body");
    // The same assertion `validate_dispatch_jitcode_payload` makes. Asserting
    // only that the offset EXISTS would pass for an offset that survived the
    // round trip pointing at the wrong byte, which is the failure the
    // encoder-side recording exists to rule out.
    assert!(opcode == insns::BC_JIT_MERGE_POINT || opcode == insns::BC_JIT_MERGE_POINT_C, "offset {offset} holds {opcode}, not a merge-point opcode",);

    // Printed in both arms, because "passed" alone does not say which one ran
    // and the empty arm asserts almost nothing.
    eprintln!(
        "loaded {count} jitcodes; the driver names portal {portal}, whose merge \
         point opcode {opcode} sits at offset {offset}"
    );
}

/// The two halves a driver has to be handed for a body lowered ahead of time.
///
/// Both are build outputs of the same pass that wrote the bodies, so they are
/// present or absent together with them. Asserting that pairing is what
/// catches the state this repository was actually in: `insns.bin` was written
/// and never read, and `all_liveness.bin` was not written at all, so a driver
/// built from these tables would have walked bodies whose `-live-` offsets
/// index an empty stream and whose opcodes resolve to no name.
#[test]
fn the_liveness_parts_are_present_exactly_when_the_bodies_are() {
    jitcodes::install();
    let (insns, all_liveness) = jitcodes::liveness_parts();
    let count = jitcodes::count();

    assert_eq!(
        insns.is_empty(),
        count == 0,
        "{} jitcodes were lowered but the insn table holds {} entries; \
         `setup_insns` sizes its opcode-name table by the numbers in here, so \
         an empty one leaves every op in those bodies unnamed",
        count,
        insns.len(),
    );
    assert_eq!(
        all_liveness.is_empty(),
        count == 0,
        "{} jitcodes were lowered but the liveness stream is {} bytes; every \
         `-live-` op in those bodies carries a baked offset into it",
        count,
        all_liveness.len(),
    );

    eprintln!("insn table = {} entries, liveness stream = {} bytes, for {count} jitcodes", insns.len(), all_liveness.len(),);
}
