//! The lowered jitcode tables, at run time.
//!
//! `build/majit_prepass.rs` lowers the dispatch loop and writes the tables it
//! produced into `OUT_DIR`. This reads them back, and then hands them
//! straight to `majit-metainterp`'s own table type. It deliberately does no
//! more than that: joining the serialized lists into runtime shells, numbering
//! the callees, and swapping the build's symbolic function addresses for the
//! real ones are all things [`EmbeddedJitCodeTable`] already does, and the
//! second implementation of any of them would be the one that drifts.
//!
//! What the deserialization has to get right is only the pairing. Every `j`
//! operand in the descr pool is an index into the jitcode list in allocation
//! order, so a list rebuilt in any other order resolves to the wrong callee
//! and says nothing about it.

use std::sync::Arc;

use majit_metainterp::{init_global_build_descr_pool, EmbeddedJitCodeTable};
use majit_translate::jitcode::{BhDescr, JitCode};

/// Concatenated jitcode bodies, in allocation order.
static JITCODES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jitcodes.bin"));
/// `(names, offsets)` — `offsets[i]..offsets[i + 1]` is jitcode `i`'s body.
static JITCODES_INDEX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jitcodes_index.bin"));
static DESCRS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descrs.bin"));
static DESCRS_INDEX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descrs_index.bin"));
static JIT_DRIVERS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jit_drivers.bin"));
static SYMBOLIC_FNADDRS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/symbolic_fnaddrs.bin"));

/// One table's entries, as byte ranges over its concatenated bodies.
///
/// `offsets` has one more entry than there are records, so a record's bounds
/// are always a pair and the last offset is the length. A build that wrote no
/// artefact writes `[0]`, which yields no ranges rather than an error.
fn entries<'a>(bodies: &'a [u8], offsets: &'a [u32]) -> impl Iterator<Item = &'a [u8]> + 'a {
    offsets
        .windows(2)
        .map(|bounds| &bodies[bounds[0] as usize..bounds[1] as usize])
}

/// The build-time tables, joined into runtime shells.
///
/// Published once and never rebuilt: the tables are a frozen build artefact,
/// and `EmbeddedJitCodeTable` hands back a `&'static` because the descr pool
/// is addressed by index from inside jitcode bodies for the life of the
/// process.
fn table() -> &'static EmbeddedJitCodeTable {
    static TABLE: once_cell::race::OnceBox<&'static EmbeddedJitCodeTable> =
        once_cell::race::OnceBox::new();
    TABLE.get_or_init(|| {
        let (_names, offsets): (Vec<String>, Vec<u32>) =
            bincode::deserialize(JITCODES_INDEX).expect("the jitcode index decodes");
        let jitcodes: Vec<Arc<JitCode>> = entries(JITCODES, &offsets)
            .map(|body| Arc::new(bincode::deserialize::<JitCode>(body).expect("a jitcode decodes")))
            .collect();

        let descr_offsets: Vec<u32> =
            bincode::deserialize(DESCRS_INDEX).expect("the descr index decodes");
        let descrs: Vec<BhDescr> = entries(DESCRS, &descr_offsets)
            .map(|body| bincode::deserialize::<BhDescr>(body).expect("a descr decodes"))
            .collect();

        let symbolic: Vec<(i64, String)> =
            bincode::deserialize(SYMBOLIC_FNADDRS).expect("the symbolic fnaddr table decodes");

        // No runtime bindings yet: nothing on this side publishes an ABI shim
        // for a symbolic path, so every unbound callee keeps the build's
        // sentinel and a trace that reaches one is expected to decline rather
        // than call it. Binding them is what makes those callees reachable.
        Box::new(EmbeddedJitCodeTable::materialize_with_symbolic_fnaddrs(
            &jitcodes,
            descrs,
            &symbolic,
            &[],
        ))
    })
}

/// Publish the tables as the process-global build-time descr pool.
///
/// The pool is what numbers the flat callee registry a driver seeds from, so
/// this has to have run before any driver is built or the registry starts
/// empty and every `j` operand resolves to nothing. Idempotent, and cheap
/// after the first call.
pub fn install() {
    init_global_build_descr_pool(table());
}

/// The index of the portal jitcode — the one carrying the merge point.
///
/// Read from the driver table the lowering wrote rather than rediscovered by
/// scanning for a name or a flag: the lowering already decided which jitcode
/// is the portal, and a second answer here could disagree with the operands
/// that were numbered against the first.
pub fn portal_index() -> Option<usize> {
    let drivers: Vec<majit_translate::CompiledJitDriver> =
        bincode::deserialize(JIT_DRIVERS).expect("the driver table decodes");
    drivers.first().map(|driver| driver.main_jitcode_index)
}

/// Where the portal's `jit_merge_point` opcode byte sits in its own body.
///
/// `JitDriver::register_dispatch_jitcode` refuses a portal that cannot say,
/// and a consumer cannot recover the answer by scanning the body: an operand
/// byte may equal the opcode byte, so only the encoder knows which position is
/// an instruction start. The assembler records it while it reserves that byte
/// and `JitCode::from_canonical` carries it across, so `None` here means the
/// lowering produced a portal the driver will not accept.
pub fn portal_merge_point_offset() -> Option<usize> {
    let portal = portal_index()?;
    table().jitcodes().get(portal)?.exec.jit_merge_point_offset
}

/// The byte at `offset` in one lowered jitcode's body, or `None` when either
/// index is out of range.
pub fn body_byte(index: usize, offset: usize) -> Option<u8> {
    table()
        .jitcodes()
        .get(index)
        .and_then(|jc| jc.code.get(offset).copied())
}

/// How many jitcodes the build lowered. Zero when it lowered none.
pub fn count() -> usize {
    table().jitcodes().len()
}
