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

use majit_metainterp::{EmbeddedJitCodeTable, init_global_build_descr_pool};
use majit_translate::jitcode::{BhDescr, JitCode};

/// Concatenated jitcode bodies, in allocation order.
static JITCODES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jitcodes.bin"));
/// `(names, offsets)` — `offsets[i]..offsets[i + 1]` is jitcode `i`'s body.
static JITCODES_INDEX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jitcodes_index.bin"));
static DESCRS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descrs.bin"));
static DESCRS_INDEX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descrs_index.bin"));
static JIT_DRIVERS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jit_drivers.bin"));
static SYMBOLIC_FNADDRS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/symbolic_fnaddrs.bin"));
/// Opcode name to opcode byte, as the lowering numbered them.
static INSNS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/insns.bin"));
/// The `(live_i, live_r, live_f)` byte stream, verbatim.
///
/// Every `-live-` op in a lowered body carries a two-byte offset into this,
/// baked at build time, so it is stored and handed over unencoded — a length
/// prefix would shift every one of those offsets by its own width.
static ALL_LIVENESS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/all_liveness.bin"));
static EFFECT_MINTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/effect_mints.bin"));

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

/// The build-time symbolic address table, retained for the diagnostic resolver.
fn symbolic_fnaddrs() -> &'static Vec<(i64, String)> {
    static PATHS: once_cell::race::OnceBox<Vec<(i64, String)>> = once_cell::race::OnceBox::new();
    PATHS.get_or_init(|| {
        Box::new(bincode::deserialize(SYMBOLIC_FNADDRS).expect("the symbolic fnaddr table decodes"))
    })
}

/// The path the build recorded for a symbolic call target, if it recorded one.
pub fn resolve_symbolic_fnaddr_path(fnaddr: i64) -> Option<&'static str> {
    symbolic_fnaddrs()
        .iter()
        .find_map(|(symbolic, path)| (*symbolic == fnaddr).then_some(path.as_str()))
}

/// Scalar ABI shims owned by this interpreter.
///
/// Build-time addresses are symbolic because build.rs runs in another
/// process.  These are the matching final-process addresses; every signature
/// is C ABI and returns at most one machine word, so no Rust enum or fat pointer
/// crosses the compiled-code boundary.
fn runtime_bindings() -> Vec<(&'static str, i64)> {
    vec![
        (
            "grain::program::Program::jit_identity",
            super::jit::program_jit_identity_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::code_byte",
            super::jit::code_byte as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::code_width",
            super::jit::code_width as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::code_u16",
            super::jit::code_u16 as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::code_u32",
            super::jit::code_u32 as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::code_form",
            super::jit::code_form as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::code_position_bits",
            super::jit::code_position_bits as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::program_constant",
            super::jit::program_constant as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::program_assign_op",
            super::jit::program_assign_op as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::track_operation_abi",
            super::jit::track_operation_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::program_token",
            super::jit::program_token as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::chain_tail",
            super::jit::chain_tail as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::assign_op_kind",
            super::jit::assign_op_kind as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::program_chain",
            super::jit::program_chain as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::program_switch",
            super::jit::program_switch as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::program_residual",
            super::jit::program_residual as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::program_name",
            super::jit::program_name as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::program_name",
            super::jit::program_name as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::program_function",
            super::jit::program_function as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::run_chain_abi",
            super::jit::run_chain_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::compiled_fn_is_plain_add",
            super::jit::compiled_fn_is_plain_add as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::plain_add_handled",
            super::jit::plain_add_handled as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::call_plain_add_abi",
            super::jit::call_plain_add_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::call_plain_add_ref_abi",
            super::jit::call_plain_add_ref_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::call_plain_abs_ref_abi",
            super::jit::call_plain_abs_ref_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::prepare_compiled_call_abi",
            super::jit::prepare_compiled_call_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::prepared_chunk_entry",
            super::jit::prepared_chunk_entry as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::finish_compiled_call_abi",
            super::jit::finish_compiled_call_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::call_syntactic_or_stacked_abi",
            super::jit::call_syntactic_or_stacked_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::call_by_reference_abi",
            super::jit::call_by_reference_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::store_scope_slot",
            super::jit::store_scope_slot as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::assign_local_abi",
            super::jit::assign_local_abi as *const () as usize as i64,
        ),
        (
            "majit_metainterp::request_walk_abort",
            super::jit::request_walk_abort_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::request_walk_abort_abi",
            super::jit::request_walk_abort_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::load_named_abi",
            super::jit::load_named_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::assign_named_abi",
            super::jit::assign_named_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::declare_local_abi",
            super::jit::declare_local_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::catch_bind_abi",
            super::jit::catch_bind_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::operator_builtin_abi",
            super::jit::operator_builtin_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::operator_builtin_handled",
            super::jit::operator_builtin_handled as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::name_is_abs",
            super::jit::name_is_abs as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::unary_builtin_abi",
            super::jit::unary_builtin_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::unary_builtin_handled",
            super::jit::unary_builtin_handled as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::make_closure_abi",
            super::jit::make_closure_abi as *const () as usize as i64,
        ),
        #[cfg(not(feature = "no_closure"))]
        (
            "rhai::grain::vm::jit::share_named_abi",
            super::jit::share_named_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::int_modulo",
            super::jit::int_modulo as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::int_modulo_result",
            super::jit::int_modulo_result as *const () as usize as i64,
        ),
        (
            "core::cmp::impls::<Impl>::max",
            super::jit::int_max as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::vm_depth",
            super::jit::vm_depth as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_dispatch",
            super::jit::switch_dispatch as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_pop_subject",
            super::jit::switch_pop_subject as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_target",
            super::jit::switch_target as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_subject_kind",
            super::jit::switch_subject_kind as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_subject_hash",
            super::jit::switch_subject_hash as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_case_count",
            super::jit::switch_case_count as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_case_hash",
            super::jit::switch_case_hash as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_case_has_int",
            super::jit::switch_case_has_int as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_case_int_key",
            super::jit::switch_case_int_key as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_case_target",
            super::jit::switch_case_target as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_default",
            super::jit::switch_default as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_range_count",
            super::jit::switch_range_count as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_range_from",
            super::jit::switch_range_from as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_range_to",
            super::jit::switch_range_to as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_range_inclusive",
            super::jit::switch_range_inclusive as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::switch_range_target",
            super::jit::switch_range_target as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::dynamic_as_fast",
            super::jit::dynamic_as_fast as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::fast_int",
            super::jit::fast_int as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::fast_bool",
            super::jit::fast_bool as *const () as usize as i64,
        ),
        #[cfg(not(feature = "no_float"))]
        (
            "rhai::grain::vm::jit::fast_float",
            super::jit::fast_float as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::array_len",
            super::jit::array_len as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::iterator_last_mut",
            super::jit::iterator_last_mut as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::iter_next_store",
            super::jit::iter_next_store as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::iter_next_produced",
            super::jit::iter_next_produced as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::store_scope_int",
            super::jit::store_scope_int as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::store_int_range_next",
            super::jit::store_int_range_next as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::store_shared_from_stack",
            super::jit::store_shared_from_stack as *const () as usize as i64,
        ),
        (
            "__len",
            super::jit::refuse_synthetic_len as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::refuse_synthetic_len",
            super::jit::refuse_synthetic_len as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::refuse_deref_write",
            super::jit::refuse_deref_write as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::stash_ok_result",
            super::jit::stash_ok_result as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::take_finished_result",
            super::jit::take_finished_result as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::array_entry_mut",
            super::jit::array_entry_mut as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::store_builtin_available",
            super::jit::store_builtin_available as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::store_builtin_apply",
            super::jit::store_builtin_apply as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::fast_operators",
            super::jit::fast_operators as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::scope_from_frame",
            super::jit::scope_from_frame as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::pin_scope_with_vm",
            super::jit::pin_scope_with_vm as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::scope_len",
            super::jit::scope_len as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::operand_stack_len",
            super::jit::operand_stack_len as *const () as usize as i64,
        ),
        (
            // The declaration-owned method path, not the adapter's path:
            // this is the native entry of the already-lowered helper.
            "grain::vm::Vm::grow_stack",
            super::jit::grow_stack_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::grow_stack_abi",
            super::jit::grow_stack_abi as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::push_fast_int",
            super::jit::push_fast_int as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::push_fast_bool",
            super::jit::push_fast_bool as *const () as usize as i64,
        ),
        #[cfg(not(feature = "no_float"))]
        (
            "rhai::grain::vm::jit::push_fast_float",
            super::jit::push_fast_float as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::push_fast_unit",
            super::jit::push_fast_unit as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::push_from_cell",
            super::jit::push_from_cell as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::operand_stack_entry",
            super::jit::operand_stack_entry as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::array_entry",
            super::jit::array_entry as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::operand_stack_take",
            super::jit::operand_stack_take as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::operand_stack_store",
            super::jit::operand_stack_store as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::operand_stack_store_int",
            super::jit::operand_stack_store_int as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::operand_stack_store_bool",
            super::jit::operand_stack_store_bool as *const () as usize as i64,
        ),
        #[cfg(not(feature = "no_float"))]
        (
            "rhai::grain::vm::jit::operand_stack_store_float",
            super::jit::operand_stack_store_float as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::operand_stack_store_unit",
            super::jit::operand_stack_store_unit as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::dynamic_store",
            super::jit::dynamic_store as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::truncate_stack",
            super::jit::truncate_stack as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::iterators_len",
            super::jit::iterators_len as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::iterators_pop",
            super::jit::iterators_pop as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::iterators_truncate",
            super::jit::iterators_truncate as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::scope_rewind",
            super::jit::scope_rewind as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::sizes_truncate",
            super::jit::sizes_truncate as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::handlers_len",
            super::jit::handlers_len as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::sizes_len",
            super::jit::sizes_len as *const () as usize as i64,
        ),
        (
            "rhai::grain::vm::jit::scope_entry",
            super::jit::scope_entry as *const () as usize as i64,
        ),
    ]
}

/// The build-time tables, joined into runtime shells.
///
/// Published once and never rebuilt: the tables are a frozen build artefact,
/// and `EmbeddedJitCodeTable` hands back a `&'static` because the descr pool
/// is addressed by index from inside jitcode bodies for the life of the
/// process.
fn table() -> &'static EmbeddedJitCodeTable {
    // Descriptor setup mutates shared GcCache objects. A racing initializer
    // may not run it twice while another thread already traces the image.
    static TABLE: std::sync::OnceLock<&'static EmbeddedJitCodeTable> = std::sync::OnceLock::new();
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
        if std::env::var_os("MAJIT_DESCR_TRACE").is_some() {
            for (index, descr) in descrs.iter().enumerate() {
                let rendered = format!("{descr:?}");
                if (215..=230).contains(&index)
                    || rendered.contains("__pos_2")
                    || rendered.contains("access")
                {
                    eprintln!("[majit-descr] {index} {rendered}");
                }
            }
        }

        let bindings = runtime_bindings();
        if std::env::var_os("MAJIT_RESIDUAL_TRACE").is_some() {
            for (name, address) in &bindings {
                eprintln!("[majit-binding] {address:#x} {name}");
            }
        }
        let effect_mints: Vec<majit_ir::effectinfo::DescrMintEntry> =
            bincode::deserialize(EFFECT_MINTS).expect("the effect descriptor image decodes");
        EmbeddedJitCodeTable::materialize_with_frozen_effects(
            &jitcodes,
            descrs,
            symbolic_fnaddrs(),
            &bindings,
            &effect_mints,
        )
    })
}

/// Publish the tables as the process-global build-time descr pool.
///
/// The pool is what numbers the flat callee registry a driver seeds from, so
/// this has to have run before any driver is built or the registry starts
/// empty and every `j` operand resolves to nothing. Idempotent, and cheap
/// after the first call.
pub fn install() {
    majit_metainterp::set_symbolic_fnaddr_path_resolver(Some(resolve_symbolic_fnaddr_path));
    init_global_build_descr_pool(table());
}

/// The driver the lowering compiled, or `None` when it compiled none.
///
/// Everything a run-time driver needs to describe this loop is decided at
/// build time and travels in here: which jitcode is the portal, and the green
/// and red names paired with the operand kinds the merge point's own operands
/// had. Re-deriving any of it on this side would be a second answer that
/// nothing keeps in step with the first.
pub fn driver() -> Option<majit_translate::CompiledJitDriver> {
    let drivers: Vec<majit_translate::CompiledJitDriver> =
        bincode::deserialize(JIT_DRIVERS).expect("the driver table decodes");
    drivers.into_iter().next()
}

/// The index of the portal jitcode — the one carrying the merge point.
///
/// Read from the driver table the lowering wrote rather than rediscovered by
/// scanning for a name or a flag: the lowering already decided which jitcode
/// is the portal, and a second answer here could disagree with the operands
/// that were numbered against the first.
pub fn portal_index() -> Option<usize> {
    driver().map(|driver| driver.main_jitcode_index)
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

/// Every lowered jitcode, in allocation order.
///
/// This is the list `MetaInterp::install_jitcodes` takes, and the order is the
/// contract: a `j` operand is an index into it, so a caller that reorders or
/// filters it resolves callees to the wrong bodies and says nothing about it.
pub fn all() -> Vec<Arc<majit_metainterp::JitCode>> {
    table().jitcodes().to_vec()
}

/// How many jitcodes the build lowered. Zero when it lowered none.
pub fn count() -> usize {
    table().jitcodes().len()
}

/// The two halves of `MetaInterp::install_liveness_from_build_parts`.
///
/// A body lowered ahead of time carries its liveness as an offset into one
/// shared stream, and its opcodes as the numbers the lowering assigned. Both
/// are decided at build time, so a driver that walks such a body has to be
/// handed them rather than deriving them from a runtime `Assembler` — that is
/// the difference between this and the proc-macro route's
/// `install_canonical_liveness`.
///
/// The map is rebuilt on each call, so a caller installs once and keeps the
/// driver rather than asking again.
pub fn liveness_parts() -> (
    majit_metainterp::indexmap::IndexMap<String, u8>,
    &'static [u8],
) {
    // Written through a `BTreeMap` for reproducible bytes; the order it comes
    // back in carries no meaning, because each entry's VALUE is the opcode
    // number `setup_insns` indexes its table by.
    let insns: std::collections::BTreeMap<String, u8> =
        bincode::deserialize(INSNS).expect("the insn table decodes");
    (insns.into_iter().collect(), ALL_LIVENESS)
}
