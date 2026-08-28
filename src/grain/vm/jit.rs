//! The tracing-JIT merge point.
//!
//! A meta-tracing JIT does not read this VM's instructions; it traces the
//! *interpreter* running them, and the merge point is where the interpreter
//! tells the tracer that one iteration of its dispatch loop has begun and what
//! the machine state is at that moment. Everything the tracer needs to decide
//! "have I been here before" is in the arguments.
//!
//! The arguments split two ways. `pc` and `program` are *green*: constant for
//! any one trace, so the tracer specialises on their values and a loop that
//! comes back to the same instruction of the same program is the same loop.
//! The rest are *red*: they vary per iteration and the trace carries them as
//! live values. Neither the split nor the order is inferable from the source,
//! so a consumer that lowers this VM declares both, and the declaration and
//! this signature have to agree position for position.
//!
//! The untranslated body also consults majit's warm state. The lowering still
//! replaces the call itself with the merge-point opcode, which is why the type
//! name, signature and `inline(never)` remain part of the build contract.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use majit_ir::{GreenKey, GreenType};
use majit_metainterp::{JitDriver, TraceAction};

use super::Vm;
use super::{jit_state, jitcodes};
use crate::grain::Program;
use crate::Scope;

/// RPython's default warm-loop threshold (`warmstate.rs` uses the same value).
const THRESHOLD: u32 = 1039;

struct Runtime {
    driver: JitDriver<jit_state::GrainJitState>,
    state: jit_state::GrainJitState,
    portal: Arc<majit_metainterp::JitCode>,
    portal_merge_point: usize,
    green_types: Vec<GreenType>,
    symbolic_residual_opcodes: [bool; 256],
}

impl Runtime {
    fn new() -> Option<Self> {
        if jitcodes::count() == 0 {
            return None;
        }

        // The portal backpointer is process-global and write-once, while
        // JitDriver itself is thread-affine. Until majit can let several TLS
        // drivers adopt one portal, the first consulting thread owns tracing;
        // other threads decline instead of attempting a second registration.
        static OWNER: OnceLock<std::thread::ThreadId> = OnceLock::new();
        let current = std::thread::current().id();
        if OWNER.get_or_init(|| current) != &current {
            bump_stats(|stats| stats.non_owner_consultations_declined += 1);
            return None;
        }

        // The build tables, liveness and dispatch identity have to be complete
        // before a trace clones staticdata. Keep them beside the state in one
        // thread-local owner because JitDriver is not a shared runtime object.
        jitcodes::install();
        let descriptor = jit_state::grain_driver_descriptor();
        let green_types = descriptor.green_args_spec();
        let mut driver = JitDriver::with_descriptor(THRESHOLD, descriptor);
        driver.ensure_descriptor_registered();
        let (insns, all_liveness) = jitcodes::liveness_parts();
        driver
            .meta_interp_mut()
            .install_liveness_from_build_parts(&insns, all_liveness);
        let mut symbolic_residual_opcodes = [false; 256];
        for (name, opcode) in &insns {
            if name.starts_with("residual_call") {
                symbolic_residual_opcodes[*opcode as usize] = true;
            }
        }

        let table = jitcodes::all();
        let portal_index = jitcodes::portal_index().expect("the driver names its portal");
        let portal = table
            .get(portal_index)
            .expect("the driver portal is present in the embedded table")
            .clone();
        driver.register_dispatch_jitcode_shared(&portal);
        let registered = driver
            .dispatch_jitcode()
            .expect("registration names the Grain portal");
        assert!(
            Arc::ptr_eq(registered, &portal),
            "the driver must adopt the embedded table's portal identity",
        );
        assert_eq!(
            driver.meta_interp_mut().jitcodes().len(),
            table.len(),
            "registration must publish the complete embedded table",
        );

        driver.set_on_trace_abort(|_, _| bump_stats(|stats| stats.traces_aborted += 1));
        driver.set_on_compile_loop(|_, _, _, _| bump_stats(|stats| stats.loops_compiled += 1));
        driver.set_on_compiled_entry(|_, _| bump_stats(|stats| stats.compiled_entries += 1));

        Some(Self {
            driver,
            state: jit_state::GrainJitState::default(),
            portal,
            portal_merge_point: jitcodes::portal_merge_point_offset()
                .expect("the registered portal names its merge point"),
            green_types,
            symbolic_residual_opcodes,
        })
    }
}

#[derive(Default)]
struct Counters {
    merge_points_consulted: usize,
    entry_door_interpret: usize,
    entry_door_already_tracing: usize,
    traces_started: usize,
    traces_aborted: usize,
    loops_compiled: usize,
    ops_recorded: usize,
    max_trace_ops: usize,
    compiled_entries: usize,
    compiled_entries_refused: usize,
    symbolic_residual_aborts: usize,
    reentrant_consultations_declined: usize,
    non_owner_consultations_declined: usize,
    abort_reasons_before: Vec<(&'static str, u64)>,
}

thread_local! {
    // JitDriver contains thread-affine tracing state. A trace can call back
    // into interpreter code, so `try_borrow_mut` declines a nested
    // consultation instead of panicking while the outer trace remains active.
    static RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };
    static COUNTERS: RefCell<Counters> = RefCell::new(Counters::default());
}

fn bump_stats(f: impl FnOnce(&mut Counters)) {
    COUNTERS.with(|cell| f(&mut cell.borrow_mut()));
}

pub(super) fn record_compiled_entry_refusal() {
    bump_stats(|stats| stats.compiled_entries_refused += 1);
}

pub(super) fn reset_stats() {
    COUNTERS.with(|cell| {
        *cell.borrow_mut() = Counters {
            abort_reasons_before: majit_metainterp::embed::abort_reasons(),
            ..Counters::default()
        };
    });
}

pub(super) fn stats() -> jit_state::GrainJitStats {
    COUNTERS.with(|cell| {
        let stats = cell.borrow();
        let mut abort_reasons = majit_metainterp::embed::render_abort_delta(
            &stats.abort_reasons_before,
            &majit_metainterp::embed::abort_reasons(),
        );
        if stats.symbolic_residual_aborts > 0 {
            if !abort_reasons.is_empty() {
                abort_reasons.push(' ');
            }
            abort_reasons.push_str(&format!(
                "unbound_symbolic_residual={}",
                stats.symbolic_residual_aborts
            ));
        }
        jit_state::GrainJitStats {
            merge_points_consulted: stats.merge_points_consulted,
            entry_door_interpret: stats.entry_door_interpret,
            entry_door_already_tracing: stats.entry_door_already_tracing,
            traces_started: stats.traces_started,
            traces_aborted: stats.traces_aborted,
            loops_compiled: stats.loops_compiled,
            ops_recorded: stats.ops_recorded,
            max_trace_ops: stats.max_trace_ops,
            compiled_entries: stats.compiled_entries,
            compiled_entries_refused: stats.compiled_entries_refused,
            symbolic_residual_aborts: stats.symbolic_residual_aborts,
            reentrant_consultations_declined: stats.reentrant_consultations_declined,
            non_owner_consultations_declined: stats.non_owner_consultations_declined,
            abort_reasons,
        }
    })
}

fn report_if_enabled(event: &str) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var_os("RHAI_GRAIN_JIT_STATS").is_some()) {
        eprintln!("[grain-jit] {event}: {:?}", stats());
    }
}

/// The driver `run_frame`'s loop reports to.
///
/// Zero-sized: the live driver/state are thread-local because a consumer
/// recognises this type's name as the marker receiver. Renaming it silently
/// disconnects the loop from the JIT rather than failing to build.
pub struct GrainJitDriver;

impl GrainJitDriver {
    /// One iteration of the dispatch loop is about to run.
    ///
    /// Greens `(pc, program)` first, then reds `(vm, scope, base, reached)`.
    #[inline(never)]
    pub fn jit_merge_point(
        &self,
        pc: usize,
        program: &Program,
        vm: &Vm<'_>,
        scope: &Scope<'_>,
        base: usize,
        reached: &usize,
    ) {
        bump_stats(|stats| stats.merge_points_consulted += 1);

        let env = [
            vm as *const Vm<'_> as usize as i64,
            scope as *const Scope<'_> as usize as i64,
            base as i64,
            reached as *const usize as usize as i64,
        ];
        assert_eq!(
            env.len(),
            jit_state::red_kinds().len(),
            "the merge point passes one raw word per declared red",
        );

        let mut report_event = None;
        RUNTIME.with(|cell| {
            let Ok(mut slot) = cell.try_borrow_mut() else {
                bump_stats(|stats| stats.reentrant_consultations_declined += 1);
                report_event = Some("reentrant consultation declined");
                return;
            };
            if slot.is_none() {
                *slot = Runtime::new();
            }
            let Some(runtime) = slot.as_mut() else {
                return;
            };

            let green_values = vec![pc as i64, program as *const Program as usize as i64];
            assert_eq!(
                green_values.len(),
                runtime.green_types.len(),
                "the merge point passes one value per declared green",
            );
            let green_hash =
                majit_metainterp::green_key_hash_typed(&green_values, &runtime.green_types);
            let green_key = || GreenKey {
                values: green_values.clone(),
                types: runtime.green_types.clone(),
            };

            let was_tracing = runtime.driver.is_tracing();
            let resume = runtime.driver.back_edge_structured(
                green_hash,
                green_key,
                runtime.portal_merge_point,
                &mut runtime.state,
                &env,
                || {},
            );
            let started = !was_tracing && runtime.driver.is_tracing();
            assert!(
                resume.is_none() || started,
                "a non-tracing resume would mean compiled code bypassed GrainJitState's refusal",
            );
            bump_stats(|stats| {
                if started {
                    stats.traces_started += 1;
                } else if was_tracing {
                    stats.entry_door_already_tracing += 1;
                } else {
                    stats.entry_door_interpret += 1;
                }
            });

            if runtime.driver.is_tracing() {
                let portal = Arc::clone(&runtime.portal);
                let header_pc = runtime.portal_merge_point;
                let green_types = runtime.green_types.clone();
                let symbolic_residual_opcodes = runtime.symbolic_residual_opcodes;
                runtime.driver.merge_point(|meta, sym| {
                    let ctx = meta.trace_ctx().expect("an active trace owns a context");
                    let before = ctx.num_ops();
                    assert_eq!(
                        sym.reds.len(),
                        jit_state::red_kinds().len(),
                        "the symbolic state carries every declared red",
                    );

                    // `trace_jitcode`'s empty-argument form is only usable for
                    // a zero-argument portal. `run_frame` starts with
                    // (vm, program, scope, base, reached, start); seed those
                    // registers from the marker's greens and reds, and verify
                    // their kinds against the jitcode's own call descriptor.
                    let green_ir: Vec<_> = green_types
                        .iter()
                        .copied()
                        .map(majit_ir::green_type_to_ir)
                        .collect();
                    let red_ir = jit_state::red_kinds();
                    let typed = [
                        (red_ir[0], sym.reds[0], env[0]),
                        (
                            green_ir[1],
                            ctx.const_ref(program as *const Program as usize as i64),
                            program as *const Program as usize as i64,
                        ),
                        (red_ir[1], sym.reds[1], env[1]),
                        (red_ir[2], sym.reds[2], env[2]),
                        (red_ir[3], sym.reds[3], env[3]),
                        (green_ir[0], ctx.const_int(pc as i64), pc as i64),
                    ];
                    let argboxes: Vec<_> = typed
                        .into_iter()
                        .map(|(tp, op, value)| {
                            (
                                majit_metainterp::JitArgKind::from_type(tp)
                                    .expect("a portal argument is not void"),
                                op,
                                value,
                            )
                        })
                        .collect();
                    let descriptor_kinds: Vec<_> = portal
                        .calldescr
                        .arg_classes
                        .bytes()
                        .map(|kind| match kind {
                            b'i' => majit_metainterp::JitArgKind::Int,
                            b'r' => majit_metainterp::JitArgKind::Ref,
                            b'f' => majit_metainterp::JitArgKind::Float,
                            other => panic!("unsupported portal argument class {other:?}"),
                        })
                        .collect();
                    assert_eq!(
                        argboxes.iter().map(|arg| arg.0).collect::<Vec<_>>(),
                        descriptor_kinds,
                        "the live portal arguments match the lowered function signature",
                    );

                    let mut frame = majit_metainterp::MIFrame::setup(
                        Arc::clone(&portal),
                        header_pc,
                        None,
                        Some(ctx),
                    );
                    frame.setup_call(&argboxes);
                    let mut stack = majit_metainterp::StandaloneFrameStack::new();
                    stack.frames.push(frame);
                    let trace_runtime = majit_metainterp::ClosureRuntime::new(|label| label);
                    majit_metainterp::JitCodeSym::begin_portal_op(sym, header_pc);

                    let action = loop {
                        let Some(frame) = stack.frames.frames.last() else {
                            break TraceAction::Continue;
                        };
                        if let Some(opcode) = frame.jitcode.code.get(frame.code_cursor) {
                            if symbolic_residual_opcodes[*opcode as usize] {
                                // The current majit walker executes residuals
                                // to obtain their concrete shadows while it
                                // records them. An unbound symbolic target
                                // therefore cannot be stepped safely: stop at
                                // the call boundary, retaining the prefix the
                                // real walker recorded so far.
                                bump_stats(|stats| stats.symbolic_residual_aborts += 1);
                                break TraceAction::Abort;
                            }
                        }
                        let step = {
                            let mut machine = majit_metainterp::JitCodeMachine::<
                                jit_state::GrainSym,
                                _,
                            >::with_framestack(
                                &mut stack.frames, &[], &[]
                            );
                            machine.run_one_step(ctx, sym, &trace_runtime)
                        };
                        match step {
                            TraceAction::Continue => {}
                            other => break other,
                        }
                    };
                    let after = ctx.num_ops();
                    let recorded = after.saturating_sub(before);
                    bump_stats(|stats| {
                        stats.ops_recorded += recorded;
                        stats.max_trace_ops = stats.max_trace_ops.max(recorded);
                    });
                    action
                });
            }

            if started {
                report_event = Some(if runtime.driver.is_tracing() {
                    "trace started"
                } else {
                    "trace decision"
                });
            }
        });

        if let Some(event) = report_event {
            report_if_enabled(event);
        }
    }
}
