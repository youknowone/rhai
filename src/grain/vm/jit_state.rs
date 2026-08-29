//! The run-time state a tracing driver reads this VM through.
//!
//! The build lowered `run_frame` to jitcode and recorded which of the merge
//! point's operands are green and which red, with the kind each one's operand
//! actually had. This is the other half: the types a `JitDriver` needs to walk
//! that jitcode -- how the reds are handed to it, how it mints a symbolic value
//! for each, and which of them the closing jump carries.
//!
//! Nothing here restates the driver's shape. Every count, name and kind is read
//! back out of [`jitcodes::driver`], because the build already checked its
//! declaration against the portal graph's own operands; a second declaration
//! here would agree with that one only until one of the two was edited.
//!
//! Compiled entry is deliberately disabled while the embedded jitcodes still
//! carry unbound symbolic function addresses. Recording such a call would be
//! safe, but majit's current walker also invokes it to obtain a concrete
//! shadow. The trace path stops before that instruction, and the state refuses
//! the driver's compatibility check before any backend body can run.

use majit_ir::{OpRef, Type};
use majit_metainterp::{JitCodeSym, JitDriverStaticData, JitState};

use super::jitcodes;

/// What the grain driver did on this thread since [`reset_stats`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GrainJitStats {
    /// Dispatch-loop markers that reached the runtime consultation.
    pub merge_points_consulted: usize,
    /// Entry-door calls that kept interpreting without starting a trace.
    pub entry_door_interpret: usize,
    /// Entry-door calls declined because a trace was already active.
    pub entry_door_already_tracing: usize,
    /// Warm-loop decisions that opened a trace.
    pub traces_started: usize,
    /// Traces the metainterpreter abandoned before compilation.
    pub traces_aborted: usize,
    /// Loop traces successfully compiled.
    pub loops_compiled: usize,
    /// Operations appended by portal walks, before optimization.
    pub ops_recorded: usize,
    /// Largest active trace observed after one portal walk.
    pub max_trace_ops: usize,
    /// Calls that reached majit's immediately-before-backend-entry hook.
    pub compiled_entries: usize,
    /// Compiled artifacts rejected by this state's compatibility gate.
    pub compiled_entries_refused: usize,
    /// Walks stopped before an unbound symbolic residual callee was invoked.
    pub symbolic_residual_aborts: usize,
    /// Consultations answered without asking the driver, because `pc` had
    /// advanced on its own and no trace was open. The door is what upstream
    /// reaches from `can_enter_jit`; these are the steps that are not back
    /// edges, and each one costs a compare rather than four allocations.
    pub forward_steps_skipped: usize,
    /// Nested consultations skipped instead of panicking on a TLS borrow.
    pub reentrant_consultations_declined: usize,
    /// Consultations skipped because another thread owns the process's driver.
    pub non_owner_consultations_declined: usize,
    /// Majit's abort-reason counter deltas for the observation window.
    pub abort_reasons: String,
}

/// Snapshot this thread's grain-JIT counters.
pub fn stats() -> GrainJitStats {
    super::jit::stats()
}

/// Start a fresh observation window for this thread's grain-JIT counters.
pub fn reset_stats() {
    super::jit::reset_stats();
}

/// Majit's own warm-up census, as `label=count` pairs, non-zero slots only.
///
/// Process-global and cumulative -- the counters live in majit, not here, so
/// [`reset_stats`] does not scope them. What it answers is which of the
/// driver's doors refused: `mst_entered` beside `mst_sync_before_false` and
/// `mst_live_values_mismatch` says whether a trace was turned away before it
/// could start, which no counter on this side can distinguish from a warm
/// threshold that was never crossed.
#[must_use]
pub fn majit_diag_summary() -> String {
    majit_metainterp::mc_diag_summary()
        .split_whitespace()
        .filter(|pair| !pair.ends_with("=0"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The compiled driver, held for the life of the process.
///
/// [`JitDriverStaticData::new`] borrows the green and red names, so they have
/// to outlive the descriptor rather than the call that builds it.
fn driver() -> &'static majit_translate::CompiledJitDriver {
    static DRIVER: std::sync::OnceLock<majit_translate::CompiledJitDriver> =
        std::sync::OnceLock::new();
    DRIVER.get_or_init(|| {
        jitcodes::driver().expect(
            "the build lowered no JIT driver; a state describing one has nothing to describe",
        )
    })
}

/// The kind of each red, in the order the merge point passes them.
///
/// This is the list [`JitDriver::live_values_match_descriptor`] checks the live
/// values against. Its default in [`JitState`] types every red `Int`, which for
/// this VM would misdescribe three of the four and make the driver decline to
/// trace without saying so.
pub fn red_kinds() -> &'static [Type] {
    &driver().red_args_types
}

/// The portal descriptor, built from what the lowering recorded.
///
/// `greens[0]` is the pc: `back_edge_structured`'s resume arm reads
/// `args.green_int.first()`, so a declaration that put anything else first
/// would resume at the wrong instruction.
pub fn grain_driver_descriptor() -> JitDriverStaticData {
    let driver = driver();
    assert_eq!(
        driver.greens.first().map(String::as_str),
        Some("pc"),
        "the first green is the one a back edge resumes at",
    );
    let pair = |names: &'static [String], kinds: &'static [Type]| -> Vec<(&'static str, Type)> {
        assert_eq!(
            names.len(),
            kinds.len(),
            "every merge-point operand is named and kinded, or neither",
        );
        names
            .iter()
            .map(String::as_str)
            .zip(kinds.iter().copied())
            .collect()
    };
    JitDriverStaticData::new(
        pair(&driver.greens, &driver.green_args_spec),
        pair(&driver.reds, &driver.red_args_types),
    )
}

/// The shape of the code being traced, captured when tracing starts.
///
/// Carries the red image itself because [`JitState::extract_live`] is handed
/// only the meta -- the live values have to survive from `build_meta` to there.
#[derive(Clone)]
pub struct GrainMeta {
    /// The merge point's own offset in the portal body.
    pub header_pc: usize,
    /// One raw word per red, in declaration order.
    pub reds: Vec<i64>,
}

/// One symbolic value per red, in the order `create_sym` numbers them.
pub struct GrainSym {
    /// The merge point's offset in the portal body, which the tracer closes
    /// the loop back to.
    pub header_pc: usize,
    /// One input arg per red, minted in declaration order and typed from the
    /// build's own record of the operand kinds.
    pub reds: Vec<OpRef>,
}

impl JitCodeSym for GrainSym {
    fn total_slots(&self) -> usize {
        // The reds are merge-point operands, not slots of a state field bank.
        0
    }

    fn loop_header_pc(&self) -> usize {
        self.header_pc
    }

    fn fail_args(&self) -> Option<Vec<OpRef>> {
        Some(self.reds.clone())
    }

    fn fail_args_types(&self) -> Option<Vec<Type>> {
        Some(red_kinds().to_vec())
    }
}

/// This VM, as a tracing driver reads it.
#[derive(Default)]
pub struct GrainJitState {
    /// What the last [`JitState::restore`] was handed.
    ///
    /// Kept rather than dropped: compiled entry is disabled in this slice, so
    /// no deoptimization path calls `restore` yet. The later binding/entry work
    /// must define how these raw words are written back into the live `Vm` and
    /// `Scope`; silently discarding them here would conceal that missing step.
    pub last_restored: Vec<i64>,
}

impl JitState for GrainJitState {
    type Meta = GrainMeta;
    type Sym = GrainSym;
    /// The red image, straight: one raw word per red in declaration order.
    type Env = [i64];

    fn build_meta(&self, header_pc: usize, env: &Self::Env) -> Self::Meta {
        assert_eq!(
            env.len(),
            red_kinds().len(),
            "the merge point passes one value per declared red",
        );
        GrainMeta {
            header_pc,
            reds: env.to_vec(),
        }
    }

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64> {
        meta.reds.clone()
    }

    fn live_value_types(&self, _meta: &Self::Meta) -> Vec<Type> {
        red_kinds().to_vec()
    }

    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym {
        let _ = meta;
        GrainSym {
            header_pc,
            reds: red_kinds()
                .iter()
                .enumerate()
                .map(|(k, tp)| OpRef::input_arg_typed(k as u32, *tp))
                .collect(),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // This is the last state-owned gate before each compiled-entry path
        // calls the backend. Every symbolic residual target in the embedded
        // table is still unbound, so refuse here until the host supplies real
        // ABI shims. Returning false does not disable recording or compilation;
        // it invalidates/declines the artifact before execute_assembler.
        let _ = meta;
        super::jit::record_compiled_entry_refusal();
        false
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        let _ = meta;
        self.last_restored = values.to_vec();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.reds.clone()
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = meta;
        sym.reds.len() == red_kinds().len()
    }
}
