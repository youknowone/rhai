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
//! At this tier the call does nothing at all. It exists so that a lowering
//! pass reading the compiled MIR finds a marker where the loop begins — which
//! is also why it is `inline(never)`: an empty function that is allowed to
//! inline is deleted before that pass ever sees it.

use super::Vm;
use crate::grain::Program;
use crate::Scope;

/// The driver `run_frame`'s loop reports to.
///
/// Zero-sized and stateless: it names the loop rather than holding anything.
/// A consumer recognises the merge point by this type's name, so renaming it
/// silently disconnects the loop from the JIT rather than failing to build.
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
        let _ = (pc, program, vm, scope, base, reached);
    }
}
