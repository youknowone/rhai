//! Process-wide collector for the grain tracing JIT.
//!
//! `Runtime::new` builds one `MiniMarkGC`. Each consulting thread registers
//! its shadow stack and holds the GIL until the thread exits, the pairing
//! `ensure_runtime_thread` uses.

#[cfg(feature = "grain-jit")]
use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(feature = "grain-jit")]
use std::sync::Once;

#[cfg(feature = "grain-jit")]
static GRAIN_FRAME_GC_TYPE_ID: AtomicU32 = AtomicU32::new(0);

#[cfg(feature = "grain-jit")]
static BUILT: Once = Once::new();

/// Id assigned by `TypeRegistry::register`. Zero until that registration runs.
#[cfg(feature = "grain-jit")]
#[allow(dead_code)]
pub(super) fn grain_frame_gc_type_id() -> u32 {
    GRAIN_FRAME_GC_TYPE_ID.load(Ordering::Acquire)
}

#[cfg(feature = "grain-jit")]
fn set_grain_frame_gc_type_id(id: u32) {
    GRAIN_FRAME_GC_TYPE_ID.store(id, Ordering::Release);
}

/// Visit GC edges of a `GrainFrame`. Host-owned fields are not edges.
#[cfg(feature = "grain-jit")]
unsafe fn grain_frame_custom_trace(_obj_addr: usize, _visit: &mut dyn FnMut(*mut majit_ir::GcRef)) {
}

#[cfg(feature = "grain-jit")]
fn build_gc_global() {
    BUILT.call_once(|| {
        if majit_gc::gc_sync::is_initialized() {
            return;
        }
        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let _jitframe_id = majit_metainterp::register_active_backend_jitframe_gc_type(&mut gc);
        let frame_id = gc.register_type(majit_gc::TypeInfo::with_custom_trace(
            std::mem::size_of::<super::GrainFrame<'static, 'static>>(),
            grain_frame_custom_trace,
        ));
        // The jitframe type is registered first, so this id is never the
        // unpublished sentinel.
        debug_assert_ne!(frame_id, 0);
        set_grain_frame_gc_type_id(frame_id);
        majit_ir::eval_breaker_word::publish_addr();
        majit_gc::gc_sync::store_singleton(Box::new(gc));
    });
}

#[cfg(feature = "grain-jit")]
thread_local! {
    static THREAD_ENTERED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static THREAD_GUARD: ThreadGuard = const { ThreadGuard };
}

#[cfg(feature = "grain-jit")]
struct ThreadGuard;

#[cfg(feature = "grain-jit")]
impl Drop for ThreadGuard {
    fn drop(&mut self) {
        majit_gc::shadow_stack::unregister_mutator();
        majit_gc::gc_sync::unregister_thread();
    }
}

#[cfg(feature = "grain-jit")]
fn ensure_thread() {
    if THREAD_ENTERED.with(|entered| entered.get()) {
        return;
    }
    THREAD_ENTERED.with(|entered| entered.set(true));
    majit_gc::gc_sync::register_thread();
    majit_gc::shadow_stack::register_mutator();
    // Touched after `register_mutator`, so this destructor runs before the
    // root-slot TLS those calls initialized.
    THREAD_GUARD.with(|_| {});
    majit_metainterp::install_active_backend_gc_standalone();
}

#[cfg(feature = "grain-jit")]
pub(super) fn ensure_collector() {
    build_gc_global();
    ensure_thread();
}
