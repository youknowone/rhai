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

#[cfg(feature = "grain-jit")]
impl majit_gc::GcType for super::GrainFrame<'_, '_> {
    fn type_id() -> u32 {
        grain_frame_gc_type_id()
    }

    const SIZE: usize = std::mem::size_of::<Self>();
}

#[cfg(feature = "grain-jit")]
#[repr(C)]
struct GrainFramePrefix<'a, 'scope> {
    gc_header: majit_gc::header::GcHeader,
    frame: std::mem::ManuallyDrop<super::GrainFrame<'a, 'scope>>,
}

#[cfg(feature = "grain-jit")]
const _: () = assert!(
    std::mem::offset_of!(GrainFramePrefix<'static, 'static>, frame)
        == majit_gc::header::GcHeader::SIZE
);

/// Owning handle for one [`super::GrainFrame`].
///
/// With `grain-jit` the frame is a non-moving collector block: a header, then
/// the frame. The address stays put for as long as `run_frame` holds
/// `&mut GrainFrame`. Allocation failure uses the same layout in a `Box`.
pub(super) struct GrainFrameBox<'a, 'scope> {
    #[cfg(not(feature = "grain-jit"))]
    frame: super::GrainFrame<'a, 'scope>,
    #[cfg(feature = "grain-jit")]
    ptr: *mut super::GrainFrame<'a, 'scope>,
    #[cfg(feature = "grain-jit")]
    owner_root: Option<majit_gc::shadow_stack::OwnerRootGuard>,
    #[cfg(feature = "grain-jit")]
    fallback: *mut GrainFramePrefix<'a, 'scope>,
}

const _: fn(super::GrainFrame<'static, 'static>) -> GrainFrameBox<'static, 'static> =
    GrainFrameBox::new;

impl<'a, 'scope> GrainFrameBox<'a, 'scope> {
    pub(super) fn new(frame: super::GrainFrame<'a, 'scope>) -> Self {
        #[cfg(not(feature = "grain-jit"))]
        {
            Self { frame }
        }
        #[cfg(feature = "grain-jit")]
        {
            ensure_collector();
            let type_id = grain_frame_gc_type_id();
            if type_id != 0 {
                let allocated = majit_gc::alloc_young_nonmoving_typed(
                    type_id,
                    std::mem::size_of::<super::GrainFrame<'a, 'scope>>(),
                );
                if allocated.0 != 0 {
                    let ptr = allocated.0 as *mut super::GrainFrame<'a, 'scope>;
                    unsafe {
                        std::ptr::write(ptr, frame);
                    }
                    let owner_root = majit_gc::shadow_stack::OwnerRootGuard::new(allocated);
                    majit_gc::gc_write_barrier_managed(owner_root.get());
                    return Self {
                        ptr,
                        owner_root: Some(owner_root),
                        fallback: std::ptr::null_mut(),
                    };
                }
            }
            Self::new_boxed(frame)
        }
    }

    #[cfg(feature = "grain-jit")]
    fn new_boxed(frame: super::GrainFrame<'a, 'scope>) -> Self {
        let raw = Box::into_raw(Box::new(GrainFramePrefix {
            gc_header: majit_gc::header::GcHeader { tid_and_flags: 0 },
            frame: std::mem::ManuallyDrop::new(frame),
        }));
        let ptr =
            unsafe { std::ptr::addr_of_mut!((*raw).frame).cast::<super::GrainFrame<'a, 'scope>>() };
        Self {
            ptr,
            owner_root: None,
            fallback: raw,
        }
    }

    #[cfg(feature = "grain-jit")]
    fn frame_ptr(&self) -> *mut super::GrainFrame<'a, 'scope> {
        match &self.owner_root {
            Some(owner_root) => owner_root.get().0 as *mut super::GrainFrame<'a, 'scope>,
            None => self.ptr,
        }
    }
}

impl<'a, 'scope> std::ops::Deref for GrainFrameBox<'a, 'scope> {
    type Target = super::GrainFrame<'a, 'scope>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        #[cfg(not(feature = "grain-jit"))]
        {
            &self.frame
        }
        #[cfg(feature = "grain-jit")]
        unsafe {
            &*self.frame_ptr()
        }
    }
}

impl<'a, 'scope> std::ops::DerefMut for GrainFrameBox<'a, 'scope> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        #[cfg(not(feature = "grain-jit"))]
        {
            &mut self.frame
        }
        #[cfg(feature = "grain-jit")]
        unsafe {
            &mut *self.frame_ptr()
        }
    }
}

#[cfg(feature = "grain-jit")]
impl<'a, 'scope> Drop for GrainFrameBox<'a, 'scope> {
    fn drop(&mut self) {
        unsafe {
            std::ptr::drop_in_place(self.frame_ptr());
        }
        self.owner_root.take();
        if !self.fallback.is_null() {
            unsafe {
                drop(Box::from_raw(self.fallback));
            }
        }
    }
}
