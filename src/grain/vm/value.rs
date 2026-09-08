//! Copying and releasing a [`Dynamic`] without leaving the dispatch loop.
//!
//! Five of `Union`'s variants own nothing: `Unit`, `Bool`, `Char`, `Int` and
//! `Float` are a discriminant, a `Tag` and an `AccessMode` beside a `Copy`
//! payload, all inside the same sixteen bytes every other variant fits a box
//! into. Copying one is a sixteen-byte move and releasing one is nothing at
//! all — but `Dynamic::clone` (`types/dynamic.rs:824-861`) and the drop glue
//! the compiler generates for `Union` are both written over all thirteen, so
//! either answer is reached by an out-of-line call across a thirteen-way
//! switch, and a sixteen-byte copy pays that call's whole prologue to get to
//! its own two stores.
//!
//! `Dynamic::clone` is reached from hundreds of places, the tree-walker among
//! them, and is the right shape for that. These helpers are the interpreter's
//! own: they test the discriminant where they stand, answer the five
//! owning-nothing variants inline, and hand every other variant back to Rhai
//! unchanged. So the coverage of this module is a performance question and
//! never a correctness one — the fast arms reproduce `Dynamic::clone`'s own
//! arms, including its rule that a copy is read-write however the original was
//! marked.
//!
//! Keeping the inlined half to a test and a move is deliberate. The dispatch
//! function is large enough that code added to it is paid for in instruction
//! fetch by every instruction that never reaches the addition — which is the
//! same reason [`Vm::call_compiled_with_this`](super::Vm) is a call and not a
//! body.

use core::mem;

use crate::Dynamic;
use crate::types::dynamic::{AccessMode, Union};

/// A copy of a value whose variant owns neither an allocation nor a reference
/// count, or `None` for one that does.
///
/// The arms are `Dynamic::clone`'s (`types/dynamic.rs:832-838`) restricted to
/// the variants whose payload is `Copy`, `AccessMode` included: a scope entry
/// bound by `const` is read-only and the value read out of it is not, so a
/// copy that carried the original's mode would make an operand refuse a write
/// the walker allows.
#[inline(always)]
fn plain_copy(value: &Dynamic) -> Option<Dynamic> {
    let copied = match value.0 {
        Union::Unit(v, tag, ..) => Union::Unit(v, tag, AccessMode::ReadWrite),
        Union::Bool(v, tag, ..) => Union::Bool(v, tag, AccessMode::ReadWrite),
        Union::Char(v, tag, ..) => Union::Char(v, tag, AccessMode::ReadWrite),
        Union::Int(v, tag, ..) => Union::Int(v, tag, AccessMode::ReadWrite),
        #[cfg(not(feature = "no_float"))]
        Union::Float(v, tag, ..) => Union::Float(v, tag, AccessMode::ReadWrite),
        _ => return None,
    };
    Some(Dynamic(copied))
}

/// Whether releasing `value` has nothing to run.
///
/// The same five variants [`plain_copy`] answers for, and the same switch over
/// the discriminant: a variant with no destructor may be forgotten instead of
/// dropped, and forgetting it is what removes the call to `Union`'s glue.
#[must_use]
#[inline(always)]
fn is_plain(value: &Dynamic) -> bool {
    match value.0 {
        Union::Unit(..) | Union::Bool(..) | Union::Char(..) | Union::Int(..) => true,
        #[cfg(not(feature = "no_float"))]
        Union::Float(..) => true,
        _ => false,
    }
}

/// [`Dynamic::clone`], with the owning-nothing variants answered here.
#[inline(always)]
pub(super) fn clone_value(value: &Dynamic) -> Dynamic {
    match plain_copy(value) {
        Some(copy) => copy,
        None => value.clone(),
    }
}

/// [`Dynamic::flatten_clone`], with the owning-nothing variants answered here.
///
/// They need no flattening to be answered: `Union::Shared` is not among them,
/// and `flatten_clone` is `clone` for everything that is not a shared cell
/// (`types/dynamic.rs:1713-1723`).
#[inline(always)]
pub(super) fn flatten_clone_value(value: &Dynamic) -> Dynamic {
    match plain_copy(value) {
        Some(copy) => copy,
        None => value.flatten_clone(),
    }
}

/// Release `value`.
///
/// A variant that owns nothing is forgotten rather than dropped, which is the
/// same thing done without the call: `Union`'s glue is one function over all
/// thirteen variants, so a value with nothing to release pays the call and its
/// prologue only to reach a branch that does nothing.
#[inline(always)]
pub(super) fn release(value: Dynamic) {
    if is_plain(&value) {
        mem::forget(value);
    } else {
        drop(value);
    }
}

/// Put `value` in `slot`, releasing what the slot held.
///
/// What `*slot = value` is, with [`release`] standing where the compiler would
/// have emitted the drop. The order differs — the new value lands before the
/// old one is released rather than after — and nothing can observe that: a
/// `Dynamic`'s destructor has no way to reach the slot it came out of.
#[inline(always)]
pub(super) fn overwrite(slot: &mut Dynamic, value: Dynamic) {
    release(mem::replace(slot, value));
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::Ordering;

    use crate::{Dynamic, ImmutableString};

    #[cfg(feature = "no_std")]
    use std::prelude::v1::*;

    /// How many `Custom` values are alive. A leak is otherwise silent, so this
    /// is what says [`release`] ran a destructor it was not allowed to skip.
    static CUSTOMS: core::sync::atomic::AtomicIsize = core::sync::atomic::AtomicIsize::new(0);

    /// The count above is one cell for the whole process and the harness runs
    /// tests in parallel, so every test that makes a `Custom` holds this for as
    /// long as it has one alive. Without it the test that reads the count sees
    /// another test's values come and go.
    static COUNTING: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Hold [`COUNTING`], whether or not an earlier failure left it poisoned:
    /// what it guards is a count, and a panic elsewhere does not make this
    /// test's own arithmetic unsound.
    fn counting() -> std::sync::MutexGuard<'static, ()> {
        COUNTING
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A type with no `Union` variant of its own — the `Union::Variant` arm,
    /// which clones through `Variant::clone_object` and owns two boxes.
    #[derive(Debug)]
    struct Custom(i64);

    impl Clone for Custom {
        fn clone(&self) -> Self {
            CUSTOMS.fetch_add(1, Ordering::Relaxed);
            Self(self.0)
        }
    }

    impl Drop for Custom {
        fn drop(&mut self) {
            CUSTOMS.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn custom(value: i64) -> Dynamic {
        CUSTOMS.fetch_add(1, Ordering::Relaxed);
        Dynamic::from(Custom(value))
    }

    /// One value of every variant a `Dynamic` can hold, so that a fast arm
    /// added or lost is a test failure rather than a wrong answer.
    ///
    /// Handed back read-only: `Dynamic::clone` marks its copy read-write
    /// whatever it was given, and a fast arm that carried the original's mode
    /// instead would be invisible against a read-write original.
    fn every_variant() -> Vec<Dynamic> {
        let mut values = vec![
            Dynamic::UNIT,
            Dynamic::from(true),
            Dynamic::from('x'),
            Dynamic::from(42 as crate::INT),
            Dynamic::from(ImmutableString::from("a string")),
            Dynamic::from(crate::FnPtr::new("f").expect("a legal function name")),
            custom(7),
        ];

        #[cfg(not(feature = "no_float"))]
        values.push(Dynamic::from(1.5 as crate::FLOAT));
        #[cfg(feature = "decimal")]
        values.push(Dynamic::from(rust_decimal::Decimal::new(15, 1)));
        #[cfg(not(feature = "no_index"))]
        {
            values.push(Dynamic::from_array(vec![Dynamic::from(1 as crate::INT)]));
            values.push(Dynamic::from_blob(vec![1_u8, 2, 3]));
        }
        #[cfg(not(feature = "no_object"))]
        {
            let mut map = crate::Map::new();
            map.insert("k".into(), Dynamic::from(1 as crate::INT));
            values.push(Dynamic::from_map(map));
        }
        #[cfg(not(feature = "no_time"))]
        values.push(Dynamic::from_timestamp(std::time::Instant::now()));
        #[cfg(not(feature = "no_closure"))]
        values.push(Dynamic::from(1 as crate::INT).into_shared());

        values.into_iter().map(Dynamic::into_read_only).collect()
    }

    /// How a copy reads, in everything the helpers are not allowed to change:
    /// the type it holds, what it prints as, and the access mode it carries.
    fn reading(value: &Dynamic) -> (std::any::TypeId, String, bool) {
        (value.type_id(), format!("{value:?}"), value.is_read_only())
    }

    #[test]
    fn a_fast_copy_reads_as_the_clone_it_replaces() {
        let _counting = counting();
        for value in every_variant() {
            assert_eq!(
                reading(&clone_value(&value)),
                reading(&value.clone()),
                "clone_value disagrees with Dynamic::clone on {value:?}",
            );
            assert_eq!(
                reading(&flatten_clone_value(&value)),
                reading(&value.flatten_clone()),
                "flatten_clone_value disagrees on {value:?}",
            );
        }
    }

    /// A shared cell is the one variant the two helpers answer differently, so
    /// it has to reach Rhai under either leg rather than be copied here.
    #[test]
    #[cfg(not(feature = "no_closure"))]
    fn a_shared_cell_is_flattened_and_never_copied_inline() {
        let cell = Dynamic::from(7 as crate::INT).into_shared();
        assert!(
            clone_value(&cell).is_shared(),
            "cloning a cell keeps the cell",
        );
        let flattened = flatten_clone_value(&cell);
        assert!(
            !flattened.is_shared(),
            "flattening a cell must reach its value",
        );
        assert_eq!(flattened.as_int().expect("the cell holds an integer"), 7);
    }

    /// Releasing runs the destructor of everything that has one, and a value
    /// forgotten instead would be a leak rather than a failure — so the live
    /// count of a type that counts itself is what says so.
    #[test]
    fn releasing_a_value_that_owns_something_still_drops_it() {
        let _counting = counting();
        let before = CUSTOMS.load(Ordering::Relaxed);
        for value in every_variant() {
            release(value);
        }
        for _ in 0..4 {
            release(clone_value(&custom(3)));
        }
        let mut slot = custom(4);
        overwrite(&mut slot, Dynamic::UNIT);
        release(slot);
        assert_eq!(
            CUSTOMS.load(Ordering::Relaxed),
            before,
            "every released copy must have run its destructor",
        );
    }

    /// Overwriting leaves the new value and releases the old one, whichever
    /// variant the slot held.
    #[test]
    fn overwriting_a_slot_leaves_the_new_value() {
        let _counting = counting();
        for value in every_variant() {
            let mut slot = value;
            overwrite(&mut slot, Dynamic::from(9 as crate::INT));
            assert_eq!(
                slot.as_int().expect("the slot now holds an integer"),
                9,
                "overwrite left the wrong value",
            );
            overwrite(&mut slot, Dynamic::UNIT);
            assert!(slot.is_unit(), "overwrite left the wrong value",);
        }
    }
}
