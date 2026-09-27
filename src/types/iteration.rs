//! A `for` iterator, the built-in the operand stack holds.
//!
//! `GET_ITER` / `FOR_ITER` keep the iterator on the frame value stack.
//! The same object is `Union::Iter`: a class the discriminant check guards.

use crate::{Dynamic, EvalAltResult, INT};

/// One `for` loop's iterator.
///
/// The count lives here, beside the cursor, because a loop long enough to
/// wrap it is an error rather than a wrap.
pub struct Iteration {
    pub(crate) items: Items,
    /// The index of the item last handed out, starting one below the first.
    pub(crate) count: INT,
}

/// Cursor of an exclusive integer range.
pub struct IntCursor {
    /// The next value to hand out; at or past `end` when exhausted.
    pub(crate) next: INT,
    pub(crate) end: INT,
}

/// Cursor of `range(from, to, step)`.
pub struct IntStepCursor {
    pub(crate) next: INT,
    pub(crate) end: INT,
    pub(crate) step: INT,
}

/// What a running `for` loop pulls its items from.
pub enum Items {
    /// An exclusive integer range, walked in place.
    IntRange(IntCursor),
    /// `range(from, to, step)` walked in place the same way.
    IntStepRange(IntStepCursor),
    /// Whatever the registry built.
    ///
    /// `FnIterator` returns this box with no `Send` bound on the iterator
    /// (`sync` puts `Send + Sync` on the function, not on the box).
    Boxed(Box<dyn Iterator<Item = Result<Dynamic, Box<EvalAltResult>>>>),
}
