//! The operators the dispatch loop runs without resolving a function.
//!
//! Every function here answers what [`get_builtin_binary_op_fn`] and
//! [`get_builtin_op_assignment_fn`] answer for the same operator and the same
//! operand pair — that is the whole contract, and it is why these bodies
//! *call* the same `packages::arithmetic` functions rather than restating what
//! they do. A `checked_add` written out again here would be a second place for
//! the overflow rule to live, and the two would drift.
//!
//! What is not restated is the *resolution*: the token match, the walk over
//! `(&x.0, &y.0, op)`, building `Some((fn_ptr, bool))`, the indirect call over
//! `&mut [&mut Dynamic]`, and the `as_int().unwrap()` downcasts on the far side
//! of it. Those are pure overhead for a site whose operands are two integers,
//! and removing them is the point.
//!
//! Anything not covered here answers `None`, and the caller dispatches exactly
//! as it did before. So the coverage of this module is a performance question
//! and never a correctness one.
//!
//! [`get_builtin_binary_op_fn`]: crate::func::get_builtin_binary_op_fn
//! [`get_builtin_op_assignment_fn`]: crate::func::get_builtin_op_assignment_fn

use crate::grain::bytecode::BinOpKind;
use crate::types::dynamic::Union;
use crate::{Dynamic, RhaiResultOf, INT};
#[cfg(feature = "no_std")]
use std::prelude::v1::*;

#[cfg(not(feature = "no_float"))]
use crate::FLOAT;

/// `x op y` for two integers, where the operator produces an integer.
///
/// Mirrors the arithmetic half of the `(Union::Int(..), Union::Int(..), _)` arm
/// of [`get_builtin_binary_op_fn`], including the split between the checked
/// build and `unchecked`: the checked arm goes through
/// `arith_basic::INT::functions`, and the unchecked one is the raw expression
/// that arm spells out — which is not the same thing as the checked function
/// with its check removed. `**` is the visible case: unchecked truncates the
/// exponent with `as u32` where `power` would refuse it.
///
/// [`get_builtin_binary_op_fn`]: crate::func::get_builtin_binary_op_fn
#[inline]
fn int_arithmetic(kind: BinOpKind, x: INT, y: INT) -> Option<RhaiResultOf<INT>> {
    use BinOpKind::{
        Add, And, Divide, Modulo, Multiply, Or, Power, ShiftLeft, ShiftRight, Subtract, Xor,
    };

    #[cfg(not(feature = "unchecked"))]
    #[allow(clippy::wildcard_imports)]
    use crate::packages::arithmetic::arith_basic::INT::functions::*;

    #[cfg(not(feature = "unchecked"))]
    return Some(match kind {
        Add => add(x, y),
        Subtract => subtract(x, y),
        Multiply => multiply(x, y),
        Divide => divide(x, y),
        Modulo => modulo(x, y),
        Power => power(x, y),
        ShiftRight => Ok(shift_right(x, y)),
        ShiftLeft => Ok(shift_left(x, y)),
        And => Ok(binary_and(x, y)),
        Or => Ok(binary_or(x, y)),
        Xor => Ok(binary_xor(x, y)),
        _ => return None,
    });

    #[cfg(feature = "unchecked")]
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    return Some(Ok(match kind {
        Add => x + y,
        Subtract => x - y,
        Multiply => x * y,
        Divide => x / y,
        Modulo => x % y,
        Power => x.pow(y as u32),
        ShiftRight => {
            if y < 0 {
                x << -y
            } else {
                x >> y
            }
        }
        ShiftLeft => {
            if y < 0 {
                x >> -y
            } else {
                x << y
            }
        }
        And => x & y,
        Or => x | y,
        Xor => x ^ y,
        _ => return None,
    }));
}

/// The comparison half of the same arm, which both builds share.
#[inline]
fn int_comparison(kind: BinOpKind, x: INT, y: INT) -> Option<bool> {
    Some(match kind {
        BinOpKind::Equals => x == y,
        BinOpKind::NotEquals => x != y,
        BinOpKind::Greater => x > y,
        BinOpKind::GreaterEquals => x >= y,
        BinOpKind::Less => x < y,
        BinOpKind::LessEquals => x <= y,
        // `..` and `..=` are the arm's remaining entries and build a range
        // rather than a value; they have no `BinOpKind` and never arrive here.
        _ => return None,
    })
}

/// `x op y` for two integers, or `None` if the arm has no entry for the
/// operator — in which case the caller dispatches.
#[inline]
pub fn int_binary(kind: BinOpKind, x: INT, y: INT) -> Option<RhaiResultOf<Dynamic>> {
    if let Some(value) = int_arithmetic(kind, x, y) {
        return Some(value.map(Into::into));
    }
    match int_comparison(kind, x, y) {
        Some(held) => Some(Ok(held.into())),
        None => None,
    }
}

/// `x op= y` for two integers, returning the value to write back.
///
/// Mirrors the `(Union::Int(..), Union::Int(..), _)` arm of
/// [`get_builtin_op_assignment_fn`].  The caller owns the `Dynamic` container
/// and writes this scalar back while preserving its tag and access mode.  This
/// value-returning boundary also matches the generated JIT's scalar ABI:
/// RPython has no pointer-to-Signed representation for a Rust `&mut i64`.
///
/// The two arms share `int_arithmetic` because Rhai's two tables do: `+=` and
/// `+` resolve to the same `add`, and an op-assignment that overflows reports
/// what the operator would have.
///
/// `None` means the operator has no op-assignment built-in, which is the
/// caller's cue to dispatch and then to expand into `x = x op y` — the same
/// two steps it already takes. The comparison kinds land here, and correctly:
/// there is no `<=`, and [`BinOpKind::of_assign`] never produces one anyway.
///
/// [`get_builtin_op_assignment_fn`]: crate::func::get_builtin_op_assignment_fn
#[inline]
pub fn int_assign(kind: BinOpKind, x: INT, y: INT) -> Option<RhaiResultOf<INT>> {
    int_arithmetic(kind, x, y)
}

/// `x op y` for a pair the float rules cover, or `None` for an operator this
/// does not run.
///
/// Only the arithmetic half of `impl_float!`, which is the half that is one
/// expression and identical under `unchecked`. The comparisons are not here:
/// a checked build compares floats through a relative-epsilon rule that would
/// have to be restated to be inlined, and restating it is exactly what this
/// module refuses to do. They dispatch, and the operator cache is what makes
/// that cheap.
///
/// The operands arrive already widened, which is what makes the three pairs
/// `impl_float!` is invoked for — float/float, float/int, int/float — one
/// function: every one of them converts to [`FLOAT`] before operating.
#[cfg(not(feature = "no_float"))]
#[inline]
pub fn float_binary(kind: BinOpKind, x: FLOAT, y: FLOAT) -> Option<Dynamic> {
    float_arithmetic(kind, x, y).map(Into::into)
}

/// The operator itself, shared by the value form and the in-place one.
#[cfg(not(feature = "no_float"))]
#[inline]
fn float_arithmetic(kind: BinOpKind, x: FLOAT, y: FLOAT) -> Option<FLOAT> {
    Some(match kind {
        BinOpKind::Add => x + y,
        BinOpKind::Subtract => x - y,
        BinOpKind::Multiply => x * y,
        BinOpKind::Divide => x / y,
        BinOpKind::Modulo => x % y,
        BinOpKind::Power => x.powf(y),
        _ => return None,
    })
}

/// `x op= y` where the target is a float, returning the value to write back.
///
/// `impl_float!` in the op-assignment table is `*write_lock::<FLOAT>() op= y`
/// for the five arithmetic forms and `x.powf(y)` for `**=`, in both builds —
/// so it is [`float_binary`] written back into the target, and the target's
/// tag and access mode survive for the same reason the integer case's do.
#[cfg(not(feature = "no_float"))]
#[inline]
pub fn float_assign(kind: BinOpKind, x: FLOAT, y: FLOAT) -> Option<FLOAT> {
    float_arithmetic(kind, x, y)
}

/// Widen an operand to [`FLOAT`] if the float rules apply to it at all.
///
/// `impl_float!` is reached for float/float, float/int and int/float, and for
/// no other pair — so an integer widens only when the *other* operand is a
/// float, and two integers are never float arithmetic. Callers check that by
/// requiring at least one operand to answer `true` here.
#[cfg(not(feature = "no_float"))]
#[inline]
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn as_float_operand(value: &Dynamic) -> Option<(FLOAT, bool)> {
    match value.0 {
        Union::Float(ref held, ..) => Some((**held, true)),
        Union::Int(held, ..) => Some((held as FLOAT, false)),
        _ => None,
    }
}
