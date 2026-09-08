//! The instruction set, the pools it indexes, and the checks a chunk must pass
//! before it runs.

pub mod code;

mod chain;
mod chunk;
mod op;
mod positions;
pub mod sites;
mod strings;
mod switch;
mod verify;

pub use chain::{Chain, Root, Step, StepFlags, Tail};
pub use chunk::Chunk;
pub use code::{AssembleError, Code, assemble, disassemble, resolve_switch_targets};
pub use op::{AssignOp, BinOpKind, BinOperand, Branch, Op, Receiver, UnOpKind};
pub(crate) use positions::site_to_position;
pub use positions::{Positions, TableError};
pub use strings::{BadTable, Strings};
pub use switch::{Switch, SwitchCase, SwitchRange, probe};
pub use verify::{Pools, VerifyError, verify};
