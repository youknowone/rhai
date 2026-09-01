use crate::tokenizer::Token;

/// A binary operator the VM can execute without resolving a function.
///
/// One of these is what [`Op::BinOp`] carries, and what an [`AssignOp`] holds
/// alongside its tokens. It names the *operator*, never the operand types: the
/// types are a run-time property and the instruction is speculative, so a site
/// that turns out to hold a string or a custom type falls back to exactly the
/// dispatch [`Op::Call`] would have done.
///
/// Only the operators [`get_builtin_binary_op_fn`] answers for integers and
/// floats are here. `..`, `..=` and any operator a host registered with
/// `Engine::register_custom_operator` have no entry and keep the dispatching
/// instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BinOpKind {
    /// `+`, `+=`
    Add = 0,
    /// `-`, `-=`
    Subtract = 1,
    /// `*`, `*=`
    Multiply = 2,
    /// `/`, `/=`
    Divide = 3,
    /// `%`, `%=`
    Modulo = 4,
    /// `**`, `**=`
    Power = 5,
    /// `<<`, `<<=`
    ShiftLeft = 6,
    /// `>>`, `>>=`
    ShiftRight = 7,
    /// `&`, `&=`
    And = 8,
    /// `|`, `|=`
    Or = 9,
    /// `^`, `^=`
    Xor = 10,
    /// `==`
    Equals = 11,
    /// `!=`
    NotEquals = 12,
    /// `>`
    Greater = 13,
    /// `>=`
    GreaterEquals = 14,
    /// `<`
    Less = 15,
    /// `<=`
    LessEquals = 16,
}

/// A unary operator the VM can execute without resolving a function.
///
/// What [`Op::UnOp`] carries. The same speculation as [`BinOpKind`]: it names
/// the operator, never the operand type, and a site holding anything the VM
/// cannot run falls back to the dispatch [`Op::Call`] would have done.
///
/// Only `!` is here, and the set is not an accident of coverage — it is the
/// walker's. `eval_fn_call_expr` short-circuits exactly one unary operator
/// under `Engine::fast_operators`, `!` on a `Union::Bool`, and dispatches
/// every other one. Adding `-` here would answer a host-registered `-` on an
/// integer differently from the tree this VM replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UnOpKind {
    /// `!`
    Not = 0,
}

impl UnOpKind {
    /// The last value, so the encoding can reject a byte that names nothing.
    pub const LAST: u8 = Self::Not as u8;

    /// Which unary operator a token is, if the VM has an implementation.
    ///
    /// `-` and `+` answer `None` and keep dispatching: they are registered
    /// functions (`packages::arithmetic` `neg` and `plus`) that a host may
    /// replace, and the walker resolves them rather than short-circuiting.
    #[must_use]
    pub const fn of(token: &Token) -> Option<Self> {
        Some(match token {
            Token::Bang => Self::Not,
            _ => return None,
        })
    }

    /// Recover a kind from the byte an instruction carries.
    ///
    /// `None` for a byte that names nothing, which is what stops a corrupt
    /// artifact from reaching an unchecked transmute.
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        Some(match byte {
            0 => Self::Not,
            _ => return None,
        })
    }
}

impl BinOpKind {
    /// The last value, so the encoding can reject a byte that names nothing.
    pub const LAST: u8 = Self::LessEquals as u8;

    /// Which binary operator a token is, if the VM has an implementation.
    ///
    /// The set is exactly what [`get_builtin_binary_op_fn`] handles for
    /// `Union::Int` and `Union::Float` operands. Anything else — a range, a
    /// custom operator — answers `None` and keeps dispatching.
    #[must_use]
    pub const fn of(token: &Token) -> Option<Self> {
        Some(match token {
            Token::Plus => Self::Add,
            Token::Minus => Self::Subtract,
            Token::Multiply => Self::Multiply,
            Token::Divide => Self::Divide,
            Token::Modulo => Self::Modulo,
            Token::PowerOf => Self::Power,
            Token::LeftShift => Self::ShiftLeft,
            Token::RightShift => Self::ShiftRight,
            Token::Ampersand => Self::And,
            Token::Pipe => Self::Or,
            Token::XOr => Self::Xor,
            Token::EqualsTo => Self::Equals,
            Token::NotEqualsTo => Self::NotEquals,
            Token::GreaterThan => Self::Greater,
            Token::GreaterThanEqualsTo => Self::GreaterEquals,
            Token::LessThan => Self::Less,
            Token::LessThanEqualsTo => Self::LessEquals,
            _ => return None,
        })
    }

    /// The same for the `x op= y` form, keyed on the op-assignment token.
    ///
    /// The set is what [`get_builtin_op_assignment_fn`] handles, which has no
    /// comparison forms — there is no `<=`.
    #[must_use]
    pub const fn of_assign(token: &Token) -> Option<Self> {
        Some(match token {
            Token::PlusAssign => Self::Add,
            Token::MinusAssign => Self::Subtract,
            Token::MultiplyAssign => Self::Multiply,
            Token::DivideAssign => Self::Divide,
            Token::ModuloAssign => Self::Modulo,
            Token::PowerOfAssign => Self::Power,
            Token::LeftShiftAssign => Self::ShiftLeft,
            Token::RightShiftAssign => Self::ShiftRight,
            Token::AndAssign => Self::And,
            Token::OrAssign => Self::Or,
            Token::XOrAssign => Self::Xor,
            _ => return None,
        })
    }

    /// Recover a kind from the byte an instruction carries.
    ///
    /// `None` for a byte that names nothing, which is what stops a corrupt
    /// artifact from reaching an unchecked transmute.
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        Some(match byte {
            0 => Self::Add,
            1 => Self::Subtract,
            2 => Self::Multiply,
            3 => Self::Divide,
            4 => Self::Modulo,
            5 => Self::Power,
            6 => Self::ShiftLeft,
            7 => Self::ShiftRight,
            8 => Self::And,
            9 => Self::Or,
            10 => Self::Xor,
            11 => Self::Equals,
            12 => Self::NotEquals,
            13 => Self::Greater,
            14 => Self::GreaterEquals,
            15 => Self::Less,
            16 => Self::LessEquals,
            _ => return None,
        })
    }
}

/// What `x op= y` needs to reproduce Rhai's resolution order.
///
/// Both the op-assignment and the plain operator are carried, because Rhai
/// tries the first and falls back to expanding into the second when no
/// op-assignment implementation exists (`eval/stmt.rs:217-236`).
///
/// Lives in the program's op-assignment pool rather than in the instruction:
/// four fields including two `Token`s do not fit an operand, and the same
/// `+=` used in ten places is one entry.
// No `Eq`: `Token` carries float literals, so it is only `PartialEq`.
#[derive(Debug, Clone, PartialEq)]
pub struct AssignOp {
    /// The `+=` token, for the built-in lookup.
    pub op_assign: Token,
    /// `"+="`, for dispatch and for error messages.
    pub op_assign_name: u32,
    /// The `+` token, for the expansion.
    pub op: Token,
    /// `"+"`.
    pub op_name: u32,
    /// Which operator `op_assign` is, when the VM can execute it directly.
    ///
    /// Derived from `op_assign` rather than stored in an artifact, and
    /// recomputed wherever an `AssignOp` is built. Keeping it here is what
    /// lets `x += y` reach an integer add without matching a `Token` first —
    /// the op-assignment instructions address this pool and would otherwise
    /// have to decode the token on every iteration of every loop.
    pub kind: Option<BinOpKind>,
}

impl AssignOp {
    /// Build one, deriving [`AssignOp::kind`] from the op-assignment token.
    #[must_use]
    pub fn new(op_assign: Token, op_assign_name: u32, op: Token, op_name: u32) -> Self {
        let kind = BinOpKind::of_assign(&op_assign);
        Self {
            op_assign,
            op_assign_name,
            op,
            op_name,
            kind,
        }
    }
}

/// Where a fused binary operator reads an operand.
///
/// The two loads in front of an operator are the commonest instruction pair
/// there is — `i < n`, `a + b`, `n - 1` — and both of them push a value the
/// operator takes straight off again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOperand {
    /// Local slot `.0`, read as [`Op::LoadLocal`] reads one: cloned out,
    /// flattening any shared cell.
    Local(u16),

    /// Constant `.0` from the pool, read as [`Op::Const`] reads one.
    Const(u32),
}

/// Where [`Op::CallRef`] finds the variable it calls through.
///
/// The two differ in how the variable is reached, not in what happens to it:
/// both take a reference where Rhai would and fall back to a value where it
/// would not, by the same rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Receiver {
    /// A local, addressed by slot. Nothing was pushed for it — the call reads
    /// the scope entry itself, and a slot always names one.
    Local(u16),

    /// A variable no slot addresses: the caller's, a module's, or nothing.
    ///
    /// [`Op::LoadNamed`] has already resolved the name and left its value as
    /// argument zero, which is what raises `ErrorVariableNotFound` against the
    /// variable rather than against the call — two positions the table cannot
    /// give one instruction. The call re-reaches the scope entry for the
    /// reference and falls back to that value when there is no entry to reach:
    /// a resolver's answer, a module's constant, a `const`.
    ///
    /// So the by-reference path pays for a clone it discards. Worth removing
    /// only if a profile of a host-heavy script says so; a local, which is the
    /// common receiver by far, never makes one.
    Named(u32),

    /// The frame's receiver, for `f(this, ..)`.
    ///
    /// Rhai applies the same rewrite to `this` as to a variable, but only when
    /// the receiver is neither shared nor curried (`func/call.rs:1409-1433`).
    /// Shared-ness is a run-time property, so the value arrives on the stack as
    /// argument zero and the call reaches for the register instead when it turns
    /// out to be usable by reference — the deferral [`Receiver::Local`] already
    /// makes for a read-only entry.
    ///
    /// Unlike either of the others, [`Op::LoadThis`] pushes it *before* the
    /// remaining arguments. Rhai's two arms disagree about when `this` is read:
    /// the by-reference one takes it after them (`func/call.rs:1417`), but the
    /// fallback that a shared or unbound receiver lands in reads and flattens it
    /// first (`:1462`). Reading first is what makes `f(this, { this = 9; 1 })`
    /// pass the pre-mutation value, and an unbound `f(this, no_such)` report
    /// `ErrorUnboundThis` rather than `ErrorVariableNotFound`.
    This,
}

/// One VM instruction, as the compiler emits it and a disassembly shows it.
///
/// **Not the executed form.** A program's code is a byte slice, assembled from
/// these by [`assemble`](crate::grain::bytecode::assemble) and dispatched on
/// directly, so a loaded program can borrow its instructions from the artifact
/// rather than building sixteen bytes of enum per instruction. See
/// [`code`](crate::grain::bytecode::code) for the encoding.
///
/// A stack machine: operands are pushed and consumed on an operand stack, and
/// locals live in slots addressed directly. `EvalAst` is the escape hatch that
/// hands a fragment back to Rhai's tree walker, so anything the compiler cannot
/// yet lower still runs, and the whole language stays covered. Lowering more of
/// it converts residuals into instructions rather than adding coverage.
///
/// Instructions carry no source position. Several of them can fail against a
/// place in the source, and the position for that comes from the program's
/// [`Positions`](crate::grain::bytecode::Positions) table, keyed on the instruction's
/// own address. Keeping it out means the diagnostics can be stripped from an
/// artifact without touching the code.
///
/// Anything too wide for an operand is a `u32` index into one of the program's
/// pools, which is also what keeps a repeated operator or name from being
/// stored twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// Push constant `.0` from the pool.
    Const(u32),
    /// Push unit.
    Unit,
    /// Push a boolean.
    Bool(bool),

    /// Push the value in local slot `.0`.
    LoadLocal(u16),
    /// Pop and write into local slot `.0`, which must already exist.
    StoreLocal {
        /// The slot index
        slot: u16,
        /// Whether the value should be stored as a constant.
        is_const: bool,
    },

    /// Push the value of the variable named `.0`, found by name.
    ///
    /// For the variables no slot can address: the ones the caller already had
    /// in its `Scope` when the program started, which sit below the base every
    /// slot is measured from. Without this, a script that reads anything its
    /// host supplied is a fragment, and so cannot be written to an artifact at
    /// all.
    ///
    /// Three places are searched, in Rhai's order (`eval/expr.rs:107-155`):
    /// the resolver a host may have registered with `Engine::on_var`, then the
    /// scope, then the modules loaded into the global namespace. Missing from
    /// all three is `ErrorVariableNotFound`.
    ///
    /// A reverse scan of the scope per read, where a slot is an index — which
    /// is why the compiler only emits this for a name it could not resolve.
    LoadNamed(u32),

    /// Pop a value and assign it to the variable named `name`, optionally
    /// through an operator.
    ///
    /// [`Op::LoadNamed`]'s counterpart, and resolved the same way. Assigning
    /// to anything that is not a scope entry it can take a reference to — a
    /// value the resolver produced, a module's constant, a `const` — is
    /// `ErrorAssignmentToConstant`, as it is in the walker.
    AssignNamed {
        /// The name of the variable
        name: u32,
        /// Index into the op-assignment pool; absent for a plain `=`.
        op: Option<u32>,
    },

    /// Pop a value and assign it to local slot `slot`, optionally through an
    /// operator.
    ///
    /// Separate from `StoreLocal` because `x += y` is not `x = x + y`: Rhai
    /// looks for an op-assignment implementation that mutates in place, and
    /// only expands to the binary form if there is none.
    AssignLocal {
        /// The slot index
        slot: u16,
        /// Names the variable in `ErrorAssignmentToConstant`.
        var_name: u32,
        /// Index into the op-assignment pool; absent for a plain `=`.
        op: Option<u32>,
    },
    /// Assign `src`'s value to local slot `slot`, optionally through an
    /// operator.
    ///
    /// [`Op::AssignLocal`] with the [`Op::LoadLocal`] or [`Op::Const`] that fed
    /// it folded in. `x op= y` where `y` is a plain local read or a constant
    /// pushes a value and pops it again on the very next instruction, and both
    /// halves of that pair are among the most frequent instructions there are —
    /// so the pair is one instruction, and the operand stack is not touched at
    /// all. A loop counter's `i += 1` is the constant case.
    ///
    /// The source is read exactly as the instruction it swallowed read it:
    /// a local cloned out and flattened, a constant cloned from the pool.
    AssignLocalFrom {
        /// The slot being assigned to.
        slot: u16,
        /// Names the variable in `ErrorAssignmentToConstant`.
        var_name: u32,
        /// Index into the op-assignment pool; absent for a plain `=`.
        op: Option<u32>,
        /// Where the value is read from.
        src: BinOperand,
    },

    /// Pop and declare it as a new local, extending the scope by one.
    ///
    /// Slots are assigned in declaration order, so the new local always lands
    /// at the top of the scope. Carries the name because locals live in the
    /// caller's `Scope`, where entries are named, and carries const-ness
    /// because Rhai enforces it through the value's own access mode.
    DeclareLocal {
        /// The name of the variable
        name: u32,
        /// Whether the variable is declared `const`.
        is_const: bool,
    },

    /// Discard the top of the operand stack.
    Pop,

    /// Jump to `.0`.
    ///
    /// An instruction index as the compiler emits it, a byte offset once
    /// assembled — instructions vary in length, so there is nothing else it
    /// could be.
    Jump(u32),
    /// Pop a condition and jump to `.0` if it is true. Mirrors
    /// [`Op::JumpIfFalse`]; both exist so short-circuit `&&` and `||` lower
    /// without an extra negation.
    JumpIfTrue {
        /// Where to jump to
        target: u32,
    },
    /// Pop a condition and jump to `.0` if it is false.
    ///
    /// Its position-table entry is the condition's own position, because Rhai
    /// rejects a non-boolean guard against the guard expression rather than the
    /// statement — and the differential harness compares error positions.
    JumpIfFalse {
        /// Where to jump to
        target: u32,
    },
    /// Inspects a condition and jump to `.0` if it is not `()`.
    /// The condition is not popped, so the caller can read it afterwards.
    /// Mirrors [`Op::JumpIfFalse`]; exists so short-circuit `??` lower.
    SkipIfNotUnit {
        /// Where to skip to
        target: u32,
    },

    /// Pop `argc` arguments and call the function named by `name`, pushing the
    /// result.
    ///
    /// Dispatch goes through Rhai, so every registered function, operator and
    /// script function resolves exactly as it would in the walker. Only calls
    /// Rhai handles syntactically before dispatch — `Fn`, `call`, `curry`,
    /// `eval`, `is_def_var` — are excluded.
    ///
    /// The position table's entry for this instruction is the call site. Rhai's
    /// dispatch path takes one and reports failures against it; `call_fn_raw`
    /// does not, so an error that comes back without a position gets this one.
    ///
    /// `op` indexes the operator pool when the call is an operator, and names
    /// the token the built-in lookup keys on. The walker short-circuits these
    /// to a function pointer rather than dispatching, and a VM that did not
    /// would be slower than the tree it replaced.
    Call {
        /// The name of the function
        name: u32,
        /// How many arguments to pop
        argc: u8,
        /// Index into the operator pool; absent unless the call is an operator.
        op: Option<u32>,
        /// This call captures the parent's scope.
        capture_parent_scope: bool,
    },

    /// Call `name` with a variable as its first argument, taken by reference.
    ///
    /// Rhai rewrites `f(x, ..)` into `x.f(..)` whenever the first argument is a
    /// plain variable, so that a `&mut` first parameter mutates the variable
    /// rather than a copy (`func/call.rs:1434-1460`). `push(a, 2)` and
    /// `a.push(2)` are the same call; only the second reached the mutation
    /// through [`Op::Chain`].
    ///
    /// Two things follow from the rewrite, and together they are why this is an
    /// instruction rather than an argument order:
    ///
    /// * the variable is read *after* the other arguments, so an argument that
    ///   writes to it is seen;
    /// * a shared or read-only variable is passed by value instead — Rhai hands
    ///   out a reference to neither (`func/call.rs:1449-1454`).
    ///
    /// Operators never reach here: under `fast_operators` a binary one
    /// short-circuits before the rewrite (`func/call.rs:1775`), so `a + b`
    /// reads `a` first and needs no reference.
    CallRef {
        /// The name of the function
        name: u32,
        /// How many arguments to pop, not counting the receiver.
        argc: u8,
        /// Where the first argument is found.
        receiver: Receiver,
        /// This call captures the parent's scope.
        capture_parent_scope: bool,
    },

    /// Move the top of the operand stack down past `.0` values.
    ///
    /// A [`Receiver::Named`] receiver is resolved by [`Op::LoadNamed`] after
    /// the other arguments, and this puts it back in argument order.
    Rotate(u8),

    /// Pop a subject and jump to wherever switch table `.0` sends it.
    ///
    /// Always jumps — the table's default is where a subject that matches
    /// nothing goes, and an absent `_` arm compiles to a jump past the
    /// statement. Arms with guards are not table entries: the table sends a
    /// subject to the head of a chain that tries each guard in source order
    /// and falls through to the default, which is what keeps dispatch a
    /// lookup.
    ///
    /// The table is in the program's switch pool rather than in the
    /// instruction because it is unbounded, and because two arms of one
    /// `switch` share it.
    Switch(u32),

    /// Turn local slot `.0` into a shared cell, so a closure can capture it.
    ///
    /// Rhai's parser emits one of these per captured variable ahead of the
    /// `curry` call that binds them (`parser.rs:3707`). Sharing is what makes
    /// the closure and the enclosing scope see the same value afterwards; the
    /// write-through in `place` is the other half.
    ///
    /// The variable resolver gets first refusal, as it does in
    /// `eval/stmt.rs:998`: if a host's `on_var` answers the name, the variable
    /// is *not* shared.
    Share(u16),

    /// The same for a variable no slot names — one the caller supplied.
    ShareNamed(u32),

    /// Push local slot `.0` without flattening it.
    ///
    /// A read normally hands back what a shared cell contains, which is right
    /// for a value and wrong for a capture: currying a closure has to bind the
    /// *cell*, or the closure gets a copy and stops being a closure.
    LoadShared(u16),

    /// The same for a variable no slot names — one the caller supplied.
    ///
    /// [`Op::LoadNamed`] flattens, and a closure capturing a caller's variable
    /// through that read binds a copy: the aliasing is dead, and a write to the
    /// variable afterwards is invisible to the closure. The slot case has had
    /// [`Op::LoadShared`] since closures were lowered at all; this is the half
    /// that was missing.
    LoadSharedNamed(u32),

    /// Push the receiver bound to the running frame.
    ///
    /// `this` is not a scope entry and no slot addresses it: Rhai threads it
    /// through evaluation as a parameter (`func/script.rs:29`) and keeps it out
    /// of the `Scope` altogether. So it gets a register of its own, and these
    /// four instructions are the only things that reach it.
    ///
    /// Flattens, as [`Op::LoadLocal`] does. Rhai reads `this` *unflattened*
    /// (`eval/expr.rs:272`) but flattens at almost every consumer — a `let`
    /// (`eval/stmt.rs:436`), an assignment's right-hand side (`:321`), a call's
    /// arguments (`func/call.rs:1428`) — so the flattening read is the common
    /// one and [`Op::LoadThisShared`] is the exception, exactly as it is for a
    /// local.
    ///
    /// `ErrorUnboundThis` when the frame has no receiver.
    LoadThis,

    /// The same without flattening.
    ///
    /// [`Op::LoadShared`]'s counterpart, for the three readers that have to see
    /// the cell rather than what it holds: a `switch` subject, `is_shared`, and
    /// a curried capture.
    LoadThisShared,

    /// Raise `ErrorUnboundThis` if the frame has no receiver. Pushes nothing.
    ///
    /// `this = v` checks *before* it evaluates `v` (`eval/stmt.rs:299-302`),
    /// unlike the variable arm, which evaluates the value first (`:319-323`).
    /// Without a check of its own, `this = no_such` in an unbound frame would
    /// report `ErrorVariableNotFound` where Rhai reports `ErrorUnboundThis`.
    RequireThis,

    /// Pop a value and assign it to the frame's receiver, optionally through an
    /// operator.
    ///
    /// [`Op::AssignLocal`] without the slot or the name, because `this` has
    /// neither — and neither does Rhai's own failure here: assigning to a
    /// read-only receiver is `ErrorAssignmentToConstant("")`
    /// (`eval/stmt.rs:118-122`), named for an expression that has no name.
    AssignThis {
        /// Index into the op-assignment pool; absent for a plain `=`.
        op: Option<u32>,
    },

    /// Pop a value and push whether it is a shared cell.
    ///
    /// Rhai answers `is_shared` before dispatch and registers no function for
    /// it (`func/call.rs:1240`), so there is nothing to call — and the value
    /// has to arrive unflattened, or the answer is always false.
    IsShared,

    /// Push a function pointer to the compiled function named `.0`.
    ///
    /// A closure's, whose name the parser makes up (`anon$…`) and which
    /// [`Op::MakeFnPtr`] would refuse — Rhai only builds pointers to names a
    /// script could have written. The name is known here, so unlike
    /// `MakeFnPtr` it needs no operand on the stack.
    MakeClosure(u32),

    /// Pop a name and push a function pointer to it.
    ///
    /// Deliberately the *late-bound* kind, carrying a name and nothing else.
    /// Rhai's other kind embeds a `ScriptFuncDef` — an AST body — which is
    /// both unreachable from outside the crate and exactly the allocation this
    /// project exists to remove. Building our own means a pointer resolves
    /// through the compiled function table like any other call, and a program
    /// holding one can still be written to an artifact.
    ///
    /// The cost is that a `Normal` pointer is late-bound where Rhai's is
    /// early-bound: redefining the function after taking a pointer to it is
    /// visible here and not in the walker.
    MakeFnPtr,

    /// Pop `.0` arguments and a function pointer, and push the pointer with
    /// those arguments bound to the front of it.
    Curry(u8),

    /// Pop `argc` arguments and a target, and call a function pointer.
    ///
    /// A compiled function of that name and arity is called directly, with the
    /// curried arguments spliced in front. Anything else — a native function,
    /// a name that resolves elsewhere — goes to Rhai's own `call_raw`.
    ///
    /// `method` distinguishes `f.call(x)` from `call(f, x)`, which are not the
    /// same call. In method position a target that is *not* a pointer is not
    /// an error: Rhai takes the first argument as the pointer and binds the
    /// target as `this` (`func/call.rs:816-919`), which is how a closure is
    /// called against a receiver.
    CallFnPtr {
        /// How many arguments to pop
        argc: u8,
        /// Whether the call is in method position (`f.call(x)`).
        method: bool,
        /// Where the receiver came from, when there is anywhere to put it back.
        ///
        /// `obj.call(f)` binds `obj` as the closure's `this` **by reference**
        /// (`func/call.rs:862`), so a closure that writes to `this` writes to
        /// `obj`. The receiver's *value* is on the operand stack either way —
        /// this only says where it came from, so the write can be carried back
        /// there.
        ///
        /// `None` in call position, and for a receiver with nowhere to write
        /// back to: `[1, 2].call(f)` mutates a temporary, as it does in Rhai.
        /// Only meaningful when `method` is set.
        receiver: Option<Receiver>,
    },

    /// Push an empty buffer for an interpolated string to be built in.
    ///
    /// Interpolation is three instructions rather than one because Rhai checks
    /// the size limit after **every** segment and blames the segment that went
    /// over. One instruction has one position-table entry, so it could not say
    /// which; a pool of per-segment positions would say it but would not be
    /// strippable, and diagnostics staying separable is the point of the
    /// table. An instruction per segment puts each position exactly where the
    /// rest of them live.
    ///
    /// The buffer is an ordinary operand, so a nested interpolation needs
    /// nothing special.
    InterpolateStart,

    /// Pop a segment and append it to the buffer beneath it.
    ///
    /// Not `+`, which is what it looks like: `+` is overridable and
    /// interpolation is not, and the string-plus-anything operator skips this
    /// size check. A string segment is written straight out and never reaches
    /// dispatch; anything else goes through Rhai's `to_string` rendering,
    /// which consults native functions only.
    InterpolateAppend,

    /// Replace the buffer with the interned string it built.
    InterpolateEnd,

    /// Pop `.0` values and push them as an array.
    ///
    /// Only for a literal whose elements are not all constant — one that is
    /// gets folded into the pool by Rhai's own optimizer before this sees it.
    MakeArray(u16),

    /// Build a map from a template and `n` key/value pairs above it.
    ///
    /// The stack holds `[template, k0, v0, .., k(n-1), v(n-1)]`. The template
    /// is a constant map that already carries every key the literal mentions,
    /// with the computed ones holding a placeholder — that is Rhai's own shape
    /// (`ast/expr.rs:283`), and it is why an entirely constant map never
    /// reaches here: the optimizer has already folded it into the template
    /// alone.
    ///
    /// Keys ride on the operand stack as string constants rather than in a
    /// pool of their own. They are constants either way, and the constant pool
    /// already deduplicates them across the program.
    MakeMap(u16),

    /// Measure the value on top of the stack into the array literal being
    /// built, and raise `ErrorDataTooLarge` if the running total is over.
    ///
    /// The operand is the element's index within its literal: zero starts a
    /// fresh total, and [`Op::MakeArray`] discards it. That is what keeps
    /// `[a, [b, c], d]` straight — the inner literal's total is pushed and
    /// popped inside the outer one's.
    ///
    /// A separate instruction rather than work inside `MakeArray` because Rhai
    /// blames the *element* that tipped the total over
    /// (`eval/expr.rs:328`), and one instruction has one position-table entry.
    /// Putting it here rather than in a pool beside the element count is what
    /// keeps those positions strippable, which matters more for a literal than
    /// for a chain: an array can have any number of elements.
    CheckSize {
        /// The element's index within its literal
        index: u16,
        /// Whether the element counts towards the map limit rather than the
        /// array one. Rhai adds one to a different member of the triple for
        /// each (`eval/expr.rs:323` against `:354`), so the same running total
        /// cannot serve both.
        map: bool,
    },

    /// Walk `a.b[i].c`, indexing the chain pool.
    ///
    /// One instruction for the whole chain rather than one per step, because
    /// the walk holds a `&mut` into the container at every level and a borrow
    /// cannot survive a trip round the dispatch loop. Index values and method
    /// arguments were pushed before it, in step order.
    ///
    /// Pushes the value for a read. An assignment pushes nothing: what it
    /// evaluates to is an [`Op::Unit`] beside it, emitted only where something
    /// reads the value — which lets the peephole that already collapses that
    /// pair for a local assignment collapse this one too.
    Chain(u32),

    /// Truncate the scope back to `.0` locals, dropping everything a block
    /// declared. The compile-time slot model unwinds in step.
    UnwindTo(u16),

    /// Count one operation against `max_operations`, and give `on_progress` a
    /// chance to terminate.
    ///
    /// This compiler emits none. A cycle always contains a backward transfer of
    /// control and the dispatch loop charges an operation there, so that a chunk
    /// it did not write is stoppable too; a loop that also carried one of these
    /// at its header paid for the same turn twice. Rhai ticks per AST node, so
    /// no count matches it either way, and what a charge preserves is that a
    /// limit is enforced and an interrupt is honoured — which is what allows
    /// `loop {}` to be killed.
    ///
    /// Still decodable, because the charge is part of the artifact format and a
    /// producer other than this compiler may spell it.
    Tick,

    /// Record the current scope length as the depth an error escaping this
    /// chunk unwinds to.
    ///
    /// Rhai rewinds a nested block whether it is left normally or by a throw,
    /// and never rewinds the top level of a chunk (`eval/stmt.rs`, and
    /// `eval_global_statements` passing `rewind_scope = false`). The normal
    /// path is [`Op::UnwindTo`], which an escaping error jumps straight past —
    /// so the frame needs a floor to fall back to, and the last top-level
    /// statement boundary is exactly it.
    ///
    /// Emitted once before each top-level statement of a chunk that runs in the
    /// caller's scope, so it costs nothing per iteration and nothing at all to
    /// a function body, whose scope is discarded whole.
    Checkpoint,

    /// A statement begins here, at nesting `depth` within its chunk.
    ///
    /// Where the debugger stops. Rhai runs its callback per AST node
    /// (`eval/stmt.rs:269`) and a chunk has no nodes, so the compiler records
    /// where the statements were and the VM stops there instead. Without this
    /// there is nothing to stop at, and break-points and stepping are inert.
    ///
    /// `depth` is how many statements enclose this one. It is what stepping
    /// re-arms against: Rhai restores the stepping state when the statement it
    /// was asked at *ends* (`eval/stmt.rs:271`), and the next marker at the
    /// same depth or shallower is where that has happened. Without it, a
    /// `next` over an `if` would step into its body.
    ///
    /// Emitted only under `debugging`. An instruction per statement is not
    /// worth carrying to a device with no callback to call, so a shipping build
    /// has none. Decoding is unconditional, so an artifact written by a
    /// debugging build still runs anywhere — it simply cannot be stopped.
    Statement {
        /// How many statements enclose this one.
        depth: u16,
    },

    /// Evaluate residual AST fragment `residual` through Rhai's walker,
    /// pushing its value.
    ///
    /// `rewind_scope` reaches `eval_stmt_block` when the fragment is a block,
    /// and decides whether locals it declares survive. Statement fragments
    /// rewind, so they cannot disturb the scope shape slots were resolved
    /// against. A whole-program fragment does not, because Rhai does not
    /// rewind top-level statements and callers can see what they declared.
    EvalAst {
        /// Index into the residual pool
        residual: u32,
        /// Whether locals the fragment declares are discarded afterwards.
        rewind_scope: bool,
    },

    /// Arm a handler covering the instructions up to the matching
    /// [`Op::PopHandler`], catching to `target`.
    ///
    /// Records where the operand stack, the scope and the iterator stack were
    /// when it was armed, because an error can be raised at any depth of all
    /// three and the catch block has to start where the `try` did.
    ///
    /// Only errors Rhai considers catchable are caught: `return`, `break`,
    /// `continue` and `exit` unwind as errors too and must pass straight
    /// through (`eval/stmt.rs:806`).
    ///
    /// `catch_var` names the variable the error is bound to. Its table entry
    /// is that variable's position, which is what Rhai reports
    /// `ErrorTooManyVariables` against.
    PushHandler {
        /// Where to jump to when an error is caught.
        target: u32,
        /// The name the error is bound to; absent for a bare `catch`.
        catch_var: Option<u32>,
    },

    /// Disarm the innermost handler.
    ///
    /// Emitted twice per `try`: once where the body ends normally, and once
    /// where the catch block does — the second ends the region in which a
    /// bare `throw;` means "re-raise the original".
    PopHandler,

    /// Pop an iterable and start iterating it.
    ///
    /// The iterator goes on a stack of the VM's own rather than the operand
    /// stack, because it is not a `Dynamic`. Rhai's iterator functions take
    /// the iterable **by value** and hand back something that cannot be
    /// re-created, so it is made once here and lives until the loop ends.
    ///
    /// Its table entry is the iterable's *start* position, which is what
    /// `ErrorFor` is reported against (`eval/stmt.rs:703`) — a different
    /// position from the one [`Op::IterNext`] uses.
    IterInit,

    /// Advance the current iterator: push the next item and fall through, or
    /// drop the iterator and jump to `exit`.
    ///
    /// The only instruction whose two edges leave different amounts on the
    /// operand stack, which is why the verifier gives it explicit successors.
    ///
    /// Its table entry is the iterable's position — `position`, not
    /// `start_position` — because that is what a fallible iterator's error is
    /// filled in with (`eval/stmt.rs:749`).
    IterNext {
        /// Where to jump to once the iterator is exhausted.
        exit: u32,
        /// `for (x, i) in seq`: the count is pushed under the item, so the two
        /// `StoreShared`s that follow pop them in declaration order.
        indexed: bool,
    },

    /// Advance the current iterator straight into local slot `slot`, or drop
    /// the iterator and jump to `exit`.
    ///
    /// [`Op::IterNext`] with the [`Op::StoreShared`] that always follows it
    /// folded in. Every `for` loop over a single variable runs both on every
    /// turn, and the item they hand between them goes onto the operand stack
    /// and straight off it again.
    ///
    /// Not emitted for `for (x, i) in seq`: that pushes a count as well, so
    /// the pair is not a pair.
    ///
    /// Its table entry is [`Op::IterNext`]'s — the iterable's position, which
    /// is what a fallible iterator's error is filled in with.
    IterNextStore {
        /// Where to jump to once the iterator is exhausted.
        exit: u32,
        /// The slot the item is written into.
        slot: u16,
    },

    /// Discard the current iterator.
    ///
    /// Emitted where a `break` leaves a loop, since the jump skips the
    /// [`Op::IterNext`] that would have dropped it on exhaustion. Leaving a
    /// frame drops whatever it left behind without this.
    IterDrop,

    /// Pop a value and write it into local slot `.0`, through a shared cell
    /// rather than over it.
    ///
    /// Distinct from [`Op::StoreLocal`] only in intent: the `for` loop
    /// variable is written once per iteration and a closure in the body may
    /// have shared it, in which case Rhai writes into the cell and every
    /// closure made in the loop sees the last value (`eval/stmt.rs:752`).
    StoreShared(u16),

    /// Pop a value and raise it as a `throw`.
    ///
    /// Always fails, with `ErrorRuntime` carrying the value — Rhai wraps
    /// nothing and converts nothing, so any type can be thrown. Its table
    /// entry is the `throw` keyword's own position, not the expression's
    /// (`eval/stmt.rs:877`).
    Throw,

    /// Apply a binary operator to two operands, pushing the result.
    ///
    /// The left operand is always popped. The right is popped too unless the
    /// instruction names it: `<expression> op <local>` and `<expression> op
    /// <constant>` push a value and take it off again on the very next
    /// instruction, exactly as [`Op::BinOpFrom`]'s pair does — the difference
    /// is only that here the left operand is not a lone read, so there is
    /// nothing to fold on that side. `i % 3 == 0` and `x * 1.5` are the shape.
    ///
    /// The specialised form of [`Op::Call`] with `argc: 2` and an operator
    /// token, and it carries the same `name` and `op` so that it can *be* that
    /// call whenever it has to be. The difference is `kind`: the operator is
    /// decoded from the instruction rather than from the token pool, so an
    /// integer add reaches an integer add without a `Token` match, a
    /// `get_builtin_binary_op_fn` type walk, or an indirect call over
    /// `&mut [&mut Dynamic]`.
    ///
    /// **The types are not proven.** The compiler emits this for every binary
    /// operator [`BinOpKind`] names, and the VM tests the operands: two
    /// integers, or a pair the float rules cover, run in the dispatch loop;
    /// anything else — a string, a custom type, a shared cell — takes exactly
    /// the [`Op::Call`] path, and reports what that path reports. That is the
    /// only way the answer can be the walker's, because a host is free to hand
    /// the same site a string on one iteration and an integer on the next.
    ///
    /// Gated at run time on `Engine::fast_operators()` for the same reason
    /// [`Op::Call`]'s built-in short-circuit is: with it off, both the walker
    /// and the VM dispatch, and a host-registered `+` on integers wins in both.
    ///
    /// A result nothing reads but the branch that follows it is not pushed at
    /// all: `branch` names where control goes when the result is false, and
    /// the instruction *is* the [`Op::JumpIfFalse`] it would have been
    /// followed by. `while i < n`, `if i % 3 == 0` and every guard written
    /// that way are one instruction rather than two.
    ///
    /// The two halves report against the same place, so one position-table
    /// entry serves both: Rhai blames a guard that is not a `bool` on the
    /// guard expression's own position, and the operator that computed it is
    /// that expression.
    BinOp {
        /// The operator's name, for the dispatch fallback and error messages.
        name: u32,
        /// Index into the operator pool, for the dispatch fallback.
        op: u32,
        /// Which operator this is.
        kind: BinOpKind,
        /// Where the right operand comes from; popped when absent.
        rhs: Option<BinOperand>,
        /// Where to go when the result is false, for an operator that is also
        /// the test of a branch; the result is consumed rather than pushed.
        branch: Option<u32>,
    },

    /// Assign into `local[index]`, with the chain it specialises beside it.
    ///
    /// [`Op::Chain`] for the one shape that dominates a loop writing an array:
    /// a root that is a local slot, exactly one [`Step::Index`], and a plain
    /// `=` tail. The compiler emits it only for that shape, so the VM does not
    /// re-derive it; what the VM still tests is the *types*, which no compiler
    /// can know — the local has to hold an unshared, writable `Array` and the
    /// index has to be a non-negative in-range integer.
    ///
    /// Anything it declines runs `chain`, which is the very chain this
    /// replaced, so a receiver that turns out to be a map, a shared cell, a
    /// host type with an indexer, or an out-of-range index is answered by the
    /// generic walk and reports exactly what it reports. That is what keeps
    /// this a speculation about shape rather than a claim about types.
    ///
    /// Leaves nothing behind, as every assigning [`Op::Chain`] does.
    ///
    /// A write whose index is a lone push names it here rather than taking it
    /// off the stack, on the same trade [`BinOperand`] is for elsewhere — see
    /// [`Op::IndexGet`], which names its index the same way. A declining write
    /// pushes the named index under the value, so the walk finds both operands
    /// where the chain record says they are.
    IndexSet {
        /// Index into the chain pool, for the fallback.
        chain: u32,
        /// The slot the root local lives in.
        slot: u16,
        /// Where the index comes from, when the instruction names it rather
        /// than reading it off the stack.
        index: Option<BinOperand>,
    },

    /// Read `local[index]`, with the chain it specialises beside it.
    ///
    /// [`Op::IndexSet`] for the read half of the same shape, decided the same
    /// way and speculating on the same types: the local has to hold an
    /// unshared `Array` and the index has to be a non-negative integer inside
    /// it. A loop that reads an array is what this exists for — the write had
    /// an instruction and the read went through the walk.
    ///
    /// Anything it declines runs `chain`, which is the very chain this
    /// replaced, so a string, a map, a bitfield, a negative index, a custom
    /// indexer and an out-of-range index are answered by the generic walk and
    /// report exactly what it reports.
    ///
    /// Leaves the element where the index was, so the stack is one deep either
    /// way — and where `index` names the index instead, one deeper than it
    /// arrived.
    ///
    /// A read whose index is a lone push names it here rather than taking it
    /// off the stack, on the same trade [`BinOperand`] is for elsewhere: the
    /// index of an array read in a loop is a counter or a literal almost
    /// every time, and pushing one to consume it on the next instruction is a
    /// second trip round the dispatch loop for a value that never outlives it.
    /// A declining read pushes the named index first, so the walk finds its
    /// operand where the chain record says it is.
    IndexGet {
        /// Index into the chain pool, for the fallback.
        chain: u32,
        /// The slot the root local lives in.
        slot: u16,
        /// Where the index comes from, when the instruction names it rather
        /// than reading it off the stack.
        index: Option<BinOperand>,
    },

    /// Apply a unary operator to the top of the stack, replacing it.
    ///
    /// [`Op::BinOp`] for one operand, and speculative in the same way: the
    /// compiler emits it for the operator, the VM tests the operand, and a
    /// value the typed arm declines takes exactly the [`Op::Call`] path and
    /// reports what that path reports.
    ///
    /// No operator-pool index rides along, unlike [`Op::BinOp`]. A unary
    /// operator has no built-in lookup to fall back *to* — `!` on a non-`bool`
    /// is an ordinary call to a registered function — and the token would not
    /// survive the artifact anyway, because `UnaryMinus` and `Minus` share the
    /// syntax `"-"`. So the fallback is the plain `CALL` arm, which is what
    /// this instruction's operand layout is: the kind byte sits where the
    /// argument count would, a unary operator's count being always one.
    ///
    /// Gated at run time on `Engine::fast_operators()`, matching the walker's
    /// own unary short-circuit in `eval_fn_call_expr`.
    UnOp {
        /// The operator's name, for the dispatch fallback and error messages.
        name: u32,
        /// Which operator this is.
        kind: UnOpKind,
    },

    /// Apply a binary operator to a local and a second operand named by the
    /// instruction, pushing the result.
    ///
    /// [`Op::BinOp`] with the two loads that fed it folded in. It carries the
    /// same `name`, `op` and `kind` and falls back to exactly the same
    /// dispatch, because the operands are still not proven — it pushes them
    /// and takes the [`Op::Call`] path whenever the typed arms decline.
    ///
    /// The left operand is always a local: an operator with a constant on the
    /// left and something else on the right is rare enough that a tag for it
    /// would be dead weight, and a constant on both sides is folded before
    /// lowering ever sees it.
    ///
    /// It takes a branch on the same terms [`Op::BinOp`] does, and for the
    /// same reason: `i < n` at the head of a loop is this instruction, and the
    /// jump that reads its result is the only thing that ever does.
    BinOpFrom {
        /// The operator's name, for the dispatch fallback and error messages.
        name: u32,
        /// Index into the operator pool, for the dispatch fallback.
        op: u32,
        /// Which operator this is.
        kind: BinOpKind,
        /// The left operand's local slot.
        lhs: u16,
        /// Where the right operand comes from.
        rhs: BinOperand,
        /// Where to go when the result is false, for an operator that is also
        /// the test of a branch; the result is consumed rather than pushed.
        branch: Option<u32>,
    },

    /// End the chunk, yielding the top of the operand stack, or unit if empty.
    Return,
}
