mod cases;
mod poolable;
mod slots;

#[cfg(not(feature = "no_function"))]
use core::mem;
#[cfg(feature = "no_std")]
use std::prelude::v1::*;

#[cfg(not(feature = "no_function"))]
use crate::ast::ScriptFuncDef;
use crate::ast::{
    ASTFlags, Expr, FlowControl, FnCallExpr, OpAssignment, Stmt, StmtBlock, SwitchCasesCollection,
};
use crate::tokenizer::Token;
use crate::types::Span;
use crate::{Dynamic, ImmutableString, Position, AST};

use crate::grain::bytecode::{
    assemble, resolve_switch_targets, AssignOp, BinOpKind, BinOperand, Chain, Chunk, Op, Positions,
    Receiver, Root, Step, StepFlags, Switch, SwitchCase, SwitchRange, Tail, UnOpKind,
};
use crate::grain::compile::poolable::is_poolable;
use crate::grain::compile::slots::Slots;
use crate::grain::format::Caps;
use crate::grain::program::{Function, Parts, Program};

/// Whether a variable reference is module-qualified, as in `foo::bar`.
///
/// `Expr::Variable`'s payload only carries a `Namespace` when modules are
/// compiled in. Under `no_module` the box is two fields rather than four and
/// nothing can be qualified, so the question has a constant answer and the
/// field it would have read does not exist.
#[cfg(not(feature = "no_module"))]
macro_rules! has_namespace {
    ($payload:expr) => {
        !$payload.2.is_empty()
    };
}
#[cfg(feature = "no_module")]
macro_rules! has_namespace {
    ($payload:expr) => {{
        let _ = $payload;
        false
    }};
}

/// The same question for a call: is it `foo::bar()` rather than `bar()`.
/// `FnCallExpr` carries no `namespace` field at all under `no_module`.
#[cfg(not(feature = "no_module"))]
macro_rules! call_has_namespace {
    ($call:expr) => {
        !$call.namespace.is_empty()
    };
}
#[cfg(feature = "no_module")]
macro_rules! call_has_namespace {
    ($call:expr) => {{
        let _ = $call;
        false
    }};
}

/// Lowers a Rhai `AST` into a [`Program`].
///
/// Anything not yet lowered is kept as an AST fragment and handed back to
/// Rhai's walker at runtime, so the output always means the same as its input.
/// Progress is [`Program::residual_count`] falling.
#[derive(Debug, Default, Clone)]
pub struct Compiler {
    _private: (),
}

impl Compiler {
    /// Create a new [`Compiler`] with default options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lower an `AST` into a [`Program`].
    #[must_use]
    pub fn compile(&self, ast: &AST) -> Program<'static> {
        // A bare script-function name used as a value is not a variable read
        // at all — Rhai turns it into a function pointer with the calling
        // environment attached (`eval/expr.rs:71-99`) — so those names must
        // not become `LoadNamed`. Carried across every restart below, because
        // function bodies are lowered after one. Under `no_function` an `AST`
        // declares none, so there is nothing to hold back.
        #[cfg(not(feature = "no_function"))]
        let script_fns: Vec<ImmutableString> = ast
            .shared_lib()
            .iter_script_fn_info()
            .map(|(.., def)| def.name.clone())
            .collect();
        #[cfg(feature = "no_function")]
        let script_fns: Vec<ImmutableString> = Vec::new();

        let fresh = |caps| Lowering {
            script_fns: script_fns.clone(),
            caps,
            ..Lowering::default()
        };

        let mut lowering = fresh(Caps::empty());

        // Anything the slot model cannot account for costs the whole program
        // its lowering rather than risking a scope it resolved slots against
        // being a different shape at runtime. Coverage is preserved either way.
        if !lowering.program(ast.statements(), true) {
            lowering = fresh(lowering.caps);
            lowering.whole_program_residual(ast.statements());
        }
        let main_ops = lowering.code.len();

        // Each function's body appends to the same instruction list, so the
        // whole program assembles as one address space. A function the slot
        // model cannot handle is simply left out, and Rhai's own copy of it
        // stays reachable through the library below.
        #[cfg(not(feature = "no_function"))]
        let (functions, skipped) = lowering.functions(ast);
        #[cfg(feature = "no_function")]
        let (functions, skipped): (Vec<LoweredFn>, usize) = (Vec::new(), 0);

        // After the functions, because they share one instruction list and a
        // chain of jumps can cross out of a function body into it.
        lowering.thread_jumps();

        // Assembly can fail the same way the slot model can, for a script with
        // more distinct names or constants than a `u16` operand can index — so
        // it takes the same exit. The fallback is a single instruction and
        // always assembles, which is what keeps coverage total.
        // Switch targets are instruction indices too, and they live in the
        // pool rather than in the code, so they are resolved separately —
        // failing the same way, into the same fallback.
        let assembled = match assemble(&lowering.code) {
            Ok((code, offsets)) => resolve_switch_targets(&mut lowering.switches, &offsets)
                .ok()
                .map(|()| (code, offsets)),
            Err(..) => None,
        };
        let (code, offsets, main_ops, functions, skipped) = match assembled {
            Some((code, offsets)) => (code, offsets, main_ops, functions, skipped),
            None => {
                lowering = fresh(lowering.caps);
                lowering.whole_program_residual(ast.statements());
                let (code, offsets) =
                    assemble(&lowering.code).expect("the fallback is one instruction");
                (code, offsets, lowering.code.len(), Vec::new(), 1)
            }
        };

        // Jump targets and the position table were both keyed on instruction
        // index while lowering; instructions vary in length once assembled.
        let mut positions = vec![rhai::Position::NONE; code.len()];
        for (index, pos) in lowering.positions.iter().enumerate() {
            positions[offsets[index] as usize] = *pos;
        }

        let main = Chunk::new(0, offsets[main_ops], lowering.max_stack);
        let functions: Vec<_> = functions
            .into_iter()
            .map(|f| Function {
                name: f.name,
                params: f.params,
                this_type: f.this_type,
                // Derived from the chunk by `Program::new`, which is the one
                // place that can see the assembled bytes.
                takes_this: false,
                chunk: Chunk::new(
                    offsets[f.first_op],
                    offsets[f.first_op + f.op_count],
                    lowering.max_stack,
                ),
            })
            .collect();

        let caps = lowering.caps;

        // Rhai's own functions are carried whenever anything might still reach
        // for them: a function this compiler skipped, or a fragment that could
        // call one. With neither, every call resolves in the table above and
        // the library — an `AST`'s whole function tree — can be dropped.
        //
        // The third case is a pointer to a `this`-taking chunk. Rhai reaches a
        // compiled function through a registered wrapper, and a wrapper is
        // registered at one arity — but a native calling a pointer against a
        // receiver decides for itself how many arguments to append beside it,
        // so no single arity is right. Rhai's own pointer carries the body and
        // sizes the call from it, which is what its copy is kept here for. See
        // `callback::wrappers`, which skips exactly these.
        #[cfg(not(feature = "no_function"))]
        let lib = {
            let escapes_as_pointer = caps.contains(Caps::FN_PTR)
                && functions
                    .iter()
                    .any(|f| crate::grain::program::takes_this(&code, f.chunk, &lowering.chains));
            let needs_walker = skipped > 0 || !lowering.residuals.is_empty() || escapes_as_pointer;
            (needs_walker && !ast.shared_lib().is_empty()).then(|| ast.shared_lib().clone())
        };
        // Under `no_function` there is no function tree to carry, whichever way
        // the fallbacks above went.
        #[cfg(feature = "no_function")]
        let lib = {
            let _ = skipped;
            None
        };

        let mut program = Program::new(
            caps,
            code.into(),
            main,
            functions,
            Parts {
                positions: Positions::dense(positions),
                // Derived from what is being compiled in.
                debug_id: None,
                residuals: lowering.residuals,
                consts: lowering.consts,
                names: crate::grain::bytecode::Strings::new(&lowering.names),
                tokens: lowering.tokens,
                assign_ops: lowering.assign_ops,
                chains: lowering.chains,
                switches: lowering.switches,
                lib,
                #[cfg(not(feature = "no_module"))]
                resolver: ast.resolver.clone(),
                source: ast.source().map(Into::into),
            },
        );

        // `max_stack` above is an upper bound the lowering can compute without
        // a depth walk. The verifier does the walk anyway, so take its answer.
        program.tighten_stack();
        program
    }
}

/// Where `break` and `continue` jump to, and what they must unwind first.
///
/// Jump targets are backpatched: `break` sites are collected as they are
/// emitted and pointed at the instruction after the loop once that address is
/// known.
struct Loop {
    /// Where `continue` goes — the condition test, or the top of the body.
    continue_target: u32,
    /// Slot depth a `break` unwinds to. For a `for` loop this is *before* the
    /// loop variable, which leaving must drop.
    break_depth: u16,
    /// Slot depth a `continue` unwinds to. Differs from `break_depth` in a
    /// `for`, where the loop variable has to survive into the next iteration —
    /// one field cannot be both.
    continue_depth: u16,
    /// How many iterators are live *inside* this loop, so a jump out of it
    /// can drop whatever was made since. A `break` inside a `try` inside a
    /// `for` skips the straight-line path that would have cleaned up.
    iters: usize,
    /// Whether the loop owns an iterator of its own. `break` drops it and
    /// `continue` must not, which is the other thing one field cannot be.
    owns_iterator: bool,
    /// How many `try` regions were armed when the loop began, so a jump out
    /// of the loop disarms the ones inside it.
    handlers: usize,
    /// `Jump` sites awaiting the address after the loop.
    breaks: Vec<usize>,
}

/// Where a `switch` table entry sends control, before the arms have
/// addresses.
#[derive(Debug, Clone, Copy)]
enum Entry {
    /// Straight to an arm's body: the group has no guard to try first.
    Body(usize),
    /// The head of a guard chain, which is already emitted.
    At(u32),
    /// Nothing in the group can run.
    Default,
}

/// A function body that lowered, before its instruction indices become byte
/// addresses.
struct LoweredFn {
    name: u32,
    params: Vec<u32>,
    /// The declared receiver type, as a name-pool index. See
    /// [`Function::this_type`](crate::grain::program::Function::this_type).
    this_type: Option<u32>,
    first_op: usize,
    op_count: usize,
}

#[derive(Default)]
struct Lowering {
    /// Capabilities required by the instructions emitted so far.
    /// The compiler does not know what the caller will do with the output,
    /// so it has to assume the worst and report everything it uses.
    caps: Caps,
    code: Vec<Op>,
    /// One per instruction, parallel to `code`. Most are `NONE`; the dense
    /// shape is what makes a lookup an index, and it compacts on the way out.
    positions: Vec<Position>,
    residuals: Vec<Expr>,
    consts: Vec<Dynamic>,
    names: Vec<ImmutableString>,
    tokens: Vec<Token>,
    assign_ops: Vec<AssignOp>,
    chains: Vec<Chain>,
    switches: Vec<Switch>,
    slots: Slots,
    max_stack: u16,
    loops: Vec<Loop>,
    /// How many iterators are live at this point in the lowering, so a jump
    /// out of a loop knows how many to drop.
    iters: usize,
    /// The same for `try` regions: a `break` out of one has to disarm it, or
    /// the next unrelated error is caught into a block already left.
    handlers: usize,
    /// Names that are script functions rather than variables.
    script_fns: Vec<ImmutableString>,
    /// How many statements enclose the one being lowered, for the marker
    /// [`Lowering::statement`] emits. Restored on the way out, so it is the
    /// nesting rather than a running count.
    #[cfg(feature = "debugging")]
    stmt_depth: u16,
    /// The furthest instruction anything emitted so far jumps to.
    ///
    /// Read by [`Lowering::drop_trailing_unit`], which may only drop the tail
    /// while nothing behind it points there. Reset per chunk with the rest of
    /// the lowering, which is what makes it comparable to an index into
    /// `code`.
    patched_max: u32,
    /// Set when something nested inside an expression defeated the slot model.
    ///
    /// [`Lowering::statement`] says so by returning false, but
    /// [`Lowering::expression`] has no way to: it is called from the middle of
    /// building other expressions, and every one of those callers would have
    /// to thread the answer back. So a block used as an expression records the
    /// failure here instead, and [`Lowering::program`] reports it.
    ///
    /// A sticky flag is enough because failure is all or nothing — the caller
    /// throws the whole lowering away and starts again as one fragment — so
    /// instructions emitted after it are discarded rather than run.
    defeated: bool,
}

/// What the code around a statement wants of the value that statement has.
///
/// Rhai gives every statement a value, but only the last one in a block is
/// ever read. Telling the lowering which position it is in is what lets the
/// value of the rest never be pushed, rather than be pushed and popped again.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Wants {
    /// Leave it on the operand stack.
    Value,
    /// Leave nothing behind.
    Effect,
}

impl Lowering {
    /// Lower a statement list as a whole chunk. Returns false if something
    /// defeated the slot model and the caller should fall back.
    ///
    /// `keeps_scope` says whether what this chunk declares outlives it, which
    /// is true of the program and false of every function body. Only then is a
    /// [`Op::Checkpoint`] worth emitting: it is what an escaping error unwinds
    /// to, and a function's scope is discarded whole however it ends.
    fn program(&mut self, statements: &[Stmt], keeps_scope: bool) -> bool {
        let Some((last, leading)) = statements.split_last() else {
            self.emit(Op::Unit);
            self.emit(Op::Return);
            return true;
        };

        for stmt in leading {
            if keeps_scope {
                self.emit(Op::Checkpoint);
            }
            // A statement's value is only the program's value if it is the
            // last one; Rhai discards the rest.
            if !self.statement_discarding(stmt) {
                return false;
            }
        }

        if keeps_scope {
            self.emit(Op::Checkpoint);
        }
        if !self.statement(last) {
            return false;
        }

        self.emit(Op::Return);
        !self.defeated
    }

    /// Lower `a.b[i].c`, either reading it or assigning to it.
    ///
    /// Returns false if the chain is not one this can express, in which case
    /// the caller keeps it as a fragment.
    ///
    /// The shape is the awkward part. Rhai does not store a chain as a list:
    /// `a.b[i]` is `Dot { lhs: a, rhs: Index { lhs: b, rhs: i } }`, where each
    /// nested node's `lhs` is the *current* step's operand and its `rhs` is the
    /// continuation. [`flatten_chain`] unpicks that into steps.
    fn chain(&mut self, expr: &Expr, tail: Tail, value: Option<&Expr>) -> bool {
        let Some((root, steps)) = flatten_chain(self, expr) else {
            return false;
        };

        // A variable root is one the chain can write back into, by slot or by
        // name; anything else has to be both a read and a value Rhai would
        // itself have evaluated into a temporary.
        //
        // `this` is deliberately not in the second class. Rhai reaches it
        // through the caller's `&mut` (`eval/chaining.rs:528`), so a method
        // step that mutates lands in the caller's value — walking a copy would
        // drop the write silently. It gets a root of its own instead.
        let root_spec = match root {
            Expr::Variable(v, ..) if !has_namespace!(v) => match self.slots.resolve(&v.1) {
                Some(slot) => Root::Local {
                    slot,
                    name: self.push_name(v.1.clone()),
                },
                // The caller's, or a module's, or nothing — decided at run
                // time, because which of the three it is decides whether the
                // chain can write through it.
                //
                // The guard is load-bearing: a bare script-function name is a
                // function pointer rather than a variable, and turning one
                // into a name lookup would report it missing where Rhai hands
                // back a pointer.
                None if self.is_variable_name(&v.1, false) => Root::Named {
                    name: self.push_name(v.1.clone()),
                    pos: root.position(),
                },
                None => return false,
            },
            Expr::ThisPtr(pos) => {
                self.caps.insert(Caps::THIS);
                Root::This { pos: *pos }
            }
            // A qualified root resolves against imported modules, which need
            // `import` — the escape hatch's job.
            Expr::Variable(..) => return false,
            _ if matches!(tail, Tail::Read) => Root::Temporary,
            // Unreachable through the parser, which refuses `f().x = 1`
            // outright (`eval/chaining.rs:559`).
            _ => return false,
        };

        // Rhai evaluates the assigned value before the lvalue's own index
        // expressions and method arguments, and `Op::Chain` wants it on top of
        // them — so the value goes into a local here and is read back once the
        // operands are down.
        //
        // Two shapes do not need the local. A chain that evaluates nothing
        // between the two points cannot tell the difference: no operands, and a
        // root that is a place rather than an expression. And a value that is a
        // single infallible push cannot tell either, whatever runs in between.
        // Both are how an indexed write is usually spelled — `a[i] = 0`,
        // `m.x += 1` — where the stash was six instructions and a scope entry
        // per turn, the scope entry an `Rc` allocation of its own.
        let evaluates_between = matches!(root_spec, Root::Temporary)
            || steps.iter().any(|step| match step {
                ChainStep::Index(..) => true,
                ChainStep::Property(..) => false,
                ChainStep::Method(call, ..) => !call.args.is_empty(),
            });
        let stashed = value.map_or(false, |value| {
            evaluates_between && !is_one_pushed_value(value)
        });

        let rewind_mark = self.mark();
        let unwind_depth = self.slots.depth();

        let value_slot = if let Some(value) = value.filter(|_| stashed) {
            if self.slots.is_full() {
                return false;
            }

            let value_name = ImmutableString::from("$CHAIN_SET_VALUE$");
            let value_name_index = self.push_name(value_name.clone());
            let value_slot = self.slots.declare(value_name);

            // First evaluate the assigned value first, stash it so the chain
            // can read it back after the lvalue steps have been resolved.
            self.emit(Op::Unit);
            self.emit(Op::DeclareLocal {
                name: value_name_index,
                is_const: false,
            });
            self.expression(value);
            self.emit(Op::StoreLocal {
                slot: value_slot,
                is_const: false,
            });

            Some(value_slot)
        } else {
            None
        };

        // Index values and method arguments are evaluated first, in step
        // order, exactly as Rhai collects them before walking
        // (`eval/chaining.rs:568`). Evaluating one partway down would need the
        // operand stack while a borrow of the container is live.
        let mut lowered = Vec::with_capacity(steps.len());
        let mut operands = 0u16;

        for step in &steps {
            match step {
                ChainStep::Index(index, bracket, flags) => {
                    self.caps.insert(Caps::INDEXING);
                    self.expression(index);
                    lowered.push(Step::Index {
                        operand: operands,
                        flags: *flags,
                        pos: index.start_position(),
                        bracket: *bracket,
                    });
                    operands += 1;
                }
                ChainStep::Property(prop, pos, flags) => {
                    self.caps.insert(Caps::PROPERTY);
                    let (getter, setter, name) = &**prop;
                    lowered.push(Step::Property {
                        name: self.push_name(name.clone()),
                        getter: self.push_name(getter.0.clone()),
                        setter: self.push_name(setter.0.clone()),
                        flags: *flags,
                        pos: *pos,
                    });
                }
                ChainStep::Method(call, pos, flags) => {
                    self.caps.insert(Caps::METHOD);
                    if !self.is_lowerable_call(call) {
                        if value_slot.is_some() {
                            self.rewind(rewind_mark);
                            self.slots.unwind_to(unwind_depth);
                        }
                        return false;
                    }
                    let Ok(argc) = u8::try_from(call.args.len()) else {
                        if value_slot.is_some() {
                            self.rewind(rewind_mark);
                            self.slots.unwind_to(unwind_depth);
                        }
                        return false;
                    };
                    let first = operands;
                    for arg in call.args.iter() {
                        self.expression(arg);
                        operands += 1;
                    }
                    lowered.push(Step::Method {
                        name: self.push_name(call.name.clone()),
                        argc,
                        operand: first,
                        flags: *flags,
                        pos: *pos,
                    });
                }
            }
        }

        // Where the steps' operands end, which for a specialised chain is where
        // its one index ends — the fold below reads that slice rather than
        // everything emitted since `rewind_mark`.
        let operands_end = self.mark();

        // Then the root, if it is one that has to be evaluated. After the
        // operands rather than before, which is Rhai's order and not the
        // reading order: `[f()][g()]` calls `g` first.
        if matches!(root_spec, Root::Temporary) {
            self.expression(root);
        }

        match (value_slot, value) {
            (Some(value_slot), _) => self.emit(Op::LoadLocal(value_slot)),
            // Not stashed, so this is where it is evaluated — the same place
            // the load would have been, so the operand stack `Op::Chain` reads
            // is laid out identically either way.
            (None, Some(value)) => self.expression(value),
            (None, None) => (),
        }

        let assigns = matches!(tail, Tail::Assign { .. });
        let chain = Chain {
            root: root_spec,
            steps: lowered,
            tail,
            operands,
        };
        // `local[i] = v` and `local[i]`, which is what a loop over an array is
        // made of, get an instruction that names the slot and skips the walk.
        // Decided on the lowered chain rather than on the tree, so a shape that
        // lowers to anything else — a second step, an op-assignment, a root
        // with no slot, a null-conditional index — keeps the general
        // instruction. See [`Op::IndexSet`] and [`Op::IndexGet`].
        let specialised = indexed_slot(&chain);
        // A specialised chain lowers its index and then, if it assigns, its
        // value — nothing else — so a lone push is still recognisable here and
        // the instruction can name it instead of taking it off the stack.
        let named_index =
            specialised.and_then(|_| self.fold_index_operand(rewind_mark, operands_end));
        let index = self.push_chain(chain);
        let op = match (specialised, assigns) {
            (Some(slot), true) => Op::IndexSet {
                chain: index,
                slot,
                index: named_index,
            },
            (Some(slot), false) => Op::IndexGet {
                chain: index,
                slot,
                index: named_index,
            },
            // Reading or assigning, with the value it arrives at left where
            // whoever wants it can see it. A statement that wants none of it
            // retags the instruction rather than popping — see
            // [`Lowering::discard_trailing_chain`].
            (None, ..) => Op::Chain {
                chain: index,
                discards: false,
            },
        };
        self.emit_at(op, expr.position());
        self.unwind_to(unwind_depth);
        // An assignment evaluates to unit, and as an instruction rather than
        // as something the chain pushes: written this way the pair is the one
        // `drop_trailing_unit` already knows how to collapse, so an assignment
        // in statement position — which is nearly all of them — costs neither
        // the push nor the `Pop` that took it off again.
        if assigns {
            self.emit(Op::Unit);
        }
        true
    }

    /// Lower a `switch` into dispatch tables plus the arms they name.
    ///
    /// The layout is: evaluate and keep the subject, [`Op::Switch`] over
    /// hashed cases, a second [`Op::Switch`] over ranges for case misses and
    /// declined guards, then the arm bodies and default. Every arm leaves one
    /// value and jumps to the end, so the statement's value is the matched
    /// arm's — or unit, which is what an absent `_` compiles to.
    ///
    /// Guards are why the table does not simply hold bodies. Rhai tries the
    /// arms sharing a case value in source order and, when they all decline,
    /// continues with ranges before the default (`eval/stmt.rs:546-571`).
    /// Nearly every arm anyone writes has no guard, and those cost no chain at
    /// all.
    fn switch(&mut self, subject: &Expr, sw: &SwitchCasesCollection, wants: Wants) -> bool {
        if self.slots.is_full() {
            return false;
        }
        let unwind_depth = self.slots.depth();

        // Sorted because Rhai's map iterates in whatever order its hasher put
        // the entries in, and an artifact should not depend on that — and
        // because `Switch::dispatch` bisects the table it ends up in.
        let mut groups: Vec<(u64, Vec<usize>)> = sw
            .cases
            .iter()
            .map(|(hash, blocks)| (*hash, blocks.to_vec()))
            .collect();
        groups.sort_unstable_by_key(|(hash, ..)| *hash);

        // Overlapping range arms have no single answer at runtime, so they are
        // cut into disjoint pieces here instead. See [`cases::split`].
        let ranges = cases::split(&sw.ranges);

        // Whether anything after the first table can still want the subject.
        // Only two things can: a range arm, and a guard that declines. With no
        // ranges and no guards the first `Op::Switch` decides every outcome —
        // hit or default — so the subject is dead the moment it is dispatched
        // on, and can be taken straight off the operand stack.
        //
        // A guard is exactly an arm whose condition is not a literal: an arm
        // written without `if` parses as `Expr::BoolConstant(true)`
        // (`parser.rs:1187`), and `arm_chain` only emits a fallback jump for a
        // group it had to test something in.
        //
        // Deciding it up front rather than after `arm_chain` costs a scan of
        // the arms and saves three instructions on every execution: the
        // `let` that stashed it, the load in front of the table, and the
        // unwind that took it back off the scope.
        let guarded = sw
            .expressions
            .iter()
            .any(|arm| !matches!(arm.lhs, Expr::BoolConstant(..)));
        let stash = guarded || !ranges.is_empty();

        let value_slot = if stash {
            let value_name = ImmutableString::from("$SWITCH_VALUE$");
            let value_name_index = self.push_name(value_name.clone());
            Some((value_name_index, self.slots.declare(value_name)))
        } else {
            None
        };

        self.expression(subject);

        // Stored only when something after the table can still need it: if all
        // the arms decline, the ranges still need the subject.
        if let Some((value_name_index, value_slot)) = value_slot {
            self.emit(Op::DeclareLocal {
                name: value_name_index,
                is_const: false,
            });
            self.emit(Op::LoadLocal(value_slot));
        }

        // The first table is for the hashed case values.
        let cases_table = self.push_switch();
        self.emit(Op::Switch(cases_table));

        // One chain per distinct list of arms, shared by every table entry
        // naming it: `1 | 2 => ..` is two case values and one chain.
        let mut case_chains: Vec<(&[usize], Entry)> = Vec::new();
        let mut to_body: Vec<(usize, usize)> = Vec::new();
        let mut to_ranges: Vec<usize> = Vec::new();
        let mut to_default: Vec<usize> = Vec::new();

        for blocks in groups.iter().map(|(.., blocks)| blocks.as_slice()) {
            if case_chains.iter().any(|(ex, ..)| *ex == blocks) {
                continue;
            }
            let entry = self.arm_chain(sw, blocks, &mut to_body, &mut to_ranges);
            case_chains.push((blocks, entry));
        }

        // Dispatch to the ranges table if no case value matches or all the guards decline.
        // The default arm is only reached when all ranges fail.
        //
        // Absent when the subject was not stashed, which is the same condition:
        // nothing jumps here, and an empty ranges table would send everything
        // that fell through to the default anyway.
        let ranges_dispatch = value_slot.map(|(.., value_slot)| {
            let at = self.here();
            self.emit(Op::LoadLocal(value_slot));

            // The second table is for the ranges.
            let ranges_table = self.push_switch();
            self.emit(Op::Switch(ranges_table));
            (at, ranges_table)
        });

        // One chain per distinct list of ranges, shared by every table entry
        let mut range_chains: Vec<(&[usize], Entry)> = Vec::new();

        for blocks in ranges.iter().map(|(.., blocks)| blocks.as_slice()) {
            if range_chains.iter().any(|(ex, ..)| *ex == blocks) {
                continue;
            }
            let entry = self.arm_chain(sw, blocks, &mut to_body, &mut to_default);
            range_chains.push((blocks, entry));
        }

        debug_assert!(
            stash || to_ranges.is_empty(),
            "an unguarded switch emitted a jump to a ranges table it has not got"
        );

        // Bodies, one per arm something can reach. An arm behind a constant
        // false guard, or one whose range the parser dropped for being empty,
        // is reachable by nothing and is not emitted.
        let mut wanted: Vec<usize> = to_body.iter().map(|(.., block)| *block).collect();
        wanted.extend(
            case_chains
                .iter()
                .chain(range_chains.iter())
                .filter_map(|(.., entry)| match entry {
                    Entry::Body(block) => Some(*block),
                    _ => None,
                }),
        );
        wanted.extend(sw.def_case);
        wanted.sort_unstable();
        wanted.dedup();

        let mut body_at: Vec<(usize, u32)> = Vec::with_capacity(wanted.len());
        let mut to_end: Vec<usize> = Vec::with_capacity(wanted.len());
        let last_body = wanted.len().wrapping_sub(1);
        for (nth, block) in wanted.into_iter().enumerate() {
            body_at.push((block, self.here()));
            // An arm body is an ordinary expression, and a block one goes
            // through the same path as `let y = { .. }`.
            self.expression_wanting(&sw.expressions[block].rhs, wants);
            if self.defeated {
                self.unwind_to(unwind_depth);
                return false;
            }
            // The body emitted last falls straight into the end of the
            // statement, so it needs no jump to get there. Only when there is
            // an `_` arm, though: without one, the unit that stands in for it
            // is emitted after the bodies and has to be jumped over.
            if nth == last_body && sw.def_case.is_some() {
                continue;
            }
            to_end.push(self.emit_jump());
        }

        let at = |block: usize| {
            body_at
                .iter()
                .find(|(candidate, ..)| *candidate == block)
                .map(|(.., at)| *at)
                .expect("every reachable arm was emitted above")
        };

        let default_at = match sw.def_case {
            Some(block) => at(block),
            None => {
                let target = self.here();
                if wants == Wants::Value {
                    self.emit(Op::Unit);
                }
                target
            }
        };

        // Where a case miss goes: the ranges table when there is one, and the
        // default directly when there is not.
        let case_fallback = ranges_dispatch.map_or(default_at, |(at, ..)| at);

        // Unwind at the end of the switch. Nothing to unwind when the subject
        // was never stashed, in which case this is only the address the arms
        // converge on.
        let unwind_at = self.here();
        self.unwind_to(unwind_depth);

        for site in to_end {
            self.patch_to(site, unwind_at);
        }
        for site in to_ranges {
            self.patch_to(site, case_fallback);
        }
        for site in to_default {
            self.patch_to(site, default_at);
        }
        for (site, block) in to_body {
            self.patch_to(site, at(block));
        }

        let case_target = |blocks: &[usize]| {
            let entry = case_chains
                .iter()
                .find(|(ex, ..)| *ex == blocks)
                .map(|(.., entry)| *entry)
                .expect("every list got a chain above");
            match entry {
                Entry::Body(block) => at(block),
                Entry::At(target) => target,
                Entry::Default => case_fallback,
            }
        };

        let range_target = |blocks: &[usize]| {
            let entry = range_chains
                .iter()
                .find(|(ex, ..)| *ex == blocks)
                .map(|(.., entry)| *entry)
                .expect("every list got a chain above");
            match entry {
                Entry::Body(block) => at(block),
                Entry::At(target) => target,
                Entry::Default => default_at,
            }
        };

        self.switches[cases_table as usize] = Switch {
            cases: groups
                .iter()
                .map(|(hash, blocks)| SwitchCase {
                    hash: *hash,
                    target: case_target(blocks),
                })
                .collect(),
            ranges: Vec::new(),
            default: case_fallback,
        };
        if let Some((.., ranges_table)) = ranges_dispatch {
            self.switches[ranges_table as usize] = Switch {
                cases: Vec::new(),
                ranges: ranges
                    .iter()
                    .map(|(range, blocks)| SwitchRange {
                        target: range_target(blocks),
                        ..*range
                    })
                    .collect(),
                default: default_at,
            };
        }

        true
    }

    /// Emit the guard chain for one group of arms, and say where the table
    /// entries naming that group should point.
    fn arm_chain(
        &mut self,
        sw: &SwitchCasesCollection,
        blocks: &[usize],
        to_body: &mut Vec<(usize, usize)>,
        to_fallback: &mut Vec<usize>,
    ) -> Entry {
        let mut entry: Option<Entry> = None;

        for block in blocks {
            match &sw.expressions[*block].lhs {
                // An arm without an `if` is a literal `true` in the tree
                // (`parser.rs:1187`), so it always runs and everything after
                // it in the group is unreachable.
                Expr::BoolConstant(true, ..) => {
                    return match entry {
                        None => Entry::Body(*block),
                        Some(entry) => {
                            to_body.push((self.emit_jump(), *block));
                            entry
                        }
                    };
                }
                // Nothing can reach this arm, so nothing is emitted for it.
                Expr::BoolConstant(false, ..) => continue,
                guard => {
                    if entry.is_none() {
                        entry = Some(Entry::At(self.here()));
                    }
                    self.expression(guard);
                    let site = self.code.len();
                    // Rhai reports a non-boolean guard against the guard, so
                    // the jump carries the guard's position.
                    self.emit_at(Op::JumpIfTrue { target: u32::MAX }, guard.position());
                    to_body.push((site, *block));
                }
            }
        }

        match entry {
            Some(entry) => {
                to_fallback.push(self.emit_jump());
                entry
            }
            // Every arm in the group is behind a constant false guard, so the
            // group is the default with extra steps.
            None => Entry::Default,
        }
    }

    /// Reserve a table, to be filled in once its arms have addresses.
    fn push_switch(&mut self) -> u32 {
        self.switches.push(Switch {
            cases: Vec::new(),
            ranges: Vec::new(),
            default: 0,
        });
        (self.switches.len() - 1) as u32
    }

    fn push_chain(&mut self, chain: Chain) -> u32 {
        if let Some(index) = self.chains.iter().position(|existing| *existing == chain) {
            return index as u32;
        }
        self.chains.push(chain);
        (self.chains.len() - 1) as u32
    }

    /// Lower every script function the `AST` declares, and count the ones the
    /// slot model turned down. Sorted for reproducability.
    #[cfg(not(feature = "no_function"))]
    fn functions(&mut self, ast: &AST) -> (Vec<LoweredFn>, usize) {
        let mut defs: Vec<_> = ast
            .shared_lib()
            .iter_script_fn_info()
            .map(|(.., def)| def)
            .collect();
        defs.sort_unstable_by(|a, b| declaration_order(a).cmp(&declaration_order(b)));

        let mut functions = Vec::new();
        let mut skipped = 0;
        for def in defs {
            match self.function(def) {
                Some(function) => {
                    functions.push(function);
                    self.caps.insert(Caps::FUNCTION);
                }
                None => skipped += 1,
            }
        }
        (functions, skipped)
    }

    /// Lower one script function's body into the same instruction list.
    ///
    /// Returns `None` if the slot model cannot account for it, in which case
    /// Rhai keeps its own copy and calls to it go through dispatch. That is a
    /// per-function decision: one awkward function does not cost the rest
    /// their lowering.
    ///
    /// The body runs in a fresh scope with the parameters already pushed
    /// (`func/script.rs:73`), so the parameters are exactly slots 0 upwards.
    #[cfg(not(feature = "no_function"))]
    fn function(&mut self, def: &ScriptFuncDef) -> Option<LoweredFn> {
        let first_op = self.code.len();
        let first_residual = self.residuals.len();
        let saved_slots = mem::take(&mut self.slots);
        let saved_loops = mem::take(&mut self.loops);
        // Per-function, like the slots: one body the model cannot handle must
        // not cost the rest of the program its lowering.
        let saved_defeated = mem::replace(&mut self.defeated, false);

        for param in def.params.iter() {
            self.slots.declare(param.clone());
        }
        let params: Vec<_> = def
            .params
            .iter()
            .map(|p| self.push_name(p.clone()))
            .collect();

        // Rhai stops once on entering a body, before its first statement, at a
        // synthetic node placed on the body itself (`func/script.rs:115-119`).
        // A marker at the same place is that stop, and puts it in the chunk
        // rather than in the VM. Depth zero, like the statements it precedes:
        // it does not enclose them, so stepping from here reaches the first one.
        #[cfg(feature = "debugging")]
        self.emit_at(Op::Statement { depth: 0 }, def.body.position());

        // A body is a statement list whose last value is the return value,
        // which is what `program` already does.
        let lowered = self.program(def.body.statements(), false);

        self.slots = saved_slots;
        self.loops = saved_loops;
        self.defeated = saved_defeated;

        if !lowered {
            // Roll back whatever the attempt emitted, so a function that could
            // not be lowered leaves no unreachable instructions behind.
            //
            // The fragments go with the instructions that referred to them.
            // Rhai keeps its own copy of a body this turned down, so it is the
            // walker that evaluates what is in there — a fragment left here
            // would be one nothing can reach, counted against a program that
            // does not need it. Only this function's are dropped: the ones
            // below `first_residual` belong to code that is staying.
            self.code.truncate(first_op);
            self.positions.truncate(first_op);
            self.residuals.truncate(first_residual);
            return None;
        }

        Some(LoweredFn {
            name: self.push_name(def.name.clone()),
            params,
            // A typed `this` is a method on a custom type, which is exactly
            // what `no_object` removes — Rhai drops the field with it.
            #[cfg(not(feature = "no_object"))]
            this_type: def
                .this_type
                .as_ref()
                .map(|typed| self.push_name(typed.clone())),
            #[cfg(feature = "no_object")]
            this_type: None,
            first_op,
            op_count: self.code.len() - first_op,
        })
    }

    /// The last-resort fallback: one fragment holding everything, evaluated
    /// without rewinding so top-level declarations still reach the caller.
    fn whole_program_residual(&mut self, statements: &[Stmt]) {
        let body = wrap_statements(statements.to_vec());
        let residual = self.push_residual(body);
        self.emit(Op::EvalAst {
            residual,
            rewind_scope: false,
        });
        self.emit(Op::Return);
    }

    /// Lower one statement, leaving its value on the stack.
    ///
    /// Marks where it begins first, which is what the debugger stops at — see
    /// [`Op::Statement`]. Every statement gets one, the ones that end up as
    /// fragments included: the walker evaluating a fragment stops at its own
    /// node as well, so such a statement stops twice at the same place. Driving
    /// the residual count to zero is what removes that.
    fn statement(&mut self, stmt: &Stmt) -> bool {
        self.statement_wanting(stmt, Wants::Value)
    }

    /// The lowering [`Lowering::statement`] and
    /// [`Lowering::statement_discarding`] both are.
    fn statement_wanting(&mut self, stmt: &Stmt, wants: Wants) -> bool {
        #[cfg(feature = "debugging")]
        let enclosing = {
            let depth = self.stmt_depth;
            self.emit_at(Op::Statement { depth }, stmt.position());
            // Saturating, so a script nested past 65,535 statements marks its
            // innermost ones as siblings rather than wrapping the depth into a
            // shallower one. `max_expr_depth` stops a parse long before.
            self.stmt_depth = depth.saturating_add(1);
            depth
        };

        // A statement whose value comes out of a block can take the discard
        // into that block, so nothing is pushed for the pop below to take off
        // again. The rest leave their value here.
        let lowered = match self.lower_statement_for_effect(stmt, wants) {
            Some(lowered) => lowered,
            None => {
                let lowered = self.lower_statement(stmt);
                if lowered && wants == Wants::Effect {
                    self.discard_value();
                }
                lowered
            }
        };

        #[cfg(feature = "debugging")]
        {
            self.stmt_depth = enclosing;
        }

        lowered
    }

    /// Lower an expression, leaving its value behind or — when nothing reads
    /// it — nothing.
    ///
    /// A block expression is where a `switch` arm's statements live, and it
    /// produces its value the way a block statement does, so the discard goes
    /// into the block rather than after it. Everything else pushes and is
    /// popped.
    fn expression_wanting(&mut self, expr: &Expr, wants: Wants) {
        if let (Wants::Effect, Expr::Stmt(block)) = (wants, expr) {
            if !self.block_wanting(block.statements(), Wants::Effect) {
                self.defeated = true;
            }
            return;
        }

        self.expression(expr);
        if wants == Wants::Effect {
            self.discard_value();
        }
    }

    /// Lower a statement in a position where its value is thrown away, when
    /// it is one that produces that value by lowering a block — so the
    /// discard reaches the blocks themselves and none of them pushes.
    ///
    /// `None` for every other statement, and for every statement in value
    /// position: the caller takes the value off instead.
    fn lower_statement_for_effect(&mut self, stmt: &Stmt, wants: Wants) -> Option<bool> {
        if wants == Wants::Value {
            return None;
        }
        match stmt {
            Stmt::Block(block) => Some(self.block_wanting(block.statements(), Wants::Effect)),
            Stmt::If(payload, ..) => Some(self.if_statement(payload, Wants::Effect)),
            Stmt::Switch(payload, ..) => {
                let (subject, cases) = &**payload;
                Some(self.switch(subject, cases, Wants::Effect))
            }
            _ => None,
        }
    }

    /// Lower `if c { .. } else { .. }`, leaving one value behind or — in
    /// statement position — nothing.
    ///
    /// With nothing wanted and no `else` written there is no unit to stand in
    /// for the branch not taken, and so nothing for the branch that is taken
    /// to jump over: the false edge lands where the true one does.
    fn if_statement(&mut self, flow: &FlowControl, wants: Wants) -> bool {
        let FlowControl { expr, body, branch } = flow;

        self.expression(expr);
        let to_else = self.emit_jump_if_false(expr.position());

        if !self.block_wanting(body.statements(), wants) {
            return false;
        }

        if wants == Wants::Effect && branch.statements().is_empty() {
            self.patch_here(to_else);
            return true;
        }

        let past_else = self.emit_jump();

        self.patch_here(to_else);
        if !self.block_wanting(branch.statements(), wants) {
            return false;
        }
        self.patch_here(past_else);
        true
    }

    /// The lowering itself, one arm per kind of statement.
    fn lower_statement(&mut self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Var(payload, flags, ..) => {
                // `export let x = ...` also binds a module alias, which the
                // slot model does not represent.
                if flags.contains(ASTFlags::EXPORTED) || self.slots.is_full() {
                    return false;
                }
                let is_const = flags.contains(ASTFlags::CONSTANT);

                let (ident, init, index) = &**payload;
                self.expression(init);

                if let Some(index) = index {
                    let slot = self.slots.depth() - index.get();
                    let slot =
                        u16::try_from(slot).expect("slot index is within the compiler's range");
                    self.emit(Op::StoreLocal { slot, is_const });
                } else {
                    let name = self.push_name(ident.name.clone());
                    self.slots.declare(ident.name.clone());
                    self.emit(Op::DeclareLocal { name, is_const });
                }

                // A declaration evaluates to unit.
                self.emit(Op::Unit);
                true
            }

            Stmt::Expr(expr) => {
                self.expression(expr);
                true
            }

            // Rhai gives a call standing alone as a statement its own node
            // rather than wrapping it in `Stmt::Expr`, and an operator is a
            // call — so without this every top-level `a * b` stayed a fragment.
            // A closure's `curry` lands here rather than in `Stmt::Expr`,
            // because Rhai gives a call standing alone as a statement its own
            // node.
            Stmt::FnCall(call, pos) if self.fn_ptr_call(call, *pos) => true,

            Stmt::FnCall(call, pos) if self.is_lowerable_call(call) => {
                self.lower_call(call, *pos);
                true
            }

            // Standing alone is the position `eval` is usually written in, and
            // Rhai gives it its own node — so this is the arm that catches it,
            // not the `Expr::FnCall` one. See there for why it defeats the
            // lowering rather than becoming a fragment.
            Stmt::FnCall(call, ..) if call.name == crate::engine::KEYWORD_EVAL => false,

            // `this` on the left. Ahead of the two variable arms because Rhai's
            // parser puts it there too (`parser.rs:2002`), and because the
            // chain arm below would otherwise take `this.x = 1`'s sibling.
            Stmt::Assignment(payload) if matches!(&payload.1.lhs, Expr::ThisPtr(..)) => {
                self.caps.insert(Caps::THIS);

                let (op_info, binary) = &**payload;

                // Before the right-hand side, not after. Rhai checks that
                // `this` is bound and returns before it evaluates the value
                // (`eval/stmt.rs:300-303`) — unlike the variable arm, which
                // evaluates first — so an unbound `this = no_such` is
                // `ErrorUnboundThis` and not the value's own failure.
                self.emit_at(Op::RequireThis, binary.lhs.position());

                self.expression(&binary.rhs);
                let op = self.op_assignment(op_info);

                self.emit_at(Op::AssignThis { op }, op_info.position());
                self.emit(Op::Unit);
                true
            }

            // A plain local on the left.
            Stmt::Assignment(payload)
                if matches!(&payload.1.lhs, Expr::Variable(v, ..)
                    if !has_namespace!(v) && self.slots.resolve(&v.1).is_some()) =>
            {
                let (op_info, binary) = &**payload;
                let Expr::Variable(v, ..) = &binary.lhs else {
                    unreachable!("checked by the guard");
                };
                let slot = self.slots.resolve(&v.1).expect("checked by the guard");
                let var_name = self.push_name(v.1.clone());

                let mark = self.mark();
                self.expression(&binary.rhs);
                let op = self.op_assignment(op_info);

                let assign = match self.fold_pushed_operand(mark) {
                    Some(src) => Op::AssignLocalFrom {
                        slot,
                        var_name,
                        op,
                        src,
                    },
                    None => Op::AssignLocal { slot, var_name, op },
                };
                self.emit_at(assign, op_info.position());
                self.emit(Op::Unit);
                true
            }

            // A variable no slot names — the caller's. Same shape as above,
            // and the same op-assignment resolution; only where the target
            // lives differs.
            Stmt::Assignment(payload)
                if matches!(&payload.1.lhs, Expr::Variable(v, ..)
                    if self.is_variable_name(&v.1, has_namespace!(v))) =>
            {
                let (op_info, binary) = &**payload;
                let Expr::Variable(v, ..) = &binary.lhs else {
                    unreachable!("checked by the guard");
                };
                let name = self.push_name(v.1.clone());

                self.expression(&binary.rhs);
                let op = self.op_assignment(op_info);

                // The variable's position, not the operator's — unlike
                // `AssignLocal`. The errors this instruction raises itself are
                // `ErrorAssignmentToConstant` and `ErrorVariableNotFound`, and
                // Rhai reports both against the variable (`eval/stmt.rs:340`
                // and `eval/stmt.rs:120`). For a local those are unreachable,
                // because the parser rejects a constant it can see; for a name
                // the caller supplied they are the common failures.
                self.emit_at(Op::AssignNamed { name, op }, binary.lhs.position());
                self.emit(Op::Unit);
                true
            }

            // A chain on the left. The value goes on the stack after the
            // chain's own operands, so the walk has everything it needs before
            // it takes a borrow of the container.
            Stmt::Assignment(payload)
                if matches!(&payload.1.lhs, Expr::Dot(..) | Expr::Index(..)) =>
            {
                if matches!(&payload.1.lhs, Expr::Dot(..)) {
                    self.caps.insert(Caps::PROPERTY);
                } else {
                    self.caps.insert(Caps::INDEXING);
                }

                let (op_info, binary) = &**payload;
                let op = self.op_assignment(op_info);

                let mark = self.mark();
                if !self.chain(&binary.lhs, Tail::Assign { op }, Some(&binary.rhs)) {
                    self.rewind(mark);
                    let residual = self.push_residual(wrap_statements(vec![stmt.clone()]));
                    self.emit(Op::EvalAst {
                        residual,
                        rewind_scope: true,
                    });
                }
                // The chain leaves unit, which is what an assignment evaluates
                // to, so there is nothing to add here.
                true
            }

            // Emitted by the parser ahead of the `curry` call that binds a
            // closure's captures (`parser.rs:3707`).
            #[cfg(not(feature = "no_closure"))]
            Stmt::Share(names) => {
                self.caps.insert(Caps::SHARING);

                for (ident, ..) in names.iter() {
                    match self.slots.resolve(&ident.name) {
                        Some(slot) => self.emit_at(Op::Share(slot), ident.pos),
                        None => {
                            // The caller's — a closure can capture something
                            // no slot addresses.
                            let name = self.push_name(ident.name.clone());
                            self.emit_at(Op::ShareNamed(name), ident.pos);
                        }
                    }
                }
                self.emit(Op::Unit);
                true
            }

            Stmt::Block(block) => self.block_wanting(block.statements(), Wants::Value),

            // `try { .. } catch (e) { .. }`.
            //
            // The catch block's value is thrown away: Rhai's whole statement
            // is the try block's value on the way through and *unit* when
            // something was caught (`.map(|_| Dynamic::UNIT)`,
            // `eval/stmt.rs:863`). So `try { throw 7 } catch (e) { e * 2 }` is
            // unit, not 14.
            Stmt::TryCatch(payload, ..) => {
                let FlowControl { expr, body, branch } = &**payload;

                // An absent catch variable is `Expr::Unit`; a present one is
                // an `Expr::Variable` whose position is what Rhai reports
                // `ErrorTooManyVariables` against.
                let catch_var = match expr {
                    Expr::Variable(v, ..) => Some(v.1.clone()),
                    _ => None,
                };

                let catch_name = catch_var.clone().map(|name| self.push_name(name));
                let site = self.code.len();
                self.emit_at(
                    Op::PushHandler {
                        target: u32::MAX,
                        catch_var: catch_name,
                    },
                    expr.position(),
                );
                self.handlers += 1;

                if !self.block(body.statements()) {
                    return false;
                }
                self.emit(Op::PopHandler);
                self.handlers -= 1;
                let past = self.emit_jump();

                // The catch block, entered with the scope back where the `try`
                // began and the variable already pushed on top of it. The
                // handler is still armed here — that is what makes a bare
                // `throw;` in this block a re-raise — so the depth goes back
                // up, and the `PopHandler` below is what ends the region.
                self.patch_to(site, self.here());
                self.handlers += 1;
                let depth = self.slots.depth();
                if let Some(name) = catch_var {
                    self.slots.declare(name);
                }
                if !self.block(branch.statements()) {
                    return false;
                }
                self.emit(Op::Pop);
                self.unwind_to(depth);
                self.emit(Op::PopHandler);
                self.handlers -= 1;
                self.emit(Op::Unit);

                self.patch_here(past);
                true
            }

            // `for x in seq` / `for (x, i) in seq`.
            //
            // The loop variable and counter are pushed once and written each
            // time round, not re-pushed — Rhai does the same (`stmt.rs:708`),
            // and it is observable: a closure made in the body captures the
            // cell, so every one of them sees the last value.
            Stmt::For(payload, ..) => {
                let (var, counter, flow) = &**payload;
                let outside = u16::try_from(self.slots.depth()).expect("slot count is bounded");

                self.expression(&flow.expr);
                // `ErrorFor` is reported against the iterable's *start*, which
                // for `a.b` or a call is not its `position`.
                self.emit_at(Op::IterInit, flow.expr.start_position());
                self.iters += 1;

                // Counter first, matching the order Rhai pushes them in, so
                // the slots line up with the scope it builds.
                let counter_slot = counter.as_ref().map(|ident| {
                    let name = self.push_name(ident.name.clone());
                    self.emit(Op::Unit);
                    self.emit(Op::DeclareLocal {
                        name,
                        is_const: false,
                    });
                    self.slots.declare(ident.name.clone());
                    self.slots.depth() as u16 - 1
                });
                let var_name = self.push_name(var.name.clone());
                self.emit(Op::Unit);
                self.emit(Op::DeclareLocal {
                    name: var_name,
                    is_const: false,
                });
                self.slots.declare(var.name.clone());
                let var_slot = self.slots.depth() as u16 - 1;

                let top = self.here();
                let exit = self.code.len();
                match counter_slot {
                    // `for (x, i) in seq` pushes a count as well, so the item
                    // and the count come off the operand stack in the order
                    // the two locals were declared.
                    Some(slot) => {
                        self.emit_at(
                            Op::IterNext {
                                exit: u32::MAX,
                                indexed: true,
                            },
                            flow.expr.position(),
                        );
                        self.emit(Op::StoreShared(var_slot));
                        self.emit(Op::StoreShared(slot));
                    }
                    // One variable, so the item goes onto the operand stack
                    // and straight off it again on the next instruction —
                    // every turn of every ordinary `for` loop. Fused, it never
                    // goes there at all.
                    None => self.emit_at(
                        Op::IterNextStore {
                            exit: u32::MAX,
                            slot: var_slot,
                        },
                        flow.expr.position(),
                    ),
                }
                self.begin_for(top, outside);
                if !self.block_discarding(flow.body.statements()) {
                    return false;
                }
                self.emit_at(Op::Jump(top), flow.body.position());
                let breaks = self.end_loop();

                // Exhausted: `IterNext` dropped the iterator on the way here.
                self.patch_to(exit, self.here());
                self.iters -= 1;
                self.emit(Op::UnwindTo(outside));
                self.slots.unwind_to(outside as usize);

                self.emit(Op::Unit);
                let past = self.emit_jump();
                for site in breaks {
                    self.patch_here(site);
                }
                self.patch_here(past);
                true
            }

            Stmt::Switch(payload, ..) => {
                let (subject, cases) = &**payload;
                self.switch(subject, cases, Wants::Value)
            }

            Stmt::If(payload, ..) => self.if_statement(payload, Wants::Value),

            // `loop` and `while true` are the same node: Rhai marks an
            // unconditional loop with a unit or `true` guard
            // (`eval/stmt.rs:575-576`).
            Stmt::While(payload, ..) => {
                let FlowControl { expr, body, .. } = &**payload;
                let unconditional = matches!(expr, Expr::Unit(..) | Expr::BoolConstant(true, ..));

                let top = self.here();

                let exit = if unconditional {
                    None
                } else {
                    self.expression(expr);
                    Some(self.emit_jump_if_false(expr.position()))
                };

                self.begin_loop(top);
                if !self.block_discarding(body.statements()) {
                    return false;
                }
                // The back edge is where the turn is charged an operation —
                // the dispatch loop meters every backward transfer, so an
                // instruction at the header metering it a second time is one
                // dispatch and one position lookup per iteration for a count
                // nothing reads. It carries the body's position because that
                // is the place the charge is reported against.
                self.emit_at(Op::Jump(top), body.position());

                let breaks = self.end_loop();
                if let Some(exit) = exit {
                    self.patch_here(exit);
                }
                // A `while` that runs to completion is unit; a `break value`
                // supplies its own. Both arrive here with the stack balanced.
                self.emit(Op::Unit);
                let past = self.emit_jump();
                for site in breaks {
                    self.patch_here(site);
                }
                self.patch_here(past);
                true
            }

            Stmt::Do(payload, flags, ..) => {
                let FlowControl { expr, body, .. } = &**payload;
                let until = flags.contains(ASTFlags::NEGATED);

                let top = self.here();

                self.begin_loop(top);
                if !self.block_discarding(body.statements()) {
                    return false;
                }
                let breaks = self.end_loop();

                self.expression(expr);
                if until {
                    // `do ... until c` loops while `c` is false, which is a
                    // false-branch straight back to the top.
                    self.emit_branch_if_false(top, expr.position());
                } else {
                    let exit = self.emit_jump_if_false(expr.position());
                    self.emit_at(Op::Jump(top), body.position());
                    self.patch_here(exit);
                }

                self.emit(Op::Unit);
                let past = self.emit_jump();
                for site in breaks {
                    self.patch_here(site);
                }
                self.patch_here(past);
                true
            }

            Stmt::BreakLoop(value, flags, pos) => {
                let Some(active) = self.loops.last() else {
                    // Outside any loop this is a parse error in Rhai, so it
                    // should be unreachable; bail rather than emit a jump to
                    // nowhere.
                    return false;
                };
                let continue_target = active.continue_target;
                let loop_iters = active.iters;
                let loop_handlers = active.handlers;
                let owns_iterator = active.owns_iterator;
                let (break_depth, continue_depth) = (active.break_depth, active.continue_depth);
                let is_break = flags.contains(ASTFlags::BREAK);

                // A jump out of a loop skips whatever the straight-line path
                // would have cleaned up. The nesting is lexical, so how many
                // iterators are live is known here — a `break` inside a `try`
                // inside a `for` has one to drop, and `continue` has none
                // because it re-enters the loop that owns it.
                if is_break {
                    match value {
                        Some(expr) => self.expression(expr),
                        None => self.emit(Op::Unit),
                    }
                    // Out of the loop entirely, so its own iterator goes too —
                    // `loop_iters` counts from inside the loop and therefore
                    // already includes it.
                    self.pop_handlers(loop_handlers);
                    self.drop_iterators(loop_iters - usize::from(owns_iterator));
                    self.emit(Op::UnwindTo(break_depth));
                    let site = self.emit_jump();
                    self.loops.last_mut().expect("checked").breaks.push(site);
                } else {
                    // Back into the same loop, so its iterator and its loop
                    // variable both have to survive.
                    self.pop_handlers(loop_handlers);
                    self.drop_iterators(loop_iters);
                    self.emit(Op::UnwindTo(continue_depth));
                    // A back edge of its own, and charged as one, so it names
                    // the place the charge is reported against.
                    self.emit_at(Op::Jump(continue_target), *pos);
                }

                // Unreachable, but every statement must leave a value for the
                // caller's `Pop`, and the verifier checks depth on every path.
                self.emit(Op::Unit);
                true
            }

            // `throw` shares this node, flagged, and unwinds as an error
            // rather than returning. The position is the keyword's, not the
            // expression's (`eval/stmt.rs:877`).
            Stmt::Return(value, flags, pos) if flags.contains(ASTFlags::BREAK) => {
                match value {
                    Some(expr) => self.expression(expr),
                    None => self.emit(Op::Unit),
                }
                self.emit_at(Op::Throw, *pos);
                // Unreachable, but every statement leaves a value for the
                // caller's `Pop` and the verifier checks depth on every path.
                self.emit(Op::Unit);
                true
            }

            Stmt::Return(value, flags, ..) if !flags.contains(ASTFlags::BREAK) => {
                match value {
                    Some(expr) => self.expression(expr),
                    None => self.emit(Op::Unit),
                }
                self.emit(Op::Return);
                self.emit(Op::Unit);
                true
            }

            // The one statement the fragment fallback below cannot hold.
            //
            // `import` declares into the imports stack rather than the scope,
            // and a fragment that rewinds truncates that stack on the way out
            // (`eval/stmt.rs:55`) — so the alias would be gone before the next
            // statement could name it, and a qualified call is its own
            // fragment. Refusing the lowering hands the body to the walker
            // whole, which is where the alias lives long enough to be used.
            #[cfg(not(feature = "no_module"))]
            Stmt::Import(..) => {
                self.caps.insert(Caps::IMPORT);
                false
            }

            // Not lowered yet, and listed rather than matched with `_` on
            // purpose. A wildcard here silently turned `import` and `eval`
            // into fragments that answered differently from the walker; naming
            // every kind means a new one added to Rhai's AST stops the build
            // until someone has decided which of the three it is — lowered,
            // fragment, or too scope-shaped to be either.
            //
            // The ones below are fragments because each either declares
            // nothing or rewinds what it declares, so the scope is the same
            // shape afterwards. That is the property to check before adding to
            // this list.
            other @ (Stmt::Noop(..)
            | Stmt::FnCall(..)
            | Stmt::Assignment(..)
            | Stmt::Return(..)) => {
                let residual = self.push_residual(wrap_statements(vec![other.clone()]));
                self.emit(Op::EvalAst {
                    residual,
                    rewind_scope: true,
                });
                true
            }

            #[cfg(not(feature = "no_module"))]
            other @ Stmt::Export(..) => {
                self.caps.insert(Caps::EXPORT);
                let residual = self.push_residual(wrap_statements(vec![other.clone()]));
                self.emit(Op::EvalAst {
                    residual,
                    rewind_scope: true,
                });
                true
            }
        }
    }

    /// Lower one expression, leaving its value on the stack.
    fn expression(&mut self, expr: &Expr) {
        match expr {
            Expr::BoolConstant(value, ..) => self.emit(Op::Bool(*value)),
            Expr::Unit(..) => self.emit(Op::Unit),

            Expr::IntegerConstant(value, ..) => self.constant(Dynamic::from(*value)),
            Expr::CharConstant(value, ..) => self.constant(Dynamic::from(*value)),
            Expr::StringConstant(value, ..) => self.constant(Dynamic::from(value.clone())),
            // Rhai has no float literal to parse under `no_float`, so there is
            // no variant to match.
            #[cfg(not(feature = "no_float"))]
            Expr::FloatConstant(value, ..) => {
                self.caps.insert(Caps::FLOAT);
                self.constant(Dynamic::from(**value))
            }
            // Folded by the optimizer, so it can hold anything a constant call
            // returned — including a function pointer, which must not be
            // copied out of a pool. See `poolable`.
            // A function pointer the optimizer folded — `Fn("f")` with a
            // constant name, or a closure literal. It cannot go in the pool:
            // a closure's carries a `ScriptFuncDef`, which is an AST body and
            // exactly what an artifact must not contain. Rebuilt by name
            // instead, which reaches the chunk we compiled from that same
            // body.
            //
            // A constant function pointer: a closure literal, or what the
            // optimizer folds `Fn("f")` into. Either way it embeds a
            // `ScriptFuncDef` — an AST body, `Fn*` in Rhai's own rendering —
            // so it cannot go in the pool. Rebuilt by name instead, reaching
            // the chunk compiled from that same body.
            //
            // There is no version of this that keeps the rendering: the thing
            // that differs *is* the tree, and carrying it is what an artifact
            // must not do. `a_closure_pointer_is_late_bound` pins the
            // difference for both spellings.
            //
            // Curried values are arbitrary `Dynamic`s with the same problem one
            // level down, and are left to the walker.
            Expr::DynamicConstant(value, ..)
                if value
                    .read_lock::<rhai::FnPtr>()
                    .map_or(false, |f| f.curry().is_empty()) =>
            {
                let name = value
                    .read_lock::<rhai::FnPtr>()
                    .expect("checked by the guard")
                    .fn_name()
                    .to_string();
                let name = self.push_name(name.into());
                self.emit_at(Op::MakeClosure(name), expr.position());
                self.caps.insert(Caps::FN_PTR);
            }

            Expr::DynamicConstant(value, ..) if is_poolable(value) => {
                #[cfg(not(feature = "no_index"))]
                if value.is_array() {
                    self.caps.insert(Caps::ARRAY);
                }
                #[cfg(not(feature = "no_index"))]
                if value.is_blob() {
                    self.caps.insert(Caps::BLOB);
                }
                #[cfg(not(feature = "no_object"))]
                if value.is_map() {
                    self.caps.insert(Caps::MAP);
                }
                #[cfg(feature = "decimal")]
                if value.is_decimal() {
                    self.caps.insert(Caps::DECIMAL);
                }
                self.constant((**value).clone());
            }

            Expr::Variable(payload, ..) => {
                // A qualified name resolves against imported modules, not the
                // scope, so it is not a slot.
                let is_qualified = has_namespace!(payload);

                match self.slots.resolve(&payload.1) {
                    Some(slot) if !is_qualified => self.emit(Op::LoadLocal(slot)),
                    // Not a local this compiler declared, so no slot can name
                    // it: it is the caller's, a module's, or nothing. Looked
                    // up by name at run time, at the cost of a scope scan.
                    _ if self.is_variable_name(&payload.1, is_qualified) => {
                        let name = self.push_name(payload.1.clone());
                        self.emit_at(Op::LoadNamed(name), expr.position());
                    }
                    // A qualified name resolves against imported modules, and
                    // a bare function name is a function pointer. Neither is a
                    // variable read, and both stay Rhai's job.
                    _ => self.residual_expr(expr),
                }
            }

            Expr::And(operands, ..) => self.short_circuit(operands, false),
            Expr::Or(operands, ..) => self.short_circuit(operands, true),
            Expr::Coalesce(operands, ..) => self.coalesce(operands),

            Expr::FnCall(call, pos) if self.fn_ptr_call(call, *pos) => {}

            Expr::FnCall(call, pos) if self.is_lowerable_call(call) => {
                self.lower_call(call, *pos);
            }

            // `eval` evaluates a script in the *caller's* scope, so what it
            // declares outlives it and the next statement can name it. The
            // slot model resolved its indices against a scope that does not
            // have those entries, so a lowered read past an `eval` looks in
            // the wrong place — `eval("let x = 40"); x + 2` found no `x` where
            // the walker found 40. Refusing the lowering hands the body to the
            // walker, which is the only thing that knows the real shape.
            Expr::FnCall(call, ..) if call.name == crate::engine::KEYWORD_EVAL => {
                self.residual_expr(expr);
                self.defeated = true;
            }

            // A literal whose elements are all constant never reaches here —
            // Rhai's optimizer folds it into a `DynamicConstant` first — so
            // this is the one that has to be built at run time.
            #[cfg(not(feature = "no_index"))]
            Expr::Array(elements, ..) if elements.len() <= u16::MAX as usize => {
                self.caps.insert(Caps::ARRAY);

                for (index, element) in elements.iter().enumerate() {
                    self.expression(element);
                    // Positioned at the element, because that is what Rhai
                    // blames when this element is the one that tips the
                    // running total over the limit.
                    self.emit_at(
                        Op::CheckSize {
                            index: index as u16,
                            map: false,
                        },
                        element.position(),
                    );
                }
                self.emit_at(Op::MakeArray(elements.len() as u16), expr.position());
            }

            // The other half of the same shape. Rhai keeps a map literal as a
            // template holding every key — the constant values already in
            // place, the computed ones as placeholders — plus the list of
            // entries still to evaluate (`ast/expr.rs:283`). An all-constant
            // map is folded into a `DynamicConstant` and never arrives here;
            // one with a single computed value does, and used to fragment.
            #[cfg(not(feature = "no_object"))]
            Expr::Map(entries, ..) if entries.0.len() <= u16::MAX as usize => {
                self.caps.insert(Caps::MAP);

                let (computed, template) = &**entries;
                let template = Dynamic::from_map(template.clone());
                // A template whose constants the pool cannot hold is a program
                // that could not be written to an artifact anyway.
                if !is_poolable(&template) {
                    self.residual_expr(expr);
                    return;
                }

                self.constant(template);
                for (index, (key, value)) in computed.iter().enumerate() {
                    self.constant(key.name.clone().into());
                    self.expression(value);
                    self.emit_at(
                        Op::CheckSize {
                            index: index as u16,
                            map: true,
                        },
                        value.position(),
                    );
                }
                self.emit_at(Op::MakeMap(computed.len() as u16), expr.position());
            }

            // A block used for its value: `let y = if c { 1 } else { 2 }`,
            // `let y = switch ..`, `let y = { let z = 1; z }`. Rhai evaluates
            // it with `restore_orig_state` set (`eval/expr.rs:434`), so it
            // rewinds what it declared — which is what `block` emits.
            Expr::Stmt(block) => {
                if !self.block(block.statements()) {
                    self.defeated = true;
                }
            }

            // The optimizer folds an all-constant interpolation away before
            // this sees it, so what arrives has at least two segments.
            Expr::InterpolatedString(segments, ..) => {
                self.emit(Op::InterpolateStart);
                for segment in segments.iter() {
                    self.expression(segment);
                    // The append carries the segment's own position, because
                    // that is what Rhai blames when the size limit goes over.
                    self.emit_at(Op::InterpolateAppend, segment.position());
                }
                self.emit(Op::InterpolateEnd);
            }

            // `f.call(x)` and `f.curry(x)` are the method spellings of the two
            // above. They arrive as chains, so they have to be taken before
            // the chain walker sees them.
            // An `rhs` that is a bare `MethodCall` is the whole chain: a
            // further step would make it a `Dot` or an `Index` instead.
            Expr::Dot(binary, ..)
                if matches!(&binary.rhs, Expr::MethodCall(m, ..)
                    if matches!(m.name.as_str(), "call" | "curry")
                        && m.args.len() <= u8::MAX as usize) =>
            {
                self.caps.insert(Caps::METHOD);

                let Expr::MethodCall(method, ..) = &binary.rhs else {
                    unreachable!("checked by the guard");
                };
                // `obj.call(f)` binds `obj` as the closure's `this` by
                // reference (`func/call.rs:862`), so a write inside the closure
                // has to reach `obj`. The value goes on the stack as it always
                // did; the receiver says where to carry a write back to.
                //
                // Unflattened, for the reason `unflattened` gives: a receiver
                // that is a shared cell has to arrive *as* the cell, so a write
                // lands where every holder can see it and no write-back is
                // needed at all.
                let receiver = self.fn_ptr_receiver(&binary.lhs);
                if receiver.is_some() {
                    self.unflattened(&binary.lhs);
                } else {
                    self.expression(&binary.lhs);
                }
                for arg in method.args.iter() {
                    self.expression(arg);
                }
                let argc = method.args.len() as u8;
                // The call's own position, which is what Rhai reports for
                // everything the pointer path can raise. The one case it is
                // not is `obj.call(x)` where `obj` is not a pointer and `x` is
                // taken as one: Rhai blames `x` (`func/call.rs:838`). Both
                // cannot come from one position-table entry, and using the
                // argument's instead was measured to move the divergence onto
                // the common path rather than remove it.
                //
                // Method style only. `curry(f, ..)` written as a call is a
                // different path in Rhai and takes the *argument's* position —
                // see `fn_ptr_call`. The two disagreeing is deliberate.
                let pos = binary.rhs.position();
                if method.name == "call" {
                    self.emit_at(
                        Op::CallFnPtr {
                            argc,
                            method: true,
                            receiver,
                        },
                        pos,
                    );
                    self.caps.insert(Caps::FN_PTR);
                } else {
                    self.emit_at(Op::Curry(argc), pos);
                    self.caps.insert(Caps::FN_PTR | Caps::CURRYING);
                }
            }

            Expr::Dot(..) | Expr::Index(..) => {
                if matches!(expr, Expr::Dot(..)) {
                    self.caps.insert(Caps::PROPERTY);
                } else {
                    self.caps.insert(Caps::INDEXING);
                }

                // A chain emits its own operands, so a failed attempt has to
                // leave nothing behind.
                let mark = self.mark();
                if !self.chain(expr, Tail::Read, None) {
                    self.rewind(mark);
                    self.residual_expr(expr);
                }
            }

            // Custom syntax runs host code against an `EvalContext`, which can
            // declare into the caller's scope. What it declares is invisible
            // here, so the slot model would be resolved against a scope shape
            // that is not the one at runtime. Refusing the lowering keeps the
            // walker's answer, as it does for `eval` above.
            #[cfg(not(feature = "no_custom_syntax"))]
            Expr::Custom(..) => {
                self.residual_expr(expr);
                self.defeated = true;
            }

            // Listed rather than matched with `_`, for the reason
            // [`Lowering::statement`] gives: a wildcard is what let `eval`
            // become a fragment that answered differently from the walker.
            //
            // These are fragments because none of them can change the shape of
            // the scope the slot model resolved its indices against. The
            // guarded arms above fall through to here when their guard fails —
            // a pool-defeating constant, a literal too long for its operand, a
            // call Rhai resolves syntactically.
            // The frame's receiver, flattened as every consumer but three
            // wants it — see [`Op::LoadThis`] and `unflattened` below. Its own
            // position, because that is what `ErrorUnboundThis` carries.
            Expr::ThisPtr(pos) => {
                self.caps.insert(Caps::THIS);
                self.emit_at(Op::LoadThis, *pos)
            }

            Expr::MethodCall(..)
            | Expr::Property(..)
            | Expr::DynamicConstant(..)
            | Expr::FnCall(..)
            | Expr::Array(..)
            | Expr::Map(..) => self.residual_expr(expr),
        }
    }

    /// Where `obj.call(f)`'s receiver came from, when a write through the
    /// closure's `this` has somewhere to land.
    ///
    /// `None` for anything Rhai would evaluate into a temporary — `[1, 2].call(f)`
    /// mutates a copy in the walker too, so there is nothing to carry back.
    fn fn_ptr_receiver(&mut self, receiver: &Expr) -> Option<Receiver> {
        match receiver {
            Expr::ThisPtr(..) => {
                self.caps.insert(Caps::THIS);
                Some(Receiver::This)
            }
            Expr::Variable(payload, ..) if !has_namespace!(payload) => {
                match self.slots.resolve(&payload.1) {
                    Some(slot) => Some(Receiver::Local(slot)),
                    None if self.is_variable_name(&payload.1, false) => {
                        Some(Receiver::Named(self.push_name(payload.1.clone())))
                    }
                    None => None,
                }
            }
            _ => None,
        }
    }

    /// Where Rhai's method-call rewrite would take this call's first argument
    /// from, if it applies at all (`func/call.rs:1434`).
    fn receiver(&mut self, call: &FnCallExpr) -> Option<Receiver> {
        // An operator short-circuits before the rewrite is reached, and a call
        // that captures the enclosing scope is excluded from it outright
        // (`func/call.rs:1387` and `:1775`).
        if call.op_token.is_some() || call.capture_parent_scope {
            return None;
        }

        // `f(this, ..)` takes the same rewrite as a variable. Rhai also requires
        // the receiver not to be shared and nothing to be curried
        // (`func/call.rs:1417`), and neither is a question the compiler can
        // answer: sharing is a run-time property, deferred to the VM as it
        // already is for a read-only local, and a curried redirect can never
        // reach this instruction because `call`/`curry` go through
        // `Op::CallFnPtr` and `is_lowerable_call` refuses them here.
        if let Some(Expr::ThisPtr(..)) = call.args.first() {
            self.caps.insert(Caps::THIS);
            return Some(Receiver::This);
        }

        let Some(Expr::Variable(payload, ..)) = call.args.first() else {
            return None;
        };
        let qualified = has_namespace!(payload);

        match self.slots.resolve(&payload.1) {
            Some(slot) if !qualified => Some(Receiver::Local(slot)),
            _ if self.is_variable_name(&payload.1, qualified) => {
                Some(Receiver::Named(self.push_name(payload.1.clone())))
            }
            _ => None,
        }
    }

    /// Push the arguments left to right, then dispatch.
    fn lower_call(&mut self, call: &FnCallExpr, pos: Position) {
        let capture_parent_scope = call.capture_parent_scope;
        let argc = u8::try_from(call.args.len()).expect("checked by is_lowerable_call");

        // `f(x, ..)` is `x.f(..)`, so the variable is read after the other
        // arguments and by reference. See [`Op::CallRef`].
        if let Some(receiver) = self.receiver(call) {
            // `this` goes on *first*, unlike either of the others. Rhai's two
            // arms disagree about when it is read: the by-reference one takes a
            // pointer after the arguments (`func/call.rs:1417`), but the
            // fallback a shared or unbound receiver lands in reads and flattens
            // it before them (`:1462`). Reading first is what makes an unbound
            // `f(this, no_such)` report `ErrorUnboundThis`, and what stops an
            // argument that writes to `this` being seen by the value passed.
            if let Receiver::This = receiver {
                self.emit_at(Op::LoadThis, call.args[0].position());
            }
            for arg in call.args.iter().skip(1) {
                self.expression(arg);
            }
            // A name is resolved here, where its own position is the one an
            // `ErrorVariableNotFound` wants, and then moved under the arguments
            // it was read after.
            if let Receiver::Named(var) = receiver {
                self.emit_at(Op::LoadNamed(var), call.args[0].position());
                if argc > 1 {
                    self.emit(Op::Rotate(argc - 1));
                }
            }

            let name = self.push_name(call.name.clone());
            self.emit_at(
                Op::CallRef {
                    name,
                    argc,
                    receiver,
                    capture_parent_scope,
                },
                pos,
            );
            return;
        }

        let mark = self.mark();
        // Where the LAST argument's code starts, which for a binary operator
        // is its right-hand side. The pair fold below wants the whole range;
        // the right-hand fold wants only this much of it.
        let mut rhs_mark = mark;
        for arg in call.args.iter() {
            rhs_mark = self.mark();
            self.expression(arg);
        }
        let name = self.push_name(call.name.clone());

        // The one unary operator the walker short-circuits, as its own
        // instruction. It is emitted before the operator pool is consulted
        // because it wants nothing from it. See [`Op::UnOp`].
        if argc == 1 {
            if let Some(kind) = call.op_token.as_ref().and_then(UnOpKind::of) {
                self.emit_at(Op::UnOp { name, kind }, pos);
                return;
            }
        }

        // Only for a binary operator, which is the only shape the built-in
        // lookup takes. A unary one could not carry an index in any case:
        // `UnaryMinus` and `Minus` share the syntax `"-"`, so it is a token
        // that cannot be written to an artifact at all.
        let token = (argc == 2).then(|| call.op_token.clone()).flatten();
        let kind = token.as_ref().and_then(BinOpKind::of);
        let op = token.map(|token| self.push_token(token));

        // The specialised instruction whenever the operator is one the VM can
        // execute. Not because the operand types are known — they are not, and
        // a site can see a different pair on every iteration — but because the
        // instruction *carries* the dispatching form and falls back to it for
        // any pair it cannot run. Emitting it is therefore never a claim about
        // types, only about the operator. See [`Op::BinOp`].
        if let (Some(kind), Some(op)) = (kind, op) {
            // Both operands pushed by one instruction each and taken off again
            // by this one — `i < n`, `a + b`, `n - 1`. Read off the instruction
            // instead. Recognised on the emitted code rather than on the tree,
            // so an operand that lowers to anything else is left alone, and
            // nothing can jump between them because no label has been taken
            // since `mark`.
            if let Some((lhs, rhs)) = self.fold_pushed_operands(mark) {
                self.emit_at(
                    Op::BinOpFrom {
                        name,
                        op,
                        kind,
                        lhs,
                        rhs,
                        branch: None,
                    },
                    pos,
                );
                return;
            }
            // Failing that, the right-hand side alone: `i % 3 == 0` computes
            // its left operand and cannot fold it, but the `0` is still a push
            // this instruction takes straight off again.
            let rhs = self.fold_pushed_operand(rhs_mark);
            self.emit_at(
                Op::BinOp {
                    name,
                    op,
                    kind,
                    rhs,
                    branch: None,
                },
                pos,
            );
            return;
        }

        self.emit_at(
            Op::Call {
                name,
                argc,
                op,
                capture_parent_scope,
            },
            pos,
        );
    }

    /// Whether a name read is a variable read at all.
    ///
    /// A qualified name resolves against imported modules rather than the
    /// scope, and a bare script-function name is a function pointer with the
    /// calling environment attached (`eval/expr.rs:71-99`). Neither is
    /// something to look up by name, and both stay fragments.
    fn is_variable_name(&self, name: &ImmutableString, qualified: bool) -> bool {
        !qualified && !self.script_fns.contains(name)
    }

    /// Pool what `x op= y` needs, if there is an operator at all.
    fn op_assignment(&mut self, op_info: &OpAssignment) -> Option<u32> {
        op_info
            .get_op_assignment_info()
            .map(|(_, _, op_assign, op_assign_str, op, op_str)| {
                let entry = AssignOp::new(
                    op_assign.clone(),
                    self.push_name(op_assign_str.into()),
                    op.clone(),
                    self.push_name(op_str.into()),
                );
                self.push_assign_op(entry)
            })
    }

    /// Read a variable without flattening it, leaving a shared cell shared.
    ///
    /// Rhai's own variable read works this way — `Target::take_or_clone` hands
    /// back the shared value untouched (`eval/target.rs:233`) — and the places
    /// that want the contents flatten for themselves. [`Op::LoadLocal`]
    /// flattens instead, which is right where the value is what matters and
    /// wrong in the two places the cell is:
    ///
    /// * a closure's captured variable, where the aliasing *is* the capture;
    /// * a `switch` subject, which Rhai refuses to match on when it is not
    ///   hashable, and a shared value is not — so a shared subject falls to the
    ///   default arm however well it would otherwise have matched.
    fn unflattened(&mut self, expr: &Expr) {
        match expr {
            Expr::Variable(payload, ..) if !has_namespace!(payload) => {
                match self.slots.resolve(&payload.1) {
                    Some(slot) => self.emit(Op::LoadShared(slot)),
                    // The caller's. A closure can capture one of those too, and
                    // reading it flat would bind a copy.
                    None if self.is_variable_name(&payload.1, false) => {
                        let name = self.push_name(payload.1.clone());
                        self.emit_at(Op::LoadSharedNamed(name), expr.position());
                    }
                    None => self.expression(expr),
                }
            }
            // The receiver can be a shared cell too — a closure capturing the
            // variable a method was called on — and the three readers that come
            // through here have to see the cell rather than what it holds.
            Expr::ThisPtr(pos) => {
                self.caps.insert(Caps::THIS);
                self.emit_at(Op::LoadThisShared, *pos)
            }
            other => self.expression(other),
        }
    }

    /// Lower `Fn(name)`, `curry(f, ..)` or `call(f, ..)`, if this is one.
    ///
    /// Rhai resolves these three by name before dispatch, but only at the
    /// arities it recognizes (`func/call.rs:1109-1245`); anything else is an
    /// ordinary call that will not find a function. Matching those arities
    /// exactly is what keeps the two agreeing on the failures as well as the
    /// successes.
    fn fn_ptr_call(&mut self, call: &FnCallExpr, pos: Position) -> bool {
        if call_has_namespace!(call) || call.capture_parent_scope {
            return false;
        }
        let argc = call.args.len();

        match (call.name.as_str(), argc) {
            // The argument has to arrive as the cell, not its contents, or the
            // answer is always false.
            //
            // Not lowered under `no_closure`: Rhai registers no `is_shared`
            // there, so the call has to reach the walker and fail the way Rhai
            // fails it. Lowering it would answer a question Rhai refuses.
            #[cfg(not(feature = "no_closure"))]
            (crate::engine::KEYWORD_IS_SHARED, 1) => {
                self.caps.insert(Caps::SHARING);
                self.unflattened(&call.args[0]);
                self.emit_at(Op::IsShared, pos);
            }
            // All of these are reported against the *argument* rather than
            // against the call: Rhai reads it, and everything it can then
            // complain about — a name that is not a string, a string that is
            // not an identifier, a first argument that is not a pointer — is
            // filled in with the argument's position (`func/call.rs:1217`,
            // `:1220`, `:1232`).
            (crate::engine::KEYWORD_FN_PTR, 1) => {
                self.expression(&call.args[0]);
                self.emit_at(Op::MakeFnPtr, call.args[0].position());
                self.caps.insert(Caps::FN_PTR);
            }
            (crate::engine::KEYWORD_FN_PTR_CURRY, _) if argc > 1 => {
                let mut args = call.args.iter();
                self.expression(args.next().expect("checked by the arity"));
                for arg in args {
                    // The captured variables. These must bind the *cell* — a
                    // flattening read would hand the closure a copy and it
                    // would stop being one.
                    self.unflattened(arg);
                }
                self.emit_at(Op::Curry((argc - 1) as u8), call.args[0].position());
                self.caps.insert(Caps::FN_PTR | Caps::CURRYING);
            }
            (crate::engine::KEYWORD_FN_PTR_CALL, _)
                if argc >= 1 && argc <= u8::MAX as usize + 1 =>
            {
                for arg in call.args.iter() {
                    self.expression(arg);
                }
                self.emit_at(
                    Op::CallFnPtr {
                        argc: (argc - 1) as u8,
                        method: false,
                        // Call position binds no receiver at all.
                        receiver: None,
                    },
                    pos,
                );
                self.caps.insert(Caps::FN_PTR);
            }
            _ => return false,
        }
        true
    }

    /// Whether a call can go through generic dispatch.
    ///
    /// Rhai resolves a handful of names syntactically before dispatch ever happens,
    /// so routing those through `call_fn_raw` would change what they mean.
    /// A call that captures the enclosing scope is closure construction,
    /// and a qualified name resolves against imported modules;
    /// neither is a plain call.
    fn is_lowerable_call(&self, call: &FnCallExpr) -> bool {
        // These are handled by `is_syntactic_call` above, but only at the
        // arities Rhai treats syntactically — at any other arity it falls
        // through to ordinary dispatch, and so must catch them here.
        const SYNTACTIC: &[&str] = &[
            crate::engine::KEYWORD_EVAL,
            crate::engine::KEYWORD_FN_PTR,
            crate::engine::KEYWORD_FN_PTR_CALL,
            crate::engine::KEYWORD_FN_PTR_CURRY,
            #[cfg(not(feature = "no_closure"))]
            crate::engine::KEYWORD_IS_SHARED,
        ];

        !call_has_namespace!(call)
            && call.args.len() <= u8::MAX as usize
            && !SYNTACTIC.contains(&call.name.as_str())
    }

    /// Lower `&&` or `||`: evaluate operands left to right, stopping at the
    /// first that decides the result.
    ///
    /// Each operand is coerced to bool at its own position, which is why the
    /// jumps carry one — Rhai reports a non-boolean operand against the
    /// operand, not the expression (`eval/expr.rs:367-399`).
    fn short_circuit(&mut self, operands: &[Expr], stop_on: bool) {
        let mut decided = Vec::new();

        for operand in operands {
            self.expression(operand);
            let pos = operand.position();
            // Only the false half folds: an operator carrying the branch that
            // reads it has one spelling, and `||` stops on the other answer.
            let site = if stop_on {
                let site = self.code.len();
                self.emit_at(Op::JumpIfTrue { target: u32::MAX }, pos);
                site
            } else {
                self.emit_jump_if_false(pos)
            };
            decided.push(site);
        }

        self.emit(Op::Bool(!stop_on));
        let past = self.emit_jump();
        for site in decided {
            self.patch_here(site);
        }
        self.emit(Op::Bool(stop_on));
        self.patch_here(past);
    }

    /// Lower `??`: evaluate operands left to right, stopping at the
    /// first that is not unit.
    fn coalesce(&mut self, operands: &[Expr]) {
        let mut decided = Vec::new();
        let last = operands.len();

        for operand in operands {
            self.expression(operand);

            // Leave the last operand to fall through, so it is the one that decides
            // if all the other operands are `()`
            if decided.len() < last - 1 {
                let pos = operand.position();
                let site = self.code.len();
                self.emit_at(Op::SkipIfNotUnit { target: u32::MAX }, pos);
                self.emit_at(Op::Pop, pos);
                decided.push(site);
            }
        }
        for site in decided {
            self.patch_here(site);
        }
    }

    /// Lower a block, leaving its value — the last statement's, or unit if
    /// empty — on the stack, and dropping anything it declared.
    fn block(&mut self, statements: &[Stmt]) -> bool {
        self.block_wanting(statements, Wants::Value)
    }

    /// The lowering [`Lowering::block`] and [`Lowering::block_discarding`]
    /// both are, with the discard — when there is one — carried into the last
    /// statement rather than applied to what it left.
    ///
    /// Carrying it in is what keeps the value from being pushed at all, and it
    /// puts the discard in front of the [`Op::UnwindTo`] instead of behind it,
    /// where [`Lowering::drop_trailing_unit`] can still see the unit it drops.
    fn block_wanting(&mut self, statements: &[Stmt], wants: Wants) -> bool {
        let depth = self.slots.depth();

        let Some((last, leading)) = statements.split_last() else {
            if wants == Wants::Value {
                self.emit(Op::Unit);
            }
            return true;
        };

        for stmt in leading {
            if !self.statement_discarding(stmt) {
                return false;
            }
        }
        if !self.statement_wanting(last, wants) {
            return false;
        }

        self.unwind_to(depth);
        true
    }

    /// Lower a block for its effects only, leaving nothing on the stack.
    ///
    /// Loop bodies discard their value: Rhai's loops yield unit or whatever a
    /// `break` supplied, never the body's last statement.
    fn block_discarding(&mut self, statements: &[Stmt]) -> bool {
        self.block_wanting(statements, Wants::Effect)
    }

    /// Lower a statement whose value is thrown away, leaving nothing behind.
    fn statement_discarding(&mut self, stmt: &Stmt) -> bool {
        self.statement_wanting(stmt, Wants::Effect)
    }

    /// Take the value on top of the operand stack off again — by not putting
    /// it there when that is possible, and popping it when it is not.
    fn discard_value(&mut self) {
        if !self.drop_trailing_unit() && !self.discard_trailing_chain() {
            self.emit(Op::Pop);
        }
    }

    /// Drop the [`Op::Unit`] a statement ended with instead of emitting the
    /// [`Op::Pop`] that would take it off again.
    ///
    /// Every assignment and every declaration evaluates to unit, and a
    /// statement anywhere but last has its value discarded — so the pair is two
    /// instructions that undo each other. They are two of the eight a turn of
    /// `for i in 0..n { t += i; }` runs, and four of the fourteen in
    /// `while i < n { s += i; i += 1; }`.
    ///
    /// Refused unless the `Unit` is the last instruction emitted and nothing
    /// already jumps to it or past it. `try`'s "past the catch block" edge is
    /// patched exactly past one, and the try block's own value arrives along
    /// that edge, so dropping the catch block's unit would leave the two edges
    /// meeting at different stack heights.
    fn drop_trailing_unit(&mut self) -> bool {
        let Some(last) = self.code.len().checked_sub(1) else {
            return false;
        };
        if !matches!(self.code[last], Op::Unit) {
            return false;
        }
        if self.patched_max as usize >= last {
            return false;
        }
        self.rewind(last);
        true
    }

    /// Retag the [`Op::Chain`] a statement ended with as the spelling that
    /// drops what the walk arrived at, instead of emitting the [`Op::Pop`]
    /// that would take it off again.
    ///
    /// [`Lowering::drop_trailing_unit`]'s trade for the chain that has no unit
    /// to drop. `a.push(i);` is a whole method call run for its effect, and the
    /// value it hands back is taken off on the very next instruction — so the
    /// pair is two trips round the dispatch loop where the walk is one, and it
    /// is the shape of nearly every chain a loop body writes.
    ///
    /// Only a reading chain is retagged. An assigning one already leaves
    /// nothing, so the value being discarded here came from somewhere before
    /// it and a retag would take off something that was never pushed.
    ///
    /// [`Op::IndexGet`] keeps its `Pop`. It is the same trade in principle,
    /// but it is three spellings rather than one, so it is three tags — and
    /// what it would buy is `a[i];` written as a statement, which reads a
    /// container and does nothing with what it read.
    ///
    /// Refused, as the unit is, unless the chain is the last instruction
    /// emitted and nothing already jumps to it or past it: an arrival carries
    /// its own stack height, and two edges meeting at different ones is what
    /// the verifier exists to catch.
    ///
    /// Retagged in place rather than re-emitted, so the instruction keeps its
    /// side-table entry — a chain that raises names the same position either
    /// way.
    fn discard_trailing_chain(&mut self) -> bool {
        let Some(last) = self.code.len().checked_sub(1) else {
            return false;
        };
        let Some(&Op::Chain {
            chain,
            discards: false,
        }) = self.code.get(last)
        else {
            return false;
        };
        if !matches!(self.chains[chain as usize].tail, Tail::Read) {
            return false;
        }
        if self.patched_max as usize >= last {
            return false;
        }
        self.code[last] = Op::Chain {
            chain,
            discards: true,
        };
        true
    }

    /// Emit the scope truncation for leaving a block, and unwind the
    /// compile-time slot model with it.
    ///
    /// The value the block produced is already on the operand stack, so it
    /// survives locals being dropped.
    fn unwind_to(&mut self, depth: usize) {
        if self.slots.depth() > depth {
            let depth = u16::try_from(depth).expect("slot count is bounded");
            self.emit(Op::UnwindTo(depth));
            self.slots.unwind_to(depth as usize);
        }
    }

    /// Point every jump past the jumps it lands on.
    ///
    /// An `if` or `switch` arm ends by jumping to where the arms converge, and
    /// when the statement is the last thing in a loop body that convergence
    /// point *is* the back edge -- so the arm dispatches `Jump` twice in a row
    /// to reach the loop header. One of the two does nothing but arrive.
    ///
    /// Run after everything is emitted, so it sees whole chains and does not
    /// have to keep [`Lowering::patched_max`] honest.
    fn thread_jumps(&mut self) {
        for site in 0..self.code.len() {
            // Not a handler: its target is where a throw lands, so what was
            // dispatched before it is whatever raised, and the reasoning in
            // [`Lowering::threaded_target`] has no predecessor to stand on.
            if matches!(self.code[site], Op::PushHandler { .. }) {
                continue;
            }
            let Some(target) = jump_target_mut(&mut self.code[site]).map(|target| *target) else {
                continue;
            };
            let Some(threaded) = self.threaded_target(site, target) else {
                continue;
            };
            if let Some(slot) = jump_target_mut(&mut self.code[site]) {
                *slot = threaded;
            }
        }
    }

    /// Where a jump at `site` to `target` really arrives, or `None` when
    /// sending it straight there would change what the JIT driver sees.
    ///
    /// The grain driver has no `can_enter_jit`. It recognises a back edge
    /// instead, by the program counter failing to advance from the last
    /// instruction dispatched (`vm/jit.rs`), so a chain of jumps opens its
    /// door once for every hop that arrives at or below where it came from.
    /// Removing hops is only sound while that count is unchanged: threading
    /// `100 -> 50 -> 200` would turn one opening into none.
    ///
    /// For the shape this exists for -- a forward jump to a convergence point
    /// that jumps back to a loop header -- the count is one either way, and
    /// the opening lands on the same instruction.
    fn threaded_target(&self, site: usize, target: u32) -> Option<u32> {
        let mut arrivals = 0;
        let mut from = site as u32;
        let mut at = target;

        // Bounded by the instruction count: `loop {}` lowers to a jump to
        // itself, and a chain of them can be a cycle.
        for _ in 0..self.code.len() {
            if at <= from {
                arrivals += 1;
            }
            match self.code.get(at as usize) {
                Some(Op::Jump(next)) if *next != at => {
                    from = at;
                    at = *next;
                }
                _ => break,
            }
        }

        let direct = u32::from(at <= site as u32);
        (at != target && arrivals == direct).then_some(at)
    }

    /// Where the instruction list currently ends, for [`Lowering::rewind`].
    fn mark(&self) -> usize {
        self.code.len()
    }

    /// Drop everything emitted since `mark`.
    ///
    /// Only safe for an attempt that emitted no jumps out of the rewound
    /// region, which is why it is used for chains and nothing else: a chain
    /// emits its operands and then one instruction, and gives up before
    /// emitting the instruction.
    fn rewind(&mut self, mark: usize) {
        self.code.truncate(mark);
        self.positions.truncate(mark);
    }

    /// Take back a lone local read emitted since `mark`, naming the slot it
    /// read, so the instruction about to be emitted can do it itself.
    ///
    /// `x op= y` where `y` is a plain local read or a constant pushes a value
    /// and pops it again on the next instruction. Recognised on the emitted
    /// code rather than on the tree, so whatever `expression` decided a bare
    /// variable is decides this too — a shared capture lowers to
    /// `Op::LoadShared` and is left alone, and a right-hand side that folded to
    /// a literal is taken whether it was written as one or not.
    ///
    /// A loop counter's `i += 1` is the constant case, and it is why the
    /// operand is not just a slot: that pair sits in the body of nearly every
    /// counted loop, and in each arm of a `switch` whose arms accumulate.
    ///
    /// Safe to take back because nothing can jump between the two: no label
    /// has been taken since `mark`, so no jump can name the instruction that
    /// followed the read.
    fn fold_pushed_operand(&mut self, mark: usize) -> Option<BinOperand> {
        let src = match self.code.get(mark..)? {
            [Op::LoadLocal(src)] => BinOperand::Local(*src),
            [Op::Const(index)] => BinOperand::Const(*index),
            _ => return None,
        };
        self.rewind(mark);
        Some(src)
    }

    /// Take back an indexed chain's index push, naming what it read.
    ///
    /// The same trade as [`Self::fold_pushed_operand`], for an operand that is
    /// not always the last thing emitted: a read lowers only its index, but a
    /// write lowers the index and then the value. `mark..end` is the index's
    /// own code, which is what decides whether there is a lone push to take —
    /// a call's last argument is a lone push too, and reading back from the
    /// end would take that one and hand the instruction an argument as its
    /// index.
    ///
    /// At most one instruction may follow it, so nothing between the index and
    /// the instruction can be a jump target and removing the push moves no
    /// address anything names.
    fn fold_index_operand(&mut self, mark: usize, end: usize) -> Option<BinOperand> {
        if self.code.len() > end + 1 {
            return None;
        }
        let src = match self.code.get(mark..end)? {
            [Op::LoadLocal(src)] => BinOperand::Local(*src),
            [Op::Const(index)] => BinOperand::Const(*index),
            _ => return None,
        };
        self.code.remove(mark);
        self.positions.remove(mark);
        Some(src)
    }

    /// Take back the two pushes emitted since `mark` when they are a local
    /// read and a local read or a constant, naming what they read.
    ///
    /// The same trade as [`Self::fold_pushed_operand`], for the operand pair in
    /// front of a binary operator.
    fn fold_pushed_operands(&mut self, mark: usize) -> Option<(u16, BinOperand)> {
        let pair = match self.code.get(mark..)? {
            [Op::LoadLocal(lhs), Op::LoadLocal(rhs)] => (*lhs, BinOperand::Local(*rhs)),
            [Op::LoadLocal(lhs), Op::Const(rhs)] => (*lhs, BinOperand::Const(*rhs)),
            _ => return None,
        };
        self.rewind(mark);
        Some(pair)
    }

    fn here(&self) -> u32 {
        u32::try_from(self.code.len()).expect("chunk length is bounded")
    }

    /// Emit a jump with a placeholder target, returning its site for patching.
    fn emit_jump(&mut self) -> usize {
        let site = self.code.len();
        self.emit(Op::Jump(u32::MAX));
        site
    }

    fn emit_jump_if_false(&mut self, pos: Position) -> usize {
        self.emit_branch_if_false(u32::MAX, pos)
    }

    /// Emit the false-branch that reads a condition, returning its site.
    ///
    /// Folded into the operator that computed the condition whenever there is
    /// one — see [`Self::fold_branch`] — so the site returned may be an
    /// instruction that was already there.
    fn emit_branch_if_false(&mut self, target: u32, pos: Position) -> usize {
        if let Some(site) = self.fold_branch(target, pos) {
            return site;
        }
        let site = self.code.len();
        self.emit_at(Op::JumpIfFalse { target }, pos);
        site
    }

    /// Give the operator that computed a condition the branch that reads it.
    ///
    /// `while i < n`, `if i % 3 == 0` and every guard written that way push a
    /// `bool` that the very next instruction takes straight off again — the
    /// same trade [`BinOperand`] is for elsewhere, made on an operator's
    /// result rather than on its operands. The operator instruction carries
    /// the dispatching form it falls back to, so a pair the typed arms decline
    /// still answers what the two instructions answered.
    ///
    /// Refused unless the operator is the last instruction emitted and its
    /// position is the branch's own, so that the one position-table entry
    /// left serves both errors: a guard that is not a `bool` is blamed on the
    /// guard expression, and that expression is the operator.
    ///
    /// Refused, too, when anything has already been patched to where the
    /// branch would go. That edge exists — an `&&` operand's short circuit
    /// leaves one behind — and it has to arrive at a test rather than past
    /// one.
    fn fold_branch(&mut self, target: u32, pos: Position) -> Option<usize> {
        let last = self.code.len().checked_sub(1)?;
        if self.patched_max as usize > last || self.positions[last] != pos {
            return None;
        }
        match &mut self.code[last] {
            Op::BinOp {
                branch: branch @ None,
                ..
            }
            | Op::BinOpFrom {
                branch: branch @ None,
                ..
            } => *branch = Some(target),
            _ => return None,
        }
        self.note_target(target);
        Some(last)
    }

    /// Point a previously emitted jump at the next instruction.
    fn patch_here(&mut self, site: usize) {
        let target = self.here();
        self.patch_to(site, target);
    }

    /// Point a previously emitted jump at an instruction already emitted.
    fn patch_to(&mut self, site: usize, target: u32) {
        let Some(slot) = jump_target_mut(&mut self.code[site]) else {
            unreachable!("patched a {:?}, which is not a jump", self.code[site]);
        };
        *slot = target;
        self.note_target(target);
    }

    /// Remember the furthest instruction anything jumps to.
    ///
    /// `u32::MAX` is the placeholder a forward jump carries until it is
    /// patched, and it is patched to wherever the code ends at that moment —
    /// which is after any tail already dropped — so it names nothing yet.
    fn note_target(&mut self, target: u32) {
        if target != u32::MAX {
            self.patched_max = self.patched_max.max(target);
        }
    }

    /// Emit an `IterDrop` for every iterator live above `floor`.
    fn drop_iterators(&mut self, floor: usize) {
        for _ in floor..self.iters {
            self.emit(Op::IterDrop);
        }
    }

    /// Disarm every `try` region entered above `floor`.
    ///
    /// A `break` or `continue` jumps over the `PopHandler` the straight-line
    /// path would have run. Left armed, the handler keeps a stale target and a
    /// stale set of depths, and the next error anywhere in the frame is caught
    /// into a `catch` block that has already been left.
    fn pop_handlers(&mut self, floor: usize) {
        for _ in floor..self.handlers {
            self.emit(Op::PopHandler);
        }
    }

    /// Open a loop whose `break` and `continue` unwind to the same place —
    /// `while`, `loop` and `do`, which declare nothing of their own.
    fn begin_loop(&mut self, continue_target: u32) {
        let depth = u16::try_from(self.slots.depth()).expect("slot count is bounded");
        self.loops.push(Loop {
            continue_target,
            break_depth: depth,
            continue_depth: depth,
            iters: self.iters,
            handlers: self.handlers,
            owns_iterator: false,
            breaks: Vec::new(),
        });
    }

    /// Open a `for`, which does declare: the loop variable and any counter
    /// live between the two depths, so leaving drops them and going round
    /// again does not.
    fn begin_for(&mut self, continue_target: u32, break_depth: u16) {
        self.loops.push(Loop {
            continue_target,
            break_depth,
            continue_depth: u16::try_from(self.slots.depth()).expect("slot count is bounded"),
            iters: self.iters,
            handlers: self.handlers,
            owns_iterator: true,
            breaks: Vec::new(),
        });
    }

    fn end_loop(&mut self) -> Vec<usize> {
        self.loops.pop().expect("loop stack is balanced").breaks
    }

    fn residual_expr(&mut self, expr: &Expr) {
        let residual = self.push_residual(expr.clone());
        self.emit(Op::EvalAst {
            residual,
            rewind_scope: true,
        });
    }

    fn constant(&mut self, value: Dynamic) {
        let index = self.push_const(value);
        self.emit(Op::Const(index));
    }

    fn push_const(&mut self, value: Dynamic) -> u32 {
        // Programs at this scale make a linear scan cheaper than a hash map,
        // and it keeps the pool in emission order for readable disassembly.
        let rendered = format!("{value:?}");
        if let Some(index) = self
            .consts
            .iter()
            .position(|existing| format!("{existing:?}") == rendered)
        {
            return index as u32;
        }
        self.consts.push(value);
        (self.consts.len() - 1) as u32
    }

    fn push_name(&mut self, name: ImmutableString) -> u32 {
        if let Some(index) = self.names.iter().position(|existing| *existing == name) {
            return index as u32;
        }
        self.names.push(name);
        (self.names.len() - 1) as u32
    }

    /// A script uses a handful of distinct operators however many times it
    /// mentions them, so the pool stays tiny and a linear scan is right.
    fn push_token(&mut self, token: Token) -> u32 {
        if let Some(index) = self.tokens.iter().position(|existing| *existing == token) {
            return index as u32;
        }
        self.tokens.push(token);
        (self.tokens.len() - 1) as u32
    }

    fn push_assign_op(&mut self, entry: AssignOp) -> u32 {
        if let Some(index) = self
            .assign_ops
            .iter()
            .position(|existing| *existing == entry)
        {
            return index as u32;
        }
        self.assign_ops.push(entry);
        (self.assign_ops.len() - 1) as u32
    }

    fn push_residual(&mut self, expr: Expr) -> u32 {
        self.residuals.push(expr);
        (self.residuals.len() - 1) as u32
    }

    fn emit(&mut self, mut op: Op) {
        // A back edge names its target when it is emitted rather than being
        // patched later, so this is the other half of what `patch_to` records.
        if let Some(&mut target) = jump_target_mut(&mut op) {
            self.note_target(target);
        }
        // An upper bound, not the answer: no instruction pushes more than one
        // value, so one slot per instruction cannot be too small. The verifier
        // replaces it with the measured high water once lowering is done.
        self.max_stack = self.max_stack.saturating_add(1);
        self.code.push(op);
        self.positions.push(Position::NONE);
    }

    /// Emit an instruction that can fail against a place in the source.
    ///
    /// The position goes to the side table rather than into the instruction, so
    /// it can be stripped from an artifact without touching the code.
    fn emit_at(&mut self, op: Op, pos: Position) {
        self.emit(op);
        *self.positions.last_mut().expect("just emitted") = pos;
    }
}

/// Does this expression lower to exactly one push that cannot fail?
///
/// Such a value can be evaluated later than Rhai evaluates it without anything
/// being able to tell: it reads no state, writes none, and raises nothing — so
/// neither the answer nor which of two failures is reported can move.
///
/// Deliberately narrower than [`Expr::is_constant`], which also admits a
/// constant array or map. Building one of those is what the data-size limits
/// refuse, and a chain that would raise on both its operands and its value has
/// to raise on the same one Rhai does.
fn is_one_pushed_value(expr: &Expr) -> bool {
    match expr {
        #[cfg(not(feature = "no_float"))]
        Expr::FloatConstant(..) => true,
        Expr::BoolConstant(..)
        | Expr::IntegerConstant(..)
        | Expr::CharConstant(..)
        | Expr::StringConstant(..)
        | Expr::Unit(..) => true,
        _ => false,
    }
}

/// The slot naming the instruction an op jumps to, for the ops that have one.
///
/// One list rather than two: a jump this does not know about would neither be
/// patchable nor keep [`Lowering::patched_max`] honest, and the second failure
/// is silent.
fn jump_target_mut(op: &mut Op) -> Option<&mut u32> {
    match op {
        Op::Jump(target)
        | Op::JumpIfFalse { target, .. }
        | Op::JumpIfTrue { target, .. }
        | Op::SkipIfNotUnit { target, .. }
        | Op::IterNext { exit: target, .. }
        | Op::IterNextStore { exit: target, .. }
        | Op::PushHandler { target, .. }
        // An operator that is also the branch reading it is patched and
        // threaded as the branch it swallowed would have been.
        | Op::BinOp {
            branch: Some(target),
            ..
        }
        | Op::BinOpFrom {
            branch: Some(target),
            ..
        } => Some(target),
        _ => None,
    }
}

/// One step, still as AST.
/// A step, and where Rhai would blame it.
///
/// The position travels with the step rather than being taken from the chain:
/// Rhai reports each kind against its own node, and one chain instruction has
/// only one position-table entry between all of them.
enum ChainStep<'a> {
    /// The index expression, and the `[` it sits behind — see [`Step::Index`].
    Index(&'a Expr, rhai::Position, crate::grain::bytecode::StepFlags),
    Property(
        &'a (
            (ImmutableString, u64),
            (ImmutableString, u64),
            ImmutableString,
        ),
        rhai::Position,
        crate::grain::bytecode::StepFlags,
    ),
    Method(
        &'a FnCallExpr,
        rhai::Position,
        crate::grain::bytecode::StepFlags,
    ),
}

/// Unpick Rhai's nested chain encoding into a root and a list of steps.
///
/// `a.b[i]` is `Dot { lhs: a, rhs: Index { lhs: b, rhs: i } }`: each nested
/// node's `lhs` is the current step's operand and its `rhs` is the
/// continuation, so the list is built by walking `rhs` and taking `lhs` at each
/// level. The innermost `rhs` is the last step rather than a continuation,
/// which is what ends the walk.
///
/// `ASTFlags::BREAK` is what ends it, and it carries real information:
/// `a[b[0]]` and `a[b][0]` have the same shape, and the flag is the only thing
/// that says the first one's `b[0]` is an index expression rather than two
/// steps (`eval/chaining.rs:698`).
///
/// Returns `None` for a dot onto anything but a property or a method.
fn flatten_chain<'a>(
    lowering: &mut Lowering,
    expr: &'a Expr,
) -> Option<(&'a Expr, Vec<ChainStep<'a>>)> {
    /// A chain node's parts: operand side, continuation side, and whether the
    /// step it introduces is a property rather than an index.
    fn parts<'a>(
        lowering: &mut Lowering,
        expr: &'a Expr,
    ) -> Option<(&'a Expr, &'a Expr, ASTFlags, bool)> {
        match expr {
            Expr::Dot(binary, flags, ..) => {
                lowering.caps.insert(Caps::METHOD);
                Some((&binary.lhs, &binary.rhs, *flags, true))
            }
            Expr::Index(binary, flags, ..) => {
                lowering.caps.insert(Caps::INDEXING);
                Some((&binary.lhs, &binary.rhs, *flags, false))
            }
            _ => None,
        }
    }

    let (root, mut rest, mut flags, mut dotted) = parts(lowering, expr)?;
    let mut steps = Vec::new();
    // Rhai's `op_pos`, which is the position of the chain node the step is
    // being taken *inside* rather than of the step's operand, and which walks
    // down with the recursion (`eval/chaining.rs:695`).
    let mut bracket = expr.position();

    loop {
        let mut step_flags = StepFlags::default();

        if flags.contains(ASTFlags::NEGATED) {
            step_flags.insert(StepFlags::SKIP_IF_UNIT);
        }

        // `rest` is the continuation only when it is a chain node *and* this
        // node is not marked as the last one. Otherwise it is this step's own
        // operand — the index expression, or the property being read.
        let next = (!flags.contains(ASTFlags::BREAK))
            .then(|| parts(lowering, rest))
            .flatten();

        let (operand, following) = match next {
            Some((operand, _, _, _)) => (operand, Some(rest)),
            None => (rest, None),
        };

        steps.push(match (dotted, operand) {
            (true, Expr::Property(prop, pos)) => {
                lowering.caps.insert(Caps::PROPERTY);
                ChainStep::Property(prop, *pos, step_flags)
            }
            (true, Expr::MethodCall(call, pos)) => {
                lowering.caps.insert(Caps::METHOD);
                ChainStep::Method(call, *pos, step_flags)
            }
            // `a.(expr)` is not syntax, so a dot onto anything else is a shape
            // the parser only makes for something handled elsewhere.
            (true, _) => return None,
            (false, index) => {
                lowering.caps.insert(Caps::INDEXING);
                ChainStep::Index(index, bracket, step_flags)
            }
        });

        match following {
            Some(node) => {
                let (_, next_rest, next_flags, next_dotted) =
                    parts(lowering, node).expect("checked by `next`");
                rest = next_rest;
                flags = next_flags;
                dotted = next_dotted;
                bracket = node.position();
            }
            None => break,
        }
    }

    Some((root, steps))
}

/// Wrap statements as a block expression.
///
/// `Expr::Stmt` is the one shape `eval_expression_tree_raw` routes to
/// `eval_stmt_block` rather than `eval_expr`, which is what lets statements go
/// back through the walker at all.
fn wrap_statements(statements: Vec<Stmt>) -> Expr {
    let span = statements.first().zip(statements.last()).map_or_else(
        || Span::new(Position::NONE, Position::NONE),
        // `crate::types`, not `crate::types::position`: `no_position` swaps the
        // module out for a zero-sized one and re-exports `Span` from whichever
        // is in play.
        |(first, last)| crate::types::Span::new(first.position(), last.position()),
    );

    Expr::Stmt(Box::new(StmtBlock::new_with_span(statements, span)))
}

/// What orders one script function against another when lowering.
///
/// Everything that tells two declarations apart, nothing that varies between
/// runs. Rhai refuses a duplicate name, arity and receiver, so this is total.
#[cfg(not(feature = "no_function"))]
fn declaration_order(def: &ScriptFuncDef) -> (&str, usize, Option<&str>) {
    #[cfg(not(feature = "no_object"))]
    let this_type = def.this_type.as_deref();
    #[cfg(feature = "no_object")]
    let this_type = None;

    (&def.name, def.params.len(), this_type)
}

/// The slot [`Op::IndexSet`] or [`Op::IndexGet`] would name, for a chain of
/// exactly their shape.
///
/// Every condition is a thing the instruction then does not have to carry or
/// re-derive: one `[i]` step and nothing after it, an index operand at the
/// bottom of the chain's operand window, no null-conditional short circuit,
/// and a root that is a slot — the only root a write can land in without a
/// write-back, and the only one a read reaches without evaluating anything.
///
/// An op-assignment is refused. `a[i] += 1` reads, applies and writes back
/// through one `Target`, and neither instruction here is built to hold the
/// element open across the operator.
fn indexed_slot(chain: &Chain) -> Option<u16> {
    let Root::Local { slot, .. } = chain.root else {
        return None;
    };
    match chain.tail {
        Tail::Read | Tail::Assign { op: None } => {}
        Tail::Assign { op: Some(..) } => return None,
    }
    let [Step::Index {
        operand: 0, flags, ..
    }] = chain.steps[..]
    else {
        return None;
    };
    flags.is_empty().then_some(slot)
}

#[cfg(test)]
#[cfg(not(feature = "no_function"))]
mod tests {
    use super::*;
    use crate::grain::bytecode::StepFlags;

    /// Lowering order fixes every address inside a function, so it has to come
    /// from the source rather than from a hash map.
    ///
    /// Checks the order itself rather than comparing two artifacts: the seed is
    /// per process, so two compiles in one process agree either way.
    #[test]
    fn functions_are_lowered_in_a_stable_order() {
        let engine = crate::Engine::new();
        let ast = engine
            .compile(
                "fn zulu(x) { x + 1 }
                 fn alpha(a, b) { a + b }
                 fn alpha(a) { a }
                 fn mike() { 1 }
                 zulu(1) + alpha(2, 3) + alpha(4) + mike()",
            )
            .expect("must compile");
        let program = Compiler::new().compile(&ast);

        let order: Vec<_> = program
            .functions()
            .iter()
            .map(|f| {
                (
                    program.name(f.name).expect("a compiled function is named"),
                    f.params.len(),
                )
            })
            .collect();

        assert_eq!(
            order,
            [("alpha", 1), ("alpha", 2), ("mike", 0), ("zulu", 1)],
            "functions must be lowered by name and arity, not by hash",
        );
    }

    /// What each benchmark source actually compiles to.
    ///
    /// `residual_count` falling is this module's stated progress metric, and a
    /// program with any residual makes `run_frame` set `always_search_scope`.
    /// The op histogram is the other half: these programs are a loop and little
    /// else, so the ops below are very nearly the ops executed per iteration,
    /// and both numbers are deterministic — they can be read on a busy machine
    /// where a timing cannot.
    #[test]
    fn benchmark_sources_shape_census() {
        // The `grain_bench` cases, verbatim, and the op count each one's
        // hot body lowers to. A ceiling rather than an equality: lowering a
        // body to fewer instructions is the point, and only raising one of
        // these numbers should have to be argued for.
        const SOURCES: &[(&str, usize, &str)] = &[
            ("tight integer loop", 4, "let s = 0; let i = 0; while i < 20000 { s += i; i += 1; } s"),
            ("float arithmetic", 7, "let x = 0.0; let i = 0; while i < 20000 { x += (i.to_float() * 1.5) / 2.5; i += 1; } x"),
            ("script fn calls", 5, "fn add(a, b) { a + b } let s = 0; for i in 0..5000 { s = add(s, i); } s"),
            ("recursive fibonacci", 13, "fn fib(n) { if n < 2 { n } else { fib(n-1) + fib(n-2) }} fib(28)"),
            (
                "switch, 4 arms",
                11,
                "let s = 0; for i in 0..20000 { \
                 switch i % 4 { 0 => s += 1, 1 => s += 2, 2 => s += 3, _ => s += 4 } \
                 } s",
            ),
            (
                "switch, 16 arms",
                35,
                "let s = 0; for i in 0..20000 { \
                 switch i % 16 { \
                 0 => s += 1, 1 => s += 2, 2 => s += 3, 3 => s += 4, \
                 4 => s += 5, 5 => s += 6, 6 => s += 7, 7 => s += 8, \
                 8 => s += 9, 9 => s += 10, 10 => s += 11, 11 => s += 12, \
                 12 => s += 13, 13 => s += 14, 14 => s += 15, _ => s += 16 } \
                 } s",
            ),
            ("branch heavy", 11, "let s = 0; for i in 0..20000 { if i % 3 == 0 { s += 1; } else if i % 3 == 1 { s += 2; } else { s -= 1; } } s"),
            ("native function calls", 7, "let a = 42; for i in 0..20000 { a = abs(abs(abs(abs(a)))); } a"),
            (
                "native callbacks",
                4,
                "let a = []; for i in 0..500 { a.push(i); } \
                 let b = a.map(|x| x * 2); b.filter(|x| x % 3 == 0).len",
            ),
            (
                "primes",
                4,
                r#"
            const SIZE = 1_000_000;

            let prime_mask = [];
            prime_mask.pad(SIZE + 1, true);

            prime_mask[0] = false;
            prime_mask[1] = false;

            let total_primes_found = 0;

            for p in 2..=SIZE {
                if !prime_mask[p] { continue; }

                total_primes_found += 1;

                for i in range(2 * p, SIZE + 1, p) {
                    prime_mask[i] = false;
                }
            }

            total_primes_found
        "#,
            ),
        ];

        let engine = crate::Engine::new();
        for (name, ceiling, source) in SOURCES {
            let ast = engine.compile(source).expect("must compile");
            let program = Compiler::new().compile(&ast);
            program
                .verify()
                .unwrap_or_else(|err| panic!("{name} does not verify: {err:?}"));
            let decoded: Vec<_> =
                crate::grain::bytecode::code::disassemble(program.code()).collect();
            // A back edge is a `Jump` to an address at or below its own. The
            // highest address any of them names is the innermost loop's
            // header, and the body that runs every iteration reaches from
            // there to the last back edge naming it. A loop has several: a
            // `continue` is one, and `Lowering::thread_jumps` turns each arm's
            // jump into another, so taking the first cuts the body off at the
            // first arm.
            let edges: Vec<(usize, usize)> = decoded
                .iter()
                .filter_map(|(at, op)| match op {
                    Op::Jump(target) if (*target as usize) <= *at => Some((*target as usize, *at)),
                    _ => None,
                })
                .collect();
            let body = edges
                .iter()
                .map(|(header, ..)| *header)
                .max()
                .map(|header| {
                    let last = edges
                        .iter()
                        .filter(|(naming, ..)| *naming == header)
                        .map(|(.., at)| *at)
                        .max()
                        .expect("the header came from one of these");
                    (header, last)
                });
            let mut histogram: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            let mut total = 0;
            for (at, op) in &decoded {
                if let Some((from, to)) = body {
                    if *at < from || *at > to {
                        continue;
                    }
                }
                let rendered = format!("{op:?}");
                let head = rendered
                    .split(|c: char| c == '(' || c == ' ' || c == '{')
                    .next()
                    .unwrap_or("?")
                    .to_string();
                *histogram.entry(head).or_default() += 1;
                total += 1;
            }
            let mut ranked: Vec<_> = histogram.into_iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let top: Vec<String> = ranked
                .iter()
                .take(8)
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            // How much pushing-to-pop is left anywhere in the chunk, hot or
            // not: `Unit` and `Pop` are what a statement whose value nothing
            // reads costs when the discard could not be carried into it.
            let discards = decoded
                .iter()
                .filter(|(.., op)| matches!(op, Op::Unit | Op::Pop))
                .count();
            println!(
                "[shape] {name}: residuals={} body_ops={total} whole={} discards={discards} | {}",
                program.residual_count(),
                decoded.len(),
                top.join(" "),
            );
            assert!(
                total <= *ceiling,
                "{name} lowers its body to {total} instructions, up from {ceiling}"
            );
        }
    }

    /// The guard's operator carries the branch that reads it, and the one
    /// edge that must not be folded is not.
    #[test]
    fn a_guards_operator_carries_its_branch() {
        let engine = crate::Engine::new();
        let lowered = |source: &str| -> Vec<Op> {
            let ast = engine.compile(source).expect("must compile");
            let program = Compiler::new().compile(&ast);
            crate::grain::bytecode::code::disassemble(program.code())
                .map(|(.., op)| op)
                .collect()
        };

        let branched = |ops: &[Op]| {
            ops.iter()
                .filter(|op| {
                    matches!(
                        op,
                        Op::BinOp {
                            branch: Some(..),
                            ..
                        } | Op::BinOpFrom {
                            branch: Some(..),
                            ..
                        }
                    )
                })
                .count()
        };
        let tests = |ops: &[Op]| {
            ops.iter()
                .filter(|op| matches!(op, Op::JumpIfFalse { .. }))
                .count()
        };

        for source in [
            "let a = 1; let b = 2; if a < b { 1 } else { 2 }",
            "let a = 1; if a < 2 { 1 } else { 2 }",
            "let a = 1; let b = 2; if a + 1 < b { 1 } else { 2 }",
            "let a = 1; let b = 2; if a < b + 1 { 1 } else { 2 }",
            "let i = 0; while i < 3 { i += 1; } i",
            "let i = 0; do { i += 1; } until i >= 3; i",
        ] {
            let ops = lowered(source);
            assert_eq!(branched(&ops), 1, "`{source}` must fold its one branch");
            assert_eq!(
                tests(&ops),
                0,
                "`{source}` must be left with no separate test"
            );
        }

        // The edge that skips a non-unit `??` operand is patched to exactly
        // where the branch goes, so the operator it ends with keeps its result
        // and the branch stays an instruction of its own.
        let ops = lowered("let a = (); let b = 1; if a ?? (b < 2) { 1 } else { 2 }");
        assert_eq!(branched(&ops), 0, "a `??` operand's operator must not fold");
        assert_eq!(tests(&ops), 1, "the branch it feeds must survive");
    }

    #[test]
    #[cfg(not(feature = "no_object"))]
    fn null_conditional_steps_are_lowered_into_chains() {
        let engine = crate::Engine::new();
        let ast = engine
            .compile("let m = #{a: #{b: 1}}; m?.a?.b")
            .expect("must compile");
        let program = Compiler::new().compile(&ast);

        let chain = program
            .chains()
            .iter()
            .find(|chain| !chain.steps.is_empty())
            .expect("the null-conditional expression must lower into a chain");

        assert!(
            chain.steps.iter().all(|step| match step {
                Step::Index { flags, .. }
                | Step::Property { flags, .. }
                | Step::Method { flags, .. } => flags.contains(StepFlags::SKIP_IF_UNIT),
            }),
            "all steps in `m?.a?.b` must short-circuit on unit",
        );
    }
}
