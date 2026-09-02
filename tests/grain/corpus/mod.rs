//! Scripts the VM must agree with Rhai on.
//!
//! Weighted towards the places a bytecode VM is most likely to drift from a
//! tree walker rather than towards breadth: scope discipline, the error-based
//! unwinding Rhai uses for `return`/`break`/`throw`, and the lvalue forms that
//! cannot be expressed as a plain `&mut` and so need explicit write-back.
//!
//! Everything here currently runs as a single `EvalAst` residual, so passing is
//! expected. That is the point — it pins the baseline, and it exercises the
//! runtime-state setup (function library, module resolver, source name,
//! `return`/`exit` mapping) which is real code that can be wrong today.

use rhai::INT;

// Only `tests/fuzz.rs` and the `grain_generated` fuzz target use this;
// the other harnesses take the module for its cases.
#[allow(dead_code)]
pub mod generate;

pub struct Case {
    pub name: &'static str,
    pub source: &'static str,
}

/// A host type with a getter, a setter, an indexer and both kinds of method.
///
/// Registered so the corpus can reach the one part of the chain walker that is
/// conservative rather than exact. Arrays and maps hand out references, so a
/// mutation partway down a chain lands in them and nothing needs writing back.
/// A getter hands back a *value*, so `w.inner.level = 1` mutates a temporary
/// and the setter is the only way home — and Rhai decides whether to call it
/// from `func.is_method()`, which the VM cannot see and therefore approximates.
/// Without a host type in the engine, nothing here is ever exercised.
/// Held as `INT` rather than `i64` throughout: `only_i32` narrows the script
/// integer, and a host type registered against the wider one would take a type
/// no script under that build can produce.
#[derive(Debug, Clone, Default)]
pub struct Widget {
    pub level: INT,
    pub cells: Vec<INT>,
}

/// A `Widget` behind a getter that hands back a *value*, which is the whole
/// point of it — and so only exists where a property does.
#[cfg(not(feature = "no_object"))]
#[derive(Debug, Clone, Default)]
pub struct Holder {
    pub inner: Widget,
}

fn out_of_range(index: INT, len: usize) -> Box<rhai::EvalAltResult> {
    Box::new(rhai::EvalAltResult::ErrorArrayBounds(len, index, rhai::Position::NONE))
}

/// The engine both sides of the differential run against.
pub fn engine() -> rhai::Engine {
    let mut engine = rhai::Engine::new();

    engine
        .register_type_with_name::<Widget>("Widget")
        .register_fn("widget", |level: INT| Widget { level, cells: vec![10, 20, 30] })
        // Takes the receiver by reference, so Rhai counts it as a method and
        // writes a temporary back afterwards.
        .register_fn("bump", |w: &mut Widget| w.level += 1)
        // Reads only. Whether Rhai still writes back after one of these is
        // exactly the question this corpus is here to settle.
        .register_fn("doubled", |w: &mut Widget| w.level * 2)
        // Mutates and *then* fails. Rhai reaches a chain root through a live
        // reference, so the mutation has already landed by the time the error
        // propagates; nothing else here can tell a write-back that happens from
        // one that is skipped because the walk raised.
        .register_fn("bump_then_fail", |w: &mut Widget| -> Result<(), Box<rhai::EvalAltResult>> {
            w.level += 1;
            Err("bump_then_fail".into())
        });

    // A host handing back a value that is already shared, which is the only way
    // a script can be given one: sharing is otherwise something Rhai does to a
    // variable a closure captured, and loading such a variable flattens it on
    // the way out. Whether the sharing survives crossing into a `let` is a
    // difference nothing else here can see.
    #[cfg(not(feature = "no_closure"))]
    {
        let cell = rhai::Dynamic::from(42 as INT).into_shared();
        engine.register_fn("shared_cell", move || cell.clone());
    }

    // A property is reached with `.`, which `no_object` removes; an indexer
    // with `[..]`, which `no_index` does. Registered apart from the chain
    // above so each can go with the feature that takes its syntax away.
    #[cfg(not(feature = "no_object"))]
    engine.register_get_set("level", |w: &mut Widget| w.level, |w: &mut Widget, v: INT| w.level = v);
    // Returning an error rather than panicking, because a panic in a
    // registered function takes the test process with it.
    #[cfg(not(all(feature = "no_index", feature = "no_object")))]
    engine.register_indexer_get_set(
        |w: &mut Widget, i: INT| -> Result<INT, Box<rhai::EvalAltResult>> { w.cells.get(i as usize).copied().ok_or_else(|| out_of_range(i, w.cells.len())) },
        |w: &mut Widget, i: INT, v: INT| -> Result<(), Box<rhai::EvalAltResult>> {
            let len = w.cells.len();
            *w.cells.get_mut(i as usize).ok_or_else(|| out_of_range(i, len))? = v;
            Ok(())
        },
    );
    #[cfg(not(all(feature = "no_index", feature = "no_object")))]
    engine.register_indexer_get_set(
        |w: &mut Widget, name: &str| -> Result<INT, Box<rhai::EvalAltResult>> {
            let index = name.len() as INT - 1;
            w.cells.get(index as usize).copied().ok_or_else(|| out_of_range(index, w.cells.len()))
        },
        |w: &mut Widget, name: &str, v: INT| -> Result<(), Box<rhai::EvalAltResult>> {
            let index = name.len() as INT - 1;
            let len = w.cells.len();
            *w.cells.get_mut(index as usize).ok_or_else(|| out_of_range(index, len))? = v;
            Ok(())
        },
    );

    #[cfg(not(feature = "no_object"))]
    {
        engine
            .register_type_with_name::<Holder>("Holder")
            .register_fn("holder", |level: INT| Holder { inner: Widget { level, cells: vec![1, 2, 3] } })
            .register_get_set("inner", |h: &mut Holder| h.inner.clone(), |h: &mut Holder, w: Widget| h.inner = w);

        // The same `Widget`, behind an *indexer* rather than a property. The
        // getter above is reached with the property name as a static operand;
        // this one is reached with an index the walk has to evaluate, and a
        // native's by-value parameter is bound by `take` — so the index a
        // write-back needs is gone by the time the setter wants it.
        #[cfg(not(feature = "no_index"))]
        engine.register_indexer_get_set(|h: &mut Holder, _: INT| h.inner.clone(), |h: &mut Holder, _: INT, w: Widget| h.inner = w);
    }

    engine
}

const fn case(name: &'static str, source: &'static str) -> Case {
    Case { name, source }
}

/// Whether a corpus case exercises anything on this build.
///
/// A restriction feature removes the syntax outright — Rhai will not parse a
/// capturing closure under `no_closure`, or a float literal under `no_float` —
/// so the case tests nothing here, and both sides agreeing on the parse failure
/// would be an empty agreement rather than a passing one.
///
/// Lives here rather than in one harness because every harness that walks
/// [`CASES`] needs the same answer.
#[must_use]
pub fn applies_to_this_build(name: &str) -> bool {
    #[cfg(feature = "no_closure")]
    if name.starts_with("closure_") || name.starts_with("is_shared") || name == "unary_not_shared" {
        return false;
    }
    #[cfg(feature = "no_module")]
    if name.starts_with("import_") || name.starts_with("export_") {
        return false;
    }
    // `unchecked` removes the arithmetic guards, so `1 / 0` panics inside
    // Rhai's own built-in rather than raising — there is no behaviour left for
    // the two sides to agree on, and the case would take the process with it.
    #[cfg(feature = "unchecked")]
    if matches!(
        name,
        "error_divide_by_zero"
            | "error_guard_operator_fails"
            | "error_temp_root_index_runs_first"
            // The guards are what these cases are, and `unchecked` removes
            // them: the operator overflows or shifts past the width in Rust
            // rather than raising, which is a panic in a test build and not a
            // behaviour two sides could agree on.
            | "error_int_add_overflow"
            | "error_int_multiply_overflow"
            | "error_int_modulo_by_zero"
            | "error_int_power_negative"
            | "error_op_assign_int_power_negative"
            // Not a guard but the same problem: `1 << 100` is a Rust shift
            // past the width, which panics in a test build. The two operator
            // censuses beside it stay in — every operand they use is in range,
            // so `unchecked` runs them, and the arm they cover there is the
            // one spelled out by hand rather than delegated.
            | "int_shift_edges"
    ) {
        return false;
    }
    // No shared prefix to key on: a float literal is incidental to most of
    // these, which are about interpolation, ranges and operator errors.
    #[cfg(feature = "no_float")]
    if matches!(
        name,
        "float_arithmetic"
            | "mixed_numeric"
            | "interpolation_of_every_type"
            | "switch_float_in_range"
            | "error_operator_undefined_for_types"
            | "error_op_assign_undefined_for_types"
            | "float_op_assign_every_form"
            | "float_int_mixed_operators"
            | "float_int_mixed_op_assign"
            | "int_float_mixed_op_assign"
            | "int_float_comparisons"
            | "error_index_assign_float_index"
            | "error_index_read_float_index"
    ) {
        return false;
    }
    // No `decimal` does not have support for `Decimal`.
    #[cfg(not(feature = "decimal"))]
    if matches!(name, "decimal_numbers" | "decimal_arithmetic") {
        return false;
    }
    // `no_function` removes `fn` and the anonymous form with it, so a case that
    // declares one, points at one, or has a `this` to be a method of does not
    // parse. The prefixes carry the families; the rest reach for a function
    // incidentally, as the subject of a `switch`, a `for` or a `try`.
    #[cfg(feature = "no_function")]
    if name.starts_with("this_")
        || name.starts_with("error_this_")
        || name.starts_with("fn_")
        || name.starts_with("call_style_")
        || matches!(
            name,
            "block_as_argument"
                | "closure_captures_a_callees_local"
                | "closure_outlives_later_calls_to_its_maker"
                | "closure_writes_a_callees_local"
                | "error_a_skipped_function_cannot_see_the_caller"
                | "error_op_assign_to_a_constant_parameter_from_a_local"
                | "error_wrong_arity"
                | "for_over_captured_array"
                | "for_return_from_body"
                | "fn_call_captures_parent_scope"
                | "guard_operator_on_a_shared_operand"
                | "is_def_fn"
                | "map_computed_order"
                | "map_read_of_absent_key_is_not_visible_to_a_closure"
                | "resolution_arity_decides"
                | "resolution_inside_a_script_function"
                | "resolution_script_function_shadows_a_native"
                | "switch_on_a_shared_subject_matches"
                | "switch_range_on_a_shared_subject_matches"
                | "temp_root_call"
                | "throw_from_a_function_leaves_the_caller_top_level_alone"
                | "throw_in_fn"
                | "try_around_a_compiled_call"
                | "try_catch_does_not_swallow_return"
                | "try_does_not_catch_return"
                | "type_of_a_pointer"
                | "if_stmt_arm_returns"
                | "index_assign_is_a_function_bodys_value"
                | "index_assign_named_value_shared_root"
                | "index_assign_through_a_shared_index"
                | "index_read_of_a_shared_element"
                | "index_read_through_a_shared_index"
                | "index_read_through_a_shared_root"
        )
    {
        return false;
    }
    // `no_index` removes the `[..]` operator outright — array and blob
    // literals, indexing, slicing a string, and every native that hands one
    // back or takes one. There is no prefix that separates them: an array is
    // incidental to most of these, which are about receivers, temporaries and
    // chain roots.
    #[cfg(feature = "no_index")]
    if matches!(
        name,
        "array_literal"
            | "array_methods"
            | "bitfield_assign"
            | "call_style_argument_replaces_the_receiver"
            | "call_style_argument_writes_the_receiver"
            | "call_style_constant_receiver"
            | "call_style_mutating_native"
            | "call_style_pure_native"
            | "call_style_receiver_is_also_an_argument"
            | "call_style_receiver_twice_over"
            | "call_style_shared_receiver"
            | "chain_in_statement_position_at_a_jump_target"
            | "chain_index_coalesce"
            | "chain_method_in_statement_position"
            | "closure_call_mutates_an_array"
            | "closure_filter_binds_this"
            | "closure_for_each_binds_this"
            | "closure_in_filter"
            | "closure_in_map"
            | "closure_made_inside_a_callback"
            | "closure_map_binds_this"
            | "closure_map_body_with_a_local"
            | "closure_map_nested"
            | "closure_map_raises_partway"
            | "closure_map_repeated"
            | "closure_map_takes_an_argument"
            | "closure_map_then_filter"
            | "closure_shared_chain_root"
            | "const_root_index_read"
            | "empty_literals_nested_in_computed_ones"
            | "empty_map_nested_in_a_computed_map"
            | "error_array_bounds"
            | "error_const_root_method_step"
            | "error_host_index_bounds"
            | "error_index_assign_constant_is_not_an_integer"
            | "error_index_assign_float_index"
            | "error_index_assign_named_value_out_of_bounds"
            | "error_index_assign_out_of_bounds"
            | "error_index_into_an_unindexable_step"
            | "error_index_into_an_unindexable_step_deep"
            | "error_index_read_constant_is_not_an_integer"
            | "error_index_read_float_index"
            | "error_index_read_out_of_bounds"
            | "error_index_read_slot_is_not_an_integer"
            | "error_no_function_for_the_receiver"
            | "error_property_on_a_temporary"
            | "error_temp_root_index_runs_first"
            | "error_temp_root_out_of_bounds"
            | "error_type_mismatch"
            | "for_array"
            | "for_over_captured_array"
            | "for_return_from_body"
            | "for_with_counter"
            | "host_index_get"
            | "host_index_set"
            | "host_index_temp_set"
            | "host_mutation_before_a_failure_survives_in_an_array"
            | "host_string_index_property_get_fallback"
            | "host_string_index_property_op_assign_fallback"
            | "host_string_index_property_set_fallback"
            | "host_temp_index_set"
            | "host_temp_string_index_property_set_fallback"
            | "index_assign_array"
            | "index_assign_call_index"
            | "index_assign_computed_index"
            | "index_assign_float_index"
            | "index_assign_is_a_function_bodys_value"
            | "index_assign_is_the_scripts_value"
            | "index_assign_local_index"
            | "index_assign_map_root"
            | "index_assign_named_bool"
            | "index_assign_named_const_const"
            | "index_assign_named_local_const"
            | "index_assign_named_value_map_root"
            | "index_assign_named_value_negative"
            | "index_assign_named_value_shared_root"
            | "index_assign_negative"
            | "index_assign_stashed_local_value"
            | "index_assign_stashed_value_const_index"
            | "index_assign_nested"
            | "index_assign_through_a_shared_index"
            | "index_expression_reads_the_root"
            | "index_read_bitfield_root"
            | "index_read_blob_root"
            | "index_read_call_index"
            | "index_read_computed_index"
            | "index_read_feeds_a_write_back"
            | "index_read_local_array"
            | "index_read_local_array_const"
            | "index_read_local_index"
            | "index_read_map_root"
            | "index_read_negative"
            | "index_read_of_a_shared_element"
            | "index_read_string_root"
            | "index_read_through_a_shared_index"
            | "index_read_through_a_shared_root"
            | "interpolation_of_containers"
            | "is_shared_after_capture"
            | "map_computed_in_array"
            | "map_read_absent_through_a_chain"
            | "map_read_of_absent_key_does_not_create_it"
            | "map_read_of_absent_key_is_not_visible_to_a_closure"
            | "nested_containers"
            | "op_assign_indexed"
            | "string_char_assign"
            | "string_slice_inclusive"
            | "string_slice_read"
            | "temp_root_array_index"
            | "temp_root_array_method"
            | "temp_root_call"
            | "temp_root_mutating_method"
            | "temp_root_nested"
            | "this_as_first_argument"
            | "this_as_first_argument_pure"
            | "this_index"
            | "this_index_assign"
            | "this_method_arity"
            | "this_method_step"
            | "this_survives_a_failed_chain"
            | "try_catch_native_error"
            | "type_of_a_container"
            | "unary_not_guard_on_an_indexed_read"
    ) {
        return false;
    }
    // `no_object` removes the `.` operator, so a map literal, a property, a
    // method call and everything reached through one all go with it — which is
    // most of what a chain is for.
    #[cfg(feature = "no_object")]
    if name.starts_with("this_")
        || name.starts_with("host_")
        || name.starts_with("map_")
        || name.starts_with("temp_root_")
        || matches!(
            name,
            "array_methods"
                | "call_style_mutating_host_type"
                | "call_style_shared_receiver"
                | "chain_in_statement_position_at_a_jump_target"
                | "chain_method_in_statement_position"
                | "chain_property_coalesce"
                | "chain_property_in_statement_position"
                | "char_ops"
                | "closure_call_mutates_an_array"
                | "closure_call_on_a_local_inline"
                | "closure_call_on_a_local_reads"
                | "closure_call_on_a_local_writes_back"
                | "closure_call_on_a_temporary"
                | "closure_call_on_this"
                | "closure_capture_mutate"
                | "closure_capture_read"
                | "closure_captures_a_callees_local"
                | "closure_filter_binds_this"
                | "closure_for_each_binds_this"
                | "closure_in_filter"
                | "closure_in_map"
                | "closure_made_inside_a_callback"
                | "closure_map_binds_this"
                | "closure_map_body_with_a_local"
                | "closure_map_nested"
                | "closure_map_raises_partway"
                | "closure_map_repeated"
                | "closure_map_takes_an_argument"
                | "closure_map_then_filter"
                | "closure_outlives_later_calls_to_its_maker"
                | "closure_shared_op_assign"
                | "closure_shared_write"
                | "closure_writes_a_callees_local"
                | "empty_literals_nested_in_computed_ones"
                | "empty_map_nested_in_a_computed_map"
                | "error_chain_in_statement_position"
                | "error_const_root_method_step"
                | "error_fn_ptr_unknown_name"
                | "error_index_into_an_unindexable_step_deep"
                | "error_map_write_through_an_absent_key"
                | "error_method_on_a_variable"
                | "error_op_assign_undefined_for_types"
                | "error_operator_undefined_for_types"
                | "error_property_deep_in_a_chain"
                | "error_property_on_a_temporary"
                | "error_property_on_a_variable"
                | "error_this_is_not_inherited"
                | "fn_mutating_method"
                | "fn_ptr_call"
                | "fn_ptr_call_repeated"
                | "fn_ptr_curried"
                | "fn_ptr_from_dynamic_name"
                | "fn_ptr_to_native"
                | "host_string_index_property_get_fallback"
                | "host_string_index_property_set_fallback"
                | "host_string_index_property_op_assign_fallback"
                | "host_temp_string_index_property_set_fallback"
                | "index_assign_map_root"
                | "index_assign_named_value_map_root"
                | "index_assign_nested"
                | "index_expression_reads_the_root"
                | "interpolation_of_containers"
                | "nested_containers"
                | "property_assign_deep"
                | "string_ops"
                | "temp_root_array_method"
                | "try_catch_native_error"
                | "type_of_method_style"
                | "index_read_map_root"
                | "property_assign_is_the_scripts_value"
        )
    {
        return false;
    }
    let _ = name;
    true
}

pub const CASES: &[Case] = &[
    // --- values and operators -------------------------------------------
    case("int_arithmetic", "let a = 7; let b = 3; a * b - a / b + a % b"),
    case("float_arithmetic", "let a = 7.5; let b = 0.5; a * b + a / b"),
    case("mixed_numeric", "1 + 2.5"),
    case("decimal_numbers", "let a = parse_decimal(\"42\")"),
    case("decimal_arithmetic", "let a = parse_decimal(\"42\"); let b = 1; a + b"),
    case("comparison_chain", "let a = 5; a > 1 && a < 10 || a == 5"),
    case("bitwise", "let a = 0b1010; (a & 0b0110) | (a ^ 0b1111) << 2"),
    // Every integer operator that has a built-in, in both spellings. The VM
    // runs these without resolving a function, so each one is a place the two
    // sides can disagree about what the operator means.
    case("int_operator_every_form", "let a = 7; let b = 3; ((a + b) - (a - b)) * (a * b) / (a / b) % (a % b) + (a ** 2) + (a << b) + (a >> 1) + (a & b) + (a | b) + (a ^ b)"),
    case("int_op_assign_every_form", "let a = 1234; a += 7; a -= 3; a *= 5; a /= 2; a %= 97; a **= 2; a <<= 3; a >>= 1; a &= 0xff; a |= 0x30; a ^= 0x0f; a"),
    // `x op= y` with `y` a local is one instruction, so both ends of it need a
    // shared cell to be run over: writing into one rather than replacing it,
    // and reading one out flattened rather than aliasing it.
    case("closure_op_assign_through_a_shared_target", "let t = 1; let u = 2; { let g = || t; } t += u; t"),
    case("closure_op_assign_from_a_shared_source", "let t = 1; let u = 2; { let g = || u; } t += u; t"),
    // The target's access mode is checked before the value is stored and after
    // the source is read. Assigning to a `const` by name is a parse error, so
    // the run-time check is reached through a parameter Rhai passes as a
    // constant instead.
    case("error_op_assign_to_a_constant_parameter_from_a_local", "fn bump(p) { let d = 1; p += d; p } const K = 1; bump(K)"),
    case("int_comparison_every_form", "let a = 3; let b = 4; (a == b) == false && (a != b) && (a < b) && (a <= b) && (a > b) == false && (a >= b) == false"),
    // Shifts by a negative and by more than the width, which the checked
    // built-in defines rather than leaving to the hardware.
    case("int_shift_edges", "let a = 1; let b = -3; (a << b) + (a >> b) + (a << 100) + (a >> 100)"),
    // The float arms, and the mixed pairs the float rules cover but the
    // op-assignment table does not: `f += 1` is a built-in and `i += 1.5` is
    // not, so the second falls back to `i = i + 1.5` and changes the local's
    // type. A typed opcode that treated the two alike would be wrong here and
    // nowhere else.
    case("float_op_assign_every_form", "let f = 1.5; f += 2.0; f -= 0.25; f *= 3.0; f /= 1.5; f %= 2.0; f **= 2.0; f"),
    case("float_int_mixed_operators", "let f = 1.5; let i = 2; (f + i) * (i + f) - (f - i) / (i - f)"),
    case("float_int_mixed_op_assign", "let f = 1.5; f += 2; f *= 3; f -= 1; f"),
    case("int_float_mixed_op_assign", "let i = 1; i += 1.5; i"),
    case("int_float_comparisons", "let i = 1; let f = 1.0; (i == f) && (f >= i) && (i < 2.0) && (2.5 > i)"),
    case("string_ops", r#"let s = "hello"; s + " " + "world" + s.len"#),
    case("string_interpolation", r#"let n = 42; `answer is ${n} and ${n * 2}`"#),
    // Every segment type goes through a different arm of Rhai's rendering:
    // strings skip dispatch entirely, unit renders empty, and a container
    // gets its debug-ish form.
    case("interpolation_of_every_type", r#"let s = "x"; let n = 1; let f = 1.5; let b = true; let u = (); `${s}|${n}|${f}|${b}|${u}|`"#),
    case("interpolation_of_containers", r#"let a = [1, 2]; let m = #{ k: 1 }; `${a}|${m}`"#),
    // A host type with no `to_string` registered falls back to the mapped
    // type name rather than to `Debug`.
    case("interpolation_of_host_type", r#"let w = widget(3); `w=${w}`"#),
    case("char_ops", r#"let c = 'a'; c.to_upper()"#),
    case("unit_value", "()"),
    // --- containers -------------------------------------------------------
    case("array_literal", "let a = [1, 2, 3]; a[0] + a[1] + a[2]"),
    case("array_methods", "let a = [3, 1, 2]; a.sort(); a"),
    case("map_literal", r#"let m = #{ a: 1, b: 2 }; m.a + m.b"#),
    // The one above is all-constant, so the optimizer folds it and the map
    // never gets built at run time. These do get built: Rhai keeps a template
    // holding every key and fills the computed ones in afterwards.
    case("map_computed_value", "let v = 7; let m = #{ a: v, b: 2 }; m.a + m.b"),
    case("map_all_computed", "let v = 7; let w = 8; #{ a: v, b: w }"),
    case("map_computed_nested", "let v = 7; #{ outer: #{ inner: v } }.outer.inner"),
    case("map_computed_in_array", "let v = 7; [#{ a: v }, #{ a: 2 }]"),
    // An empty literal inside one that is not. It contributes no size check of
    // its own, so it must not consume the enclosing literal's running total.
    case("empty_literals_nested_in_computed_ones", "let v = 7; [v, [], #{}, v]"),
    case("empty_map_nested_in_a_computed_map", "let v = 7; #{ a: v, b: #{}, c: [] }"),
    // The value is a call, so the order it runs in relative to the rest of the
    // literal is observable.
    case("map_computed_order", r#"let log = ""; fn note(s, c) { s + c } let m = #{ a: note("", "x"), b: note("", "y") }; m.a + m.b"#),
    case("nested_containers", r#"let m = #{ xs: [1, 2, #{ y: 3 }] }; m.xs[2].y"#),
    // --- unary operators --------------------------------------------------
    // `!` on a `bool` is the one unary operator the walker short-circuits
    // under fast operators, and so the only one the VM runs itself
    // (`Op::UnOp`). The rest of these are the shapes that must still reach a
    // registered function.
    case("unary_not_bool", "let b = true; !b"),
    case("unary_not_guard", "let b = false; let n = 0; if !b { n = 1; } n"),
    case("unary_not_twice", "let b = true; !!b"),
    // Not a `bool`, so the typed arm declines and the dispatch answers — with
    // an error, because no `!` is registered for an integer.
    case("error_unary_not_int", "let i = 1; !i"),
    // A shared cell holding a `bool`. The typed arm does not flatten, so this
    // is the dispatching path reaching the same answer. The capture is scoped
    // to a block: a closure left in the top-level scope renders differently on
    // the two sides, which would fail this for a reason that is not `!`.
    case("unary_not_shared", "let b = true; { let f = || b; } b = false; !b"),
    // Neither `-` nor `+` is short-circuited by the walker, so both stay
    // ordinary calls to `packages::arithmetic` and must keep answering as one.
    case("unary_neg_int", "let i = 7; -i"),
    case("unary_plus_int", "let i = 7; +i"),
    case("unary_neg_expr", "let i = 7; let j = 2; -(i * j)"),
    // --- control flow -----------------------------------------------------
    case("if_else", "let a = 5; if a > 3 { \"big\" } else { \"small\" }"),
    // A statement-position `if` is lowered for effect, so neither arm pushes
    // the value the statement after it would have popped. What the walker
    // says these are is what says that was only ever a pop of a unit: the
    // ones ending in an expression still discard a real value, and the ones
    // that leave the block early never reach the join at all.
    case("if_stmt_no_else", "let s = 0; if s == 0 { s = 1; } s"),
    case("if_stmt_no_else_untaken", "let s = 9; if s == 0 { s = 1; } s"),
    case("if_stmt_chain", "let s = 0; let i = 7; if i % 3 == 0 { s += 1; } else if i % 3 == 1 { s += 2; } else { s -= 1; } s"),
    case("if_stmt_arm_is_expression", "let s = 0; if s == 0 { 42 } else { 7 }; s"),
    case("if_stmt_arm_returns", "fn f(n) { if n > 0 { return n * 2; } n - 1 } f(3) + f(-3)"),
    case("if_stmt_arm_breaks", "let s = 0; for i in 0..10 { if i > 4 { break; } s += i; } s"),
    case("if_stmt_declares", "let s = 0; if s == 0 { let t = 5; s = t; } s"),
    case("if_stmt_nested", "let s = 0; for i in 0..6 { if i % 2 == 0 { if i > 2 { s += 10; } } else { s += 1; } } s"),
    case("if_value_position_kept", "let a = 5; let b = if a > 3 { 1 } else { 2 }; b"),
    case("bare_block_stmt", "let s = 0; { let t = 3; s = t; } s"),
    // A statement-position `switch` is lowered the same way, arm bodies
    // included, and the unit standing in for an absent `_` is not emitted.
    case("switch_stmt_all_arms", "let s = 0; for i in 0..8 { switch i % 4 { 0 => s += 1, 1 => s += 2, 2 => s += 3, _ => s += 4 } } s"),
    case("switch_stmt_no_default", "let s = 0; for i in 0..8 { switch i % 4 { 0 => s += 1, 1 => s += 2 } } s"),
    case("switch_stmt_range_arm", "let s = 0; for i in 0..8 { switch i { 3 => s += 2, 0..=2 => s += 1, _ => s += 3 } } s"),
    case("switch_stmt_guarded", "let s = 0; for i in 0..8 { switch i % 4 { 0 if i > 3 => s += 1, 0 => s += 2, _ => s += 3 } } s"),
    case("switch_value_position_kept", "let i = 2; let s = switch i { 0 => \"a\", 2 => \"c\", _ => \"z\" }; s"),
    // A guard that is an operator: the instruction that computes it is the
    // branch that reads it, so what it does when the result is not a `bool`,
    // when the operands are a pair the typed arms decline, and when the
    // operator itself fails all have to be what the pair of instructions did.
    case("guard_operator_local_and_local", "let a = 1; let b = 2; if a < b { 1 } else { 2 }"),
    case("guard_operator_local_and_constant", "let a = 1; if a < 2 { 1 } else { 2 }"),
    case("guard_operator_computed_left", "let a = 1; let b = 2; if a + 1 < b { 1 } else { 2 }"),
    case("guard_operator_computed_left_and_constant", "let a = 1; if a + 1 < 2 { 1 } else { 2 }"),
    case("guard_operator_computed_right", "let a = 1; let b = 2; if a < b + 1 { 1 } else { 2 }"),
    case("guard_operator_on_strings", r#"let a = "x"; let b = "y"; if a < b { 1 } else { 2 }"#),
    case("guard_operator_on_a_shared_operand", "let a = 1; let b = 2; let r = 0; { let f = || a; if a < b { r = 1; } } r"),
    case("guard_operator_two_comparisons", "let a = 1; let b = 2; if a < b && b < 3 { 1 } else { 2 }"),
    case("error_guard_operator_result_is_not_a_bool", "let i = 1; if i + 1 { 1 } else { 2 }"),
    case("error_guard_operator_fails", "let i = 1; if i / 0 { 1 } else { 2 }"),
    case("error_and_operand_operator_result_is_not_a_bool", "let i = 1; if i + 1 && true { 1 } else { 2 }"),
    case("error_while_guard_operator_result_is_not_a_bool", "let i = 0; while i + 1 { i += 1; } i"),
    case("error_do_until_guard_operator_result_is_not_a_bool", "let i = 0; do { i += 1; } until i + 1; i"),
    // The operator a `??` ends with is not the branch's to take: the edge
    // that skips a non-unit operand is patched to exactly where the branch
    // goes, and it has to arrive at a test rather than past one.
    case("coalesce_guard_ends_in_a_comparison", "let a = (); let b = 1; if a ?? (b < 2) { 1 } else { 2 }"),
    case("coalesce_guard_skips_to_the_branch", "let a = true; let b = 1; if a ?? (b < 2) { 1 } else { 2 }"),
    // A guard that is a unary operator, which carries its branch the same way.
    // `!` is the one the walker short-circuits, so the guard reaches the typed
    // arm for a `bool` and the dispatch for anything else.
    case("unary_not_guard_on_an_indexed_read", "let a = [true, false]; let i = 1; if !a[i] { 1 } else { 2 }"),
    case("unary_not_while_guard", "let b = true; let n = 0; while !b { n += 1; } n"),
    case("error_unary_not_guard_int", "let i = 1; if !i { 1 } else { 2 }"),
    // A chain in statement position, which is what a loop writing a container
    // is made of. The walk runs for the method's effect and nothing reads what
    // it arrived at, so the instruction drops the value rather than pushing it
    // for the next one to take off — and what the container ends up holding is
    // how the two sides say whether the walk still happened.
    case("chain_method_in_statement_position", "let a = [1]; a.push(2); a"),
    // A property read there instead, which has no effect at all to run: the
    // whole statement is a value nothing wants. Rhai still evaluates it, and a
    // getter registered on a host type would still be called.
    case("chain_property_in_statement_position", "let m = #{ a: 1 }; m.a; m"),
    // One that mutates and then raises. What it evaluates to is discarded
    // either way, so the error is the only thing left to disagree about — and
    // the mutation before it has to have landed.
    case("error_chain_in_statement_position", "let w = widget(1); w.bump_then_fail(); w"),
    // A chain in statement position that something already jumps to: the false
    // edge of the `if` lands on the `else` arm's own instruction. Arriving
    // there carries a stack height, so the value has to be pushed and popped
    // after all rather than never pushed.
    case("chain_in_statement_position_at_a_jump_target", "let a = [1]; let n = 1; if n > 0 { a.push(2); } else { a.clear(); } a"),
    case("while_loop", "let i = 0; let s = 0; while i < 5 { s += i; i += 1; } s"),
    case("while_loop_local_bound", "let n = 5; let i = 0; let s = 0; while i < n { s += i; i += 1; } s"),
    case("do_while", "let i = 0; do { i += 1; } while i < 3; i"),
    case("do_until", "let i = 0; do { i += 1; } until i >= 3; i"),
    case("loop_break_value", "let i = 0; loop { i += 1; if i > 4 { break i * 10; } }"),
    case("continue_skips", "let s = 0; for i in 0..10 { if i % 2 == 0 { continue; } s += i; } s"),
    case("for_range", "let s = 0; for i in 0..5 { s += i; } s"),
    case("for_array", "let s = 0; for x in [10, 20, 30] { s += x; } s"),
    case("for_with_counter", "let s = 0; for (x, i) in [10, 20, 30] { s += x * i; } s"),
    // The loop variable is pushed once and mutated in place rather than
    // re-pushed each iteration (eval/stmt.rs:752); a VM that re-pushes would
    // leave the scope a different depth.
    case("for_loop_var_not_leaked", "let x = 99; for x in 0..3 { } x"),
    case("nested_loops_break", "let s = 0; for i in 0..3 { for j in 0..3 { if j == 2 { break; } s += 1; } } s"),
    // An empty body is a separate path in Rhai that never touches the loop
    // variable or the counter (`eval/stmt.rs:719`).
    case("for_empty_body", "let s = 0; for i in 0..5 { } s"),
    // `return` out of a `for` skips the exhaustion path, so the iterator and
    // both loop variables have to go with the frame.
    case("for_return_from_body", "fn find(xs) { for (x, i) in xs { if x > 1 { return i; } } -1 } find([1, 2, 3])"),
    // A `break` out of a `while` nested in a `for` must drop nothing, and out
    // of the `for` must drop one — the two are easy to get the wrong way round.
    case("for_around_while_break", "let s = 0; for i in 0..3 { let j = 0; while true { j += 1; if j > 2 { break; } s += 1; } } s"),
    // Iterating a shared cell walks a snapshot, because Rhai flattens the
    // iterable before asking for an iterator (`eval/stmt.rs:677`).
    case("for_over_captured_array", "let a = [1, 2, 3]; { let f = || a; } let s = 0; for x in a { s += x; } s"),
    // The loop variable is written *through* its cell, so a closure made on one
    // turn sees the value the last turn wrote — which is the whole reason the
    // write is not a plain slot store.
    // `call(f)` rather than `f.call()`: method-call syntax is `no_object`'s to
    // remove, and this case is about the loop variable, not about the syntax.
    case("closure_captures_the_for_loop_variable_cell", "let r = 0; { let f = (); for x in 0..3 { if x == 0 { f = || x; } } r = call(f); } r"),
    // --- switch -----------------------------------------------------------
    case("switch_literal", "let x = 2; switch x { 1 => \"one\", 2 => \"two\", _ => \"other\" }"),
    case("switch_range", "let x = 42; switch x { 0..=9 => \"small\", 10..=99 => \"medium\", _ => \"large\" }"),
    // A failing guard must fall through to the next matching case, not to the
    // default, so both single-digit arms are needed to tell those apart.
    case("switch_guard", "let x = 5; switch x { 0..=9 if x % 2 == 1 => \"odd digit\", 0..=9 => \"even digit\", _ => \"big\" }"),
    case("switch_default_only", "switch 999 { 1 => \"a\", _ => \"fallback\" }"),
    // Two case values, one arm: the table has two entries pointing at one
    // body, which a compiler emitting a body per entry would duplicate.
    case("switch_shared_body", "let x = 2; switch x { 1 | 2 => \"low\", 3 => \"three\", _ => \"other\" }"),
    // A case value that matched but whose guard declined must continue to the
    // range arms before the default (`eval/stmt.rs:546-571`).
    case("switch_declined_case_checks_ranges", "let f = false; let x = 1; switch x { 1 if f => \"guarded\", 0..=5 => \"range\", _ => \"default\" }"),
    // No `_` arm at all, so the miss has to produce unit from nowhere.
    case("switch_no_default", "let x = 9; switch x { 1 => \"a\" }"),
    case("switch_string", "let s = \"b\"; switch s { \"a\" => 1, \"b\" => 2, _ => 0 }"),
    // A range arm covers the reals between its bounds, so a float lands in one
    // even though the bounds are integers.
    case("switch_float_in_range", "let x = 5.5; switch x { 0..10 => \"in\", _ => \"out\" }"),
    // Hashing a host type panics, so the subject has to be checked before it
    // reaches a hasher — and must still find the default.
    case("switch_non-hashable_subject", "let w = widget(3); switch w { 1 => \"int\", _ => \"other\" }"),
    // A shared value is normally hashable, so it is the same as an unshared one.
    case("switch_on_an_unshared_subject_matches", r#"let v = 0; switch v { 0 => "case", _ => "default" }"#),
    case("switch_on_a_shared_subject_matches", r#"let v = 0; { let f = || v; } switch v { 0 => "case", _ => "default" }"#),
    case("switch_range_on_a_shared_subject_matches", r#"let v = 5; { let f = || v; } switch v { 0..=9 => "range", _ => "default" }"#),
    // An arm body is a block: it declares, and it has to leave the scope the
    // depth it found it — which only shows up in something that reads a local
    // afterwards.
    case("switch_block_body_scope", "let x = 1; let y = 0; switch x { 1 => { let z = 5; y = z * 2 }, _ => () } y"),
    // A jump out of an arm and out of the switch, which is where the operand
    // stack most plausibly ends up a different depth on the two paths.
    // A range is a host type as far as `Dynamic` is concerned, and
    // `is_hashable` says no to those — even though `Hash for Dynamic` would in
    // fact hash a range (types/dynamic.rs:465). So Rhai never matches a range
    // *subject* against anything, and neither may the VM: mirroring the gate
    // matters more than being clever about it.
    case("switch_range_subject_never_matches", "let r = 0..5; switch r { 0..5 => \"same\", _ => \"no\" }"),
    case("switch_break_from_loop", "let s = 0; let i = 0; while i < 10 { switch i { 3 => break, _ => () } s += 1; i += 1; } s"),
    // --- blocks used for their value ---------------------------------------
    // Rhai wraps a block in `Expr::Stmt` wherever a value is wanted, so these
    // are one construct in three disguises. Each declares inside the block, so
    // a lowering that forgot to rewind would leave the scope a different depth
    // and every slot after it would name the wrong variable.
    case("let_from_switch", "let x = 2; let y = switch x { 1 => \"one\", 2 => \"two\", _ => \"other\" }; y"),
    case("let_from_if", "let c = true; let y = if c { let a = 1; a } else { let b = 2; b }; y + 10"),
    case("let_from_block", "let a = 3; let y = { let z = a; z * 2 }; y"),
    case("is_def_var_true", r#"let a = 42; is_def_var("a")"#),
    case("is_def_var_false", r#"let a = 42; is_def_var("b")"#),
    // A block among a call's arguments, where the scope grows while operands
    // are already on the stack.
    case("block_as_argument", "fn add(a, b) { a + b } let n = 2; add({ let t = n; t + 1 }, 10)"),
    // --- scoping ----------------------------------------------------------
    case("shadowing_nested", "let x = 1; { let x = 2; { let x = 3; } } x"),
    case("block_scope_discarded", "let x = 1; { let y = 2; x += y; } x"),
    case("const_read", "const K = 10; K * 2"),
    // --- functions --------------------------------------------------------
    case("fn_call", "fn add(a, b) { a + b } add(2, 3)"),
    case("fn_call_captures_parent_scope", r#"fn foo(x) { x + y * z }  let x = 42; let y = 1; let z = 9; foo!(x)"#),
    case("is_def_fn", r#"fn add(x, y) { x + y } is_def_fn("add", 2)"#),
    // Kept shallow deliberately: Rhai's default call-depth limit is far lower
    // in debug builds than in release, and this case is about recursion working
    // at all, not about the limit. The limit gets its own case.
    case("fn_recursion", "fn fib(n) { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } } fib(6)"),
    case("fn_early_return", "fn f(x) { if x > 0 { return \"pos\"; } \"non_pos\" } f(1) + f(-1)"),
    // A script method mutating its receiver: `this` is bound by reference, so
    // the write has to land back in the caller's variable.
    case("fn_mutating_method", "fn double() { this *= 2; } let v = 21; v.double(); v"),
    case("top_level_return", "let x = 5; if x > 0 { return x * 2; } 0"),
    // A script function this compiler could not lower still runs — Rhai finds
    // it in `global.lib` — and it must run in a scope of its own. Handing it
    // this frame's would let the body read the caller's locals, where Rhai
    // gives it an empty one (`func/call.rs:1476`), and the read is the whole
    // difference: the walker cannot find `secret` and a VM that leaked its
    // scope answers with 42.
    //
    // `this` is what leaves the body non-lowerable, and calling it by name
    // rather than as a method is what routes it through generic dispatch.
    case("error_a_skipped_function_cannot_see_the_caller", "fn peek(k) { let seen = secret; this } let secret = 42; peek(1)"),
    // --- a variable in first-argument position ------------------------------
    // Rhai rewrites `f(x, ..)` into `x.f(..)` so a `&mut` first parameter
    // mutates the variable (`func/call.rs:1434`). These are the same calls the
    // chain cases make in method syntax, and they have to mean the same thing.
    case("call_style_mutating_native", "let a = [1]; push(a, 2); a"),
    case("call_style_mutating_host_type", "let w = widget(4); bump(w); w.level"),
    case("call_style_pure_native", "let a = [1, 2]; len(a)"),
    // Rhai reads the variable *after* the other arguments, so an argument that
    // writes to it is seen. Two shapes of write, because one goes through the
    // rewrite itself and the other does not.
    case("call_style_argument_writes_the_receiver", "let a = [1]; push(a, { push(a, 9); 2 }); a"),
    case("call_style_argument_replaces_the_receiver", "let a = [1]; push(a, { a = [7]; 2 }); a"),
    // The receiver appearing again among the arguments, which is where a live
    // reference into the scope would be most likely to show. It does not: the
    // later argument was read and flattened before the reference was taken, so
    // it is a copy of what the receiver held then.
    case("call_style_receiver_is_also_an_argument", "let a = [1]; push(a, a); a"),
    case("call_style_receiver_twice_over", "let a = [1]; insert(a, 0, a); a"),
    // Neither of these can be handed out by reference, so both are passed by
    // value and the mutation is discarded (`func/call.rs:1449-1454`).
    case("call_style_constant_receiver", "const a = [1]; push(a, 2); a"),
    // The closure is made in a block so the scope the two sides are compared on
    // does not end up holding a pointer, which they render differently on
    // purpose — see `a_closure_pointer_is_late_bound` in `tests/scope.rs`.
    case("call_style_shared_receiver", "let a = [1]; { let f = || a.len(); } push(a, 2); a"),
    // A script function copies its first argument whichever way it arrives, so
    // the rewrite is invisible here — which is the thing to pin.
    case("call_style_script_fn", "fn bump_it(x) { x += 1; x } let n = 3; bump_it(n); n"),
    // The receiver resolves last, so a missing one is reported after a missing
    // argument rather than before it.
    case("error_receiver_resolves_after_arguments", "no_such(receiver, argument)"),
    case("error_receiver_not_found", "let ok = 1; no_such(missing, ok)"),
    // Dispatch still fails at the call, not at the variable that reached it.
    case("error_no_function_for_the_receiver", "let a = [1]; no_such(a, 2)"),
    // An error a native *returned*, which Rhai positions at the call site like
    // everything else dispatch produces (`func/call.rs:413`). Both argument
    // shapes, because one goes through the rewrite and the other does not.
    case("error_returned_by_a_native", r#"parse_int("zz")"#),
    case("error_returned_by_a_native_by_reference", r#"let s = "zz"; parse_int(s)"#),
    // --- closures ---------------------------------------------------------
    // Function pointers are invoked via `.call()`; `f(5)` would look for a
    // function literally named `f`.
    // The closure is kept inside a block in all of these. Not incidental: the
    // pointer we build is late-bound where Rhai's is early-bound, so Rhai
    // renders it `Fn*+("anon$..")` and we render it `Fn("anon$..")`. That
    // difference is the price of not shipping an AST body, it is script-
    // visible, and `a_closure_pointer_is_late_bound` is where it is pinned —
    // so these cases test the capture rather than re-testing the rendering.
    case("closure_capture_read", "let n = 10; let r = 0; { let f = |x| x + n; r = f.call(5); } r"),
    // Capture is by shared cell, so the mutation must be visible outside.
    case("closure_capture_mutate", "let n = 0; { let f = || n += 1; f.call(); f.call(); } n"),
    // In a block for the same reason the closure cases are: what the pointer
    // *does* matches, what it renders as does not.
    case("fn_ptr_call", "fn triple(x) { x * 3 } let r = 0; { let f = Fn(\"triple\"); r = f.call(4); } r"),
    // The same site taken repeatedly, which is the other door a compiled body
    // is reached through and so the other place a call's scope is lent from
    // ([`Vm::take_scope`]). A single call cannot tell a scope built afresh from
    // one handed on.
    case("fn_ptr_call_repeated", "fn add(a, b) { a + b } let s = 0; { let p = Fn(\"add\"); let i = 0; while i < 3 { s = p.call(s, i); i += 1; } } s"),
    // The same through a name that is not a constant, so Rhai's optimizer
    // cannot fold it into a pointer carrying an environment.
    case("fn_ptr_from_dynamic_name", "fn triple(x) { x * 3 } let n = \"trip\" + \"le\"; let f = Fn(n); f.call(4)"),
    case("fn_ptr_curried", "fn add(a, b) { a + b } let n = \"a\" + \"dd\"; let f = Fn(n).curry(10); f.call(5)"),
    // A pointer to a native function goes to Rhai's own dispatch rather than
    // to a chunk of ours.
    case("fn_ptr_to_native", "let n = \"ab\" + \"s\"; let f = Fn(n); f.call(-7)"),
    // Deliberately absent: `let x = 1; x.call(2)`. That is not an error in
    // Rhai — a non-pointer target means the *argument* is the pointer and the
    // target is `this` — and the VM reproduces the behaviour, but not the
    // position. Rhai blames the argument there and the call everywhere else,
    // and one instruction has one position-table entry; using the argument's
    // was measured to move the divergence onto the common path instead of
    // removing it. A pool of positions would fix it and would not be
    // strippable.
    case("error_fn_ptr_unknown_name", "let n = \"no\" + \"pe\"; let f = Fn(n); f.call(1)"),
    // `Fn` and `curry` read their first argument and blame everything they can
    // then complain about on *it* rather than on the call — a name that is not
    // a string, a string that is not an identifier, a first argument that is
    // not a pointer (`func/call.rs:1217`, `:1220`, `:1232`).
    case("error_fn_ptr_from_a_non_string", "Fn(())"),
    case("error_fn_ptr_from_an_unusable_name", r#"Fn("not an identifier!")"#),
    case("error_curry_of_a_non_pointer", "curry(1, 2)"),
    // Capturing a variable turns its slot into a shared cell, which changes
    // what every later read and write of that slot means. Writing the slot
    // instead of writing *through* it severs the closure silently — the value
    // is right and the aliasing is dead — so this needs a write after the
    // capture to catch it.
    case("closure_shared_write", "let x = 1; let r = 0; { let f = || x; x = 42; r = f.call(); } r"),
    case("closure_shared_op_assign", "let x = 1; let r = 0; { let f = || x; x += 41; r = f.call(); } r"),
    // The same cell as the root of a chain. `get_indexed_mut` refuses a shared
    // value outright, so walking one takes the host down rather than returning
    // an error (`eval/chaining.rs:461`).
    case("closure_shared_chain_root", "let a = [1, 2, 3]; { let f = || a[0]; } a[1] = 20; a[1]"),
    // A host handing back a cell that is *already* shared. Rhai flattens a
    // declaration's initializer (`eval/stmt.rs:438`), so the sharing stops at
    // the boundary and the local holds a plain value. Reaching this needs a
    // native, because loading a variable flattens on the way out either way.
    // The `closure_` prefix is what keeps it out of a `no_closure` build, where
    // there is no such thing as a shared value to hand back.
    case("closure_shared_from_a_native", "let m = shared_cell(); m"),
    // Rhai answers this syntactically and registers no function for it, so a
    // lowered call would fail to resolve where the walker returns a bool.
    case("is_shared_after_capture", "let x = 1; let r = false; { let f = || x; r = is_shared(f); } [is_shared(x), r]"),
    case("closure_in_map", "[1, 2, 3].map(|x| x * 2)"),
    case("closure_in_filter", "[1, 2, 3, 4].filter(|x| x % 2 == 0)"),
    // Everything above captures out of the top-level scope, which lives for the
    // whole run. These capture out of a *call's* scope, which does not: the
    // callee's scope is emptied when the call ends and the VM may lend it to
    // the next call ([`Vm::take_scope`]). A closure holding a cell that came
    // out of one is the case where lending would be visible if what was lent
    // were an entry rather than an array — the second call would answer with
    // the first call's value, or the closure with the second call's.
    //
    // Three calls with two pointers still alive across the third, so a lent
    // scope has to be handed on twice before the answers are read.
    case("closure_captures_a_callees_local", "fn make(k) { let t = k; || t } let r = 0; { let p = make(1); let q = make(2); make(3); r = p.call() * 10 + q.call(); } r"),
    // The same with the calls after the capture rather than around it, so the
    // pointer is read once every later call has had the scope.
    case("closure_outlives_later_calls_to_its_maker", "fn make(k) { let t = k; || t } let r = 0; { let p = make(1); make(2); make(3); r = p.call(); } r"),
    // And a write through the cell rather than a read, since a write is what
    // severs silently: the second call must see its own argument, not the
    // first call's incremented one.
    case("closure_writes_a_callees_local", "fn bump(k) { let t = k; { let g = || t += 1; g.call(); } t } bump(1) + bump(10)"),
    // --- chained lvalues --------------------------------------------------
    // Each of these needs a different `Target` variant and its write-back.
    case("index_assign_array", "let a = [1, 2, 3]; a[1] = 99; a"),
    case("index_assign_nested", "let m = #{ xs: [1, 2, 3] }; m.xs[2] = 42; m.xs"),
    case("property_assign_deep", "let m = #{ a: #{ b: #{ c: 1 } } }; m.a.b.c = 7; m.a.b.c"),
    case("map_auto_vivify", "let m = #{}; m.fresh = 1; m"),
    // The other half of it: only a write creates a key. Reading one that is
    // not there gives unit and must leave the map alone — the map is returned
    // so the test can see whether it grew.
    case("map_read_of_absent_key_does_not_create_it", "let m = #{ a: 1 }; let r = m.b; [m, r]"),
    case("map_read_absent_through_a_chain", "let m = #{ a: #{} }; let r = m.a.b; [m, r]"),
    // Walking through an absent key reaches a detached unit, so the write has
    // nowhere to land and Rhai says so rather than creating the path.
    case("error_map_write_through_an_absent_key", "let m = #{}; m.a.b = 1; m"),
    // A closure holds the same cell, so a key invented by a read would be
    // visible from outside the expression that invented it.
    case("map_read_of_absent_key_is_not_visible_to_a_closure", "let m = #{}; let r = 0; { let f = || m; r = m.zz; } [m, r]"),
    case("op_assign_indexed", "let a = [1, 2, 3]; a[0] += 10; a"),
    // The shapes `Op::IndexSet` speculates on but must hand back: an index
    // that counts from the end, one that is off the end, a root that is a map
    // rather than an array, and one whose cell is shared with a closure. Each
    // has to reach the same answer the general walk gives, error included.
    case("index_assign_negative", "let a = [1, 2, 3]; a[-1] = 9; a"),
    case("error_index_assign_out_of_bounds", "let a = [1, 2, 3]; a[10] = 9; a"),
    case("index_assign_map_root", r#"let m = #{}; m["k"] = 9; m"#),
    case("index_assign_float_index", "let a = [1, 2, 3]; let i = 1; a[i] = 9; a"),
    // An assignment through a chain in *value* position. It evaluates to unit
    // like every other assignment, and the instruction that says so is emitted
    // only here — a statement-position one leaves nothing at all — so the
    // three places a value is read from are each worth a case.
    // The value an assigning instruction names rather than takes off the
    // stack. Four spellings, because the index is named on its own terms and
    // the value on its own: the tag carries the pair.
    case("index_assign_named_local_const", "let a = [1, 2, 3]; let i = 1; a[i] = 9; a"),
    case("index_assign_named_const_const", "let a = [1, 2, 3]; a[1] = 9; a"),
    // A value that is not a literal is stashed into a local before the
    // operands run, and the stash is what leaves nothing for the fold to take
    // -- so these keep the general instruction, and prove the fold does not
    // reach through a stash and take the load it ends with.
    case("index_assign_stashed_local_value", "let a = [1, 2, 3]; let i = 1; let v = 9; a[i] = v; a"),
    case("index_assign_stashed_value_const_index", "let a = [1, 2, 3]; let v = 9; a[1] = v; a"),
    // A boolean is pushed by an instruction of its own rather than out of the
    // constant pool, and folding it names a constant that nothing else in the
    // program mentions.
    case("index_assign_named_bool", "let a = [true, true]; a[1] = false; a"),
    // The same four, declined: the walk reads its operands off the stack, and
    // a named value was never pushed there. An index off the end is the
    // cheapest way to reach that path, and a map root reaches it without
    // raising at all.
    case("error_index_assign_named_value_out_of_bounds", "let a = [1, 2]; let v = 9; a[7] = v; a"),
    case("index_assign_named_value_map_root", r#"let m = #{}; let v = 9; m["k"] = v; m"#),
    case("index_assign_named_value_negative", "let a = [1, 2, 3]; a[-1] = 9; a"),
    case("index_assign_named_value_shared_root", "let a = [1, 2]; let v = 9; { let f = || a; a[0] = v; } a"),
    case("index_assign_is_the_scripts_value", "let a = [1, 2, 3]; a[1] = 99"),
    case("property_assign_is_the_scripts_value", "let m = #{ a: 1 }; m.a = 7"),
    case("index_assign_is_a_function_bodys_value", "fn place(a) { a[0] = 9 } let a = [1, 2, 3]; [place(a), a]"),
    // `Op::IndexGet` speculates on the same shape as `Op::IndexSet` and has
    // the same set of things to hand back: a root that is not an array, an
    // index that is not a non-negative integer inside it, and an element that
    // is shared. Each has to reach the answer the general walk gives.
    case("index_read_local_array", "let a = [1, 2, 3]; a[1]"),
    case("index_read_local_array_const", "const A = [1, 2, 3]; A[2]"),
    case("index_read_negative", "let a = [1, 2, 3]; a[-1]"),
    case("error_index_read_out_of_bounds", "let a = [1, 2, 3]; a[10]"),
    case("error_index_read_float_index", "let a = [1, 2, 3]; a[1.5]"),
    case("index_read_map_root", r#"let m = #{ k: 9 }; m["k"]"#),
    case("index_read_string_root", r#"let s = "hello"; s[1]"#),
    case("index_read_bitfield_root", "let x = 5; x[0]"),
    case("index_read_blob_root", "let b = blob(3, 7); b[1]"),
    case("index_read_of_a_shared_element", "let a = [1, 2]; let r = 0; { let f = || a; r = a[0]; } [a, r]"),
    case("index_read_through_a_shared_root", "let a = [1, 2]; let r = 0; { let f = || a; r = a[1]; } r"),
    case("index_read_feeds_a_write_back", "let a = [1, 2, 3]; a[0] = a[2]; a"),
    // An index the instruction names rather than takes off the stack. The two
    // sources are a slot and a constant, and each has to decline to the same
    // walk when what it names is not an integer inside the array — including
    // the slot that has become a shared cell, which is not read the way a
    // plain one is.
    case("error_index_read_slot_is_not_an_integer", r#"let a = [1, 2, 3]; let k = "x"; a[k]"#),
    case("error_index_read_constant_is_not_an_integer", r#"let a = [1, 2, 3]; a["k"]"#),
    case("index_read_through_a_shared_index", "let a = [1, 2]; let i = 1; let r = 0; { let f = || i; r = a[i]; } r"),
    case("index_assign_through_a_shared_index", "let a = [1, 2]; let i = 1; { let f = || i; a[i] = 99; } a"),
    case("error_index_assign_float_index", "let a = [1, 2, 3]; a[1.5] = 9; a"),
    case("error_index_assign_constant_is_not_an_integer", r#"let a = [1, 2, 3]; a["k"] = 9; a"#),
    case("index_read_local_index", "let a = [1, 2, 3]; let i = 1; a[i]"),
    case("index_assign_local_index", "let a = [1, 2, 3]; let i = 1; a[i] = 9; a"),
    // An index that is more than one push, which the instruction cannot name —
    // so it arrives on the stack, under the value where there is one.
    case("index_read_computed_index", "let a = [1, 2, 3]; let i = 0; a[i + 1]"),
    // The index's own last push is not the index. Reading only the code
    // emitted for the whole chain would take this one back and hand the
    // instruction an argument as its index.
    case("index_read_call_index", "let a = [1, 2, 3]; let i = 0; let j = 1; a[max(i, j)]"),
    case("index_assign_call_index", "let a = [1, 2, 3]; let i = 0; let j = 1; a[max(i, j)] = 9; a"),
    case("index_assign_computed_index", "let a = [1, 2, 3]; let i = 0; a[i + 1] = 9; a"),
    // A chain is walked where its root lives rather than in a copy of it, so
    // the access mode of the entry is what refuses a write — not the fact that
    // the walk was handed something detached. All three have to agree with
    // Rhai, which reaches the same entry through a `Target`.
    // No `A[0] = 9` case: Rhai blames `ErrorAssignmentToConstant` on the
    // variable and the VM blames it on the `[`, because `Root::Local` carries
    // no position of its own the way `Root::Named` and `Root::This` do. That
    // predates the borrow and needs a field in the chain pool to fix.
    case("const_root_index_read", "const A = [1, 2, 3]; A[1]"),
    case("error_const_root_method_step", "const A = [1, 2]; A.push(3); A"),
    // The index is evaluated before the root is reached, so a method call in it
    // sees the root as it was — and cannot be holding it when the walk starts.
    case("index_expression_reads_the_root", "let a = [1, 2, 3]; a[a.len() - 1] = 9; a"),
    case("bitfield_assign", "let x = 0; x[2] = true; x"),
    case("string_char_assign", r#"let s = "hello"; s[0] = 'H'; s"#),
    case("string_slice_read", r#"let s = "hello world"; s[0..5]"#),
    // The inclusive form is a different `TypeId` and a different pool tag, so
    // one does not cover the other. A string rather than an array, because
    // Rhai indexes arrays with integers only and slices them with `extract`.
    case("string_slice_inclusive", r#"let s = "hello world"; s[6..=9]"#),
    // --- coalesce ---
    case("coalesce", "let a = (); let b = (); let c = 42; a ?? b ?? c"),
    case("coalesce_middle", "let a = (); let b = 42; let c = 123; a ?? b ?? c"),
    case("chain_index_coalesce", "let a = (); a?[1]?[2]"),
    case("chain_property_coalesce", "let a = (); a?.b?.c"),
    // --- chains rooted at something that is not a variable ------------------
    // Rhai evaluates the root into a temporary and walks that
    // (`eval/chaining.rs:561-571`), so there is no scope entry behind it and
    // nothing is written back. One case per root shape, because each reaches a
    // different `Target`.
    case("temp_root_array_method", "[3, 1, 2].len()"),
    case("temp_root_array_index", "[10, 20, 30][1]"),
    case("temp_root_string", r#""hello".to_upper()"#),
    case("temp_root_map_property", "#{ a: 1, b: 2 }.b"),
    case("temp_root_call", "fn make() { [1, 2, 3] } make().len()"),
    case("temp_root_parenthesised", "let a = 1; let b = 2; (a + b).to_string()"),
    case("temp_root_nested", "[[1, 2], [3, 4]][1][0]"),
    // A mutating method on a temporary. The mutation has nowhere to land, and
    // the point is that both sides agree it is discarded rather than one of
    // them inventing a place to put it.
    case("temp_root_mutating_method", "let a = [1, 2, 3]; [a.len()].push(9)"),
    case("temp_root_host_mutates", "widget(4).bump()"),
    case("temp_root_host_pure", "widget(4).doubled()"),
    // Order, which is the part that is not obvious: Rhai collects a chain's
    // indices *before* it evaluates what they apply to. Both halves fail, so
    // the position in the reported error is which one ran first.
    // An operator with no implementation for the types it got. The corpus
    // reaches `ErrorFunctionNotFound` through a named call elsewhere, which is
    // a different dispatch path and positions itself differently.
    case("error_operator_undefined_for_types", "let a = 1.0; a + #{ b: 1 }"),
    // The arithmetic guards, which are the reason an integer operator is a
    // fallible one — and which `unchecked` removes, so these do not run there.
    // Every one of them reports `ErrorArithmetic` with no position at all,
    // because a built-in operator's error comes back untouched under
    // `fast_operators` (`func/call.rs:1798`).
    case("error_int_add_overflow", "let a = 1; let n = 0; while n < 200 { a += a; n += 1; } a"),
    case("error_int_multiply_overflow", "let a = 3; let n = 0; while n < 64 { a = a * a; n += 1; } a"),
    case("error_int_modulo_by_zero", "let z = 0; 7 % z"),
    case("error_int_power_negative", "let a = 2; let b = -1; a ** b"),
    case("error_op_assign_int_power_negative", "let a = 2; let b = -1; a **= b; a"),
    // A chain step that fails, one per kind. Rhai blames the step rather than
    // the chain, and a chain is one instruction with one position-table entry,
    // so these are what make each step carry its own.
    case("error_property_on_a_variable", "let x = 1; x.a"),
    case("error_property_on_a_temporary", "[1, 2].a"),
    case("error_method_on_a_variable", "let x = 1; x.to_upper()"),
    case("error_property_deep_in_a_chain", "let m = #{ a: #{} }; m.a.b.c"),
    // The op-assign form, which falls back to the plain operator when no
    // `+=` is registered and used to lose the position on the way.
    case("error_op_assign_undefined_for_types", "let a = 1.0; a += #{ b: 1 }; a"),
    case("error_temp_root_index_runs_first", "let z = 0; [1 / z][9 / z]"),
    // Two positions belong to an index step, not one: the index expression,
    // which an out-of-bounds is blamed on, and the `[` in front of it, which
    // indexing something unindexable is blamed on. They only come apart in a
    // chain of more than one step — here `n[0]` bit-indexes an integer and
    // yields a bool, and Rhai names the *second* `[`.
    case("error_index_into_an_unindexable_step", "let n = 0; n[0][5]"),
    case("error_index_into_an_unindexable_step_deep", "let m = #{ a: 1 }; m.a[0][5]"),
    // --- what an escaping error leaves in the scope -------------------------
    // Rhai rewinds a block whether it is left normally or by a throw, and
    // rewinds nothing at the top level. The comparison that matters in all of
    // these is the leftover scope rather than the error.
    case("throw_from_a_block_rewinds_it", "let a = 1; { let b = 2; throw 3; }"),
    case("throw_from_a_for_body_drops_the_loop_var", "let a = 1; for i in 0..3 { throw i; }"),
    case("throw_from_a_while_body_drops_its_locals", "let a = 1; let n = 0; while n < 3 { let b = n; throw b; }"),
    // A catch block is a block too, and its variable is the one Rhai pushes
    // rather than the script.
    case("throw_from_a_catch_drops_the_catch_var", "let a = 1; try { throw 2; } catch (e) { throw e; }"),
    case("throw_from_a_nested_block_drops_every_level", "let a = 1; { let b = 2; { let c = 3; for i in 0..2 { throw i; } } }"),
    // The frame boundary: a function's own locals go with its scope, and the
    // caller's top-level ones stay.
    case("throw_from_a_function_leaves_the_caller_top_level_alone", "fn boom() { let inner = 9; throw inner; } let a = 1; { let b = 2; boom(); }"),
    // Nothing to rewind, which is the case a floor set too low would break.
    case("throw_at_the_top_level_keeps_what_ran", "let a = 1; let b = 2; throw 3;"),
    case("error_temp_root_out_of_bounds", "[1, 2, 3][99]"),
    // --- errors -----------------------------------------------------------
    // Compared by variant and position, so a VM that reports the right failure
    // at the wrong place still fails the test.
    case("error_unknown_variable", "let a = 1; a + nonexistent"),
    // Positions on call failures are set by different code paths depending on
    // whether the callee was found, so both need pinning.
    case("error_unknown_function", "let a = 1; no_such_function(a)"),
    case("error_wrong_arity", "fn f(a, b) { a } f(1)"),
    case("error_array_bounds", "let a = [1, 2]; a[10]"),
    // Rhai maps the *expected* type through its registered names and leaves the
    // *actual* one raw, so a range guard reports `core::ops::range::Range<INT>`
    // rather than the `range` the same engine prints everywhere else. Mapping
    // both is the obvious mistake, and only a type with a registered name shows
    // it up.
    case("error_condition_is_a_range", "if 0..1 { 1 } else { 2 }"),
    case("error_condition_is_a_host_type", "let w = widget(1); while w { 1 }"),
    case("error_type_mismatch", r#"let a = 1; a + "string" + [1]"#),
    case("error_divide_by_zero", "let a = 1; a / 0"),
    case("throw_value", "throw 42"),
    case("throw_in_fn", "fn f() { throw \"boom\"; } f()"),
    // --- try / catch ------------------------------------------------------
    case("try_catch_value", "try { throw 7; } catch (e) { e * 2 }"),
    case("try_catch_native_error", "try { let a = [1]; a[9] } catch (e) { e.message != () }"),
    case("try_catch_rethrow", "try { try { throw 1; } catch { throw; } } catch (e) { e }"),
    // `return` unwinds as an error but must pass straight through a catch.
    case("try_catch_does_not_swallow_return", "fn f() { try { return 1; } catch { return 2; } } f()"),
    case("try_catch_no_error", "try { 5 } catch { 6 }"),
    // The catch block's value is discarded — the statement is unit on the
    // caught path and the try block's value otherwise (`eval/stmt.rs:863`).
    case("try_catch_discards_its_value", "try { throw 1; } catch { 99 }"),
    // A jump out of a `try` skips the `PopHandler` the straight-line path
    // would have run. Left armed, the next error is caught into a block that
    // has already been left — so the second failure here must not be caught.
    case("break_out_of_try_disarms_it", "let s = 0; while true { try { throw 1; } catch { break; } } try { throw 2; } catch (e) { s = e; } s"),
    case("break_out_of_for_inside_try", "let s = 0; for i in 0..5 { try { if i == 2 { break; } s += i; } catch { s = -1; } } s"),
    case("continue_out_of_try_inside_for", "let s = 0; for i in 0..5 { try { if i % 2 == 0 { continue; } s += i; } catch { s = -1; } } s"),
    // An error out of a called function arrives wrapped in
    // `ErrorInFunctionCall`, which is catchable, and `unwrap_inner` is what
    // still binds the bare thrown value.
    case("try_around_a_compiled_call", "fn boom() { throw 7; } try { boom(); } catch (e) { e }"),
    // `return` is a pseudo error and must pass straight through a handler.
    case("try_does_not_catch_return", "fn f() { try { return 1; } catch { 2 } } f()"),
    // --- host types -------------------------------------------------------
    //
    // The one part of the chain walker that approximates rather than
    // reproduces. A getter hands back a value, so anything below it mutates a
    // temporary that only the setter can put back — and Rhai decides whether
    // to call the setter from `func.is_method()`, which is not visible from
    // outside the crate.
    case("host_get", "let w = widget(4); w.level"),
    case("host_set", "let w = widget(4); w.level = 9; w.level"),
    case("host_op_assign", "let w = widget(4); w.level += 5; w.level"),
    case("host_index_get", "let w = widget(1); w[1]"),
    case("host_index_set", "let w = widget(1); w[1] = 99; w[1]"),
    case("host_method_mutates", "let w = widget(4); w.bump(); w.level"),
    case("host_method_pure", "let w = widget(4); w.doubled()"),
    case("host_string_index_property_get_fallback", "let w = widget(1); w.a"),
    case("host_string_index_property_set_fallback", "let w = widget(1); w.ab = 77; w.ab"),
    case("host_string_index_property_op_assign_fallback", "let w = widget(1); w.a += 5; w.a"),
    // Two levels, so the middle one is a temporary.
    case("host_temp_set", "let h = holder(3); h.inner.level = 8; h.inner.level"),
    case("host_temp_index_set", "let h = holder(3); h.inner[0] = 7; h.inner[0]"),
    case("host_temp_string_index_property_set_fallback", "let h = holder(3); h.inner.ab = 7; h.inner.ab"),
    // The mirror of it: an *index* step handing back the temporary, with the
    // property below. The index has to survive the getter to address the setter
    // with afterwards.
    case("host_index_temp_set", "let h = holder(3); h[0].level = 8; h.inner.level"),
    // A mutating call on a temporary: Rhai writes it back, so the change
    // survives.
    case("host_temp_mutates", "let h = holder(3); h.inner.bump(); h.inner.level"),
    // A read-only call on a temporary, which is where Rhai's own flag decides
    // whether a setter runs at all.
    case("host_temp_pure", "let h = holder(3); h.inner.doubled()"),
    case("error_host_index_bounds", "let w = widget(1); w[99]"),
    // A step that mutates and then raises. The error is caught, so what is
    // being compared is whether the mutation reached the variable — Rhai's does,
    // because it never walked a copy.
    case("host_mutation_before_a_failure_survives", "let w = widget(1); try { w.bump_then_fail(); } catch(e) {} w.level"),
    case("host_mutation_before_a_failure_survives_in_a_map", "let m = #{ w: widget(1) }; try { m.w.bump_then_fail(); } catch(e) {} m.w.level"),
    case("host_mutation_before_a_failure_survives_in_an_array", "let a = [widget(1)]; try { a[0].bump_then_fail(); } catch(e) {} a[0].level"),
    // `this`, which is a register rather than a scope entry and so is reached
    // by instructions of its own.
    case("this_read", "fn get() { this } let v = 7; v.get()"),
    case("this_in_an_expression", "fn double() { this * 2 } let v = 21; v.double()"),
    case("this_assign", "fn set() { this = 9; } let v = 1; v.set(); v"),
    case("this_op_assign", "fn bump(n) { this += n; } let v = 1; v.bump(4); v"),
    case("this_op_assign_on_a_string", "fn add(s) { this += s; } let v = \"a\"; v.add(\"b\"); v"),
    case("this_is_the_body_value", "fn twice() { this + this } let v = 4; v.twice()"),
    // Never inherited: a plain call from a bound body gets no receiver.
    case("error_this_is_not_inherited", "fn outer() { inner() } fn inner() { this } let v = 1; v.outer()"),
    case("error_this_unbound_in_call_style", "fn get() { this } get()"),
    // The check precedes the right-hand side, unlike the variable arm.
    case("error_this_assign_unbound_beats_a_bad_value", "fn set() { this = no_such; } set()"),
    // Chains rooted at `this`, which must write back into the caller's value.
    case("this_property", "fn count() { this.n } let m = #{ n: 5 }; m.count()"),
    case("this_property_assign", "fn set() { this.n = 9; } let m = #{ n: 1 }; m.set(); m.n"),
    case("this_index", "fn first() { this[0] } let a = [3, 4]; a.first()"),
    case("this_index_assign", "fn set() { this[0] = 9; } let a = [1, 2]; a.set(); a"),
    case("this_method_step", "fn grow() { this.push(3); } let a = [1, 2]; a.grow(); a"),
    // The receiver is moved out of the register for the walk, so a chain that
    // fails partway has to put it back — otherwise the next `this` in the same
    // body is unbound.
    case("this_survives_a_failed_chain", "fn f() { try { this[9] } catch(e) { this[0] } } let a = [1, 2]; a.f()"),
    case("this_host_method", "fn raise() { this.bump(); } let w = widget(4); w.raise(); w.level"),
    case("this_host_property", "fn read() { this.level } let w = widget(4); w.read()"),
    // A method on `this` that reaches another compiled function.
    case("this_nested_method", "fn outer() { this.inner() } fn inner() { this * 2 } let v = 5; v.outer()"),
    // `f(this, ..)`, which Rhai rewrites to `this.f(..)` by reference.
    case("this_as_first_argument", "fn grow() { push(this, 3); } let a = [1, 2]; a.grow(); a"),
    case("this_as_first_argument_pure", "fn size() { len(this) } let a = [1, 2]; a.size()"),
    case("this_as_a_later_argument", "fn plus(n) { n + this } let v = 1; v.plus(2)"),
    // The same site taken repeatedly. A chained method call starts a scope of
    // its own, and the VM may lend that scope on to the next call
    // ([`Vm::take_scope`]) — which a single call cannot distinguish from
    // building one, so the loop is the assertion.
    case("this_method_call_repeated", "fn plus(k) { this + k } let s = 0; let i = 0; while i < 3 { s = s.plus(i); i += 1; } s"),
    // Arity excludes the receiver, so these are two different functions.
    case("this_method_arity", "fn f() { 1 } fn f(x) { this + x } let v = 10; [v.f(), v.f(5)]"),
    // `obj.call(f)` binds `obj` as the closure's `this` by reference, so a
    // write inside the closure reaches `obj`. The operand stack only ever holds
    // a copy of it, which is why the instruction carries where it came from.
    //
    // The pointer is scoped to a block throughout, as the other closure cases
    // are: a compiled closure's `FnPtr` carries a name where Rhai's carries the
    // body and its environment, so one left in the scope compares unequal for a
    // reason that has nothing to do with the call.
    case("closure_call_on_a_local_writes_back", "let v = 21; { let f = || { this *= 2; }; v.call(f); } v"),
    case("closure_call_on_a_local_inline", "let v = 21; v.call(|| { this *= 2; }); v"),
    case("closure_call_on_a_local_reads", "let v = 21; let r = 0; { let f = || this * 2; r = v.call(f); } r"),
    case("closure_call_mutates_an_array", "let a = [1]; { let f = || { this.push(2); }; a.call(f); } a"),
    // And the receiver can be the frame's own receiver.
    case("closure_call_on_this", "fn twice() { let f = || { this *= 2; }; this.call(f); } let v = 21; v.twice(); v"),
    // A temporary receiver has nowhere to write back to, and Rhai mutates a
    // copy of it too.
    case("closure_call_on_a_temporary", "let r = 0; { let f = || { this *= 2; }; r = (20 + 1).call(f); } r"),
    // A native calling a pointer back against a receiver. How many arguments it
    // appends beside the receiver is the native's business — `map` adds an
    // index, `reduce` the running result — so no single wrapper arity is right
    // and these have to stay reachable by Rhai itself.
    case("closure_map_binds_this", "[1, 2, 3].map(|| this * 2)"),
    case("closure_filter_binds_this", "[1, 2, 3].filter(|| this > 1)"),
    case("closure_for_each_binds_this", "let t = 0; [1, 2, 3].for_each(|| t += this); t"),
    // And the argument form, which takes the element as a parameter instead.
    case("closure_map_takes_an_argument", "[1, 2, 3].map(|x| x * 2)"),
    // A crossing hands what it finished with to the next one, so what one
    // leaves has to be invisible to the next. Each of these is a shape where
    // something lent that should not have been would answer differently, and
    // none of them can be told from a crossing that inherits nothing by
    // anything but its answer. See `grain::vm::callback`.
    //
    // Nested first, because that is where two crossings are live at once and
    // the inner one cannot be handed what the outer is still using.
    case("closure_map_nested", "let a = [[1, 2], [3, 4]]; a.map(|r| r.map(|x| x * 2))"),
    // Repeated further than a pool can be primed by, over a body whose answer
    // depends on its own argument and nothing else.
    case("closure_map_repeated", "let a = []; for i in 0..40 { a.push(i); } a.map(|x| x * x)"),
    // Two crossings of one run through two different bodies, so what the
    // second inherits was filled by a chunk that is not its own.
    case("closure_map_then_filter", "let a = [1, 2, 3, 4, 5, 6]; let b = a.map(|x| x * 3); b.filter(|x| x % 2 == 0)"),
    // A body with a local, so every crossing binds a scope the one before it
    // gave back — and a longer array than the pool holds entries.
    case("closure_map_body_with_a_local", "let a = [1, 2, 3, 4]; a.map(|x| { let t = x + 1; t * t })"),
    // A crossing that raises. Its parts go back however it ended, and the
    // crossings after it have to be as clean as the ones before it.
    case("closure_map_raises_partway", "let a = [1, 2, 3]; let r = 0; try { a.map(|x| if x == 2 { throw x } else { x }); } catch (e) { r = e; } [r, a.map(|x| x + 1)]"),
    // A closure made inside a crossing and called after every later crossing
    // has had the parts: the cell it captured came out of a scope the pool
    // lends on, which is where lending an entry rather than an array would be
    // visible. The maker is a named function because an anonymous one declared
    // inside a callback body is still a fragment, and a fragment would hand the
    // whole thing back to the walker; the pointers are read out into locals
    // because calling one through an index is a fragment too.
    case("closure_made_inside_a_callback", "fn make(k) { let t = k * 10; || t } let a = [1, 2]; let r = 0; { let fs = a.map(|x| make(x)); let p = fs[0]; let q = fs[1]; r = p.call() + q.call(); } r"),
    // `type_of` has no registered implementation anywhere — Rhai answers it by
    // name — so it is reached through the same door every other call is.
    // A constant argument is folded by the optimizer and proves nothing.
    case("type_of_a_variable", "let x = 1; type_of(x)"),
    case("type_of_a_container", "let a = [1]; type_of(a)"),
    case("type_of_a_host_type", "let w = widget(1); type_of(w)"),
    case("type_of_method_style", "let s = \"a\"; s.type_of()"),
    case("type_of_a_pointer", "let r = \"\"; { let f = |x| x; r = type_of(f); } r"),
    // --- call resolution ---
    // A site that runs more than once resolves once and remembers, so what it
    // remembered has to be right for every later turn. Each of these puts a
    // site in a loop and changes one thing under it. A constant argument is
    // folded by the optimizer, so every one of them computes its argument.
    case("resolution_repeated_native", "let s = 0; for i in 0..5 { s += abs(0 - i - 1); } s"),
    case("resolution_two_sites_one_name", "let s = 0; for i in 0..5 { s += abs(0 - i - 1) + abs(0 - i - 2); } s"),
    case("resolution_alternating_argument_types", "let t = \"\"; for i in 0..4 { let x = if i % 2 == 0 { 7 + i } else { \"s\" }; t += to_string(x); } t"),
    case("resolution_arity_decides", "fn abs(x, y) { x + y } let s = 0; for i in 0..4 { s += abs(0 - i - 1) + abs(i, 1); } s"),
    case("resolution_script_function_shadows_a_native", "fn abs(x) { 999 } let s = 0; for i in 0..3 { s += abs(0 - i - 1); } s"),
    case("resolution_syntactic_name_in_a_loop", "let t = \"\"; for i in 0..3 { t += type_of(i + 0); } t"),
    case("resolution_inside_a_script_function", "fn g(n) { abs(n) } let s = 0; for i in 0..4 { s += g(0 - i - 1); } s"),
    case("error_resolution_function_not_found", "let s = 0; for i in 0..2 { s += nosuch(i); } s"),
    // A shared cell holds whatever is inside the lock, so a site that sees one
    // may not be answered out of what it resolved for anything else.
    case("closure_shared_argument_to_a_native", "let x = 0 - 5; { let f = || x; } let s = 0; for i in 0..3 { s += abs(x); } s"),
    // --- optimizer ---
    case("optimizer_folding_switch", "let a = 1; { let b = 99; switch b { _ => b } }"),
    case("optimizer_folding_variables_access", "let a = 1; { let b = 99; b; b; b; b }"),
    case("optimizer_folding_internal_variables_access", "let a = 1; { let b = 99; b; b; b; b } a"),
];
