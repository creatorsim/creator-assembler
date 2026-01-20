/*
 * Copyright 2018-2024 Felix Garcia Carballeira, Alejandro Calderon Mateos, Diego Camarmas Alonso,
 * Álvaro Guerrero Espinosa
 *
 * This file is part of CREATOR.
 *
 * CREATOR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CREATOR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with CREATOR.  If not, see <http://www.gnu.org/licenses/>.
 */

//! Module containing the interpreter engine used to expand pseudoinstructions
//!
//! The entry point for expanding pseudoinstructions is the [`expand()`] function

// NOTE: for compatibility with pseudoinstruction definitions written for the old compiler, this
// module reimplements the same functionality used by the old compiler with minimal changes. This
// should be completely redesigned from scratch once we are ready to make a breaking change in the
// definition of the architecture

use num_bigint::{BigInt, BigUint};
use regex::{Captures, Regex};

use std::fmt::Write as _;
use std::sync::LazyLock;

use crate::architecture::Pseudoinstruction;
use crate::number::Number;
use crate::parser::{ParseError, Token};
use crate::span::Range;

use super::{ArgumentType, Context, ErrorData, ErrorKind, InstructionDefinition};
use super::{Expr, ParsedArgs, ParsedArgument};
use super::{Span, Spanned, SpannedErr};

/// Pseudoinstruction evaluation error kind
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    UnknownFieldName(String),
    UnknownFieldNumber { idx: usize, size: usize },
    UnknownFieldType(String),
    EmptyBitRange,
    BitRangeOutOfBounds { upper_bound: usize, msb: usize },
    EvaluationError(String),
    ParseError(ParseError),
}

/// Pseudoinstruction evaluation error
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Definition of the string at the point of the error
    pub definition: String,
    /// Location in the definition that caused the error
    pub span: Range,
    /// Type of the error
    pub kind: Kind,
}

impl Error {
    /// Adds a span and a pseudoinstruction name to the error, promoting it to a [`ErrorData`]
    ///
    /// # Parameters
    ///
    /// * `def`: definition of the pseudoinstruction
    /// * `span`: location in the assembly code that caused the error
    #[must_use]
    fn compile_error(self, def: &Pseudoinstruction, span: Span) -> ErrorData {
        ErrorKind::PseudoinstructionError {
            name: def.name.to_owned(),
            error: Box::new(self),
        }
        .add_span(span)
    }
}

#[cfg(not(feature = "pseudoinstructions"))]
mod js {
    #[derive(Debug)]
    pub enum Never {}

    pub fn eval_expr(_: &str) -> Result<Never, String> {
        unimplemented!("Evaluating js code during pseudoinstruction expansion requires the `pseudoinstruction` feature flag");
    }

    pub fn eval_fn(_: &str) -> Result<Never, String> {
        unimplemented!("Evaluating js code during pseudoinstruction expansion requires the `pseudoinstruction` feature flag");
    }

    #[must_use]
    pub fn to_string<T>(_: T) -> String {
        unimplemented!("Evaluating js code during pseudoinstruction expansion requires the `pseudoinstruction` feature flag");
    }
}

#[cfg(feature = "pseudoinstructions")]
mod js {
    use js_sys::wasm_bindgen::JsValue;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        /// Converts a `JS` value to a `JS` string
        #[wasm_bindgen(js_name = String)]
        fn string(x: JsValue) -> js_sys::JsString;
    }

    // Function
    // NOTE: Modification of [`js_sys::Function`] to add `catch` to the constructor
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(extends = js_sys::Object, is_type_of = JsValue::is_function, typescript_type = "Function")]
        #[derive(Clone, Debug, PartialEq, Eq)]
        type Function;

        /// The `Function` constructor creates a new `Function` object. Calling the
        /// constructor directly can create functions dynamically, but suffers from
        /// security and similar (but far less significant) performance issues
        /// similar to `eval`. However, unlike `eval`, the `Function` constructor
        /// allows executing code in the global scope, prompting better programming
        /// habits and allowing for more efficient code minification.
        ///
        /// [MDN documentation](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Function)
        #[wasm_bindgen(constructor, catch)]
        fn new_no_args(body: &str) -> Result<Function, JsValue>;

        /// The `call()` method calls a function with a given this value and
        /// arguments provided individually.
        ///
        /// [MDN documentation](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Function/call)
        #[wasm_bindgen(method, catch, js_name = call)]
        fn call0(this: &Function, context: &JsValue) -> Result<JsValue, JsValue>;
    }

    /// Evaluates code corresponding to a `JS` expression
    ///
    /// # Errors
    ///
    /// Errors if there is any exception during the execution of the code
    pub fn eval_expr(src: &str) -> Result<JsValue, String> {
        js_sys::eval(src).map_err(to_string)
    }

    /// Evaluates code corresponding to the body of a `JS` function
    ///
    /// # Errors
    ///
    /// Errors if there is any exception during the execution of the code
    pub fn eval_fn(src: &str) -> Result<JsValue, String> {
        Function::new_no_args(src)
            .map_err(to_string)?
            .call0(&JsValue::TRUE)
            .map_err(to_string)
    }

    /// Converts a `JS` value to a string
    #[must_use]
    pub fn to_string(x: JsValue) -> String {
        String::from(string(x))
    }
}

/// Unwraps an expression containing a register name
///
/// # Errors
///
/// Errors if the expression doesn't contain a register name
fn reg_name(arg: &Spanned<Expr>) -> Result<String, ErrorData> {
    match &arg.0 {
        Expr::Identifier((name, _)) => Ok(name.clone()),
        Expr::Integer(i) => Ok(i.to_string()),
        _ => Err(ErrorKind::IncorrectArgumentType {
            expected: ArgumentType::RegisterName,
            found: ArgumentType::Expression,
        }
        .add_span(arg.1)),
    }
}

/// Gets the [`Span`] of a capture group
///
/// # Parameters
///
/// * `captures`: list of capture groups
/// * `i`: index of the capture group to get
///
/// # Panics
///
/// Panics if the index is out of bounds
#[must_use]
fn capture_span(captures: &Captures, i: usize) -> Range {
    captures
        .get(i)
        .expect("The capture group number given should exist")
        .range()
}

/// Result of expanding a pseudoinstruction
type ExpandedInstructions<'a> = Vec<Spanned<(InstructionDefinition<'a>, ParsedArgs)>>;

/// Expands a pseudoinstruction to a sequence of instructions, which might be real or another
/// pseudoinstruction
///
/// # Parameters
///
/// * `ctx`: compilation context to use
/// * `address`: address in which the instruction is being compiled into
/// * `instruction`: pseudoinstruction definition to use
/// * `args`: arguments of the instruction being expanded
///
/// # Errors
///
/// Errors if there is any problem expanding the pseudoinstruction
#[allow(clippy::too_many_lines)]
pub fn expand<'arch>(
    ctx: &mut Context<'arch>,
    address: &BigUint,
    instruction: (&'arch Pseudoinstruction, Span),
    args: &ParsedArgs,
) -> Result<ExpandedInstructions<'arch>, ErrorData> {
    // Regex used
    // Gets the value of the i-th argument from bits j to k, evaluating the argument as the given
    // type
    static FIELD_VALUE: LazyLock<Regex> = crate::regex!(r"Field\.(\d+)\.\((\d+),(\d+)\)\.(\w+)");
    // Gets the size of the i-th argument
    static FIELD_SIZE: LazyLock<Regex> = crate::regex!(r"Field\.(\d+)\.SIZE");
    // Gets the register name of the i-th argument
    static REG_NAME: LazyLock<Regex> = crate::regex!(r"reg_name\{(\d+)\}");
    // Evaluates a `JS` expression that doesn't return a value
    static NO_RET_OP: LazyLock<Regex> = crate::regex!(r"no_ret_op\{([^}]*?)\};");
    // Evaluates a `JS` expression should be replaced with its return value
    static OP: LazyLock<Regex> = crate::regex!(r"op\{([^}]*?)\}");
    // Block of code containing a list of instructions
    static INSTRUCTIONS: LazyLock<Regex> = crate::regex!(r"\{(.*?)\}");

    // Function to evaluate a label within an expression
    let ident_eval = |label: &str| super::label_eval(&ctx.label_table, address, label);

    // Parse a number from a string that already matched a number regex
    let num = |x: &str| {
        x.parse()
            .expect("This should have already matched a number")
    };
    let (instruction, span) = instruction;
    let arch = ctx.arch;

    // Get the argument corresponding to the field with the given name
    let get_arg = |name: &str| {
        args.iter()
            .find(|arg| instruction.syntax.fields[arg.field_idx].name == name)
    };

    // Expansion
    let mut def = instruction.definition.replace('\n', "");
    let mods = &arch.modifiers;

    // Replace occurrences of `Field.number`
    while let Some(x) = FIELD_VALUE.captures(&def) {
        let (_, [arg, start_bit, end_bit, ty]) = x.extract();
        let arg_num = num(arg) - 1;
        // Get the user's argument expression
        let arg: &ParsedArgument = args.get(arg_num).ok_or_else(|| {
            Error {
                definition: def.clone(),
                span: capture_span(&x, 1),
                kind: Kind::UnknownFieldNumber {
                    idx: arg_num + 1,
                    size: args.len(),
                },
            }
            .compile_error(instruction, span)
        })?;
        let (value, value_span) = &arg.value;
        // Get the range of bits requested
        let start_bit = num(start_bit);
        let end_bit = num(end_bit);
        if start_bit < end_bit {
            return Err(Error {
                definition: def.clone(),
                span: capture_span(&x, 2).start..capture_span(&x, 3).end,
                kind: Kind::EmptyBitRange,
            }
            .compile_error(instruction, span));
        }
        // Evaluate the expression according to the requested type
        #[allow(clippy::cast_possible_truncation)]
        let field = match ty {
            "int" => {
                // Convert the number to binary using two's complement
                let s = BigInt::try_from(value.eval(ident_eval, mods)?)
                    .add_span(*value_span)?
                    .to_signed_bytes_be()
                    .iter()
                    .fold(String::new(), |mut s, byte| {
                        write!(s, "{byte:08b}").expect("Writing to a string shouldn't fail");
                        s
                    });
                // Pad the number to `start_bit` bits, using sign extension
                let msb = start_bit + 1;
                if s.len() >= msb {
                    s
                } else {
                    let pad = s
                        .chars()
                        .next()
                        .expect("There should always be at least 1 character");
                    let mut pad = std::iter::repeat_n(pad, msb - s.len()).collect::<String>();
                    pad.push_str(&s);
                    pad
                }
            }
            "float" => format!("{:032b}", value.eval_no_ident(mods)?.to_f32().to_bits()),
            "double" => format!("{:064b}", value.eval_no_ident(mods)?.to_f64().to_bits()),
            ty => {
                return Err(Error {
                    definition: def.clone(),
                    span: capture_span(&x, 4),
                    kind: Kind::UnknownFieldType(ty.to_owned()),
                }
                .compile_error(instruction, span))
            }
        };
        let msb = field.len() - 1;
        if start_bit > msb {
            return Err(Error {
                definition: def.clone(),
                span: capture_span(&x, 2).start..capture_span(&x, 3).end,
                kind: Kind::BitRangeOutOfBounds {
                    upper_bound: start_bit,
                    msb,
                },
            }
            .compile_error(instruction, span));
        }
        // Replace the matched string with the corresponding bits
        let mut field = format!("0b{}", &field[msb - start_bit..=msb - end_bit]);
        // If the number is bigger than 32 bits, add the `n` postfix to mark it as a big integer
        if start_bit - end_bit + 1 > 32 {
            field.push('n');
        }
        def.replace_range(capture_span(&x, 0), &field);
    }

    // Replace occurrences of `Field.size`
    while let Some(x) = FIELD_SIZE.captures(&def) {
        let (_, [arg]) = x.extract();
        let arg_num = num(arg) - 1;
        // Get the user's argument expression
        let (value, _) = &args
            .get(arg_num)
            .ok_or_else(|| {
                Error {
                    definition: def.clone(),
                    span: capture_span(&x, 1),
                    kind: Kind::UnknownFieldNumber {
                        idx: arg_num + 1,
                        size: args.len(),
                    },
                }
                .compile_error(instruction, span)
            })?
            .value;
        // Calculate the size of the expression
        #[allow(clippy::cast_possible_truncation)]
        let size = match value.eval(ident_eval, mods)? {
            Number::Int(x) => x.bits() + 1,
            // If the result is a float, assume it is in single precision
            Number::Float { .. } => 32,
        };
        def.replace_range(capture_span(&x, 0), &size.to_string());
    }

    // Replace occurrences of `reg_name`
    while let Some(x) = REG_NAME.captures(&def) {
        let (_, [arg]) = x.extract();
        let arg_num = num(arg) - 1;
        // Get the user's argument expression
        let value = &args
            .get(arg_num)
            .ok_or_else(|| {
                Error {
                    definition: def.clone(),
                    span: capture_span(&x, 1),
                    kind: Kind::UnknownFieldNumber {
                        idx: arg_num + 1,
                        size: args.len(),
                    },
                }
                .compile_error(instruction, span)
            })?
            .value;
        let name = reg_name(value)?;
        def.replace_range(capture_span(&x, 0), &format!("\"{name}\""));
    }

    // Replace occurrences of `reg.pc` and update its value on the JS side
    if def.contains("reg.pc") {
        let code = format!("pc = {address} + 4");
        js::eval_fn(&code).expect("The code we are running should never fail");
        def = def.replace("reg.pc", "pc");
    }

    // Evaluate occurrences of `no_ret_op{}`
    while let Some(x) = NO_RET_OP.captures(&def) {
        let (_, [code]) = x.extract();
        js::eval_fn(code).map_err(|error| {
            Error {
                definition: def.clone(),
                span: capture_span(&x, 1),
                kind: Kind::EvaluationError(error),
            }
            .compile_error(instruction, span)
        })?;
        def.replace_range(capture_span(&x, 0), "");
    }

    // Evaluate and replace occurrences of `op{}`
    while let Some(x) = OP.captures(&def) {
        let (_, [code]) = x.extract();
        let result = js::eval_expr(code).map_err(|error| {
            Error {
                definition: def.clone(),
                span: capture_span(&x, 1),
                kind: Kind::EvaluationError(error),
            }
            .compile_error(instruction, span)
        })?;
        def.replace_range(capture_span(&x, 0), &js::to_string(result));
    }

    // Wrap instruction sequences in quotes and with a return statement, so we can treat the
    // definition code as the body of a function
    let mut start = 0;
    while let Some(x) = INSTRUCTIONS.captures_at(&def, start) {
        let (_, [instructions]) = x.extract();
        let replacement = format!("{{return \"{instructions}\"}}");
        let range = capture_span(&x, 0);
        start = range.start + replacement.len();
        def.replace_range(range, &replacement);
    }
    // If start isn't 0, there has been at least 1 replacement, meaning that the definition uses
    // code we must execute to get the replacement instruction sequence. Otherwise, the entire
    // string is the instruction sequence
    if start != 0 {
        let result = js::eval_fn(&def).map_err(|error| {
            Error {
                definition: def.clone(),
                span: 0..def.len(),
                kind: Kind::EvaluationError(error),
            }
            .compile_error(instruction, span)
        })?;
        def = js::to_string(result);
    }

    // Lex the instructions
    let (def, file_id) = ctx.file_cache.add(def, span);
    let parse_err = |error| {
        Error {
            definition: def.to_string(),
            span: 0..def.len(),
            kind: Kind::ParseError(error),
        }
        .compile_error(instruction, span)
    };
    let tokens = crate::parser::Instruction::lex(def, file_id).map_err(parse_err)?;
    // Process the resulting instruction sequence
    tokens
        .split(|(t, _)| *t == Token::Literal(';'))
        .filter(|t| !t.is_empty()) // split leaves an empty element after the last `;`
        .map(|tokens| {
            let (name, args) = crate::parser::Instruction::parse_name((def.len(), file_id), tokens)
                .map_err(parse_err)?;
            let span = chumsky::span::Span::new(file_id, name.1.start..args.1.end);
            // Parse the instruction
            let (inst_def, mut args) = super::parse_instruction(arch, name, args)?;
            // Replace the argument names that match those of the pseudoinstruction being expanded
            // with the values provided by the user
            for arg in &mut args {
                let value = &arg.value;
                if let Expr::Identifier((ident, _)) = &value.0 {
                    if let Some(pseudoinstruction_arg) = get_arg(ident) {
                        arg.value.0 = pseudoinstruction_arg.value.0.clone();
                    }
                }
            }
            Ok(((inst_def, args), span))
        })
        .collect()
}
