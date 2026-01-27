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

//! Module containing the definition of the assembly compiler
//!
//! The entry point for compiler code is the [`compile()`] function. Users are expected to parse
//! the code first to an AST with [`crate::parser::parse()`]

// # Compiler architecture
//
// ## General process
//
// 1. Process the AST and split the statements into instruction and data, processing non-data
//    directives (directives that don't add elements to the data segment of the compiled program)
// 2. Perform the 1st pass over the statements, determining the value of all labels defined
//    1. Process the data directives, determining the address of each element. Delay expression
//       evaluation on cases where labels are allowed until the 2nd pass to allow for forward
//       references
//    2. Process the instruction statements, determining the address of each real instruction. For
//       each instruction:
//       1. Find a valid definition to parse its arguments
//       2. Evaluate the instruction with the definition found, expanding pseudoinstructions
//          recursively if the definition corresponds with one. If the definition corresponds with a
//          real instruction, extract its arguments as expressions (delaying their evaluation until
//          the 2nd pass to allow for forward references)
// 3. Perform the 2nd pass over the statements, compiling the program (order doesn't matter)
//    * Evaluate the remaining arguments of data directives that were delayed during the 1st pass
//    * Translate instructions
//      1. Translate each field according to its type, evaluating its arguments
//      2. Add the `cop` fields' values to the binary instruction
//
// ## Reasons
//
// The hardest problem to solve that leads to the above process is answering the question:
//
// "How do we handle forward references?"
//
// The assembly code might have labels with forward references (i.e. used before they are
// defined in the assembly code. This is necessary for e.g. forward jumps). This means we can't
// just go through the statements one by one compiling them, since we might get to a point where
// the compilation might need to know the value of a label that's defined later in the code
//
// In order to solve this, we need to split the compilation in 2 passes:
//
// 1. Determine the values of all the labels without compiling anything, storing them on a symbol
//    table. Since the value of labels is the address in which the element (instruction or data
//    to add to the data segment) they point to is, this requires determining the address in which
//    each element will be stored. This, in turn, requires determining how much memory each element
//    will use, as the address of each element is the sum of the address of the start of the segment
//    and the memory used by the elements that appeared before
//
//    This allows the 2nd pass to know the values of all labels, regardless of where they are
//    defined relative to where they are used, solving the problem of forward references
// 2. Perform the actual compilation of the elements using the values of the labels obtained in the
//    1st pass. The processing of all statements during this pass is independent: the only global
//    information used is contained in the symbol table created during the 1st pass, and that
//    symbol table is never modified
//
// This approach, however, has some problems with pseudoinstructions (instructions that are allowed
// in the assembly code but aren't defined in the actual architecture, and thus need to be replaced
// with a sequence of instructions that implement an equivalent functionality):
//
// 1. The amount of memory used by a pseudoinstruction depends on the instruction sequence it gets
//    replaced with after it is expanded
// 2. Pseudoinstruction expansion might use the exact numeric value of its arguments to determine
//    the instruction sequence it will expand into, which means we might need to evaluate its
//    arguments during pseudoinstruction expansion. Different sequences might take different amounts
//    of space in memory (instruction sizes can change, and they might have different amounts of
//    instructions)
// 3. Instruction arguments might contain labels with forward references
// 4. Defining a label requires knowing how much memory has been used up to the element it points
//    to
//
// Problem 1 means we need to perform the pseudoinstruction expansion during the 1st pass, as
// that's when we need to determine the amount of space used by all elements. However, problems 2
// and 3 mean we need to be able to use forward references during pseudoinstruction expansion,
// which we can only do during the 2nd pass
//
// The core issue is that these problems create a cyclic dependency: expanding a pseudoinstruction
// might require knowing the address of elements that appear later in the assembly code, but those
// addresses might depend on how the pseudoinstruction is expanded. Thus, expanding a
// pseudoinstruction might require knowing how that pseudoinstruction should be expanded, which is
// impossible
//
// In order to solve this, we need to do one of these options:
//
// 1. Disallow forward references on pseudoinstruction expansions that need to know the exact
//    numeric value of their arguments
// 2. Add padding no-op instructions after the instruction sequence of a pseudoinstruction to
//    guarantee all possible expansions end up using the same amount of memory, allowing the
//    expansion to be performed during the 2nd pass
// 3. Rework the pseudoinstruction definition specification to disallow instructions from using the
//    exact numeric value of their arguments, only giving them access to their arguments by name and
//    to get estimates of their size. This allows to decouple the process of determining the
//    instruction sequence to replace the pseudoinstruction with from from the process of evaluating
//    the arguments of the resulting instructions. Thus, the former could be performed during the
//    1st pass while the later could be done during the 2nd pass
//
// Solution 3 is the best one, but requires a breaking change in the architecture specification
// that would make some current pseudoinstruction definitions invalid. Option 2 would be the
// simplest one that still allows forward references, but its implementation is extremely complex
// without requiring the addition of extra metadata to the pseudoinstruction definition: a
// pseudoinstruction that can be expanded into multiple different instruction sequences requires
// running `JS` code, which would be very hard to analyze to determine all possible instruction
// sequences it might result in
//
// This only leaves option 1, which is the solution currently being used. We can, however, be a
// littler clever with it to allow some forward references during pseudoinstruction expansions that
// need to know the exact numeric value of their arguments: pseudoinstruction expansions can only
// influence the address used by other instructions, so forward references are only a problem when
// they point to another instruction. Forward references to data elements aren't an issue as those
// are never in the same segment as the problematic pseudoinstructions
//
// We can take advantage of this by, during the 1st pass, processing all data elements before any
// instructions are processed. In order to achieve this, we must pre-process the assembly code to
// split it into instruction and data statements, and from there perform the 1st pass for data
// statements, the 1st pass for instructions, and finally the 2nd pass (the processing order doesn't
// matter, since the symbol table isn't modified). This process is precisely a summary of the full
// compilation process described above

use chumsky::span::Span as _;
use num_bigint::{BigInt, BigUint};
use regex::{NoExpand, Regex};

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use crate::architecture::{
    AlignmentType, Architecture, DirectiveAction, DirectiveData, DirectiveSegment, FieldType,
    FloatType, IntegerType, RegisterType, StringType,
};
use crate::parser::instruction::{ParsedArgs, ParsedArgument};
use crate::parser::{
    Data as DataNode, Expr, InstructionNode, Statement as StatementNode, Token, AST,
};
use crate::span::{Range, Span, Spanned, DEFAULT_SPAN};

mod label;
pub use label::{Label, Table as LabelTable};

pub mod error;
use error::{ArgumentType, SpannedErr};
pub use error::{Data as ErrorData, Error as CompileError, Kind as ErrorKind};

mod bit_field;
pub use bit_field::BitField;

mod section;
use section::Section;

mod integer;
pub use integer::Integer;

mod pseudoinstruction;
pub use pseudoinstruction::{Error as PseudoinstructionError, Kind as PseudoinstructionErrorKind};

mod file_cache;
pub use file_cache::{FileCache, FileID};

/* TODO:
*  - Combine `crate::parser::Error` with `crate::compiler::Error`
*  - Add `Vec<T>` to [`Section`] type
**/

/// Global compilation context
#[derive(Debug, Clone)]
pub struct Context<'arch> {
    /// Architecture used during the compilation
    pub arch: &'arch Architecture<'arch>,
    /// Labels defined
    pub label_table: LabelTable,
    /// Pseudoinstruction definitions expanded
    pub file_cache: FileCache,
}

/// Definition of an instruction, either a real instruction or a pseudoinstruction
#[derive(Debug)]
enum InstructionDefinition<'arch> {
    Real(&'arch crate::architecture::Instruction<'arch>),
    Pseudo(&'arch crate::architecture::Pseudoinstruction<'arch>),
}

/// Parse the arguments of an instruction according to any of its definitions. If the arguments
/// match the syntax of multiple definitions, one of them is selected according to the following
/// criteria:
///
/// * If there are multiple definitions in which the arguments fit, the first one is used. For
///   arguments whose size can't be known, it's assumed that they will fit
/// * If there are no definitions in which the arguments fit, the last one that parsed correctly is
///   used
///
/// # Parameters
///
/// * `arch`: architecture definition
/// * `name`: name of the instruction
/// * `args`: vector of tokens that form the instruction arguments
/// * `origin`: origin span of the instruction
///
/// # Errors
///
/// Errors if the instruction name doesn't exist, or if the arguments doesn't match the syntax of
/// any of the instruction definitions for this instruction name
fn parse_instruction<'a>(
    arch: &'a Architecture,
    name: Spanned<&str>,
    args: Spanned<&[Spanned<Token>]>,
) -> Result<(InstructionDefinition<'a>, ParsedArgs), ErrorData> {
    let mut possible_def = None;
    // Errors produced on each of the attempted parses
    let mut errs = Vec::new();
    for inst in arch.find_instructions(name.0) {
        match inst.syntax.parser.parse(args) {
            Ok(parsed_args) => {
                // Check if all the arguments fit in the current instruction definition
                let ok = parsed_args.iter().all(|arg| {
                    let field = &inst.syntax.fields[arg.field_idx];
                    let (value, span) = &arg.value;
                    let value = match field.r#type {
                        // If the field expects a number and the argument value is a number,
                        // evaluate the number. If any forward reference labels are used, we can't
                        // know the exact value of the expression, so don't try to evaluate
                        // expressions using labels
                        FieldType::Address
                        | FieldType::ImmSigned
                        | FieldType::ImmUnsigned
                        | FieldType::OffsetBytes
                        | FieldType::OffsetWords => match value
                            .eval_no_ident(&arch.modifiers)
                            .and_then(|x| BigInt::try_from(x).add_span(*span))
                        {
                            Ok(val) => val,
                            // If there was any error, assume the argument fits. This should be
                            // properly handled later when we can fully evaluate the expressions
                            Err(_) => return true,
                        },
                        // Otherwise, assume the argument fits to avoid circular dependencies
                        _ => return true,
                    };
                    // Check if it fits by trying to build an integer with the required size
                    Integer::build(
                        value,
                        field.range.size(),
                        None,
                        Some(matches!(
                            field.r#type,
                            FieldType::ImmSigned | FieldType::OffsetBytes | FieldType::OffsetWords
                        )),
                    )
                    .is_ok()
                });
                // If all arguments fit the current definition, use it as the correct one
                if ok {
                    return Ok((InstructionDefinition::Real(inst), parsed_args));
                }
                // Otherwise, store it in case this is the only matching definition
                possible_def = Some((inst, parsed_args));
            }
            Err(e) => errs.push((inst.syntax.parser.syntax().to_string(), e)),
        }
    }
    for inst in arch.find_pseudoinstructions(name.0) {
        match inst.syntax.parser.parse(args) {
            // If parsing is successful, assume this definition is the correct one and return it
            Ok(parsed_args) => return Ok((InstructionDefinition::Pseudo(inst), parsed_args)),
            Err(e) => errs.push((inst.syntax.parser.syntax().to_string(), e)),
        }
    }
    // None of the definitions matched perfectly. If there is a matching definition that failed due
    // to argument sizes, use it
    if let Some((def, args)) = possible_def {
        return Ok((InstructionDefinition::Real(def), args));
    }
    // Otherwise, return the appropriate error. If we didn't get any errors, we didn't find any
    // definitions for the instruction
    Err(if errs.is_empty() {
        ErrorKind::UnknownInstruction(name.0.to_owned()).add_span(name.1)
    } else {
        ErrorKind::IncorrectInstructionSyntax(errs).add_span(args.1)
    })
}

/// Processed instruction pending argument evaluation
#[derive(Debug, Clone)]
struct PendingInstruction<'arch> {
    /// Address of the instruction
    address: BigUint,
    /// Labels pointing to this instruction
    labels: Vec<String>,
    /// Span of the instruction in the user's assembly code
    user_span: Range,
    /// Span of the instruction in pseudoinstruction expansions
    span: Span,
    /// Instruction definition selected
    definition: &'arch crate::architecture::Instruction<'arch>,
    /// Arguments parsed for this instruction
    args: ParsedArgs,
}

/// Process instruction, recursively expanding it if the selected definition corresponds to a
/// pseudoinstruction
///
/// # Parameters
///
/// * `ctx`: compilation context to use
/// * `section`: memory section in which the instructions should be stored
/// * `pending_instructions`: vector in which to append the real instructions
/// * `instruction`: instruction definition with its arguments
/// * `span`: span of the instruction, range in the user's assembly code and span in
///   pseudoinstruction expansions
///
/// # Errors
///
/// Errors if there is not enough space for any of the final instructions, or if there is an error
/// while expanding a pseudoinstruction
fn process_instruction<'arch>(
    ctx: &mut Context<'arch>,
    section: &mut Section,
    pending_instructions: &mut Vec<PendingInstruction<'arch>>,
    instruction: (InstructionDefinition<'arch>, ParsedArgs),
    span: (Range, Span),
) -> Result<(), ErrorData> {
    let (user_span, span) = span;
    match instruction.0 {
        // Base case: we have a real instruction => push it to the parsed instructions normally
        InstructionDefinition::Real(definition) => {
            let word_size = BigUint::from(ctx.arch.word_size().div_ceil(8));
            let address = section
                .try_reserve(&(word_size * definition.nwords))
                .add_span(span)?;
            pending_instructions.push(PendingInstruction {
                labels: Vec::new(),
                args: instruction.1,
                address,
                user_span,
                span,
                definition,
            });
        }
        // Recursive case: we have a pseudoinstruction => expand it into multiple instructions and
        // process each of them recursively
        InstructionDefinition::Pseudo(def) => {
            let inst = (def, span);
            let instructions = pseudoinstruction::expand(ctx, section.get(), inst, &instruction.1)?;
            for (instruction, span) in instructions {
                process_instruction(
                    ctx,
                    section,
                    pending_instructions,
                    instruction,
                    (user_span.clone(), span),
                )?;
            }
        }
    }
    Ok(())
}

/// Compiled instruction
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Instruction {
    /// Address of the instruction
    pub address: BigUint,
    /// Labels pointing to this instruction
    pub labels: Vec<String>,
    /// Translated instruction to a simplified syntax
    pub loaded: String,
    /// Instruction encoded in binary
    pub binary: BitField,
    /// Span of the instruction in the assembly code
    pub user: Range,
}

/// Value to add to the data segment pending argument evaluation
#[derive(Debug, PartialEq, Clone)]
enum PendingValue {
    /// Integer value
    Integer(Spanned<Expr>, usize, IntegerType),
    /// Reserved space initialized to 0
    Space(BigUint),
    /// Padding added to align elements
    Padding(BigUint),
    /// Single precision floating point value
    Float(f32),
    /// Double precision floating point value
    Double(f64),
    /// UTF-8 string
    String {
        /// Byte sequence of the string, encoded in UTF-8
        data: String,
        /// Whether the string is terminated by a null byte
        null_terminated: bool,
    },
}

/// Value to add to the data segment
#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    /// Integer value
    Integer(Integer),
    /// Reserved space initialized to 0
    Space(BigUint),
    /// Padding added to align elements
    Padding(BigUint),
    /// Single precision floating point value
    Float(f32),
    /// Double precision floating point value
    Double(f64),
    /// UTF-8 string
    String {
        /// Byte sequence of the string, encoded in UTF-8
        data: String,
        /// Whether the string is terminated by a null byte
        null_terminated: bool,
    },
}

/// Compiled data segment element
#[derive(Debug, PartialEq, Clone)]
struct PendingData {
    /// Address of the element
    address: BigUint,
    /// Labels pointing to this data element
    labels: Vec<String>,
    /// Value of the data element
    value: PendingValue,
}

/// Compiled data segment element
#[derive(Debug, PartialEq, Clone)]
pub struct Data {
    /// Address of the element
    pub address: BigUint,
    /// Labels pointing to this data element
    pub labels: Vec<String>,
    /// Value of the data element
    pub value: Value,
}

impl DataNode {
    /// Convert the value to a string
    ///
    /// # Parameters
    ///
    /// * `span`: span of the value in the assembly code
    ///
    /// # Errors
    ///
    /// Errors if the value doesn't contain a string
    fn into_string(self, span: Span) -> Result<String, ErrorData> {
        match self {
            Self::String(s) => Ok(s),
            Self::Number(_) => Err(ErrorKind::IncorrectArgumentType {
                expected: ArgumentType::String,
                found: ArgumentType::Expression,
            }),
        }
        .add_span(span)
    }

    /// Convert the value to an expression
    ///
    /// # Parameters
    ///
    /// * `span`: span of the value in the assembly code
    ///
    /// # Errors
    ///
    /// Errors if the value doesn't contain an expression
    fn into_expr(self, span: Span) -> Result<Expr, ErrorData> {
        match self {
            Self::Number(expr) => Ok(expr),
            Self::String(_) => Err(ErrorKind::IncorrectArgumentType {
                expected: ArgumentType::Expression,
                found: ArgumentType::String,
            }),
        }
        .add_span(span)
    }
}

/// Compilation result
#[derive(Debug, PartialEq, Clone)]
pub struct CompiledCode {
    /// Symbol table for labels
    pub label_table: LabelTable,
    /// Table indicating which labels are global
    pub global_symbols: GlobalSymbols,
    /// Compiled instructions
    pub instructions: Vec<Instruction>,
    /// Compiled data elements
    pub data_memory: Vec<Data>,
}

/// Convert a vector of spanned elements to a vector of elements, leaving an empty vector
///
/// # Parameters
///
/// * `src`: source vector to take the data from
#[must_use]
fn take_spanned_vec<T>(src: &mut Vec<Spanned<T>>) -> Vec<T> {
    std::mem::take(src).into_iter().map(|x| x.0).collect()
}

/// Statement extracted from the AST
#[derive(Debug, PartialEq, Clone)]
struct Statement<T> {
    /// Labels attached to the node
    labels: Vec<Spanned<String>>,
    /// Whether the statement is in a kernel section (`true`) or not (`false`)
    kernel: bool,
    /// Statement of the node
    value: Spanned<T>,
}

/// Data values to add to the data segment
struct DataValue {
    /// Expected type of the values
    data_type: DirectiveData,
    /// Values to be added to the data segment
    values: Spanned<Vec<Spanned<DataNode>>>,
}

/// Vector of instruction statements
type Instructions = Vec<Statement<InstructionNode>>;
/// Vector of data directive statements
type DataDirectives = Vec<Statement<DataValue>>;
/// Global symbols table, indicating which symbols are global
pub type GlobalSymbols = HashSet<String>;

/// Split the vector of statements represented by the AST into a vector of instruction statements,
/// a vector of data directive statements, and a global symbols table
///
/// # Parameters
///
/// * `arch`: architecture definition
/// * `ast`: AST of the assembly code
///
/// # Errors
///
/// Errors if there is any error while processing any of the statements
fn split_statements(
    arch: &Architecture,
    ast: AST,
) -> Result<(Instructions, DataDirectives, GlobalSymbols), ErrorData> {
    // Section currently being processed
    let mut current_section: Option<Spanned<DirectiveSegment>> = None;
    // Resulting data directives vector
    let mut data_directives = Vec::new();
    // Resulting instructions vector
    let mut instructions = Vec::new();
    // Resulting global symbols table
    let mut global_symbols = HashSet::new();

    // For each statement in the assembly code
    for node in ast {
        match node.statement.0 {
            StatementNode::Directive(directive) => {
                let action = arch.find_directive(&directive.name.0).ok_or_else(|| {
                    ErrorKind::UnknownDirective(directive.name.0).add_span(directive.name.1)
                })?;
                // Execute the directive
                match action {
                    // No-op, ignore it and its arguments
                    DirectiveAction::Nop(_) => {}
                    // Change the current section
                    DirectiveAction::Segment(new_section) => {
                        ArgumentNumber::new(0, false).check(&directive.args)?;
                        current_section = Some((new_section, node.statement.1));
                    }
                    // Add new global symbols
                    DirectiveAction::GlobalSymbol(_) => {
                        ArgumentNumber::new(1, true).check(&directive.args)?;
                        for (label, span) in directive.args.0 {
                            // Extract the name from the argument, should be an identifier
                            let label = match label {
                                DataNode::Number(Expr::Identifier(label)) => label,
                                DataNode::Number(_) => Err(ErrorKind::IncorrectArgumentType {
                                    expected: ArgumentType::Identifier,
                                    found: ArgumentType::Expression,
                                }
                                .add_span(span))?,
                                DataNode::String(_) => Err(ErrorKind::IncorrectArgumentType {
                                    expected: ArgumentType::Identifier,
                                    found: ArgumentType::String,
                                }
                                .add_span(span))?,
                            };
                            global_symbols.insert(label.0);
                        }
                    }
                    // Add new data elements
                    DirectiveAction::Data(data_type) => match current_section {
                        // If the current section allows data, add it to the data directives vector
                        Some((section, _)) if !section.is_code() => {
                            let span = node.statement.1;
                            data_directives.push(Statement {
                                labels: node.labels,
                                value: (
                                    DataValue {
                                        data_type,
                                        values: directive.args,
                                    },
                                    span,
                                ),
                                kernel: section == DirectiveSegment::KernelData,
                            });
                        }
                        // Otherwise, the statement is in the wrong section
                        _ => {
                            return Err(ErrorKind::UnallowedStatementType {
                                section: current_section,
                                found: DirectiveSegment::Data,
                            }
                            .add_span(node.statement.1));
                        }
                    },
                }
            }
            StatementNode::Instruction(instruction) => match &current_section {
                // If the current section allows code, add it to the instructions vector
                Some((section, _)) if section.is_code() => instructions.push(Statement {
                    labels: node.labels,
                    value: (instruction, node.statement.1),
                    kernel: *section == DirectiveSegment::KernelCode,
                }),
                // Otherwise, the statement is in the wrong section
                _ => {
                    return Err(ErrorKind::UnallowedStatementType {
                        section: current_section,
                        found: DirectiveSegment::Code,
                    }
                    .add_span(node.statement.1));
                }
            },
        }
    }
    Ok((instructions, data_directives, global_symbols))
}

/// Amount of arguments expected by a directive
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArgumentNumber {
    /// Minimum expected amount
    pub amount: usize,
    /// Whether it's allowed to use more arguments
    pub at_least: bool,
}

impl ArgumentNumber {
    /// Creates a new [`ArgumentNumber`]
    ///
    /// # Parameters
    ///
    /// * `amount`: minimum expected amount of arguments
    /// * `at_least`: whether it's allowed to use more arguments
    #[must_use]
    pub const fn new(amount: usize, at_least: bool) -> Self {
        Self { amount, at_least }
    }

    /// Checks if the argument number matches the amount specified
    ///
    /// # Parameters
    ///
    /// * `args`: arguments found
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::IncorrectDirectiveArgumentNumber`] if the amount of arguments doesn't
    /// match the specified amount
    fn check<T>(self, args: &Spanned<Vec<T>>) -> Result<(), ErrorData> {
        let len = args.0.len();
        if len < self.amount || (!self.at_least && len != self.amount) {
            return Err(ErrorKind::IncorrectDirectiveArgumentNumber {
                expected: self,
                found: args.0.len(),
            }
            .add_span(args.1));
        }
        Ok(())
    }

    /// Checks that the argument number is exactly one, and returns that argument
    ///
    /// # Parameters
    ///
    /// * `args`: arguments found
    ///
    /// # Errors
    ///
    /// Errors if the amount of arguments isn't exactly 1
    fn exactly_one<T>(args: Spanned<Vec<T>>) -> Result<T, ErrorData> {
        Self::new(1, false).check(&args)?;
        let mut iter = args.0.into_iter();
        Ok(iter.next().expect("We should have exactly 1 value"))
    }
}

/// Combines a kernel section with a user section into a single memory vector. It is a logical
/// error for the sections to overlap
///
/// # Parameters
///
/// * `kernel`: kernel section with its vector of elements
/// * `user`: user section with its vector of elements
#[must_use]
fn combine_sections<T>(kernel: (Section, Vec<T>), user: (Section, Vec<T>)) -> Vec<T> {
    // Sort the sections
    let (mem1, mem2) = if kernel.0.get() <= user.0.get() {
        (kernel.1, user.1)
    } else {
        (user.1, kernel.1)
    };
    // Chain the elements
    mem1.into_iter().chain(mem2).collect()
}

/// Performs the 1st pass of the compilation of the data directive statements
///
/// # Parameters
///
/// * `ctx`: compilation context to use
/// * `elements`: data directives to compile
///
/// # Errors
///
/// Errors if there is any problem processing the data elements
#[allow(clippy::too_many_lines)]
fn compile_data(
    ctx: &mut Context,
    elements: Vec<Statement<DataValue>>,
) -> Result<Vec<PendingData>, ErrorData> {
    let size = elements.len();
    // User data section
    let mut user = (
        Section::new("Data", Some(ctx.arch.data_section())),
        Vec::with_capacity(size),
    );
    // Kernel data section
    let mut kernel = (
        Section::new("KernelData", ctx.arch.kernel_data_section()),
        Vec::with_capacity(size),
    );
    let word_size_bytes = ctx.arch.word_size().div_ceil(8);

    for data_directive in elements {
        let mut labels = data_directive.labels;
        // Get the corresponding section according to where the element should be placed
        let (section, memory) = if data_directive.kernel {
            &mut kernel
        } else {
            &mut user
        };
        // Add the labels to the label table
        for (label, span) in &labels {
            ctx.label_table
                .insert(label.to_owned(), *span, section.get().clone())?;
        }
        let (statement, statement_span) = data_directive.value;
        let args = statement.values;
        // Process the directive according to its type
        match statement.data_type {
            DirectiveData::Alignment(align_type) => {
                let (value, span) = ArgumentNumber::exactly_one(args)?;
                let value = value.into_expr(span)?.eval_no_ident(&ctx.arch.modifiers)?;
                let value = BigUint::try_from(value).add_span(span)?;
                // Calculate the align size in bytes
                let align = match align_type {
                    // Calculate 2^argument
                    AlignmentType::Exponential => {
                        // Convert the input to a fixed sized int, as shifts aren't implemented
                        // with big ints as the second argument. `u16` gives more than enough size
                        // for normal usage, anything bigger will likely be user error
                        let value = u16::try_from(value).map_err(|e| {
                            ErrorKind::IntegerOutOfRange(
                                e.into_original().into(),
                                0.into()..=u16::MAX.into(),
                            )
                            .add_span(span)
                        })?;
                        BigUint::from(1u8) << value
                    }
                    // If the argument is already in bytes, we don't need to do anything
                    AlignmentType::Byte => value,
                };
                let (start, size) = section.try_align(&align).add_span(statement_span)?;
                // If we needed to add any padding, store it in the result vector
                if size != BigUint::ZERO {
                    memory.push(PendingData {
                        address: start,
                        labels: take_spanned_vec(&mut labels),
                        value: PendingValue::Padding(size),
                    });
                }
            }
            DirectiveData::Space(size) => {
                let (value, span) = ArgumentNumber::exactly_one(args)?;
                let value = value.into_expr(span)?.eval_no_ident(&ctx.arch.modifiers)?;
                let size = BigUint::try_from(value).add_span(span)? * size;
                memory.push(PendingData {
                    address: section.try_reserve(&size).add_span(span)?,
                    labels: take_spanned_vec(&mut labels),
                    value: PendingValue::Space(size),
                });
            }
            DirectiveData::Int(size, int_type) => {
                ArgumentNumber::new(1, true).check(&args)?;
                for (value, span) in args.0 {
                    let value = value.into_expr(span)?;
                    memory.push(PendingData {
                        address: section
                            .try_reserve_aligned(&size.into(), word_size_bytes)
                            .add_span(span)?,
                        labels: take_spanned_vec(&mut labels),
                        value: PendingValue::Integer((value, span), size, int_type),
                    });
                }
            }
            DirectiveData::Float(float_type) => {
                ArgumentNumber::new(1, true).check(&args)?;
                for (value, span) in args.0 {
                    let value = value.into_expr(span)?.eval_no_ident(&ctx.arch.modifiers)?;
                    let value = value.to_f64();
                    // We intentionally want to truncate the number from f64 to f32 if the user
                    // asked for an f32
                    #[allow(clippy::cast_possible_truncation)]
                    let (value, size) = match float_type {
                        FloatType::Float => (PendingValue::Float(value as f32), 4u8),
                        FloatType::Double => (PendingValue::Double(value), 8),
                    };
                    memory.push(PendingData {
                        address: section
                            .try_reserve_aligned(&size.into(), word_size_bytes)
                            .add_span(span)?,
                        labels: take_spanned_vec(&mut labels),
                        value,
                    });
                }
            }
            DirectiveData::String(str_type) => {
                ArgumentNumber::new(1, true).check(&args)?;
                for (value, span) in args.0 {
                    let data = value.into_string(span)?;
                    let null_terminated = str_type == StringType::AsciiNullEnd;
                    let size = BigUint::from(data.len()) + u8::from(null_terminated);
                    memory.push(PendingData {
                        address: section.try_reserve(&size).add_span(span)?,
                        labels: take_spanned_vec(&mut labels),
                        value: PendingValue::String {
                            data,
                            null_terminated,
                        },
                    });
                }
            }
        }
    }
    Ok(combine_sections(kernel, user))
}

/// Performs the 1st pass of the compilation of the instruction statements
///
/// # Parameters
///
/// * `ctx`: compilation context to use
/// * `elements`: instructions to compile
/// * `reserved_offset`: amount of addresses reserved by the library
///
/// # Errors
///
/// Errors if there is any problem processing the data elements
fn compile_instructions<'arch>(
    ctx: &mut Context<'arch>,
    instructions: Vec<Statement<InstructionNode>>,
    reserved_offset: &BigUint,
) -> Result<Vec<PendingInstruction<'arch>>, ErrorData> {
    let size = instructions.len();
    // User data section
    let mut user = (
        Section::new("Instructions", Some(ctx.arch.code_section())),
        Vec::with_capacity(size),
    );
    // Kernel data section
    let mut kernel = (
        Section::new("KernelInstructions", ctx.arch.kernel_code_section()),
        Vec::with_capacity(size),
    );
    // Reserve space in the user section for the library instructions
    user.0.try_reserve(reserved_offset).add_span(DEFAULT_SPAN)?;

    for mut instruction in instructions {
        let (name, name_span) = instruction.value.0.name;
        let (args, args_span) = instruction.value.0.args;
        let span = instruction.value.1;
        // Get the corresponding section according to where the element should be placed
        let (section, memory) = if instruction.kernel {
            &mut kernel
        } else {
            &mut user
        };
        // Add the labels to the label table
        for (label, span) in &instruction.labels {
            ctx.label_table
                .insert(label.clone(), *span, section.get().clone())?;
        }
        // Parse the instruction, finding a valid definition to use for the compilation
        let parsed_instruction =
            parse_instruction(ctx.arch, (&name, name_span), (&args, args_span))?;
        // Store the next index, so we can do a small post-processing to the processed instructions
        let first_idx = memory.len();
        process_instruction(
            ctx,
            section,
            memory,
            parsed_instruction,
            (span.into_range(), span),
        )?;
        let mut iter = memory[first_idx..].iter_mut().fuse();
        // Add the labels attached to the instruction in the assembly code to the first generated
        // instruction
        if let Some(inst) = iter.next() {
            inst.labels = take_spanned_vec(&mut instruction.labels);
        }
        // Remove the user span from all generated instructions except the first, so the source
        // isn't repeated in the UI
        for inst in iter {
            inst.user_span = 0..0;
        }
    }
    Ok(combine_sections(kernel, user))
}

/// Evaluates an identifier used as a label within an expression
///
/// # Parameters
///
/// * `label_table`: symbol table for labels
/// * `address`: address in which the value is being compiled into
/// * `label`: identifier to evaluate
///
/// # Errors
///
/// Returns a [`ErrorKind::UnknownLabel`] if the label isn't defined
fn label_eval(
    label_table: &LabelTable,
    address: &BigUint,
    label: &str,
) -> Result<BigInt, ErrorKind> {
    // The identifier `.` should always correspond to the address in which
    // the value is being compiled into. Otherwise, try to find the label
    // name in the label table
    let value = if label == "." {
        address
    } else {
        label_table
            .get(label)
            .ok_or_else(|| ErrorKind::UnknownLabel(label.to_owned()))?
            .address()
    };
    Ok(value.clone().into())
}

/// Checks that the main label (entry point of an executable program) isn't misplaced
///
/// # Parameters
///
/// * `ctx`: compilation context to use
/// * `library`: whether to compile the assembly code as a library (`true`) or executable (`false`)
/// * `eof`: byte index of the end of the last instruction statement
///
/// # Errors
///
/// Returns [`ErrorKind::MissingMainLabel`], [`ErrorKind::MainInLibrary`], or
/// [`ErrorKind::MainOutsideCode`] if the main label is misplaced
fn check_main_location(ctx: &Context, library: bool, eof: usize) -> Result<(), ErrorData> {
    let add_main_span =
        |e: ErrorKind, main: &Label| e.add_span(main.span().unwrap_or(DEFAULT_SPAN));
    match (ctx.label_table.get(ctx.arch.main_label()), library) {
        // Main label wasn't used but we aren't compiling a library => main is missing
        (None, false) => {
            Err(ErrorKind::MissingMainLabel.add_span(Span::new(FileID::SRC, eof..eof)))
        }
        // Main label was used but we are compiling a library => main shouldn't be used
        (Some(main), true) => Err(add_main_span(ErrorKind::MainInLibrary, main)),
        // Main label was used and we aren't compiling a library, but it doesn't point to an
        // instruction => main is misplaced
        (Some(main), false) if !ctx.arch.code_section().contains(main.address()) => {
            Err(add_main_span(ErrorKind::MainOutsideCode, main))
        }
        // Otherwise, the main label is correctly placed
        _ => Ok(()),
    }
}

/// Evaluates the argument of an instruction, returning its value as a number and as text
///
/// # Parameters
///
/// * `ctx`: compilation context to use
/// * `address`: address of the instruction
/// * `definition`: definition of the instruction
/// * `arg`: expression containing the argument to evaluate
///
/// # Errors
///
/// Errors if there is any problem evaluating the argument
fn evaluate_instruction_field(
    ctx: &Context,
    address: &BigUint,
    def: &crate::architecture::Instruction,
    arg: ParsedArgument,
) -> Result<(BigInt, String), ErrorData> {
    let field = &def.syntax.fields[arg.field_idx];
    Ok(match &field.r#type {
        FieldType::Cop { .. } => unreachable!("Cop shouldn't be used in instruction args"),
        FieldType::Co => (def.co.0.clone().into(), def.name.to_string()),
        // Numeric fields
        FieldType::Address
        | FieldType::ImmSigned
        | FieldType::ImmUnsigned
        | FieldType::OffsetBytes
        | FieldType::OffsetWords => {
            let label_eval = |label: &str| {
                let value = label_eval(&ctx.label_table, address, label)?;
                // Function to calculate the offset between a given address and the
                // address in which the value is being compiled into
                let offset = |x| x - BigInt::from(address.clone());
                Ok(match field.r#type {
                    FieldType::OffsetWords => offset(value) / ctx.arch.word_size().div_ceil(8),
                    FieldType::OffsetBytes => offset(value),
                    _ => value,
                })
            };
            let value = arg.value.0.eval(label_eval, &ctx.arch.modifiers)?;
            let value = BigInt::try_from(value).add_span(arg.value.1)?;
            // Remove the least significant bits according to the padding specified by
            // the field
            let padding = field.range.padding();
            let value = (value >> padding) << padding;
            let value_str = value.to_string();
            (value, value_str)
        }
        // Register fields
        FieldType::IntReg
        | FieldType::CtrlReg
        | FieldType::SingleFPReg
        | FieldType::DoubleFPReg => {
            // Get the name of the register. We only allow identifiers and integers to
            // be used as register names
            let name = match arg.value.0 {
                Expr::Identifier((name, _)) => name,
                Expr::Integer(x) => x.to_string(),
                _ => {
                    return Err(ErrorKind::IncorrectArgumentType {
                        expected: ArgumentType::RegisterName,
                        found: ArgumentType::Expression,
                    }
                    .add_span(arg.value.1))
                }
            };
            // Convert the generic field type to an specific register type
            let file_type = match field.r#type {
                FieldType::IntReg => RegisterType::Int,
                FieldType::CtrlReg => RegisterType::Ctrl,
                FieldType::SingleFPReg => RegisterType::Float(FloatType::Float),
                FieldType::DoubleFPReg => RegisterType::Float(FloatType::Double),
                _ => unreachable!("We already matched one of these variants"),
            };
            // Find the register files with the requested type, and verify that at
            // least one file is found
            let mut files = ctx.arch.find_reg_files(file_type).peekable();
            files
                .peek()
                .ok_or_else(|| ErrorKind::UnknownRegisterFile(file_type).add_span(arg.value.1))?;
            let case = ctx.arch.config.sensitive_register_name;
            // Find the register with the given name
            let (reg, name) = files
                .find_map(|file| file.find_register(&name, case))
                .ok_or_else(|| {
                    ErrorKind::UnknownRegister {
                        name: name.clone(),
                        file: file_type,
                    }
                    .add_span(arg.value.1)
                })?;
            (reg.encoding.0.clone().into(), name.to_string())
        }
        // Enumerated fields
        FieldType::Enum { enum_name } => {
            // Find the definition of the enum
            let enum_def = ctx.arch.enums.get(enum_name);
            let enum_def = enum_def
                .ok_or_else(|| ErrorKind::UnknownEnumType((*enum_name).to_string()))
                .add_span(arg.value.1)?;
            // Get the identifier used
            let Expr::Identifier((name, _)) = arg.value.0 else {
                return Err(ErrorKind::IncorrectArgumentType {
                    expected: ArgumentType::Identifier,
                    found: ArgumentType::Expression,
                }
                .add_span(arg.value.1));
            };
            // Map the identifier to its value according to the definition of the enum
            let Some(value) = enum_def.get(name.as_str()) else {
                return Err(ErrorKind::UnknownEnumValue {
                    value: name,
                    enum_name: (*enum_name).to_string(),
                })
                .add_span(arg.value.1);
            };
            (value.0.clone().into(), name)
        }
    })
}

/// Translates the data of an instruction to the final result of the compilation, performing the
/// 2nd pass over the instructions
///
/// # Parameters
///
/// * `ctx`: compilation context to use
/// * `inst`: instruction to translate
///
/// # Errors
///
/// Errors if there is any problem translating the instruction
fn translate_instruction(
    ctx: &Context,
    inst: PendingInstruction,
) -> Result<Instruction, ErrorData> {
    // Regex for replacement templates in the translation spec of instructions
    // Surround the argument placeholder pattern (`[fF]([0-9]+)`) with word boundary assertions to
    // make sure we match a full identifier rather than part of one
    // SEE: https://docs.rs/regex/latest/regex/#empty-matches
    static RE: LazyLock<Regex> = crate::regex!(r"\b{start-half}[fF]([0-9]+)\b{end-half}");
    static FIELD: LazyLock<Regex> = crate::regex!("\0([0-9]+)");
    let def = inst.definition;
    let mut binary_instruction = BitField::new(ctx.arch.word_size().saturating_mul(def.nwords));
    // Replace the field placeholders in the translation spec with `\0N`. Since null bytes
    // shouldn't appear in the translation spec/register names, this avoids issues when a
    // placeholder is replaced with a register name with the same format as another placeholder
    let mut translated_instruction = RE.replace_all(def.syntax.output_syntax, "\0$1").to_string();
    for arg in inst.args {
        let field = &def.syntax.fields[arg.field_idx];
        let span = arg.value.1;
        let (value, value_str) = evaluate_instruction_field(ctx, &inst.address, def, arg)?;
        // Update the binary/translated instruction using the values obtained
        let signed = matches!(
            field.r#type,
            FieldType::ImmSigned | FieldType::OffsetBytes | FieldType::OffsetWords
        );
        binary_instruction
            .replace(&field.range, value, signed)
            .add_span(span)?;
        translated_instruction = FIELD
            .replace(&translated_instruction, NoExpand(&value_str))
            .to_string();
    }
    // Add the `Cop` fields' value to the binary instruction
    let fields = def.syntax.fields.iter();
    for (range, value) in fields.filter_map(|field| match &field.r#type {
        FieldType::Cop { value } => Some((&field.range, value)),
        _ => None,
    }) {
        binary_instruction
            .replace(range, value.0.clone().into(), false)
            .add_span(inst.span)?;
    }
    Ok(Instruction {
        labels: inst.labels,
        address: inst.address,
        binary: binary_instruction,
        loaded: translated_instruction,
        user: inst.user_span,
    })
}

/// Translates the data of a data element to the final result of the compilation, performing the
/// 2nd pass over the data elements
///
/// # Parameters
///
/// * `ctx`: compilation context to use
/// * `data`: data element to translate
///
/// # Errors
///
/// Errors if there is any problem translating the instruction
fn translate_data(ctx: &Context, data: PendingData) -> Result<Data, ErrorData> {
    Ok(Data {
        labels: data.labels,
        value: match data.value {
            // Evaluate the expression used as value for integer directives
            PendingValue::Integer((value, span), size, int_type) => {
                let (label_table, mods) = (&ctx.label_table, &ctx.arch.modifiers);
                let value =
                    value.eval(|label| label_eval(label_table, &data.address, label), mods)?;
                let value = BigInt::try_from(value).add_span(span)?;
                let int = Integer::build(value, size.saturating_mul(8), Some(int_type), None);
                Value::Integer(int.add_span(span)?)
            }
            // Copy the data from the other types of data directives
            PendingValue::Space(x) => Value::Space(x),
            PendingValue::Padding(x) => Value::Padding(x),
            PendingValue::Float(x) => Value::Float(x),
            PendingValue::Double(x) => Value::Double(x),
            PendingValue::String {
                data,
                null_terminated,
            } => Value::String {
                data,
                null_terminated,
            },
        },
        address: data.address,
    })
}

/// Main function handling the compilation
///
/// # Parameters
///
/// * `ctx`: compilation context to use
/// * `ast`: AST of the assembly code
/// * `reserved_offset`: amount of addresses reserved for the instructions of the library used
/// * `library`: whether to compile the assembly code as a library (`true`) or executable (`false`)
///
/// # Errors
///
/// Errors if there is any problem compiling the assembly code
fn compile_inner(
    ctx: &mut Context,
    ast: AST,
    reserved_offset: &BigUint,
    library: bool,
) -> Result<(GlobalSymbols, Vec<Instruction>, Vec<Data>), ErrorData> {
    // Split the statements into different sections
    let (instructions, data_directives, global_symbols) = split_statements(ctx.arch, ast)?;
    // Compile each of the sections
    let instruction_eof = instructions.last().map_or(0, |inst| inst.value.1.end);
    let pending_data = compile_data(ctx, data_directives)?;
    let pending_instructions = compile_instructions(ctx, instructions, reserved_offset)?;
    check_main_location(ctx, library, instruction_eof)?;
    // Perform the 2nd pass of instruction processing
    let instructions = pending_instructions
        .into_iter()
        .map(|inst| translate_instruction(ctx, inst))
        .collect::<Result<Vec<_>, _>>()?;
    // Perform the 2nd pass of data directives processing
    let data_memory = pending_data
        .into_iter()
        .map(|data| translate_data(ctx, data))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((global_symbols, instructions, data_memory))
}

/// Compiles an AST obtained from an assembly code into a set of instructions, data elements, and a
/// symbol table with the data from each label used
///
/// # Parameters
///
/// * `arch`: architecture definition
/// * `ast`: AST of the assembly code
/// * `reserved_offset`: amount of addresses reserved for the instructions of the library used
/// * `labels`: global labels defined by the library used
/// * `library`: whether to compile the assembly code as a library (`true`) or executable (`false`)
///
/// # Errors
///
/// Errors if there is any problem compiling the assembly code
pub fn compile<'arch, S: std::hash::BuildHasher>(
    arch: &'arch Architecture,
    ast: AST,
    reserved_offset: &BigUint,
    labels: HashMap<String, BigUint, S>,
    library: bool,
) -> Result<CompiledCode, CompileError<'arch>> {
    let mut ctx = Context {
        arch,
        label_table: LabelTable::from(labels),
        file_cache: FileCache::default(),
    };
    // Wrap the result of the inner compilation function. We need to wrap the internal compilation
    // function to add extra metadata added to all error types that isn't directly related with
    // the errors (like the architecture definition and the label table)
    match compile_inner(&mut ctx, ast, reserved_offset, library) {
        Ok((global_symbols, instructions, data_memory)) => Ok(CompiledCode {
            label_table: ctx.label_table,
            global_symbols,
            instructions,
            data_memory,
        }),
        Err(error) => Err(CompileError { ctx, error }),
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod test {
    use super::*;
    use crate::architecture::{Architecture, BitRange, IntegerType, NonEmptyRangeInclusive};
    use crate::span::test::*;

    fn compile_with(
        src: &str,
        reserved_offset: &BigUint,
        labels: HashMap<String, BigUint>,
        library: bool,
    ) -> Result<CompiledCode, ErrorData> {
        let arch = Architecture::from_json(include_str!("../tests/architecture.json")).unwrap();
        let ast = crate::parser::parse(arch.comment_prefix(), src).unwrap();
        super::compile(&arch, ast, reserved_offset, labels, library).map_err(|e| e.error)
    }

    fn compile_arch(src: &str, arch: &str) -> Result<CompiledCode, ErrorData> {
        let arch = Architecture::from_json(arch).unwrap();
        let ast = crate::parser::parse(arch.comment_prefix(), src).unwrap();
        super::compile(&arch, ast, &BigUint::ZERO, HashMap::new(), false).map_err(|e| e.error)
    }

    fn compile(src: &str) -> Result<CompiledCode, ErrorData> {
        compile_with(src, &BigUint::ZERO, HashMap::new(), false)
    }

    #[must_use]
    fn label_table(labels: impl IntoIterator<Item = (&'static str, u64, Range)>) -> LabelTable {
        let mut tbl = LabelTable::default();
        for v in labels {
            tbl.insert(v.0.into(), v.2.span(), v.1.into()).unwrap();
        }
        tbl
    }

    #[must_use]
    fn bitfield(bits: &str) -> BitField {
        let mut field = BitField::new(bits.len());
        for (i, c) in bits.chars().enumerate() {
            if c == '1' {
                let i = bits.len() - i - 1;
                let ranges = vec![NonEmptyRangeInclusive::<usize>::build(i, i).unwrap()];
                field
                    .replace(&BitRange::build(ranges, 0).unwrap(), 1.into(), false)
                    .unwrap();
            }
        }
        field
    }

    #[must_use]
    fn inst(address: u64, labels: &[&str], loaded: &str, binary: &str, user: Range) -> Instruction {
        Instruction {
            address: address.into(),
            labels: labels.iter().map(|&x| x.to_owned()).collect(),
            loaded: loaded.into(),
            binary: bitfield(binary),
            user,
        }
    }

    static NOP_BINARY: &str = "11110000000000000000000001111111";

    #[must_use]
    fn main_nop(span: Range) -> Instruction {
        inst(0, &["main"], "nop", NOP_BINARY, span)
    }

    #[must_use]
    fn data(address: u64, labels: &[&str], value: Value) -> Data {
        Data {
            address: address.into(),
            value,
            labels: labels.iter().map(|&x| x.to_owned()).collect(),
        }
    }

    #[must_use]
    fn int_val(x: i64, size: usize, ty: IntegerType) -> Value {
        Value::Integer(Integer::build(x.into(), size, Some(ty), None).unwrap())
    }

    #[test]
    fn nop() {
        // Minimal
        let x = compile(".text\nmain: nop").unwrap();
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(x.instructions, vec![main_nop(12..15)]);
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
        // 2 instructions
        let x = compile(".text\nmain: nop\nnop").unwrap();
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![
                main_nop(12..15),
                inst(4, &[], "nop", "11110000000000000000000001111111", 16..19)
            ]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_multiword() {
        let x = compile(".text\nmain: nop2\nnop").unwrap();
        let binary = "1001000000000000000000000000000000000000000000000000000001000001";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![
                inst(0, &["main"], "nop2", binary, 12..16),
                inst(8, &[], "nop", "11110000000000000000000001111111", 17..20),
            ]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_multiple_defs() {
        // Definition 1
        let x = compile(".text\nmain: multi 15").unwrap();
        let binary = "11110000000000000000000001110011";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "multi 15", binary, 12..20)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
        // Definition 2
        let x = compile(".text\nmain: multi $").unwrap();
        let binary = "00000000000000000000000001011101";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "multi $", binary, 12..19)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
        // Definition 3
        let x = compile(".text\nmain: multi 17").unwrap();
        let binary = "10001000000000000000000001000001";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "multi 17", binary, 12..20)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_fields_regs() {
        // Simple
        let x = compile(".text\nmain: reg ctrl1, x2, fs1, ft2").unwrap();
        let binary = "01001000000000000000000000010010";
        let result = "reg ctrl1, x2, fs1, ft2";
        let tbl = label_table([("main", 0, 6..11)]);
        assert_eq!(x.label_table, tbl);
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], result, binary, 12..35)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
        // Aliases
        let x = compile(".text\nmain: reg ctrl1, two, f1, Field2").unwrap();
        assert_eq!(x.label_table, tbl);
        let instruction = "reg ctrl1, two, f1, Field2";
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], instruction, binary, 12..38)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
        // Number aliases
        let x = compile(".text\nmain: reg ctrl1, 2, fs1, ft2").unwrap();
        assert_eq!(x.label_table, tbl);
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "reg ctrl1, 2, fs1, ft2", binary, 12..34)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
        // Case insensitive names (when disabled in the architecture) get replaced with the name
        // defined in the architecture
        let arch = include_str!("../tests/architecture2.json");
        let x = compile_arch(".text\nmain: int zero", arch).unwrap();
        let binary = "00000000000000000000000001111111";
        assert_eq!(x, compile_arch(".text\nmain: int ZERO", arch).unwrap());
        assert_eq!(x, compile_arch(".text\nmain: int zErO", arch).unwrap());
        assert_eq!(x.label_table, label_table([("main", 32, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(32, &["main"], "int ZeRo", binary, 12..20)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_fields_immediate() {
        let x = compile(".text\nmain: imm -7, 255, 11").unwrap();
        let binary = "00100100000000000010110111111110";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "imm -7, 255, 11", binary, 12..27)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_fields_immediate_labels() {
        let x = compile(".text\nmain: nop\na: imm a, c, b\nb: nop\n.data\nc: .zero 1").unwrap();
        let binary = "00010000000000000010000000100000";
        assert_eq!(
            x.label_table,
            label_table([
                ("main", 0, 6..11),
                ("a", 4, 16..18),
                ("b", 8, 31..33),
                ("c", 16, 44..46)
            ])
        );
        assert_eq!(
            x.instructions,
            vec![
                main_nop(12..15),
                inst(4, &["a"], "imm 4, 16, 8", binary, 19..30),
                inst(8, &["b"], "nop", NOP_BINARY, 34..37),
            ]
        );
        assert_eq!(
            x.data_memory,
            vec![data(16, &["c"], Value::Space(1u8.into()))]
        );
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_fields_offsets_aligned() {
        let x = compile(".text\nmain: off 7, -8").unwrap();
        let binary = "01110000000000000000000000001000";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "off 7, -8", binary, 12..21)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_fields_offsets_aligned_labels() {
        let x = compile(".text\nmain: nop\noff main, main").unwrap();
        let binary = "11000000000000000000000000001111";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![main_nop(12..15), inst(4, &[], "off -4, -1", binary, 16..30),]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());

        let x = compile(".text\na: nop\noff main, main\nmain: nop").unwrap();
        let binary = "01000000000000000000000000000001";
        assert_eq!(
            x.label_table,
            label_table([("a", 0, 6..8), ("main", 8, 28..33)])
        );
        assert_eq!(
            x.instructions,
            vec![
                inst(0, &["a"], "nop", NOP_BINARY, 9..12),
                inst(4, &[], "off 4, 1", binary, 13..27),
                inst(8, &["main"], "nop", NOP_BINARY, 34..37),
            ]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_fields_offsets_unaligned() {
        let x = compile(".text\nmain: off 6, 7").unwrap();
        let binary = "01100000000000000000000000000111";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "off 6, 7", binary, 12..20)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_fields_offsets_unaligned_labels() {
        let x = compile(".text\nmain: off 1, a\n.data\n.zero 1\na: .zero 1").unwrap();
        let binary = "00010000000000000000000000000100";
        assert_eq!(
            x.label_table,
            label_table([("main", 0, 6..11), ("a", 17, 35..37)])
        );
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "off 1, 4", binary, 12..20)]
        );
        assert_eq!(
            x.data_memory,
            vec![
                data(16, &[], Value::Space(1u8.into())),
                data(17, &["a"], Value::Space(1u8.into()))
            ]
        );
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_fields_enums() {
        let x = compile(".text\nmain: enum a, b, value, last").unwrap();
        let binary = "01010000000000011111110110000101";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "enum a, b, value, last", binary, 12..34)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_fields_padding() {
        let x = compile(".text\nmain: pad 15, 4").unwrap();
        let binary = "01100000000000011111010000000100";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "pad 12, 4", binary, 12..21)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());

        let x = compile(".text\nmain: pad -15, -1").unwrap();
        let binary = "10000000000000011111010000111100";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "pad -16, -4", binary, 12..23)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn instruction_fields_literals() {
        let x = compile(".text\nmain: lit F1a, aF1, 3").unwrap();
        let binary = "00000000000000011110000000000011";
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(0, &["main"], "lit F1a, aF1, 3", binary, 12..27)]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn kernel_text() {
        let x = compile(".text\nmain: nop\n.ktext\nfoo: nop").unwrap();
        assert_eq!(
            x.label_table,
            label_table([("main", 0, 6..11), ("foo", 32, 23..27)])
        );
        assert_eq!(
            x.instructions,
            vec![
                main_nop(12..15),
                inst(32, &["foo"], "nop", NOP_BINARY, 28..31),
            ]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());

        let x = compile(".ktext\nfoo: nop\n.text\nmain: nop").unwrap();
        assert_eq!(
            x.label_table,
            label_table([("main", 0, 22..27), ("foo", 32, 7..11)])
        );
        assert_eq!(
            x.instructions,
            vec![
                main_nop(28..31),
                inst(32, &["foo"], "nop", NOP_BINARY, 12..15),
            ]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());

        let x = compile_arch(
            ".text\nmain: nop\n.ktext\nfoo: nop",
            include_str!("../tests/architecture2.json"),
        )
        .unwrap();
        assert_eq!(
            x.label_table,
            label_table([("main", 32, 6..11), ("foo", 0, 23..27)])
        );
        assert_eq!(
            x.instructions,
            vec![
                inst(0, &["foo"], "nop", NOP_BINARY, 28..31),
                inst(32, &["main"], "nop", NOP_BINARY, 12..15),
            ]
        );
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn space() {
        let x = compile(".data\n.zero 3\n.zero 1\n.text\nmain: nop").unwrap();
        assert_eq!(x.label_table, label_table([("main", 0, 28..33)]));
        assert_eq!(x.instructions, vec![main_nop(34..37)]);
        assert_eq!(
            x.data_memory,
            vec![
                data(16, &[], Value::Space(3u8.into())),
                data(19, &[], Value::Space(1u8.into())),
            ]
        );
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn int() {
        let test_cases = [
            ("byte ", 1u8, IntegerType::Byte, 2u64.pow(8) - 128),
            ("half ", 2, IntegerType::HalfWord, 2u64.pow(16) - 128),
            ("word ", 4, IntegerType::Word, 2u64.pow(32) - 128),
            ("dword", 8, IntegerType::DoubleWord, u64::MAX - 127),
        ];
        for (name, size, r#type, val) in test_cases {
            let bits = usize::from(size * 8);
            // 1 argument
            let x = compile(&format!(".data\na: .{name} 1\n.text\nmain: nop")).unwrap();
            assert_eq!(
                x.label_table,
                label_table([("main", 0, 24..29), ("a", 16, 6..8)])
            );
            assert_eq!(x.instructions, vec![main_nop(30..33)]);
            assert_eq!(
                x.data_memory,
                vec![data(16, &["a"], int_val(1, bits, r#type))]
            );
            assert_eq!(x.global_symbols, HashSet::new());
            // Multiple arguments
            let x = compile(&format!(".data\nb: .{name} -128, 255\n.text\nmain: nop")).unwrap();
            assert_eq!(
                x.label_table,
                label_table([("main", 0, 32..37), ("b", 16, 6..8)])
            );
            assert_eq!(x.instructions, vec![main_nop(38..41)]);
            assert_eq!(
                x.data_memory,
                vec![
                    data(16, &["b"], int_val(val.cast_signed(), bits, r#type)),
                    data((16 + size).into(), &[], int_val(255, bits, r#type))
                ]
            );
            assert_eq!(x.global_symbols, HashSet::new());
        }
    }

    #[test]
    fn int_label() {
        let x = compile(".data\na: .byte a, b\nb: .byte main\n.text\nmain: nop").unwrap();
        assert_eq!(
            x.label_table,
            label_table([("main", 0, 40..45), ("a", 16, 6..8), ("b", 18, 20..22)])
        );
        assert_eq!(x.instructions, vec![main_nop(46..49)]);
        assert_eq!(
            x.data_memory,
            vec![
                data(16, &["a"], int_val(16, 8, IntegerType::Byte)),
                data(17, &[], int_val(18, 8, IntegerType::Byte)),
                data(18, &["b"], int_val(0, 8, IntegerType::Byte)),
            ]
        );
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn calc_reserved_space() {
        let code = ".data\nsize: .byte end-begin\nbegin: .zero 14\nend: .byte 0\n.text\nmain: nop";
        let x = compile(code).unwrap();
        assert_eq!(
            x.label_table,
            label_table([
                ("main", 0, 63..68),
                ("size", 16, 6..11),
                ("begin", 17, 28..34),
                ("end", 31, 44..48),
            ])
        );
        assert_eq!(x.instructions, vec![main_nop(69..72)]);
        assert_eq!(
            x.data_memory,
            vec![
                data(16, &["size"], int_val(14, 8, IntegerType::Byte)),
                data(17, &["begin"], Value::Space(14u8.into())),
                data(31, &["end"], int_val(0, 8, IntegerType::Byte)),
            ]
        );
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn float() {
        let test_cases = [
            ("float ", 4, FloatType::Float),
            ("double", 8, FloatType::Double),
        ];
        let value = |x, ty| match ty {
            FloatType::Float => Value::Float(x),
            FloatType::Double => Value::Double(x.into()),
        };
        for (name, size, r#type) in test_cases {
            // 1 argument
            let x = compile(&format!(".data\na: .{name} 1\n.text\nmain: nop")).unwrap();
            assert_eq!(
                x.label_table,
                label_table([("main", 0, 25..30), ("a", 16, 6..8)])
            );
            assert_eq!(x.instructions, vec![main_nop(31..34)]);
            assert_eq!(x.data_memory, vec![data(16, &["a"], value(1.0, r#type))]);
            assert_eq!(x.global_symbols, HashSet::new());
            // Multiple arguments
            let x = compile(&format!(".data\nb: .{name} 1, 2\n.text\nmain: nop")).unwrap();
            assert_eq!(
                x.label_table,
                label_table([("main", 0, 28..33), ("b", 16, 6..8)])
            );
            assert_eq!(x.instructions, vec![main_nop(34..37)]);
            assert_eq!(
                x.data_memory,
                vec![
                    data(16, &["b"], value(1.0, r#type)),
                    data((16 + size).try_into().unwrap(), &[], value(2.0, r#type)),
                ]
            );
            assert_eq!(x.global_symbols, HashSet::new());
        }
    }

    #[test]
    fn string() {
        let test_cases = [("string ", true), ("stringn", false)];
        for (name, null_terminated) in test_cases {
            // 1 argument
            let x = compile(&format!(".data\na: .{name} \"a\"\n.text\nmain: nop")).unwrap();
            assert_eq!(
                x.label_table,
                label_table([("main", 0, 28..33), ("a", 16, 6..8)])
            );
            assert_eq!(x.instructions, vec![main_nop(34..37)]);
            let str = |x: &str| Value::String {
                data: x.into(),
                null_terminated,
            };
            assert_eq!(x.data_memory, vec![data(16, &["a"], str("a"))]);
            assert_eq!(x.global_symbols, HashSet::new());
            // Multiple arguments
            let x = compile(&format!(".data\nb: .{name} \"b\", \"0\"\n.text\nmain: nop")).unwrap();
            assert_eq!(
                x.label_table,
                label_table([("main", 0, 33..38), ("b", 16, 6..8)])
            );
            assert_eq!(x.instructions, vec![main_nop(39..42)]);
            assert_eq!(
                x.data_memory,
                vec![
                    data(16, &["b"], str("b")),
                    data(17 + u64::from(null_terminated), &[], str("0")),
                ]
            );
            assert_eq!(x.global_symbols, HashSet::new());
        }
    }

    #[test]
    fn global() {
        // Before definition
        let x = compile(".global main\n.text\nmain: nop").unwrap();
        assert_eq!(x.label_table, label_table([("main", 0, 19..24)]));
        assert_eq!(x.instructions, vec![main_nop(25..28)]);
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::from(["main".into()]));
        // After definition
        let x = compile(".text\nmain: nop\n.global main").unwrap();
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(x.instructions, vec![main_nop(12..15)]);
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::from(["main".into()]));
        // Multiple arguments
        let x = compile(".text\nmain: nop\ntest: nop\n.global main, test").unwrap();
        assert_eq!(
            x.label_table,
            label_table([("main", 0, 6..11), ("test", 4, 16..21)])
        );
        let nop = inst(4, &["test"], "nop", NOP_BINARY, 22..25);
        assert_eq!(x.instructions, vec![main_nop(12..15), nop]);
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(
            x.global_symbols,
            HashSet::from(["main".into(), "test".into()])
        );
    }

    #[test]
    fn exp_align() {
        for size in [1, 3] {
            let x = compile(&format!(
                ".data\n.zero 1\na: b: .align {size}\n.zero 1\n.text\nmain: nop"
            ))
            .unwrap();
            assert_eq!(
                x.label_table,
                label_table([("main", 0, 43..48), ("a", 17, 14..16), ("b", 17, 17..19)])
            );
            assert_eq!(x.instructions, vec![main_nop(49..52)]);
            let alignment = 2u64.pow(size) - 1;
            assert_eq!(
                x.data_memory,
                vec![
                    data(16, &[], Value::Space(1u8.into())),
                    data(17, &["a", "b"], Value::Padding(alignment.into())),
                    data(17 + alignment, &[], Value::Space(1u8.into()))
                ]
            );
            assert_eq!(x.global_symbols, HashSet::new());
        }
    }

    #[test]
    fn byte_align() {
        for size in [2, 8] {
            let x = compile(&format!(
                ".data\n.zero 1\na: .balign {size}\n.zero 1\n.text\nmain: nop"
            ))
            .unwrap();
            assert_eq!(
                x.label_table,
                label_table([("main", 0, 41..46), ("a", 17, 14..16)])
            );
            assert_eq!(x.instructions, vec![main_nop(47..50)]);
            let alignment = size - 1;
            assert_eq!(
                x.data_memory,
                vec![
                    data(16, &[], Value::Space(1u8.into())),
                    data(17, &["a"], Value::Padding(alignment.into())),
                    data(17 + alignment, &[], Value::Space(1u8.into()))
                ]
            );
            assert_eq!(x.global_symbols, HashSet::new());
        }
    }

    #[test]
    fn align_end() {
        for s in [1, 3] {
            let x = compile(&format!(".data\n.zero 1\n.align {s}\n.text\nmain: nop")).unwrap();
            assert_eq!(x.label_table, label_table([("main", 0, 29..34)]));
            assert_eq!(x.instructions, vec![main_nop(35..38)]);
            let alignment = 2u64.pow(s) - 1;
            assert_eq!(
                x.data_memory,
                vec![
                    data(16, &[], Value::Space(1u8.into())),
                    data(17, &[], Value::Padding(alignment.into())),
                ]
            );
            assert_eq!(x.global_symbols, HashSet::new());
        }
    }

    #[test]
    fn already_aligned() {
        for size in [0, 1, 2] {
            let x = compile(&format!(
                ".data\n.zero 4\na: .align {size}\n.zero 1\n.text\nmain: nop"
            ))
            .unwrap();
            assert_eq!(
                x.label_table,
                label_table([("main", 0, 40..45), ("a", 20, 14..16)])
            );
            assert_eq!(x.instructions, vec![main_nop(46..49)]);
            assert_eq!(
                x.data_memory,
                vec![
                    data(16, &[], Value::Space(4u8.into())),
                    data(20, &[], Value::Space(1u8.into()))
                ]
            );
            assert_eq!(x.global_symbols, HashSet::new());
        }
    }

    #[test]
    fn align_decrease() {
        let x =
            compile(".data\n.zero 4\na: .align 3\nb: .align 2\n.zero 1\n.text\nmain: nop").unwrap();
        assert_eq!(
            x.label_table,
            label_table([("main", 0, 52..57), ("a", 20, 14..16), ("b", 24, 26..28)])
        );
        assert_eq!(x.instructions, vec![main_nop(58..61)]);
        assert_eq!(
            x.data_memory,
            vec![
                data(16, &[], Value::Space(4u8.into())),
                data(20, &["a"], Value::Padding(4u8.into())),
                data(24, &[], Value::Space(1u8.into()))
            ]
        );
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn kernel_data() {
        let x = compile(".text\nmain: nop\n.kdata\n.zero 1\n.data\n.zero 2").unwrap();
        let data_mem = vec![
            data(16, &[], Value::Space(2u8.into())),
            data(48, &[], Value::Space(1u8.into())),
        ];
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(x.instructions, vec![main_nop(12..15)]);
        assert_eq!(x.data_memory, data_mem);
        assert_eq!(x.global_symbols, HashSet::new());

        let x = compile(".text\nmain: nop\n.data\n.zero 2\n.kdata\n.zero 1").unwrap();
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        assert_eq!(x.instructions, vec![main_nop(12..15)]);
        assert_eq!(x.data_memory, data_mem);
        assert_eq!(x.global_symbols, HashSet::new());

        let x = compile_arch(
            ".text\nmain: nop\n.data\n.zero 2\n.kdata\n.zero 1",
            include_str!("../tests/architecture2.json"),
        )
        .unwrap();
        assert_eq!(x.label_table, label_table([("main", 32, 6..11)]));
        assert_eq!(
            x.instructions,
            vec![inst(32, &["main"], "nop", NOP_BINARY, 12..15)]
        );
        assert_eq!(
            x.data_memory,
            vec![
                data(16, &[], Value::Space(1u8.into())),
                data(48, &[], Value::Space(2u8.into()))
            ]
        );
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn read_pc() {
        let x = compile(".text\nmain: nop\nimm ., 0, 0\n.data\n.word .").unwrap();
        assert_eq!(x.label_table, label_table([("main", 0, 6..11)]));
        let binary = "00010000000000000000000000000000";
        let result = "imm 4, 0, 0";
        assert_eq!(
            x.instructions,
            vec![main_nop(12..15), inst(4, &[], result, binary, 16..27)]
        );
        assert_eq!(
            x.data_memory,
            vec![data(16, &[], int_val(16, 32, IntegerType::Word)),]
        );
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn unallowed_statement() {
        assert_eq!(
            compile(".data\nnop\n.text\nmain: nop"),
            Err(ErrorKind::UnallowedStatementType {
                section: Some((DirectiveSegment::Data, (0..5).span())),
                found: DirectiveSegment::Code,
            }
            .add_span((6..9).span())),
        );
        assert_eq!(
            compile(".kdata\nnop\n.text\nmain: nop"),
            Err(ErrorKind::UnallowedStatementType {
                section: Some((DirectiveSegment::KernelData, (0..6).span())),
                found: DirectiveSegment::Code,
            }
            .add_span((7..10).span())),
        );
        assert_eq!(
            compile(".text\nmain: nop\n.byte 1"),
            Err(ErrorKind::UnallowedStatementType {
                section: Some((DirectiveSegment::Code, (0..5).span())),
                found: DirectiveSegment::Data,
            }
            .add_span((16..23).span())),
        );
        assert_eq!(
            compile(".ktext\nmain: nop\n.byte 1"),
            Err(ErrorKind::UnallowedStatementType {
                section: Some((DirectiveSegment::KernelCode, (0..6).span())),
                found: DirectiveSegment::Data,
            }
            .add_span((17..24).span())),
        );
        assert_eq!(
            compile("nop\n.text\nmain: nop"),
            Err(ErrorKind::UnallowedStatementType {
                section: None,
                found: DirectiveSegment::Code,
            }
            .add_span((0..3).span())),
        );
        assert_eq!(
            compile(".byte 1\n.text\nmain: nop"),
            Err(ErrorKind::UnallowedStatementType {
                section: None,
                found: DirectiveSegment::Data,
            }
            .add_span((0..7).span())),
        );
    }

    #[test]
    fn unknown_directive() {
        assert_eq!(
            compile(".test\n.text\nmain: nop"),
            Err(ErrorKind::UnknownDirective(".test".into()).add_span((0..5).span())),
        );
    }

    #[test]
    fn unknown_instruction() {
        assert_eq!(
            compile(".text\nmain: test"),
            Err(ErrorKind::UnknownInstruction("test".into()).add_span((12..16).span())),
        );
    }

    #[test]
    fn unknown_label() {
        assert_eq!(
            compile(".text\nmain: imm 0, 0, test"),
            Err(ErrorKind::UnknownLabel("test".into()).add_span((22..26).span())),
        );
    }

    #[test]
    fn unknown_register_file() {
        let arch = include_str!("../tests/architecture2.json");
        assert_eq!(
            compile_arch(".text\nmain: ctrl pc", arch),
            Err(ErrorKind::UnknownRegisterFile(RegisterType::Ctrl).add_span((17..19).span())),
        );
        assert_eq!(
            compile_arch(".text\nmain: float ft0", arch),
            Err(
                ErrorKind::UnknownRegisterFile(RegisterType::Float(FloatType::Float))
                    .add_span((18..21).span())
            ),
        );
        assert_eq!(
            compile_arch(".text\nmain: double ft0", arch),
            Err(
                ErrorKind::UnknownRegisterFile(RegisterType::Float(FloatType::Double))
                    .add_span((19..22).span())
            ),
        );
    }

    #[test]
    fn unknown_register() {
        assert_eq!(
            compile(".text\nmain: reg x0, x0, ft1, ft2"),
            Err(ErrorKind::UnknownRegister {
                name: "x0".into(),
                file: RegisterType::Ctrl,
            }
            .add_span((16..18).span())),
        );
        assert_eq!(
            compile(".text\nmain: reg 2, x0, ft1, ft2"),
            Err(ErrorKind::UnknownRegister {
                name: "2".into(),
                file: RegisterType::Ctrl,
            }
            .add_span((16..17).span())),
        );
        // Register names should be case sensitive if enabled in the architecture
        assert_eq!(
            compile(".text\nmain: reg pc, x0, ft1, ft2"),
            Err(ErrorKind::UnknownRegister {
                name: "pc".into(),
                file: RegisterType::Ctrl,
            }
            .add_span((16..18).span())),
        );
        assert_eq!(
            compile(".text\nmain: reg PC, PC, ft1, ft2"),
            Err(ErrorKind::UnknownRegister {
                name: "PC".into(),
                file: RegisterType::Int,
            }
            .add_span((20..22).span())),
        );
        assert_eq!(
            compile(".text\nmain: reg PC, x0, x0, ft2"),
            Err(ErrorKind::UnknownRegister {
                name: "x0".into(),
                file: RegisterType::Float(FloatType::Float),
            }
            .add_span((24..26).span())),
        );
        assert_eq!(
            compile(".text\nmain: reg PC, x0, F1, ft2"),
            Err(ErrorKind::UnknownRegister {
                name: "F1".into(),
                file: RegisterType::Float(FloatType::Float),
            }
            .add_span((24..26).span())),
        );
        assert_eq!(
            compile(".text\nmain: reg PC, x0, fs1, fs2"),
            Err(ErrorKind::UnknownRegister {
                name: "fs2".into(),
                file: RegisterType::Float(FloatType::Double),
            }
            .add_span((29..32).span())),
        );
    }

    #[test]
    fn unknown_enum_value() {
        assert_eq!(
            compile(".text\nmain: enum a, b, value, wrong"),
            Err(ErrorKind::UnknownEnumValue {
                value: "wrong".into(),
                enum_name: "test".into(),
            }
            .add_span((30..35).span())),
        );
        assert_eq!(
            compile(".text\nmain: enum a, b, value, a"),
            Err(ErrorKind::UnknownEnumValue {
                value: "a".into(),
                enum_name: "test".into(),
            }
            .add_span((30..31).span())),
        );
        assert_eq!(
            compile(".text\nmain: enum a, c, value, last"),
            Err(ErrorKind::UnknownEnumValue {
                value: "c".into(),
                enum_name: "enum1".into(),
            }
            .add_span((20..21).span())),
        );
        // Enum names should be case sensitive
        assert_eq!(
            compile(".text\nmain: enum a, b, value, OTHER"),
            Err(ErrorKind::UnknownEnumValue {
                value: "OTHER".into(),
                enum_name: "test".into(),
            }
            .add_span((30..35).span())),
        );
    }

    #[test]
    fn section_args() {
        assert_eq!(
            compile(".data 1\n.text\nmain: nop"),
            Err(ErrorKind::IncorrectDirectiveArgumentNumber {
                expected: ArgumentNumber::new(0, false),
                found: 1
            }
            .add_span((6..7).span())),
        );
    }

    #[test]
    fn padding_args_number() {
        for directive in ["zero  ", "align ", "balign"] {
            assert_eq!(
                compile(&format!(".data\n.{directive}\n.text\nmain: nop")),
                Err(ErrorKind::IncorrectDirectiveArgumentNumber {
                    expected: ArgumentNumber::new(1, false),
                    found: 0
                }
                .add_span((13..13).span())),
                "{directive}"
            );
            assert_eq!(
                compile(&format!(".data\n.{directive} 1, 2\n.text\nmain: nop")),
                Err(ErrorKind::IncorrectDirectiveArgumentNumber {
                    expected: ArgumentNumber::new(1, false),
                    found: 2
                }
                .add_span((14..18).span())),
                "{directive}"
            );
        }
    }

    #[test]
    fn padding_negative() {
        for directive in ["zero  ", "align ", "balign"] {
            assert_eq!(
                compile(&format!(".data\n.{directive} -1\n.text\nmain: nop")),
                Err(ErrorKind::UnallowedNegativeValue((-1).into()).add_span((14..16).span())),
                "{directive}"
            );
        }
    }

    #[test]
    fn int_args_type() {
        for directive in [
            "zero  ", "align ", "balign", "byte  ", "half  ", "word  ", "dword ",
        ] {
            assert_eq!(
                compile(&format!(".data\n.{directive} \"a\"\n.text\nmain: nop")),
                Err(ErrorKind::IncorrectArgumentType {
                    expected: ArgumentType::Expression,
                    found: ArgumentType::String
                }
                .add_span((14..17).span())),
                "{directive}"
            );
            assert_eq!(
                compile(&format!(".data\n.{directive} 1.0\n.text\nmain: nop")),
                Err(ErrorKind::UnallowedFloat((14..17).span()).add_span((14..17).span())),
                "{directive}"
            );
        }
        assert_eq!(
            compile(".text\nmain: imm 0, 0, 1.0"),
            Err(ErrorKind::UnallowedFloat((22..25).span()).add_span((22..25).span())),
        );
        assert_eq!(
            compile(".text\nmain: reg PC, 0+2, ft1, ft2"),
            Err(ErrorKind::IncorrectArgumentType {
                expected: ArgumentType::RegisterName,
                found: ArgumentType::Expression,
            }
            .add_span((20..23).span())),
        );
        assert_eq!(
            compile(".text\nmain: enum a, b, value, 0"),
            Err(ErrorKind::IncorrectArgumentType {
                expected: ArgumentType::Identifier,
                found: ArgumentType::Expression,
            }
            .add_span((30..31).span())),
        );
    }

    #[test]
    fn data_no_args() {
        for directive in ["byte  ", "float ", "string"] {
            assert_eq!(
                compile(&format!(".data\n.{directive}\n.text\nmain: nop")),
                Err(ErrorKind::IncorrectDirectiveArgumentNumber {
                    expected: ArgumentNumber::new(1, true),
                    found: 0
                }
                .add_span((13..13).span())),
                "{directive}"
            );
        }
    }

    #[test]
    fn data_unaligned() {
        for (directive, size) in [
            ("half  ", 2u8),
            ("word  ", 4),
            ("dword ", 8),
            ("float ", 4),
            ("double", 8),
        ] {
            assert_eq!(
                compile(&format!(".data\n.byte 0\n.{directive} 1\n.text\nmain: nop")),
                Err(ErrorKind::DataUnaligned {
                    address: 17u8.into(),
                    alignment: size.into(),
                }
                .add_span((22..23).span())),
                "{directive}",
            );
        }
    }

    #[test]
    fn int_args_size() {
        let range = |x: std::ops::Range<i32>| x.start.into()..=(x.end - 1).into();

        // Data directives
        assert_eq!(
            compile(".data\n.byte 256\n.text\nmain: nop"),
            Err(ErrorKind::IntegerOutOfRange(256.into(), range(-128..256))
                .add_span((12..15).span())),
        );
        let s = (12..16).span();
        assert_eq!(
            compile(".data\n.byte -129\n.text\nmain: nop"),
            Err(ErrorKind::IntegerOutOfRange((-129).into(), range(-128..256)).add_span(s)),
        );
        let s = (12..17).span();
        assert_eq!(
            compile(".data\n.half 65536\n.text\nmain: nop"),
            Err(ErrorKind::IntegerOutOfRange(65536.into(), range(-32768..65536)).add_span(s)),
        );
        // Instruction arguments
        assert_eq!(
            compile(".text\nmain: imm 8, 0, 0"),
            Err(ErrorKind::IntegerOutOfRange(8.into(), range(-8..8)).add_span((16..17).span())),
        );
        assert_eq!(
            compile(".text\nmain: imm -9, 0, 0"),
            Err(ErrorKind::IntegerOutOfRange((-9).into(), range(-8..8)).add_span((16..18).span())),
        );
        assert_eq!(
            compile(".text\nmain: imm 0, 256, 0"),
            Err(ErrorKind::IntegerOutOfRange(256.into(), range(0..256)).add_span((19..22).span())),
        );
        assert_eq!(
            compile(".text\nmain: imm 0, -1, 0"),
            Err(ErrorKind::IntegerOutOfRange((-1).into(), range(0..256)).add_span((19..21).span())),
        );
        assert_eq!(
            compile(".text\nmain: imm 0, 0, 20"),
            Err(ErrorKind::IntegerOutOfRange(20.into(), range(0..16)).add_span((22..24).span())),
        );
    }

    #[test]
    fn float_args_type() {
        assert_eq!(
            compile(".data\n.float \"a\"\n.text\nmain: nop"),
            Err(ErrorKind::IncorrectArgumentType {
                expected: ArgumentType::Expression,
                found: ArgumentType::String
            }
            .add_span((13..16).span())),
        );
    }

    #[test]
    fn string_args_type() {
        assert_eq!(
            compile(".data\n.string 1\n.text\nmain: nop"),
            Err(ErrorKind::IncorrectArgumentType {
                expected: ArgumentType::String,
                found: ArgumentType::Expression
            }
            .add_span((14..15).span())),
        );
    }

    #[test]
    fn global_args_type() {
        assert_eq!(
            compile(".global \"test\"\n.text\nmain: nop"),
            Err(ErrorKind::IncorrectArgumentType {
                expected: ArgumentType::Identifier,
                found: ArgumentType::String
            }
            .add_span((8..14).span())),
        );
        assert_eq!(
            compile(".global 123\n.text\nmain: nop"),
            Err(ErrorKind::IncorrectArgumentType {
                expected: ArgumentType::Identifier,
                found: ArgumentType::Expression
            }
            .add_span((8..11).span())),
        );
    }

    #[test]
    fn incorrect_instruction_syntax() {
        let assert = |err, syntaxes: &[&str], expected_span: Range| match err {
            Err(ErrorData { span, kind }) => match *kind {
                ErrorKind::IncorrectInstructionSyntax(s) => {
                    assert_eq!(span, (expected_span).span());
                    assert_eq!(s.into_iter().map(|x| x.0).collect::<Vec<_>>(), syntaxes);
                }
                x => panic!(
                    "Incorrect result, expected ErrorKind::IncorrectInstructionSyntax: {x:?}"
                ),
            },
            x => panic!("Incorrect result, expected Err variant: {x:?}"),
        };
        assert(compile(".text\nmain: nop 1"), &["nop"], 16..17);
        assert(
            compile(".text\nmain: multi &, 1"),
            &["multi imm4", "multi $", "multi imm5"],
            18..22,
        );
    }

    #[test]
    fn expr_eval() {
        use error::OperationKind;
        assert_eq!(
            compile(".data\n.byte 1/0\n.text\nmain: nop"),
            Err(ErrorKind::DivisionBy0((14..15).span()).add_span((13..14).span())),
        );
        assert_eq!(
            compile(".text\nmain: imm 0, 0, 1/0"),
            Err(ErrorKind::DivisionBy0((24..25).span()).add_span((23..24).span())),
        );
        assert_eq!(
            compile(".text\nmain: imm 0, 0, 1%0"),
            Err(ErrorKind::RemainderWith0((24..25).span()).add_span((23..24).span())),
        );
        assert_eq!(
            compile(".data\n.float ~1.0\n.text\nmain: nop"),
            Err(
                ErrorKind::UnallowedFloatOperation(OperationKind::Complement, (14..17).span())
                    .add_span((13..14).span())
            ),
        );
        for (op, c) in [
            (OperationKind::BitwiseOR, '|'),
            (OperationKind::BitwiseAND, '&'),
            (OperationKind::BitwiseXOR, '^'),
        ] {
            assert_eq!(
                compile(&format!(".data\n.float 1.0 {c} 2.0\n.text\nmain: nop")),
                Err(ErrorKind::UnallowedFloatOperation(op, (13..16).span())
                    .add_span((17..18).span())),
            );
        }
    }

    #[test]
    fn missing_main() {
        assert_eq!(
            compile(".text\nnop"),
            Err(ErrorKind::MissingMainLabel.add_span((9..9).span())),
        );
        assert_eq!(
            compile(".text\nnop\n.data"),
            Err(ErrorKind::MissingMainLabel.add_span((9..9).span())),
        );
    }

    #[test]
    fn main_outside_code() {
        assert_eq!(
            compile(".data\nmain: .byte 1\n.text\nnop"),
            Err(ErrorKind::MainOutsideCode.add_span((6..11).span())),
        );
        assert_eq!(
            compile(".kdata\nmain: .byte 1\n.text\nnop"),
            Err(ErrorKind::MainOutsideCode.add_span((7..12).span())),
        );
        assert_eq!(
            compile(".ktext\nmain: nop\n.text\nnop"),
            Err(ErrorKind::MainOutsideCode.add_span((7..12).span())),
        );
    }

    #[test]
    fn empty_code() {
        let s = DEFAULT_SPAN;
        assert_eq!(compile(""), Err(ErrorKind::MissingMainLabel.add_span(s)));
    }

    #[test]
    fn duplicate_label() {
        let s = (16..21).span();
        assert_eq!(
            compile(".text\nmain: nop\nmain: nop"),
            Err(ErrorKind::DuplicateLabel("main".into(), Some((6..11).span())).add_span(s)),
        );
        let s = (23..29).span();
        assert_eq!(
            compile(".text\nmain: nop\nlabel:\nlabel: nop"),
            Err(ErrorKind::DuplicateLabel("label".into(), Some((16..22).span())).add_span(s)),
        );
    }

    #[test]
    fn section_full() {
        // Instructions
        assert_eq!(
            compile(".text\nmain: nop\nnop\nnop\nnop\nimm 0, 0, 0"),
            Err(ErrorKind::MemorySectionFull("Instructions").add_span((28..39).span())),
        );
        assert_eq!(
            compile(".text\nmain: nop\nnop\nnop\nnop2"),
            Err(ErrorKind::MemorySectionFull("Instructions").add_span((24..28).span())),
        );
        // Data directives
        for (directive, span) in [
            ("zero 5", 21..22),
            ("word 0\n.byte 0", 29..30),
            ("dword 0", 22..23),
            ("double 0", 23..24),
            ("string \"1234\"", 23..29),
            ("stringn \"1234\"\n.stringn \"5\"", 40..43),
            ("balign 64\n.byte 0", 15..25),
        ] {
            assert_eq!(
                compile(&format!(".data\n.zero 12\n.{directive}\n.text\nmain: nop")),
                Err(ErrorKind::MemorySectionFull("Data").add_span((span).span())),
                "{directive}",
            );
        }
    }

    #[test]
    fn no_kernel() {
        let compile = |src| compile_arch(src, include_str!("../tests/architecture_no_kernel.json"));
        assert_eq!(
            compile(".ktext\nfoo: nop\n.text\nmain: nop"),
            Err(ErrorKind::MemorySectionFull("KernelInstructions").add_span((12..15).span())),
        );
        assert_eq!(
            compile(".kdata\n.zero 1\n.text\nmain: nop"),
            Err(ErrorKind::MemorySectionFull("KernelData").add_span((13..14).span())),
        );
    }

    #[test]
    fn library_labels() {
        let src = ".data\n.word test\n.text\nmain: nop";
        for val in [3u8, 11, 27] {
            let labels = HashMap::from([("test".into(), val.into())]);
            let x = compile_with(src, &BigUint::ZERO, labels.clone(), false).unwrap();
            let mut labels = LabelTable::from(labels);
            labels
                .insert("main".into(), (23..28).span(), BigUint::ZERO)
                .unwrap();
            assert_eq!(x.label_table, labels);
            assert_eq!(x.instructions, vec![main_nop(29..32)]);
            let val = val.into();
            assert_eq!(
                x.data_memory,
                vec![data(16, &[], int_val(val, 32, IntegerType::Word))]
            );
            assert_eq!(x.global_symbols, HashSet::new());
        }
    }

    #[test]
    fn library_offset() {
        let src = ".text\nmain: nop";
        for val in [5u64, 8, 11] {
            let x = compile_with(src, &val.into(), HashMap::new(), false).unwrap();
            assert_eq!(x.label_table, label_table([("main", val, 6..11)]));
            let nop = inst(val, &["main"], "nop", NOP_BINARY, 12..15);
            assert_eq!(x.instructions, vec![nop]);
            assert_eq!(x.data_memory, vec![]);
            assert_eq!(x.global_symbols, HashSet::new());
        }
    }

    #[test]
    fn compile_library() {
        let x = compile_with(".text\ntest: nop", &BigUint::ZERO, HashMap::new(), true).unwrap();
        assert_eq!(x.label_table, label_table([("test", 0, 6..11)]));
        let nop = inst(0, &["test"], "nop", NOP_BINARY, 12..15);
        assert_eq!(x.instructions, vec![nop]);
        assert_eq!(x.data_memory, vec![]);
        assert_eq!(x.global_symbols, HashSet::new());
    }

    #[test]
    fn main_in_library() {
        assert_eq!(
            compile_with(".text\nmain: nop", &BigUint::ZERO, HashMap::new(), true),
            Err(ErrorKind::MainInLibrary.add_span((6..11).span()))
        );
        let labels = HashMap::from([("main".into(), 4u8.into())]);
        assert_eq!(
            compile_with(".text\ntest: nop", &BigUint::ZERO, labels, true),
            Err(ErrorKind::MainInLibrary.add_span(DEFAULT_SPAN))
        );
    }
}
