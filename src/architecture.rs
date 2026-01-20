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

//! Module containing the specification of the architecture definition structure
//!
//! The entry point for the specification is the [`Architecture`] struct

use num_bigint::BigUint;
use schemars::{schema_for, JsonSchema};
use serde::{de::Error, Deserialize};
use serde_json::Number;

use std::collections::HashMap;

mod utils;
pub use utils::NonEmptyRangeInclusive;
pub use utils::{BaseN, Integer, RangeFrom};

mod json;

/// Architecture description
#[derive(Deserialize, JsonSchema, Debug, Clone)]
pub struct Architecture<'a> {
    /// Metadata about the architecture
    /// Order of elements is assumed to be name, bits, description, data format,
    /// memory alignment, main function, passing convention, and sensitive register
    /// name
    #[serde(borrow)]
    pub config: Config<'a>,
    /// Components (register files) of the architecture. It's assumed that the first register of
    /// the first file will contain the program counter
    pub components: Vec<Component<'a>>,
    /// Instructions allowed
    pub instructions: Vec<Instruction<'a>>,
    /// Pseudoinstructions allowed
    pub pseudoinstructions: Vec<Pseudoinstruction<'a>>,
    /// Directives allowed
    pub directives: Vec<Directive<'a>>,
    /// Memory layout of the architecture. Order of elements is assumed to be optionally ktext
    /// start/end and kdata start/end, followed by text start/end, data start/end, and stack
    /// start/end
    pub memory_layout: MemoryLayout,
    /// Interrupt configuration
    #[serde(default)]
    pub interrupts: Option<Interrupts>,
    /// Timer configuration
    #[serde(default)]
    pub timer: Option<Timer>,
    /// Definitions of possible enumerated instruction fields
    #[serde(default)]
    pub enums: HashMap<&'a str, EnumDefinition<'a>>,
    /// Definitions of possible modifiers
    #[serde(default)]
    pub modifiers: ModifierDefinitions<'a>,
}

/// Definitions of possible modifiers
pub type ModifierDefinitions<'a> = HashMap<&'a str, Modifier>;

/// Definition of a expression modifier, an operator which returns a slice of bits from its input
#[derive(Deserialize, JsonSchema, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Modifier {
    /// Range of bits to select from the expression's value
    pub range: RangeFrom,
    /// Whether to account for the lower (unselected) bits being treated as a signed number. If
    /// `true`, will add 1 to the output if that number would have been negative
    pub lower_signed: bool,
    /// Whether to return the result as a signed (`true`) or unsigned (`false`) number. If the
    /// range of bits selected has no upper bound, the sign of the input will always be preserved
    pub output_signed: bool,
}

/// Definition of an enumerated field
pub type EnumDefinition<'a> = HashMap<&'a str, Integer>;

/// Architecture metadata attributes
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Config<'a> {
    /// Name of the architecture
    pub name: &'a str,
    /// Word size in bits
    pub word_size: usize,
    /// Description of the architecture
    pub description: &'a str,
    /// Storage format of the architecture (big/little endian)
    pub endianness: Endianness,
    /// Whether to enable memory alignment
    pub memory_alignment: bool,
    /// Name of the `main` function of the program
    pub main_function: &'a str,
    /// Whether to enable function parameter passing convention checks
    pub passing_convention: bool,
    /// Whether the register names should be case sensitive (`true`) or not (`false`)
    pub sensitive_register_name: bool,
    /// String to use as line comment prefix
    pub comment_prefix: &'a str,
}

/// Endianness of data in the architecture
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum Endianness {
    BigEndian,
    LittleEndian,
}

/// Register file
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
pub struct Component<'a> {
    /// Name of the register file
    pub name: &'a str,
    /// Type of the registers
    r#type: ComponentType,
    /// Whether the registers have double the word size
    double_precision: bool,
    /// Registers in this file
    pub elements: Vec<Register<'a>>,
}

/// Types of register files allowed
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
pub enum ComponentType {
    /// Control registers
    #[serde(rename = "ctrl_registers")]
    Ctrl,
    /// Integer registers
    #[serde(rename = "int_registers")]
    Int,
    /// Floating point registers
    #[serde(rename = "fp_registers")]
    Float,
}

/// Type of registers allowed
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RegisterType {
    /// Control registers
    Ctrl,
    /// Integer registers
    Int,
    /// Floating point registers
    Float(FloatType),
}

/// Register specification
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
pub struct Register<'a> {
    /// List of aliases
    #[serde(borrow)]
    pub name: Vec<&'a str>,
    /// Encoding of the register in an instruction
    pub encoding: Integer,
    /// Size
    pub nbits: Integer,
    /// Current value of the register
    pub value: Number,
    /// Default value of the register
    #[serde(default)]
    pub default_value: Option<Number>,
    /// Properties of this register
    pub properties: Vec<RegisterProperty>,
}

/// Properties of a register
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum RegisterProperty {
    /// Register can be read
    Read,
    /// Register can be written
    Write,
    /// Register should be preserved across routine calls
    Saved,
    /// Register contains the program counter
    ProgramCounter,
    /// Register to save PC in during interruptions
    ExceptionProgramCounter,
    /// Interruption ID register
    EventCause,
    /// Register can only be used on kernel mode
    StatusRegister,
    /// Writes to this register are ignored. Ignored if `Write` is set
    IgnoreWrite,
    /// Register contains the stack pointer
    StackPointer,
    /// Register contains the global pointer. Only used in the UI
    GlobalPointer,
    /// Register contains the stack frame pointer. Only used in the UI
    FramePointer,
}

/// Instruction specification
#[derive(Deserialize, JsonSchema, Debug, Clone)]
pub struct Instruction<'a> {
    /// Name of the instruction
    pub name: &'a str,
    /// Type of the instruction
    pub r#type: InstructionType,
    /// Syntax of the instruction
    #[serde(flatten)]
    pub syntax: InstructionSyntax<'a, BitRange>,
    /// Binary op code
    pub co: BaseN<2>,
    /// Size of the instruction
    pub nwords: usize,
    /// Execution time of the instruction
    pub clk_cycles: Option<Integer>,
    /// Code to execute for the instruction
    // Can't be a reference because there might be escape sequences, which require
    // modifying the data on deserialization
    pub definition: String,
    /// Properties of the instruction
    pub properties: Option<Vec<InstructionProperties>>,
}

/// Allowed instruction types
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
pub enum InstructionType {
    #[serde(rename = "Arithmetic integer")]
    ArithmeticInteger,
    #[serde(rename = "Arithmetic floating point")]
    ArithmeticFloat,
    Logic,
    Comparison,
    Control,
    #[serde(rename = "I/O")]
    IO,
    #[serde(rename = "Conditional bifurcation")]
    ConditionalBifurcation,
    #[serde(rename = "Unconditional bifurcation")]
    UnconditionalBifurcation,
    Syscall,
    #[serde(rename = "Function call")]
    FunctionCall,
    #[serde(rename = "Transfer between registers")]
    TransferRegister,
    #[serde(rename = "Memory access")]
    MemoryAccess,
    Other,
}

/// Instruction syntax specification
#[derive(Deserialize, Debug, Clone)]
#[serde(try_from = "json::InstructionSyntax<BitRange>")]
pub struct InstructionSyntax<'a, BitRange> {
    /// Parser for the syntax of the instruction
    pub parser: crate::parser::Instruction,
    /// Translated instruction's syntax
    pub output_syntax: &'a str,
    /// Parameters of the instruction
    pub fields: Vec<InstructionField<'a, BitRange>>,
}
utils::schema_from!(InstructionSyntax<'a, T>, json::InstructionSyntax<T>);

/// Allowed instruction properties
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum InstructionProperties {
    ExitSubroutine,
    EnterSubroutine,
    Privileged,
}

/// Instruction field specification
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
pub struct InstructionField<'a, BitRange> {
    /// Name of the field
    pub name: &'a str,
    /// Type of the field
    #[serde(flatten)]
    pub r#type: FieldType<'a>,
    /// Range of bits of the field. Ignored for pseudoinstructions
    #[serde(flatten)]
    pub range: BitRange,
}

/// Range of bits of a field in a binary instruction
#[derive(Deserialize, Debug, PartialEq, Eq, Clone)]
#[serde(try_from = "json::BitRange")]
pub struct BitRange {
    /// Ranges of bits where to place the field, applied from MSB of the value to LSB
    ranges: Vec<NonEmptyRangeInclusive<usize>>,
    /// Amount of LSB to discard from the value before placing it in the binary instruction
    padding: usize,
}
utils::schema_from!(BitRange, json::BitRange);

/// Allowed instruction field types
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum FieldType<'a> {
    /// Opcode of the instruction
    Co,
    /// Extended operation code
    Cop {
        /// Fixed value of this field in the binary instruction (specified as a binary string)
        value: BaseN<2>,
    },
    /// Immediate signed integer
    #[serde(rename = "imm-signed")]
    ImmSigned,
    /// Immediate unsigned integer
    #[serde(rename = "imm-unsigned")]
    ImmUnsigned,
    /// Offset from the next instruction's address in bytes
    #[serde(rename = "offset_bytes")]
    OffsetBytes,
    /// Offset from the next instruction's address in words
    #[serde(rename = "offset_words")]
    OffsetWords,
    /// Control register
    #[serde(rename = "Ctrl-Reg")]
    CtrlReg,
    /// Integer register
    #[serde(rename = "INT-Reg")]
    IntReg,
    /// Double precision floating point register
    #[serde(rename = "DFP-Reg")]
    DoubleFPReg,
    /// Single precision floating point register
    #[serde(rename = "SFP-Reg")]
    SingleFPReg,
    /// Immediate address, equivalent to `ImmUnsigned`
    Address,
    /// Enumerated field that only allows a predefined set of names to be used
    Enum {
        /// Name of the enumeration, defined in [`Architecture::enums`]
        enum_name: &'a str,
    },
}

/// Pseudoinstruction specification
#[derive(Deserialize, JsonSchema, Debug, Clone)]
pub struct Pseudoinstruction<'a> {
    /// Name of the pseudoinstruction
    pub name: &'a str,
    /// Syntax of the instruction
    #[serde(flatten)]
    pub syntax: InstructionSyntax<'a, ()>,
    /// Code to execute for the instruction
    // Can't be a reference because there might be escape sequences, which require
    // modifying the data on deserialization
    pub definition: String,
    /// Properties of the instruction
    pub properties: Option<Vec<InstructionProperties>>,
}

/// Directive specification
#[derive(Deserialize, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(try_from = "json::Directive")]
pub struct Directive<'a> {
    /// Name of the directive
    pub name: &'a str,
    /// Action of the directive
    pub action: DirectiveAction<DirectiveData>,
}
utils::schema_from!(Directive<'a>, json::Directive);

/// Allowed actions for directives
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
pub enum DirectiveAction<DirectiveData> {
    /// Ignore this directive
    Nop(Nop),
    /// Switch to the given segment
    Segment(DirectiveSegment),
    /// Store symbols in an external symbols table
    GlobalSymbol(GlobalSymbol),
    /// Add data to the data segment
    Data(DirectiveData),
}

/// Store symbols in an external symbols table
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum GlobalSymbol {
    GlobalSymbol,
}

/// Ignore this directive
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum Nop {
    Nop,
}

/// Memory segment to switch to
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
pub enum DirectiveSegment {
    #[serde(rename = "kernel_code_segment")]
    KernelCode,
    #[serde(rename = "kernel_data_segment")]
    KernelData,
    #[serde(rename = "code_segment")]
    Code,
    #[serde(rename = "data_segment")]
    Data,
}

/// Data segment types
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DirectiveData {
    /// Store n * size null bytes in the data segment
    Space(usize),
    /// Store string
    String(StringType),
    /// Store integer
    Int(usize, IntegerType),
    /// Store floating point value
    Float(FloatType),
    /// Align the memory to a given size
    Alignment(AlignmentType),
}

/// Types of strings allowed
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum StringType {
    /// Ascii string with a terminating null byte (`\0`)
    AsciiNullEnd,
    /// Ascii string without a terminating null byte (`\0`)
    AsciiNotNullEnd,
}

/// Types of integers allowed
#[derive(Deserialize, JsonSchema, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IntegerType {
    Byte,
    HalfWord,
    Word,
    DoubleWord,
}

/// Types of floats allowed
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum FloatType {
    /// 32 bit float
    Float,
    /// 64 bit double
    Double,
}

/// Data alignment types
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
pub enum AlignmentType {
    /// Align data to n bytes
    #[serde(rename = "balign")]
    Byte,
    /// Align data to 2^n bytes
    #[serde(rename = "align")]
    Exponential,
}

/// Memory layout of the architecture
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
pub struct MemoryLayout {
    /// Addresses reserved for the kernel text segment
    ktext: Option<NonEmptyRangeInclusive<BigUint>>,
    /// Addresses reserved for the kernel data segment
    kdata: Option<NonEmptyRangeInclusive<BigUint>>,
    /// Addresses reserved for the text segment
    text: NonEmptyRangeInclusive<BigUint>,
    /// Addresses reserved for the data segment
    data: NonEmptyRangeInclusive<BigUint>,
    /// Addresses reserved for the stack segment
    stack: NonEmptyRangeInclusive<BigUint>,
}

#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
pub struct InterruptHandlers {
    /// JS Handler for CREATOR interrupt handler's syscall interrupt
    pub creator_syscall: Option<String>,
    /// JS Handler for the custom interrupt handler
    pub custom: Option<String>,
}

#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
pub struct Interrupts {
    /// Interrupt handler configuration
    pub handlers: InterruptHandlers,
    /// JS code to be executed in order to check what type of interrupt occurred.
    /// It must return an `InterruptType` (if an interrupt happened) or `null`
    /// (if it didn't)
    pub check: String,
    /// JS code to be executed in order to enable the specified interrupt
    /// `type`. Defaults to `global_enable`
    pub enable: Option<String>,
    /// JS code to be executed in order to disable the specified interrupt
    /// `type`. Defaults to `global_disable`
    pub disable: Option<String>,
    /// JS code to be executed in order to globally enable interrupts
    pub global_enable: String,
    /// JS code to be executed in order to globally disable interrupts
    pub global_disable: String,
    /// JS code to be executed in order to clear an interrupt of the specified
    /// `type`. Defaults to `global_clear`
    pub clear: Option<String>,
    /// JS code to be executed in order to clear all interrupts
    pub global_clear: String,
    /// JS code to be executed in order to set an interrupt given an interrupt
    /// `type`
    pub create: String,
    /// JS code to check whether the specified interrupt `type` is enabled. Must
    /// return a boolean. Defaults to `is_global_enabled`
    pub is_enabled: Option<String>,
    /// JS code to check whether interrupts are globally is enabled. Must return
    /// a boolean
    pub is_global_enabled: String,
}

#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
pub struct Timer {
    /// Number of clock cycles that correspond to one timer tick
    pub tick_cycles: usize,
    /// JS code to be executed each tick in order to advance the tick
    pub advance: String,
    /// JS code to be executed each tick in order to check the timer and act (e.g. launch an
    /// interrupt)
    pub handler: String,
    /// JS code to be executed in order to check whether the timer is enabled
    pub is_enabled: String,
    /// JS code to be executed in order to enable timer
    pub enable: String,
    /// JS code to be executed in order to disable timer
    pub disable: String,
}

impl Architecture<'_> {
    /// Generate a `JSON` schema
    #[must_use]
    #[allow(clippy::missing_panics_doc)] // This should never panic at runtime from user error
    pub fn schema() -> String {
        let schema = schema_for!(Architecture);
        serde_json::to_string_pretty(&schema)
            .expect("Input is known and fixed, so it shouldn't error out")
    }

    /// Load architecture data from `JSON`
    ///
    /// # Parameters
    ///
    /// * `src`: `JSON` data to deserialize
    ///
    /// # Errors
    ///
    /// Errors if the input `JSON` data is invalid, either because it's ill-formatted or because it
    /// doesn't conform to the specification
    pub fn from_json(src: &str) -> serde_json::Result<Architecture<'_>> {
        let arch = serde_json::from_str::<Architecture>(src)?;
        let word_size = arch.config.word_size;
        for instruction in &arch.instructions {
            let size = instruction.nwords.saturating_mul(word_size);
            for field in &instruction.syntax.fields {
                for range in &field.range.ranges {
                    let end = range.end();
                    if end >= size {
                        let msg = format!("instruction `{}` contains fields out of bounds (size: {size}, field index: {end})", instruction.name);
                        return Err(serde_json::Error::custom(msg));
                    }
                }
            }
        }
        Ok(arch)
    }

    /// Finds the action associated with the directive with the given name
    ///
    /// # Parameters
    ///
    /// * `name`: name to search for
    #[must_use]
    pub fn find_directive(&self, name: &str) -> Option<DirectiveAction<DirectiveData>> {
        self.directives
            .iter()
            .find(|directive| directive.name == name)
            .map(|directive| directive.action)
    }

    /// Gets the word size of the architecture
    #[must_use]
    pub const fn word_size(&self) -> usize {
        self.config.word_size
    }

    /// Gets the name of the label used as the entry point of the code
    #[must_use]
    pub const fn main_label(&self) -> &str {
        self.config.main_function
    }

    /// Gets the string to use as the line comment prefix
    #[must_use]
    pub const fn comment_prefix(&self) -> &str {
        self.config.comment_prefix
    }

    /// Gets the code section's start/end addresses
    #[must_use]
    pub const fn code_section(&self) -> &NonEmptyRangeInclusive<BigUint> {
        &self.memory_layout.text
    }

    /// Gets the kernel's code section's start/end addresses
    #[must_use]
    pub const fn kernel_code_section(&self) -> Option<&NonEmptyRangeInclusive<BigUint>> {
        self.memory_layout.ktext.as_ref()
    }

    /// Gets the data section's start/end addresses
    #[must_use]
    pub const fn data_section(&self) -> &NonEmptyRangeInclusive<BigUint> {
        &self.memory_layout.data
    }

    /// Gets the kernel's data section's start/end addresses
    #[must_use]
    pub const fn kernel_data_section(&self) -> Option<&NonEmptyRangeInclusive<BigUint>> {
        self.memory_layout.kdata.as_ref()
    }

    /// Gets the instructions with the given name
    ///
    /// # Parameters
    ///
    /// * `name`: name to search for
    pub fn find_instructions<'b: 'c, 'c>(
        &'b self,
        name: &'c str,
    ) -> impl Iterator<Item = &'b Instruction<'b>> + 'c {
        self.instructions
            .iter()
            .filter(move |instruction| instruction.name == name)
    }

    /// Gets the pseudoinstructions with the given name
    ///
    /// # Parameters
    ///
    /// * `name`: name to search for
    pub fn find_pseudoinstructions<'b: 'c, 'c>(
        &'b self,
        name: &'c str,
    ) -> impl Iterator<Item = &'b Pseudoinstruction<'b>> + 'c {
        self.pseudoinstructions
            .iter()
            .filter(move |instruction| instruction.name == name)
    }

    /// Gets the register files with registers of the given type
    ///
    /// # Parameters
    ///
    /// * `type`: type of the file wanted
    pub fn find_reg_files(&self, r#type: RegisterType) -> impl Iterator<Item = &Component<'_>> {
        let eq = move |file: &&Component| match r#type {
            RegisterType::Int => file.r#type == ComponentType::Int,
            RegisterType::Ctrl => file.r#type == ComponentType::Ctrl,
            RegisterType::Float(x) => {
                file.r#type == ComponentType::Float
                    && (x == FloatType::Double) == file.double_precision
            }
        };
        self.components.iter().filter(eq)
    }
}

impl Component<'_> {
    /// Finds the register with the given name, returning its index in its register file, the
    /// register definition, and the name that matched
    ///
    /// # Parameters
    ///
    /// * `name`: name of the register to search for
    /// * `case`: whether the find should be case sensitive (`true`) or not (`false`)
    #[must_use]
    pub fn find_register(&self, name: &str, case: bool) -> Option<(&Register<'_>, &str)> {
        self.elements.iter().find_map(|reg| {
            let name = reg.name.iter().find(|&&n| {
                if case {
                    n == name
                } else {
                    n.eq_ignore_ascii_case(name)
                }
            });
            name.map(|&n| (reg, n))
        })
    }
}

impl DirectiveSegment {
    /// Checks whether the segment allows adding instructions
    #[must_use]
    pub const fn is_code(&self) -> bool {
        matches!(self, Self::Code | Self::KernelCode)
    }
}

impl BitRange {
    /// Calculates the size of this range in bits
    #[must_use]
    pub fn size(&self) -> usize {
        let size = self
            .ranges
            .iter()
            .map(|x| *x.size())
            .reduce(usize::saturating_add)
            .unwrap_or_default();
        size + self.padding
    }

    /// Gets an iterator of the ranges of bits specified
    pub fn iter(&self) -> impl Iterator<Item = &NonEmptyRangeInclusive<usize>> {
        self.ranges.iter()
    }

    /// Creates a new [`BitRange`]
    #[must_use]
    pub fn build(ranges: Vec<NonEmptyRangeInclusive<usize>>, padding: usize) -> Option<Self> {
        if ranges.is_empty() {
            return None;
        }
        Some(Self { ranges, padding })
    }

    /// Gets the amount of LSB to discard from the value before placing it in the binary instruction
    #[must_use]
    pub const fn padding(&self) -> usize {
        self.padding
    }
}
