/*
 * Copyright 2018-2026 CREATOR Team.
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

//! Module containing the definition of the compiler errors
//!
//! The main type is [`Error`]

use ariadne::{Color, Config, Fmt, IndexType, Label, Report, ReportKind, Source};
use num_bigint::{BigInt, BigUint};

use std::collections::HashMap;
use std::fmt;
use std::ops::RangeInclusive;
use std::{fmt::Write as _, io::Write as _};

use crate::architecture::{DirectiveAction, DirectiveSegment, FloatType, RegisterType};
use crate::error_rendering as utils;
use crate::error_rendering::{ArgNum, Colored, DisplayList};
use crate::parser::ParseError;
use crate::span::{Span, Spanned};

use super::{ArgumentNumber, Context, FileCache};
use super::{PseudoinstructionError, PseudoinstructionErrorKind};

/// Type of arguments for directives/instructions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArgumentType {
    String,
    Expression,
    RegisterName,
    Identifier,
}

/// Unsupported operations for floating point numbers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperationKind {
    Complement,
    BitwiseOR,
    BitwiseAND,
    BitwiseXOR,
    Shl,
    Shr,
}

/// Error type
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Kind {
    UnknownDirective(String),
    UnknownInstruction(String),
    UnknownLabel(String),
    UnknownRegisterFile(RegisterType),
    UnknownRegister {
        name: String,
        file: RegisterType,
    },
    UnknownEnumType(String),
    UnknownEnumValue {
        value: String,
        enum_name: String,
    },
    UnknownModifier(String),
    IncorrectInstructionSyntax(Vec<(String, ParseError)>),
    IncorrectDirectiveArgumentNumber {
        expected: ArgumentNumber,
        found: usize,
    },
    IncorrectArgumentType {
        expected: ArgumentType,
        found: ArgumentType,
    },
    DuplicateLabel(String, Option<Span>),
    MissingMainLabel,
    MainInLibrary,
    MainOutsideCode,
    MemorySectionFull {
        section: &'static str,
        requested: BigUint,
        available: BigUint,
    },
    DataUnaligned {
        address: BigUint,
        alignment: BigUint,
    },
    UnallowedStatementType {
        section: Option<Spanned<DirectiveSegment>>,
        found: DirectiveSegment,
    },
    UnallowedLabel,
    UnallowedFloat(Span),
    UnallowedFloatOperation(OperationKind, Span),
    UnallowedNegativeValue(BigInt),
    IntegerOutOfRange(BigInt, RangeInclusive<BigInt>),
    DivisionBy0(Span),
    RemainderWith0(Span),
    ShiftOutOfRange(Span, BigInt),
    PseudoinstructionError {
        name: String,
        error: Box<PseudoinstructionError>,
    },
}

/// Information about the error
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Data {
    /// Location in the assembly that produced the error
    pub span: Span,
    /// Type of the error
    pub kind: Box<Kind>,
}

/// Compiler error type
#[derive(Debug, Clone)]
pub struct Error<'arch> {
    /// Global compilation context
    pub ctx: Context<'arch>,
    /// Information about the error
    pub error: Data,
}

impl fmt::Display for ArgumentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Debug>::fmt(self, f)
    }
}

impl fmt::Display for OperationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Complement => write!(f, "complement"),
            Self::BitwiseOR => write!(f, "bitwise OR"),
            Self::BitwiseAND => write!(f, "bitwise AND"),
            Self::BitwiseXOR => write!(f, "bitwise XOR"),
            Self::Shl => write!(f, "shift left"),
            Self::Shr => write!(f, "shift right"),
        }
    }
}

impl fmt::Display for RegisterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ctrl => write!(f, "Control"),
            Self::Int => write!(f, "Integer"),
            Self::Float(FloatType::Float) => write!(f, "SingleFloatingPoint"),
            Self::Float(FloatType::Double) => write!(f, "DoubleFloatingPoint"),
        }
    }
}

impl fmt::Display for DirectiveSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", if self.is_code() { "Text" } else { "Data" })
    }
}

impl Kind {
    /// Adds a span to the error kind, promoting it to a [`Data`]
    ///
    /// # Parameters
    ///
    /// * `span`: location in the assembly code that caused the error
    #[must_use]
    pub(crate) fn add_span(self, span: Span) -> Data {
        Data {
            span,
            kind: Box::new(self),
        }
    }
}

/// Trait to get different types of data from an error, such as the error message, label message,
/// error code, or notes/hints
#[allow(unused_variables)]
pub trait Info {
    /// Gets the numeric error code of the error
    #[must_use]
    fn code(&self) -> u32 {
        0
    }

    /// Gets a note with extra information about the error if available
    ///
    /// # Parameters
    ///
    /// * `color`: whether the message should be formatted (`true`) or plain text (`false`)
    #[must_use]
    fn note(&self, color: bool) -> Option<String> {
        None
    }

    /// Gets a hint about how to fix the error if available
    ///
    /// # Parameters
    ///
    /// * `color`: whether the message should be formatted (`true`) or plain text (`false`)
    #[must_use]
    fn hint(&self, color: bool) -> Option<String> {
        None
    }

    /// Gets a list of extra context labels related to the error
    ///
    /// # Parameters
    ///
    /// * `color`: whether the message should be formatted (`true`) or plain text (`false`)
    #[must_use]
    fn context(&self, color: bool) -> Vec<(Span, String)> {
        vec![]
    }

    /// Gets the label text describing the error
    ///
    /// # Parameters
    ///
    /// * `color`: whether the message should be formatted (`true`) or plain text (`false`)
    #[must_use]
    fn label(&self, color: bool) -> String;

    /// Gets the error message of the error
    ///
    /// # Parameters
    ///
    /// * `color`: whether the message should be formatted (`true`) or plain text (`false`)
    #[must_use]
    fn msg(&self, color: bool) -> String;
}

impl Info for Error<'_> {
    fn code(&self) -> u32 {
        match self.error.kind.as_ref() {
            Kind::UnknownDirective(..) => 1,
            Kind::UnknownInstruction(..) => 2,
            Kind::UnknownLabel(..) => 3,
            Kind::UnknownRegisterFile(..) => 4,
            Kind::UnknownRegister { .. } => 5,
            Kind::UnknownEnumType { .. } => 6,
            Kind::UnknownEnumValue { .. } => 7,
            Kind::UnknownModifier { .. } => 8,
            Kind::IncorrectInstructionSyntax(..) => 9,
            Kind::IncorrectDirectiveArgumentNumber { .. } => 10,
            Kind::IncorrectArgumentType { .. } => 11,
            Kind::DuplicateLabel(..) => 12,
            Kind::MissingMainLabel => 13,
            Kind::MainInLibrary => 14,
            Kind::MainOutsideCode => 15,
            Kind::MemorySectionFull { .. } => 16,
            Kind::DataUnaligned { .. } => 17,
            Kind::UnallowedStatementType { .. } => 18,
            Kind::UnallowedLabel => 19,
            Kind::UnallowedFloat(..) => 20,
            Kind::UnallowedFloatOperation(..) => 21,
            Kind::UnallowedNegativeValue(..) => 22,
            Kind::IntegerOutOfRange(..) => 23,
            Kind::DivisionBy0(..) => 24,
            Kind::RemainderWith0(..) => 25,
            Kind::ShiftOutOfRange(..) => 26,
            Kind::PseudoinstructionError { .. } => 27,
        }
    }

    fn note(&self, _: bool) -> Option<String> {
        Some(match self.error.kind.as_ref() {
            Kind::IntegerOutOfRange(_, bounds) => {
                format!("Allowed range is [{}, {}]", bounds.start(), bounds.end())
            }
            Kind::ShiftOutOfRange(_, x) if *x >= BigInt::ZERO => {
                format!("Allowed range is [{}, {}]", u16::MIN, u16::MAX)
            }
            Kind::IncorrectInstructionSyntax(errs) => {
                let mut res = "Allowed formats:".to_string();
                for (syntax, _) in errs {
                    write!(res, "\n{syntax}").expect("The write macro can't fail for `String`s");
                }
                res
            }
            Kind::DuplicateLabel(_, None) => "Label also defined in library".into(),
            Kind::UnallowedStatementType { section: None, .. } => {
                "No section previously started".into()
            }
            Kind::MemorySectionFull {
                requested,
                available,
                ..
            } => {
                format!("Element requires {requested} bytes, but only {available} are free")
            }
            _ => return None,
        })
    }

    fn hint(&self, color: bool) -> Option<String> {
        Some(match self.error.kind.as_ref() {
            Kind::UnknownDirective(s) => {
                let names = utils::get_similar(s, self.ctx.arch.directives.iter().map(|d| d.name));
                format!("Did you mean {}?", DisplayList::non_empty(names, color)?)
            }
            Kind::UnknownInstruction(s) => {
                let inst_names = self.ctx.arch.instructions.iter().map(|i| i.name);
                let pseudo_names = self.ctx.arch.pseudoinstructions.iter().map(|i| i.name);
                let names = utils::get_similar(s, inst_names.chain(pseudo_names));
                format!("Did you mean {}?", DisplayList::non_empty(names, color)?)
            }
            Kind::UnknownLabel(s) => {
                let names =
                    utils::get_similar(s, self.ctx.label_table.iter().map(|(n, _)| n.as_str()));
                format!("Did you mean {}?", DisplayList::non_empty(names, color)?)
            }
            Kind::UnknownRegister { name, file } => {
                let files = self.ctx.arch.find_reg_files(*file);
                let registers = files.flat_map(|file| {
                    file.registers
                        .iter()
                        .flat_map(|reg| reg.name.iter().copied())
                });
                let names = utils::get_similar(name, registers);
                format!("Did you mean {}?", DisplayList::non_empty(names, color)?)
            }
            Kind::UnknownEnumValue { value, enum_name } => {
                let enums = &self.ctx.arch.enums;
                let default = HashMap::default();
                let enum_def = enums.get(enum_name.as_str()).unwrap_or(&default);
                let names = utils::get_similar(value, enum_def.keys().copied());
                format!("Did you mean {}?", DisplayList::non_empty(names, color)?)
            }
            Kind::UnknownModifier(s) => {
                let names = utils::get_similar(s, self.ctx.arch.modifiers.keys().copied());
                format!("Did you mean {}?", DisplayList::non_empty(names, color)?)
            }
            Kind::DuplicateLabel(.., Some(_)) => "Consider renaming either of the labels".into(),
            Kind::DuplicateLabel(.., None) | Kind::MainInLibrary => {
                "Consider renaming the label".into()
            }
            Kind::MainOutsideCode => "Consider moving the label to a user instruction".into(),
            Kind::IncorrectDirectiveArgumentNumber { expected, found } => {
                let expected = expected.amount;
                let (msg, n) = if expected > *found {
                    ("adding the missing", expected - found)
                } else {
                    ("removing the extra", found - expected)
                };
                let color = color.then_some(Color::Green);
                format!("Consider {msg} {}", ArgNum(n, color))
            }
            Kind::UnallowedStatementType { found, .. } => {
                let names: Vec<_> = self.ctx.arch.directives.iter()
                    .filter(|dir| matches!(dir.action, DirectiveAction::Segment(s) if s.is_code() == found.is_code()))
                    .map(|dir| dir.name)
                    .collect();
                let section_type = Colored(found, color.then_some(Color::BrightBlue));
                format!(
                    "Consider changing the section to {}{}{}",
                    section_type,
                    if names.is_empty() { "" } else { ", using " },
                    DisplayList::new(names, color)
                )
            }
            _ => return None,
        })
    }

    fn context(&self, color: bool) -> Vec<(Span, String)> {
        let red = color.then_some(Color::Red);
        match self.error.kind.as_ref() {
            Kind::DuplicateLabel(_, Some(span)) => {
                vec![(*span, "Label also defined here".into())]
            }
            Kind::UnallowedStatementType {
                section: Some(section),
                ..
            } => {
                vec![(section.1, "Section previously started here".into())]
            }
            Kind::UnallowedFloat(span) if *span != self.error.span => {
                vec![(*span, "Expression evaluates to a float due to this".into())]
            }
            Kind::UnallowedFloatOperation(_, span) => {
                vec![(*span, "Operands are converted to floats due to this".into())]
            }
            Kind::DivisionBy0(span) | Kind::RemainderWith0(span) => {
                vec![(
                    *span,
                    format!("This expression has value {}", Colored(0, red)),
                )]
            }
            Kind::ShiftOutOfRange(span, x) => {
                vec![(
                    *span,
                    format!("This expression has value {}", Colored(x, red)),
                )]
            }
            _ => Vec::new(),
        }
    }

    fn label(&self, color: bool) -> String {
        let red = color.then_some(Color::Red);
        match self.error.kind.as_ref() {
            Kind::UnknownDirective(..) => "Unknown directive".into(),
            Kind::UnknownInstruction(..) => "Unknown instruction".into(),
            Kind::UnknownLabel(..) => "Unknown label".into(),
            Kind::UnknownRegisterFile(..) => "Unknown register file".into(),
            Kind::UnknownRegister { .. } => "Unknown register".into(),
            Kind::UnknownEnumType { .. } => "Unknown enum type".into(),
            Kind::UnknownEnumValue { .. } => "Unknown enum value".into(),
            Kind::UnknownModifier { .. } => "Unknown modifier name".into(),
            Kind::IncorrectInstructionSyntax(..) => "Incorrect syntax".into(),
            Kind::IncorrectDirectiveArgumentNumber { found, .. } => {
                format!("This directive has {}", ArgNum(*found, red))
            }
            Kind::IncorrectArgumentType { found, .. } => {
                format!("This argument has type {}", Colored(found, red))
            }
            Kind::DuplicateLabel(..) => "Duplicate label".into(),
            Kind::MissingMainLabel => {
                let main = Colored(self.ctx.arch.main_label(), color.then_some(Color::Green));
                format!("Consider adding a label called {main} to an instruction")
            }
            Kind::MainInLibrary | Kind::MainOutsideCode => "Label defined here".into(),
            Kind::MemorySectionFull { .. } => {
                "This element doesn't fit in the available space".into()
            }
            Kind::DataUnaligned { .. } => "This value isn't aligned".into(),
            Kind::UnallowedStatementType { .. } => {
                "This statement can't be used in the current section".into()
            }
            Kind::UnallowedLabel | Kind::UnallowedFloat(..) => "This value can't be used".into(),
            Kind::UnallowedFloatOperation(..)
            | Kind::DivisionBy0(..)
            | Kind::RemainderWith0(..)
            | Kind::ShiftOutOfRange(..) => "This operation can't be performed".into(),
            Kind::UnallowedNegativeValue(val) | Kind::IntegerOutOfRange(val, _) => {
                format!("This expression has value {}", Colored(val, red))
            }
            Kind::PseudoinstructionError { .. } => "While expanding this pseudoinstruction".into(),
        }
    }

    fn msg(&self, color: bool) -> String {
        let red = color.then_some(Color::Red);
        let blue = color.then_some(Color::BrightBlue);
        let main = Colored(self.ctx.arch.main_label(), red);
        match self.error.kind.as_ref() {
            Kind::UnknownDirective(s) => {
                format!("Directive {} isn't defined", Colored(s, red))
            }
            Kind::UnknownInstruction(s) => {
                format!("Instruction {} isn't defined", Colored(s, red))
            }
            Kind::UnknownLabel(s) => format!("Label {} isn't defined", Colored(s, red)),
            Kind::UnknownRegisterFile(s) => {
                format!("Register file of type {} isn't defined", Colored(s, red))
            }
            Kind::UnknownRegister { name, file } => format!(
                "Register {} isn't defined in file type {}",
                Colored(name, red),
                Colored(file, blue)
            ),
            Kind::UnknownEnumType(t) => format!("Enum type {} isn't defined", Colored(t, red)),
            Kind::UnknownEnumValue { value, enum_name } => format!(
                "Value {} isn't defined in enum type {}",
                Colored(value, red),
                Colored(enum_name, blue)
            ),
            Kind::UnknownModifier(s) => {
                format!(
                    "Modifier {}{} isn't defined",
                    Colored('%', red),
                    Colored(s, red)
                )
            }
            Kind::IncorrectInstructionSyntax(..) => "Incorrect instruction syntax".into(),
            Kind::IncorrectDirectiveArgumentNumber { expected, found } => format!(
                "Incorrect amount of arguments, expected {}{} but found {}",
                if expected.at_least { "at least " } else { "" },
                Colored(expected.amount, blue),
                Colored(found, red),
            ),
            Kind::IncorrectArgumentType { expected, found } => format!(
                "Incorrect argument type, expected {} but found {}",
                Colored(expected, blue),
                Colored(found, red),
            ),
            Kind::DuplicateLabel(s, _) => {
                format!("Label {} is already defined", Colored(s, red))
            }
            Kind::MissingMainLabel => format!("Main label {main} not found"),
            Kind::MainInLibrary => format!("Main label {main} can't be used in libraries"),
            Kind::MainOutsideCode => {
                format!("Main label {main} defined outside of the text segment")
            }
            Kind::MemorySectionFull { section, .. } => {
                format!("{section} memory segment isn't big enough")
            }
            Kind::DataUnaligned { address, alignment } => format!(
                "Data at address {} isn't aligned to size {} nor word size {}",
                Colored(format!("{address:#X}"), red),
                Colored(alignment, blue),
                Colored(self.ctx.arch.word_size().div_ceil(8), blue),
            ),
            Kind::UnallowedStatementType { section, found } => {
                let found = if found.is_code() {
                    "instruction"
                } else {
                    "data directive"
                };
                let found = Colored(found, red);
                let section = section
                    .as_ref()
                    .map_or_else(|| "None".into(), |(s, _)| s.to_string());
                let section = Colored(section, blue);
                format!("Can't use {found} statements while in section {section}",)
            }
            Kind::UnallowedLabel => "Can't use labels in literal expressions".into(),
            Kind::UnallowedFloat(..) => {
                "Can't use floating point values in integer expressions".into()
            }
            Kind::UnallowedFloatOperation(op, ..) => format!(
                "Can't perform the {} operation with floating point numbers",
                Colored(op, red),
            ),
            Kind::UnallowedNegativeValue(_) => "Negative values aren't allowed here".into(),
            Kind::IntegerOutOfRange(val, _) => format!(
                "Value {} is outside of the valid range of the field",
                Colored(val, red)
            ),
            Kind::DivisionBy0(..) => "Can't divide by 0".into(),
            Kind::RemainderWith0(..) => "Can't take the remainder of a division by 0".into(),
            Kind::ShiftOutOfRange(_, x) if *x < BigInt::ZERO => {
                "Can't perform a shift by negative".into()
            }
            Kind::ShiftOutOfRange(..) => "Can't perform a shift by big integer".into(),
            Kind::PseudoinstructionError { name, .. } => {
                let name = Colored(name, red);
                format!("Error while expanding pseudoinstruction {name}")
            }
        }
    }
}

/// Wrapper for the cache of buffers to display in the error message
struct SourceCache<'src, 'name>(Vec<Spanned<Source<&'src str>>>, &'name str);

impl<'src> ariadne::Cache<usize> for SourceCache<'src, '_> {
    type Storage = &'src str;

    fn fetch(&mut self, id: &usize) -> Result<&Source<Self::Storage>, impl fmt::Debug> {
        self.0
            .get(*id)
            .map(|(s, _)| s)
            .ok_or_else(|| format!("Failed to fetch source `{id}`"))
    }

    fn display<'a>(&self, id: &'a usize) -> Option<impl fmt::Display + 'a> {
        Some(self.display(*id))
    }
}

impl<'src, 'name> SourceCache<'src, 'name> {
    /// Display a given source ID
    fn display(&self, id: usize) -> String {
        if id == self.0.len() - 1 {
            // The last id is the user code
            self.1.to_owned()
        } else {
            format!("<expansion {}>", self.0.len() - id - 1)
        }
    }

    /// Creates a new [`SourceCache`] with the given data
    ///
    /// # Parameters
    ///
    /// * `file_cache`: pseudoinstruction definition cache
    /// * `src`: user assembly code
    /// * `filename`: filename for the user assembly code
    /// * `span`: location span where the error occurred
    fn new(file_cache: &'src FileCache, src: &'src str, filename: &'name str, span: Span) -> Self {
        let sources = file_cache
            .context(src, span)
            .into_iter()
            .map(|(buf, range)| (Source::from(buf), range))
            .collect();
        Self(sources, filename)
    }
}

impl crate::RenderError for Error<'_> {
    fn format(&self, filename: &str, src: &str, mut buffer: &mut Vec<u8>, color: bool) {
        let mut sources = SourceCache::new(&self.ctx.file_cache, src, filename, self.error.span);
        let file_id = 0;
        let span = sources.0[file_id].1.into_range();
        let blue = color.then_some(Color::BrightBlue);
        let config = Config::default()
            .with_color(color)
            .with_index_type(IndexType::Byte);
        let mut report = Report::build(ReportKind::Error, (file_id, span.clone()))
            .with_config(config)
            .with_code(format!("E{:02}", self.code()))
            .with_message(self.msg(color))
            .with_label(
                Label::new((file_id, span))
                    .with_message(self.label(color))
                    .with_color(Color::Red),
            )
            .with_labels(
                // Add pseudoinstruction context first to order sources in output
                sources.0[1..]
                    .iter()
                    .enumerate()
                    // Reverse sources to display user code at the top
                    .rev()
                    .map(|(id, (_, span))| {
                        Label::new((id + 1, span.into_range()))
                            .with_message("Generated by this pseudoinstruction")
                            .with_color(Color::BrightCyan)
                    }),
            )
            .with_labels(self.context(color).into_iter().map(|label| {
                let mut ids = sources.0.iter().enumerate();
                let id = ids
                    .find_map(|(i, (_, id))| (id.context == label.0.context).then_some(i))
                    .expect("Context labels should always point to a source in the cache");
                Label::new((id, label.0.into_range()))
                    .with_message(format!("{} {}", "Note:".fg(blue), label.1))
                    .with_color(Color::BrightBlue)
            }));
        if let Some(note) = self.note(color) {
            report.set_note(note);
        }
        if let Some(hint) = self.hint(color) {
            report.set_help(hint);
        }

        report
            .finish()
            .write(&mut sources, &mut buffer)
            .expect("Writing to an in-memory vector shouldn't fail");

        match self.error.kind.as_ref() {
            Kind::IncorrectInstructionSyntax(errs) => {
                for (syntax, err) in errs {
                    writeln!(
                        &mut buffer,
                        "\nThe syntax {} failed with the following reason:",
                        Colored(syntax, blue)
                    )
                    .expect("Writing to an in-memory vector can't fail");
                    let src = sources.0[file_id].0.text();
                    err.format(&sources.display(file_id), src, buffer, color);
                }
            }
            Kind::PseudoinstructionError { error, .. } => {
                writeln!(&mut buffer).expect("Writing to an in-memory vector can't fail");
                error.format(filename, src, buffer, color);
            }
            _ => {}
        }
    }
}

impl Info for PseudoinstructionError {
    fn note(&self, color: bool) -> Option<String> {
        use PseudoinstructionErrorKind as Kind;
        Some(match &self.kind {
            Kind::UnknownFieldNumber { size, .. } => {
                format!(
                    "The pseudoinstruction has {}",
                    ArgNum(*size, color.then_some(Color::BrightBlue))
                )
            }
            _ => return None,
        })
    }

    fn label(&self, _: bool) -> String {
        use PseudoinstructionErrorKind as Kind;
        match &self.kind {
            Kind::UnknownFieldName(..) => "Unknown field name",
            Kind::UnknownFieldNumber { .. } => "Field index out of bounds",
            Kind::UnknownFieldType(..) => "Unknown field type",
            Kind::EmptyBitRange => "Empty bit range",
            Kind::BitRangeOutOfBounds { .. } => "Bit range out of bounds",
            Kind::EvaluationError(..) => "While evaluating this code",
            Kind::ParseError { .. } => "While parsing this instruction",
        }
        .into()
    }

    fn msg(&self, color: bool) -> String {
        use PseudoinstructionErrorKind as Kind;
        let red = color.then_some(Color::Red);
        match &self.kind {
            Kind::UnknownFieldName(s) => format!("Field {} isn't defined", Colored(s, red)),
            Kind::UnknownFieldNumber { idx, .. } => {
                format!("Field index {} is out of bounds", Colored(idx, red))
            }
            Kind::UnknownFieldType(s) => format!("Unknown field type {}", Colored(s, red)),
            Kind::EmptyBitRange => "Bit range is empty".into(),
            Kind::BitRangeOutOfBounds { upper_bound, msb } => format!(
                "Bit range is of bounds, upper bound is {} but the MSB is {}",
                Colored(upper_bound, red),
                Colored(msb, color.then_some(Color::BrightBlue)),
            ),
            Kind::EvaluationError(s) => format!("Error evaluating JS code:\n{s}"),
            Kind::ParseError(_) => "Error parsing instruction".into(),
        }
    }
}

impl crate::RenderError for PseudoinstructionError {
    fn format(&self, _: &str, _: &str, mut buffer: &mut Vec<u8>, color: bool) {
        static FILENAME: &str = "<pseudoinstruction expansion>";
        let src = &self.definition;
        let config = Config::default()
            .with_color(color)
            .with_index_type(IndexType::Byte);
        let mut report = Report::build(ReportKind::Error, (FILENAME, self.span.clone()))
            .with_config(config)
            .with_message(self.msg(color))
            .with_label(
                Label::new((FILENAME, self.span.clone()))
                    .with_message(self.label(color))
                    .with_color(Color::Red),
            );
        if let Some(note) = self.note(color) {
            report.set_note(note);
        }
        report
            .finish()
            .write((FILENAME, Source::from(src)), &mut buffer)
            .expect("Writing to an in-memory vector shouldn't fail");
        writeln!(&mut buffer).expect("Writing to an in-memory vector can't fail");
        if let PseudoinstructionErrorKind::ParseError(err) = &self.kind {
            err.format(FILENAME, src, buffer, color);
        }
    }
}

/// Trait for promoting an error [`Kind`] wrapped in a [`Result`] to an [`Data`]
pub(crate) trait SpannedErr {
    /// Type wrapped in the Ok variant
    type T;

    /// Adds a span to the error kind, promoting it to an [`Data`]
    ///
    /// # Parameters
    ///
    /// * `span`: location in the assembly code that caused the error
    fn add_span(self, span: Span) -> Result<Self::T, Data>;
}

impl<T> SpannedErr for Result<T, Kind> {
    type T = T;
    fn add_span(self, span: Span) -> Result<T, Data> {
        self.map_err(|e| e.add_span(span))
    }
}
