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

//! Module containing conversion methods between the format used by the architecture JSON
//! specification and our internal representation

use schemars::JsonSchema;
use serde::Deserialize;

use super::{utils, DirectiveAction};
use super::{AlignmentType, FloatType, IntegerType, StringType};
use utils::NonEmptyRangeInclusive;

/// Directive specification
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Directive<'a> {
    /// Name of the directive
    pub name: &'a str,
    /// Action of the directive
    pub action: DirectiveAction<DirectiveData>,
    /// Size in bytes of values associated with this directive
    #[serde(default)]
    pub size: Option<usize>,
}

/// Data segment types
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(untagged)]
pub enum DirectiveData {
    /// Store n * size null bytes in the data segment
    Space(Space),
    /// Store string
    String(StringType),
    /// Store integer
    Int(IntegerType),
    /// Store floating point value
    Float(FloatType),
    /// Align the next data value to a given size
    Alignment(AlignmentType),
}

/// Store n * size null bytes in the data segment
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum Space {
    Space,
}

impl<'a> TryFrom<Directive<'a>> for super::Directive<'a> {
    type Error = &'static str;

    fn try_from(value: Directive<'a>) -> Result<Self, Self::Error> {
        use super::DirectiveData as SDD;
        use DirectiveData as DD;

        let size = value.size.ok_or("missing required size field value");
        Ok(Self {
            name: value.name,
            action: match value.action {
                DirectiveAction::Data(data_type) => DirectiveAction::Data(match data_type {
                    DD::Space(_) => SDD::Space(size?),
                    DD::Int(x) => SDD::Int(size?, x),
                    DD::Float(x) => SDD::Float(x),
                    DD::String(x) => SDD::String(x),
                    DD::Alignment(x) => SDD::Alignment(x),
                }),
                DirectiveAction::Segment(x) => DirectiveAction::Segment(x),
                DirectiveAction::GlobalSymbol(x) => DirectiveAction::GlobalSymbol(x),
                DirectiveAction::Nop(x) => DirectiveAction::Nop(x),
            },
        })
    }
}

/// Range of bits of a field in a binary instruction
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
pub struct BitRange {
    /// Starting position of the field, ignored for pseudoinstructions. Will be applied from the
    /// MSB of the value to the LSB
    pub startbit: BitPosition,
    /// End position of the field, ignored for pseudoinstructions. Will be applied from the MSB of
    /// the value to the LSB
    pub stopbit: BitPosition,
    /// Amount of least significant bits from the value that should be ignored
    #[serde(default)]
    pub padding: usize,
}

/// Position of the start/end bit of a field in a binary instruction
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
#[serde(untagged)]
pub enum BitPosition {
    // Field uses a single, contiguous bit range
    Single(usize),
    // Field uses multiple, discontiguous bit ranges
    Multiple(Vec<usize>),
}

impl TryFrom<BitRange> for super::BitRange {
    type Error = &'static str;

    fn try_from(value: BitRange) -> Result<Self, Self::Error> {
        let range = |(msb, lsb)| {
            NonEmptyRangeInclusive::<usize>::build(lsb, msb).ok_or("invalid empty range")
        };
        let ranges = match (value.startbit, value.stopbit) {
            (BitPosition::Single(msb), BitPosition::Single(lsb)) => vec![range((msb, lsb))?],
            (BitPosition::Multiple(msb), BitPosition::Multiple(lsb)) => {
                if msb.len() != lsb.len() {
                    return Err("the startbit and endbit fields must have the same length if they are vectors");
                }
                std::iter::zip(msb, lsb)
                    .map(range)
                    .collect::<Result<_, _>>()?
            }
            _ => return Err("the type of the startbit and endbit fields should be the same"),
        };
        Self::build(ranges, value.padding)
            .ok_or("the startbit and endbit fields must not be empty if they are vectors")
    }
}

/// Instruction syntax specification
#[derive(Deserialize, JsonSchema, Debug, PartialEq, Eq, Clone)]
pub struct InstructionSyntax<'a, BitRange> {
    /// Syntax specification of the instruction. `[fF]\d+` is interpreted as the field with index
    /// `i` of the instruction. Other characters are interpreted literally. Ex: `F0 F3 F1 (F2)`
    pub signature_definition: &'a str,
    /// Parameters of the instruction
    pub fields: Vec<super::InstructionField<'a, BitRange>>,
}

impl<'a, T> TryFrom<InstructionSyntax<'a, T>> for super::InstructionSyntax<'a, T> {
    type Error = &'static str;

    fn try_from(value: InstructionSyntax<'a, T>) -> Result<Self, Self::Error> {
        let parser = crate::parser::Instruction::build(value.signature_definition, &value.fields)?;
        Ok(Self {
            parser,
            output_syntax: value.signature_definition,
            fields: value.fields,
        })
    }
}
