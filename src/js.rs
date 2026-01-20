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

//! Module containing the definition of wrappers for the compiler and generation of `JS` bindings
//! for interoperability

use std::collections::HashMap;
use std::hash::RandomState;
use std::str::FromStr;

use js_sys::BigInt;
use self_cell::self_cell;
use wasm_bindgen::prelude::*;

use crate::architecture::{Architecture, Integer};
use crate::RenderError;

// Creates a hook for panics to improve error messages
pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

self_cell!(
    /// Architecture definition
    #[wasm_bindgen]
    pub struct ArchitectureJS {
        owner: String,
        #[covariant]
        dependent: Architecture,
    }
);

/// Converts a given string with ANSI escape codes to HTML
///
/// # Panics
///
/// Panics if the string contains invalid ANSI escape codes
#[must_use]
fn to_html(str: &str) -> String {
    let converter = ansi_to_html::Converter::default().four_bit_var_prefix(Some("err-".into()));
    converter
        .convert(str)
        .expect("We should only generate valid ANSI escapes")
}

/// Converts a number to a `JS` big integer
#[must_use]
fn to_js_bigint<T: num_traits::Num + ToString>(x: &T) -> BigInt {
    BigInt::from_str(&x.to_string())
        .expect("Converting a number to string should always return a valid format")
}

/// Method used to render colors in error messages
#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    /// Use HTML tags, intended for display in browsers
    Html,
    /// Use ANSI escape codes, intended for display in terminals
    Ansi,
    /// Disable all formatting, using only plain text
    Off,
}

#[wasm_bindgen]
#[allow(clippy::use_self)] // wasm_bindgen doesn't support using `Self` on nested types
impl ArchitectureJS {
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
    pub fn from_json(json: String) -> Result<ArchitectureJS, String> {
        set_panic_hook();
        Self::try_new(json, |json| Architecture::from_json(json)).map_err(|e| e.to_string())
    }

    /// Converts the architecture to a pretty printed string for debugging
    #[wasm_bindgen(js_name = toString)]
    #[must_use]
    pub fn debug(&self) -> String {
        format!("{:#?}", self.borrow_dependent())
    }

    /// Compiles an assembly source according to the architecture description
    ///
    /// # Parameters
    ///
    /// * `src`: assembly code to compile
    /// * `reserved_offset`: amount of bytes that should be reserved for library instructions
    /// * `labels`: mapping from label names specified in the library to their addresses, in `JSON`
    /// * `library`: whether the code should be compiled as a library (`true`) or not (`false`)
    /// * `color`: method used to render colors in error messages
    ///
    /// # Errors
    ///
    /// Errors if the assembly code has a syntactical or semantical error, or if the `labels`
    /// parameter is either an invalid `JSON` or has invalid mappings
    pub fn compile(
        &self,
        src: &str,
        reserved_offset: usize,
        labels: &str,
        library: bool,
        color: Color,
    ) -> Result<CompiledCodeJS, String> {
        const FILENAME: &str = "assembly";
        let format_err = |e: String| if color == Color::Html { to_html(&e) } else { e };
        let labels: HashMap<String, Integer> =
            serde_json::from_str(labels).map_err(|e| e.to_string())?;
        let labels: HashMap<_, _, RandomState> =
            labels.into_iter().map(|(k, v)| (k, v.0)).collect();
        let arch = self.borrow_dependent();
        // Parse the source to an AST
        let ast = crate::parser::parse(arch.comment_prefix(), src)
            .map_err(|e| format_err(e.render(FILENAME, src, color != Color::Off)))?;
        // Compile the AST
        let compiled =
            crate::compiler::compile(arch, ast, &reserved_offset.into(), labels, library)
                .map_err(|e| format_err(e.render(FILENAME, src, color != Color::Off)))?;
        // Wrap the instructions in a type that can be returned to `JS`
        let instructions = compiled
            .instructions
            .into_iter()
            .map(|x| InstructionJS {
                address: format!("0x{:X}", x.address),
                labels: x.labels,
                loaded: x.loaded,
                binary: x.binary.into(),
                user: src[x.user].to_owned(),
            })
            .collect();
        // Wrap the data elements in a type that can be returned to `JS`
        let data = compiled.data_memory.into_iter().map(DataJS).collect();
        // Convert the label table to a type that can be returned to `JS`
        let label_table = compiled
            .label_table
            .into_iter()
            .map(|(name, label)| {
                let global = compiled.global_symbols.contains(&name);
                let address = to_js_bigint(label.address());
                LabelJS {
                    name,
                    address,
                    global,
                }
            })
            .collect();
        Ok(CompiledCodeJS {
            instructions,
            data,
            label_table,
        })
    }

    /// Generate a `JSON` schema
    #[must_use]
    pub fn schema() -> String {
        Architecture::schema()
    }
}

/// Assembly compilation output
#[wasm_bindgen(getter_with_clone)]
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledCodeJS {
    /// Compiled instructions to execute
    #[wasm_bindgen(readonly)]
    pub instructions: Vec<InstructionJS>,
    /// Compiled data to add to the data segment
    #[wasm_bindgen(readonly)]
    pub data: Vec<DataJS>,
    /// Symbol table for labels
    #[wasm_bindgen(readonly)]
    pub label_table: Vec<LabelJS>,
}

#[wasm_bindgen]
impl CompiledCodeJS {
    /// Converts the compiled code to a pretty printed string for debugging
    #[wasm_bindgen(js_name = toString)]
    #[must_use]
    pub fn debug(&self) -> String {
        format!("{self:#?}")
    }
}

/// Label table entry wrapper
#[wasm_bindgen(getter_with_clone)]
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LabelJS {
    /// Name of the label
    pub name: String,
    /// Address to which the label points
    pub address: BigInt,
    /// Whether the label is local to the file (`false`) or global
    pub global: bool,
}

/// Compiled instruction wrapper
#[wasm_bindgen(getter_with_clone)]
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct InstructionJS {
    /// Address of the instruction in hexadecimal (`0xABCD`)
    pub address: String,
    /// Labels pointing to this instruction
    pub labels: Vec<String>,
    /// Translated instruction to a simplified syntax
    pub loaded: String,
    /// Instruction encoded in binary
    pub binary: String,
    /// Instruction in the code
    pub user: String,
}

/// Compiled data wrapper
#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq)]
pub struct DataJS(crate::compiler::Data);

/// General category of a compiled data element
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataCategoryJS {
    /// Element represents a number
    Number,
    /// Element represents a string
    String,
    /// Element represents a reserved amount of space initialized to 0
    Space,
    /// Element represents padding that was added to align values
    Padding,
}

#[wasm_bindgen]
#[allow(clippy::missing_const_for_fn)] // wasm_bindgen doesn't support const functions
impl DataJS {
    /// Address of the data element
    #[must_use]
    pub fn address(&self) -> BigInt {
        to_js_bigint(&self.0.address)
    }

    /// Labels pointing to this data element
    #[must_use]
    pub fn labels(&self) -> Vec<String> {
        self.0.labels.clone()
    }

    /// Value of the data element:
    ///
    /// * For integers/floating point values, it's their value either in hexadecimal without the
    ///   `0x` prefix or as a number, depending on the `human` parameter
    /// * For strings, it's their contents
    /// * For empty spaces/padding, it's their size as a string
    ///
    /// # Parameters
    ///
    /// * `human`: whether to return the value as a human-readable representation or in hexadecimal
    #[must_use]
    pub fn value(&self, human: bool) -> String {
        use crate::compiler::Value;
        match (&self.0.value, human) {
            (Value::Integer(int), true) => format!("{}", int.value()),
            (Value::Integer(int), false) => format!("{:X}", int.value()),
            (Value::Float(float), true) => format!("{float}"),
            (Value::Float(float), false) => format!("{:X}", float.to_bits()),
            (Value::Double(double), true) => format!("{double}"),
            (Value::Double(double), false) => format!("{:X}", double.to_bits()),
            (Value::String { data, .. }, _) => data.clone(),
            (Value::Space(n) | Value::Padding(n), _) => n.to_string(),
        }
    }

    /// Precise type of the data element
    #[must_use]
    pub fn r#type(&self) -> String {
        use crate::architecture::IntegerType;
        use crate::compiler::Value;
        match &self.0.value {
            Value::Integer(int) => match int.r#type() {
                None => "bits",
                Some(IntegerType::Byte) => "byte",
                Some(IntegerType::HalfWord) => "half",
                Some(IntegerType::Word) => "word",
                Some(IntegerType::DoubleWord) => "double_word",
            },
            Value::Float(_) => "float",
            Value::Double(_) => "double",
            Value::String {
                null_terminated, ..
            } => match null_terminated {
                true => "asciiz",
                false => "ascii",
            },
            Value::Space(_) => "space",
            Value::Padding(_) => "padding",
        }
        .into()
    }

    /// General category of the data element
    #[must_use]
    pub fn data_category(&self) -> DataCategoryJS {
        use crate::compiler::Value;
        match self.0.value {
            Value::Integer(_) | Value::Float(_) | Value::Double(_) => DataCategoryJS::Number,
            Value::String { .. } => DataCategoryJS::String,
            Value::Space(_) => DataCategoryJS::Space,
            Value::Padding(_) => DataCategoryJS::Padding,
        }
    }

    /// Size of the data element in bytes
    #[must_use]
    pub fn size(&self) -> BigInt {
        use crate::compiler::Value;
        match &self.0.value {
            Value::Integer(int) => to_js_bigint(&int.size().div_ceil(8)),
            Value::Float(_) => BigInt::from(4),
            Value::Double(_) => BigInt::from(8),
            Value::String {
                data,
                null_terminated,
            } => to_js_bigint(&(data.len() + usize::from(*null_terminated))),
            Value::Space(x) | Value::Padding(x) => to_js_bigint(x),
        }
    }
}
