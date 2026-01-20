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

#![doc = include_str!("../README.md")]
//! # Example
//!
//! Example usage of the assembler from Rust:
//!
//! ```
//! use creator_assembler::prelude::*;
//! use std::collections::HashMap;
//!
//! let arch_json = include_str!("../tests/architecture.json");
//! let arch = Architecture::from_json(arch_json).expect("The architecture should be correct");
//!
//! let code = "
//! .data
//! value: .word 5
//!        .zero 1
//!        .align 2
//! address: .word value
//! .text
//! main: nop
//! a: b: imm -3, 3, a
//!       reg PC, x2, fs0, F1
//! ";
//!
//! // Parse the code
//! let ast = parser::parse(arch.config.comment_prefix, code)
//!     .map_err(|e| eprintln!("{}", e.clone().render("file.s", code, true)))
//!     .expect("The code should be valid");
//! // Compile the code
//! let compiled = compiler::compile(&arch, ast, &0u8.into(), HashMap::new(), false)
//!     .map_err(|e| eprintln!("{}", e.clone().render("file.s", code, true)))
//!     .expect("The code should be valid");
//! ```

pub mod architecture;
pub mod compiler;
mod error_rendering;
#[cfg(feature = "js")]
mod js;
mod number;
pub mod parser;
pub mod span;

/// Module containing the default exports
pub mod prelude {
    pub use crate::architecture::Architecture;
    pub use crate::compiler;
    pub use crate::error_rendering::RenderError;
    pub use crate::parser;
    pub use num_bigint::BigUint;
}

use error_rendering::RenderError;

/// Builds a new lazily-initialized regex with a given literal string
///
/// # Panics
///
/// Panics if the literal string isn't a valid regex
macro_rules! build_regex {
    ($re:expr) => {
        LazyLock::new(|| Regex::new($re).expect("All regexes should compile"))
    };
}
use build_regex as regex;
