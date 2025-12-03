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

#![cfg(feature = "cli")]

use std::{collections::HashMap, process::ExitCode};

use clap::{Parser, Subcommand};

use num_bigint::BigUint;

use creator_assembler::prelude::*;

/// Command-line arguments parser
#[derive(Parser)]
#[command(version, about, long_about = None, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Available CLI commands
#[derive(Subcommand, PartialEq, Clone)]
enum Command {
    /// Print the architecture schema to `stdout` and exit
    ///
    /// This schema can be used to validate the architecture JSON file, although it's not
    /// guaranteed to catch all possible errors. It also provides documentation for all the
    /// attributes
    Schema,
    /// Validate the given architecture specification
    ///
    /// Load the architecture JSON file, validate it, and print to `stdout` a debug representation
    /// of the specification as loaded for use during compilation
    Validate {
        /// Path to the architecture specification file
        architecture: String,
    },
    /// Compile a given assembly code according to an architecture specification and print the
    /// result to `stdout`
    ///
    /// Compiles an assembly code and prints a debug representation of the result. This includes:
    ///
    ///   * Symbol table for labels defined, including the address they point to and the span where
    ///     they were defined
    ///   * Instructions compiled, including their address, binary representation, textual
    ///     representation with labels replaced with their addresses, and span where they were
    ///     specified
    ///   * Data elements compiled, including their type, value, and addresses
    #[command(verbatim_doc_comment)]
    Compile {
        /// Path to the architecture specification file
        architecture: String,
        /// Path to the assembly code file
        code: String,
        /// Enable verbose output. Prints the assembly code AST as well
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Execution error
#[derive(Debug)]
enum Error {
    /// Error reading a file
    ReadFile(String, std::io::Error),
    /// Error parsing/validating architecture specification
    ParseArchitecture(serde_json::Error),
    /// Error parsing/compiling the assembly code
    Compilation(String),
}

/// Reads a file to a string
fn read_file(filename: &str) -> Result<String, Error> {
    std::fs::read_to_string(filename).map_err(|e| Error::ReadFile(filename.to_owned(), e))
}

/// Parses and validates an architecture from a JSON string
fn build_architecture(arch: &str) -> Result<Architecture<'_>, Error> {
    Architecture::from_json(arch).map_err(Error::ParseArchitecture)
}

/// Runs the application
fn run() -> Result<(), Error> {
    let args = Cli::parse(); // Parse command-line arguments
    match args.command {
        Command::Schema => println!("{}", Architecture::schema()),
        Command::Validate { architecture } => {
            let arch = read_file(&architecture)?;
            let arch = build_architecture(&arch)?;
            println!("{arch:#?}");
        }
        Command::Compile {
            architecture,
            code,
            verbose,
        } => {
            // Read the source files
            let arch = read_file(&architecture)?;
            let src = read_file(&code)?;
            // Parse the architecture
            let arch = build_architecture(&arch)?;
            // Parse the assembly code
            let ast = parser::parse(arch.comment_prefix(), &src)
                .map_err(|e| Error::Compilation(e.render(&code, &src, true)))?;
            // Print AST if asked
            if verbose {
                println!("\n\x1B[1;32m============================== AST ==============================\x1B[0m\n");
                println!("{ast:#?}");
            }
            // Compile the assembly code
            let compiled = compiler::compile(&arch, ast, &BigUint::ZERO, HashMap::new(), false)
                .map_err(|e| Error::Compilation(e.render(&code, &src, true)))?;
            // Print the compiled code
            println!("\n\x1B[1;32m========================= Compiled Code =========================\x1B[0m\n");
            println!("{compiled:#?}");
        }
    }
    Ok(())
}

/// Main entry point
fn main() -> ExitCode {
    let (x, msg) = match run() {
        Err(Error::ReadFile(file, e)) => {
            (1, format!("Can't read file `\x1B[33m{file}\x1B[0m`: {e}"))
        }
        Err(Error::ParseArchitecture(e)) => (2, format!("Can't parse architecture: {e}")),
        Err(Error::Compilation(e)) => {
            eprintln!("{e}");
            return 0.into();
        }
        _ => return ExitCode::SUCCESS,
    };
    eprintln!("\x1B[1;31m[Error]\x1B[0m {msg}");
    x.into()
}
