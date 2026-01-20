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

//! Module containing the definition of the error type used by syntax errors

use ariadne::{Color, Config, IndexType, Label, Report, ReportKind, Source};
use chumsky::error::{Rich, RichPattern, RichReason};

use super::Token;
use crate::error_rendering::{Colored, DisplayList};
use crate::span::Span;

/// Error representing a syntax error during parsing
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Error {
    /// The error happened during the lexing stage
    Lexer(Vec<Rich<'static, char, Span>>),
    /// The error happened during the parsing stage
    Parser(Vec<Rich<'static, Token, Span>>),
}

impl<'src> From<Vec<Rich<'src, char, Span>>> for Error {
    fn from(value: Vec<Rich<'src, char, Span>>) -> Self {
        Self::Lexer(value.into_iter().map(Rich::into_owned).collect())
    }
}

impl<'src> From<Vec<Rich<'src, Token, Span>>> for Error {
    fn from(value: Vec<Rich<'src, Token, Span>>) -> Self {
        Self::Parser(value.into_iter().map(Rich::into_owned).collect())
    }
}

impl<T: std::fmt::Display> crate::RenderError for Vec<Rich<'_, T, Span>> {
    fn format(&self, filename: &str, src: &str, mut buffer: &mut Vec<u8>, color: bool) {
        // Configure the error reports
        let config = Config::default()
            .with_color(color)
            .with_index_type(IndexType::Byte);
        let yellow = color.then_some(Color::Yellow);
        // Generate a report for each error
        for e in self {
            Report::build(ReportKind::Error, (filename, e.span().into_range()))
                .with_config(config)
                .with_message(match e.reason() {
                    RichReason::Custom(msg) => msg.clone(),
                    RichReason::ExpectedFound { expected, found } => {
                        let found = found
                            .as_deref()
                            .map_or_else(|| "end of input".into(), ToString::to_string);
                        let disp = |e: &RichPattern<'_, T>| match e {
                            // The implementation of ToString for Rich pattern adds quotes
                            // automatically, but we will add them later
                            RichPattern::Token(t) => t.to_string(),
                            RichPattern::Identifier(i) => i.clone(),
                            _ => e.to_string(),
                        };
                        format!(
                            "found {} but expected {}",
                            Colored(found, color.then_some(Color::Red)),
                            DisplayList::new(expected.iter().map(disp).collect(), color)
                        )
                    }
                })
                .with_label(
                    Label::new((filename, e.span().into_range()))
                        .with_message(match e.reason() {
                            RichReason::ExpectedFound { .. } => "Unexpected input",
                            RichReason::Custom(_) => "Caused by this",
                        })
                        .with_color(Color::Red),
                )
                .with_labels(e.contexts().map(|(label, span)| {
                    Label::new((filename, span.into_range()))
                        .with_message(format!("While parsing this {}", Colored(label, yellow)))
                        .with_color(Color::Yellow)
                }))
                .finish()
                .write((filename, Source::from(src)), &mut buffer)
                .expect("Writing to an in-memory vector can't fail");
        }
    }
}

impl crate::RenderError for Error {
    fn format(&self, filename: &str, src: &str, buffer: &mut Vec<u8>, color: bool) {
        match self {
            Self::Lexer(errs) => errs.format(filename, src, buffer, color),
            Self::Parser(errs) => errs.format(filename, src, buffer, color),
        }
    }
}
