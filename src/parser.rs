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

//! Module containing the definition of the assembly parser
//!
//! Contains the definition of the AST, with the entry point for parsing code being the [`parse()`]
//! function

use chumsky::{input::ValueInput, prelude::*};

use crate::span::{Span, Spanned};

mod expression;
pub use expression::Expr;

mod lexer;
pub use lexer::Token;

mod error;
pub use error::Error as ParseError;

pub mod instruction;
pub use instruction::Instruction;

/// Generic parser type definition
macro_rules! Parser {
    ($ilt:lifetime, $i:ty, $o:ty) => { impl Parser<$ilt, $i, $o, extra::Err<Rich<$ilt, <$i as Input<$ilt>>::Token, <$i as Input<$ilt>>::Span>>> + Clone };
    (boxed: $ilt:lifetime, $i:ty, $o:ty) => { Boxed<$ilt, $ilt, $i, $o, extra::Err<Rich<$ilt, <$i as Input<$ilt>>::Token, <$i as Input<$ilt>>::Span>>> };
}
use Parser;

/// AST node for directive arguments
#[derive(Debug, PartialEq, Clone)]
pub enum Data {
    /// String literal
    String(String),
    /// Numeric expression
    Number(Expr),
}

/// AST node for instructions
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct InstructionNode {
    /// Name of the instruction
    pub name: Spanned<String>,
    /// Arguments of the instruction, to be checked/used on the semantic stage
    pub args: Spanned<Vec<Spanned<Token>>>,
}

/// AST node for directives
#[derive(Debug, PartialEq, Clone)]
pub struct DirectiveNode {
    /// Name of the directive
    pub name: Spanned<String>,
    /// Arguments of the directive
    pub args: Spanned<Vec<Spanned<Data>>>,
}

/// AST node for statements
#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Directive(DirectiveNode),
    Instruction(InstructionNode),
}

/// Top level AST node
#[derive(Debug, PartialEq, Clone)]
pub struct ASTNode {
    /// Labels attached to the node
    pub labels: Vec<Spanned<String>>,
    /// Statement of the node
    pub statement: Spanned<Statement>,
}

/// Abstract Syntax Tree representing the assembly code
pub type AST = Vec<ASTNode>;

/// Creates a parser for the tokenized input
#[must_use]
fn parser<'tokens, I>() -> Parser!('tokens, I, AST)
where
    I: ValueInput<'tokens, Token = Token, Span = Span>,
{
    // Newline token
    let newline = || just(Token::Ctrl('\n'));
    // Identifiers
    let ident = select! { Token::Identifier(ident) = e => (ident, e.span())}.labelled("identifier");
    let label = select! { Token::Label(name) = e => (name, e.span())}
        .padded_by(newline().repeated())
        .labelled("label");
    let directive_name =
        select! { Token::Directive(name) = e => (name, e.span())}.labelled("directive name");

    // Any amount of labels: `labels -> label*`
    let labels = label.repeated().collect().labelled("labels");

    // Directive statement:
    // `directive -> (directive_name \n)
    //             | (directive_name expression (\n* , \n* expression)* \n)`
    let directive = directive_name
        .then(
            // Arguments of the directive. Comma-separated list of expressions. Each expression can
            // have any amount of newlines prefixing it except the first (to avoid ambiguities in
            // cases where a directive without arguments is followed by an instruction), and any
            // amount of newlines following it if they are followed by a comma (indicating that more
            // expressions will follow, otherwise a single newline is required as the statement end)
            newline()
                .map_with(|_, e| {
                    let mut s: Span = e.span();
                    // Don't include the newline in the span. We need to do this manually here
                    // because getting the span after `.rewind()` gives the wrong span
                    s.end = s.start;
                    (Vec::new(), s)
                })
                .rewind()
                .or(expression::parser()
                    .map(|(expr, _)| Data::Number(expr))
                    .or(select! { Token::String(s) => Data::String(s) }.labelled("string"))
                    .map_with(|x, e| (x, e.span()))
                    .separated_by(just(Token::Ctrl(',')).padded_by(newline().repeated()))
                    .collect()
                    .map_with(|x, e| (x, e.span()))),
        )
        .map(|(name, args)| Statement::Directive(DirectiveNode { name, args }))
        .labelled("directive");

    // Instruction: `instruction -> ident [^\n]*`
    let instruction = ident
        .then(
            none_of([Token::Ctrl('\n')])
                .map_with(|token, e| (token, e.span()))
                .repeated()
                .collect()
                .map_with(|args: Vec<_>, e| {
                    // Fix the span being wrong when there are no arguments (when sub-parser doesn't
                    // consume input). SEE: <https://github.com/zesterer/chumsky/issues/870>
                    let mut s: Span = e.span();
                    if args.is_empty() {
                        s.start = s.end;
                    }
                    (args, s)
                }),
        )
        .map(|(name, args)| Statement::Instruction(InstructionNode { name, args }))
        .labelled("instruction");

    // Statement: `statement -> labels [instruction | directive]`
    let statement = labels
        .then(directive.or(instruction).map_with(|x, e| (x, e.span())))
        .map(|(labels, statement)| ASTNode { labels, statement });

    // `code -> statement*`
    statement
        .separated_by(newline().repeated().at_least(1))
        .allow_leading()
        .allow_trailing()
        .collect()
}

/// Tokenizes an input and parses it with a given parser
///
/// # Parameters
///
/// * `parser`: parser to use with the tokenized input
/// * `comment_prefix`: string to use as line comment prefix
/// * `src`: code to tokenize and parse
///
/// # Errors
///
/// Errors if the input either has an invalid token or it's syntactically invalid according to the
/// given parser
// NOTE: this has to be implemented with a macro in order for type inference to be able to get the
// correct types, since we can't specify them ourselves in a function signature
macro_rules! parse_with {
    ($parser:expr, $comment_prefix:expr, $src:expr) => {{
        use $crate::parser::{lexer, ParseError};
        use $crate::span::FileID;
        let end = Span::new(FileID::SRC, $src.len()..$src.len());
        let src = $src.with_context(FileID::SRC);
        || -> Result<_, ParseError> {
            let tokens = lexer::lexer($comment_prefix).parse(src).into_result()?;
            let tokens = tokens.map(end, |(x, s)| (x, s));
            let res = $parser.parse(tokens).into_result()?;
            Ok(res)
        }()
    }};
}
#[allow(unused_imports)] // This is used below, but clippy doesn't seem to detect it
use parse_with;

/// Parses the input creating an abstract syntax tree
///
/// # Parameters
///
/// * `comment_prefix`: string to use as line comment prefix
/// * `src`: code to parse
///
/// # Errors
///
/// Errors if the input is syntactically invalid
pub fn parse(comment_prefix: &str, src: &str) -> Result<AST, ParseError> {
    parse_with!(parser(), comment_prefix, src)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::span::test::*;

    fn test(test_cases: Vec<(&str, AST)>) {
        for (src, ast) in test_cases {
            assert_eq!(super::parse("#", src), Ok(ast), "`{src}`");
        }
    }

    #[must_use]
    fn into<T>(x: Ranged<T>) -> Spanned<T> {
        (x.0, x.1.span())
    }

    #[must_use]
    fn owned<O, T: ToOwned<Owned = O> + ?Sized>(x: Ranged<&T>) -> Spanned<O> {
        (x.0.to_owned(), x.1.span())
    }

    #[must_use]
    fn directive(
        labels: Vec<Ranged<&str>>,
        name: Ranged<&str>,
        args: Ranged<Vec<Ranged<Data>>>,
        span: Range,
    ) -> ASTNode {
        ASTNode {
            labels: labels.into_iter().map(owned).collect(),
            statement: (
                Statement::Directive(DirectiveNode {
                    name: owned(name),
                    args: (args.0.into_iter().map(into).collect(), args.1.span()),
                }),
                span.span(),
            ),
        }
    }

    #[must_use]
    fn instruction(
        labels: Vec<Ranged<&str>>,
        name: Ranged<&str>,
        args: Ranged<Vec<Ranged<Token>>>,
        span: Range,
    ) -> ASTNode {
        ASTNode {
            labels: labels.into_iter().map(owned).collect(),
            statement: (
                Statement::Instruction(InstructionNode {
                    name: owned(name),
                    args: (args.0.into_iter().map(into).collect(), args.1.span()),
                }),
                span.span(),
            ),
        }
    }

    #[test]
    fn directives() {
        let empty = vec![directive(vec![], (".name", 0..5), (vec![], 5..5), 0..5)];
        test(vec![
            (".name\n", empty.clone()),
            (".name\r", empty.clone()),
            (".name\r\n", empty.clone()),
            (".name", empty),
            (
                ".name \"a\"\n",
                vec![directive(
                    vec![],
                    (".name", 0..5),
                    (vec![(Data::String("a".into()), 6..9)], 6..9),
                    0..9,
                )],
            ),
            (".name \"󰉀\"\n", {
                let len = 8 + '󰉀'.len_utf8();
                vec![directive(
                    vec![],
                    (".name", 0..5),
                    (vec![(Data::String("󰉀".into()), 6..len)], 6..len),
                    0..len,
                )]
            }),
            (
                ".name \"a\", 1, b\n",
                vec![directive(
                    vec![],
                    (".name", 0..5),
                    (
                        vec![
                            (Data::String("a".into()), 6..9),
                            (Data::Number(Expr::Integer(1u8.into())), 11..12),
                            (
                                Data::Number(Expr::Identifier(("b".into(), (14..15).span()))),
                                14..15,
                            ),
                        ],
                        6..15,
                    ),
                    0..15,
                )],
            ),
            (
                "label: .name\n",
                vec![directive(
                    vec![("label", 0..6)],
                    (".name", 7..12),
                    (vec![], 12..12),
                    7..12,
                )],
            ),
            (
                "a: b: .name\n",
                vec![directive(
                    vec![("a", 0..2), ("b", 3..5)],
                    (".name", 6..11),
                    (vec![], 11..11),
                    6..11,
                )],
            ),
        ]);
    }

    #[test]
    fn directives_newlines() {
        test(vec![
            (
                ".name 1,\n 2\n",
                vec![directive(
                    vec![],
                    (".name", 0..5),
                    (
                        vec![
                            (Data::Number(Expr::Integer(1u8.into())), 6..7),
                            (Data::Number(Expr::Integer(2u8.into())), 10..11),
                        ],
                        6..11,
                    ),
                    0..11,
                )],
            ),
            (
                ".name 1\n, 2\n",
                vec![directive(
                    vec![],
                    (".name", 0..5),
                    (
                        vec![
                            (Data::Number(Expr::Integer(1u8.into())), 6..7),
                            (Data::Number(Expr::Integer(2u8.into())), 10..11),
                        ],
                        6..11,
                    ),
                    0..11,
                )],
            ),
            (
                ".name 1\n,\n 2\n",
                vec![directive(
                    vec![],
                    (".name", 0..5),
                    (
                        vec![
                            (Data::Number(Expr::Integer(1u8.into())), 6..7),
                            (Data::Number(Expr::Integer(2u8.into())), 11..12),
                        ],
                        6..12,
                    ),
                    0..12,
                )],
            ),
            (
                ".name 1\n\n,\n\n\n 2\n",
                vec![directive(
                    vec![],
                    (".name", 0..5),
                    (
                        vec![
                            (Data::Number(Expr::Integer(1u8.into())), 6..7),
                            (Data::Number(Expr::Integer(2u8.into())), 14..15),
                        ],
                        6..15,
                    ),
                    0..15,
                )],
            ),
        ]);
    }

    #[test]
    fn instructions() {
        let empty = vec![instruction(vec![], ("name", 0..4), (vec![], 4..4), 0..4)];
        test(vec![
            ("name\n", empty.clone()),
            ("name\r", empty.clone()),
            ("name\r\n", empty.clone()),
            ("name \n", empty.clone()),
            ("name", empty),
            (
                "name a ",
                vec![instruction(
                    vec![],
                    ("name", 0..4),
                    (vec![(Token::Identifier("a".into()), 5..6)], 5..6),
                    0..6,
                )],
            ),
            (
                "name a\n",
                vec![instruction(
                    vec![],
                    ("name", 0..4),
                    (vec![(Token::Identifier("a".into()), 5..6)], 5..6),
                    0..6,
                )],
            ),
            (
                "name a, 1\n",
                vec![instruction(
                    vec![],
                    ("name", 0..4),
                    (
                        vec![
                            (Token::Identifier("a".into()), 5..6),
                            (Token::Ctrl(','), 6..7),
                            (Token::Integer(1u8.into()), 8..9),
                        ],
                        5..9,
                    ),
                    0..9,
                )],
            ),
            (
                "label: name\n",
                vec![instruction(
                    vec![("label", 0..6)],
                    ("name", 7..11),
                    (vec![], 11..11),
                    7..11,
                )],
            ),
            (
                "a: b: name\n",
                vec![instruction(
                    vec![("a", 0..2), ("b", 3..5)],
                    ("name", 6..10),
                    (vec![], 10..10),
                    6..10,
                )],
            ),
            (
                "\na: \n\tb: \nname\n",
                vec![instruction(
                    vec![("a", 1..3), ("b", 6..8)],
                    ("name", 10..14),
                    (vec![], 14..14),
                    10..14,
                )],
            ),
        ]);
    }

    #[test]
    fn mixed() {
        test(vec![
            (
                "name\n .dir\n",
                vec![
                    instruction(vec![], ("name", 0..4), (vec![], 4..4), 0..4),
                    directive(vec![], (".dir", 6..10), (vec![], 10..10), 6..10),
                ],
            ),
            (
                "name\r\n .dir\r\n",
                vec![
                    instruction(vec![], ("name", 0..4), (vec![], 4..4), 0..4),
                    directive(vec![], (".dir", 7..11), (vec![], 11..11), 7..11),
                ],
            ),
            (
                ".dir\n name\n",
                vec![
                    directive(vec![], (".dir", 0..4), (vec![], 4..4), 0..4),
                    instruction(vec![], ("name", 6..10), (vec![], 10..10), 6..10),
                ],
            ),
            (
                "a: .dir 1\n b: name\n",
                vec![
                    directive(
                        vec![("a", 0..2)],
                        (".dir", 3..7),
                        (vec![(Data::Number(Expr::Integer(1u8.into())), 8..9)], 8..9),
                        3..9,
                    ),
                    instruction(
                        vec![("b", 11..13)],
                        ("name", 14..18),
                        (vec![], 18..18),
                        14..18,
                    ),
                ],
            ),
        ]);
    }

    #[test]
    fn empty() {
        test(vec![
            ("", vec![]),
            ("\n", vec![]),
            ("\r", vec![]),
            ("\r\n", vec![]),
            ("\n\n", vec![]),
            ("\r\n\n\r", vec![]),
        ]);
    }
}
