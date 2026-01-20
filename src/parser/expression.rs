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

//! Module containing the definition of the expressions sub-parser as well as their evaluation
//!
//! The main entry point for creating the parser is the [`parser()`] function, with the evaluation
//! of methods being defined in the methods of the [`Expr`] type

use chumsky::pratt::{infix, left, prefix};
use chumsky::{input::ValueInput, prelude::*};
use num_bigint::{BigInt, BigUint};
use std::cmp::Ordering;

use super::{lexer::Operator, Parser, Span, Spanned, Token};
use crate::architecture::ModifierDefinitions;
use crate::compiler::error::SpannedErr;
use crate::compiler::{ErrorData, ErrorKind};
use crate::number::Number;

/// Allowed unary operations
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnaryOp {
    /// Unary plus, essentially a no-op
    Plus,
    /// Unary negation
    Minus,
    /// Unary binary complement
    Complement,
    /// Unary Modifier
    Modifier(String),
}

/// Allowed binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BinaryOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Remainder, with the same sign as the left operand
    Rem,
    /// Bitwise OR
    BitwiseOR,
    /// Bitwise AND
    BitwiseAND,
    /// Bitwise XOR
    BitwiseXOR,
    /// Greater
    Gt,
    /// Less
    Lt,
    /// Greater or equal
    Ge,
    /// Less or equal
    Le,
    /// Not equal
    Ne,
    /// Equal
    Eq,
    /// Boolean AND
    LogicalAnd,
    /// Boolean OR
    LogicalOr,
    /// Shift left
    Shl,
    /// Shift right
    Shr,
}

/// Mathematical expression on constant values
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Expr {
    /// Integer literal
    Integer(BigUint),
    /// Float literal
    Float(Spanned<f64>),
    /// Character literal
    Character(char),
    /// Identifier
    Identifier(Spanned<String>),
    /// Unary operation on other expressions
    UnaryOp {
        /// Operation to perform
        op: Spanned<UnaryOp>,
        /// Operand to perform the operation on
        operand: Box<Spanned<Self>>,
    },
    /// Binary operation on other expressions
    BinaryOp {
        /// Operation to perform
        op: Spanned<BinaryOp>,
        /// Left operand of the operation
        lhs: Box<Spanned<Self>>,
        /// Right operand of the operation
        rhs: Box<Spanned<Self>>,
    },
}

impl Expr {
    /// Evaluates the expression
    ///
    /// # Parameters
    ///
    /// * `ident_eval`: callback function to evaluate identifiers
    /// * `modifiers`: definitions of the modifiers allowed
    ///
    /// # Errors
    ///
    /// Returns a [`ErrorKind::UnallowedFloatOperation`] if an operation that's undefined with
    /// floats is attempted, a [`ErrorKind::DivisionBy0`] if a division by 0 is attempted, or any
    /// [`ErrorKind`] returned by the callback function
    pub fn eval(
        &self,
        ident_eval: impl Copy + Fn(&str) -> Result<BigInt, ErrorKind>,
        modifiers: &ModifierDefinitions,
    ) -> Result<Number, ErrorData> {
        match self {
            Self::Integer(value) => Ok(value.clone().into()),
            Self::Float((value, span)) => Ok(Number::from((*value, *span))),
            Self::Character(c) => Ok((*c as u32).into()),
            Self::Identifier((ident, span)) => Ok(ident_eval(ident).add_span(*span)?.into()),
            Self::UnaryOp { op, operand } => {
                let rhs = operand.0.eval(ident_eval, modifiers)?;
                match &op.0 {
                    UnaryOp::Plus => Ok(rhs),
                    UnaryOp::Minus => Ok(-rhs),
                    UnaryOp::Complement => (!rhs).add_span(op.1),
                    UnaryOp::Modifier(name) => {
                        let err = || ErrorKind::UnknownModifier(name.clone()).add_span(op.1);
                        let modifier = modifiers.get(name.as_str()).ok_or_else(err)?;
                        Ok(rhs.modify(*modifier))
                    }
                }
            }
            Self::BinaryOp { op, lhs, rhs } => {
                let lhs = lhs.0.eval(ident_eval, modifiers)?;
                let span = rhs.1;
                let rhs = rhs.0.eval(ident_eval, modifiers)?;
                #[rustfmt::skip]
                let to_int = |cond, ret: i32| Ok(if cond { ret.into() } else { BigInt::ZERO.into() });
                let replace_span = |e, span| match e {
                    ErrorKind::ShiftOutOfRange(_, x) => ErrorKind::ShiftOutOfRange(span, x),
                    e => e,
                };
                match op.0 {
                    BinaryOp::Add => Ok(lhs + rhs),
                    BinaryOp::Sub => Ok(lhs - rhs),
                    BinaryOp::Mul => Ok(lhs * rhs),
                    BinaryOp::Div => (lhs / rhs).ok_or(ErrorKind::DivisionBy0(span)),
                    BinaryOp::Rem => (lhs % rhs).ok_or(ErrorKind::RemainderWith0(span)),
                    BinaryOp::BitwiseOR => lhs | rhs,
                    BinaryOp::BitwiseAND => lhs & rhs,
                    BinaryOp::BitwiseXOR => lhs ^ rhs,
                    BinaryOp::Gt => to_int(lhs > rhs, -1),
                    BinaryOp::Lt => to_int(lhs < rhs, -1),
                    BinaryOp::Ge => to_int(lhs >= rhs, -1),
                    BinaryOp::Le => to_int(lhs <= rhs, -1),
                    BinaryOp::Ne => to_int(lhs.partial_cmp(&rhs) != Some(Ordering::Equal), -1),
                    BinaryOp::Eq => to_int(lhs.partial_cmp(&rhs) == Some(Ordering::Equal), -1),
                    BinaryOp::LogicalAnd => to_int(lhs.to_bool() && rhs.to_bool(), 1),
                    BinaryOp::LogicalOr => to_int(lhs.to_bool() || rhs.to_bool(), 1),
                    BinaryOp::Shl => (lhs << rhs).map_err(|e| replace_span(e, span)),
                    BinaryOp::Shr => (lhs >> rhs).map_err(|e| replace_span(e, span)),
                }
                .add_span(op.1)
            }
        }
    }

    /// Identifier evaluator utility function that doesn't allow any identifier in the expression
    ///
    /// # Errors
    ///
    /// Always errors with a [`ErrorKind::UnallowedLabel`]
    pub const fn unallowed_ident<T>(_: &str) -> Result<T, ErrorKind> {
        Err(ErrorKind::UnallowedLabel)
    }

    /// Utility function to evaluate an expression without allowing identifiers
    ///
    /// # Errors
    ///
    /// Errors in the same cases as [`Expr::eval`], but any identifier usage results in a
    /// [`ErrorKind::UnallowedLabel`]
    pub fn eval_no_ident(&self, modifiers: &ModifierDefinitions) -> Result<Number, ErrorData> {
        self.eval(Self::unallowed_ident, modifiers)
    }
}

/// Creates a parser for expressions
#[must_use]
pub fn parser<'tokens, I>() -> Parser!('tokens, I, Spanned<Expr>)
where
    I: ValueInput<'tokens, Token = Token, Span = Span>,
{
    // Literal values
    let literal = select! {
        Token::Integer(x) => Expr::Integer(x),
        Token::Float(x) = e => Expr::Float((x.into(), e.span())),
        Token::Character(c) => Expr::Character(c),
        Token::Identifier(ident) = e => Expr::Identifier((ident, e.span())),
        Token::Directive(ident) = e => Expr::Identifier((ident, e.span())),
    }
    .labelled("literal");

    // Operator parser
    macro_rules! op {
        (:$name:literal: $($i:ident => $o:expr),+ $(,)?) => {
            select! { $(Token::Operator(Operator::$i) => $o,)+ }
                .map_with(|x, e| (x, e.span()))
                .labelled(concat!($name, " operator"))
        };
        ($($i:ident => $o:expr),+ $(,)?) => { op!(:"binary": $($i => $o,)+) };
    }

    // Folding function for binary operations. We need to define it with a macro because the
    // closure doesn't use the generic lifetime bounds required when stored in a variable
    macro_rules! fold {
        () => {
            |lhs: Spanned<Expr>, op: Spanned<BinaryOp>, rhs: Spanned<Expr>, _| {
                let span = lhs.1.start..rhs.1.end;
                (
                    Expr::BinaryOp {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    Span::new(op.1.context, span),
                )
            }
        };
    }

    recursive(|expr| {
        // paren_expr: `paren_expr -> ( expression )`
        let paren_expr = expr.delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')));
        // modifier: `modifier -> % ident paren_expr`
        let modifier = just(Token::Operator(Operator::Percent))
            .ignore_then(select! {Token::Identifier(name) => name }.labelled("identifier"))
            .map_with(|name, e| (UnaryOp::Modifier(name), e.span()))
            .then(paren_expr.clone())
            .map(|(op, expr)| Expr::UnaryOp {
                op,
                operand: Box::new(expr),
            });
        // Remove span to replace it with one including the parenthesis
        let paren_expr = paren_expr.map(|(x, _)| x);

        // atom: `atom -> literal | modifier | paren_expr`
        let atom = choice((literal, modifier, paren_expr)).map_with(|atom, e| (atom, e.span()));
        let atom = atom.labelled("expression").as_context();

        let high_precedence = op!(
            Star => BinaryOp::Mul,
            Slash => BinaryOp::Div,
            Percent => BinaryOp::Rem,
            Shl => BinaryOp::Shl,
            Shr => BinaryOp::Shr,
        );
        let medium_precedence = op!(
            Or => BinaryOp::BitwiseOR,
            And => BinaryOp::BitwiseAND,
            Caret => BinaryOp::BitwiseXOR,
        );
        let low_precedence = op!(
            Plus => BinaryOp::Add,
            Minus => BinaryOp::Sub,
            Gt => BinaryOp::Gt,
            Lt => BinaryOp::Lt,
            Ge => BinaryOp::Ge,
            Le => BinaryOp::Le,
            Ne => BinaryOp::Ne,
            Eq => BinaryOp::Eq,
        );
        let expr = atom.pratt((
            prefix(
                6,
                op!(:"unary": Plus => UnaryOp::Plus, Minus => UnaryOp::Minus, Tilde => UnaryOp::Complement),
                |op: Spanned<UnaryOp>, rhs: Spanned<Expr>, _| {
                    let span = op.1.start..rhs.1.end;
                    let ctx = op.1.context;
                    (
                        Expr::UnaryOp {
                            op,
                            operand: Box::new(rhs),
                        },
                        Span::new(ctx, span),
                    )
                },
            ),
            infix(left(5), high_precedence, fold!()),
            infix(left(4), medium_precedence, fold!()),
            infix(left(3), low_precedence, fold!()),
            infix(left(2), op!(LogicalAnd => BinaryOp::LogicalAnd), fold!()),
            infix(left(1), op!(LogicalOr => BinaryOp::LogicalOr), fold!()),
        ));
        expr.labelled("expression").as_context()
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::architecture::Modifier;
    use crate::compiler::error::OperationKind;
    use crate::span::test::*;

    fn parse(code: &str) -> Result<Spanned<Expr>, ()> {
        super::super::parse_with!(super::parser(), "#", code).map_err(|_| ())
    }

    type ExprResult = Result<Number, ErrorData>;

    fn test(test_cases: impl IntoIterator<Item = (&'static str, Spanned<Expr>, ExprResult)>) {
        let ident_eval = |ident: &str| {
            if ident.len() == 1 {
                Ok(BigInt::from(ident.as_bytes()[0] - b'a' + 5))
            } else {
                Err(ErrorKind::UnknownLabel(ident.to_owned()))
            }
        };
        #[allow(clippy::unwrap_used)]
        let modifier = |range: (u64, Option<u64>), lower_signed, output_signed| Modifier {
            range: range.try_into().unwrap(),
            lower_signed,
            output_signed,
        };
        let modifiers = ModifierDefinitions::from([
            ("hi", modifier((12, Some(32)), true, false)),
            ("low", modifier((0, Some(12)), false, true)),
        ]);
        for (src, expr, expected) in test_cases {
            assert_eq!(parse(src), Ok(expr.clone()), "`{src:?}`");
            let res = expr.0.eval(ident_eval, &modifiers);
            assert_eq!(res, expected, "`{src:?}`\n{:?}", expr.0);
        }
    }

    #[must_use]
    fn float_op<S: IntoSpan>(op: OperationKind, float_span: S, op_span: S) -> ErrorData {
        ErrorKind::UnallowedFloatOperation(op, float_span.span()).add_span(op_span.span())
    }

    #[test]
    fn unallowed_ident() {
        for i in &["a", "b", "test", "identifier"] {
            assert_eq!(
                Expr::unallowed_ident::<i32>(i),
                Err(ErrorKind::UnallowedLabel)
            );
        }
    }

    #[test]
    fn literal() {
        let int = BigUint::from(2u8).pow(128) - 1u8;
        let span = |e, s: Range| (e, s.span());
        test([
            ("16", span(Expr::Integer(16u8.into()), 0..2), Ok(16.into())),
            (
                "\t 16",
                span(Expr::Integer(16u8.into()), 2..4),
                Ok(16.into()),
            ),
            (
                "'a'",
                span(Expr::Character('a'), 0..3),
                Ok(('a' as u32).into()),
            ),
            (
                "a",
                span(Expr::Identifier(("a".into(), (0..1).span())), 0..1),
                Ok(5.into()),
            ),
            (
                "test",
                span(Expr::Identifier(("test".into(), (0..4).span())), 0..4),
                Err(ErrorKind::UnknownLabel("test".into()).add_span((0..4).span())),
            ),
            (
                ".test",
                span(Expr::Identifier((".test".into(), (0..5).span())), 0..5),
                Err(ErrorKind::UnknownLabel(".test".into()).add_span((0..5).span())),
            ),
            (
                "1.0",
                span(Expr::Float((1.0, (0..3).span())), 0..3),
                Ok((1.0, 0..3).into()),
            ),
            (
                "340282366920938463463374607431768211455",
                span(Expr::Integer(int.clone()), 0..39),
                Ok(int.into()),
            ),
        ]);
    }

    #[must_use]
    fn int(x: u32, s: impl IntoSpan) -> Spanned<Expr> {
        (Expr::Integer(x.into()), s.span())
    }

    #[must_use]
    fn float(x: f64, s: impl IntoSpan) -> Spanned<Expr> {
        let s = s.span();
        (Expr::Float((x, s)), s)
    }

    #[must_use]
    fn un_op(op: (UnaryOp, impl IntoSpan), operand: (Expr, impl IntoSpan)) -> Spanned<Expr> {
        let op_span = op.1.span();
        let operand_span = operand.1.span();
        (
            Expr::UnaryOp {
                op: (op.0, op_span),
                operand: Box::new((operand.0, operand_span)),
            },
            (op_span.start..operand_span.end).span(),
        )
    }

    #[must_use]
    fn bin_op<S1, S2, S3>(op: (BinaryOp, S1), lhs: (Expr, S2), rhs: (Expr, S3)) -> Spanned<Expr>
    where
        S1: IntoSpan,
        S2: IntoSpan,
        S3: IntoSpan,
    {
        let l_span = lhs.1.span();
        let r_span = rhs.1.span();
        (
            Expr::BinaryOp {
                op: (op.0, op.1.span()),
                lhs: Box::new((lhs.0, l_span)),
                rhs: Box::new((rhs.0, r_span)),
            },
            (l_span.start..r_span.end).span(),
        )
    }

    #[test]
    fn unary() {
        let modifier = |n: &str, s| (UnaryOp::Modifier(n.into()), s);
        let mod_span = |e: Spanned<Expr>, s: Range| (e.0, s.span());
        test([
            (
                "+2",
                un_op((UnaryOp::Plus, 0..1), int(2, 1..2)),
                Ok(2.into()),
            ),
            (
                "+2.2",
                un_op((UnaryOp::Plus, 0..1), float(2.2, 1..4)),
                Ok((2.2, 1..4).into()),
            ),
            (
                "\t + 2",
                un_op((UnaryOp::Plus, 2..3), int(2, 4..5)),
                Ok(2.into()),
            ),
            (
                " \t+\t2.2",
                un_op((UnaryOp::Plus, 2..3), float(2.2, 4..7)),
                Ok((2.2, 4..7).into()),
            ),
            (
                "-2",
                un_op((UnaryOp::Minus, 0..1), int(2, 1..2)),
                Ok((-2).into()),
            ),
            (
                "~2",
                un_op((UnaryOp::Complement, 0..1), int(2, 1..2)),
                Ok((!2).into()),
            ),
            (
                "~2.75",
                un_op((UnaryOp::Complement, 0..1), float(2.75, 1..5)),
                Err(float_op(OperationKind::Complement, 1..5, 0..1)),
            ),
            (
                "%hi(0xABCDE701)",
                mod_span(un_op(modifier("hi", 0..3), int(0xABCD_E701, 4..14)), 0..15),
                Ok((0xABCDE).into()),
            ),
            (
                "%low(0xABCDE701)",
                mod_span(un_op(modifier("low", 0..4), int(0xABCD_E701, 5..15)), 0..16),
                Ok((0x701).into()),
            ),
            (
                "%mod(0xABCDE701)",
                mod_span(un_op(modifier("mod", 0..4), int(0xABCD_E701, 5..15)), 0..16),
                Err(ErrorKind::UnknownModifier("mod".into()).add_span((0..4).span())),
            ),
        ]);
    }

    #[test]
    fn binary_add() {
        test([
            (
                "5 + 7",
                bin_op((BinaryOp::Add, 2..3), int(5, 0..1), int(7, 4..5)),
                Ok(12.into()),
            ),
            (
                "\t5 \t\t+ \t7",
                bin_op((BinaryOp::Add, 5..6), int(5, 1..2), int(7, 8..9)),
                Ok(12.into()),
            ),
            (
                "2147483647 + 1",
                bin_op(
                    (BinaryOp::Add, 11..12),
                    int(i32::MAX as u32, 0..10),
                    int(1, 13..14),
                ),
                Ok(2_147_483_648_u32.into()),
            ),
            (
                "2.5 + 7",
                bin_op((BinaryOp::Add, 4..5), float(2.5, 0..3), int(7, 6..7)),
                Ok((9.5, 0..3).into()),
            ),
            (
                "2.5 + 7.25",
                bin_op((BinaryOp::Add, 4..5), float(2.5, 0..3), float(7.25, 6..10)),
                Ok((9.75, 0..3).into()),
            ),
        ]);
    }

    #[test]
    fn binary_sub() {
        test([
            (
                "4294967295 - 4294967295",
                bin_op(
                    (BinaryOp::Sub, 11..12),
                    int(u32::MAX, 0..10),
                    int(u32::MAX, 13..23),
                ),
                Ok(0.into()),
            ),
            (
                "d - a",
                bin_op(
                    (BinaryOp::Sub, 2..3),
                    (Expr::Identifier(("d".into(), (0..1).span())), 0..1),
                    (Expr::Identifier(("a".into(), (4..5).span())), 4..5),
                ),
                Ok(3.into()),
            ),
        ]);
    }

    #[test]
    fn binary_mul() {
        test([
            (
                "5 * 7",
                bin_op((BinaryOp::Mul, 2..3), int(5, 0..1), int(7, 4..5)),
                Ok(35.into()),
            ),
            (
                "\t5 \t\t* \t7",
                bin_op((BinaryOp::Mul, 5..6), int(5, 1..2), int(7, 8..9)),
                Ok(35.into()),
            ),
            (
                "2147483647 * 2147483648",
                bin_op(
                    (BinaryOp::Mul, 11..12),
                    int(i32::MAX as u32, 0..10),
                    int(1 << 31, 13..23),
                ),
                Ok((BigInt::from(2_147_483_647) * BigInt::from(2_147_483_648_u32)).into()),
            ),
        ]);
    }

    #[test]
    fn binary_div() {
        test([
            (
                "8 / 2",
                bin_op((BinaryOp::Div, 2..3), int(8, 0..1), int(2, 4..5)),
                Ok(4.into()),
            ),
            (
                "10 / 0",
                bin_op((BinaryOp::Div, 3..4), int(10, 0..2), int(0, 5..6)),
                Err(ErrorKind::DivisionBy0((5..6).span()).add_span((3..4).span())),
            ),
            (
                "10 / 0.0",
                bin_op((BinaryOp::Div, 3..4), int(10, 0..2), float(0.0, 5..8)),
                Ok((f64::INFINITY, 5..8).into()),
            ),
        ]);
    }

    #[test]
    fn binary_rem() {
        test([
            (
                "7 % 5",
                bin_op((BinaryOp::Rem, 2..3), int(7, 0..1), int(5, 4..5)),
                Ok(2.into()),
            ),
            (
                "7 % 0",
                bin_op((BinaryOp::Rem, 2..3), int(7, 0..1), int(0, 4..5)),
                Err(ErrorKind::RemainderWith0((4..5).span()).add_span((2..3).span())),
            ),
            (
                "7.2 % 5",
                bin_op((BinaryOp::Rem, 4..5), float(7.2, 0..3), int(5, 6..7)),
                Ok((2.2, 0..3).into()),
            ),
        ]);
    }

    #[test]
    fn binary_bitwise() {
        test([
            (
                "0b0101 | 0b0011",
                bin_op(
                    (BinaryOp::BitwiseOR, 7..8),
                    int(0b0101, 0..6),
                    int(0b0011, 9..15),
                ),
                Ok(0b0111.into()),
            ),
            (
                "0b0101 & 0b0011",
                bin_op(
                    (BinaryOp::BitwiseAND, 7..8),
                    int(0b0101, 0..6),
                    int(0b0011, 9..15),
                ),
                Ok(0b0001.into()),
            ),
            (
                "0b0101 ^ 0b0011",
                bin_op(
                    (BinaryOp::BitwiseXOR, 7..8),
                    int(0b0101, 0..6),
                    int(0b0011, 9..15),
                ),
                Ok(0b0110.into()),
            ),
            (
                "\t0b0101 \t\t^ \t0b0011",
                bin_op(
                    (BinaryOp::BitwiseXOR, 10..11),
                    int(0b0101, 1..7),
                    int(0b0011, 13..19),
                ),
                Ok(0b0110.into()),
            ),
            (
                "\t0b0101 \t\t^ \t1.1",
                bin_op(
                    (BinaryOp::BitwiseXOR, 10..11),
                    int(0b0101, 1..7),
                    float(1.1, 13..16),
                ),
                Err(float_op(OperationKind::BitwiseXOR, 13..16, 10..11)),
            ),
        ]);
    }

    #[test]
    fn binary_comparison() {
        test([
            (
                "5 < 3",
                bin_op((BinaryOp::Lt, 2..3), int(5, 0..1), int(3, 4..5)),
                Ok(0.into()),
            ),
            (
                "5 > 3",
                bin_op((BinaryOp::Gt, 2..3), int(5, 0..1), int(3, 4..5)),
                Ok((-1).into()),
            ),
            (
                "5 <= 3",
                bin_op((BinaryOp::Le, 2..4), int(5, 0..1), int(3, 5..6)),
                Ok(0.into()),
            ),
            (
                "5 >= 3",
                bin_op((BinaryOp::Ge, 2..4), int(5, 0..1), int(3, 5..6)),
                Ok((-1).into()),
            ),
            (
                "5 == 3",
                bin_op((BinaryOp::Eq, 2..4), int(5, 0..1), int(3, 5..6)),
                Ok(0.into()),
            ),
            (
                "5 != 3",
                bin_op((BinaryOp::Ne, 2..4), int(5, 0..1), int(3, 5..6)),
                Ok((-1).into()),
            ),
            (
                "3 >= 3",
                bin_op((BinaryOp::Ge, 2..4), int(3, 0..1), int(3, 5..6)),
                Ok((-1).into()),
            ),
            (
                "0 == 0.0",
                bin_op((BinaryOp::Eq, 2..4), int(0, 0..1), float(0.0, 5..8)),
                Ok((-1).into()),
            ),
            (
                "3 >= 3.1",
                bin_op((BinaryOp::Ge, 2..4), int(3, 0..1), float(3.1, 5..8)),
                Ok(0.into()),
            ),
        ]);
    }

    #[test]
    fn binary_boolean() {
        test([
            (
                "5 || 3",
                bin_op((BinaryOp::LogicalOr, 2..4), int(5, 0..1), int(3, 5..6)),
                Ok(1.into()),
            ),
            (
                "0 || 3",
                bin_op((BinaryOp::LogicalOr, 2..4), int(0, 0..1), int(3, 5..6)),
                Ok(1.into()),
            ),
            (
                "5 || 0",
                bin_op((BinaryOp::LogicalOr, 2..4), int(5, 0..1), int(0, 5..6)),
                Ok(1.into()),
            ),
            (
                "0 || 0",
                bin_op((BinaryOp::LogicalOr, 2..4), int(0, 0..1), int(0, 5..6)),
                Ok(0.into()),
            ),
            (
                "0.2 || 0",
                bin_op((BinaryOp::LogicalOr, 4..6), float(0.2, 0..3), int(0, 7..8)),
                Ok(1.into()),
            ),
            (
                "5 && 3",
                bin_op((BinaryOp::LogicalAnd, 2..4), int(5, 0..1), int(3, 5..6)),
                Ok(1.into()),
            ),
            (
                "0 && 3",
                bin_op((BinaryOp::LogicalAnd, 2..4), int(0, 0..1), int(3, 5..6)),
                Ok(0.into()),
            ),
            (
                "5 && 0",
                bin_op((BinaryOp::LogicalAnd, 2..4), int(5, 0..1), int(0, 5..6)),
                Ok(0.into()),
            ),
            (
                "0 && 0",
                bin_op((BinaryOp::LogicalAnd, 2..4), int(0, 0..1), int(0, 5..6)),
                Ok(0.into()),
            ),
            (
                "0.2 && 1.4",
                bin_op(
                    (BinaryOp::LogicalAnd, 4..6),
                    float(0.2, 0..3),
                    float(1.4, 7..10),
                ),
                Ok(1.into()),
            ),
        ]);
    }

    #[test]
    fn binary_shift() {
        test([
            (
                "5 << 3",
                bin_op((BinaryOp::Shl, 2..4), int(5, 0..1), int(3, 5..6)),
                Ok(40.into()),
            ),
            (
                "5 >> 1",
                bin_op((BinaryOp::Shr, 2..4), int(5, 0..1), int(1, 5..6)),
                Ok(2.into()),
            ),
            (
                "5.2 << 3",
                bin_op((BinaryOp::Shl, 4..6), float(5.2, 0..3), int(3, 7..8)),
                Err(float_op(OperationKind::Shl, 0..3, 4..6)),
            ),
            (
                "5.2 >> 3",
                bin_op((BinaryOp::Shr, 4..6), float(5.2, 0..3), int(3, 7..8)),
                Err(float_op(OperationKind::Shr, 0..3, 4..6)),
            ),
            (
                "5 << -1",
                bin_op(
                    (BinaryOp::Shl, 2..4),
                    int(5, 0..1),
                    un_op((UnaryOp::Minus, 5..6), int(1, 6..7)),
                ),
                Err(ErrorKind::ShiftOutOfRange((5..7).span(), (-1).into()).add_span((2..4).span())),
            ),
            (
                "5 >> -1",
                bin_op(
                    (BinaryOp::Shr, 2..4),
                    int(5, 0..1),
                    un_op((UnaryOp::Minus, 5..6), int(1, 6..7)),
                ),
                Err(ErrorKind::ShiftOutOfRange((5..7).span(), (-1).into()).add_span((2..4).span())),
            ),
        ]);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn precedence() {
        test([
            (
                "1 + 2 - 3",
                bin_op(
                    (BinaryOp::Sub, 6..7),
                    bin_op((BinaryOp::Add, 2..3), int(1, 0..1), int(2, 4..5)),
                    int(3, 8..9),
                ),
                Ok(0.into()),
            ),
            (
                "1 + \t(\t2 - 3\t)",
                bin_op(
                    (BinaryOp::Add, 2..3),
                    int(1, 0..1),
                    (
                        bin_op((BinaryOp::Sub, 9..10), int(2, 7..8), int(3, 11..12)).0,
                        5..14,
                    ),
                ),
                Ok(0.into()),
            ),
            (
                "1 | 6 & 3 ^ 9",
                bin_op(
                    (BinaryOp::BitwiseXOR, 10..11),
                    bin_op(
                        (BinaryOp::BitwiseAND, 6..7),
                        bin_op((BinaryOp::BitwiseOR, 2..3), int(1, 0..1), int(6, 4..5)),
                        int(3, 8..9),
                    ),
                    int(9, 12..13),
                ),
                Ok(10.into()),
            ),
            (
                "1 * 6 / 3 % 2",
                bin_op(
                    (BinaryOp::Rem, 10..11),
                    bin_op(
                        (BinaryOp::Div, 6..7),
                        bin_op((BinaryOp::Mul, 2..3), int(1, 0..1), int(6, 4..5)),
                        int(3, 8..9),
                    ),
                    int(2, 12..13),
                ),
                Ok(0.into()),
            ),
            (
                "\t- \t\t+ \t1",
                un_op(
                    (UnaryOp::Minus, 1..2),
                    un_op((UnaryOp::Plus, 5..6), int(1, 8..9)),
                ),
                Ok((-1).into()),
            ),
            (
                "~-+1",
                un_op(
                    (UnaryOp::Complement, 0..1),
                    un_op(
                        (UnaryOp::Minus, 1..2),
                        un_op((UnaryOp::Plus, 2..3), int(1, 3..4)),
                    ),
                ),
                Ok(0.into()),
            ),
            (
                "1 + 6 | +3 * +9",
                bin_op(
                    (BinaryOp::Add, 2..3),
                    int(1, 0..1),
                    bin_op(
                        (BinaryOp::BitwiseOR, 6..7),
                        int(6, 4..5),
                        bin_op(
                            (BinaryOp::Mul, 11..12),
                            un_op((UnaryOp::Plus, 8..9), int(3, 9..10)),
                            un_op((UnaryOp::Plus, 13..14), int(9, 14..15)),
                        ),
                    ),
                ),
                Ok(32.into()),
            ),
            (
                "1 + %low(0x1234) * 3",
                bin_op(
                    (BinaryOp::Add, 2..3),
                    int(1, 0..1),
                    bin_op(
                        (BinaryOp::Mul, 17..18),
                        (
                            un_op((UnaryOp::Modifier("low".into()), 4..8), int(0x1234, 9..15)).0,
                            4..16,
                        ),
                        int(3, 19..20),
                    ),
                ),
                Ok((1 + 0x234 * 3).into()),
            ),
            (
                "1 + 2 > 3 < 4 != 0 + 1",
                bin_op(
                    (BinaryOp::Add, 19..20),
                    bin_op(
                        (BinaryOp::Ne, 14..16),
                        bin_op(
                            (BinaryOp::Lt, 10..11),
                            bin_op(
                                (BinaryOp::Gt, 6..7),
                                bin_op((BinaryOp::Add, 2..3), int(1, 0..1), int(2, 4..5)),
                                int(3, 8..9),
                            ),
                            int(4, 12..13),
                        ),
                        int(0, 17..18),
                    ),
                    int(1, 21..22),
                ),
                Ok(0.into()),
            ),
            (
                "1 + 2 >= 3 <= 4 == 0 + 1",
                bin_op(
                    (BinaryOp::Add, 21..22),
                    bin_op(
                        (BinaryOp::Eq, 16..18),
                        bin_op(
                            (BinaryOp::Le, 11..13),
                            bin_op(
                                (BinaryOp::Ge, 6..8),
                                bin_op((BinaryOp::Add, 2..3), int(1, 0..1), int(2, 4..5)),
                                int(3, 9..10),
                            ),
                            int(4, 14..15),
                        ),
                        int(0, 19..20),
                    ),
                    int(1, 23..24),
                ),
                Ok(1.into()),
            ),
            (
                "0 || 1 && 2 + 3",
                bin_op(
                    (BinaryOp::LogicalOr, 2..4),
                    int(0, 0..1),
                    bin_op(
                        (BinaryOp::LogicalAnd, 7..9),
                        int(1, 5..6),
                        bin_op((BinaryOp::Add, 12..13), int(2, 10..11), int(3, 14..15)),
                    ),
                ),
                Ok(1.into()),
            ),
            (
                "2 * 1 << 2 >> 1 * 2",
                bin_op(
                    (BinaryOp::Mul, 16..17),
                    bin_op(
                        (BinaryOp::Shr, 11..13),
                        bin_op(
                            (BinaryOp::Shl, 6..8),
                            bin_op((BinaryOp::Mul, 2..3), int(2, 0..1), int(1, 4..5)),
                            int(2, 9..10),
                        ),
                        int(1, 14..15),
                    ),
                    int(2, 18..19),
                ),
                Ok(8.into()),
            ),
        ]);
    }
}
