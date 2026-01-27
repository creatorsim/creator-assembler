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

//! Module containing the definition of [`Number`]

use std::ops;

use num_bigint::{BigInt, BigUint};
use num_traits::cast::ToPrimitive;

use crate::architecture::Modifier;
use crate::compiler::{error::OperationKind, ErrorKind};
use crate::span::{Span, Spanned};

/// Generic number type, either an integer or a floating-point number
#[derive(Debug, Clone, PartialEq)]
pub enum Number {
    Int(BigInt),
    Float {
        // Value of the number
        value: f64,
        // Span that caused the number to be casted to a float
        origin: Span,
    },
}

impl Number {
    /// Combines the origin spans of 2 numbers, assuming either one or both of them are floats
    ///
    /// # Panics
    ///
    /// Panics if both numbers are integers
    fn combine_origin(&self, rhs: &Self) -> Span {
        match (self, rhs) {
            (Self::Float { origin, .. }, _) | (_, Self::Float { origin, .. }) => *origin,
            _ => unreachable!("We shouldn't try to combine the origin spans of 2 integers"),
        }
    }
}

/// Generates implementations of <code>[From]\<int></code> for [`Number`]
macro_rules! impl_from_int {
    ($($ty:ty),+) => {
        $(
            impl From<$ty> for Number {
                fn from(value: $ty) -> Self {
                    Self::Int(value.into())
                }
            }
        )+
    };
}
impl_from_int!(BigUint, BigInt, u32, i32);

impl From<Spanned<f64>> for Number {
    fn from(value: Spanned<f64>) -> Self {
        Self::Float {
            value: value.0,
            origin: value.1,
        }
    }
}

impl TryFrom<Number> for BigInt {
    type Error = ErrorKind;
    fn try_from(value: Number) -> Result<Self, Self::Error> {
        match value {
            Number::Int(x) => Ok(x),
            Number::Float { origin, .. } => Err(ErrorKind::UnallowedFloat(origin)),
        }
    }
}

impl TryFrom<Number> for BigUint {
    type Error = ErrorKind;
    fn try_from(value: Number) -> Result<Self, Self::Error> {
        match value {
            Number::Int(x) => {
                Self::try_from(x).map_err(|e| ErrorKind::UnallowedNegativeValue(e.into_original()))
            }
            Number::Float { origin, .. } => Err(ErrorKind::UnallowedFloat(origin)),
        }
    }
}

impl ops::Neg for Number {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Int(value) => Self::Int(-value),
            Self::Float { value, origin } => Self::Float {
                origin,
                value: -value,
            },
        }
    }
}

impl ops::Not for Number {
    type Output = Result<Self, ErrorKind>;

    fn not(self) -> Self::Output {
        match self {
            Self::Int(value) => Ok(Self::Int(!value)),
            Self::Float { origin, .. } => Err(ErrorKind::UnallowedFloatOperation(
                OperationKind::Complement,
                origin,
            )),
        }
    }
}

/// Generates an implementation of a binary operation for [`Number`]
macro_rules! impl_bin_op {
    // Generic interface
    ($trait:path, $name:ident, ($lhs:ident, $rhs:ident), $int:expr, |$orig:ident| $float:expr, $out:ty$(: $wrap:ident)?) => {
        impl $trait for Number {
            type Output = $out;

            fn $name(self, rhs: Self) -> Self::Output {
                $($wrap)?(match (self, rhs) {
                    (Self::Int($lhs), Self::Int($rhs)) => Self::Int($int),
                    (lhs, rhs) => {
                        let $orig = lhs.combine_origin(&rhs);
                        let _value = match (lhs.to_f64(), rhs.to_f64()) {
                            ($lhs, $rhs) => $float
                        };
                        #[allow(unreachable_code)]
                        Self::Float { origin: $orig, value: _value }
                    },
                })
            }
        }
    };
    // Convenience shorthands that forward the arguments to the generic interface
    // No possible errors, just forwards the numbers to the operator
    ($trait:path, $name:ident, $op:tt) => {
        impl_bin_op!($trait, $name, (lhs, rhs), lhs $op rhs, |origin| lhs $op rhs, Self);
    };
    // Integer implementation that can fail in a single way, result wrapped in an `Option`
    ($trait:path, $name:ident, ($lhs:ident, $rhs:ident), $int:expr, $float:expr) => {
        impl_bin_op!($trait, $name, ($lhs, $rhs), $int, |origin| $float, Option<Self>: Some);
    };
    // Implementation for bitwise operators that aren't allowed with floats, result wrapped in
    // `Result`
    ($trait:path, $name:ident, $int:tt, $float:expr) => {
        impl_bin_op!(
            $trait,
            $name,
            (_lhs, _rhs),
            _lhs $int _rhs,
            |origin| return Err(ErrorKind::UnallowedFloatOperation($float, origin)),
            Result<Self, ErrorKind>: Ok
        );
    };
    // Implementation for shifts
    ($trait:path, $name:ident, shift $int:tt, $float:expr) => {
        impl_bin_op!(
            $trait,
            $name,
            (_lhs, _rhs),
            {
                let rhs = u16::try_from(_rhs).map_err(|e| {
                    ErrorKind::ShiftOutOfRange(
                        // NOTE: we don't have information here about the position of the int
                        // TODO: implement operations in Spanned<Number> instead to solve this?
                        crate::span::DEFAULT_SPAN,
                        e.into_original().into(),
                    )
                })?;
                _lhs $int rhs
            },
            |origin| return Err(ErrorKind::UnallowedFloatOperation($float, origin)),
            Result<Self, ErrorKind>: Ok
        );
    };
}

impl_bin_op!(ops::Add, add, +);
impl_bin_op!(ops::Sub, sub, -);
impl_bin_op!(ops::Mul, mul, *);
impl_bin_op!(ops::Div, div, (lhs, rhs), lhs.checked_div(&rhs)?, lhs / rhs);
impl_bin_op!(
    ops::Rem,
    rem,
    (lhs, rhs),
    (rhs != BigInt::ZERO).then(|| lhs % rhs)?,
    lhs % rhs
);
impl_bin_op!(ops::BitOr, bitor, |, OperationKind::BitwiseOR);
impl_bin_op!(ops::BitAnd, bitand, &, OperationKind::BitwiseAND);
impl_bin_op!(ops::BitXor, bitxor, ^, OperationKind::BitwiseXOR);
impl_bin_op!(ops::Shl, shl, shift <<, OperationKind::Shl);
impl_bin_op!(ops::Shr, shr, shift >>, OperationKind::Shr);

impl Number {
    /// Converts this number to a [`f64`]
    pub fn to_f64(&self) -> f64 {
        match self {
            Self::Int(x) => x.to_f64().expect("Converting a bigint to f64 can't fail"),
            Self::Float { value, .. } => *value,
        }
    }

    /// Converts this number to a [`f32`]
    pub fn to_f32(&self) -> f32 {
        match self {
            Self::Int(x) => x.to_f32().expect("Converting a bigint to f32 can't fail"),
            #[allow(clippy::cast_possible_truncation)]
            Self::Float { value, .. } => *value as f32,
        }
    }

    /// Converts this number to a [`bool`]
    pub fn to_bool(&self) -> bool {
        match self {
            Self::Int(x) => *x != BigInt::ZERO,
            Self::Float { value, .. } => *value != 0.0,
        }
    }

    /// Applies a modifier to this number, taking a slice of bits from it, manipulating it as
    /// specified, and returning it as an integer
    pub fn modify(self, modifier: Modifier) -> Self {
        // Convert the number to an int, preserving the bit pattern for floats
        let mut x = match self {
            Self::Int(x) => x,
            Self::Float { value, .. } => value.to_bits().into(),
        };
        // Discard the bits below the range, if any
        if modifier.range.start > 0 {
            let lower = modifier.range.start - 1;
            let inc = modifier.lower_signed && x.bit(lower);
            x = (x >> modifier.range.start) + u8::from(inc);
        }
        // Discard the bits above the range, if any
        if let Some(size) = modifier.range.size {
            // NOTE: Bitwise operations are performed in two's complement. Positive numbers have
            // infinite leading 0s while negative numbers have infinite leading 1s
            x &= (BigInt::from(1) << size) - 1;
            if modifier.output_signed && x.bit(size - 1) {
                x |= BigInt::from(-1) << size;
            }
        }
        Self::Int(x)
    }
}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Int(lhs), Self::Int(rhs)) => Some(lhs.cmp(rhs)),
            (lhs, rhs) => lhs.to_f64().partial_cmp(&rhs.to_f64()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod test {
    use super::*;
    use crate::span::{test::*, DEFAULT_SPAN};

    impl From<Ranged<f64>> for Number {
        fn from(value: Ranged<f64>) -> Self {
            Self::Float {
                value: value.0,
                origin: value.1.span(),
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn to_float() {
        assert_eq!(Number::Int(123.into()).to_f64(), 123.0);
        assert_eq!(
            Number::Float {
                value: 101.5,
                origin: (0..0).span()
            }
            .to_f64(),
            101.5
        );
        assert_eq!(Number::Int(123.into()).to_f32(), 123.0);
        assert_eq!(
            Number::Float {
                value: 101.5,
                origin: (0..0).span()
            }
            .to_f32(),
            101.5
        );
    }

    #[test]
    fn to_int() {
        assert_eq!(BigInt::try_from(Number::Int(123.into())), Ok(123.into()));
        assert_eq!(
            BigInt::try_from(Number::Float {
                value: 101.5,
                origin: (1..3).span()
            }),
            Err(ErrorKind::UnallowedFloat((1..3).span()))
        );
    }

    #[test]
    fn to_uint() {
        assert_eq!(BigUint::try_from(Number::Int(123.into())), Ok(123u8.into()));
        assert_eq!(
            BigUint::try_from(Number::Int((-123).into())),
            Err(ErrorKind::UnallowedNegativeValue((-123).into()))
        );
        assert_eq!(
            BigUint::try_from(Number::Float {
                value: 101.5,
                origin: (1..3).span()
            }),
            Err(ErrorKind::UnallowedFloat((1..3).span()))
        );
    }

    #[test]
    fn to_bool() {
        assert!(Number::Int(123.into()).to_bool());
        assert!(!Number::Int(0.into()).to_bool());
        assert!(Number::Int((-1).into()).to_bool());
        assert!(Number::Float {
            value: 101.5,
            origin: (1..3).span()
        }
        .to_bool());
        assert!(Number::Float {
            value: -1.5,
            origin: (3..5).span()
        }
        .to_bool());
        assert!(!Number::Float {
            value: 0.0,
            origin: (1..3).span()
        }
        .to_bool());
    }

    #[test]
    fn from_f64() {
        assert_eq!(
            Number::from((12.5, 1..4)),
            Number::Float {
                value: 12.5,
                origin: (1..4).span()
            }
        );
    }

    fn int() -> BigInt {
        BigInt::from(2u8).pow(128) - 1u8
    }

    #[test]
    fn neg() {
        assert_eq!(-Number::from(123), Number::from(-123));
        assert_eq!(-Number::from((1.5, 1..2)), Number::from((-1.5, 1..2)));
        assert_eq!(
            -Number::from(BigInt::from(2).pow(200)),
            Number::from(-BigInt::from(2).pow(200))
        );
    }

    #[test]
    fn not() {
        assert_eq!(!Number::from(123), Ok(Number::from(!123)));
        assert_eq!(!Number::from(-1), Ok(Number::from(0)));
        assert_eq!(!Number::from(0), Ok(Number::from(-1)));
        assert_eq!(
            !Number::from((1.5, 1..2)),
            Err(ErrorKind::UnallowedFloatOperation(
                OperationKind::Complement,
                (1..2).span()
            ))
        );
    }

    #[test]
    fn add() {
        let op = |a: &Number, b: &Number| a.clone() + b.clone();
        let opint = |a: BigInt, b: BigInt| op(&Number::from(a), &Number::from(b));
        let i1 = Number::from(123);
        let i2 = Number::from(-2);
        let f1 = Number::from((1.2, 1..3));
        let f2 = Number::from((2.5, 5..6));
        assert_eq!(op(&i1, &i2), Number::from(121));
        assert_eq!(op(&i1, &f2), Number::from((125.5, 5..6)));
        assert_eq!(op(&f1, &i2), Number::from((-0.8, 1..3)));
        assert_eq!(op(&f1, &f2), Number::from((3.7, 1..3)));
        assert_eq!(opint(int(), int() + 1), Number::from(2 * int() + 1));
    }

    #[test]
    fn sub() {
        let op = |a: &Number, b: &Number| a.clone() - b.clone();
        let opint = |a: BigInt, b: BigInt| op(&Number::from(a), &Number::from(b));
        let i1 = Number::from(123);
        let i2 = Number::from(-2);
        let f1 = Number::from((1.2, 1..3));
        let f2 = Number::from((2.5, 5..6));
        assert_eq!(op(&i1, &i2), Number::from(125));
        assert_eq!(op(&i1, &f2), Number::from((120.5, 5..6)));
        assert_eq!(op(&f1, &i2), Number::from((3.2, 1..3)));
        assert_eq!(op(&f1, &f2), Number::from((-1.3, 1..3)));
        assert_eq!(opint(int() + 10, int() - 2), Number::from(12));
        assert_eq!(opint(int(), int() + 10), Number::from(-10));
    }

    #[test]
    fn mul() {
        let op = |a: &Number, b: &Number| a.clone() * b.clone();
        let opint = |a: BigInt, b: BigInt| op(&Number::from(a), &Number::from(b));
        let i1 = Number::from(12);
        let i2 = Number::from(-2);
        let f1 = Number::from((1.2, 1..3));
        let f2 = Number::from((2.5, 5..6));
        assert_eq!(op(&i1, &i2), Number::from(-24));
        assert_eq!(op(&i1, &f2), Number::from((30.0, 5..6)));
        assert_eq!(op(&f1, &i2), Number::from((-2.4, 1..3)));
        assert_eq!(op(&f1, &f2), Number::from((3.0, 1..3)));
        assert_eq!(opint(int(), int()), Number::from(int() * int()));
    }

    #[test]
    fn div() {
        const INF: f64 = f64::INFINITY;
        let op = |a: &Number, b: &Number| a.clone() / b.clone();
        let opint = |a: BigInt, b: BigInt| op(&Number::from(a), &Number::from(b));
        let i1 = Number::from(9);
        let i2 = Number::from(-5);
        let i3 = Number::from(0);
        let f1 = Number::from((1.2, 1..3));
        let f2 = Number::from((-2.5, 5..6));
        let f3 = Number::from((0.0, 5..6));
        let f4 = Number::from((-0.0, 5..6));
        assert_eq!(op(&i1, &i2), Some(Number::from(-1)));
        assert_eq!(op(&i1, &f2), Some(Number::from((9.0 / -2.5, 5..6))));
        assert_eq!(op(&f1, &i2), Some(Number::from((1.2 / -5.0, 1..3))));
        assert_eq!(op(&f1, &f2), Some(Number::from((1.2 / -2.5, 1..3))));
        assert_eq!(opint(int(), int()), Some(Number::from(1)));

        assert_eq!(op(&i1, &i3), None);
        assert_eq!(op(&i1, &f3), Some(Number::from((INF, 5..6))));
        assert_eq!(op(&f1, &i3), Some(Number::from((INF, 1..3))));
        assert_eq!(op(&f1, &f3), Some(Number::from((INF, 1..3))));
        assert_eq!(op(&f2, &f4), Some(Number::from((INF, 5..6))));
        assert_eq!(op(&i2, &f3), Some(Number::from((-INF, 5..6))));
        assert_eq!(op(&f2, &i3), Some(Number::from((-INF, 5..6))));
        assert_eq!(op(&f1, &f4), Some(Number::from((-INF, 1..3))));
    }

    #[test]
    fn rem() {
        let op = |a: &Number, b: &Number| a.clone() % b.clone();
        let opint = |a: BigInt, b: BigInt| op(&Number::from(a), &Number::from(b));
        let i1 = Number::from(9);
        let i2 = Number::from(5);
        let i3 = Number::from(-5);
        let i4 = Number::from(0);
        let f1 = Number::from((1.2, 1..3));
        let f2 = Number::from((2.5, 5..6));
        let f3 = Number::from((-2.5, 5..6));
        let f4 = Number::from((0.0, 5..6));
        assert_eq!(op(&i1, &i2), Some(Number::from(4)));
        assert_eq!(op(&i1, &f2), Some(Number::from((9.0 % 2.5, 5..6))));
        assert_eq!(op(&f1, &i2), Some(Number::from((1.2 % 5.0, 1..3))));
        assert_eq!(op(&f1, &f2), Some(Number::from((1.2 % 2.5, 1..3))));
        assert_eq!(opint(2 * int() + 10, int() - 2), Some(Number::from(14)));

        assert_eq!(op(&i1, &i3), Some(Number::from(4)));
        assert_eq!(op(&i3, &i1), Some(Number::from(-5)));
        assert_eq!(op(&f1, &f3), Some(Number::from((1.2, 1..3))));
        assert_eq!(op(&f3, &f1), Some(Number::from((-2.5 % 1.2, 5..6))));

        assert_eq!(i1.clone() % i4.clone(), None);
        let test = |a: &Number, b: &Number| {
            op(a, b).is_some_and(|x| match x {
                Number::Float { value, .. } => value.is_nan(),
                Number::Int(_) => false,
            })
        };
        assert!(test(&i1, &f4));
        assert!(test(&f1, &i4));
        assert!(test(&f1, &f4));
    }

    #[test]
    fn bitor() {
        let op = |a: &Number, b: &Number| a.clone() | b.clone();
        let opint = |a: BigInt, b: BigInt| op(&Number::from(a), &Number::from(b));
        let i1 = Number::from(12);
        let i2 = Number::from(-2);
        let f1 = Number::from((1.2, 1..3));
        let f2 = Number::from((2.5, 5..6));
        let err = |s: Range| ErrorKind::UnallowedFloatOperation(OperationKind::BitwiseOR, s.span());
        assert_eq!(op(&i1, &i2), Ok(Number::from(12 | -2)));
        assert_eq!(op(&i1, &f2), Err(err(5..6)));
        assert_eq!(op(&f1, &i2), Err(err(1..3)));
        assert_eq!(op(&f1, &f2), Err(err(1..3)));
        assert_eq!(
            opint(
                BigInt::from(0xAAAA_AAAA_AAAA_AAAA_AAAB_u128),
                BigInt::from(0x5555_5555_5555_5555_5555_u128)
            ),
            Ok(Number::from(BigInt::from(0xFFFF_FFFF_FFFF_FFFF_FFFF_u128)))
        );
        assert_eq!(
            op(&Number::from(-1), &Number::from(123)),
            Ok(Number::from(-1))
        );
    }

    #[test]
    fn bitand() {
        let op = |a: &Number, b: &Number| a.clone() & b.clone();
        let opint = |a: BigInt, b: BigInt| op(&Number::from(a), &Number::from(b));
        let i1 = Number::from(12);
        let i2 = Number::from(-2);
        let f1 = Number::from((1.2, 1..3));
        let f2 = Number::from((2.5, 5..6));
        let err =
            |s: Range| ErrorKind::UnallowedFloatOperation(OperationKind::BitwiseAND, s.span());
        assert_eq!(op(&i1, &i2), Ok(Number::from(12 & -2)));
        assert_eq!(op(&i1, &f2), Err(err(5..6)));
        assert_eq!(op(&f1, &i2), Err(err(1..3)));
        assert_eq!(op(&f1, &f2), Err(err(1..3)));
        assert_eq!(
            opint(
                BigInt::from(0xAAAA_AAAA_AAAA_AAAA_AAAB_u128),
                BigInt::from(0x5555_5555_5555_5555_5555_u128)
            ),
            Ok(Number::from(1))
        );
        assert_eq!(
            opint(
                BigInt::from(0xAAAA_AAAA_AAAA_AAAA_AAAB_u128),
                BigInt::from(-1)
            ),
            Ok(Number::from(BigInt::from(0xAAAA_AAAA_AAAA_AAAA_AAAB_u128)))
        );
        assert_eq!(
            op(&Number::from(-1), &Number::from(-1)),
            Ok(Number::from(-1))
        );
    }

    #[test]
    fn bitxor() {
        let op = |a: &Number, b: &Number| a.clone() ^ b.clone();
        let opint = |a: BigInt, b: BigInt| op(&Number::from(a), &Number::from(b));
        let i1 = Number::from(12);
        let i2 = Number::from(-2);
        let f1 = Number::from((1.2, 1..3));
        let f2 = Number::from((2.5, 5..6));
        let err =
            |s: Range| ErrorKind::UnallowedFloatOperation(OperationKind::BitwiseXOR, s.span());
        assert_eq!(op(&i1, &i2), Ok(Number::from(12 ^ -2)));
        assert_eq!(op(&i1, &f2), Err(err(5..6)));
        assert_eq!(op(&f1, &i2), Err(err(1..3)));
        assert_eq!(op(&f1, &f2), Err(err(1..3)));
        assert_eq!(
            opint(
                BigInt::from(0xAAAA_AAAA_AAAA_AAAA_AAAB_u128),
                BigInt::from(0x5555_5555_5555_5555_5555_u128)
            ),
            Ok(Number::from(BigInt::from(0xFFFF_FFFF_FFFF_FFFF_FFFE_u128)))
        );
    }

    #[test]
    fn shl() {
        let op = |a: &Number, b: &Number| a.clone() << b.clone();
        let i1 = Number::from(0b1010);
        let i2 = Number::from(2);
        let i3 = Number::from(-3);
        let f1 = Number::from((1.2, 1..3));
        let f2 = Number::from((2.5, 5..6));
        let err = |s: Range| ErrorKind::UnallowedFloatOperation(OperationKind::Shl, s.span());
        assert_eq!(op(&i1, &i2), Ok(Number::from(0b10_1000)));
        assert_eq!(op(&i1, &f2), Err(err(5..6)));
        assert_eq!(op(&f1, &i2), Err(err(1..3)));
        assert_eq!(op(&f1, &f2), Err(err(1..3)));
        assert_eq!(
            op(&i1, &i3),
            Err(ErrorKind::ShiftOutOfRange(DEFAULT_SPAN, (-3).into()))
        );
        let m = u32::from(u16::MAX) + 1;
        assert_eq!(
            op(&i1, &Number::from(m)),
            Err(ErrorKind::ShiftOutOfRange(DEFAULT_SPAN, m.into()))
        );
    }

    #[test]
    fn shr() {
        let op = |a: &Number, b: &Number| a.clone() >> b.clone();
        let opint = |a: BigInt, b: BigInt| op(&Number::from(a), &Number::from(b));
        let i1 = Number::from(0b1010);
        let i2 = Number::from(2);
        let i3 = Number::from(-3);
        let f1 = Number::from((1.2, 1..3));
        let f2 = Number::from((2.5, 5..6));
        let err = |s: Range| ErrorKind::UnallowedFloatOperation(OperationKind::Shr, s.span());
        assert_eq!(op(&i1, &i2), Ok(Number::from(0b10)));
        assert_eq!(op(&i1, &f2), Err(err(5..6)));
        assert_eq!(op(&f1, &i2), Err(err(1..3)));
        assert_eq!(op(&f1, &f2), Err(err(1..3)));
        assert_eq!(
            op(&i1, &i3),
            Err(ErrorKind::ShiftOutOfRange(DEFAULT_SPAN, (-3).into()))
        );
        let m = u32::from(u16::MAX) + 1;
        assert_eq!(
            op(&i1, &Number::from(m)),
            Err(ErrorKind::ShiftOutOfRange(DEFAULT_SPAN, m.into()))
        );
        assert_eq!(
            opint(
                BigInt::from(0xAAAA_AAAA_AAAA_AAAA_AAAB_u128),
                BigInt::from(u16::MAX)
            ),
            Ok(Number::from(0))
        );
        assert_eq!(
            opint(
                BigInt::from(-0xAAAA_AAAA_AAAA_AAAA_AAAB_i128),
                BigInt::from(u16::MAX)
            ),
            Ok(Number::from(-1))
        );
    }

    #[test]
    fn modify_split() {
        let hi = Modifier {
            range: (12, Some(32)).try_into().unwrap(),
            lower_signed: true,
            output_signed: false,
        };
        let lo = Modifier {
            range: (0, Some(12)).try_into().unwrap(),
            lower_signed: false,
            output_signed: true,
        };
        let neg = |x: u16| x.cast_signed().into();

        let x = Number::from(0xdead_beef_u32);
        let x_hi: BigInt = x.clone().modify(hi).try_into().unwrap();
        let x_lo: BigInt = x.clone().modify(lo).try_into().unwrap();
        assert_eq!(x_hi, 0xdeadc.into());
        assert_eq!(x_lo, neg(0xfeef));
        let res = (x_hi << 12) + x_lo;
        assert_eq!(res, x.try_into().unwrap());

        let x = Number::from(0xFFFF_FFFF_u32);
        let x_hi: BigInt = x.clone().modify(hi).try_into().unwrap();
        let x_lo: BigInt = x.modify(lo).try_into().unwrap();
        assert_eq!(x_hi, 0.into());
        assert_eq!(x_lo, neg(0xffff));
        let res = (x_hi << 12) + x_lo;
        assert_eq!(res, (-1).into());
    }

    #[test]
    fn modify() {
        let modifier = |range: (u64, Option<u64>), lower_signed, output_signed| Modifier {
            range: range.try_into().unwrap(),
            lower_signed,
            output_signed,
        };
        let neg = |x: u8| x.cast_signed().into();
        let mod1 = modifier((4, None), true, false);
        let mod2 = modifier((4, None), false, false);
        let mod3 = modifier((4, Some(8)), true, false);
        let mod4 = modifier((4, Some(8)), false, false);
        let mod5 = modifier((4, Some(8)), true, true);
        let test_cases = [
            (0b0101_0110, mod1, 0b0101),
            (0b1001_0110, mod1, 0b1001),
            (0b1001_0001, mod1, 0b1001),
            (0b1001_1001, mod1, 0b1010),
            (0b1001_1001, mod2, 0b1001),
            (0b1001_0110, mod2, 0b1001),
            (neg(0xA7), mod2, neg(0xFA)),
            (neg(0x7F), mod2, neg(0x07)),
            (0b1_1001_0110, mod3, 0b1001),
            (0b1_1001_0001, mod3, 0b1001),
            (0b1_1001_1001, mod3, 0b1010),
            (0b1_1111_1001, mod3, 0b0000),
            (0b1_1001_1001, mod4, 0b1001),
            (0b1_1001_0110, mod4, 0b1001),
            (0b1_0101_1001, mod4, 0b0101),
            (neg(0xA7), mod5, neg(0xFA)),
            (neg(0x7F), mod5, neg(0xF8)),
            (neg(0x74), mod5, neg(0x07)),
        ];
        for (i, (x, modifier, expected)) in test_cases.into_iter().enumerate() {
            let x = Number::from(x);
            let res: BigInt = x.modify(modifier).try_into().unwrap();
            assert_eq!(res, expected.into(), "{i}, {modifier:?}");
        }
        let x = Number::from((1.25, 0..4));
        let res: BigInt = x
            .modify(modifier((0, None), false, false))
            .try_into()
            .unwrap();
        assert_eq!(res, 1.25_f64.to_bits().into());
    }

    #[test]
    fn partial_ord() {
        use std::cmp::Ordering;
        let test_cases: [(Number, Number, Ordering); _] = [
            (1.into(), 2.into(), Ordering::Less),
            (2.into(), 2.into(), Ordering::Equal),
            (3.into(), 2.into(), Ordering::Greater),
            ((1.5, 0..1).into(), (2.5, 0..1).into(), Ordering::Less),
            ((1.5, 0..1).into(), (2.5, 3..5).into(), Ordering::Less),
            ((2.5, 0..1).into(), (2.5, 0..1).into(), Ordering::Equal),
            ((2.5, 0..1).into(), (2.5, 3..5).into(), Ordering::Equal),
            ((3.5, 0..1).into(), (2.5, 0..1).into(), Ordering::Greater),
            ((3.5, 0..1).into(), (2.5, 3..5).into(), Ordering::Greater),
            (2.into(), (3.0, 0..1).into(), Ordering::Less),
            (3.into(), (3.0, 0..1).into(), Ordering::Equal),
            (4.into(), (3.0, 0..1).into(), Ordering::Greater),
            ((2.0, 0..1).into(), 3.into(), Ordering::Less),
            ((3.0, 0..1).into(), 3.into(), Ordering::Equal),
            ((4.0, 0..1).into(), 3.into(), Ordering::Greater),
            (2.into(), (2.5, 0..1).into(), Ordering::Less),
            (3.into(), (2.5, 0..1).into(), Ordering::Greater),
        ];
        for (i, (a, b, res)) in test_cases.into_iter().enumerate() {
            assert_eq!(a.partial_cmp(&b), Some(res), "{i}, a: {a:?}, b: {b:?}");
        }
    }
}
