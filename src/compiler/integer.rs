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

//! Module containing the definition of integers with specific sizes

use num_bigint::{BigInt, BigUint};

use super::ErrorKind;
use crate::architecture::IntegerType;

/// Sized integer
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Integer {
    /// Numeric value
    value: BigUint,
    /// Size of the integer in bits
    size: usize,
    /// Type of the integer
    r#type: Option<IntegerType>,
}

impl Integer {
    /// Creates a new [`Integer`]
    ///
    /// # Parameters
    ///
    /// * `value`: numeric value of the integer
    /// * `size`: size in bits of the integer
    /// * `r#type`: type of the integer
    /// * `signed`: If [`Some`], strictly checks if value fits in the size given as signed/unsigned.
    ///   If [`None`], casts the value to unsigned before checking
    ///
    /// # Errors
    ///
    /// Returns a [`ErrorKind::IntegerOutOfRange`] if the value doesn't fit in the specified size
    #[allow(clippy::missing_panics_doc)] // Function should never panic
    pub fn build(
        value: BigInt,
        size: usize,
        r#type: Option<IntegerType>,
        signed: Option<bool>,
    ) -> Result<Self, ErrorKind> {
        let pow = |n: usize| -> BigInt { BigInt::from(1) << n };
        // Check that the given value fits in the specified size
        let bounds = signed.map_or_else(
            || -pow(size - 1)..pow(size),
            |signed| {
                if signed {
                    let max = pow(size - 1);
                    -max.clone()..max
                } else {
                    BigInt::ZERO..pow(size)
                }
            },
        );
        if !bounds.contains(&value) {
            #[allow(clippy::range_minus_one)] // We need an inclusive range due to the error type
            return Err(ErrorKind::IntegerOutOfRange(
                value,
                bounds.start..=bounds.end - 1,
            ));
        }
        // Mask for bits outside of the specified size
        let mask = (BigInt::from(1u8) << size) - 1u8;
        let value = BigUint::try_from(value & mask);
        Ok(Self {
            value: value.expect("AND'ing a bigint with a positive should always return a positive"),
            size,
            r#type,
        })
    }

    /// Gets the value of the integer
    #[must_use]
    pub const fn value(&self) -> &BigUint {
        &self.value
    }

    /// Gets the type of the integer
    #[must_use]
    pub const fn r#type(&self) -> Option<IntegerType> {
        self.r#type
    }

    /// Gets the size of the integer in bits
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }
}

impl std::fmt::Display for Integer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self.r#type {
            None => write!(f, "{:01$b}", self.value, self.size),
            Some(_) => write!(f, "{:01$X}", self.value, self.size / 4),
        }
    }
}

#[allow(clippy::unwrap_used)]
#[allow(clippy::cast_sign_loss)]
#[cfg(test)]
mod test {
    use super::{ErrorKind, Integer, IntegerType};

    #[test]
    fn bits_signed() {
        for (x, x_str) in [(-8, "1000"), (-5, "1011"), (4, "0100"), (7, "0111")] {
            let val = Integer::build(x.into(), 4, None, Some(true));
            assert_eq!(
                val,
                Ok(Integer {
                    value: (x as u32 & 0b1111).into(),
                    size: 4,
                    r#type: None,
                })
            );
            assert_eq!(val.unwrap().to_string(), x_str);
        }
        for x in [8, -9] {
            assert_eq!(
                Integer::build(x.into(), 4, None, Some(true)),
                Err(ErrorKind::IntegerOutOfRange(
                    x.into(),
                    (-8).into()..=7.into()
                ))
            );
        }
        for x in [i64::MAX, i64::MIN] {
            assert_eq!(
                Integer::build(x.into(), 64, None, Some(true)),
                Ok(Integer {
                    value: (x as u64).into(),
                    size: 64,
                    r#type: None,
                })
            );
        }
    }

    #[test]
    fn bits_unsigned() {
        #[allow(clippy::cast_sign_loss)]
        for (x, x_str) in [(0, "0000"), (4, "0100"), (15, "1111")] {
            let val = Integer::build(x.into(), 4, None, Some(false));
            assert_eq!(
                val,
                Ok(Integer {
                    value: (x as u32 & 0b1111).into(),
                    size: 4,
                    r#type: None,
                })
            );
            assert_eq!(val.unwrap().to_string(), x_str);
        }
        for x in [-1, 16] {
            assert_eq!(
                Integer::build(x.into(), 4, None, Some(false)),
                Err(ErrorKind::IntegerOutOfRange(x.into(), 0.into()..=15.into()))
            );
        }
        for x in [0, i64::MAX] {
            assert_eq!(
                Integer::build(x.into(), 64, None, Some(false)),
                Ok(Integer {
                    value: (x as u64).into(),
                    size: 64,
                    r#type: None,
                })
            );
        }
    }

    #[test]
    fn byte() {
        #[allow(clippy::cast_sign_loss)]
        for (x, x_str) in [(0, "0"), (4, "4"), (15, "F"), (-8, "8"), (-5, "B")] {
            let val = Integer::build(x.into(), 4, Some(IntegerType::Byte), None);
            assert_eq!(
                val,
                Ok(Integer {
                    value: (x as u32 & 0b1111).into(),
                    size: 4,
                    r#type: Some(IntegerType::Byte),
                })
            );
            assert_eq!(val.unwrap().to_string(), x_str);
        }
        for x in [-9, 16] {
            assert_eq!(
                Integer::build(x.into(), 4, Some(IntegerType::Byte), None),
                Err(ErrorKind::IntegerOutOfRange(
                    x.into(),
                    (-8).into()..=15.into()
                ))
            );
        }
    }
}
