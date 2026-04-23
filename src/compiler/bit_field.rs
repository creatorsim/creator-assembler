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

//! Module containing the definition of a bit field

use num_bigint::BigInt;

use super::{ErrorKind, Integer};
use crate::architecture::BitRange;

/// Bit field
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitField(String);

impl BitField {
    /// Creates a new bit field with the given size, initialized to all 0s
    ///
    /// # Parameters
    ///
    /// * `size`: size of the bit field
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self("0".repeat(size))
    }

    /// Replaces a range of bits in the bit field
    ///
    /// # Parameters
    ///
    /// * `start`: starting bit of the range
    /// * `data`: binary string of bits to use as a replacement
    fn replace_range(&mut self, start: usize, data: &str) {
        assert!(start < self.0.len(), "{} <= {}", start, self.0.len());
        let end = self.0.len() - start;
        let start = end - data.len();
        self.0.replace_range(start..end, data);
    }

    /// Replaces a range of bits in the bit field with the given values
    ///
    /// # Parameters
    ///
    /// * `range`: ranges of bits to replace
    /// * `data`: data to use as a replacement
    /// * `signed`: whether the data contains a signed or unsigned number
    ///
    /// # Errors
    ///
    /// Returns a [`ErrorKind::IntegerOutOfRange`] if the data doesn't fit in the bit ranges
    pub fn replace(
        &mut self,
        range: &BitRange,
        data: BigInt,
        signed: bool,
    ) -> Result<(), ErrorKind> {
        let data = Integer::build(data, range.size(), None, Some(signed))?.to_string();
        let mut data = &data[..data.len() - range.padding()];
        for segment in range.iter() {
            let size = *segment.size();
            self.replace_range(*segment.start(), &data[..size]);
            data = &data[size..];
        }
        Ok(())
    }

    /// Extracts a string slice containing the entire [`BitField`]
    #[must_use]
    pub const fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<BitField> for String {
    fn from(value: BitField) -> Self {
        value.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type NonEmptyRangeInclusive = crate::architecture::NonEmptyRangeInclusive<usize>;

    #[must_use]
    fn range(segments: Vec<(usize, usize)>, padding: usize) -> BitRange {
        let ranges = segments
            .into_iter()
            .map(|(a, b)| NonEmptyRangeInclusive::build(b, a).expect("This shouldn't fail"))
            .collect();
        BitRange::build(ranges, padding).expect("this shouldn't fail")
    }

    #[test]
    fn new() {
        assert_eq!(
            BitField::new(32).as_str(),
            "00000000000000000000000000000000"
        );
        assert_eq!(BitField::new(8).as_str(), "00000000");
    }

    #[test]
    fn replace_contiguous() {
        let mut field = BitField::new(16);
        assert_eq!(
            field.replace(&range(vec![(15, 12)], 0), 0b1111.into(), false),
            Ok(())
        );
        assert_eq!(field.as_str(), "1111000000000000");
        assert_eq!(
            field.replace(&range(vec![(2, 0)], 0), 0b101.into(), false),
            Ok(())
        );
        assert_eq!(field.as_str(), "1111000000000101");
    }

    #[test]
    fn replace_separated() {
        let mut field = BitField::new(16);
        assert_eq!(
            field.replace(&range(vec![(15, 12), (7, 6)], 0), 0b10_0111.into(), false),
            Ok(())
        );
        assert_eq!(field.as_str(), "1001000011000000");
    }

    #[test]
    fn replace_padding() {
        let mut field = BitField::new(10);
        assert_eq!(
            field.replace(&range(vec![(7, 2)], 2), 0b1110_0111.into(), false),
            Ok(())
        );
        assert_eq!(field.as_str(), "0011100100");

        let mut field = BitField::new(10);
        assert_eq!(
            field.replace(&range(vec![(7, 2)], 2), (-1).into(), true),
            Ok(())
        );
        assert_eq!(field.as_str(), "0011111100");
    }

    #[test]
    fn replace_error() {
        let mut field = BitField::new(16);
        assert_eq!(
            field.replace(&range(vec![(15, 12)], 0), 18.into(), false,),
            Err(ErrorKind::IntegerOutOfRange(
                18.into(),
                0.into()..=15.into()
            ))
        );
        assert_eq!(
            field.replace(&range(vec![(15, 12)], 0), 8.into(), true),
            Err(ErrorKind::IntegerOutOfRange(
                8.into(),
                (-8).into()..=7.into()
            ))
        );
        assert_eq!(
            field.replace(&range(vec![(3, 1)], 1), 16.into(), false),
            Err(ErrorKind::IntegerOutOfRange(
                16.into(),
                0.into()..=15.into()
            ))
        );
    }
}
