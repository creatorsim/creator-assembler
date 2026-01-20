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

//! Module containing the definition of the memory sections

use num_bigint::BigUint;
use num_traits::Zero as _;

use super::ErrorKind;
use crate::architecture::NonEmptyRangeInclusive;

/// Memory section manager
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// Address of the start of the section
    address: BigUint,
    /// Address of the end of the section
    end: BigUint,
    /// Name of the section
    name: &'static str,
}

impl Section {
    /// Creates a new [`Section`] with the given data
    ///
    /// # Parameters
    ///
    /// * `name`: name of the memory section
    /// * `bounds`: start/end addresses of the section
    #[must_use]
    pub fn new(name: &'static str, bounds: Option<&NonEmptyRangeInclusive<BigUint>>) -> Self {
        bounds.map_or_else(
            || Self {
                name,
                address: 1u8.into(),
                end: BigUint::ZERO,
            },
            |bounds| Self {
                name,
                address: bounds.start().clone(),
                end: bounds.end(),
            },
        )
    }

    /// Gets the first available address
    #[must_use]
    pub const fn get(&self) -> &BigUint {
        &self.address
    }

    /// Reserves space for `size` addresses and returns the address of the beginning of the
    /// reserved space
    ///
    /// # Parameters
    ///
    /// * `size`: amount of addresses to reserve space for
    ///
    /// # Errors
    ///
    /// Returns a [`ErrorKind::MemorySectionFull`] if the there is not enough space in the section
    /// left for the requested allocation
    pub fn try_reserve(&mut self, size: &BigUint) -> Result<BigUint, ErrorKind> {
        let res = self.address.clone();
        self.address += size;
        if self.address > &self.end + 1u8 {
            Err(ErrorKind::MemorySectionFull(self.name))
        } else {
            Ok(res)
        }
    }

    /// Aligns the first available address with the size given in bytes and returns the skipped
    /// region as `(start_addr, size)`. Size is guaranteed to be 0 if the address was already
    /// aligned
    ///
    /// # Parameters
    ///
    /// * `size`: size of the data values
    ///
    /// # Errors
    ///
    /// Returns a [`ErrorKind::MemorySectionFull`] if the there is not enough space in the section
    /// left for the requested alignment
    pub fn try_align(&mut self, align_size: &BigUint) -> Result<(BigUint, BigUint), ErrorKind> {
        let offset = &self.address % align_size;
        if offset.is_zero() {
            return Ok((self.address.clone(), BigUint::ZERO));
        }
        let size = align_size - offset;
        let start = self.try_reserve(&size)?;
        Ok((start, size))
    }

    /// Reserves space for `size` addresses and returns the address of the beginning of the
    /// reserved space, checking that the region is aligned with its size
    ///
    /// # Parameters
    ///
    /// * `size`: amount of addresses to reserve space for
    /// * `word_size`: size of a word in the architecture, in bytes
    ///
    /// # Errors
    ///
    /// Returns a [`ErrorKind::MemorySectionFull`] if the there is not enough space in the section
    /// left for the requested allocation, or a [`ErrorKind::DataUnaligned`] if the region isn't
    /// aligned
    pub fn try_reserve_aligned(
        &mut self,
        size: &BigUint,
        word_size: usize,
    ) -> Result<BigUint, ErrorKind> {
        if !(&self.address % size).is_zero() && !(&self.address % word_size).is_zero() {
            return Err(ErrorKind::DataUnaligned {
                address: self.address.clone(),
                alignment: size.clone(),
            });
        }
        self.try_reserve(size)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[must_use]
    fn range(start: u64, end: u64) -> NonEmptyRangeInclusive<BigUint> {
        NonEmptyRangeInclusive::<BigUint>::build(start.into(), end.into())
            .expect("This shouldn't fail")
    }

    #[test]
    fn reserve1() {
        let one = 1u8.into();
        let mut section = Section::new("test", Some(&range(0, 3)));
        assert_eq!(section.try_reserve(&one), Ok(BigUint::ZERO));
        assert_eq!(section.try_reserve(&one), Ok(1u8.into()));
        assert_eq!(section.try_reserve(&one), Ok(2u8.into()));
        assert_eq!(section.try_reserve(&one), Ok(3u8.into()));
        assert_eq!(
            section.try_reserve(&1u8.into()),
            Err(ErrorKind::MemorySectionFull("test"))
        );
    }

    #[test]
    fn new() {
        let one = 1u8.into();
        let mut section = Section::new("test", Some(&range(0, 0)));
        assert_eq!(section.try_reserve(&one), Ok(BigUint::ZERO));
        assert_eq!(
            section.try_reserve(&1u8.into()),
            Err(ErrorKind::MemorySectionFull("test"))
        );
        let mut section = Section::new("test", None);
        assert_eq!(
            section.try_reserve(&1u8.into()),
            Err(ErrorKind::MemorySectionFull("test"))
        );
    }

    #[test]
    fn reserve4() {
        let four = 4u8.into();
        for i in 1u8..=4 {
            let mut section = Section::new("test2", Some(&range(0, 11)));
            assert_eq!(section.try_reserve(&i.into()), Ok(BigUint::ZERO));
            assert_eq!(section.try_reserve(&four), Ok(i.into()));
            assert_eq!(section.try_reserve(&four), Ok((i + 4).into()));
            assert_eq!(
                section.try_reserve(&four),
                Err(ErrorKind::MemorySectionFull("test2"))
            );
        }
    }

    #[test]
    fn reserve6() {
        let six = 6u8.into();
        for i in 1u8..=6 {
            let mut section = Section::new("test3", Some(&range(0, 17)));
            assert_eq!(section.try_reserve(&i.into()), Ok(BigUint::ZERO));
            assert_eq!(section.try_reserve(&six), Ok(i.into()));
            assert_eq!(section.try_reserve(&six), Ok((i + 6).into()));
            assert_eq!(
                section.try_reserve(&six),
                Err(ErrorKind::MemorySectionFull("test3"))
            );
        }
    }

    #[test]
    fn already_aligned() {
        let four = 4u8.into();
        let mut section = Section::new("test4", Some(&range(0, 11)));
        assert_eq!(section.try_align(&four), Ok((BigUint::ZERO, BigUint::ZERO)));
        assert_eq!(section.try_reserve(&four), Ok(BigUint::ZERO));
        assert_eq!(section.get(), &four);
        assert_eq!(section.try_align(&four), Ok((four.clone(), BigUint::ZERO)));
        assert_eq!(section.get(), &four);
    }

    #[test]
    fn align_memory_limit() {
        for i in 1u8..4 {
            let mut section = Section::new("test5", Some(&range(0, 3)));
            assert_eq!(section.try_reserve(&i.into()), Ok(BigUint::ZERO));
            assert_eq!(
                section.try_align(&4u8.into()),
                Ok((i.into(), (4 - i).into()))
            );
        }
    }

    #[test]
    fn align_fail() {
        let four = 4u8.into();
        for i in 1u8..2 {
            let mut section = Section::new("test6", Some(&range(0, 2)));
            assert_eq!(section.try_align(&four), Ok((BigUint::ZERO, BigUint::ZERO)));
            assert_eq!(section.try_reserve(&i.into()), Ok(BigUint::ZERO));
            assert_eq!(
                section.try_align(&four),
                Err(ErrorKind::MemorySectionFull("test6"))
            );
        }
    }

    #[test]
    fn align4() {
        let four = 4u8.into();
        for i in 1u8..4 {
            let mut section = Section::new("test7", Some(&range(0, 11)));
            assert_eq!(section.try_reserve(&i.into()), Ok(BigUint::ZERO));
            assert_eq!(section.try_align(&four), Ok((i.into(), (4 - i).into())));
            assert_eq!(section.get(), &four);
            assert_eq!(section.try_align(&four), Ok((four.clone(), BigUint::ZERO)));
        }
    }

    #[test]
    fn align6() {
        let six = 6u8.into();
        for i in 1u8..6 {
            let mut section = Section::new("test8", Some(&range(0, 17)));
            assert_eq!(section.try_reserve(&i.into()), Ok(BigUint::ZERO));
            assert_eq!(section.try_align(&six), Ok((i.into(), (6 - i).into())));
            assert_eq!(section.get(), &six);
            assert_eq!(section.try_align(&six), Ok((six.clone(), BigUint::ZERO)));
        }
    }

    #[test]
    fn try_reserve_aligned_ok() {
        let mut section = Section::new("test9", Some(&range(0, 17)));
        assert_eq!(section.try_reserve_aligned(&2u8.into(), 4), Ok(0u8.into()));
        assert_eq!(section.try_reserve_aligned(&2u8.into(), 4), Ok(2u8.into()));
        assert_eq!(section.try_reserve_aligned(&8u8.into(), 4), Ok(4u8.into()));
        assert_eq!(section.try_reserve_aligned(&3u8.into(), 4), Ok(12u8.into()));
        assert_eq!(section.try_reserve_aligned(&3u8.into(), 4), Ok(15u8.into()));
    }

    #[test]
    fn try_reserve_aligned_fail() {
        let mut section = Section::new("test10", Some(&range(0, 17)));
        assert_eq!(
            section.try_reserve_aligned(&1u8.into(), 3),
            Ok(BigUint::ZERO)
        );
        assert_eq!(
            section.try_reserve_aligned(&2u8.into(), 3),
            Err(ErrorKind::DataUnaligned {
                address: 1u8.into(),
                alignment: 2u8.into(),
            })
        );
    }
}
