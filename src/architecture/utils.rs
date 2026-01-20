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

//! Module containing general utilities for deserialization of different types of values

use num_bigint::BigUint;
use num_traits::{Num as _, One as _};
use schemars::JsonSchema;
use serde::{de::Error, Deserialize, Deserializer};

use core::{ops::RangeInclusive, str::FromStr};

/// Thin wrapper for big integers that can be deserialized from JSON, either from a JSON integer or
/// a string representing an integer
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct Integer(pub BigUint);

impl JsonSchema for Integer {
    fn is_referenceable() -> bool {
        false
    }
    fn schema_name() -> String {
        "Integer".to_owned()
    }
    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("Integer")
    }
    fn json_schema(_: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        schemars::schema::SchemaObject {
            instance_type: Some(schemars::schema::InstanceType::Integer.into()),
            ..Default::default()
        }
        .into()
    }
}

impl FromStr for Integer {
    type Err = <BigUint as FromStr>::Err;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(s.parse()?))
    }
}

/// Wrapper for integers that can be deserialized from a string representing an integer in base N
#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord, JsonSchema)]
pub struct BaseN<const N: u8>(#[schemars(with = "String")] pub BigUint);

impl<'de> Deserialize<'de> for Integer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = serde_json::Number::deserialize(deserializer)?;
        Ok(Self(s.as_str().parse().map_err(Error::custom)?))
    }
}

impl<'de, const N: u8> Deserialize<'de> for BaseN<N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s: &str = Deserialize::deserialize(deserializer)?;
        BigUint::from_str_radix(s.trim_start_matches("0x"), N.into())
            .map(Self)
            .map_err(Error::custom)
    }
}

/// Inclusive range guaranteed to be non-empty
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct NonEmptyRangeInclusive<Idx> {
    /// Start of the range
    start: Idx,
    /// Size of the range
    size: Idx,
}

/// Macro to generate methods for non-empty ranges, taking as input the list of concrete types to
/// be used as the generic type parameter
macro_rules! impl_NonEmptyRangeInclusive {
    ($($ty:ty),+) => {
        $(
            impl NonEmptyRangeInclusive<$ty> {
                /// Creates a new [`NonEmptyRangeInclusive`]
                ///
                /// # Parameters
                ///
                /// * `start`: starting value of the range (inclusive)
                /// * `end`: ending value of the range (inclusive)
                #[must_use]
                pub fn build(start: $ty, end: $ty) -> Option<Self> {
                    if start > end {
                        return None;
                    }
                    let size = end - (&start) + <$ty>::one();
                    Some(Self { start, size })
                }

                /// Get the starting value of the range
                #[must_use]
                pub const fn start(&self) -> &$ty {
                    &self.start
                }

                /// Get the size of the range
                #[must_use]
                pub const fn size(&self) -> &$ty {
                    &self.size
                }

                /// Get the ending value of the range
                #[must_use]
                pub fn end(&self) -> $ty {
                    &self.start + &self.size - <$ty>::one()
                }

                /// Check if a value is contained in this range
                #[must_use]
                pub fn contains(&self, x: &$ty) -> bool {
                    *x >= self.start && x - (&self.start) < self.size
                }
            }
        )+
    };
}

impl_NonEmptyRangeInclusive!(BigUint, usize);

impl<'de> Deserialize<'de> for NonEmptyRangeInclusive<BigUint> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let range: RangeInclusive<Integer> = Deserialize::deserialize(deserializer)?;
        let (start, end) = range.into_inner();
        Self::build(start.0, end.0)
            .ok_or("section can't be empty")
            .map_err(Error::custom)
    }
}

impl JsonSchema for NonEmptyRangeInclusive<BigUint> {
    schema_from!(impl: RangeInclusive<Integer>);
}

/// Exclusive non-empty range with a possibly unbound end
#[derive(Deserialize, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(try_from = "(u64, Option<u64>)")]
pub struct RangeFrom {
    /// Start of the range
    pub start: u64,
    /// End of the range
    pub size: Option<u64>,
}
schema_from!(RangeFrom, (u64, Option<u64>));

impl TryFrom<(u64, Option<u64>)> for RangeFrom {
    type Error = &'static str;

    fn try_from((start, end): (u64, Option<u64>)) -> Result<Self, Self::Error> {
        let size = match end.map(|end| end.checked_sub(start)) {
            Some(None) => return Err("the range end must be bigger than the start"),
            Some(Some(x)) => Some(x),
            None => None,
        };
        Ok(Self { start, size })
    }
}

/// Derive implementation of [`JsonSchema`] from the implementation of a different type
macro_rules! schema_from {
    (impl: $src:ty) => {
        fn schema_name() -> String {
            <$src as JsonSchema>::schema_name()
        }

        fn schema_id() -> std::borrow::Cow<'static, str> {
            <$src as JsonSchema>::schema_id()
        }

        fn json_schema(gen: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
            <$src as JsonSchema>::json_schema(gen)
        }

        fn is_referenceable() -> bool {
            <$src as JsonSchema>::is_referenceable()
        }
    };
    ($dst:ident$(<$($lt:lifetime)? $($(,)? $t:ident)?>)?, $src:ty) => {
        impl $(<$($lt)? $(, $t: JsonSchema)?>)? JsonSchema for $dst$(<$($lt)? $(, $t)?>)? {
            $crate::architecture::utils::schema_from!(impl: $src);
        }
    };
}
pub(super) use schema_from;
