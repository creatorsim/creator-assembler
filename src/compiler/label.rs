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

//! Module containing the definition of assembly labels and their symbol table

use num_bigint::BigUint;
use std::collections::{hash_map::Entry, HashMap};

use super::{ErrorData, ErrorKind};
use crate::span::Span;

/// Assembly label semantic data
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label {
    /// Address to which the label points
    address: BigUint,
    /// Location of the definition of the label in the assembly. [`None`] if the label comes from a
    /// library
    span: Option<Span>,
}

impl Label {
    /// Gets the address this label is pointing in
    #[must_use]
    pub const fn address(&self) -> &BigUint {
        &self.address
    }

    /// Gets the [`Span`] where the label was defined
    #[must_use]
    pub const fn span(&self) -> Option<Span> {
        self.span
    }
}

/// Symbol table for labels
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table(HashMap<String, Label>);

impl<S: std::hash::BuildHasher> From<HashMap<String, BigUint, S>> for Table {
    fn from(value: HashMap<String, BigUint, S>) -> Self {
        Self(
            value
                .into_iter()
                .map(|(name, address)| {
                    (
                        name,
                        Label {
                            address,
                            span: None,
                        },
                    )
                })
                .collect(),
        )
    }
}

impl IntoIterator for Table {
    type Item = (String, Label);
    type IntoIter = <HashMap<String, Label> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Table {
    /// Inserts a new label
    ///
    /// # Parameters
    ///
    /// * `label`: Label name to insert
    /// * `span`: [`Span`] where the label was defined
    /// * `address`: memory address the label points at
    ///
    /// # Errors
    ///
    /// Errors with [`ErrorKind::DuplicateLabel`] if the label has already been inserted
    pub fn insert(&mut self, label: String, span: Span, address: BigUint) -> Result<(), ErrorData> {
        match self.0.entry(label) {
            Entry::Vacant(e) => {
                let span = Some(span);
                e.insert(Label { address, span });
                Ok(())
            }
            Entry::Occupied(e) => {
                Err(ErrorKind::DuplicateLabel(e.key().clone(), e.get().span).add_span(span))
            }
        }
    }

    /// Returns a reference to the label data corresponding to the given label name
    ///
    /// # Parameters
    ///
    /// * `label`: name of the label to search
    #[must_use]
    pub fn get(&self, label: &str) -> Option<&Label> {
        self.0.get(label)
    }

    /// An iterator visiting all key-value pairs in arbitrary order
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Label)> {
        self.0.iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::span::test::*;

    #[test]
    fn from_library() {
        let label = |s: &str, x: usize| {
            (
                s.into(),
                Label {
                    address: x.into(),
                    span: None,
                },
            )
        };
        let value = |s: &str, x: usize| (s.into(), x.into());
        let labels = HashMap::from([value("test", 10), value("func", 17), value("obj", 101)]);
        let correct = HashMap::from([label("test", 10), label("func", 17), label("obj", 101)]);
        let table = Table::from(labels);
        assert_eq!(table, Table(correct));
    }

    #[test]
    fn insert() {
        let mut table = Table::default();
        let s = (0..2).span();
        assert_eq!(table.insert("test".to_string(), s, 12u8.into()), Ok(()));
        let s = (6..10).span();
        assert_eq!(table.insert("test2".to_string(), s, 0u8.into()), Ok(()));
        let s = (13..17).span();
        assert_eq!(
            table.insert("test".to_string(), (13..17).span(), 4u8.into()),
            Err(ErrorKind::DuplicateLabel("test".to_string(), Some((0..2).span())).add_span(s))
        );
        let s = (20..22).span();
        assert_eq!(
            table.insert("test2".to_string(), (20..22).span(), 128u8.into()),
            Err(ErrorKind::DuplicateLabel("test2".to_string(), Some((6..10).span())).add_span(s))
        );
    }

    #[test]
    fn get() {
        let mut table = Table::default();
        let s = (2..4).span();
        assert_eq!(table.insert("test".to_string(), s, 12u8.into()), Ok(()));
        let s = (5..10).span();
        assert_eq!(table.insert("test2".to_string(), s, 0u8.into()), Ok(()));
        let label = Label {
            address: 12u8.into(),
            span: Some((2..4).span()),
        };
        assert_eq!(table.get("test"), Some(&label));
        let label = Label {
            address: 0u8.into(),
            span: Some((5..10).span()),
        };
        assert_eq!(table.get("test2"), Some(&label));
        assert_eq!(table.get("none"), None);
    }
}
