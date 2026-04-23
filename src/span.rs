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

//! Module containing the definition of the spans used to track regions of the assembly source code
//! throughout the crate

pub use crate::compiler::FileID;

/// Range of characters in the source code of an element
pub type Span = chumsky::span::SimpleSpan<usize, FileID>;
/// Value with an attached [`Span`]
pub type Spanned<T> = (T, Span);
/// Simple range of elements
pub type Range = std::ops::Range<usize>;

pub const DEFAULT_SPAN: Span = Span {
    start: 0,
    end: 0,
    context: FileID::SRC,
};

#[cfg(test)]
pub mod test {
    pub use super::{FileID, Range, Span, Spanned};

    pub type Ranged<T> = (T, Range);

    pub trait IntoSpan {
        fn span(self) -> Span;
    }

    impl IntoSpan for Span {
        fn span(self) -> Span {
            self
        }
    }

    impl IntoSpan for Range {
        fn span(self) -> Span {
            chumsky::span::Span::new(FileID::SRC, self)
        }
    }
}
