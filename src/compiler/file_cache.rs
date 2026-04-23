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

use crate::span::{Span, Spanned};

/// ID of a file
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FileID(usize);

/// Expanded pseudoinstruction definition data
#[derive(Debug, Clone, PartialEq, Eq)]
struct File {
    /// Expanded definition
    code: String,
    /// Location where the pseudoinstruction originates from
    origin: Span,
}

/// Cache of expanded pseudoinstruction definitions
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FileCache(Vec<File>);

impl FileID {
    /// File ID of the user code
    pub const SRC: Self = Self(0);
}

impl FileCache {
    /// Add a new pseudoinstruction definition to the cache, returning a reference to the
    /// definition and its ID
    ///
    /// # Parameters
    ///
    /// * `code`: pseudoinstruction definition code
    /// * `origin`: location where the pseudoinstruction originates from
    pub fn add(&mut self, code: String, origin: Span) -> (&str, FileID) {
        let i = self.0.len();
        self.0.push(File { code, origin });
        (self.0[i].code.as_str(), FileID(i + 1))
    }

    /// Returns all buffers related to a given location span, along with their relevant range span,
    /// ordered from closest to the error up to the user assembly code
    ///
    /// # Parameters
    ///
    /// * `src`: user assembly code
    /// * `location`: location span to fetch
    #[must_use]
    pub fn context<'a>(&'a self, src: &'a str, mut location: Span) -> Vec<Spanned<&'a str>> {
        let mut res = Vec::new();
        while location.context.0 != 0 {
            let parent = &self.0[location.context.0 - 1];
            let code = parent.code.as_str();
            res.push((code, location));
            location = parent.origin;
        }
        res.push((src, location));
        res
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use chumsky::span::Span as _;

    #[test]
    fn context() {
        let mut cache = FileCache::default();
        cache.add("gen 1".into(), Span::new(FileID(0), 1..2));
        cache.add("other".into(), Span::new(FileID(0), 0..2));
        cache.add("gen 2".into(), Span::new(FileID(1), 2..3));
        cache.add("gen 3".into(), Span::new(FileID(3), 4..5));
        cache.add("last".into(), Span::new(FileID(1), 3..4));
        assert_eq!(
            cache.context("user code", Span::new(FileID(4), 0..1)),
            vec![
                ("gen 3", Span::new(FileID(4), 0..1)),
                ("gen 2", Span::new(FileID(3), 4..5)),
                ("gen 1", Span::new(FileID(1), 2..3)),
                ("user code", Span::new(FileID(0), 1..2))
            ]
        );
    }
}
