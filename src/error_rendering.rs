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

//! Module containing utilities for rendering compiler errors

use ariadne::{Color, Fmt};

use std::fmt;

/// Wrapper to display elements with an optional color
#[derive(Debug, PartialEq, Eq)]
pub struct Colored<T>(pub T, pub Option<Color>);

impl<T: fmt::Display> fmt::Display for Colored<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(color) = self.1 {
            write!(f, "{}", (&self.0).fg(color))
        } else {
            write!(f, "`{}`", self.0)
        }
    }
}

/// Wrapper for a vector to display it as a list of values
#[derive(Debug, PartialEq, Eq)]
pub struct DisplayList<T> {
    /// List of values to display
    pub values: Vec<T>,
    /// Whether to display the names with colors or not
    pub color: bool,
}

impl<T: std::cmp::Ord> DisplayList<T> {
    /// Creates a new [`DisplayList`], checking that it isn't empty
    #[must_use]
    pub fn non_empty(names: Vec<T>, color: bool) -> Option<Self> {
        (!names.is_empty()).then_some(Self::new(names, color))
    }

    /// Creates a new [`DisplayList`]
    #[must_use]
    pub fn new(mut values: Vec<T>, color: bool) -> Self {
        values.sort_unstable();
        Self { values, color }
    }
}

impl<T: fmt::Display> fmt::Display for DisplayList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Get the last value, if we can't the list is empty and we don't have to do anything
        let Some(last) = self.values.last() else {
            return Ok(());
        };
        let color = self.color.then_some(Color::Green);
        // Only use a comma to separate values if there are more than 2
        let comma = if self.values.len() > 2 { "," } else { "" };
        // Write each of the values except the last, appending an optional comma
        for x in &self.values[..self.values.len() - 1] {
            write!(f, "{}{comma} ", Colored(x, color))?;
        }
        // If there are multiple values add an `or` before the last one
        if self.values.len() > 1 {
            write!(f, "or ")?;
        }
        // Write the last value
        write!(f, "{}", Colored(last, color))
    }
}

/// Wrapper to display an amount of arguments
#[derive(Debug, PartialEq, Eq)]
pub struct ArgNum(pub usize, pub Option<Color>);

impl fmt::Display for ArgNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Append an `s` if the amount of arguments isn't 1
        let s = if self.0 == 1 { "" } else { "s" };
        write!(f, "{} argument{}", Colored(self.0, self.1), s)
    }
}

// NOTE: obtained and adapted from rustc
// https://github.com/rust-lang/rust/blob/master/compiler/rustc_span/src/edit_distance.rs
/// Finds the [edit distance] between two strings.
///
/// Returns `None` if the distance exceeds the limit.
///
/// [edit distance]: https://en.wikipedia.org/wiki/Edit_distance
// We need to collect the string in a vector of chars to avoid iterating over it for each char
#[allow(clippy::needless_collect)]
fn edit_distance(a: &str, b: &str, limit: usize) -> Option<usize> {
    use std::{cmp, mem};
    let mut a = &a.chars().collect::<Vec<_>>()[..];
    let mut b = &b.chars().collect::<Vec<_>>()[..];

    // Ensure that `b` is the shorter string, minimizing memory use.
    if a.len() < b.len() {
        mem::swap(&mut a, &mut b);
    }

    let min_dist = a.len() - b.len();
    // If we know the limit will be exceeded, we can return early.
    if min_dist > limit {
        return None;
    }

    // Strip common prefix.
    while let Some(((b_char, b_rest), (a_char, a_rest))) = b.split_first().zip(a.split_first()) {
        if a_char != b_char {
            break;
        }
        a = a_rest;
        b = b_rest;
    }
    // Strip common suffix.
    while let Some(((b_char, b_rest), (a_char, a_rest))) = b.split_last().zip(a.split_last()) {
        if a_char != b_char {
            break;
        }
        a = a_rest;
        b = b_rest;
    }

    // If either string is empty, the distance is the length of the other.
    // We know that `b` is the shorter string, so we don't need to check `a`.
    if b.is_empty() {
        return Some(min_dist);
    }

    let mut prev_prev = vec![usize::MAX; b.len() + 1];
    let mut prev = (0..=b.len()).collect::<Vec<_>>();
    let mut current = vec![0; b.len() + 1];

    // row by row
    for i in 1..=a.len() {
        current[0] = i;
        let a_idx = i - 1;

        // column by column
        for j in 1..=b.len() {
            let b_idx = j - 1;

            // There is no cost to substitute a character with itself.
            let substitution_cost = usize::from(a[a_idx] != b[b_idx]);

            current[j] = cmp::min(
                // deletion
                prev[j] + 1,
                cmp::min(
                    // insertion
                    current[j - 1] + 1,
                    // substitution
                    prev[j - 1] + substitution_cost,
                ),
            );

            if (i > 1) && (j > 1) && (a[a_idx] == b[b_idx - 1]) && (a[a_idx - 1] == b[b_idx]) {
                // transposition
                current[j] = cmp::min(current[j], prev_prev[j - 2] + 1);
            }
        }

        // Rotate the buffers, reusing the memory.
        [prev_prev, prev, current] = [prev, current, prev_prev];
    }

    // `prev` because we already rotated the buffers.
    let distance = prev[b.len()];
    (distance <= limit).then_some(distance)
}

/// Gets the names from a list that are the most similar to the given name
///
/// # Parameters
///
/// * `target`: target name to match against
/// * `names`: iterator of possible names
#[must_use]
pub fn get_similar<'a>(target: &str, names: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    use std::collections::hash_map::Entry;
    let mut distances = std::collections::HashMap::new();
    let limit = std::cmp::max(target.len() / 3, 1);
    // For each candidate name, calculate its distance to the target if we haven't processed it yet
    for name in names {
        if let Entry::Vacant(e) = distances.entry(name) {
            if let Some(d) = edit_distance(name, target, limit) {
                e.insert(d);
            }
        }
    }
    // Get the names with the minimum distance
    distances
        .iter()
        .map(|(_, &d)| d)
        .min()
        .map(|min| {
            // Get the names with the minimum distance
            distances
                .iter()
                .filter(|(_, &d)| d == min)
                .map(|(&name, _)| name)
                .collect()
        })
        .unwrap_or_default()
}

/// Trait representing an error that can be rendered for display
pub trait RenderError {
    /// Write the formatted error to a buffer. The written bytes should correspond to valid UTF-8
    ///
    /// # Parameters
    ///
    /// * `filename`: name of the file with the code
    /// * `src`: original source code parsed
    /// * `buffer`: writer in which to write the formatted error
    /// * `color`: whether to enable colors or not
    fn format(&self, filename: &str, src: &str, buffer: &mut Vec<u8>, color: bool);

    /// Render the error to a string
    ///
    /// # Parameters
    ///
    /// * `filename`: name of the file with the code
    /// * `src`: original source code parsed
    /// * `color`: whether to enable colors or not
    #[must_use]
    fn render(&self, filename: &str, src: &str, color: bool) -> String {
        let mut buffer = Vec::new();
        self.format(filename, src, &mut buffer, color);
        String::from_utf8(buffer).expect("the rendered error should be valid UTF-8")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn display_colored() {
        assert_eq!(&Colored(1, None).to_string(), "`1`");
        assert_eq!(
            &Colored("test", Some(Color::Red)).to_string(),
            &"test".fg(Color::Red).to_string()
        );
        assert_eq!(
            &Colored(10, Some(Color::Blue)).to_string(),
            &"10".fg(Color::Blue).to_string()
        );
    }

    #[test]
    fn display_name_list() {
        let list = |vals: Vec<i32>, color| DisplayList::new(vals, color).to_string();
        let green = |x: &'static str| x.fg(Color::Green);
        assert_eq!(&list(vec![], false), "");
        assert_eq!(&list(vec![], true), "");
        assert_eq!(&list(vec![1], false), "`1`");
        assert_eq!(&list(vec![1, 2], false), "`1` or `2`");
        assert_eq!(&list(vec![1, 3, 2], false), "`1`, `2`, or `3`");
        assert_eq!(&list(vec![4, 3, 2, 1], false), "`1`, `2`, `3`, or `4`");
        assert_eq!(&list(vec![1], true), &1.fg(Color::Green).to_string());
        assert_eq!(
            &DisplayList::new(vec!["foo", "bar"], true).to_string(),
            &format!("{} or {}", green("bar"), green("foo"))
        );
        assert_eq!(
            &DisplayList::new(vec!["c", "a", "b"], true).to_string(),
            &format!("{}, {}, or {}", green("a"), green("b"), green("c"))
        );
    }

    #[test]
    fn display_list_new() {
        assert_eq!(
            DisplayList::<i8>::new(vec![], false),
            DisplayList {
                values: vec![],
                color: false
            }
        );
        assert_eq!(
            DisplayList::new(vec!["a"], true),
            DisplayList {
                values: vec!["a"],
                color: true
            }
        );
        assert_eq!(
            DisplayList::new(vec![1, 2, 3], false),
            DisplayList {
                values: vec![1, 2, 3],
                color: false
            }
        );
    }

    #[test]
    fn display_list_non_empty() {
        assert_eq!(DisplayList::<u8>::non_empty(vec![], false), None);
        assert_eq!(DisplayList::<u8>::non_empty(vec![], true), None);
        assert_eq!(
            DisplayList::non_empty(vec!["a"], true),
            Some(DisplayList::new(vec!["a"], true))
        );
        assert_eq!(
            DisplayList::non_empty(vec![1, 2, 3], false),
            Some(DisplayList::new(vec![1, 2, 3], false))
        );
    }

    #[test]
    fn display_arg_num() {
        assert_eq!(&ArgNum(0, None).to_string(), "`0` arguments");
        assert_eq!(&ArgNum(1, None).to_string(), "`1` argument");
        assert_eq!(&ArgNum(2, None).to_string(), "`2` arguments");
        assert_eq!(&ArgNum(10, None).to_string(), "`10` arguments");

        assert_eq!(
            &ArgNum(0, Some(Color::Red)).to_string(),
            &format!("{} arguments", 0.fg(Color::Red))
        );
        assert_eq!(
            &ArgNum(1, Some(Color::Green)).to_string(),
            &format!("{} argument", 1.fg(Color::Green))
        );
        assert_eq!(
            &ArgNum(2, Some(Color::Blue)).to_string(),
            &format!("{} arguments", 2.fg(Color::Blue))
        );
    }

    #[test]
    fn similar_names() {
        let sim = |target, values| {
            let mut res = get_similar(target, values);
            res.sort_unstable();
            res
        };
        assert_eq!(sim("test", vec!["testtest"]), Vec::<&str>::new());
        assert_eq!(sim("tests0", vec!["test"]), vec!["test"]);
        assert_eq!(sim("tes", vec!["te", "te", "te"]), vec!["te"]);
        assert_eq!(sim("x2", vec!["x0", "x1"]), vec!["x0", "x1"]);
        assert_eq!(sim("x20", vec!["x0", "x1"]), vec!["x0"]);
        assert_eq!(
            sim("test", vec!["aest", "tst", "tests", "tset", "tsts", "aa"]),
            vec!["aest", "tests", "tset", "tst"]
        );
    }
}
