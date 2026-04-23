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

//! Module containing the definition of the lexer
//!
//! The main entry point for creating the parser is the [`lexer()`] function

use chumsky::{input::WithContext, prelude::*, text::Char as _};
use num_bigint::BigUint;
use num_traits::Num as _;
use std::fmt;

use super::{Parser, Span, Spanned};

/// Thin wrapper for an [`f64`] value that implements `Eq`
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Float(u64);

impl From<Float> for f64 {
    fn from(value: Float) -> Self {
        Self::from_bits(value.0)
    }
}

impl From<f64> for Float {
    fn from(value: f64) -> Self {
        Self(value.to_bits())
    }
}

// Macro to generate the operator enum
macro_rules! operator_token {
    (@count) => (0usize);
    (@count $x:tt $($xs:tt)*) => (1usize + operator_token!(@count $($xs)*));
    ($($i:tt => $o:ident),+ $(,)?) => {
        /// Expression operator token
        #[derive(Debug, PartialEq, Eq, Clone, Copy)]
        pub enum Operator {
            $(#[doc = concat!("`", stringify!($i), "`")] $o,)*
        }

        impl fmt::Display for Operator {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self {
                    $(Self::$o => write!(f, stringify!($i)),)+
                }
            }
        }

        #[cfg(test)]
        static OPERATORS: [(&'static str, Operator); operator_token!(@count $($o)+)] = [
            $((stringify!($i), Operator::$o),)+
        ];
    };
}

operator_token! {
    + => Plus,
    - => Minus,
    * => Star,
    / => Slash,
    % => Percent,
    | => Or,
    & => And,
    ^ => Caret,
    ~ => Tilde,
    > => Gt,
    < => Lt,
    >= => Ge,
    <= => Le,
    != => Ne,
    == => Eq,
    && => LogicalAnd,
    || => LogicalOr,
    << => Shl,
    >> => Shr,
}

/// Tokens created by the lexer
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Token {
    /// Integer literal
    Integer(BigUint),
    /// Floating point literal
    Float(Float),
    /// String literal
    String(String),
    /// Character literal
    Character(char),
    /// Identifier name
    Identifier(String),
    /// Label name
    Label(String),
    /// Directive name
    Directive(String),
    /// Numeric expression operators
    Operator(Operator),
    /// Control characters
    Ctrl(char),
    /// Other literal characters
    Literal(char),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Integer(n) => write!(f, "integer ({n})"),
            Self::Float(x) => write!(f, "floating-point number ({})", f64::from(*x)),
            Self::String(s) => write!(f, "string ({s:?})"),
            Self::Character(c) => write!(f, "character literal ({c:?})"),
            Self::Identifier(i) => write!(f, "identifier ({i})"),
            Self::Label(l) => write!(f, "label ({l})"),
            Self::Directive(d) => write!(f, "directive ({d})"),
            Self::Operator(c) => write!(f, "{c}"),
            Self::Ctrl(c) | Self::Literal(c) => {
                write!(f, "{}", c.escape_debug())
            }
        }
    }
}

/// Creates a lexer for integer literals
#[must_use]
fn int_lexer<'src>() -> Parser!('src, WithContext<Span, &'src str>, Token) {
    static EXPECT_MSG: &str = "The parsed string should always correspond with a valid number";

    // Decimal: integer not followed by a decimal part/exponent
    let decimal = text::int(10)
        // Disambiguate integer literals from the integer part of a floating point literal
        .then_ignore(none_of(".eE").rewind().ignored().or(end()))
        .from_str()
        .map(|x| x.expect(EXPECT_MSG));

    // Generic base N literals
    let to_int = move |x, n| BigUint::from_str_radix(x, n).expect(EXPECT_MSG);
    let base_n = |n| text::digits(n).to_slice().map(move |x| to_int(x, n));
    let hex = one_of("xX").ignore_then(base_n(16));
    let bin = one_of("bB").ignore_then(base_n(2));
    let octal = base_n(8);
    let base_n = just("0").ignore_then(choice((hex, bin, octal)));

    // Integer token
    let int = base_n.or(decimal).map(Token::Integer);
    int.labelled("integer").as_context()
}

/// Creates a lexer for floating point literals
#[must_use]
fn float_lexer<'src>() -> Parser!('src, WithContext<Span, &'src str>, Token) {
    let int = text::int(10); // Integer part
    let frac = just('.').then(text::digits(10)); // Fractional part
    let exp = one_of("eE").then(one_of("+-").or_not()).then(int); // Exponent part

    // Float literal: `float -> int [frac] [exp]`
    let float = int
        // Disambiguate integer literals from the integer part of a floating point literal
        .then_ignore(one_of(".eE").rewind())
        .then(frac.or_not())
        .then(exp.or_not())
        .to_slice()
        .from_str()
        .map(|res: Result<f64, _>| res.expect("We already parsed it as a float literal"));

    // named constants: `inf`, `infinity`, and `nan`
    let named_constant = text::ident().try_map(|ident: &str, span| {
        Ok(match ident.to_lowercase().as_str() {
            "inf" | "infinity" => f64::INFINITY,
            "nan" => f64::NAN,
            _ => return Err(Rich::custom(span, "Unallowed float literal")),
        })
    });

    // Float token
    choice((float, named_constant))
        .map(|x| Token::Float(x.into()))
        .labelled("float")
        .as_context()
}

/// Creates a lexer for string and character literals
#[must_use]
// We are using `impl Trait` types, so we can't split them into type aliases
#[allow(clippy::type_complexity)]
fn str_lexer<'src>() -> (
    Parser!('src, WithContext<Span, &'src str>, Token),
    Parser!('src, WithContext<Span, &'src str>, Token),
) {
    // Escape sequences in strings
    let escape = just('\\').ignore_then(
        choice((
            just('\\'),
            just('"'),
            just('\''),
            just('a').to('\x07'),
            just('b').to('\x08'),
            just('e').to('\x1B'),
            just('f').to('\x0C'),
            just('n').to('\n'),
            just('r').to('\r'),
            just('t').to('\t'),
            just('0').to('\0'),
        ))
        .map_err(|e: Rich<'_, char, Span>| {
            let mut s = *e.span();
            s.start -= 1; // Include the `\` prefix in the span
            Rich::custom(s, "Invalid escape sequence")
        }),
    );

    // Characters allowed inside string/character literals: anything that isn't their delimiter,
    // a backslash, or a new line
    let char = |delimiter| {
        // This would be better written with `.filter()` and `Char::is_newline()`, but `.filter()`
        // doesn't give the correct span due to a bug, so we need to manually copy over the newline
        // list from `Char::is_newline()`
        // TODO: replace with `any().filter(|c| !['\\', delimiter].contains(c) && !c.is_newline())`
        // on chumsky 0.10.2
        none_of([
            '\\', delimiter, '\n', '\r', '\x0B', '\x0C', '\u{0085}', '\u{2028}', '\u{2029}',
        ])
        .or(escape)
    };
    let err = |msg| move |e: Rich<'_, _, Span>| Rich::custom(*e.span(), msg);

    // Literal strings: `string -> " char* "`
    let string = char('"')
        .repeated()
        .collect()
        .delimited_by(
            just('"'),
            just('"').map_err(err("Unterminated string literal")),
        )
        .map(Token::String)
        .labelled("string")
        .as_context();

    // Literal characters: `character -> ' char '`
    let character = char('\'')
        .delimited_by(
            just('\''),
            just('\'').map_err(err("Unterminated character literal")),
        )
        .map(Token::Character)
        .labelled("character")
        .as_context();

    (string, character)
}

/// Creates a lexer for the input
///
/// # Parameters
///
/// * `comment_prefix`: string to use as line comment prefix
#[must_use]
pub fn lexer<'src, 'arch: 'src>(
    comment_prefix: &'arch str,
) -> Parser!('src, WithContext<Span, &'src str>, Vec<Spanned<Token>>) {
    let newline = text::newline().to('\n');

    // Integer literals
    let int = int_lexer();
    // Float literals
    let float = float_lexer();
    // Number literals can be either integers or floats
    let num = int.or(float);

    // Expression operators
    let single_char_op = select! {
        '+' => Operator::Plus,
        '-' => Operator::Minus,
        '*' => Operator::Star,
        '/' => Operator::Slash,
        '%' => Operator::Percent,
        '|' => Operator::Or,
        '&' => Operator::And,
        '^' => Operator::Caret,
        '~' => Operator::Tilde,
        '>' => Operator::Gt,
        '<' => Operator::Lt,
    };
    let double_char_op = any().repeated().exactly(2).to_slice().try_map(|op, span| {
        match op {
            ">=" => Ok(Operator::Ge),
            "<=" => Ok(Operator::Le),
            "!=" => Ok(Operator::Ne),
            "==" => Ok(Operator::Eq),
            "&&" => Ok(Operator::LogicalAnd),
            "||" => Ok(Operator::LogicalOr),
            "<<" => Ok(Operator::Shl),
            ">>" => Ok(Operator::Shr),
            // Use a generic error if we don't match a valid operator. The characters can always be
            // lexed as a combination of other tokens so this will never appear in the output
            _ => Err(Rich::custom(span, "unknown operator")),
        }
    });
    let op = choice((double_char_op, single_char_op))
        .map(Token::Operator)
        .labelled("operator");

    // Control characters used in the grammar
    let ctrl = one_of(",()")
        .or(newline)
        .map(Token::Ctrl)
        .labelled("control character");

    // Other literal punctuation characters. This should be the last option if all other patterns
    // fail, so we don't need to be too specific to avoid ambiguities with other patterns
    let literal = any()
        .filter(|c: &char| !(c.is_ascii_alphanumeric() || "\"'_.".contains(*c)))
        .map(Token::Literal)
        .labelled("literal");

    // Generic identifiers
    let ident = any()
        .filter(|c: &char| c.is_ascii_alphabetic() || "_.".contains(*c))
        .then(
            any()
                .filter(|c: &char| c.is_ascii_alphanumeric() || "_.".contains(*c))
                .repeated(),
        )
        .to_slice()
        .map(ToString::to_string)
        .labelled("identifier")
        .as_context();

    // Identifiers (names/labels/directives)
    let identifier = ident
        .then(just(':').or_not().map(|x| x.is_some()))
        .map(|(ident, label)| {
            if label {
                Token::Label(ident)
            } else if ident.starts_with('.') && ident != "." {
                Token::Directive(ident)
            } else {
                Token::Identifier(ident)
            }
        });

    // String/character literals
    let (string, character) = str_lexer();

    // Any of the previous patterns can be a token
    let token = choice((op, ctrl, num, identifier, string, character, literal));

    // Comments
    let line_comment = just(comment_prefix)
        .then(any().and_is(newline.not()).repeated())
        .ignored();
    let not = |x| any().and_is(just(x).not()).repeated();
    let multiline_comment = not("*/").delimited_by(just("/*"), just("*/"));
    // Whitespace that isn't new lines
    let whitespace = any()
        .filter(|c: &char| c.is_whitespace() && !c.is_newline())
        .ignored();

    let padding = choice((line_comment, multiline_comment, whitespace)).repeated();

    // Definition of a token
    token
        .map_with(|tok, e| (tok, e.span()))
        .separated_by(padding)
        .allow_leading()
        .allow_trailing()
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod test {
    use chumsky::label::LabelError;

    use super::*;
    use crate::span::test::*;

    fn lex(code: &str) -> Result<Vec<Spanned<Token>>, ()> {
        lexer("#")
            .parse(code.with_context(FileID::SRC))
            .into_result()
            .map_err(|e| eprintln!("{e:?}"))
    }

    fn error(err: Ranged<&str>, context: &[Ranged<&'static str>]) -> Rich<'static, char, Span> {
        let mut err = Rich::custom(err.1.span(), err.0);
        for (label, span) in context {
            <_ as LabelError<'_, WithContext<_, &str>, _>>::in_context(
                &mut err,
                *label,
                span.clone().span(),
            );
        }
        err
    }

    #[test]
    fn int() {
        let test_cases = [
            // decimal
            ("0", 0),
            ("1", 1),
            ("1234", 1234),
            (&u32::MAX.to_string(), u128::from(u32::MAX)),
            (&u128::MAX.to_string(), u128::MAX),
            // octal
            ("00", 0),
            ("01", 1),
            ("010", 8),
            // hex
            ("0x0", 0),
            ("0x1", 1),
            ("0xf", 15),
            ("0x10", 16),
            ("0X10", 16),
            ("0xFf", 255),
            ("0xAAAAAAAAAAAAAAAAAAAB", 0xAAAA_AAAA_AAAA_AAAA_AAAB_u128),
            // binary
            ("0b0", 0),
            ("0b1", 1),
            ("0b10", 2),
            ("0B10", 2),
            ("0b10011001100110011001100110011001100110011001100110011001100110011001100110011001", 0x9999_9999_9999_9999_9999_u128),
        ];
        for (s, v) in test_cases {
            let v = v.into();
            let span = (0..s.len()).span();
            assert_eq!(lex(s), Ok(vec![(Token::Integer(v), span)]), "`{s}`");
        }
    }

    #[test]
    fn float() {
        let float_tok = |x: &str| Token::Float(x.parse::<f64>().unwrap().into());
        let test_cases = [
            "0.0", "1.0", "0.1", "100.0", "100.01", "100e1", "100E1", "0.5e1", "0.5e1", "0.5e+1",
            "0.5e-1", "0.5e0", "0.5e+0", "0.5e-0", "1e300", "1e400", "1e-30", "inf", "INF", "Inf",
            "infinity", "Infinity", "INFINITY", "nan", "NAN", "NaN",
        ];
        for s in test_cases {
            let span = (0..s.len()).span();
            assert_eq!(lex(s), Ok(vec![(float_tok(s), span)]), "`{s}`");
        }
    }

    const ESCAPE_SEQUENCES: [(&str, char); 11] = [
        ("\"", '\"'),
        ("\'", '\''),
        ("\\", '\\'),
        ("n", '\n'),
        ("r", '\r'),
        ("t", '\t'),
        ("0", '\0'),
        ("a", '\x07'),
        ("b", '\x08'),
        ("e", '\x1B'),
        ("f", '\x0C'),
    ];

    const NEWLINES: [char; 7] = [
        '\n', '\r',       // Common newlines
        '\x0B',     // Vertical tab
        '\x0C',     // Form feed
        '\u{0085}', // Next line
        '\u{2028}', // Line separator
        '\u{2029}', // Paragraph separator
    ];

    #[test]
    fn string() {
        // normal strings
        for s in ["", "a", "test", "TEST", "0a", "π √  🅐 󰸞"] {
            assert_eq!(
                lex(&format!("\"{s}\"")),
                Ok(vec![(Token::String(s.into()), (0..s.len() + 2).span())])
            );
        }
        // escape sequences
        for (s, res) in ESCAPE_SEQUENCES {
            let span = (0..s.len() + 3).span();
            assert_eq!(
                lex(&format!("\"\\{s}\"")),
                Ok(vec![(Token::String(res.to_string()), span)])
            );
        }
        let msg = "\"a string with escape sequences like newline `\\n` or tabs `\\t`, also quotes `\\\"` and literal backslashes `\\\\`\"";
        assert_eq!(lex(msg), Ok(vec![(Token::String("a string with escape sequences like newline `\n` or tabs `\t`, also quotes `\"` and literal backslashes `\\`".into()), (0..msg.len()).span())]));
        let err = error(("Invalid escape sequence", 8..10), &[("string", 0..9)]);
        let src = "\"invalid\\z\"".with_context(FileID::SRC);
        assert_eq!(lexer("#").parse(src).into_result(), Err(vec![err]));
        for newline in NEWLINES {
            let s = 5..5 + newline.len_utf8();
            let err = error(("Unterminated string literal", s), &[("string", 0..5)]);
            let src = format!("\"test{newline}test");
            let src = src.with_context(FileID::SRC);
            let res = lexer("#").parse(src).into_result();
            assert_eq!(res, Err(vec![err]), "{newline:?}");
        }
    }

    #[test]
    fn char() {
        let ascii = ('!'..='~').filter(|c| !"\\\'".contains(*c));
        let chars = ascii
            .chain('¡'..='±')
            .chain('Σ'..='ω')
            .chain('ᴀ'..='ᴊ')
            .chain('←'..='↙')
            .chain('⎛'..='⎿')
            .chain('─'..='⎮')
            .chain('龱'..='龺')
            .chain('Ꭓ'..='ꞷ')
            .chain(''..='')
            .chain('𐝈'..='𐝌')
            .chain('𛰙'..='𛰜')
            .chain('🮤'..='🮧')
            .chain('󰀁'..='󰀘');
        for c in chars {
            assert_eq!(
                lex(&format!("'{c}'")),
                Ok(vec![(Token::Character(c), (0..c.len_utf8() + 2).span())])
            );
        }
        for (s, res) in ESCAPE_SEQUENCES {
            assert_eq!(
                lex(&format!("'\\{s}'")),
                Ok(vec![(Token::Character(res), (0..s.len() + 3).span())])
            );
        }
        let err = error(("Invalid escape sequence", 1..3), &[("character", 0..2)]);
        let src = "'\\z'".with_context(FileID::SRC);
        assert_eq!(lexer("#").parse(src).into_result(), Err(vec![err]));
        let err = error(
            ("Unterminated character literal", 2..3),
            &[("character", 0..2)],
        );
        let src = "'a\ntest".with_context(FileID::SRC);
        assert_eq!(lexer("#").parse(src).into_result(), Err(vec![err]));
        for newline in NEWLINES {
            let s = 2..2 + newline.len_utf8();
            let err = error(
                ("Unterminated character literal", s),
                &[("character", 0..2)],
            );
            let src = format!("'a{newline}test");
            let src = src.with_context(FileID::SRC);
            let res = lexer("#").parse(src).into_result();
            assert_eq!(res, Err(vec![err]), "{newline:?}");
        }
    }

    #[test]
    fn ident() {
        let test_cases = [
            "addi",
            "fclass.s",
            "fmul.d",
            "addi0",
            "addi1",
            "addi2",
            "ident_with_underscores_and.dots.",
            "_start_underscore",
            "a._123",
            "z....___1",
            "_1_",
            "_",
            ".",
        ];
        for s in test_cases {
            let span = (0..s.len()).span();
            assert_eq!(lex(s), Ok(vec![(Token::Identifier(s.into()), span)]));
        }
    }

    #[test]
    fn label() {
        let test_cases = [
            "label:",
            ".label:",
            "label_with_underscores_and.dots.:",
            "label3:",
            ".L3:",
            "L0_:",
            "L_1:",
            ".a...___12:",
            "z....___1:",
            "z....___1..:",
            "z....___1__:",
            ".1_:",
            "_.1_a:",
            ".:",
        ];
        for s in test_cases {
            let l = s.len();
            let span = (0..l).span();
            assert_eq!(lex(s), Ok(vec![(Token::Label(s[..l - 1].into()), span)]));
        }
    }

    #[test]
    fn directive() {
        let test_cases = [
            ".directive",
            ".dir_with_underscores_and.dots.",
            ".string",
            ".data",
            ".L3",
            ".L_1",
            ".z....___1",
            ".z....___1__",
            ".z....___1..",
            ".1_",
            "._1_a",
        ];
        for s in test_cases {
            let span = (0..s.len()).span();
            assert_eq!(lex(s), Ok(vec![(Token::Directive(s.into()), span)]));
        }
    }

    #[test]
    fn operator() {
        for (s, op) in OPERATORS {
            #[allow(clippy::range_plus_one)]
            let span = (1..s.len() + 1).span();
            assert_eq!(
                lex(&format!(" {s} ")),
                Ok(vec![(Token::Operator(op), span)])
            );
        }
    }

    #[test]
    fn ctrl() {
        for c in ",()".chars() {
            let span = (0..1).span();
            assert_eq!(lex(&c.to_string()), Ok(vec![(Token::Ctrl(c), span)]));
        }
    }

    #[test]
    fn newline() {
        for s in NEWLINES {
            let span = (0..s.len_utf8()).span();
            let src = s.to_string();
            assert_eq!(lex(&src), Ok(vec![(Token::Ctrl('\n'), span)]), "{s:?}");
        }
        assert_eq!(lex("\r\n"), Ok(vec![(Token::Ctrl('\n'), (0..2).span())]));
    }

    #[test]
    fn literal() {
        for c in "@!?=:;${}[]\\".chars() {
            let span = (0..1).span();
            assert_eq!(lex(&c.to_string()), Ok(vec![(Token::Literal(c), span)]));
        }
    }

    #[test]
    fn padding() {
        let utf8_len = "/* π √  🅐 󰸞 */".len();
        let test_cases = [
            ("  a", 2..3),
            ("\u{A0}a", 2..3),
            ("a  ", 0..1),
            ("  abc    ", 2..5),
            ("  \ta\t\t", 3..4),
            ("\t\t\ta", 3..4),
            ("a\t\u{A0}\t\u{1680}", 0..1),
            ("\ta\t\t", 1..2),
            (" \t\ttest  \u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\u{2006}\u{2007}\u{2008}\u{2009}\u{200A} \t \t", 3..7),
            (" \t\ttest\u{202F}\u{205F}\u{3000}", 3..7),
            ("/* inline comment */ test", 21..25),
            ("test /* inline comment */", 0..4),
            ("/* inline comment */ test  /* inline comment */", 21..25),
            (
                "/* π √  🅐 󰸞 */ test  /* inline comment */",
                utf8_len + 1..utf8_len + 5,
            ),
            ("test # asd", 0..4),
            ("test #asd", 0..4),
        ];
        for (s, v) in test_cases {
            assert_eq!(
                lex(s),
                Ok(vec![(Token::Identifier(s[v.clone()].into()), v.span())]),
                "`{s}`"
            );
        }
        assert_eq!(
            lex("#comment\ntest"),
            Ok(vec![
                (Token::Ctrl('\n'), (8..9).span()),
                (Token::Identifier("test".into()), (9..13).span())
            ])
        );
        for (s, v) in [("test // asd", 0..4), ("test //asd", 0..4)] {
            let res = lexer("//").parse(s.with_context(FileID::SRC)).into_result();
            assert_eq!(
                res.map_err(|e| eprintln!("{e:?}")),
                Ok(vec![(Token::Identifier(s[v.clone()].into()), v.span())]),
                "`{s}`"
            );
        }
    }

    #[test]
    fn sequence() {
        let src = "a 1 .z +- test:  ]\t='x'\"string\" <= >= &<";
        let tokens = [
            (Token::Identifier("a".into()), 0..1),
            (Token::Integer(1u8.into()), 2..3),
            (Token::Directive(".z".into()), 4..6),
            (Token::Operator(Operator::Plus), 7..8),
            (Token::Operator(Operator::Minus), 8..9),
            (Token::Label("test".into()), 10..15),
            (Token::Literal(']'), 17..18),
            (Token::Literal('='), 19..20),
            (Token::Character('x'), 20..23),
            (Token::String("string".into()), 23..31),
            (Token::Operator(Operator::Le), 32..34),
            (Token::Operator(Operator::Ge), 35..37),
            (Token::Operator(Operator::And), 38..39),
            (Token::Operator(Operator::Lt), 39..40),
        ]
        .into_iter()
        .map(|(t, s)| (t, s.span()))
        .collect();
        assert_eq!(lex(src), Ok(tokens));
    }

    #[test]
    fn empty() {
        let test_cases = [
            "",
            " ",
            "    ",
            "  \t  ",
            "#a",
            "/*a*/",
            " \t #a",
            " /*a*/ \t /*b*/ #c",
        ];
        for s in test_cases {
            assert_eq!(lex(s), Ok(vec![]), "`{s}`");
        }
    }
}
