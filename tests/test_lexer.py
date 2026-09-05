"""Comprehensive test suite for Straylight hand-written lexer (Phase 1).

Validates tokenization determinism, exact 1-based line and column tracking,
lexical classification, string escape handling, and error conditions.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

# Ensure project root is in sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from straylight.lexer import LexError, Lexer, Tok, tokenize


def test_empty_input() -> None:
    """Empty source or whitespace-only source produces zero tokens."""
    assert tokenize("") == []
    assert tokenize("   \t  \n  \r\n  ") == []


def test_all_token_kinds() -> None:
    """All 8 token kinds (LPAREN, RPAREN, SYMBOL, INT, FLOAT, STRING, BOOL, NIL) are recognized."""
    src = '( ) my_sym 42 3.14 "hello" true false nil'
    toks = tokenize(src)
    kinds = [t.kind for t in toks]
    values = [t.value for t in toks]
    assert kinds == [
        "LPAREN",
        "RPAREN",
        "SYMBOL",
        "INT",
        "FLOAT",
        "STRING",
        "BOOL",
        "BOOL",
        "NIL",
    ]
    assert values == [
        "(",
        ")",
        "my_sym",
        "42",
        "3.14",
        "hello",
        "true",
        "false",
        "nil",
    ]


def test_integers_positive_and_zero() -> None:
    """Positive integers and zero are correctly classified as INT."""
    toks = tokenize("0 1 42 999999")
    assert all(t.kind == "INT" for t in toks)
    assert [t.value for t in toks] == ["0", "1", "42", "999999"]


def test_integers_negative() -> None:
    """Negative integers starting with '-' followed by digits are INT."""
    toks = tokenize("-0 -1 -42 -999999")
    assert all(t.kind == "INT" for t in toks)
    assert [t.value for t in toks] == ["-0", "-1", "-42", "-999999"]


def test_floats_positive() -> None:
    """Floats with digits on both sides of '.' are correctly classified as FLOAT."""
    toks = tokenize("0.0 3.14 0.5 123.456")
    assert all(t.kind == "FLOAT" for t in toks)
    assert [t.value for t in toks] == ["0.0", "3.14", "0.5", "123.456"]


def test_floats_negative() -> None:
    """Negative floats with digits on both sides of '.' are correctly classified as FLOAT."""
    toks = tokenize("-0.0 -3.14 -0.5 -123.456")
    assert all(t.kind == "FLOAT" for t in toks)
    assert [t.value for t in toks] == ["-0.0", "-3.14", "-0.5", "-123.456"]


def test_floats_require_digits_both_sides() -> None:
    """Floats missing digits on either side or with multiple dots raise LexError."""
    # Trailing dot
    with pytest.raises(LexError) as exc1:
        tokenize("1.")
    assert "invalid token '1.'" in str(exc1.value)

    # Leading dot
    with pytest.raises(LexError) as exc2:
        tokenize(".5")
    assert "invalid token '.5'" in str(exc2.value)

    # Multiple dots
    with pytest.raises(LexError) as exc3:
        tokenize("1.2.3")
    assert "invalid token '1.2.3'" in str(exc3.value)

    # Negative trailing dot
    with pytest.raises(LexError) as exc4:
        tokenize("-1.")
    assert "invalid token '-1.'" in str(exc4.value)

    # Negative leading dot
    with pytest.raises(LexError) as exc5:
        tokenize("-.5")
    assert "invalid token '-.5'" in str(exc5.value)


def test_lone_minus_is_symbol() -> None:
    """A lone '-' is the subtraction operator SYMBOL, not a number or invalid token."""
    toks = tokenize("-")
    assert len(toks) == 1
    assert toks[0] == Tok("SYMBOL", "-", 1, 1)

    # In expression context
    expr_toks = tokenize("(- 10 5)")
    assert expr_toks[1] == Tok("SYMBOL", "-", 1, 2)


def test_hyphenated_and_arrow_symbols() -> None:
    """Symbols containing '-' and '>' like 'int->str', 'is-nil', 'str->int' scan as single SYMBOLs."""
    toks = tokenize("int->str is-nil str->int int->float str-cat str-len")
    assert all(t.kind == "SYMBOL" for t in toks)
    assert [t.value for t in toks] == [
        "int->str",
        "is-nil",
        "str->int",
        "int->float",
        "str-cat",
        "str-len",
    ]


def test_operators_as_heads() -> None:
    """All arithmetic and comparison operators are scanned as SYMBOL tokens."""
    ops = "+ - * / == != <= >= < >"
    toks = tokenize(ops)
    assert all(t.kind == "SYMBOL" for t in toks)
    assert [t.value for t in toks] == ops.split()


def test_booleans() -> None:
    """Booleans 'true' and 'false' are scanned as distinct BOOL tokens."""
    toks = tokenize("true false")
    assert len(toks) == 2
    assert toks[0] == Tok("BOOL", "true", 1, 1)
    assert toks[1] == Tok("BOOL", "false", 1, 6)


def test_nil() -> None:
    """The 'nil' keyword is scanned as a NIL token."""
    toks = tokenize("nil")
    assert len(toks) == 1
    assert toks[0] == Tok("NIL", "nil", 1, 1)


def test_strings_with_all_four_escapes() -> None:
    r"""All 4 valid escapes (\n, \t, \", \\) are correctly decoded."""
    src = r'"line\nnewline\ttab\"quote\\backslash"'
    toks = tokenize(src)
    assert len(toks) == 1
    assert toks[0].kind == "STRING"
    assert toks[0].value == 'line\nnewline\ttab"quote\\backslash'


def test_string_invalid_escape_rejected() -> None:
    r"""Invalid escape sequences like \x or \a raise LexError with exact position."""
    src = r'"valid \q invalid"'
    with pytest.raises(LexError) as exc:
        tokenize(src)
    assert "invalid escape \\q" in str(exc.value)
    # Col of escape start: line 1, col 8 is '\'
    assert exc.value.line == 1
    assert exc.value.col == 8


def test_unterminated_string_eof() -> None:
    """String not closed before EOF raises LexError with start position."""
    with pytest.raises(LexError) as exc:
        tokenize('"unclosed string')
    assert "unterminated string literal" in str(exc.value)
    assert exc.value.line == 1
    assert exc.value.col == 1


def test_multiline_string_rejected() -> None:
    """Multi-line strings are forbidden in v0.1 and raise LexError."""
    src = '"first line\nsecond line"'
    with pytest.raises(LexError) as exc:
        tokenize(src)
    assert "unterminated string literal" in str(exc.value)
    assert exc.value.line == 1
    assert exc.value.col == 1


def test_unterminated_string_dangling_escape() -> None:
    """Dangling escape at EOF raises LexError."""
    with pytest.raises(LexError) as exc:
        tokenize('"dangling\\')
    assert "unterminated string literal" in str(exc.value)


def test_comments_ignored() -> None:
    """Comments starting with ';' to end of line are ignored, advancing coordinates."""
    src = """
    ; header comment
    (def x 42) ; inline comment
    ; footer comment
    """
    toks = tokenize(src)
    assert [t.value for t in toks] == ["(", "def", "x", "42", ")"]
    assert toks[0].line == 3
    assert toks[0].col == 5
    assert toks[1].value == "def"
    assert toks[1].line == 3
    assert toks[1].col == 6


def test_exact_line_col_multiline() -> None:
    """Exact 1-based (line, col) coordinates are asserted across multi-line source."""
    src = "(def x\n  (+ 1\n     2))"
    toks = tokenize(src)
    expected = [
        ("LPAREN", "(", 1, 1),
        ("SYMBOL", "def", 1, 2),
        ("SYMBOL", "x", 1, 6),
        ("LPAREN", "(", 2, 3),
        ("SYMBOL", "+", 2, 4),
        ("INT", "1", 2, 6),
        ("INT", "2", 3, 6),
        ("RPAREN", ")", 3, 7),
        ("RPAREN", ")", 3, 8),
    ]
    actual = [(t.kind, t.value, t.line, t.col) for t in toks]
    assert actual == expected


def test_parens_self_delimit() -> None:
    """Parentheses delimit themselves and surrounding tokens without requiring whitespace."""
    toks = tokenize("((+ 1 2))")
    assert [t.kind for t in toks] == [
        "LPAREN",
        "LPAREN",
        "SYMBOL",
        "INT",
        "INT",
        "RPAREN",
        "RPAREN",
    ]
    assert [t.value for t in toks] == ["(", "(", "+", "1", "2", ")", ")"]


def test_identifier_symbols() -> None:
    """Various valid identifier symbols with ?, _, -, digits are classified as SYMBOL."""
    src = "valid? is-zero? _private sum-to fib_2"
    toks = tokenize(src)
    assert all(t.kind == "SYMBOL" for t in toks)
    assert [t.value for t in toks] == ["valid?", "is-zero?", "_private", "sum-to", "fib_2"]


def test_invalid_token_rejected() -> None:
    """Tokens that cannot be classified raise LexError with coordinates."""
    with pytest.raises(LexError) as exc:
        tokenize("(1abc 2)")
    assert "invalid token '1abc'" in str(exc.value)
    assert exc.value.line == 1
    assert exc.value.col == 2
