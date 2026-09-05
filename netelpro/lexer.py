"""Netelpro lexer -- Phase 1 hand-written tokenizer.

The prosecutor thesis:
Netelpro is designed for LLMs to generate and mechanically verify syntax
by counting parentheses and operands against a fixed-arity table. For this
contract to hold, tokenization must be completely deterministic, transparent,
and report exact 1-based line and column coordinates for every token.

Syntax rules enforced at lexical stage:
- Delimiters: '(' and ')' self-delimit and delimit surrounding tokens.
- Comments: ';' to end-of-line is ignored; advances line/col tracking.
- Integers: -?[0-9]+
- Floats: -?[0-9]+\\.[0-9]+ (digits required on both sides of '.')
- Strings: "..." with escapes \\n, \\t, \\", \\\\ only. Unterminated strings
  or invalid escapes abort with exact positions. Multi-line strings are
  forbidden in v0.1.
- Booleans: 'true' and 'false'.
- Nil: 'nil' (the empty list).
- Symbols: [A-Za-z_][A-Za-z0-9_?->]* or operator heads (+, -, *, /, ==, !=,
  <=, >=, <, >). Lone '-' is a SYMBOL; '-' followed by a digit at token start
  starts an INT or FLOAT. Edge cases like 'int->str', 'is-nil', 'str->int'
  are scanned as single atomic SYMBOL tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DELIMS = " \t\r\n();"
ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}

INT_RE = re.compile(r"-?[0-9]+\Z")
FLOAT_RE = re.compile(r"-?[0-9]+\.[0-9]+\Z")
OP_RE = re.compile(r"(\+|-|\*|/|==|!=|<=|>=|<|>)\Z")
# Identifiers start with letter/underscore or '-' (when not followed by digits),
# and can contain letters, digits, _, ?, >, -
SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_?>-]*\Z")


class LexError(Exception):
    """Fatal scanner error raised when tokenization cannot proceed."""

    def __init__(self, line: int, col: int, message: str) -> None:
        self.line = line
        self.col = col
        self.message = message
        super().__init__(f"line {line}, col {col}: {message}")


LexerError = LexError  # Canonical alias


@dataclass(frozen=True)
class Tok:
    """An atomic lexical token with exact 1-based source coordinates."""

    kind: str  # LPAREN, RPAREN, SYMBOL, INT, FLOAT, STRING, BOOL, NIL
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Tok({self.kind}, {self.value!r}, line={self.line}, col={self.col})"


Token = Tok  # Canonical alias


def classify(text: str, line: int, col: int) -> str:
    """Classify a non-delimiter text slice into its lexical category.

    Order matters: INT and FLOAT are checked before SYMBOL so negative numbers
    like '-5' are classified as INT rather than an operator expression.
    Lone '-' and symbols like 'int->str' fall through to SYMBOL.
    """
    if INT_RE.match(text):
        return "INT"
    if FLOAT_RE.match(text):
        return "FLOAT"
    if text in ("true", "false"):
        return "BOOL"
    if text == "nil":
        return "NIL"
    if OP_RE.match(text) or SYMBOL_RE.match(text):
        return "SYMBOL"
    raise LexError(line, col, f"invalid token {text!r}")


class Lexer:
    """Hand-written stateful scanner for Netelpro source text."""

    def __init__(self, src: str) -> None:
        self.src = src
        self.n = len(src)
        self.i = 0
        self.line = 1
        self.col = 1

    def _advance(self, ch: str) -> None:
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1

    def tokenize(self) -> list[Tok]:
        """Scan full source string into a list of Tok objects.

        Raises LexError with exact position on unclosed strings,
        illegal escapes, or unrecognized character sequences.
        """
        toks: list[Tok] = []
        src = self.src
        n = self.n

        while self.i < n:
            c = src[self.i]

            # 1. Whitespace
            if c in " \t\r\n":
                self._advance(c)
                self.i += 1
                continue

            # 2. Line comment: ; to end-of-line
            if c == ";":
                while self.i < n and src[self.i] != "\n":
                    self._advance(src[self.i])
                    self.i += 1
                continue

            # 3. Delimiters: ( and )
            if c in "()":
                kind = "LPAREN" if c == "(" else "RPAREN"
                toks.append(Tok(kind, c, self.line, self.col))
                self._advance(c)
                self.i += 1
                continue

            # 4. String literals: "..."
            if c == '"':
                sl, sc = self.line, self.col
                buf: list[str] = []
                self.i += 1
                self._advance(c)

                while True:
                    if self.i >= n:
                        raise LexError(sl, sc, "unterminated string literal")

                    ch = src[self.i]

                    # Multi-line strings not allowed in v0.1
                    if ch == "\n":
                        raise LexError(sl, sc, "unterminated string literal")

                    if ch == "\\":
                        esc_line, esc_col = self.line, self.col
                        if self.i + 1 >= n:
                            raise LexError(sl, sc, "unterminated string literal (dangling escape)")
                        esc = src[self.i + 1]
                        if esc not in ESCAPES:
                            raise LexError(esc_line, esc_col, f"invalid escape \\{esc}")
                        buf.append(ESCAPES[esc])
                        self._advance("\\")
                        self._advance(esc)
                        self.i += 2
                        continue

                    if ch == '"':
                        self._advance(ch)
                        self.i += 1
                        break

                    buf.append(ch)
                    self._advance(ch)
                    self.i += 1

                toks.append(Tok("STRING", "".join(buf), sl, sc))
                continue

            # 5. Symbols, numbers, booleans, nil
            sl, sc = self.line, self.col
            j = self.i
            buf2: list[str] = []
            while j < n and src[j] not in DELIMS:
                buf2.append(src[j])
                j += 1
            text = "".join(buf2)
            for ch in text:
                self._advance(ch)
            self.i = j
            kind = classify(text, sl, sc)
            toks.append(Tok(kind, text, sl, sc))

        return toks


def tokenize(src: str) -> list[Tok]:
    """Convenience functional interface for scanning source code."""
    return Lexer(src).tokenize()
