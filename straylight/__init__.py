"""Straylight language package -- Phase 1 compiler frontend.

The compiler-as-prosecutor:
Deterministic lexical analysis, mechanical arity verification against spec/arity_table.json,
and typed, frozen AST construction with exact source coordinates.
"""
from __future__ import annotations

from straylight.ast_nodes import (
    And,
    BoolLit,
    Call,
    Def,
    Defn,
    FloatLit,
    Fn,
    Grant,
    If,
    IntLit,
    Let,
    ListLit,
    Literal,
    NilLit,
    Node,
    Or,
    Program,
    Sorry,
    StrLit,
    Sym,
    Symbol,
)
from straylight.lexer import (
    LexError,
    Lexer,
    LexerError,
    Tok,
    Token,
    tokenize,
)
from straylight.parser import (
    ParseError,
    ParseResult,
    Parser,
    parse,
)
from straylight.evaluator import (
    StrayError,
    StrayHoleError,
    StrayList,
    StrayRuntimeError,
    Closure,
    Environment,
    eval_node,
    evaluate,
    format_value,
    is_nil,
    run_source,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Lexer exports
    "Lexer",
    "LexError",
    "LexerError",
    "Tok",
    "Token",
    "tokenize",
    # Parser exports
    "Parser",
    "ParseError",
    "ParseResult",
    "parse",
    # AST Node exports
    "Node",
    "Sym",
    "Symbol",
    "Literal",
    "IntLit",
    "FloatLit",
    "StrLit",
    "BoolLit",
    "NilLit",
    "ListLit",
    "Def",
    "Defn",
    "Fn",
    "Let",
    "If",
    "And",
    "Or",
    "Sorry",
    "Grant",
    "Call",
    "Program",
    # Evaluator exports
    "StrayError",
    "StrayRuntimeError",
    "StrayHoleError",
    "StrayList",
    "evaluate",
    "run_source",
    # Evaluator internals (public API for tooling)
    "Closure",
    "Environment",
    "eval_node",
    "format_value",
    "is_nil",
]
