"""Netelpro AST nodes -- Phase 1 typed abstract syntax tree.

The prosecutor thesis:
Every AST node carries exact source coordinates (line, col) from the token stream
so any downstream phase (structural validation, typing, capability auditing,
or code generation) can report prosecutorial diagnostics with exact source
provenance.

Why literal nodes are separate subclasses (IntLit, FloatLit, StrLit, BoolLit, NilLit):
Netelpro v0.1 has a strictly typed data model (Int, Float, Str, Bool, List<T>).
In Python, 'bool' is a subclass of 'int' (isinstance(True, int) is True), which
creates insidious type-coercion bugs in compiler passes that inspect native values.
By modeling each literal as a distinct, dedicated AST subclass, AST passes and pattern
matching ('match node: case IntLit(v): ...') are guaranteed to be type-safe,
unambiguous, and auditable without relying on dynamic runtime value inspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Node:
    """Base class for all Netelpro AST nodes.

    Carries 1-based (line, col) coordinates. Coordinates are excluded from
    structural equality comparison so AST assertions in tests and transformations
    can verify semantic tree shapes directly while retaining access to exact
    source positions on every node.

    Fields are declared kw_only so they always sit at the END of every
    subclass __init__ signature (dataclass inheritance would otherwise put
    them FIRST and shift every positional argument, e.g. IntLit(42) would
    bind 42 to `line`). kw_only fields are moved after all positional fields
    by the dataclasses machinery, so IntLit(42, line=2, col=19) and
    Sym("x") both work as the parser and tests expect.
    """

    line: int = field(default=0, compare=False, kw_only=True)
    col: int = field(default=0, compare=False, kw_only=True)


@dataclass(frozen=True)
class Sym(Node):
    """An identifier symbol (variable, function name, or capability)."""

    name: str = ""

    def __repr__(self) -> str:
        return f"Sym({self.name!r})"


Symbol = Sym  # Canonical alias


# ---------------------------------------------------------------------------
# Literal Nodes (separate subclasses for strict type discrimination)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Literal(Node):
    """Abstract base class for all literal value nodes."""


@dataclass(frozen=True)
class IntLit(Literal):
    """Integer literal node, e.g. 42 or -7."""

    value: int = 0

    def __repr__(self) -> str:
        return f"IntLit({self.value})"


@dataclass(frozen=True)
class FloatLit(Literal):
    """Floating-point literal node, e.g. 3.14 or -0.5."""

    value: float = 0.0

    def __repr__(self) -> str:
        return f"FloatLit({self.value})"


@dataclass(frozen=True)
class StrLit(Literal):
    """String literal node with escapes already decoded."""

    value: str = ""

    def __repr__(self) -> str:
        return f"StrLit({self.value!r})"


@dataclass(frozen=True)
class BoolLit(Literal):
    """Boolean literal node: 'true' or 'false'."""

    value: bool = False

    def __repr__(self) -> str:
        return f"BoolLit({self.value})"


@dataclass(frozen=True)
class NilLit(Literal):
    """Nil literal node: 'nil' (the empty list)."""

    value: None = field(default=None, compare=False)

    def __repr__(self) -> str:
        return "NilLit()"


# ---------------------------------------------------------------------------
# Composite & Special Form Nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ListLit(Node):
    """List construction literal for '(list ...)', the only open-arity form."""

    items: list[Node] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.items, list):
            object.__setattr__(self, "items", list(self.items))


@dataclass(frozen=True)
class Def(Node):
    """Constant binding: '(def name value)'."""

    name: Sym = field(default_factory=lambda: Sym(""))
    value: Node = field(default_factory=Node)

    def __post_init__(self) -> None:
        if isinstance(self.name, str):
            object.__setattr__(self, "name", Sym(self.name, line=self.line, col=self.col))


@dataclass(frozen=True)
class Defn(Node):
    """Named function definition: '(defn name (params...) body)'."""

    name: Sym = field(default_factory=lambda: Sym(""))
    params: list[Sym] = field(default_factory=list)
    body: Node = field(default_factory=Node)

    def __post_init__(self) -> None:
        if isinstance(self.name, str):
            object.__setattr__(self, "name", Sym(self.name, line=self.line, col=self.col))
        if any(isinstance(p, str) for p in self.params):
            object.__setattr__(
                self,
                "params",
                [Sym(p, line=self.line, col=self.col) if isinstance(p, str) else p for p in self.params],
            )
        elif not isinstance(self.params, list):
            object.__setattr__(self, "params", list(self.params))


@dataclass(frozen=True)
class Fn(Node):
    """Anonymous function: '(fn (params...) body)'."""

    params: list[Sym] = field(default_factory=list)
    body: Node = field(default_factory=Node)

    def __post_init__(self) -> None:
        if any(isinstance(p, str) for p in self.params):
            object.__setattr__(
                self,
                "params",
                [Sym(p, line=self.line, col=self.col) if isinstance(p, str) else p for p in self.params],
            )
        elif not isinstance(self.params, list):
            object.__setattr__(self, "params", list(self.params))


@dataclass(frozen=True)
class Let(Node):
    """Single lexical binding: '(let name value body)'."""

    name: Sym = field(default_factory=lambda: Sym(""))
    value: Node = field(default_factory=Node)
    body: Node = field(default_factory=Node)

    def __post_init__(self) -> None:
        if isinstance(self.name, str):
            object.__setattr__(self, "name", Sym(self.name, line=self.line, col=self.col))


@dataclass(frozen=True)
class If(Node):
    """Conditional branching: '(if cond then else)'. Both branches mandatory."""

    cond: Node = field(default_factory=Node)
    then: Node = field(default_factory=Node)
    else_: Node = field(default_factory=Node)


@dataclass(frozen=True)
class And(Node):
    """Short-circuit logical AND: '(and l r)'."""

    l: Node = field(default_factory=Node)
    r: Node = field(default_factory=Node)


@dataclass(frozen=True)
class Or(Node):
    """Short-circuit logical OR: '(or l r)'."""

    l: Node = field(default_factory=Node)
    r: Node = field(default_factory=Node)


@dataclass(frozen=True)
class Sorry(Node):
    """Prosecutor typed hole placeholder: '(sorry \"reason\")'."""

    reason: StrLit = field(default_factory=lambda: StrLit(""))

    def __post_init__(self) -> None:
        if isinstance(self.reason, str):
            object.__setattr__(self, "reason", StrLit(self.reason, line=self.line, col=self.col))


@dataclass(frozen=True)
class Grant(Node):
    """Top-level capability declaration: '(grant cap1 cap2 ...)'."""

    caps: list[Sym] = field(default_factory=list)

    def __post_init__(self) -> None:
        if any(isinstance(c, str) for c in self.caps):
            object.__setattr__(
                self,
                "caps",
                [Sym(c, line=self.line, col=self.col) if isinstance(c, str) else c for c in self.caps],
            )
        elif not isinstance(self.caps, list):
            object.__setattr__(self, "caps", list(self.caps))


@dataclass(frozen=True)
class Call(Node):
    """Primitive application or user function invocation: '(head arg1 arg2 ...)'."""

    head: str = ""
    args: list[Node] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.head, Sym):
            object.__setattr__(self, "head", self.head.name)
        if not isinstance(self.args, list):
            object.__setattr__(self, "args", list(self.args))


@dataclass(frozen=True)
class Program(Node):
    """Top-level AST root containing a sequence of forms."""

    forms: list[Node] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.forms, list):
            object.__setattr__(self, "forms", list(self.forms))
