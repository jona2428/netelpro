"""Netelpro evaluator -- Phase 2 tree-walking interpreter with TCO.

Semantic contract:
1. Value model:
   - Int -> Python int
   - Float -> Python float
   - Str -> Python str
   - Bool -> Python bool
   - List -> StrayList (tuple-backed frozen class; nil -> empty StrayList)
   - Python trap avoided: bool is a subclass of int, so always discriminated via `type(x) is bool`.
2. Lexical scope chain: Environment with parent pointers. Top-level def/defn bind into global scope.
3. Strict booleans: if/and/or conditions must be Bool, no truthiness coercions.
4. and/or short-circuit; result always Bool.
5. Exact call arity check against parameter list.
6. Arithmetic: +,-,* int*int->int, any float->float; / always promotes to float; zero divisor errors; quot/rem Int-only.
7. Comparisons: ordering numeric-only; cross-type equality returns False (not error).
8. Lists: StrayList operations cons, head, tail, is-nil, len, nth.
9. Strings: str-cat, str-len, int->str, str->int, int->float.
10. sorry: raises StrayHoleError.
11. grant: records declared capabilities without error.
12. print: human-readable stdout, returns nil.
13. TCO: tail call optimization for function calls in tail positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence
import re
import sys

from netelpro.ast_nodes import (
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
    NilLit,
    Node,
    Or,
    Program,
    Sorry,
    StrLit,
    Sym,
)
from netelpro.parser import parse, ParseResult


# ---------------------------------------------------------------------------
# Error Hierarchy
# ---------------------------------------------------------------------------


class StrayError(Exception):
    """Base exception for all Netelpro compiler and runtime errors."""

    def __init__(self, message: str = "", errors: Optional[list[Any]] = None) -> None:
        self.message = message
        self.errors = errors if errors is not None else []
        super().__init__(message)


class StrayRuntimeError(StrayError):
    """Prosecutorial runtime diagnostic record with exact source coordinates."""

    def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
        self.message = message
        self.line = line
        self.col = col
        super().__init__(f"runtime error at line {line}, col {col}: {message}")


class StrayHoleError(StrayError):
    """Exception raised when a 'sorry' hole placeholder is evaluated."""

    def __init__(self, reason: str, line: int = 0, col: int = 0) -> None:
        self.reason = reason
        self.line = line
        self.col = col
        super().__init__(f"runtime error at line {line}, col {col}: {reason}")


# ---------------------------------------------------------------------------
# StrayList Value Model
# ---------------------------------------------------------------------------


class StrayList:
    """Frozen tuple-backed list representation for Netelpro values."""

    __slots__ = ("_items",)
    _items: tuple[Any, ...]

    def __init__(self, *args: Any, items: Any = None) -> None:
        if items is not None:
            raw = items
        elif len(args) == 1:
            first = args[0]
            if isinstance(first, (list, tuple, StrayList)):
                raw = first
            else:
                raw = (first,)
        elif len(args) > 1:
            raw = args
        else:
            raw = ()

        if isinstance(raw, StrayList):
            object.__setattr__(self, "_items", raw._items)
        elif isinstance(raw, tuple):
            object.__setattr__(self, "_items", raw)
        else:
            object.__setattr__(self, "_items", tuple(raw))

    @property
    def items(self) -> tuple[Any, ...]:
        return self._items

    @property
    def elements(self) -> tuple[Any, ...]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index: Any) -> Any:
        return self._items[index]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, StrayList):
            return False
        return self._items == other._items

    def __hash__(self) -> int:
        return hash(self._items)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("StrayList is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("StrayList is immutable")

    def __repr__(self) -> str:
        return f"StrayList({list(self._items)!r})"

    def __str__(self) -> str:
        if not self._items:
            return "nil"
        return f"({' '.join(format_value(x) for x in self._items)})"


NIL = StrayList()


def is_nil(v: Any) -> bool:
    """Check if value is nil (the empty StrayList)."""
    return isinstance(v, StrayList) and len(v) == 0


def format_value(val: Any) -> str:
    """Format Netelpro value for human-readable output."""
    if type(val) is bool:
        return "true" if val else "false"
    if isinstance(val, StrayList):
        if len(val) == 0:
            return "nil"
        return f"({' '.join(format_value(x) for x in val._items)})"
    if isinstance(val, Closure):
        name_part = f" {val.name}" if val.name else ""
        return f"<fn{name_part}>"
    return str(val)


# ---------------------------------------------------------------------------
# Scoping & Environment
# ---------------------------------------------------------------------------


class Environment:
    """Lexical environment for Netelpro scoping."""

    def __init__(
        self,
        parent: Optional[Environment] = None,
        bindings: Optional[dict[str, Any]] = None,
    ) -> None:
        self.parent = parent
        self.bindings: dict[str, Any] = bindings if bindings is not None else {}

    def get(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise KeyError(name)

    def contains(self, name: str) -> bool:
        if name in self.bindings:
            return True
        if self.parent is not None:
            return self.parent.contains(name)
        return False

    def define(self, name: str, value: Any) -> None:
        self.bindings[name] = value

    def set_global(self, name: str, value: Any) -> None:
        env = self
        while env.parent is not None:
            env = env.parent
        env.bindings[name] = value

    def extend(self, bindings: dict[str, Any]) -> Environment:
        return Environment(parent=self, bindings=dict(bindings))


@dataclass
class Closure:
    """Represents a user-defined function closure."""

    params: list[Sym]
    body: Node
    env: Environment
    name: Optional[str] = None
    line: int = 0
    col: int = 0


# ---------------------------------------------------------------------------
# Reserved Heads & Constants
# ---------------------------------------------------------------------------

RESERVED_HEADS = {
    # Special forms
    "def",
    "defn",
    "let",
    "fn",
    "if",
    "and",
    "or",
    "sorry",
    "grant",
    # Primitives
    "+",
    "-",
    "*",
    "/",
    "quot",
    "rem",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "not",
    "list",
    "cons",
    "head",
    "tail",
    "is-nil",
    "len",
    "nth",
    "str-cat",
    "str-len",
    "int->str",
    "str->int",
    "int->float",
    "prefix?",
    "print",
}

PRIMITIVES = {
    "+",
    "-",
    "*",
    "/",
    "quot",
    "rem",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "not",
    "list",
    "cons",
    "head",
    "tail",
    "is-nil",
    "len",
    "nth",
    "str-cat",
    "str-len",
    "int->str",
    "str->int",
    "int->float",
    "prefix?",
    "print",
}


# ---------------------------------------------------------------------------
# Equality & Type Discriminations
# ---------------------------------------------------------------------------


def _type_class(val: Any) -> str:
    if type(val) is bool:
        return "bool"
    if isinstance(val, (int, float)):
        return "number"
    if type(val) is str:
        return "str"
    if isinstance(val, StrayList):
        return "list"
    return "other"


def stray_equals(a: Any, b: Any) -> bool:
    class_a = _type_class(a)
    class_b = _type_class(b)
    if class_a != class_b or class_a == "other":
        return False
    if class_a == "bool":
        return a is b
    if class_a == "number":
        return a == b
    if class_a == "str":
        return a == b
    if class_a == "list":
        if len(a) != len(b):
            return False
        return all(stray_equals(x, y) for x, y in zip(a._items, b._items))
    return False


# ---------------------------------------------------------------------------
# Primitives Implementation
# ---------------------------------------------------------------------------


def _exec_primitive(head: str, args: list[Any], node: Call, capabilities: Optional[set[str]] = None) -> Any:
    line, col = node.line, node.col

    if head in ("+", "-", "*", "/", "quot", "rem", "==", "!=", "<", "<=", ">", ">=", "cons", "nth", "str-cat"):
        if len(args) != 2:
            raise StrayRuntimeError(f"'{head}' expects 2 arguments, got {len(args)}", line, col)
    elif head in ("not", "head", "tail", "is-nil", "len", "str-len", "int->str", "str->int", "int->float", "print"):
        if len(args) != 1:
            raise StrayRuntimeError(f"'{head}' expects 1 argument, got {len(args)}", line, col)
    elif head == "prefix?":
        if len(args) != 2:
            raise StrayRuntimeError(f"'prefix?' expects 2 arguments, got {len(args)}", line, col)
    elif head == "list":
        return StrayList(args)

    if head in ("+", "-", "*"):
        a, b = args
        if type(a) is bool or type(b) is bool or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise StrayRuntimeError(f"'{head}' operands must be Int or Float", line, col)
        if head == "+":
            res = a + b
        elif head == "-":
            res = a - b
        else:
            res = a * b
        if type(a) is int and type(b) is int:
            return res
        return float(res)

    if head == "/":
        a, b = args
        if type(a) is bool or type(b) is bool or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise StrayRuntimeError("'/' operands must be Int or Float", line, col)
        if b == 0:
            raise StrayRuntimeError("division by zero", line, col)
        return float(a) / float(b)

    if head == "quot":
        a, b = args
        if type(a) is not int or type(b) is not int:
            raise StrayRuntimeError("'quot' operands must be Int", line, col)
        if b == 0:
            raise StrayRuntimeError("'quot' division by zero", line, col)
        return int(a / b)

    if head == "rem":
        a, b = args
        if type(a) is not int or type(b) is not int:
            raise StrayRuntimeError("'rem' operands must be Int", line, col)
        if b == 0:
            raise StrayRuntimeError("'rem' division by zero", line, col)
        q = int(a / b)
        return a - q * b

    if head == "==":
        return stray_equals(args[0], args[1])

    if head == "!=":
        return not stray_equals(args[0], args[1])

    if head in ("<", "<=", ">", ">="):
        a, b = args
        if type(a) is bool or type(b) is bool or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise StrayRuntimeError(f"'{head}' cannot compare non-numeric types", line, col)
        if head == "<":
            return a < b
        if head == "<=":
            return a <= b
        if head == ">":
            return a > b
        if head == ">=":
            return a >= b

    if head == "not":
        a = args[0]
        if type(a) is not bool:
            raise StrayRuntimeError("'not' operand must be Bool", line, col)
        return not a

    if head == "cons":
        x, xs = args
        if not isinstance(xs, StrayList):
            raise StrayRuntimeError("'cons' second operand must be List", line, col)
        return StrayList((x,) + xs._items)

    if head == "head":
        xs = args[0]
        if not isinstance(xs, StrayList):
            raise StrayRuntimeError("'head' operand must be List", line, col)
        if len(xs) == 0:
            raise StrayRuntimeError("'head' called on empty list", line, col)
        return xs._items[0]

    if head == "tail":
        xs = args[0]
        if not isinstance(xs, StrayList):
            raise StrayRuntimeError("'tail' operand must be List", line, col)
        if len(xs) == 0:
            raise StrayRuntimeError("'tail' called on empty list", line, col)
        return StrayList(xs._items[1:])

    if head == "is-nil":
        xs = args[0]
        if not isinstance(xs, StrayList):
            raise StrayRuntimeError("'is-nil' operand must be List", line, col)
        return len(xs) == 0

    if head == "len":
        xs = args[0]
        if not isinstance(xs, StrayList):
            raise StrayRuntimeError("'len' operand must be List", line, col)
        return len(xs)

    if head == "nth":
        xs, i = args
        if not isinstance(xs, StrayList):
            raise StrayRuntimeError("'nth' first operand must be List", line, col)
        if type(i) is not int:
            raise StrayRuntimeError("'nth' index must be Int", line, col)
        if i < 0 or i >= len(xs):
            raise StrayRuntimeError(f"'nth' index out of range: {i}", line, col)
        return xs._items[i]

    if head == "str-cat":
        a, b = args
        if type(a) is not str or type(b) is not str:
            raise StrayRuntimeError("'str-cat' operands must be Str", line, col)
        return a + b

    if head == "str-len":
        s = args[0]
        if type(s) is not str:
            raise StrayRuntimeError("'str-len' operand must be Str", line, col)
        return len(s)

    if head == "int->str":
        n = args[0]
        if type(n) is not int:
            raise StrayRuntimeError("'int->str' operand must be Int", line, col)
        return str(n)

    if head == "str->int":
        s = args[0]
        if type(s) is not str:
            raise StrayRuntimeError("'str->int' operand must be Str", line, col)
        if not re.fullmatch(r"-?[0-9]+", s):
            raise StrayRuntimeError(f"invalid integer string: {s!r}", line, col)
        try:
            return int(s)
        except ValueError:
            raise StrayRuntimeError(f"invalid integer string: {s!r}", line, col)

    if head == "int->float":
        n = args[0]
        if type(n) is not int:
            raise StrayRuntimeError("'int->float' operand must be Int", line, col)
        return float(n)

    if head == "prefix?":
        text, prefix = args
        if type(text) is not str or type(prefix) is not str:
            raise StrayRuntimeError("'prefix?' operands must be Str", line, col)
        return text.startswith(prefix)

    if head == "print":
        caps = capabilities if capabilities is not None else set()
        if "io" not in caps:
            raise StrayRuntimeError("capability 'io' required by 'print' but not granted", line, col)
        x = args[0]
        print(format_value(x))
        return NIL

    raise StrayRuntimeError(f"unknown primitive '{head}'", line, col)


# ---------------------------------------------------------------------------
# Evaluator Loop (Iterative TCO)
# ---------------------------------------------------------------------------


def _validate_params(params: Sequence[Sym | str], line: int, col: int) -> list[str]:
    param_names: list[str] = []
    for p in params:
        name = p.name if isinstance(p, Sym) else str(p)
        if name in RESERVED_HEADS:
            p_line = getattr(p, "line", 0) or line
            p_col = getattr(p, "col", 0) or col
            raise StrayRuntimeError(f"parameter '{name}' shadows reserved head", p_line, p_col)
        param_names.append(name)
    return param_names


def eval_loop(node: Node, env: Environment, capabilities: Optional[set[str]] = None) -> Any:
    """Iterative evaluation loop supporting tail-call optimization."""
    curr_node: Node = node
    curr_env: Environment = env
    must_be_bool_stack: list[tuple[int, int]] = []

    while True:
        if isinstance(curr_node, IntLit):
            result = curr_node.value
        elif isinstance(curr_node, FloatLit):
            result = curr_node.value
        elif isinstance(curr_node, StrLit):
            result = curr_node.value
        elif isinstance(curr_node, BoolLit):
            result = curr_node.value
        elif isinstance(curr_node, NilLit):
            result = NIL
        elif isinstance(curr_node, Sym):
            try:
                result = curr_env.get(curr_node.name)
            except KeyError:
                raise StrayRuntimeError(f"unbound symbol '{curr_node.name}'", curr_node.line, curr_node.col)
        elif isinstance(curr_node, ListLit):
            items = [eval_loop(item, curr_env, capabilities) for item in curr_node.items]
            result = StrayList(items)
        elif isinstance(curr_node, Fn):
            _validate_params(curr_node.params, curr_node.line, curr_node.col)
            result = Closure(
                params=list(curr_node.params),
                body=curr_node.body,
                env=curr_env,
                name=None,
                line=curr_node.line,
                col=curr_node.col,
            )
        elif isinstance(curr_node, Sorry):
            reason_str = curr_node.reason.value if isinstance(curr_node.reason, StrLit) else str(curr_node.reason)
            raise StrayHoleError(reason_str, curr_node.line, curr_node.col)
        elif isinstance(curr_node, Grant):
            result = NIL
        elif isinstance(curr_node, Def):
            val = eval_loop(curr_node.value, curr_env, capabilities)
            curr_env.set_global(curr_node.name.name, val)
            result = NIL
        elif isinstance(curr_node, Defn):
            _validate_params(curr_node.params, curr_node.line, curr_node.col)
            closure = Closure(
                params=list(curr_node.params),
                body=curr_node.body,
                env=curr_env,
                name=curr_node.name.name,
                line=curr_node.line,
                col=curr_node.col,
            )
            curr_env.set_global(curr_node.name.name, closure)
            result = NIL
        elif isinstance(curr_node, If):
            cond_val = eval_loop(curr_node.cond, curr_env, capabilities)
            if type(cond_val) is not bool:
                c_line = getattr(curr_node.cond, "line", 0) or curr_node.line
                c_col = getattr(curr_node.cond, "col", 0) or curr_node.col
                raise StrayRuntimeError("condition is not Bool", c_line, c_col)
            if cond_val:
                curr_node = curr_node.then
            else:
                curr_node = curr_node.else_
            continue
        elif isinstance(curr_node, Let):
            val = eval_loop(curr_node.value, curr_env, capabilities)
            curr_env = curr_env.extend({curr_node.name.name: val})
            curr_node = curr_node.body
            continue
        elif isinstance(curr_node, And):
            l_val = eval_loop(curr_node.l, curr_env, capabilities)
            if type(l_val) is not bool:
                l_line = getattr(curr_node.l, "line", 0) or curr_node.line
                l_col = getattr(curr_node.l, "col", 0) or curr_node.col
                raise StrayRuntimeError("condition is not Bool", l_line, l_col)
            if not l_val:
                result = False
            else:
                curr_node = curr_node.r
                r_line = getattr(curr_node, "line", 0) or curr_node.line
                r_col = getattr(curr_node, "col", 0) or curr_node.col
                must_be_bool_stack.append((r_line, r_col))
                continue
        elif isinstance(curr_node, Or):
            l_val = eval_loop(curr_node.l, curr_env, capabilities)
            if type(l_val) is not bool:
                l_line = getattr(curr_node.l, "line", 0) or curr_node.line
                l_col = getattr(curr_node.l, "col", 0) or curr_node.col
                raise StrayRuntimeError("condition is not Bool", l_line, l_col)
            if l_val:
                result = True
            else:
                curr_node = curr_node.r
                r_line = getattr(curr_node, "line", 0) or curr_node.line
                r_col = getattr(curr_node, "col", 0) or curr_node.col
                must_be_bool_stack.append((r_line, r_col))
                continue
        elif isinstance(curr_node, Call):
            head = curr_node.head
            if head in PRIMITIVES:
                arg_vals = [eval_loop(a, curr_env, capabilities) for a in curr_node.args]
                result = _exec_primitive(head, arg_vals, curr_node, capabilities)
            else:
                try:
                    fn_val = curr_env.get(head)
                except KeyError:
                    raise StrayRuntimeError(f"undefined function '{head}'", curr_node.line, curr_node.col)
                if not isinstance(fn_val, Closure):
                    raise StrayRuntimeError(f"'{head}' is not a function", curr_node.line, curr_node.col)
                if len(curr_node.args) != len(fn_val.params):
                    raise StrayRuntimeError(
                        f"'{head}' expects {len(fn_val.params)} argument(s), got {len(curr_node.args)}",
                        curr_node.line,
                        curr_node.col,
                    )
                arg_vals = [eval_loop(a, curr_env, capabilities) for a in curr_node.args]
                param_bindings = {
                    (p.name if isinstance(p, Sym) else str(p)): v
                    for p, v in zip(fn_val.params, arg_vals)
                }
                curr_env = fn_val.env.extend(param_bindings)
                curr_node = fn_val.body
                continue
        elif isinstance(curr_node, Program):
            result = evaluate(curr_node, curr_env, capabilities=capabilities)
        else:
            raise StrayRuntimeError(
                f"unsupported AST node: {type(curr_node).__name__}",
                getattr(curr_node, "line", 0),
                getattr(curr_node, "col", 0),
            )

        # Check bool requirement if coming from And/Or right-side tail
        while must_be_bool_stack:
            pos_line, pos_col = must_be_bool_stack.pop()
            if type(result) is not bool:
                raise StrayRuntimeError("condition is not Bool", pos_line, pos_col)

        return result


def eval_node(node: Node, env: Environment, capabilities: Optional[set[str]] = None) -> Any:
    """Evaluate an individual AST node within an environment."""
    return eval_loop(node, env, capabilities)


# ---------------------------------------------------------------------------
# Evaluator & Public API
# ---------------------------------------------------------------------------


class Evaluator:
    """Stateful evaluator instance tracking capabilities and global environment."""

    def __init__(self, env: Optional[Environment] = None, capabilities: Optional[set[str]] = None) -> None:
        self.env = env if env is not None else Environment()
        self.capabilities: set[str] = set(capabilities) if capabilities else set()

    def evaluate(self, program: Program) -> Any:
        last_val: Any = NIL
        has_expr: bool = False

        for form in program.forms:
            if isinstance(form, Grant):
                for cap in form.caps:
                    name = cap.name if isinstance(cap, Sym) else str(cap)
                    self.capabilities.add(name)
            elif isinstance(form, Def):
                val = eval_loop(form.value, self.env, self.capabilities)
                self.env.set_global(form.name.name, val)
            elif isinstance(form, Defn):
                _validate_params(form.params, form.line, form.col)
                closure = Closure(
                    params=list(form.params),
                    body=form.body,
                    env=self.env,
                    name=form.name.name,
                    line=form.line,
                    col=form.col,
                )
                self.env.set_global(form.name.name, closure)
            else:
                last_val = eval_loop(form, self.env, self.capabilities)
                has_expr = True

        return last_val if has_expr else NIL


def _guard_recursion(program: Program, env: Optional[Environment], capabilities: Optional[set[str]] = None) -> Any:
    """Translate raw Python RecursionError into a prosecutorial StrayRuntimeError.

    NON-tail recursion depth is bounded by the host interpreter stack; exceeding
    it is a resource limit of the runtime, not a raw Python crash. The prosecutor
    reports it instead of leaking a traceback.
    """
    try:
        return Evaluator(env=env, capabilities=capabilities).evaluate(program)
    except RecursionError:
        raise StrayRuntimeError(
            "non-tail recursion exceeded host stack depth (no TCO on this call path)",
            1,
            1,
        ) from None


def evaluate(program: Program, env: Optional[Environment] = None, capabilities: Optional[set[str]] = None) -> Any:
    """Evaluate a Program AST and return the value of the last evaluated expression form (nil if none)."""
    return _guard_recursion(program, env, capabilities)


def run_source(src: str, env: Optional[Environment] = None) -> Any:
    """Parse and evaluate Netelpro source text.

    Raises StrayError carrying all parse diagnostic messages if parsing fails.
    Otherwise returns the value of the last evaluated expression form.
    """
    parse_result = parse(src)
    if not parse_result.ok:
        all_msgs = "\n".join(str(e) for e in parse_result.errors)
        raise StrayError(f"parse error(s):\n{all_msgs}", errors=parse_result.errors)
    return evaluate(parse_result.program, env=env)
