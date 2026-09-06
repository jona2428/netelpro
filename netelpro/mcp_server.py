"""Netelpro Model Context Protocol (MCP) Server.

Line-delimited JSON-RPC 2.0 stdio server for the Netelpro programming language.
Transport protocol: stdio, line-delimited JSON (Content-Length NOT required).
Each incoming line from sys.stdin is parsed as a JSON-RPC 2.0 message, and each
response is written as a single newline-terminated JSON line to sys.stdout and flushed.

Tools exposed:
- netelpro_compile: Static prosecutorial validation (parse, caps, holes, codegen).
- netelpro_eval: Subprocess execution with resource limits and stdout capture.
- netelpro_verify: Differential parity verification between native JIT and interpreter.
- netelpro_spec: Static language specification knowledge, forms, and capabilities.

Limits (module-level overridable attributes):
- MAX_SOURCE_BYTES: 65536
- PARSE_DEPTH_BUDGET: 64
- EVAL_TIMEOUT_S: 3.0
- MAX_CASES: 100
- RESULT_STRING_CAP: 65536
- MAX_LINE_BYTES: 8 * MAX_SOURCE_BYTES (stdio frame cap, fail-closed)
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

from netelpro.caps import check_capabilities, collect_grants
from netelpro.evaluator import (
    Closure,
    StrayError,
    StrayHoleError,
    StrayList,
    StrayRuntimeError,
    evaluate,
    format_value,
    is_nil,
)
from netelpro.holes import check_holes
from netelpro.parser import parse

# ---------------------------------------------------------------------------
# Fail-closed limits (overridable for tests via module attributes)
# ---------------------------------------------------------------------------

MAX_SOURCE_BYTES: int = 65536
PARSE_DEPTH_BUDGET: int = 64
EVAL_TIMEOUT_S: float = 3.0
MAX_CASES: int = 100
RESULT_STRING_CAP: int = 65536
MAX_LINE_BYTES: int = 8 * MAX_SOURCE_BYTES
VERIFY_STRING_ARG_CAP: int = 4096


def _kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Best-effort process-tree kill: Windows taskkill /T /F, then kill() fallback."""
    with contextlib.suppress(Exception):
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    with contextlib.suppress(Exception):
        proc.kill()


def _close_pipes(proc: subprocess.Popen[Any]) -> None:
    """Release child pipe handles to avoid leaks after a failed drain."""
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            with contextlib.suppress(Exception):
                pipe.close()


def _validated_out_path(argv: Sequence[str]) -> str | None:
    """Accept --out only if it resolves under the OS temp dir (fail-closed)."""
    if "--out" not in argv:
        return None
    idx = argv.index("--out")
    if idx + 1 >= len(argv):
        return None
    path = argv[idx + 1]
    try:
        temp_root = os.path.realpath(tempfile.gettempdir())
        candidate = os.path.realpath(path)
        if os.path.commonpath([candidate, temp_root]) != temp_root:
            return None
    except Exception:
        return None
    return path


def _validate_verify_cases(cases: list[Any]) -> list[dict[str, Any]]:
    """Type/size validation for netelpro_verify cases (fail-closed, no coercion)."""
    errors: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        if isinstance(case, dict):
            c_args: Any = case.get("args", [])
            c_exp: Any = case.get("expected", True)
        elif isinstance(case, (list, tuple)) and len(case) >= 2:
            c_args, c_exp = case[0], case[1]
        else:
            errors.append(
                {"phase": "runtime", "line": 0, "col": 0, "message": f"case {i}: expected args/expected object"}
            )
            continue
        if not isinstance(c_args, (list, tuple)):
            errors.append(
                {"phase": "runtime", "line": 0, "col": 0, "message": f"case {i}: 'args' must be a list"}
            )
            continue
        for j, arg in enumerate(c_args):
            if isinstance(arg, (bool, int)):
                continue
            if isinstance(arg, str):
                if len(arg) > VERIFY_STRING_ARG_CAP:
                    errors.append(
                        {
                            "phase": "limit",
                            "line": 0,
                            "col": 0,
                            "message": f"case {i} arg {j}: string arg exceeds {VERIFY_STRING_ARG_CAP} chars",
                        }
                    )
                continue
            errors.append(
                {
                    "phase": "runtime",
                    "line": 0,
                    "col": 0,
                    "message": f"case {i} arg {j}: arg must be int, bool or str",
                }
            )
        if not isinstance(c_exp, bool):
            errors.append(
                {"phase": "runtime", "line": 0, "col": 0, "message": f"case {i}: 'expected' must be a bool"}
            )
    return errors

# ---------------------------------------------------------------------------
# Static Knowledge Database for Netelpro Spec
# ---------------------------------------------------------------------------

SPECIAL_FORMS: dict[str, dict[str, Any]] = {
    "def": {
        "arity": [2, 2],
        "sig": "(def name expr)",
        "scope": "top-level only",
        "desc": "Constant binding in global scope.",
    },
    "defn": {
        "arity": [3, 3],
        "sig": "(defn name (params...) body)",
        "scope": "top-level only",
        "desc": "Named function definition; body is exactly one expression; supports recursive and mutual calls.",
    },
    "let": {
        "arity": [3, 3],
        "sig": "(let name expr body)",
        "scope": "lexical",
        "desc": "Lexical binding of a single symbol within body; chain lets for multiple bindings.",
    },
    "fn": {
        "arity": [2, 2],
        "sig": "(fn (params...) body)",
        "scope": "lexical",
        "desc": "Anonymous function closure; body is exactly one expression.",
    },
    "if": {
        "arity": [3, 3],
        "sig": "(if cond then else)",
        "desc": "Conditional branching; strict boolean condition (i1); both then and else branches mandatory.",
    },
    "and": {
        "arity": [2, 2],
        "sig": "(and a b)",
        "eval": "short-circuit",
        "desc": "Short-circuit boolean AND; strict boolean operands; result is always Bool.",
    },
    "or": {
        "arity": [2, 2],
        "sig": "(or a b)",
        "eval": "short-circuit",
        "desc": "Short-circuit boolean OR; strict boolean operands; result is always Bool.",
    },
    "sorry": {
        "arity": [1, 1],
        "sig": "(sorry \"reason\")",
        "phase": "grammar reserved, enforced in Phase 4",
        "desc": "Typed hole placeholder; reason must be a string literal; compiles clean, collected in hole manifest, raises StrayHoleError if executed.",
    },
    "grant": {
        "arity": [1, None],
        "sig": "(grant cap...)",
        "scope": "top-level only",
        "phase": "grammar reserved, enforced in Phase 3",
        "desc": "Top-level capability declaration; grants effects (e.g. io) file-wide.",
    },
}

PRIMITIVES: dict[str, dict[str, Any]] = {
    "+": {"arity": [2, 2], "sig": "(+ a b)", "desc": "Addition: Int+Int->Int, Float+Float->Float."},
    "-": {"arity": [2, 2], "sig": "(- a b)", "desc": "Subtraction: Int-Int->Int, Float-Float->Float."},
    "*": {"arity": [2, 2], "sig": "(* a b)", "desc": "Multiplication: Int*Int->Int, Float*Float->Float."},
    "/": {
        "arity": [2, 2],
        "sig": "(/ a b)",
        "semantics": "Int/Int -> Float (always promotes); requires nonzero divisor.",
        "desc": "Division returning Float.",
    },
    "quot": {
        "arity": [2, 2],
        "sig": "(quot a b)",
        "semantics": "integer division truncated toward zero",
        "desc": "Integer quotient.",
    },
    "rem": {
        "arity": [2, 2],
        "sig": "(rem a b)",
        "semantics": "remainder of quot",
        "desc": "Integer remainder.",
    },
    "==": {"arity": [2, 2], "sig": "(== a b)", "desc": "Equality comparison."},
    "!=": {"arity": [2, 2], "sig": "(!= a b)", "desc": "Inequality comparison."},
    "<": {"arity": [2, 2], "sig": "(< a b)", "desc": "Numeric less-than ordering."},
    "<=": {"arity": [2, 2], "sig": "(<= a b)", "desc": "Numeric less-than-or-equal ordering."},
    ">": {"arity": [2, 2], "sig": "(> a b)", "desc": "Numeric greater-than ordering."},
    ">=": {"arity": [2, 2], "sig": "(>= a b)", "desc": "Numeric greater-than-or-equal ordering."},
    "not": {"arity": [1, 1], "sig": "(not a)", "desc": "Boolean negation."},
    "list": {
        "arity": [0, None],
        "sig": "(list a b c ...)",
        "desc": "List construction; only open-arity form in Netelpro, closed by ')'.",
    },
    "cons": {"arity": [2, 2], "sig": "(cons x xs)", "desc": "Prepend element x to list xs."},
    "head": {"arity": [1, 1], "sig": "(head xs)", "desc": "Return the first element of list xs."},
    "tail": {"arity": [1, 1], "sig": "(tail xs)", "desc": "Return list xs without its first element."},
    "is-nil": {"arity": [1, 1], "sig": "(is-nil xs)", "desc": "Check if value is nil (the empty list)."},
    "len": {"arity": [1, 1], "sig": "(len xs)", "desc": "Return the number of elements in list xs."},
    "nth": {"arity": [2, 2], "sig": "(nth xs i)", "desc": "Return the 0-indexed element at index i in list xs."},
    "str-cat": {"arity": [2, 2], "sig": "(str-cat a b)", "desc": "Concatenate two strings."},
    "str-len": {"arity": [1, 1], "sig": "(str-len s)", "desc": "Return the character count of string s."},
    "int->str": {"arity": [1, 1], "sig": "(int->str n)", "desc": "Format integer as string."},
    "str->int": {"arity": [1, 1], "sig": "(str->int s)", "desc": "Parse string as integer."},
    "int->float": {"arity": [1, 1], "sig": "(int->float n)", "desc": "Convert integer to floating-point number."},
    "prefix?": {"arity": [2, 2], "sig": "(prefix? text prefix)", "desc": "Test whether text starts with prefix."},
    "print": {
        "arity": [1, 1],
        "sig": "(print x)",
        "capabilities": ["io"],
        "desc": "Print human-readable value to stdout; requires 'io' capability granted via (grant io).",
    },
}

CAPABILITIES_SPEC: dict[str, Any] = {
    "system_summary": (
        "Netelpro uses a static effect/capability system (Phase 3). Effects like I/O "
        "must be granted file-wide via top-level (grant ...) forms. Using an un-granted "
        "capability is prosecuted as a static compile error, not a runtime exception."
    ),
    "known_capabilities": {
        "io": "Standard output access. Required by the 'print' primitive. Granted via '(grant io)'."
    },
    "sorry_holes": {
        "summary": (
            "Typed hole placeholder (sorry \"reason\") from Phase 4. Enforces honesty in incomplete "
            "implementations. Compiles cleanly, manifests are extracted into audit tables, and "
            "raises StrayHoleError at runtime only if the hole branch is executed."
        )
    },
}

# ---------------------------------------------------------------------------
# MCP Tool Schemas
# ---------------------------------------------------------------------------

TOOLS_LIST: list[dict[str, Any]] = [
    {
        "name": "netelpro_compile",
        "description": "Statically compiles and audits Netelpro source code without execution. Runs parse, capability, and hole auditing, plus optional LLVM codegen validation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The Netelpro source code to compile and validate.",
                },
                "backend": {
                    "type": "string",
                    "enum": ["interpreter", "native"],
                    "default": "interpreter",
                    "description": "Compilation backend target ('interpreter' or 'native').",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "netelpro_eval",
        "description": "Executes Netelpro source code in an isolated subprocess with strict limits. Returns final evaluated value and captured stdout.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The Netelpro source code to execute.",
                },
                "native": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to execute natively via LLVM JIT instead of the reference interpreter.",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "netelpro_verify",
        "description": "Performs differential parity verification on a Netelpro rule filter (defn filter-rule ...), testing cases across native LLVM JIT and reference interpreter.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Netelpro source code defining (defn filter-rule ...).",
                },
                "cases": {
                    "type": "array",
                    "description": "Array of test vectors with args and expected boolean result (max 100).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "args": {
                                "type": "array",
                                "items": {"type": ["integer", "boolean", "string"]},
                                "description": "Positional arguments to pass to filter-rule.",
                            },
                            "expected": {
                                "type": "boolean",
                                "description": "Expected boolean decision outcome.",
                            },
                        },
                        "required": ["args", "expected"],
                    },
                },
            },
            "required": ["source", "cases"],
        },
    },
    {
        "name": "netelpro_spec",
        "description": "Provides static language specification knowledge, special forms, primitives, capabilities, and sorry-hole mechanisms.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["all", "special_forms", "primitives", "capabilities"],
                    "default": "all",
                    "description": "Category of specification knowledge to inspect.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional search term to filter forms and capabilities.",
                },
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Value Serialization & Limit Enforcement Helpers
# ---------------------------------------------------------------------------


def to_json_val(val: Any) -> Any:
    """Convert Netelpro evaluator return value into JSON-serializable Python data."""
    if is_nil(val):
        return None
    if type(val) is bool:
        return bool(val)
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        return float(val)
    if isinstance(val, str):
        return str(val)
    if isinstance(val, StrayList):
        return [to_json_val(x) for x in val]
    if isinstance(val, Closure):
        name_part = f" {val.name}" if val.name else ""
        return f"<fn{name_part}>"
    return format_value(val)


def check_source_bytes(source: str) -> list[dict[str, Any]]:
    """Enforce MAX_SOURCE_BYTES limit."""
    encoded = source.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        return [
            {
                "phase": "limit",
                "line": 0,
                "col": 0,
                "message": f"source size ({len(encoded)} bytes) exceeds MAX_SOURCE_BYTES ({MAX_SOURCE_BYTES})",
            }
        ]
    return []


def check_nesting_depth(source: str) -> list[dict[str, Any]]:
    """Enforce PARSE_DEPTH_BUDGET bracket nesting depth limit."""
    depth = 0
    in_str = False
    in_comment = False
    escape = False
    line = 1
    col = 0

    for ch in source:
        if ch == "\n":
            line += 1
            col = 0
            in_comment = False
            if in_str and not escape:
                in_str = False
            escape = False
            continue

        col += 1

        if in_comment:
            continue

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == ";":
            in_comment = True
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "([{":
            depth += 1
            if depth > PARSE_DEPTH_BUDGET:
                return [
                    {
                        "phase": "limit",
                        "line": line,
                        "col": col,
                        "message": f"bracket nesting depth {depth} exceeds budget {PARSE_DEPTH_BUDGET}",
                    }
                ]
        elif ch in ")]}":
            depth = max(0, depth - 1)

    return []


# ---------------------------------------------------------------------------
# Tool 1: netelpro_compile
# ---------------------------------------------------------------------------


def tool_compile(source: str, backend: str = "interpreter") -> dict[str, Any]:
    """Execute static validation only: parse, caps, holes, and optional codegen check."""
    byte_errs = check_source_bytes(source)
    if byte_errs:
        return {"ok": False, "errors": byte_errs, "hole_manifest": [], "grants": [], "effects": {}}

    depth_errs = check_nesting_depth(source)
    if depth_errs:
        return {"ok": False, "errors": depth_errs, "hole_manifest": [], "grants": [], "effects": {}}

    parse_res = parse(source)
    if not parse_res.ok:
        errs = [
            {"phase": "parse", "line": e.line, "col": e.col, "message": e.message}
            for e in parse_res.errors
        ]
        return {"ok": False, "errors": errs, "hole_manifest": [], "grants": [], "effects": {}}

    program = parse_res.program
    granted = collect_grants(program)
    grants = sorted(list(granted))

    errors: list[dict[str, Any]] = []

    # Per-function effect typing (v0.6): expose inferred effect sets so an LLM
    # consumer can mechanically verify gate-rule purity before trusting a rule.
    effects: dict[str, Any] = {}
    try:
        from netelpro.effects import infer_effects

        effect_sets = infer_effects(program)
        effects = {name: sorted(list(effs)) for name, effs in effect_sets.items()}
    except Exception as e:
        errors.append({"phase": "effect", "line": 0, "col": 0, "message": str(e)})

    cap_errors = check_capabilities(program, granted)
    if cap_errors:
        errors.extend(
            [
                {"phase": "cap", "line": e.line, "col": e.col, "message": e.message}
                for e in cap_errors
            ]
        )

    hole_errors, hole_manifest = check_holes(program)
    if hole_errors:
        errors.extend(
            [
                {"phase": "hole", "line": e.line, "col": e.col, "message": e.message}
                for e in hole_errors
            ]
        )

    if errors:
        return {
            "ok": False,
            "errors": errors,
            "hole_manifest": hole_manifest,
            "grants": grants,
            "effects": effects,
        }

    if backend == "native":
        try:
            import llvmlite  # noqa: F401

            from netelpro.codegen import CodegenError, compile_program
        except ImportError:
            return {
                "ok": False,
                "errors": [
                    {
                        "phase": "codegen",
                        "line": 0,
                        "col": 0,
                        "message": "llvmlite not available",
                    }
                ],
                "hole_manifest": hole_manifest,
                "grants": grants,
                "effects": effects,
            }

        try:
            compile_program(program)
        except CodegenError as e:
            return {
                "ok": False,
                "errors": [
                    {
                        "phase": "codegen",
                        "line": e.line,
                        "col": e.col,
                        "message": e.message,
                    }
                ],
                "hole_manifest": hole_manifest,
                "grants": grants,
                "effects": effects,
            }
        except StrayError as e:
            line = getattr(e, "line", 0)
            col = getattr(e, "col", 0)
            msg = getattr(e, "message", str(e))
            return {
                "ok": False,
                "errors": [{"phase": "codegen", "line": line, "col": col, "message": msg}],
                "hole_manifest": hole_manifest,
                "grants": grants,
                "effects": effects,
            }
        except Exception as e:
            return {
                "ok": False,
                "errors": [{"phase": "codegen", "line": 0, "col": 0, "message": str(e)}],
                "hole_manifest": hole_manifest,
                "grants": grants,
                "effects": effects,
            }

    return {
        "ok": True,
        "errors": [],
        "hole_manifest": hole_manifest,
        "grants": grants,
        "effects": effects,
    }


# ---------------------------------------------------------------------------
# Tool 2: netelpro_eval (Subprocess Execution)
# ---------------------------------------------------------------------------


def tool_eval(source: str, native: bool = False) -> dict[str, Any]:
    """Execute Netelpro source code in an isolated subprocess with limits."""
    byte_errs = check_source_bytes(source)
    if byte_errs:
        return {
            "ok": False,
            "result": None,
            "stdout": "",
            "errors": byte_errs,
            "hole_manifest": [],
        }

    depth_errs = check_nesting_depth(source)
    if depth_errs:
        return {
            "ok": False,
            "result": None,
            "stdout": "",
            "errors": depth_errs,
            "hole_manifest": [],
        }

    # SIM115 justification: close-before-spawn is REQUIRED on Windows — an open
    # parent handle would block the child worker from writing this file (lock).
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")  # noqa: SIM115
    temp_path = temp_file.name
    temp_file.close()

    try:
        cmd = [sys.executable, "-m", "netelpro.mcp_server", "--exec-eval", "--out", temp_path]
        payload = json.dumps({"source": source, "native": native})

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        try:
            stdout_text, stderr_text = proc.communicate(input=payload, timeout=EVAL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            _close_pipes(proc)
            return {
                "ok": False,
                "result": None,
                "stdout": "",
                "errors": [{"phase": "limit", "line": 0, "col": 0, "message": "evaluation timed out"}],
                "hole_manifest": [],
            }

        captured_stdout = (stdout_text or "").rstrip("\r\n")[:RESULT_STRING_CAP]

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            try:
                with open(temp_path, encoding="utf-8") as f:
                    res_data = json.load(f)
                if isinstance(res_data, dict):
                    if not res_data.get("stdout"):
                        res_data["stdout"] = captured_stdout
                    return res_data
            except Exception:
                pass

        if proc.returncode != 0:
            err_msg = (
                captured_stdout.strip()
                or (stderr_text or "").strip()
                or f"process exited with code {proc.returncode}"
            )
            return {
                "ok": False,
                "result": None,
                "stdout": captured_stdout,
                "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": err_msg}],
                "hole_manifest": [],
            }

        return {
            "ok": True,
            "result": None,
            "stdout": captured_stdout,
            "errors": [],
            "hole_manifest": [],
        }

    finally:
        if os.path.exists(temp_path):
            with contextlib.suppress(Exception):
                os.remove(temp_path)


def _worker_eval(argv: list[str] | None = None) -> int:
    """Worker entrypoint for --exec-eval running in a child process."""
    if argv is None:
        argv = sys.argv[1:]
    out_path = None
    if "--out" in argv:
        idx = argv.index("--out")
        if idx + 1 < len(argv):
            out_path = argv[idx + 1]

    raw_input = sys.stdin.read()
    if not raw_input.strip():
        res = {
            "ok": False,
            "result": None,
            "stdout": "",
            "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": "empty input"}],
            "hole_manifest": [],
        }
        _write_worker_result(res, out_path)
        return 0

    try:
        data = json.loads(raw_input)
    except Exception as e:
        res = {
            "ok": False,
            "result": None,
            "stdout": "",
            "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": f"invalid JSON: {e}"}],
            "hole_manifest": [],
        }
        _write_worker_result(res, out_path)
        return 0

    source = data.get("source", "")
    native = bool(data.get("native", False))

    byte_errs = check_source_bytes(source)
    if byte_errs:
        res = {"ok": False, "result": None, "stdout": "", "errors": byte_errs, "hole_manifest": []}
        _write_worker_result(res, out_path)
        return 0

    depth_errs = check_nesting_depth(source)
    if depth_errs:
        res = {"ok": False, "result": None, "stdout": "", "errors": depth_errs, "hole_manifest": []}
        _write_worker_result(res, out_path)
        return 0

    parse_res = parse(source)
    if not parse_res.ok:
        errs = [
            {"phase": "parse", "line": e.line, "col": e.col, "message": e.message}
            for e in parse_res.errors
        ]
        res = {"ok": False, "result": None, "stdout": "", "errors": errs, "hole_manifest": []}
        _write_worker_result(res, out_path)
        return 0

    program = parse_res.program

    granted = collect_grants(program)
    cap_errors = check_capabilities(program, granted)
    if cap_errors:
        errs = [
            {"phase": "cap", "line": e.line, "col": e.col, "message": e.message}
            for e in cap_errors
        ]
        res = {"ok": False, "result": None, "stdout": "", "errors": errs, "hole_manifest": []}
        _write_worker_result(res, out_path)
        return 0

    hole_errors, hole_manifest = check_holes(program)
    if hole_errors:
        errs = [
            {"phase": "hole", "line": e.line, "col": e.col, "message": e.message}
            for e in hole_errors
        ]
        res = {
            "ok": False,
            "result": None,
            "stdout": "",
            "errors": errs,
            "hole_manifest": hole_manifest,
        }
        _write_worker_result(res, out_path)
        return 0

    if not native:
        try:
            val = evaluate(program)
            json_val = to_json_val(val)
            res = {
                "ok": True,
                "result": json_val,
                "stdout": "",
                "errors": [],
                "hole_manifest": hole_manifest,
            }
        except RecursionError:
            res = {
                "ok": False,
                "result": None,
                "stdout": "",
                "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": "recursion depth exceeded"}],
                "hole_manifest": hole_manifest,
            }
        except StrayHoleError as e:
            res = {
                "ok": False,
                "result": None,
                "stdout": "",
                "errors": [{"phase": "hole", "line": e.line, "col": e.col, "message": e.reason}],
                "hole_manifest": hole_manifest,
            }
        except StrayRuntimeError as e:
            res = {
                "ok": False,
                "result": None,
                "stdout": "",
                "errors": [{"phase": "runtime", "line": e.line, "col": e.col, "message": e.message}],
                "hole_manifest": hole_manifest,
            }
        except StrayError as e:
            res = {
                "ok": False,
                "result": None,
                "stdout": "",
                "errors": [
                    {
                        "phase": "runtime",
                        "line": getattr(e, "line", 0),
                        "col": getattr(e, "col", 0),
                        "message": getattr(e, "message", str(e)),
                    }
                ],
                "hole_manifest": hole_manifest,
            }
        except Exception as e:
            res = {
                "ok": False,
                "result": None,
                "stdout": "",
                "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": str(e)}],
                "hole_manifest": hole_manifest,
            }

        _write_worker_result(res, out_path)
        return 0
    else:
        try:
            import llvmlite  # noqa: F401

            from netelpro.codegen import CodegenError, compile_and_run
        except ImportError:
            res = {
                "ok": False,
                "result": None,
                "stdout": "",
                "errors": [{"phase": "codegen", "line": 0, "col": 0, "message": "llvmlite not available"}],
                "hole_manifest": hole_manifest,
            }
            _write_worker_result(res, out_path)
            return 0

        try:
            val = compile_and_run(program)
            # llvmlite JIT returns a raw ctypes int for top-level programs;
            # wrap it so the result is JSON-serializable (M6, auditor finding).
            native_val: Any = int(val) if isinstance(val, int) and not isinstance(val, bool) else to_json_val(val)
            res = {
                "ok": True,
                "result": native_val,
                "stdout": "",
                "errors": [],
                "hole_manifest": hole_manifest,
            }
        except CodegenError as e:
            res = {
                "ok": False,
                "result": None,
                "stdout": "",
                "errors": [{"phase": "codegen", "line": e.line, "col": e.col, "message": e.message}],
                "hole_manifest": hole_manifest,
            }
        except StrayError as e:
            res = {
                "ok": False,
                "result": None,
                "stdout": "",
                "errors": [
                    {
                        "phase": "runtime",
                        "line": getattr(e, "line", 0),
                        "col": getattr(e, "col", 0),
                        "message": getattr(e, "message", str(e)),
                    }
                ],
                "hole_manifest": hole_manifest,
            }
        except Exception as e:
            res = {
                "ok": False,
                "result": None,
                "stdout": "",
                "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": str(e)}],
                "hole_manifest": hole_manifest,
            }

        _write_worker_result(res, out_path)
        return 0


# ---------------------------------------------------------------------------
# Tool 3: netelpro_verify (Subprocess Differential Testing)
# ---------------------------------------------------------------------------


def tool_verify(source: str, cases: list[Any]) -> dict[str, Any]:
    """Execute differential parity verification across native LLVM JIT and interpreter."""
    if len(cases) > MAX_CASES:
        return {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": [
                {
                    "phase": "limit",
                    "line": 0,
                    "col": 0,
                    "message": f"cases count ({len(cases)}) exceeds MAX_CASES ({MAX_CASES})",
                }
            ],
        }

    byte_errs = check_source_bytes(source)
    if byte_errs:
        return {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": byte_errs,
        }

    depth_errs = check_nesting_depth(source)
    if depth_errs:
        return {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": depth_errs,
        }

    case_errs = _validate_verify_cases(cases)
    if case_errs:
        return {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": case_errs,
        }

    # SIM115 justification: close-before-spawn is REQUIRED on Windows — an open
    # parent handle would block the child worker from writing this file (lock).
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")  # noqa: SIM115
    temp_path = temp_file.name
    temp_file.close()

    try:
        cmd = [sys.executable, "-m", "netelpro.mcp_server", "--exec-verify", "--out", temp_path]
        payload = json.dumps({"source": source, "cases": cases})

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        try:
            stdout_text, stderr_text = proc.communicate(input=payload, timeout=EVAL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            _close_pipes(proc)
            return {
                "ok": False,
                "mismatches": [],
                "manifest": [],
                "errors": [{"phase": "limit", "line": 0, "col": 0, "message": "evaluation timed out"}],
            }

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            try:
                with open(temp_path, encoding="utf-8") as f:
                    res_data = json.load(f)
                if isinstance(res_data, dict):
                    return res_data
            except Exception:
                pass

        err_msg = (
            (stdout_text or "").strip()
            or (stderr_text or "").strip()
            or f"process exited with code {proc.returncode}"
        )
        return {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": err_msg}],
        }
    finally:
        if os.path.exists(temp_path):
            with contextlib.suppress(Exception):
                os.remove(temp_path)


def _worker_verify(argv: list[str] | None = None) -> int:
    """Worker entrypoint for --exec-verify running in a child process."""
    if argv is None:
        argv = sys.argv[1:]
    out_path = _validated_out_path(argv)
    raw_input = sys.stdin.read(MAX_LINE_BYTES)
    if not raw_input.strip():
        res = {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": "empty input"}],
        }
        _write_worker_result(res, out_path)
        return 0

    try:
        data = json.loads(raw_input)
    except Exception as e:
        res = {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": f"invalid JSON: {e}"}],
        }
        _write_worker_result(res, out_path)
        return 0

    source = data.get("source", "")
    cases_raw = data.get("cases", [])

    if len(cases_raw) > MAX_CASES:
        res = {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": [
                {
                    "phase": "limit",
                    "line": 0,
                    "col": 0,
                    "message": f"cases count ({len(cases_raw)}) exceeds MAX_CASES ({MAX_CASES})",
                }
            ],
        }
        _write_worker_result(res, out_path)
        return 0

    byte_errs = check_source_bytes(source)
    if byte_errs:
        res = {"ok": False, "mismatches": [], "manifest": [], "errors": byte_errs}
        _write_worker_result(res, out_path)
        return 0

    depth_errs = check_nesting_depth(source)
    if depth_errs:
        res = {"ok": False, "mismatches": [], "manifest": [], "errors": depth_errs}
        _write_worker_result(res, out_path)
        return 0

    try:
        from netelpro.codegen import CodegenError
        from netelpro.rule_filter import RuleFilterError, compile_filter
    except ImportError:
        res = {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": [{"phase": "codegen", "line": 0, "col": 0, "message": "rule_filter or codegen not available"}],
        }
        _write_worker_result(res, out_path)
        return 0

    try:
        rf = compile_filter(source)
    except RuleFilterError as e:
        res = {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": [{"phase": "rule_filter", "line": e.line, "col": e.col, "message": e.message}],
        }
        _write_worker_result(res, out_path)
        return 0
    except CodegenError as e:
        res = {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": [{"phase": "codegen", "line": e.line, "col": e.col, "message": e.message}],
        }
        _write_worker_result(res, out_path)
        return 0
    except StrayError as e:
        res = {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": [
                {
                    "phase": "runtime",
                    "line": getattr(e, "line", 0),
                    "col": getattr(e, "col", 0),
                    "message": getattr(e, "message", str(e)),
                }
            ],
        }
        _write_worker_result(res, out_path)
        return 0
    except Exception as e:
        res = {
            "ok": False,
            "mismatches": [],
            "manifest": [],
            "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": str(e)}],
        }
        _write_worker_result(res, out_path)
        return 0

    manifest = rf.manifest()

    cases_tuples: list[tuple[Any, bool]] = []
    for c in cases_raw:
        if isinstance(c, dict):
            c_args = c.get("args", [])
            c_exp = bool(c.get("expected", True))
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            c_args = c[0]
            c_exp = bool(c[1])
        else:
            continue
        c_args_tuple = tuple(c_args) if isinstance(c_args, (list, tuple)) else (c_args,)
        cases_tuples.append((c_args_tuple, c_exp))

    try:
        raw_mismatches = rf.verify(cases_tuples)
        mismatches_formatted = [
            {
                "args": list(m[0]) if isinstance(m[0], (list, tuple)) else [m[0]],
                "expected": m[1],
                "interpreted": m[2],
                "native": m[3],
            }
            for m in raw_mismatches
        ]
        res = {
            "ok": len(mismatches_formatted) == 0,
            "mismatches": mismatches_formatted,
            "manifest": manifest,
            "errors": [],
        }
    except RuleFilterError as e:
        res = {
            "ok": False,
            "mismatches": [],
            "manifest": manifest,
            "errors": [{"phase": "rule_filter", "line": e.line, "col": e.col, "message": e.message}],
        }
    except Exception as e:
        res = {
            "ok": False,
            "mismatches": [],
            "manifest": manifest,
            "errors": [{"phase": "runtime", "line": 0, "col": 0, "message": str(e)}],
        }

    _write_worker_result(res, out_path)
    return 0


def _write_worker_result(res: dict[str, Any], out_path: str | None) -> None:
    """Write worker structured result to out_path or stdout."""
    text = json.dumps(res)
    if out_path:
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            return
        except Exception:
            pass
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Tool 4: netelpro_spec (Static Language Knowledge)
# ---------------------------------------------------------------------------


def tool_spec(category: str = "all", query: str | None = None) -> dict[str, Any]:
    """Export static language knowledge about special forms, primitives, and capabilities."""
    forms: dict[str, Any] = {}
    if category in ("all", "special_forms"):
        forms.update(SPECIAL_FORMS)
    if category in ("all", "primitives"):
        forms.update(PRIMITIVES)

    capabilities: dict[str, Any] = {}
    if category in ("all", "capabilities"):
        capabilities = dict(CAPABILITIES_SPEC)

    q = (query or "").strip().lower()
    if q:
        filtered_forms = {}
        for name, info in forms.items():
            sig = str(info.get("sig", "")).lower()
            desc = str(info.get("desc", "")).lower()
            if q in name.lower() or q in sig or q in desc:
                filtered_forms[name] = info
        forms = filtered_forms

        filtered_caps: dict[str, Any] = {}
        if "known_capabilities" in capabilities:
            known = {
                k: v
                for k, v in capabilities["known_capabilities"].items()
                if q in k.lower() or q in str(v).lower()
            }
            if known:
                filtered_caps["known_capabilities"] = known
        if "sorry_holes" in capabilities and q in str(capabilities["sorry_holes"]).lower():
            filtered_caps["sorry_holes"] = capabilities["sorry_holes"]
        if "system_summary" in capabilities and q in str(capabilities["system_summary"]).lower():
            filtered_caps["system_summary"] = capabilities["system_summary"]
        capabilities = filtered_caps

    return {
        "version": "0.1.0",
        "forms": forms,
        "capabilities": capabilities,
    }


# ---------------------------------------------------------------------------
# Importable API: dispatch(name, args)
# ---------------------------------------------------------------------------


def dispatch(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch a tool call by name and arguments, returning a structured result dictionary.

    Args:
        name: One of 'netelpro_compile', 'netelpro_eval', 'netelpro_verify', 'netelpro_spec'.
        args: Dictionary of arguments matching the tool's inputSchema.

    Returns:
        A dictionary containing the tool's structured outcome.

    Raises:
        ValueError: If the tool name is unknown.
    """
    if args is None:
        args = {}

    if name == "netelpro_compile":
        source = args.get("source", "")
        backend = args.get("backend", "interpreter")
        return tool_compile(source=source, backend=backend)
    elif name == "netelpro_eval":
        source = args.get("source", "")
        native = bool(args.get("native", False))
        return tool_eval(source=source, native=native)
    elif name == "netelpro_verify":
        source = args.get("source", "")
        cases = args.get("cases", [])
        return tool_verify(source=source, cases=cases)
    elif name == "netelpro_spec":
        category = args.get("category", "all")
        query = args.get("query")
        return tool_spec(category=category, query=query)
    else:
        raise ValueError(f"Unknown tool '{name}'")


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 Stdio Loop & Transport
# ---------------------------------------------------------------------------


def _send_jsonrpc_response(resp: dict[str, Any]) -> None:
    """Serialize and write a line-delimited JSON-RPC response to stdout."""
    text = json.dumps(resp, ensure_ascii=False)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _handle_request(method: str, params: Any, req_id: Any) -> dict[str, Any] | None:
    """Handle a parsed JSON-RPC request and return the response dict (or None for notifications)."""
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "serverInfo": {
                    "name": "netelpro-mcp",
                    "version": "0.1.0",
                },
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS_LIST,
            },
        }

    if method == "tools/call":
        if not isinstance(params, dict):
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: params must be an object",
                    },
                }
            return None

        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not isinstance(tool_name, str):
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: 'name' is required and must be a string",
                    },
                }
            return None

        if not isinstance(arguments, dict):
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: 'arguments' must be an object",
                    },
                }
            return None

        try:
            tool_res = dispatch(tool_name, arguments)
        except ValueError as e:
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": str(e),
                    },
                }
            return None
        except Exception as e:
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {e}",
                    },
                }
            return None

        if req_id is not None:
            is_error = not tool_res.get("ok", True)
            json_text = json.dumps(tool_res)
            result_obj = {
                **tool_res,
                "content": [{"type": "text", "text": json_text}],
                "structuredContent": tool_res,
                "isError": is_error,
            }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result_obj,
            }
        return None

    # Unknown method
    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }
    return None


def _run_stdio_server() -> int:
    """Run the MCP JSON-RPC 2.0 stdio server loop.

    Reads line-delimited JSON-RPC messages from sys.stdin, executes requested
    methods, and writes line-delimited JSON responses to sys.stdout followed by flush.
    Content-Length header is NOT required.
    """
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdin.reconfigure(encoding="utf-8")

    def _read_capped_line() -> str:
        """Read one stdio line with a hard byte cap (never buffers beyond MAX_LINE_BYTES).

        Returns '' on EOF. Oversized lines are truncated in memory and returned
        with a sentinel marker so the caller can reject them without having
        buffered the whole payload.
        """
        buf: list[str] = []
        total = 0
        while True:
            chunk = sys.stdin.readline(MAX_LINE_BYTES)  # caps chars read per call
            if not chunk:
                return ""
            total += len(chunk)
            buf.append(chunk)
            if chunk.endswith("\n"):
                line = "".join(buf)
                if total > MAX_LINE_BYTES:
                    return line[:MAX_LINE_BYTES] + "\x00OVERSIZE\x00"
                return line
            if total > MAX_LINE_BYTES:
                # Still mid-line: drain remaining chunks without buffering beyond cap.
                while True:
                    tail = sys.stdin.readline(MAX_LINE_BYTES)
                    if not tail or tail.endswith("\n"):
                        break
                return "\x00OVERSIZE\x00"

    while True:
        try:
            line = _read_capped_line()
        except Exception:
            break

        if not line:
            break

        if line.endswith("\x00OVERSIZE\x00"):
            # Oversized frame: rejected without ever buffering the full payload.
            _send_jsonrpc_response(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32600,
                        "message": f"Invalid Request: line exceeds MAX_LINE_BYTES ({MAX_LINE_BYTES}) bytes",
                    },
                }
            )
            continue

        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {e}",
                },
            }
            _send_jsonrpc_response(err_resp)
            continue

        if not isinstance(req, dict):
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: expected JSON object",
                },
            }
            _send_jsonrpc_response(err_resp)
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params")

        if not isinstance(method, str):
            if req_id is not None:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: 'method' must be a string",
                    },
                }
                _send_jsonrpc_response(err_resp)
            continue

        resp = _handle_request(method, params, req_id)
        if resp is not None:
            _send_jsonrpc_response(resp)

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Netelpro MCP server."""
    if argv is None:
        argv = sys.argv[1:]

    if "--exec-eval" in argv:
        return _worker_eval(argv)
    if "--exec-verify" in argv:
        return _worker_verify(argv)

    return _run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())
