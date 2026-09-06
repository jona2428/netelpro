"""Netelpro rule filter -- Host bridge for compiled pure gate rules.

The prosecutor thesis applied to host security and gatekeeping:
In Neuromancer and host environments, access control and decision gates are
modeled as pure Netelpro functions. The host requires deterministic, zero-overhead
evaluation of decision rules compiled directly to native machine code via LLVM,
with prosecutorial static auditing and differential verification against the
reference interpreter.

Architecture & Contracts:
1. Purpose & Neuromancer Gate Rules:
   Hosts define gate rules as pure Netelpro source units. The bridge compiles
   the source natively into an LLVM JIT execution engine (`CompiledProgram`),
   enforcing that rules are static, capability-audited, and syntactically clean
   before execution. Python callers invoke rules with native execution speed
   through `RuleFilter.decide(*args)`.

2. The Required `(defn filter-rule ...)` Convention:
   Every valid rule filter source unit must define a top-level named function:
       (defn filter-rule (param1 param2 ...) body)
   Parameters of `filter-rule` resolve statically to `Int` (i64), `Bool` (i1) or
   `Str` (i8* NUL-terminated) by bidirectional type inference, driven by how each
   parameter is used: an unused or Int-context parameter binds to `Int`; a parameter
   used in a boolean context (`if`/`and`/`or`/`not` operands) binds to `Bool`; a
   parameter compared with string literals (`==`/`prefix?`) or printed binds to `Str`.
   Mixed use (e.g. Bool demanded where Int is required, or a concrete heterogeneous
   comparison) is a compile error with exact coordinates.
   String params are READ-ONLY at the native boundary: they cross as `c_char_p`
   (UTF-8 encoded in decide()), and the rule must not return a string (return type
   must be i1 or i64 -- return-Str is prosecuted at compile time).
   If `filter-rule` is not defined in the source, compilation is rejected immediately
   with `RuleFilterError` carrying the precise line and column coordinates of the definition.

3. Bool Calling Convention Findings at Machine Boundary:
   Investigation of Netelpro's LLVM backend (`netelpro/codegen.py`):
   - In `codegen.py`, top-level expressions inside `main()` widen boolean expressions
     to 64-bit signed integers via zero-extension (`mb.zext(last_val, i64)`).
   - In contrast, user function definitions (`defn`) preserve un-widened native types:
     functions returning `Bool` emit LLVM functions returning 1-bit flags (`i1`, i.e.,
     `ir.IntType(1)`), while functions returning `Int` emit `i64` (`ir.IntType(64)`).
   - In machine ABIs (both Windows x64 and System V AMD64), 1-bit return values are
     placed in the low 8 bits of RAX (the AL register). LLVM does not guarantee that
     the upper 56 bits of RAX are cleared upon returning an `i1`.
   - Declaring the native foreign function interface using `ctypes.c_bool` as the return
     type ensures that ctypes reads only the 8-bit `AL` register, avoiding false positives
     from residual dirty bits in RAX. For integer-returning rules, `ctypes.c_int64` is used.
     The resulting value is coerced to Python `bool` to provide a consistent boolean API.

4. Differential Verification Model:
   To ensure native LLVM code generator fidelity against the specification, `verify(cases)`
   executes differential testing against the reference tree-walking interpreter
   (`netelpro.evaluator.run_source`). For each test vector `(args, expected)`, the call form
   `(filter-rule arg1 arg2 ...)` is appended to the source text and evaluated in a freshly
   initialized, isolated `Environment`. Any discrepancy between the native execution,
   the interpreted result, or the expected boolean is captured as an audit record
   `(args, expected, interpreted, native)`.
"""
from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

import llvmlite.ir as ir

from netelpro.ast_nodes import Defn, Sym
from netelpro.caps import check_capabilities, collect_grants
from netelpro.codegen import CodegenError, CompiledProgram, compile_program
from netelpro.evaluator import Environment, StrayError, run_source
from netelpro.holes import check_holes
from netelpro.parser import parse
from netelpro.str_native import StrNativeError, read_arena_string


class RuleFilterError(Exception):
    """Prosecutorial diagnostic record for rule filtering and compilation failures.

    Carries exact source coordinates (line, col) and human-readable message.
    """

    def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
        self.message: str = message
        self.line: int = line
        self.col: int = col
        super().__init__(f"line {line}, col {col}: {message}")

    def __str__(self) -> str:
        return f"line {self.line}, col {self.col}: {self.message}"


class RuleFilter:
    """Compiled native Netelpro gate rule bridge for host execution."""

    def __init__(self, source: str, defn_name: str = "filter-rule") -> None:
        """Compile a Netelpro gate rule source immediately to native machine code.

        Executes the prosecutorial pipeline:
        1. Parse source text and validate AST structure (rejecting parse errors).
        2. Verify capability grants (rejecting unauthorized effects).
        3. Collect holes manifest (declared sorries are legal and listed, not rejected).
        4. Validate that `filter-rule` is defined and its parameters resolve cleanly
           to native types (Int i64 / Bool i1).
        5. Compile to native machine code via LLVM codegen (if no declared sorries).

        Raises:
            RuleFilterError: If parsing, capability checking, hole auditing, or
                codegen fails, or if `filter-rule` is missing.
        """
        self._source: str = source
        self.source: str = source

        # 1. Parse phase
        parse_result = parse(source)
        if not parse_result.ok:
            first_err = parse_result.errors[0]
            raise RuleFilterError(first_err.message, line=first_err.line, col=first_err.col)

        program = parse_result.program
        self.program = program

        # 2. Capability verification
        granted = collect_grants(program)
        cap_errors = check_capabilities(program, granted)
        if cap_errors:
            first_cap = cap_errors[0]
            raise RuleFilterError(first_cap.message, line=first_cap.line, col=first_cap.col)

        # 3. Static hole auditing (collecting manifest; declared sorries are legal)
        hole_errors, manifest_entries = check_holes(program)
        if hole_errors:
            first_hole = hole_errors[0]
            raise RuleFilterError(first_hole.message, line=first_hole.line, col=first_hole.col)

        self._manifest: list[str] = [
            f"line {entry['line']}, col {entry['col']}: {entry['reason']}"
            for entry in manifest_entries
        ]
        self._sorry_entries = manifest_entries

        # 4. Filter-rule definition audit
        entry_defn: Defn | None = None
        for form in program.forms:
            if isinstance(form, Defn):
                name = form.name.name if isinstance(form.name, Sym) else str(form.name)
                if name == defn_name:
                    entry_defn = form
                    break

        if entry_defn is None:
            raise RuleFilterError(
                f"function '{defn_name}' is not defined in source",
                line=0,
                col=0,
            )

        self._defn_name: str = defn_name
        self._defn_line: int = entry_defn.line
        self._defn_col: int = entry_defn.col
        self._arity: int = len(entry_defn.params)
        self._entry_defn: Defn = entry_defn
        self._param_names: list[str] = [
            p.name if isinstance(p, Sym) else str(p) for p in entry_defn.params
        ]

        # 5. Native compilation
        self._compiled: CompiledProgram | None = None
        if not manifest_entries:
            try:
                compiled = compile_program(program)
            except CodegenError as e:
                raise RuleFilterError(e.message, line=e.line, col=e.col) from e
            except StrayError as e:
                line = getattr(e, "line", 0)
                col = getattr(e, "col", 0)
                msg = getattr(e, "message", str(e))
                raise RuleFilterError(msg, line=line, col=col) from e
            except Exception as e:
                line = getattr(e, "line", 0)
                col = getattr(e, "col", 0)
                raise RuleFilterError(str(e), line=line, col=col) from e

            # Audit LLVM parameter types and determine return calling convention
            llvm_fn: ir.Function | None = None
            for fn in compiled.module.functions:
                if fn.name == defn_name:
                    llvm_fn = fn
                    break

            if llvm_fn is None:
                raise RuleFilterError(
                    f"compiled function '{defn_name}' not found in native module",
                    line=self._defn_line,
                    col=self._defn_col,
                )

            # Per-parameter type audit against the codegen-resolved signature.
            # Params are Int (i64), Bool (i1) or Str (i8* NUL-terminated, read-
            # only); anything else is a bridge bug and is prosecuted as one.
            for arg in llvm_fn.args:
                w = getattr(arg.type, "width", None)
                if w is None:
                    continue  # pointer param (Str): legal since v0.3
                if w not in (64, 1):
                    raise RuleFilterError(
                        f"parameter of '{defn_name}' resolved to unsupported type i{w}",
                        line=self._defn_line,
                        col=self._defn_col,
                    )
            if defn_name == "filter-rule" and (
                llvm_fn.function_type.return_type.width != 1
                and llvm_fn.function_type.return_type.width != 64
            ):
                raise RuleFilterError(
                    "'filter-rule' must return Bool (i1) or Int (i64) -- strings cannot cross "
                    "the native boundary as return values (read-only boundary, v0.3)",
                    line=self._defn_line,
                    col=self._defn_col,
                )

            self._compiled = compiled
            if llvm_fn.function_type.return_type == ir.PointerType():
                # v0.5 builder: Str products cross as c_char_p.
                self._restype = ctypes.c_char_p
            elif llvm_fn.function_type.return_type == ir.IntType(1):
                self._restype = ctypes.c_bool
            else:
                self._restype = ctypes.c_int64

            # Per-parameter ctypes prototype: i1 params cross as c_bool (ABI:
            # ctypes reads the low byte), i64 as c_int64, i8* (Str, NUL-
            # terminated read-only) as c_char_p. Python str args are encoded
            # UTF-8 at call time in decide(); bytes pass through unchanged.
            self._argtypes: list[type[ctypes._SimpleCData]] = [
                ctypes.c_bool
                if (w := getattr(arg.type, "width", None)) == 1
                else ctypes.c_int64
                if w == 64
                else ctypes.c_char_p
                for arg in llvm_fn.args
            ]
        else:
            self._compiled = None
            self._restype = ctypes.c_bool
            self._argtypes: list[type[ctypes._SimpleCData]] = [ctypes.c_int64] * self._arity

    def manifest(self) -> list[str]:
        """Return the declared sorry holes as human-readable diagnostic strings.

        Returns:
            A list of diagnostic strings formatted as 'line <line>, col <col>: <reason>'.
        """
        return list(self._manifest)

    def decide(self, *args: int | bool) -> bool:
        """Call the compiled defn named `filter-rule` natively and return a boolean decision.

        Args:
            *args: Positional arguments matching the arity of `filter-rule` --
                `int` for Int params, `bool` for Bool params (per-param ctypes
                prototype is built from the compiled LLVM signature).

        Returns:
            Boolean outcome of the gate rule decision.

        Raises:
            RuleFilterError: If the rule contains unimplemented sorry holes,
                if arity mismatches, or if machine address lookup fails.
        """
        if self._compiled is None:
            first_sorry = self._sorry_entries[0] if self._sorry_entries else None
            line = first_sorry["line"] if first_sorry else self._defn_line
            col = first_sorry["col"] if first_sorry else self._defn_col
            raise RuleFilterError(
                f"cannot decide: rule contains declared sorry hole(s): {', '.join(self._manifest)}",
                line=line,
                col=col,
            )

        if len(args) != self._arity:
            raise RuleFilterError(
                f"'filter-rule' expects {self._arity} argument(s), got {len(args)}",
                line=self._defn_line,
                col=self._defn_col,
            )

        addr = self._compiled.engine.get_function_address("filter-rule")
        if not addr:
            raise RuleFilterError(
                "failed to resolve machine address for 'filter-rule'",
                line=self._defn_line,
                col=self._defn_col,
            )

        c_fn = ctypes.CFUNCTYPE(self._restype, *self._argtypes)(addr)
        call_args: list[Any] = [
            a.encode("utf-8") if (isinstance(a, str) and at is ctypes.c_char_p) else a
            for a, at in zip(args, self._argtypes, strict=True)
        ]
        raw_result = c_fn(*call_args)
        return bool(raw_result)

    def verify(
        self, cases: Sequence[tuple[Any, bool]]
    ) -> list[tuple[Any, bool, Any, bool]]:
        """Perform differential testing between native JIT execution and the reference interpreter.

        Args:
            cases: A sequence of `(args_tuple, expected_bool)` test vectors.

        Returns:
            A list of `(args, expected, interpreted, native)` tuples for any mismatched case.
            An empty list indicates complete agreement across all cases.
        """
        mismatches: list[tuple[Any, bool, Any, bool]] = []
        for args_case, expected_bool in cases:
            args_tuple = tuple(args_case) if isinstance(args_case, (tuple, list)) else (args_case,)
            native_res = self.decide(*args_tuple)

            env = Environment()
            # Interpreter-side serialization: Python bools render as the
            # Netelpro literals `true`/`false` (str(True) would emit 'True',
            # which is not a token of the language); Python strings render as
            # quoted literals with the language's own escape rules.
            def _render(a: Any) -> str:
                if a is True:
                    return "true"
                if a is False:
                    return "false"
                if isinstance(a, str):
                    escaped = a.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
                    return f'"{escaped}"'
                return str(a)

            args_str = " ".join(_render(a) for a in args_tuple)
            call_form = f"(filter-rule {args_str})" if args_str else "(filter-rule)"
            interp_src = f"{self._source}\n{call_form}"
            interp_res = run_source(interp_src, env=env)

            if (native_res != interp_res) or (native_res != expected_bool) or (interp_res != expected_bool):
                mismatches.append((args_case, expected_bool, interp_res, native_res))

        return mismatches


def compile_filter(source: str) -> RuleFilter:
    """Convenience factory to parse, validate, and compile a Netelpro gate rule.

    Args:
        source: Netelpro source text containing `(defn filter-rule ...)`.

    Returns:
        A compiled `RuleFilter` ready for host evaluation.
    """
    return RuleFilter(source)


class RuleBuilder(RuleFilter):
    """Compiled native Netelpro builder bridge: string products at the boundary.

    v0.5 capability, exclusive to `(defn build-rule ...)` -- the only defn whose
    return type may be Str (i8* into a per-call-reset 64KB arena). Gate rules
    (`filter-rule`) remain prohibited from returning strings: they decide, they
    never produce.

    Raises:
        RuleFilterError: On any prosecutorial failure (parse, caps, holes,
            codegen), on arena overflow (NULL product), or if `build-rule` is
            missing from the source.
    """

    def __init__(self, source: str) -> None:
        """Compile a builder program immediately to native machine code.

        Verifies that the source defines `build-rule` and that its return
        type resolved to Str; a builder without a string product is a
        prosecutorial error (the capability has a purpose or it does not exist).
        """
        RuleFilter.__init__(self, source, defn_name="build-rule")
        builder_defn = self._entry_defn
        # The builder's compiled return type must be a pointer (Str product).
        # A builder returning Int/Bool is a semantic misuse: prosecute it.
        llvm_fn = None
        if self._compiled is not None:
            for fn in self._compiled.module.functions:
                if fn.name == "build-rule":
                    llvm_fn = fn
                    break
        if llvm_fn is None or llvm_fn.function_type.return_type != ir.PointerType():
            raise RuleFilterError(
                "'build-rule' must return Str (it exists to produce rule text)",
                line=builder_defn.line,
                col=builder_defn.col,
            )

    def build(self, *args: int | bool | str) -> str:
        """Call the compiled `build-rule` natively and return its string product.

        The product is read up to NUL from the returned i8* (per-call arena),
        decoded as UTF-8, and returned as a Python str. A NULL product (arena
        overflow) is prosecuted as a bridge error mentioning both 'arena' and
        'overflow'.
        """
        if self._compiled is None:
            raise RuleFilterError(
                "cannot build: rule contains declared sorry hole(s): " + ", ".join(self._manifest)
            )
        if len(args) != self._arity:
            raise RuleFilterError(
                f"'build-rule' expects {self._arity} argument(s), got {len(args)}",
                line=self._defn_line,
                col=self._defn_col,
            )
        addr = self._compiled.engine.get_function_address("build-rule")
        if not addr:
            raise RuleFilterError(
                "failed to resolve machine address for 'build-rule'",
                line=self._defn_line,
                col=self._defn_col,
            )
        c_fn = ctypes.CFUNCTYPE(ctypes.c_char_p, *self._argtypes)(addr)
        call_args: list[Any] = [
            a.encode("utf-8") if isinstance(a, str) else a for a in args
        ]
        raw = c_fn(*call_args)
        try:
            return read_arena_string(raw)
        except StrNativeError as e:
            raise RuleFilterError(
                f"arena overflow: {e.message}",
                line=self._defn_line,
                col=self._defn_col,
            ) from e

    def verify_build(
        self, cases: Sequence[tuple[Any, str]]
    ) -> list[tuple[Any, str, Any, str]]:
        """Differential testing: native builder vs reference interpreter.

        Args:
            cases: Sequence of `(args_tuple, expected_str)` vectors.

        Returns:
            Mismatches as `(args, expected, interpreter_result, native_result)`
            tuples; empty list means full parity across all cases.
        """
        mismatches: list[tuple[Any, str, Any, str]] = []
        for args_case, expected in cases:
            args_tuple = tuple(args_case) if isinstance(args_case, (tuple, list)) else (args_case,)

            native_res = self.build(*args_tuple)

            def _render(a: Any) -> str:
                if a is True:
                    return "true"
                if a is False:
                    return "false"
                if isinstance(a, str):
                    escaped = (
                        a.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
                    )
                    return f'"{escaped}"'
                return str(a)

            args_str = " ".join(_render(a) for a in args_tuple)
            call_form = f"(build-rule {args_str})" if args_str else "(build-rule)"
            interp_src = f"{self._source}\n{call_form}"
            interp_res = run_source(interp_src, env=Environment())

            if (native_res != interp_res) or (native_res != expected):
                mismatches.append((args_case, expected, interp_res, native_res))

        return mismatches


__all__ = [
    "RuleFilter",
    "RuleFilterError",
    "compile_filter",
]
