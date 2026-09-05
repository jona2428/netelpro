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
   Parameters of `filter-rule` resolve statically to `Int` (i64) or `Bool` (i1) by
   bidirectional type inference, driven by how each parameter is used: an unused or
   Int-context parameter binds to `Int`; a parameter used in a boolean context
   (`if`/`and`/`or`/`not` operands) binds to `Bool`. Mixed use (Bool demanded where
   Int is required, or vice versa) is a compile error with exact coordinates.
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
from typing import Any, Sequence

import llvmlite.ir as ir

from netelpro.ast_nodes import Defn, Sym
from netelpro.caps import check_capabilities, collect_grants
from netelpro.codegen import CodegenError, CompiledProgram, compile_program
from netelpro.evaluator import Environment, StrayError, run_source
from netelpro.holes import check_holes
from netelpro.parser import parse


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

    def __init__(self, source: str) -> None:
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
        filter_defn: Defn | None = None
        for form in program.forms:
            if isinstance(form, Defn):
                name = form.name.name if isinstance(form.name, Sym) else str(form.name)
                if name == "filter-rule":
                    filter_defn = form
                    break

        if filter_defn is None:
            raise RuleFilterError("function 'filter-rule' is not defined in source", line=0, col=0)

        self._defn_line: int = filter_defn.line
        self._defn_col: int = filter_defn.col
        self._arity: int = len(filter_defn.params)
        self._param_names: list[str] = [
            p.name if isinstance(p, Sym) else str(p) for p in filter_defn.params
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
                if fn.name == "filter-rule":
                    llvm_fn = fn
                    break

            if llvm_fn is None:
                raise RuleFilterError(
                    "compiled function 'filter-rule' not found in native module",
                    line=self._defn_line,
                    col=self._defn_col,
                )

            # Per-parameter type audit against the codegen-resolved signature.
            # Params are Int (i64) or Bool (i1); anything else is a bridge bug
            # and is prosecuted as one.
            param_arg_types: list[int] = [arg.type.width for arg in llvm_fn.args]
            for arg_width in param_arg_types:
                if arg_width not in (64, 1):
                    raise RuleFilterError(
                        f"parameter of 'filter-rule' resolved to unsupported type i{arg_width}",
                        line=self._defn_line,
                        col=self._defn_col,
                    )

            self._compiled = compiled
            if llvm_fn.function_type.return_type == ir.IntType(1):
                self._restype = ctypes.c_bool
            else:
                self._restype = ctypes.c_int64

            # Per-parameter ctypes prototype: i1 params cross the boundary as
            # c_bool (ctypes reads only the low byte, per the ABI finding in
            # the module docstring), i64 params as c_int64.
            self._argtypes: list[type[ctypes._SimpleCData]] = [
                ctypes.c_bool if width == 1 else ctypes.c_int64 for width in param_arg_types
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
        raw_result = c_fn(*args)
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
            # Interpreter-side serialization: Python bools must render as the
            # Netelpro literals `true`/`false` (str(True) would emit 'True',
            # which is not a token of the language).
            args_str = " ".join(
                ("true" if a is True else "false" if a is False else str(a)) for a in args_tuple
            )
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


__all__ = [
    "RuleFilter",
    "RuleFilterError",
    "compile_filter",
]
