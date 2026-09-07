"""Verificador RLVR: pipeline estático (parse+capabilities+holes) + ejecución
por intérprete contra la referencia Python de la tarea.

Ver docs/superpowers/specs/2026-09-07-rlvr-netelpro-raft-design.md §4 para el
razonamiento completo -- en particular, por qué no se exige compilación
nativa LLVM: no soporta listas (netelpro/codegen.py:354 levanta
CodegenError("List literals are not supported in native codegen v0.1", ...)).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

from netelpro.caps import check_capabilities, collect_grants
from netelpro.evaluator import (
    Environment,
    StrayError,
    StrayHoleError,
    StrayList,
    StrayRuntimeError,
    run_source,
)
from netelpro.holes import check_holes
from netelpro.parser import parse


@dataclass(frozen=True)
class VerifyResult:
    """Resultado de verificar un candidato .sl contra una tarea."""

    compiled: bool
    passed: bool
    cases_total: int
    cases_passed: int
    error: str | None


def _render_literal(value: Any) -> str:
    """Renderiza un valor Python como literal Netelpro (misma técnica que
    RuleFilter.verify() en netelpro/rule_filter.py: bools -> true/false,
    strings escapadas, listas -> (list e1 e2 ...))."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        if not value:
            return "nil"
        return "(list " + " ".join(_render_literal(v) for v in value) + ")"
    return str(value)


def _to_python(value: Any) -> Any:
    """Convierte un valor devuelto por el intérprete (StrayList incluida) a
    un valor Python plano, comparable directo contra la referencia."""
    if isinstance(value, StrayList):
        return tuple(_to_python(v) for v in value)
    return value


def verify_program(
    sl_source: str, task_module: ModuleType, num_cases: int = 20, seed: int = 0
) -> VerifyResult:
    """Verifica un programa Netelpro candidato contra una tarea del corpus.

    Pipeline: (1) parse + capabilities + holes -- "compiló" en el sentido
    correcto para este lenguaje (ver docstring del módulo); si falla ahí,
    compiled=False y no se ejecuta nada. (2) Si compiló, ejecuta vía
    intérprete contra cada caso de gen_inputs() y compara contra reference().
    Binario: pasa TODOS los casos o se descarta (sin partial credit, RAFT
    filtra por diseño).
    """
    parse_result = parse(sl_source)
    if not parse_result.ok:
        first = parse_result.errors[0]
        return VerifyResult(
            compiled=False,
            passed=False,
            cases_total=0,
            cases_passed=0,
            error=f"line {first.line}, col {first.col}: {first.message}",
        )

    program = parse_result.program
    granted = collect_grants(program)
    cap_errors = check_capabilities(program, granted)
    if cap_errors:
        first_cap = cap_errors[0]
        return VerifyResult(
            compiled=False,
            passed=False,
            cases_total=0,
            cases_passed=0,
            error=f"line {first_cap.line}, col {first_cap.col}: {first_cap.message}",
        )

    hole_errors, _manifest = check_holes(program)
    if hole_errors:
        first_hole = hole_errors[0]
        return VerifyResult(
            compiled=False,
            passed=False,
            cases_total=0,
            cases_passed=0,
            error=f"line {first_hole.line}, col {first_hole.col}: {first_hole.message}",
        )

    cases = task_module.gen_inputs(num_cases, seed)
    cases_total = len(cases)
    cases_passed = 0
    last_error: str | None = None

    for args in cases:
        expected = task_module.reference(*args)
        args_text = " ".join(_render_literal(a) for a in args)
        call_form = (
            f"({task_module.FN_NAME} {args_text})"
            if args_text
            else f"({task_module.FN_NAME})"
        )
        candidate_source = f"{sl_source}\n{call_form}"
        try:
            raw_result = run_source(candidate_source, env=Environment())
        except (StrayRuntimeError, StrayHoleError, StrayError) as e:
            last_error = str(e)
            continue

        actual = _to_python(raw_result)
        if actual == expected:
            cases_passed += 1
        else:
            last_error = f"caso {args!r}: esperado {expected!r}, obtuvo {actual!r}"

    passed = cases_total > 0 and cases_passed == cases_total
    return VerifyResult(
        compiled=True,
        passed=passed,
        cases_total=cases_total,
        cases_passed=cases_passed,
        error=None if passed else last_error,
    )


__all__ = ["VerifyResult", "verify_program"]
