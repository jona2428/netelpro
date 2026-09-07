"""Arma el prompt fijo de muestreo para el RAFT de Netelpro (ver
docs/superpowers/specs/2026-09-07-rlvr-netelpro-raft-design.md §5).

El mismo prompt, palabra por palabra, se usa en las mediciones del criterio
de éxito (baseline y RAFT, §7) -- es el control, no una variable.
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

_FEW_SHOT_FILES = ("fib.sl", "sum_to.sl")

_SPEC_SUMMARY = """\
Netelpro es un lenguaje tipo Lisp de aridad fija. Formas principales:
  (defn nombre (params...) cuerpo)   -- define una función (un único cuerpo, sin secuencia)
  (let nombre expr cuerpo)           -- liga un nombre, anida lets para varias ligaduras
  (if cond entonces sino)            -- ambas ramas son obligatorias
  (and a b) / (or a b)               -- corto-circuito, aridad 2
Aritmética: (+ a b) (- a b) (* a b) (/ a b) (quot a b) (rem a b)
Comparación: (== a b) (!= a b) (< a b) (<= a b) (> a b) (>= a b) (not a)
Listas: (list a b c ...) (cons x xs) (head xs) (tail xs) (is-nil xs) (len xs) (nth xs i)
Strings: (str-cat a b) (str-len s) (int->str n) (str->int s) (prefix? text prefix)
No hay map/filter/fold, ni indexado de caracteres de string, ni bucles --
todo repetitivo se escribe como recursión (frecuentemente cola-recursiva,
con un acumulador como parámetro extra).
"""


def _load_few_shot_examples() -> str:
    blocks = []
    for filename in _FEW_SHOT_FILES:
        path = _EXAMPLES_DIR / filename
        blocks.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(blocks)


def build_prompt(task_module: ModuleType) -> str:
    """Arma el prompt completo: spec resumida + few-shot + la tarea."""
    few_shot = _load_few_shot_examples()
    return (
        f"{_SPEC_SUMMARY}\n"
        f"Ejemplos de programas Netelpro válidos:\n\n{few_shot}\n\n"
        f"Ahora escribí un programa Netelpro para esta tarea:\n"
        f"{task_module.DESCRIPTION_ES}\n"
        f"Firma esperada: {task_module.SIGNATURE}\n"
    )


__all__ = ["build_prompt"]
