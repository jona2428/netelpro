"""Tests de examples/gates/*.sl: cada regla de dominio se compila via el
Gate generico y se verifica contra su tabla de verdad documentada.

Los examples no son decoracion: si una regla y su contrato divergen, el
test falla. El espacio probado es la proyeccion completa de sus flags
(0/1) con puntos de borde numericos de la tabla.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from netelpro.gate import Gate, check_file  # noqa: E402

_GATES_DIR = _REPO_ROOT / "examples" / "gates"


def _gate(name: str, tmp_path: Path) -> Gate:
    src = (_GATES_DIR / name).read_text(encoding="utf-8")
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return Gate(p)


def _truth_rows(gate: Gate, rows: list[tuple], cols: tuple[int, ...]) -> None:
    """Verifica (args..., expected) contra gate.check para cada fila."""
    for row in rows:
        args, expected = row[:-1], row[-1]
        allow, reason = gate.check(*args)
        assert reason is None, f"{args}: fallo del gate, no decision: {reason}"
        assert allow == bool(expected), f"{args}: esperado {expected}, obtuvo {allow}"


def test_expense_approval_full_flag_space(tmp_path: Path):
    """8/8 combinaciones de flags con bordes numericos de la tabla."""
    gate = _gate("expense_approval.sl", tmp_path)
    rows: list[tuple] = []
    for emergency in (0, 1):
        for manager in (0, 1):
            if manager == 1:
                rows += [(0, 1, emergency, 1), (500, 1, emergency, 1), (501, 1, emergency, 0)]
            if emergency == 1 and manager == 0:
                rows += [(50, 0, 1, 1), (51, 0, 1, 0)]
    _truth_rows(gate, rows, (0, 1))
    # Bordes puros de la tabla documentada:
    assert gate.check(500, 1, 0) == (True, None)
    assert gate.check(501, 1, 0) == (False, None)
    assert gate.check(50, 0, 1) == (True, None)
    assert gate.check(51, 0, 1) == (False, None)
    assert gate.check(500, 0, 0) == (False, None)


def test_content_moderation_boundary_and_flags(tmp_path: Path):
    gate = _gate("content_moderation.sl", tmp_path)
    assert gate.check(69, 0, 0) == (True, None)
    assert gate.check(70, 0, 1) == (False, None)
    assert gate.check(10, 3, 0) == (False, None)
    assert gate.check(10, 3, 1) == (True, None)
    assert gate.check(69, 0, 1) == (True, None)
    for toxicity, reports, verified in product((0, 69), (0, 3), (0, 1)):
        expected = toxicity < 70 and (reports == 0 or verified == 1)
        assert gate.check(toxicity, reports, verified) == (expected, None)


def test_robot_interlock_estop_dominates(tmp_path: Path):
    """Estop latched => nunca PERMIT (dominancia de seguridad)."""
    gate = _gate("robot_interlock.sl", tmp_path)
    assert gate.check(0, 100, 0) == (True, None)
    assert gate.check(1, 100, 0) == (False, None)
    assert gate.check(1, 0, 0) == (True, None)
    for door in (0, 1):
        for speed in (0, 100):
            assert gate.check(door, speed, 1) == (False, None)


def test_examples_work_one_shot_from_repo_paths(tmp_path: Path):
    """Un host externo: check_file directo sobre el .sl del repo."""
    rule = _GATES_DIR / "robot_interlock.sl"
    assert check_file(str(rule), 0, 100, 0) == (True, None)
    assert check_file(str(rule), 1, 100, 0)[0] is False