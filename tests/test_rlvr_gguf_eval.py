"""Tests de rlvr/gguf_eval.py: la herramienta oficial de eval local (ollama)
que produjo la medicion GGUF pass@8 OOD 80% del 2026-09-07.

Todos los tests corren OFFLINE: generate_fn es inyectable y los tests usan
un fake determinista. Nada aqui toca ollama ni la red.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlvr.gguf_eval import evaluate_task, extract_src, pass_rate, run_ood_eval  # noqa: E402
from rlvr.tasks import double_value  # noqa: E402

_GOOD_DOUBLE = "(defn double (x) (+ x x))"


def _identity_loader(task_id: str):
    return {"double_value": double_value}[task_id]


def _fake_prompt(task_module) -> str:
    return f"PROMPT::{task_module.__name__}"


# ---------------------------------------------------------------------------
# extract_src: extraccion de codigo desde la salida cruda del modelo
# ---------------------------------------------------------------------------


def test_extract_src_prefers_netelpro_fence():
    out = "Blabla\n```netelpro\n(defn double (x) (+ x x))\n```\nfin"
    assert extract_src(out) == "(defn double (x) (+ x x))"


def test_extract_src_falls_back_to_any_fence():
    out = "```python\nprint('hola')\n```"
    assert extract_src(out) == "print('hola')"


def test_extract_src_none_without_fence():
    """Salida sin fence: no hay candidato (cuenta como muestra fallida)."""
    assert extract_src("no hay codigo aqui") is None


# ---------------------------------------------------------------------------
# evaluate_task / run_ood_eval con fake determinista
# ---------------------------------------------------------------------------


def test_evaluate_task_counts_passes_with_real_verifier():
    """Programa correcto en muestras con seed par, prose sin fence en impar: 4/8.

    Usa el verificador real (verify_program + casos de double_value): las
    4 muestras con fence correcto pasan los 20 casos; las impares no
    producen candidato y cuentan como fallo.
    """
    calls = []

    def fake_gen(prompt: str, seed: int) -> str:
        calls.append(seed)
        if seed % 2 == 0:
            return f"```netelpro\n{_GOOD_DOUBLE}\n```"
        return "texto sin fence"

    ok = evaluate_task("double_value", _identity_loader, _fake_prompt, fake_gen)
    assert ok == 4
    assert calls == list(range(8))


def test_evaluate_task_all_pass():
    def fake_gen(prompt: str, seed: int) -> str:
        return f"```netelpro\n{_GOOD_DOUBLE}\n```"

    assert evaluate_task("double_value", _identity_loader, _fake_prompt, fake_gen) == 8


def test_run_ood_eval_aggregates_and_pass_rate():
    def fake_gen(prompt: str, seed: int) -> str:
        return f"```netelpro\n{_GOOD_DOUBLE}\n```"

    done = run_ood_eval(
        ["double_value"],
        _identity_loader,
        _fake_prompt,
        fake_gen,
        num_samples=3,
    )
    assert done == {"double_value": 3}
    assert pass_rate(done) == pytest.approx(1.0)


def test_pass_rate_semantics():
    """pass@k por tarea: resuelta si >=1 muestra pasa; 0 tareas -> 0.0."""
    assert pass_rate({"a": 8, "b": 0}) == pytest.approx(0.5)
    assert pass_rate({"a": 0, "b": 0}) == pytest.approx(0.0)
    assert pass_rate({}) == pytest.approx(0.0)


def test_checkpoint_skips_already_done_tasks():
    """Tarea en checkpoint: no se regenera (0 llamadas a generate_fn).

    Guard de la propiedad que protegio la medicion real: si el proceso se
    corta a mitad, re-ejecutar no repite lo ya evaluado.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.json"
        ckpt.write_text(json.dumps({"double_value": 8}))
        calls = []

        def fake_gen(prompt: str, seed: int) -> str:
            calls.append(seed)
            return f"```netelpro\n{_GOOD_DOUBLE}\n```"

        done = run_ood_eval(
            ["double_value"],
            _identity_loader,
            _fake_prompt,
            fake_gen,
            ckpt_path=ckpt,
        )
        assert done == {"double_value": 8}
        assert calls == []


def test_checkpoint_written_incrementally():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.json"

        def fake_gen(prompt: str, seed: int) -> str:
            return f"```netelpro\n{_GOOD_DOUBLE}\n```"

        run_ood_eval(
            ["double_value"],
            _identity_loader,
            _fake_prompt,
            fake_gen,
            num_samples=2,
            ckpt_path=ckpt,
        )
        assert json.loads(ckpt.read_text()) == {"double_value": 2}