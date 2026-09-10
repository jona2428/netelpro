"""Tests del generador train_raft_lfm_kaggle.ipynb (run #3, eje A).

El notebook fuente (train_raft_kaggle.ipynb) es el probado que entreno los
Qwen RAFT en Kaggle; el generador lo transforma a base LFM2.5-1.2B con la
lista LoRA oficial de la arquitectura hibrida. Estos tests congelan esa
transformacion: si alguien edita el fuente o el generador y una ancla
desaparece, la regeneracion debe fallar (no producir un notebook roto).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1] / "training"
GENERATOR = TRAINING / "create_raft_lfm_kaggle_notebook.py"
OUTPUT = TRAINING / "train_raft_lfm_kaggle.ipynb"

LFM_BASE = "LiquidAI/LFM2.5-1.2B-Instruct"
LFM_TARGET_MODULES = 'target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]'


def _load_generator():
    spec = importlib.util.spec_from_file_location("create_raft_lfm_kaggle_notebook", GENERATOR)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _notebook_text() -> str:
    nb = json.loads(OUTPUT.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"]
    )


def test_output_es_json_valido_con_misma_estructura_del_fuente() -> None:
    nb = json.loads(OUTPUT.read_text(encoding="utf-8"))
    src = json.loads((TRAINING / "train_raft_kaggle.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == src["nbformat"] == 4
    assert len(nb["cells"]) == len(src["cells"]) == 25


def test_base_lfm_y_target_modules_oficiales_presentes() -> None:
    text = _notebook_text()
    assert f'MODEL_NAME = "{LFM_BASE}"' in text
    assert LFM_TARGET_MODULES in text
    assert "in_proj" in text  # cubre los 10 bloques conv de la arquitectura hibrida


def test_sin_residuos_qwen_en_codigo() -> None:
    text = _notebook_text()
    assert "Qwen/Qwen2.5-1.5B-Instruct" not in text
    assert "netelpro_qwen1.5b_raft" not in text
    assert '"gate_proj"' not in text  # la lista transformer no debe sobrevivir en el branch LFM


def test_regeneracion_deterministica() -> None:
    mod = _load_generator()
    before = OUTPUT.read_text(encoding="utf-8")
    mod.main()
    after = OUTPUT.read_text(encoding="utf-8")
    assert before == after, "el generador no es deterministico"


def test_ancla_ausente_aborta_en_lugar_de_producir_roto() -> None:
    mod = _load_generator()
    original = mod.OLD_MODEL_LINE
    try:
        mod.OLD_MODEL_LINE = "ancla_inexistente_para_el_test"
        import pytest

        with pytest.raises(AssertionError):
            mod.main()
    finally:
        mod.OLD_MODEL_LINE = original