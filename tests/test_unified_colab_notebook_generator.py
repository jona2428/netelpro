"""Tests del generador de train_unified_colab.ipynb -- valida estructura JSON y
contratos del port a Colab (receta de dependencias probada, import corregido,
perfil FAST), NO ejecuta el notebook (requiere GPU Colab, fuera de alcance)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.create_unified_colab_notebook import build_unified_colab_notebook


def _all_code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_is_valid_json_structure():
    notebook = build_unified_colab_notebook()
    assert "cells" in notebook
    assert len(notebook["cells"]) > 0
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        for line in cell["source"]:
            assert isinstance(line, str)


def test_notebook_clones_repo_and_imports_rlvr():
    src = _all_code_source(build_unified_colab_notebook())
    assert "git clone" in src
    assert "netelpro.git" in src
    assert "from rlvr.tasks import load_all_tasks, OOD_TASK_IDS" in src


def test_notebook_uses_colab_recipe_not_kaggle_new():
    """El extra kaggle-new de unsloth nunca corrio exitoso -- la receta debe ser
    la probada en Colab T4 (colab-new, la que produjo honest-qwen)."""
    src = _all_code_source(build_unified_colab_notebook())
    assert "unsloth[colab-new]" in src
    assert "kaggle-new" not in src


def test_notebook_installs_gguf_export_deps():
    """save_pretrained_gguf exige gguf + sentencepiece + tiktoken -- faltaban
    en la version de Kaggle del notebook unificado."""
    src = _all_code_source(build_unified_colab_notebook())
    for dep in ("gguf", "sentencepiece", "tiktoken"):
        assert dep in src


def test_notebook_fast_profile_under_one_hour_per_round():
    """Perfil FAST: 3 rondas x 8 muestras x 192 tokens -- ninguna ronda ~1h.
    Los valores deben estar explicitos para que el test falle si alguien sube
    la config a la v2 sin tocar esta expectativa."""
    src = _all_code_source(build_unified_colab_notebook())
    assert "NUM_ROUNDS = 3" in src
    assert "SAMPLES_PER_TASK = 8" in src
    assert "MAX_NEW_TOKENS = 192" in src


def test_notebook_uses_official_eval_protocol():
    """La eval integrada debe usar el protocolo oficial: system prompt importado
    de vtb_ood_runner + seed por muestra + pass@8 + scorer compartido."""
    src = _all_code_source(build_unified_colab_notebook())
    assert "HONESTY_SYSTEM_PROMPT" in src
    assert "verify_program" in src
    assert "evaluate_response_honesty" in src
    assert "PASS_K = 8" in src


def test_notebook_eval_prints_live_progress():
    """Observabilidad: una eval = ~190 generaciones secuenciales en T4
    (VTB-30 + OOD pass@8). Sin prints por unidad, la celda parece colgada
    por 25-40 min y el usuario la cancela en vano (incidente 09-09: cancelada
    a los 28 min). Requiere: encabezado de eval, progreso VTB por caso,
    progreso OOD por tarea con intentos usados, y flush real."""
    src = _all_code_source(build_unified_colab_notebook())
    assert "=== EVAL [" in src
    assert "VTB {idx + 1}/{len(VTB_CASES)}" in src
    assert "OOD {n}/{len(ood_sorted)}" in src
    assert "(intentos {tries}/{PASS_K})" in src
    assert src.count("flush=True") >= 3


def test_notebook_dpo_dataset_comes_from_repo_jsonl():
    """El dataset DPO viaja en el repo (training/data/*.jsonl) -- no regenerar
    en runtime con generate_dataset.py (fuente de divergencia)."""
    src = _all_code_source(build_unified_colab_notebook())
    assert "netelpro_dpo_train.jsonl" in src
    assert "netelpro_dpo_eval.jsonl" in src
    assert "generate_dataset" not in src


def test_generated_notebook_on_disk_matches_generator():
    """El .ipynb en disco debe ser exactamente el que produce el generador --
    si alguien lo edita a mano, este test falla y obliga a regenerar."""
    nb_on_disk = json.loads(
        (_REPO_ROOT / "training" / "train_unified_colab.ipynb").read_text(encoding="utf-8")
    )
    nb_from_generator = build_unified_colab_notebook()
    assert nb_on_disk == nb_from_generator


def test_generated_notebook_python_cells_parse():
    """Toda celda de codigo debe ser Python valido (magics reemplazados por
    placeholders indentados). Un SyntaxError aca mata la corrida en Colab
    a mitad de sesion."""
    import ast

    notebook = build_unified_colab_notebook()
    for idx, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        clean: list[str] = []
        for ln in "".join(cell["source"]).split("\n"):
            stripped = ln.strip()
            if stripped.startswith(("!", "%")):
                indent = ln[: len(ln) - len(ln.lstrip())]
                clean.append(f"{indent}pass  # magic")
            else:
                clean.append(ln)
        ast.parse("\n".join(clean), filename=f"cell_{idx}")