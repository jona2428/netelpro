"""Tests del generador de train_raft_colab.ipynb -- valida estructura JSON,
NO ejecuta el notebook (requiere GPU Colab, fuera de alcance de pytest local)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.create_raft_notebook import build_raft_notebook  # noqa: E402


def _all_code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_is_valid_json_structure():
    notebook = build_raft_notebook()
    assert "cells" in notebook
    assert len(notebook["cells"]) > 0
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)


def test_notebook_clones_repo_and_imports_rlvr():
    src = _all_code_source(build_raft_notebook())
    assert "git clone" in src
    assert "netelpro.git" in src
    assert "import rlvr" in src or "from rlvr" in src


def test_notebook_loads_base_model_not_dpo_checkpoint():
    src = _all_code_source(build_raft_notebook())
    assert "Qwen/Qwen2.5-1.5B-Instruct" in src
    assert "netelpro_qwen1.5b_honest" not in src  # el checkpoint DPO -- no se usa (spec §1, §9)


def test_notebook_references_verify_program_and_split():
    src = _all_code_source(build_raft_notebook())
    assert "verify_program" in src
    assert "split_train_ood" in src


def test_notebook_checks_gpu_before_unsloth():
    src = _all_code_source(build_raft_notebook())
    gpu_idx = src.find("torch.cuda.is_available()")
    unsloth_idx = src.find("from unsloth")
    assert gpu_idx != -1, "falta guard de GPU"
    assert unsloth_idx != -1
    assert gpu_idx < unsloth_idx, "el guard de GPU debe correr antes del import de unsloth"


def test_notebook_measures_baseline_before_raft_rounds():
    src = _all_code_source(build_raft_notebook())
    baseline_idx = src.find("baseline")
    raft_loop_idx = src.find("for round_num in range")
    assert baseline_idx != -1
    assert raft_loop_idx != -1
    assert baseline_idx < raft_loop_idx


def test_notebook_exports_gguf_at_the_end():
    src = _all_code_source(build_raft_notebook())
    assert "save_pretrained_gguf" in src


def test_raft_loop_conditions_sft_on_prompt_via_formatting_func():
    """El SFT tiene que condicionar al prompt de la tarea, no entrenar solo
    sobre el texto crudo de la completion (bug de fix round 1: con
    trl<0.15.0, dataset_text_field solo usa esa UNA columna, así que el
    modelo nunca vería el prompt durante el fine-tune)."""
    src = _all_code_source(build_raft_notebook())
    assert 'dataset_text_field="completion"' not in src
    assert "formatting_func=format_sft_example" in src


def test_generator_writes_valid_ipynb_file(tmp_path):
    from training.create_raft_notebook import main as generator_main

    output_path = tmp_path / "train_raft_colab.ipynb"
    generator_main(output_path)
    with open(output_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert "cells" in loaded
