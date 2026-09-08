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


def test_sft_disables_completion_only_loss_for_unsloth_fork():
    """TRL clasifica el dataset (claves prompt/completion) como prompt-completion
    y activa completion_only_loss=True por default; el fork de Unsloth rechaza
    esa combinación con formatting_func en el init del trainer (ValueError, no
    ejecutó el formatter nunca). False explícito = full-sequence loss (RAFT
    canónico) y formatter desbloqueado."""
    src = _all_code_source(build_raft_notebook())
    assert "completion_only_loss=False" in src


def test_generator_writes_valid_ipynb_file(tmp_path):
    from training.create_raft_notebook import main as generator_main

    output_path = tmp_path / "train_raft_colab.ipynb"
    generator_main(output_path)
    with open(output_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert "cells" in loaded


def test_v2_pool_accumulates_between_rounds():
    """v2: el SFT de cada ronda entrena sobre el pool acumulado (RAFT canónico),
    no solo sobre el harvest fresco -- la v1 entrenaba 8-10 ejemplos nuevos por
    ronda y arriesgaba borrar lo aprendido en rondas previas (plateau ronda 2)."""
    src = _all_code_source(build_raft_notebook())
    assert "all_sft_examples" in src
    assert "all_sft_examples.extend(sft_examples)" in src
    assert "Dataset.from_list(all_sft_examples)" in src
    assert "Dataset.from_list(sft_examples)" not in src  # no re-caer en harvest fresco


def test_v2_evals_are_seeded_paired():
    """v2: evaluate_pass_rate fija la seed antes de muestrear -- baseline,
    rondas y eval final comparten draws del sampler (comparación pareada;
    en la v1 el sampler no seeded dejaba el salto sin poder atribuir)."""
    src = _all_code_source(build_raft_notebook())
    def_idx = src.find("def evaluate_pass_rate")
    seed_idx = src.find("torch.manual_seed(EVAL_SEED)")
    # v3: evaluate_pass_rate devuelve (rate, passed_ids) -- la asignación del
    # baseline pasa a ser una tupla, no un `=` simple.
    baseline_idx = src.find("baseline_pass_rate, baseline_passed_ids = ")
    assert def_idx != -1 and seed_idx != -1 and baseline_idx != -1
    assert def_idx < seed_idx < baseline_idx, "la seed se fija dentro de evaluate_pass_rate, antes del baseline"


def test_v2_levers():
    """Palancas v2 declaradas: 5 rondas, 16 muestras/tarea, logging_steps=1."""
    src = _all_code_source(build_raft_notebook())
    assert "NUM_ROUNDS = 5" in src
    assert "SAMPLES_PER_TASK = 16" in src
    assert "logging_steps=1" in src
    assert "logging_steps=5" not in src
