"""Generador de train_raft_lfm_kaggle.ipynb -- Run #3 (eje A): portar el
curriculum RAFT v2 (el mismo loop probado en train_raft_kaggle.ipynb que
entreno los Qwen RAFT/raft-v2) al base LiquidAI LFM2.5-1.2B-Instruct.

Motivacion (ledger f6ae110f): el LFM es el modelo mas descargado del roster
(2:1 sobre el Qwen honest) y el que mejor respondio al DPO de honestidad
(+133% relativo), pero es 0% OOD -- solo voz, no escribe reglas. Este run
equilibra: misma senal de recompensa (compilo + paso los tests), mismo
protocolo congelado (temp 0.8, seed 0, pass@16, 5 rondas, pool acumulado).
Unico cambio experimental: la base.

Diferencias con train_raft_kaggle.ipynb (Qwen):
- MODEL_NAME -> base LFM2.5-1.2B-Instruct, extraido de train_lfm_colab.ipynb
  (el notebook que entreno el modelo publicado) -- no hardcodeado a mano.
- target_modules -> lista oficial Unsloth para la arquitectura hibrida LFM2.5
  (q/k/v/out/in_proj + w1/w2/w3): cubre los 6 bloques GQA Y los 10 bloques
  conv. La lista transformer (q/k/v/o/gate/up/down) en LFM solo matchea 6
  modulos y se salta los conv en silencio (hallazgo auditoria doc Unsloth).
- Nombres de export GGUF -> netelpro_lfm1.2b_raft.

El notebook fuente no se edita a mano: este script lo transforma de forma
deterministica con aserciones -- si el fuente cambia y un ancla desaparece,
aborta en vez de producir un notebook roto.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
SOURCE_NOTEBOOK = TRAINING_DIR / "train_raft_kaggle.ipynb"
LFM_DPO_NOTEBOOK = TRAINING_DIR / "train_lfm_colab.ipynb"
OUTPUT_NOTEBOOK = TRAINING_DIR / "train_raft_lfm_kaggle.ipynb"

EXPORT_DIR = "netelpro_lfm1.2b_raft"

OLD_MODEL_LINE = (
    'MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # base limpio, '
    "NO el checkpoint DPO de honestidad (spec Sec 1, Sec 9)"
)
OLD_TARGET_MODULES = (
    'target_modules=["q_proj", "k_proj", "v_proj", "o_proj", '
    '"gate_proj", "up_proj", "down_proj"],'
)
NEW_TARGET_MODULES = (
    'target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],'
    "  # lista oficial Unsloth LFM2.5: 6 GQA + 10 conv (la lista transformer se salta los conv)"
)


def extract_lfm_base_name() -> str:
    """Base exacto con el que se entreno el LFM publicado (train_lfm_colab.ipynb)."""
    nb = json.loads(LFM_DPO_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        match = re.search(r'[\'"](LiquidAI/[\w.\-]+)[\'"]', src)
        if match:
            return match.group(1)
    raise AssertionError("No se encontro el base LiquidAI/... en train_lfm_colab.ipynb")


def transform_cell(src: str, lfm_base: str) -> str:
    # Orden importa: el reemplazo de MODEL_NAME va ANTES del generico
    # "Qwen2.5-1.5B" -> "LFM2.5-1.2B", si no el anchor queda mutilado
    # ("Qwen/LFM2.5-1.2B-Instruct") y el notebook sale con un repo inexistente.
    new_model_line = (
        f'MODEL_NAME = "{lfm_base}"'
        "  # base LFM2.5-1.2B-Instruct, mismo blanco que el DPO de honestidad publicado"
    )
    src = src.replace(OLD_MODEL_LINE, new_model_line)
    src = src.replace(OLD_TARGET_MODULES, NEW_TARGET_MODULES)
    src = src.replace("netelpro_qwen1.5b_raft", EXPORT_DIR)
    src = src.replace("Qwen2.5-1.5B", "LFM2.5-1.2B")
    return src


def main() -> None:
    lfm_base = extract_lfm_base_name()
    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    # Anclas contra el TEXTO REAL de las celdas (join), no contra json.dumps:
    # en la serializacion JSON las comillas van escapadas (\") y ninguna
    # ancla con comillas matchearia jamas.
    full_text = "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"]
    )
    anchors = {
        "MODEL_NAME": OLD_MODEL_LINE,
        "target_modules": OLD_TARGET_MODULES,
        "export_name": "netelpro_qwen1.5b_raft",
        "titulo": "Qwen2.5-1.5B",
    }
    missing = [name for name, anchor in anchors.items() if anchor not in full_text]
    if missing:
        raise AssertionError(f"Anclas ausentes en el fuente (el notebook cambio?): {missing}")

    for cell in nb["cells"]:
        was_list = isinstance(cell["source"], list)
        joined = "".join(cell["source"]) if was_list else cell["source"]
        new_src = transform_cell(joined, lfm_base)
        cell["source"] = new_src.splitlines(keepends=True) if was_list else new_src

    out_text = "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"]
    )
    for residuo in ("Qwen/Qwen2.5-1.5B-Instruct", "netelpro_qwen1.5b_raft", '"gate_proj"'):
        assert residuo not in out_text, f"Residuo Qwen en el output: {residuo}"
    assert f'MODEL_NAME = "{lfm_base}"' in out_text, "MODEL_NAME LFM ausente"
    assert NEW_TARGET_MODULES.split(",")[0] in out_text, "target_modules LFM ausente"

    leftovers = [ln.strip() for ln in out_text.splitlines() if "Qwen" in ln]
    if leftovers:
        print(f"AVISO: quedan {len(leftovers)} menciones de 'Qwen' (revisar):")
        for ln in leftovers[:5]:
            print(f"  {ln[:120]}")

    OUTPUT_NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_export = out_text.count(EXPORT_DIR)
    print(
        f"OK: {OUTPUT_NOTEBOOK.name} generado -- {len(nb['cells'])} celdas, "
        f"base={lfm_base}, export={EXPORT_DIR} ({n_export} menciones)"
    )


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_NOTEBOOK
    if target != OUTPUT_NOTEBOOK:
        OUTPUT_NOTEBOOK = target  # type: ignore[assignment]
    main()