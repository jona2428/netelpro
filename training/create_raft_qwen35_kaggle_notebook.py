"""Generador de train_raft_qwen35_kaggle.ipynb -- Run Qwen3.5-2B (Instruct).

Porta el curriculum RAFT v2 al modelo de nueva generación Qwen3.5-2B (Instruct)
usando 'unsloth/Qwen3.5-2B' en 4-bit sobre Kaggle (GPU T4 x2 / P100).
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
SOURCE_NOTEBOOK = TRAINING_DIR / "train_raft_kaggle.ipynb"
OUTPUT_NOTEBOOK = TRAINING_DIR / "train_raft_qwen35_kaggle.ipynb"

TARGET_MODEL = "unsloth/Qwen3.5-2B"
EXPORT_DIR = "netelpro_qwen35_2b_raft"


def main() -> None:
    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

        # Reemplazar modelo base específicamente primero
        if "MODEL_NAME =" in src and "Qwen2.5-1.5B" in src:
            src = src.replace(
                'MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # base limpio, NO el checkpoint DPO de honestidad (spec Sec 1, Sec 9)',
                f'MODEL_NAME = "{TARGET_MODEL}"  # Qwen 3.5 2B Instruct (4-bit optimizado para Unsloth en Kaggle)',
            )

        # Reemplazar menciones de Qwen2.5-1.5B por Qwen3.5-2B en títulos y textos
        src = src.replace("Qwen2.5-1.5B", "Qwen3.5-2B")

        # Reemplazar directorio de exportación GGUF
        if "netelpro_qwen1.5b_raft" in src:
            src = src.replace("netelpro_qwen1.5b_raft", EXPORT_DIR)

        # Parchear sample_completions para modelos multimodales (VL de Qwen3.5)
        # Qwen3.5 carga un Qwen3VLProcessor cuyo 1er argumento posicional es `images`, no `text`.
        # Extraemos el tokenizer de texto interno con getattr(tokenizer, "tokenizer", tokenizer)
        old_inputs = 'inputs = tokenizer([chat] * n, return_tensors="pt", padding=True).to("cuda")'
        new_inputs = (
            'text_tok = getattr(tokenizer, "tokenizer", tokenizer)\n'
            '    inputs = text_tok([chat] * n, return_tensors="pt", padding=True).to("cuda")'
        )
        if old_inputs in src:
            src = src.replace(old_inputs, new_inputs)
            src = src.replace("texts = tokenizer.batch_decode(", "texts = text_tok.batch_decode(")

        # Reconstruir source respetando si era lista o string
        if isinstance(cell["source"], list):
            cell["source"] = [line + "\n" for line in src.split("\n")][:-1]
            if src and not src.endswith("\n"):
                cell["source"].append(src.split("\n")[-1])
        else:
            cell["source"] = src

    OUTPUT_NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Generado exitosamente: {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
