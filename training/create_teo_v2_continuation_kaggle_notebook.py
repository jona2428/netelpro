"""Generador del notebook train_teo_v2_continuation_kaggle.ipynb para Kaggle.

Notebook de continuación y entrenamiento masivo (Fase 2) para Teo v2 (Netelpro SBT):
- Reanuda automáticamente desde el checkpoint guardado en la Fase 1.
- Construye el corpus masivo de 250M tokens (Cosmopedia v2 + OpenAssistant + Rust/C++/C#/Python + Razonamiento).
- Entrenamiento profundo con GPU T4 hasta 50,000 - 100,000 pasos.
- Inferencia en vivo y empaquetado del checkpoint final para descarga.
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
OUTPUT_NOTEBOOK = TRAINING_DIR / "train_teo_v2_continuation_kaggle.ipynb"


def build_notebook() -> dict:
    cells = [
        # Celda 1: Portada
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Teo v2 (Netelpro SBT): Fase 2 - Entrenamiento Masivo y Reanudación Profunda\n",
                "### *Cosmopedia v2 + OpenAssistant v2 + Sistemas (Rust/C++/C#/Python) + Razonamiento*\n",
                "\n",
                "Este notebook ejecuta la **Fase 2 de entrenamiento continuo** para el modelo **Teo v2 (Netelpro SBT-124M)**.\n",
                "\n",
                "### 🎯 Objetivos de la sesión:\n",
                "1. **Reanudación transparente:** Toma el checkpoint existente (`checkpoint.pt`) de la Fase 1 y continúa exactamente donde quedó el descenso de gradiente.\n",
                "2. **Corpus Masivo:** Empaqueta 250 Millones de tokens con libros de texto científicos (Cosmopedia v2), diálogos de múltiples turnos y proyectos de sistemas.\n",
                "3. **Inferencia en vivo:** Evaluación directa de razonamiento paso a paso (`<|thought|> ... <|endthought|>`).\n",
                "\n",
                "---\n",
                "### ⚙️ Configuración en Kaggle:\n",
                "- **Accelerator:** GPU T4 x2 (o P100)\n",
                "- **Internet:** **On**\n",
                "- **Persistence:** Variables / Files on\n",
            ],
        },
        # Celda 2: Hardware
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Verificación de Hardware y GPU T4"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi\n",
                "import torch\n",
                "print(f'CUDA: {torch.cuda.is_available()} | Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')\n",
            ],
        },
        # Celda 3: Repositorio
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Actualización del Repositorio Netelpro"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /kaggle/working\n",
                "import os\n",
                "if not os.path.exists('netelpro'):\n",
                "    !git clone https://github.com/jona2428/netelpro.git\n",
                "else:\n",
                "    !cd /kaggle/working/netelpro && git pull origin master\n",
                "%cd /kaggle/working/netelpro\n",
                "\n",
                "!pip install -q tokenizers datasets pyarrow requests\n",
            ],
        },
        # Celda 4: Construcción del Corpus Masivo (250M tokens)
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Generación del Corpus Masivo (250 Millones de Tokens)\n",
                "Descarga y empaqueta en streaming Cosmopedia v2, OpenAssistant y código de sistemas a ~150,000 tokens/s."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python training/data/build_massive_corpus.py \\\n",
                "    --out-dir data/teo_v2_massive \\\n",
                "    --tokenizer-path data/teo_v2/tokenizer.json \\\n",
                "    --target-shards 10 \\\n",
                "    --shard-size 25000000 \\\n",
                "    --persona-multiplier 100\n",
            ],
        },
        # Celda 5: Entrenamiento Continuo con Auto-Resume
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Entrenamiento Continuo de Teo v2 (Netelpro SBT)\n",
                "Detecta automáticamente el último `checkpoint.pt` y continúa el entrenamiento sin reiniciar."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "os.makedirs('models/teo_v2', exist_ok=True)\n",
                "\n",
                "# Si subiste un checkpoint de una sesión anterior como Kaggle Dataset:\n",
                "# !cp /kaggle/input/teo-v2-checkpoint/checkpoint.pt models/teo_v2/checkpoint.pt\n",
                "\n",
                "!python training/train_teo_v2.py \\\n",
                "    --data-dir data/teo_v2_massive \\\n",
                "    --checkpoint-dir models/teo_v2 \\\n",
                "    --device cuda \\\n",
                "    --autocast \\\n",
                "    --batch-size 8 \\\n",
                "    --context-len 512 \\\n",
                "    --lr 2.5e-4 \\\n",
                "    --save-interval 500 \\\n",
                "    --log-interval 50\n",
            ],
        },
        # Celda 6: Inferencia con Pensamiento en Vivo
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Prueba de Inferencia y Razonamiento en Vivo con Teo"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "from netelpro.neuro.tokenizer_bpe import NetelproBPETokenizer\n",
                "from training.train_teo_v2 import TeoV2Config, TeoV2Transformer, load_checkpoint\n",
                "\n",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "tok = NetelproBPETokenizer.load('data/teo_v2/tokenizer.json')\n",
                "cfg = TeoV2Config.from_json('training/teo_v2_config.json')\n",
                "cfg.context = 512\n",
                "\n",
                "model = TeoV2Transformer(cfg).to(device)\n",
                "load_checkpoint('models/teo_v2/checkpoint.pt', model=model, device=device)\n",
                "model.eval()\n",
                "print('✅ Modelo Netelpro SBT cargado exitosamente.')\n",
                "\n",
                "test_prompts = [\n",
                "    '<|user|>\\nHola Teo, soy Jona tu creador. ¿Cómo operan tus compuertas en silicio?<|assistant|>\\n',\n",
                "    '<|user|>\\nExplícame cómo funciona un Mutex en Rust con un ejemplo.<|thought|>\\n',\n",
                "]\n",
                "\n",
                "for p in test_prompts:\n",
                "    print('=' * 60)\n",
                "    print('PROMPT:', p)\n",
                "    input_ids = torch.tensor([tok.encode(p)], device=device)\n",
                "    with torch.no_grad():\n",
                "        for _ in range(150):\n",
                "            logits = model(input_ids)[:, -1, :]\n",
                "            next_tok = torch.argmax(logits, dim=-1, keepdim=True)\n",
                "            input_ids = torch.cat([input_ids, next_tok], dim=1)\n",
                "            if next_tok.item() == tok.eos_token_id:\n",
                "                break\n",
                "    print('RESPUESTA:', tok.decode(input_ids[0].tolist()))\n",
            ],
        },
        # Celda 7: Empaquetar
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Empaquetar Checkpoint Final de la Fase 2"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!tar -czvf /kaggle/working/teo_v2_phase2_checkpoint.tar.gz -C models/teo_v2 checkpoint.pt\n",
                "print('🎉 Checkpoint Fase 2 empaquetado en /kaggle/working/teo_v2_phase2_checkpoint.tar.gz')\n",
            ],
        },
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.10.12"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return notebook


def main() -> None:
    nb = build_notebook()
    with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated continuation notebook at: {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
