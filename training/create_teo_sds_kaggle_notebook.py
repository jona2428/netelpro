"""Generador del notebook train_teo_sds_kaggle.ipynb para Kaggle.

Fase 3 del roadmap Teo-SDS 7B MoE: entrena el motor no-transformer Netelpro
Silicon Dynamic State (netelpro/neuro/dynamic_state.py) sobre el corpus masivo
compilado en la Fase 2 (ver training/create_massive_corpus_kaggle_notebook.py).

A diferencia del notebook de compilación del corpus, este SÍ entrena — usa
GPU T4 en serio, monta el corpus ya empaquetado como Kaggle Dataset (no
vuelve a golpear HuggingFace), y no debería toparse con el crash de
PyGILState de `datasets` porque no hace streaming de red durante el
entrenamiento.
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
OUTPUT_NOTEBOOK = TRAINING_DIR / "train_teo_sds_kaggle.ipynb"


def build_notebook() -> dict:
    cells = [
        # Celda 1: Portada
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ⚡ Netelpro: Fase 3 - Entrenamiento de Teo-SDS (Silicon Dynamic State)\n",
                "### *Motor no-transformer, memoria O(1), cero KV-Cache*\n",
                "\n",
                "Entrena el motor **Netelpro SDS** (`netelpro/neuro/dynamic_state.py`) — reemplaza\n",
                "la atención de Teo v2 por un espacio de estado dinámico (tipo Mamba/SSM) con\n",
                "compuertas de silicio formales (STE, banda `[-5000, 5000]`, fail-closed real: si\n",
                "`control_flag=0`, toda la celda colapsa a 0.0, no solo el estado interno).\n",
                "\n",
                "Config por defecto (Fase 3 del roadmap): 16 capas, `d_model=1024`, `d_state=16`,\n",
                "`expand=2` (`d_inner=2048`), vocabulario 32.768 — ~300-350M parámetros.\n",
                "\n",
                "---\n",
                "### ⚙️ Configuración en Kaggle:\n",
                "- **Accelerator:** GPU T4 x2 (o P100)\n",
                "- **Internet:** On (para clonar el repo — el corpus se monta como Dataset, no se\n",
                "  vuelve a descargar de HuggingFace)\n",
                "- **Persistence:** Variables / Files on\n",
                "- **Input:** montá acá tu Kaggle Dataset con el corpus masivo empaquetado\n",
                "  (`Add Input` -> el dataset que subiste con `massive_corpus_partial.tar.gz`)\n",
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
            "source": [
                "## 2. Clonar/Actualizar Repositorio Netelpro (privado)\n",
                "Mismo patrón que el notebook del corpus: `GITHUB_TOKEN` vía Kaggle Secrets\n",
                "(`Add-ons -> Secrets`) o pegado manual abajo."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /kaggle/working\n",
                "import os\n",
                "\n",
                "GITHUB_TOKEN = \"\"  # <-- si no usás Kaggle Secrets, pegá tu token acá\n",
                "\n",
                "if not GITHUB_TOKEN:\n",
                "    try:\n",
                "        from kaggle_secrets import UserSecretsClient\n",
                "        GITHUB_TOKEN = UserSecretsClient().get_secret(\"GITHUB_TOKEN\")\n",
                "    except Exception:\n",
                "        GITHUB_TOKEN = \"\"\n",
                "\n",
                "REPO_URL = (\n",
                "    f\"https://{GITHUB_TOKEN}@github.com/jona2428/netelpro.git\"\n",
                "    if GITHUB_TOKEN\n",
                "    else \"https://github.com/jona2428/netelpro.git\"\n",
                ")\n",
                "\n",
                "if not os.path.exists('netelpro'):\n",
                "    !git clone {REPO_URL}\n",
                "else:\n",
                "    !cd /kaggle/working/netelpro && git pull origin master\n",
                "%cd /kaggle/working/netelpro\n",
                "\n",
                "!pip install -q tokenizers numpy\n",
            ],
        },
        # Celda 4: Montar corpus
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Montar el Corpus Masivo (Kaggle Dataset)\n",
                "Ajustá `CORPUS_DATASET_SLUG` al nombre del Dataset que subiste (el `.tar.gz`\n",
                "generado por `build_massive_corpus_kaggle.ipynb`). Si lo montaste con\n",
                "`Add Input` en vez de subir el tar suelto, ajustá el path de origen."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "CORPUS_DATASET_SLUG = \"teo-sds-massive-corpus\"  # <-- ajustá al nombre real de tu Dataset\n",
                "\n",
                "os.makedirs('data', exist_ok=True)\n",
                "\n",
                "candidates = list(Path(f'/kaggle/input/{CORPUS_DATASET_SLUG}').glob('*.tar.gz'))\n",
                "if not candidates:\n",
                "    candidates = list(Path('/kaggle/input').glob('*/*.tar.gz'))\n",
                "\n",
                "assert candidates, (\n",
                "    'No encontré ningún .tar.gz en /kaggle/input — montá el Dataset del corpus '\n",
                "    'con Add Input antes de correr esta celda.'\n",
                ")\n",
                "corpus_tar = candidates[0]\n",
                "print(f'📦 Extrayendo {corpus_tar} ...')\n",
                "!tar -xzf {corpus_tar} -C data\n",
                "\n",
                "shard_dirs = [d for d in Path('data').iterdir() if d.is_dir() and list(d.glob('shard_*.bin'))]\n",
                "assert shard_dirs, 'La extracción no produjo ningún shard_*.bin — revisá el .tar.gz.'\n",
                "CORPUS_DIR = shard_dirs[0]\n",
                "n_shards = len(list(CORPUS_DIR.glob('shard_*.bin')))\n",
                "print(f'✅ Corpus listo en {CORPUS_DIR} — {n_shards} shard(s) encontrados.')\n",
            ],
        },
        # Celda 5: Smoke test CPU (opcional, rápido)
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. (Opcional) Smoke Test Rápido\n",
                "Verifica en ~30s que el trainer, el checkpointing y el resume funcionan antes\n",
                "de lanzar el entrenamiento real de horas. Corre en CPU con datos sintéticos,\n",
                "no toca tu corpus ni tus checkpoints reales."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python training/train_teo_sds.py --smoke-test --checkpoint-dir /kaggle/working/smoke_test_ckpt\n",
                "!rm -rf /kaggle/working/smoke_test_ckpt training/data/teo_sds_corpus\n",
            ],
        },
        # Celda 6: Entrenamiento real
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Entrenamiento de Teo-SDS (300-350M params) con Auto-Resume\n",
                "Detecta automáticamente `checkpoint.pt` si ya existe y continúa. Si la sesión\n",
                "de Kaggle se corta por límite de tiempo, volvé a correr esta misma celda —\n",
                "retoma en el step exacto donde quedó (optimizador, RNG, todo)."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "os.makedirs('models/teo_sds', exist_ok=True)\n",
                "\n",
                "!python training/train_teo_sds.py \\\n",
                "    --data-dir {CORPUS_DIR} \\\n",
                "    --checkpoint-dir models/teo_sds \\\n",
                "    --device cuda \\\n",
                "    --batch-size 8 \\\n",
                "    --context-len 512 \\\n",
                "    --lr 3e-4 \\\n",
                "    --max-steps 100000 \\\n",
                "    --save-interval 500 \\\n",
                "    --log-interval 50\n",
            ],
        },
        # Celda 7: Inferencia en vivo
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Prueba de Inferencia O(1) en Vivo (stepping recurrente, cero KV-Cache)"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "from netelpro.neuro.tokenizer_bpe import NetelproBPETokenizer\n",
                "from training.train_teo_sds import TeoSDSConfig, load_checkpoint\n",
                "from netelpro.neuro.dynamic_state import NetelproSDSModel\n",
                "\n",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "tok = NetelproBPETokenizer.load('data/teo_v2/tokenizer.json')\n",
                "\n",
                "ckpt_probe = torch.load('models/teo_sds/checkpoint.pt', map_location=device, weights_only=False)\n",
                "cfg = TeoSDSConfig.from_dict(ckpt_probe['config'])\n",
                "\n",
                "model = NetelproSDSModel(cfg.to_sds_config()).to(device)\n",
                "load_checkpoint('models/teo_sds/checkpoint.pt', model=model, device=device)\n",
                "model.eval()\n",
                "print(f'✅ Teo-SDS cargado — step {ckpt_probe[\"step\"]}, loss {ckpt_probe[\"loss\"]:.4f}')\n",
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
                "    out = model.generate(input_ids, max_new_tokens=150, temperature=0.8)\n",
                "    print('RESPUESTA:', tok.decode(out[0].tolist()))\n",
            ],
        },
        # Celda 8: Empaquetar
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Empaquetar Checkpoint Final de Teo-SDS"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!tar -czvf /kaggle/working/teo_sds_checkpoint.tar.gz -C models/teo_sds checkpoint.pt\n",
                "print('🎉 Checkpoint Teo-SDS empaquetado en /kaggle/working/teo_sds_checkpoint.tar.gz')\n",
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
    print(f"✅ Generated Teo-SDS training notebook at: {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
