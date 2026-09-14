"""Generador del notebook build_massive_corpus_kaggle.ipynb para Kaggle.

Notebook dedicado a la Fase 2 del roadmap Teo-SDS 7B MoE: compila el corpus
masivo (250-500M tokens: Cosmopedia v2 + OpenAssistant/Alpaca-ES + FineWeb-2
Español + código de sistemas Python/Rust/C++/C# + lore/razonamiento Netelpro)
en shards binarios packed uint16, listos para alimentar tanto a
`training/train_teo_v2.py` como al nuevo `training/train_teo_sds.py`.

Este notebook SOLO construye el corpus (no entrena). Al final empaqueta los
shards en un .tar.gz para subirlos como Kaggle Dataset privado y reutilizarlos
en notebooks de entrenamiento posteriores sin volver a descargar de HuggingFace.
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
OUTPUT_NOTEBOOK = TRAINING_DIR / "build_massive_corpus_kaggle.ipynb"


def build_notebook() -> dict:
    cells = [
        # Celda 1: Portada
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📦 Netelpro: Fase 2 - Compilación del Corpus Masivo (250-500M tokens)\n",
                "### *Cosmopedia v2 + OpenAssistant/Alpaca-ES + FineWeb-2 Español + Código de Sistemas + Lore Netelpro*\n",
                "\n",
                "Este notebook **solo construye el corpus** — no entrena ningún modelo. Genera shards binarios\n",
                "packed `uint16` (mismo formato que usa `train_teo_v2.py` y el nuevo `train_teo_sds.py`) y los\n",
                "empaqueta al final para subirlos como **Kaggle Dataset privado**, reutilizable en cualquier\n",
                "notebook de entrenamiento posterior sin re-descargar de HuggingFace.\n",
                "\n",
                "---\n",
                "### ⚙️ Configuración en Kaggle:\n",
                "- **Accelerator:** CPU o GPU (no importa, este notebook no entrena — GPU T4 ayuda solo por más RAM/CPU asignada)\n",
                "- **Internet:** **On** (obligatorio — streaming desde HuggingFace)\n",
                "- **Persistence:** Files on (para que el corpus sobreviva si la sesión se corta y hay que reanudar)\n",
            ],
        },
        # Celda 2: Repositorio y dependencias
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Clonar/Actualizar Repositorio Netelpro (privado) e Instalar Dependencias\n",
                "Repo privado: usa **Kaggle Secrets** (`Add-ons` -> `Secrets` -> agregá un secret llamado `GITHUB_TOKEN`\n",
                "con tu Personal Access Token de GitHub). Si no configurás el secret, pegá el token manualmente\n",
                "en la variable `GITHUB_TOKEN` de la celda de abajo antes de correrla (y no subas/compartas el\n",
                "notebook con el token pegado adentro).\n",
                "\n",
                "`bigcode/the-stack-smol` (fuente de código de la Celda 4) también es un dataset gated en\n",
                "HuggingFace: agregá OTRO secret llamado `HF_TOKEN` (token de `huggingface.co/settings/tokens`,\n",
                "scope Read alcanza) y **aceptá el acceso** en la página de cada subset que uses antes de correr\n",
                "(`huggingface.co/datasets/bigcode/the-stack-smol` -> botón de aceptar términos). Sin esto, esa\n",
                "fuente cae sola al banco local chico y no bloquea el resto de la compilación."
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
                "HF_TOKEN = \"\"       # <-- idem, para bigcode/the-stack-smol (dataset gated)\n",
                "\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    _secrets = UserSecretsClient()\n",
                "except Exception:\n",
                "    _secrets = None\n",
                "\n",
                "if not GITHUB_TOKEN and _secrets is not None:\n",
                "    try:\n",
                "        GITHUB_TOKEN = _secrets.get_secret(\"GITHUB_TOKEN\")\n",
                "    except Exception:\n",
                "        GITHUB_TOKEN = \"\"\n",
                "\n",
                "if not HF_TOKEN and _secrets is not None:\n",
                "    try:\n",
                "        HF_TOKEN = _secrets.get_secret(\"HF_TOKEN\")\n",
                "    except Exception:\n",
                "        HF_TOKEN = \"\"\n",
                "\n",
                "if HF_TOKEN:\n",
                "    os.environ[\"HF_TOKEN\"] = HF_TOKEN\n",
                "    print('✅ HF_TOKEN cargado — datasets gated (the-stack-smol) van a poder autenticar.')\n",
                "else:\n",
                "    print('⚠️ HF_TOKEN no configurado — the-stack-smol caerá al banco local de código.')\n",
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
                "!pip install -q tokenizers datasets pyarrow requests numpy\n",
            ],
        },
        # Celda 3: Reanudar corpus previo (si se subió como Kaggle Dataset)
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. (Opcional) Reanudar desde un Corpus Parcial Anterior\n",
                "Si esta es la continuación de una sesión de Kaggle previa que se cortó a mitad de camino,\n",
                "subí el `.tar.gz` empaquetado en la celda final de esa sesión como Kaggle Dataset\n",
                "(ej: `teo-sds-massive-corpus-partial`) y montalo acá antes de correr la Celda 4.\n",
                "El builder detecta `dedup_state.json` y los shards ya completos, y sigue desde ahí\n",
                "(`--resume` está activo por default)."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "os.makedirs('data/teo_sds_massive', exist_ok=True)\n",
                "\n",
                "# Descomentar y ajustar el nombre del dataset si estás reanudando:\n",
                "# !tar -xzvf /kaggle/input/teo-sds-massive-corpus-partial/massive_corpus_partial.tar.gz -C data/teo_sds_massive\n",
                "\n",
                "print('📂 Contenido actual de data/teo_sds_massive:', os.listdir('data/teo_sds_massive') if os.path.exists('data/teo_sds_massive') else 'vacío')\n",
            ],
        },
        # Celda 4: Construcción del corpus masivo
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Compilar Corpus Masivo (20 shards × 25M tokens = 500M tokens)\n",
                "Streaming simultáneo de 5 dominios (razonamiento/lore, conversaciones, Cosmopedia STEM, código\n",
                "de sistemas, filosofía/ciencia en español), deduplicado, con checkpoint de progreso cada shard\n",
                "completo (`dedup_state.json` + shards ya cerrados). Ajustá `--target-shards` a 10 (250M) si\n",
                "preferís una corrida más corta primero.\n",
                "\n",
                "**Nota:** la librería `datasets` a veces crashea el proceso (`Fatal Python error:\n",
                "PyGILState_Release`) al reintentar una descarga de Parquet en el sandbox de red de Kaggle —\n",
                "no es un bug del script, y no pierde progreso (los shards ya cerrados y el estado de\n",
                "deduplicación quedan en disco). La celda de abajo reintenta sola hasta 15 veces si eso pasa."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Nota: NO uses %%bash acá — Jupyter/Kaggle bufferea toda la salida de una\n",
                "# celda %%bash hasta que termina, así que no verías ningún progreso en vivo.\n",
                "# subprocess.run con stdout heredado transmite en vivo igual que !python.\n",
                "import subprocess\n",
                "import sys\n",
                "import time\n",
                "from pathlib import Path\n",
                "\n",
                "MAX_ATTEMPTS = 15\n",
                "TARGET_SHARDS = 20\n",
                "OUT_DIR = Path('data/teo_sds_massive')\n",
                "\n",
                "for attempt in range(1, MAX_ATTEMPTS + 1):\n",
                "    print(f'=== Intento {attempt}/{MAX_ATTEMPTS} ===', flush=True)\n",
                "    result = subprocess.run([\n",
                "        sys.executable, '-u', 'training/data/build_massive_corpus.py',\n",
                "        '--out-dir', str(OUT_DIR),\n",
                "        '--tokenizer-path', 'data/teo_v2/tokenizer.json',\n",
                "        '--target-shards', str(TARGET_SHARDS),\n",
                "        '--shard-size', '25000000',\n",
                "        '--persona-multiplier', '100',\n",
                "    ])\n",
                "    if result.returncode == 0:\n",
                "        print(f'✅ Compilación terminada en el intento {attempt}.')\n",
                "        break\n",
                "    n_shards = len(list(OUT_DIR.glob('shard_*.bin'))) if OUT_DIR.exists() else 0\n",
                "    print(f'⚠️ Se cortó (exit {result.returncode}). Shards completos: {n_shards}/{TARGET_SHARDS}. Reintentando...', flush=True)\n",
                "    time.sleep(3)\n",
            ],
        },
        # Celda 5: Verificación / estadísticas
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Verificación del Corpus Compilado"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.insert(0, '.')\n",
                "from pathlib import Path\n",
                "from training.data.compile_packed import load_stream, stream_stats\n",
                "\n",
                "shard_dir = Path('data/teo_sds_massive')\n",
                "shards = sorted(shard_dir.glob('shard_*.bin'))\n",
                "print(f'📂 Shards completos: {len(shards)}')\n",
                "\n",
                "total_tokens = 0\n",
                "for s in shards:\n",
                "    stream = load_stream(s)\n",
                "    stats = stream_stats(stream)\n",
                "    total_tokens += stats['total_tokens']\n",
                "    assert stats['pad_fraction'] == 0.0, f'{s.name} tiene padding! {stats}'\n",
                "    print(f\"  {s.name}: {stats['total_tokens']:,} tokens, pad_fraction={stats['pad_fraction']}\")\n",
                "\n",
                "print(f'\\n🎉 Total: {total_tokens:,} tokens en {len(shards)} shard(s) — CERO padding verificado.')\n",
            ],
        },
        # Celda 6: Empaquetado para subir como Kaggle Dataset
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Empaquetar Corpus para Reutilización (Kaggle Dataset)\n",
                "Comprime todos los shards + tokenizer + metadata. Si el corpus quedó incompleto (sesión\n",
                "cortada por límite de tiempo), este mismo archivo sirve para retomar en la Celda 2 de la\n",
                "próxima sesión. Si quedó completo, subilo como Dataset (ej. `teo-v2-massive-corpus`) para\n",
                "usarlo directamente en `train_teo_sds.py` sin repetir la descarga."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!tar -czvf /kaggle/working/massive_corpus_partial.tar.gz -C data teo_sds_massive\n",
                "print('🎉 Corpus empaquetado en /kaggle/working/massive_corpus_partial.tar.gz')\n",
                "print('   Subilo como Kaggle Dataset privado (ej: teo-v2-massive-corpus) para usarlo')\n",
                "print('   en el notebook de entrenamiento de Teo-SDS (train_teo_sds.py) sin re-descargar.')\n",
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
    print(f"✅ Generated massive corpus notebook at: {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
