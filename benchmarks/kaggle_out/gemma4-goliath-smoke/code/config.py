"""Config única del experimento Gemma 4 E2B x Goliath Killer (RAFT verificado)."""
import os
from pathlib import Path

NETELPRO_REPO = Path(os.environ.get("NETELPRO_REPO", r"C:\Users\Jona\Documents\netelpro"))   # rlvr/ (tareas, verificador, prompt)
KAGGLE_MODEL = "google/gemma-4/transformers/gemma-4-e2b-it"  # slug verificado en Kaggle
HF_MODEL = os.environ.get("MODEL_PATH", "google/gemma-4-e2b-it")   # en Kaggle: carpeta local del modelo adjunto

K = 8            # muestras por tarea (igual que el protocolo previo con Qwen 1.5B)
TEMP = 0.8
MAX_NEW_TOKENS = 512
SEED = 0
N_SAMPLES_TRAIN = 16      # muestras por tarea train para armar el dataset RAFT
OUT = Path(os.environ.get("OUT_DIR", str(Path(__file__).parent / "out")))
