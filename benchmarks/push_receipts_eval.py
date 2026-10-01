"""Build (and optionally push) a Kaggle kernel that runs the receipts
benchmark v0.2 on a GGUF: held-out set + frozen OOD set, 10 repeats each,
llama-cpp with full GPU offload on a T4 (spec v0.2 §5, D9, D10).

The model comes either from Hugging Face (the base: the official
Qwen/Qwen2.5-1.5B-Instruct-GGUF q4_k_m, the same file as the 2026-10-01 runs)
or from a previous kernel's output (a Receipts-RAFT stage-5 GGUF), found by
glob under /kaggle/input. Outputs land in /kaggle/working:
receipts_<set>_<name>_v02.json plus a meta file with the model's sha256 and
library versions. Nothing is labelled here; labels are human (§5).

    python benchmarks/push_receipts_eval.py --name base                     # build only
    python benchmarks/push_receipts_eval.py --name base --push              # send
    python benchmarks/push_receipts_eval.py --name armA --kernel-source receipts-raft-a-s5 \\
        --model-glob "/kaggle/input/*/netelpro_qwen1.5b_receipts_raft_a/*.gguf" --push
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
KERNELS_DIR = REPO / "training" / "kernels"
KAGGLE_USER = "jonacgerizo"
BASE_HF = ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf")


def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def notebook(name: str, model_glob: str | None) -> dict:
    install = '''
# llama-cpp-python con CUDA: wheel precompilada si existe para este Python/CUDA; si no, compilar con CUDA.
import subprocess, sys, os
def pip(*a, env=None):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], env=env).returncode
ok = pip("llama-cpp-python", "--only-binary=:all:", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124") == 0
if ok:
    import llama_cpp
    ok = llama_cpp.llama_supports_gpu_offload()
    print("wheel precompilada, GPU offload:", ok)
if not ok:
    print("compilando llama-cpp-python con CUDA (~10-15 min)...")
    env = {**os.environ, "CMAKE_ARGS": "-DGGML_CUDA=on", "FORCE_CMAKE": "1"}
    assert pip("--force-reinstall", "--no-cache-dir", "llama-cpp-python", env=env) == 0
'''
    clone = '''
# GIT_TERMINAL_PROMPT=0: si el repo no es público, fallar en segundos en vez de colgar esperando credenciales.
!GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/jona2428/netelpro.git
import glob, hashlib, json, subprocess, sys
from pathlib import Path
'''
    if model_glob is None:
        get_model = f'''
from huggingface_hub import hf_hub_download
MODEL = hf_hub_download("{BASE_HF[0]}", "{BASE_HF[1]}")
'''
    else:
        get_model = f'''
found = glob.glob({model_glob!r})
assert len(found) == 1, f"esperaba un GGUF, hay {{found}}"
MODEL = found[0]
'''
    run = f'''
NAME = {name!r}
sha = hashlib.sha256(Path(MODEL).read_bytes()).hexdigest()
import llama_cpp
assert llama_cpp.llama_supports_gpu_offload(), "llama-cpp sin GPU -- no correr 320 trials en CPU"
meta = {{"name": NAME, "model": MODEL, "sha256": sha, "llama_cpp": llama_cpp.__version__, "repeats": 10,
        "temperature": 0.5, "max_tokens": 160}}
Path(f"/kaggle/working/receipts_eval_{{NAME}}_meta.json").write_text(json.dumps(meta, indent=2))
print(meta)
for s in ("bench", "ood"):
    out = f"/kaggle/working/receipts_{{s}}_{{NAME}}_v02.json"
    r = subprocess.run([sys.executable, "-m", "benchmarks.receipts_qwen_live_bench", "--model", MODEL, "--set", s,
                        "--repeats", "10", "--n-gpu-layers", "-1", "--out", out], cwd="netelpro", capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0, f"{{s}} falló"
'''
    return {"cells": [_code(install), _code(clone), _code(get_model), _code(run)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def build(name: str, *, kernel_source: str | None = None, model_glob: str | None = None,
          user: str = KAGGLE_USER, out_root: Path = KERNELS_DIR) -> Path:
    if (kernel_source is None) != (model_glob is None):
        raise ValueError("--kernel-source and --model-glob go together (or neither, for the HF base)")
    slug = f"receipts-eval-{name.lower()}"
    out = out_root / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(notebook(name, model_glob), indent=1) + "\n", encoding="utf-8")
    meta = {
        "id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "keywords": ["gpu"],
        "dataset_sources": [], "competition_sources": [], "model_sources": [],
        "kernel_sources": [f"{user}/{kernel_source}"] if kernel_source else [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="base | armA | armB | ...")
    ap.add_argument("--kernel-source", default=None, help="slug of the kernel whose output holds the GGUF")
    ap.add_argument("--model-glob", default=None, help="glob under /kaggle/input for that GGUF")
    ap.add_argument("--push", action="store_true", help="send to Kaggle (spends GPU quota)")
    a = ap.parse_args(argv)
    out = build(a.name, kernel_source=a.kernel_source, model_glob=a.model_glob)
    print(f"built {out}")
    if a.push:
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(out)], capture_output=True, text=True)
        print(r.stdout or r.stderr)
        return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
