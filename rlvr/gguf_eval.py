"""Eval pass@k de un GGUF RAFT via ollama local sobre el split OOD explicito.

Protocolo identico al notebook de Colab (tools/make_rlvr_notebook.py):
temp 0.8, 8 muestras/tarea, verificacion real via verify_program.
Checkpoint incremental en JSON: si el proceso se corta, las tareas ya
evaluadas no se repiten.

Procedencia: este modulo es la ingesta oficial de la medicion del
2026-09-07 (GGUF q4_k_m pass@8 OOD 80% = 4/5 tareas, modelo
JonaECG/netelpro-qwen2.5-1.5b-raft en ollama local), que re-verifico
independientemente el 40% medido en fp16 en Colab. La herramienta que
produjo el headline del experimento vive ahora en el repo, y el numero
que produce es reproducible por cualquiera con ollama + el GGUF publico.

Uso (desde la raiz del repo, con ollama corriendo y el modelo instalado):
    python -m rlvr.gguf_eval --model netelpro-qwen1.5b-raft

Para tests, generate_fn es inyectable: nada de este modulo exige ollama.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path

from rlvr.verify import verify_program

OLLAMA_URL = "http://localhost:11434/api/generate"
NUM_SAMPLES = 8
TEMP = 0.8
MAX_NEW_TOKENS = 256

FENCE_NETELPRO = re.compile(r"```netelpro\s*(.*?)```", re.DOTALL)
FENCE_ANY = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)


def extract_src(out: str) -> str | None:
    """Extrae el codigo del modelo: prefiere fence ```netelpro, cae a cualquier fence.

    Sin fence no hay candidato: None (la muestra cuenta como fallida).
    """
    m = FENCE_NETELPRO.search(out) or FENCE_ANY.search(out)
    return m.group(1).strip() if m else None


def ollama_generate(
    prompt: str,
    seed: int,
    *,
    model: str,
    temp: float = TEMP,
    url: str = OLLAMA_URL,
    num_predict: int = MAX_NEW_TOKENS,
) -> str:
    """Genera una muestra via API /api/generate de ollama (no streaming)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temp,
            "num_predict": num_predict,
            "seed": seed,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]


def evaluate_task(
    task_id: str,
    load_task_fn: Callable,
    build_prompt_fn: Callable[[object], str],
    generate_fn: Callable[..., str],
    *,
    num_samples: int = NUM_SAMPLES,
    max_steps: int | None = None,
) -> int:
    """Muestras que pasan los 20 casos de la tarea: genera, extrae, verifica.

    Muestra sin fence extraible cuenta como fallo (no se relanza).
    """
    task = load_task_fn(task_id)
    prompt = build_prompt_fn(task)
    ok = 0
    for i in range(num_samples):
        out = generate_fn(prompt, seed=i)
        src = extract_src(out)
        if src is None:
            continue
        if max_steps is None:
            result = verify_program(src, task)
        else:
            result = verify_program(src, task, max_steps=max_steps)
        if result.passed:
            ok += 1
    return ok


def run_ood_eval(
    task_ids: list[str],
    load_task_fn: Callable,
    build_prompt_fn: Callable[[object], str],
    generate_fn: Callable[..., str],
    *,
    num_samples: int = NUM_SAMPLES,
    ckpt_path: Path | None = None,
    log_fn: Callable[[str], None] = print,
    max_steps: int | None = None,
) -> dict[str, int]:
    """Evalua todas las tareas con checkpoint incremental {task_id: passes}."""
    done: dict[str, int] = {}
    if ckpt_path is not None and ckpt_path.exists():
        done = json.loads(ckpt_path.read_text())
        log_fn(f"checkpoint: {len(done)} tareas ya evaluadas")
    for tid in task_ids:
        if tid in done:
            log_fn(f"{tid}: {done[tid]}/{num_samples} (checkpoint)")
            continue
        ok = evaluate_task(
            tid,
            load_task_fn,
            build_prompt_fn,
            generate_fn,
            num_samples=num_samples,
            max_steps=max_steps,
        )
        done[tid] = ok
        if ckpt_path is not None:
            ckpt_path.write_text(json.dumps(done))
        log_fn(f"{tid}: {ok}/{num_samples} muestras pasan")
    return done


def pass_rate(done: dict[str, int]) -> float:
    """pass@k a nivel de tarea: tarea resuelta si >=1 muestra pasa los 20 casos."""
    solved = sum(1 for v in done.values() if v > 0)
    return solved / max(len(done), 1)


def resolve_ckpt_path(ckpt: Path) -> Path | None:
    """Resuelve el flag --ckpt a Path o None (checkpoint desactivado).

    """
    return None if str(ckpt) in ("", ".") else ckpt


def main() -> None:
    from rlvr.prompting import build_prompt
    from rlvr.tasks import OOD_TASK_IDS, load_task

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="netelpro-qwen1.5b-raft")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--temp", type=float, default=TEMP)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("gguf_ood_ckpt.json"),
        help="checkpoint incremental (pasa --ckpt '' para desactivar)",
    )
    args = parser.parse_args()

    def generate_fn(prompt: str, seed: int) -> str:
        return ollama_generate(prompt, seed, model=args.model, temp=args.temp)

    ckpt_path = resolve_ckpt_path(args.ckpt)
    done = run_ood_eval(
        list(OOD_TASK_IDS),
        load_task,
        build_prompt,
        generate_fn,
        num_samples=args.samples,
        ckpt_path=ckpt_path,
    )
    rate = pass_rate(done)
    print(f"\nGGUF pass@{args.samples} OOD: {rate:.0%} ({rate * len(done):.0f}/{len(done)} tareas)")


if __name__ == "__main__":
    main()