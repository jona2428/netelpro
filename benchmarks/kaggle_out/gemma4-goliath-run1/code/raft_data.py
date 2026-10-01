"""Arma los datasets de SFT desde el modelo base sobre las tareas TRAIN.

  verified.jsonl    solo muestras que el verificador acepta (método Goliath/RAFT)
  unverified.jsonl  control: MISMA cantidad y mismas tareas, sin filtrar (incluye fallos)

Si verified y unverified rinden igual, el verificador no aporta nada: solo el volumen de SFT.
"""
import argparse, json, os, random
from config import *
from common import *
from eval_arm import make_generate, mock_generate, v_sl

CAP_PER_TASK = 4   # tope de muestras aceptadas por tarea (evita que dominen las fáciles)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SAMPLES_TRAIN)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--prev", default="", help="verified.jsonl de una ronda anterior para fusionar")
    a = ap.parse_args()
    gen = mock_generate if a.mock else make_generate(None)
    ids = TRAIN_IDS[:2] if a.mock else TRAIN_IDS
    rng = random.Random(SEED)
    verified, pool = [], []
    for tid in ids:
        task = load_task(tid)
        prompt = build_prompt_sl(task)
        good, seen = 0, set()
        outs = gen.many(prompt, a.n, seed=a.seed) if hasattr(gen, "many") else [gen(prompt, seed=a.seed + i) for i in range(a.n)]
        for out in outs:
            src = extract_src(out)
            ok = v_sl(out, task)
            row = {"task": tid, "prompt": prompt, "response": f"```netelpro\n{src}\n```" if src else out, "ok": ok}
            pool.append(row)
            if ok and src not in seen and good < CAP_PER_TASK:
                seen.add(src); good += 1; verified.append(row)
        print(f"{tid}: {good} aceptadas de {a.n}", flush=True)
    if a.prev and os.path.exists(a.prev):     # fusiona con la ronda anterior: dedupe por respuesta y tope por tarea
        by_task = {}
        for r in [json.loads(l) for l in open(a.prev, encoding="utf-8")] + verified:
            lst = by_task.setdefault(r["task"], [])
            if len(lst) < CAP_PER_TASK and all(r["response"] != x["response"] for x in lst):
                lst.append(r)
        verified = [r for lst in by_task.values() for r in lst]
        print(f"fusionado con {a.prev}: {len(verified)} verificados en {len(by_task)} tareas")
    unverified = rng.sample(pool, min(len(verified), len(pool)))
    OUT.mkdir(exist_ok=True)
    for name, rows in (("verified", verified), ("unverified", unverified)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"verified={len(verified)}  unverified={len(unverified)} (de ellas ok={sum(r['ok'] for r in unverified)})")
