"""Arma los datasets de SFT desde el modelo base sobre las tareas TRAIN.

  verified.jsonl    solo muestras que el verificador acepta (método Goliath/RAFT)
  unverified.jsonl  control: MISMA cantidad y mismas tareas, sin filtrar (incluye fallos)

Si verified y unverified rinden igual, el verificador no aporta nada: solo el volumen de SFT.
"""
import argparse, json, random
from config import *
from common import *
from eval_arm import make_generate, mock_generate, v_sl

CAP_PER_TASK = 4   # tope de muestras aceptadas por tarea (evita que dominen las fáciles)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SAMPLES_TRAIN)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    gen = mock_generate if a.mock else make_generate(None)
    ids = TRAIN_IDS[:2] if a.mock else TRAIN_IDS
    rng = random.Random(SEED)
    verified, pool = [], []
    for tid in ids:
        task = load_task(tid)
        prompt = build_prompt_sl(task)
        good, seen = 0, set()
        outs = gen.many(prompt, a.n, seed=1000) if hasattr(gen, "many") else [gen(prompt, seed=1000 + i) for i in range(a.n)]
        for out in outs:
            src = extract_src(out)
            ok = v_sl(out, task)
            row = {"task": tid, "prompt": prompt, "response": f"```netelpro\n{src}\n```" if src else out, "ok": ok}
            pool.append(row)
            if ok and src not in seen and good < CAP_PER_TASK:
                seen.add(src); good += 1; verified.append(row)
        print(f"{tid}: {good} aceptadas de {a.n}")
    unverified = rng.sample(pool, min(len(verified), len(pool)))
    OUT.mkdir(exist_ok=True)
    for name, rows in (("verified", verified), ("unverified", unverified)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"verified={len(verified)}  unverified={len(unverified)} (de ellas ok={sum(r['ok'] for r in unverified)})")
