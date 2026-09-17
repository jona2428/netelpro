"""Probe de verificación del dataset grammar-v3 (solo lectura, Teo).

Verifica: composición del train_mix, disjunción de semánticas train/heldout
(patrón de casos), masa de patrones activadores, y un gate real: compilar
N muestras del heldout con el compilador netelpro real.
Uso: C:/Python314/python.exe scripts/probe_grammar_v3.py
"""
import json
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "training" / "data"


def load(name):
    path = DATA / name
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def case_pattern(row):
    """Signature semántica: la lista de casos de la tabla (hashable)."""
    return json.dumps(row["netelpro_cases"], sort_keys=True)


def main():
    train = load("train.jsonl")
    heldout = load("heldout.jsonl")
    mix = load("train_mix.jsonl")
    v2 = load("dataset_v2.jsonl")

    print(f"counts: train={len(train)} heldout={len(heldout)} mix={len(mix)} v2={len(v2)}")

    # Composición del mix: cuántas líneas provienen de dataset_v2 (anti-forgetting)
    v2_prompts = {row["prompt"] for row in v2}
    v2_in_mix = sum(1 for row in mix if row["prompt"] in v2_prompts)
    print(f"mix: {len(mix) - v2_in_mix} grammar + {v2_in_mix} v2-replay "
          f"({v2_in_mix / len(mix):.1%})")

    # Disjunción de semánticas (el heldout puede compartir FORMAS de tabla,
    # pero la disjunción nominal ya se verificó: 660 vs 60 señales, 0 overlap).
    pat_train = Counter(case_pattern(r) for r in train)
    pat_ho = Counter(case_pattern(r) for r in heldout)
    shared = set(pat_train) & set(pat_ho)
    print(f"distinct case-patterns: train={len(pat_train)} heldout={len(pat_ho)} "
          f"shared={len(shared)}")
    if shared:
        # ¿Es grave? Solo si un patrón compartido domina el heldout.
        ho_total = sum(pat_ho.values())
        shared_total = sum(pat_ho[p] for p in shared)
        print(f"  heldout rows with a shared pattern: {shared_total}/{ho_total}")
        top_shared = sorted(shared, key=lambda p: pat_ho[p], reverse=True)[:2]
        for p in top_shared:
            print(f"  pattern in train={pat_train[p]}x heldout={pat_ho[p]}x")

    # Gate de compilador: 12 muestras del heldout, mezcla aleatoria con seed.
    import sys
    sys.path.insert(0, str(REPO))
    from benchmarks.run5_gate import extract_block, compile_verdict

    rng = random.Random(20260915)
    sample = rng.sample(heldout, 12)
    ok = 0
    for row in sample:
        contract = extract_block(row["completion"], "netelpro")
        verdict = compile_verdict(contract)
        ok += 1 if verdict else 0
    print(f"compiler gate on heldout sample: {ok}/12 compiled")


if __name__ == "__main__":
    main()