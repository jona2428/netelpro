"""Verificación del cierre de la corrida v2 (scratch, descartable)."""
from __future__ import annotations

import json
from pathlib import Path

P = Path(__file__).resolve().parent
v2 = P / "vtb_ood_benchmark_results_v2.json"
partial = P / "vtb_ood_benchmark_results_v2.partial.json"

print("v2.json existe:", v2.exists(), "| tam:", v2.stat().st_size if v2.exists() else 0)
print("partial existe:", partial.exists(), "| tam:", partial.stat().st_size if partial.exists() else 0)

if v2.exists():
    d = json.loads(v2.read_text(encoding="utf-8"))
    print("v2 keys:", sorted(d.keys()))
    print("v2 total_cases:", d.get("total_cases"))
    print(json.dumps(d.get("metrics", {}), indent=1, ensure_ascii=False))