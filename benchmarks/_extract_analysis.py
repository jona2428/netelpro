"""One-shot: extract self_history cases + aligned THEATER cases from OOD results."""
import json
from pathlib import Path

data = json.loads(
    Path(__file__).with_name("vtb_ood_benchmark_results.json").read_text(encoding="utf-8")
)
cases = data["case_comparisons"]

print("=== SELF_HISTORY (base | aligned) ===")
for c in cases:
    if c["category"] != "self_history":
        continue
    print(f"\n[{c['id']}] base={c['base_status']} | aligned={c['aligned_status']}")
    print(f"  BASE:    {c['base_response'][:220]}")
    print(f"  ALIGNED: {c['aligned_response'][:220]}")

print("\n\n=== ALL ALIGNED THEATER ===")
for c in cases:
    if c["aligned_status"] == "THEATER":
        print(f"\n[{c['id']}] ({c['category']})")
        print(f"  ALIGNED: {c['aligned_response'][:400]}")

print("\n\n=== ALL BASE THEATER ===")
for c in cases:
    if c["base_status"] == "THEATER":
        print(f"\n[{c['id']}] ({c['category']})")
        print(f"  BASE: {c['base_response'][:400]}")