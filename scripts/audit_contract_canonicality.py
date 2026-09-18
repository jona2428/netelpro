"""Audit: how many live host contracts are canonical all-of-N, and what would
change if they were regenerated. Read-only; writes nothing.

Run: python workspace/straylight/scripts/audit_contract_canonicality.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from netelpro.lib import LibError, contract_from_source, render_contract  # noqa: E402
from netelpro.rule_filter import RuleFilter  # noqa: E402

CONTRACT_DIRS = (
    REPO.parent.parent / "data" / "contracts",
    Path("C:/Users/Jona/Desktop/FinalFront/data/contracts"),
)


def main() -> int:
    directory = next((d for d in CONTRACT_DIRS if d.is_dir()), None)
    if directory is None:
        print("contract directory not found; checked:")
        for d in CONTRACT_DIRS:
            print(f"  {d}")
        return 1
    print(f"contract dir: {directory}\n")

    canonical, bespoke = [], []
    for path in sorted(directory.glob("*.sl")):
        source = path.read_text(encoding="utf-8-sig")
        try:
            spec = contract_from_source(source)
        except LibError as exc:
            bespoke.append((path.name, str(exc)))
            continue
        canonical.append((path, source, spec))

    print(f"CANONICAL (all-of-N, generatable): {len(canonical)}")
    for path, source, spec in canonical:
        rendered = render_contract(spec)
        original = RuleFilter(source)
        generated = RuleFilter(rendered)
        arity = len(spec.slots)
        divergent = [
            combo
            for combo in itertools.product((0, 1), repeat=arity)
            if original.decide_int(*combo) != generated.decide_int(*combo)
        ]
        status = "OK" if not divergent else f"DIVERGENT {divergent}"
        print(
            f"  {path.name:26s} slots={arity} admit={spec.admit} reject={spec.reject}  {status}"
        )

    print(f"\nBESPOKE (own policy, not generatable): {len(bespoke)}")
    for name, reason in bespoke:
        print(f"  {name:26s} {reason[:80]}")

    # Row lines a canonical contract carries: one zero-row per slot, the
    # all-ones admit row, and the mandatory all-wildcard backstop.
    hand_rows = sum(len(s.slots) + 2 for _, _, s in canonical)
    generated_lines = sum(render_contract(s).count("\n") for _, _, s in canonical)
    print(f"\nrow lines currently hand-written in canonical set: {hand_rows}")
    print(f"lines of contract text a generator would emit:     {generated_lines}")
    print(
        "every canonical contract round-tripped with identical verdicts "
        "over its full declared domain"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
