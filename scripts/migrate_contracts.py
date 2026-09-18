"""Migrate the live host contracts to generated formal blocks.

Two modes, and the safe one is the default:

``--check`` (default)
    Read-only. Reports, per contract, whether its formal block already matches
    what the generator would emit. Exit code 1 if any contract drifts. This is
    the CI / preflight guard: hand-editing a generated block fails the check.

``--write``
    Rewrites the formal block of every canonical contract in place, preserving
    the prose header verbatim. Before writing anything it compiles both the old
    and the new source and asserts identical verdicts over the full declared
    domain -- a contract that would change meaning is refused, not written.

Nothing here touches the contracts' *meaning*: slot names and their
descriptions stay in the file. What gets normalised is the shape (parameter
declarations, one zero-rejection row per slot, the all-ones admit row, the
mandatory all-wildcard backstop).

Run:
    python workspace/straylight/scripts/migrate_contracts.py --check
    python workspace/straylight/scripts/migrate_contracts.py --write
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from netelpro.lib import (  # noqa: E402
    LibError,
    contract_from_source,
    migrate_source,
)
from netelpro.rule_filter import RuleFilter  # noqa: E402

CONTRACT_DIRS = (
    REPO.parent.parent / "data" / "contracts",
    Path("C:/Users/Jona/Desktop/FinalFront/data/contracts"),
    Path("/mnt/c/Users/Jona/Desktop/FinalFront/data/contracts"),
)


def _contract_dir() -> Path | None:
    return next((d for d in CONTRACT_DIRS if d.is_dir()), None)


def _verify_equivalent(name: str, old_src: str, new_src: str, arity: int) -> None:
    """Refuse the migration unless both sources decide identically everywhere.

    Declared domain: full differential check on both sources. Out-of-domain
    probes: native-vs-native only, because the reference interpreter rejects
    out-of-range literals at parse time.
    """
    old = RuleFilter(old_src)
    new = RuleFilter(new_src)
    domain = list(itertools.product((0, 1), repeat=arity))
    for combo in domain:
        a, b = old.decide_int(*combo), new.decide_int(*combo)
        if a != b:
            raise LibError(f"{name}: verdict changed at {combo!r}: {a} -> {b}")
    for probe in (2, -1, 7):
        for i in range(arity):
            combo = tuple(probe if j == i else 1 for j in range(arity))
            if old.decide_int(*combo) != new.decide_int(*combo):
                raise LibError(f"{name}: backstop changed at {combo!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="migrate_contracts")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check", action="store_true", help="read-only drift report (default)"
    )
    group.add_argument(
        "--write", action="store_true", help="rewrite formal blocks in place"
    )
    args = parser.parse_args(argv)
    write = args.write

    directory = _contract_dir()
    if directory is None:
        print("contract directory not found; checked:")
        for d in CONTRACT_DIRS:
            print(f"  {d}")
        return 1
    print(f"contract dir: {directory}")
    print(f"mode:         {'WRITE' if write else 'CHECK (read-only)'}\n")

    drifted: list[str] = []
    migrated = 0
    skipped: list[tuple[str, str]] = []
    refused: list[tuple[str, str]] = []

    for path in sorted(directory.glob("*.sl")):
        source = path.read_text(encoding="utf-8-sig")
        try:
            spec = contract_from_source(source)
        except LibError as exc:
            skipped.append((path.name, str(exc)))
            continue

        try:
            new_source = migrate_source(source)
        except LibError as exc:
            refused.append((path.name, str(exc)))
            continue

        if new_source == source:
            print(f"  {path.name:26s} already canonical")
            continue

        drifted.append(path.name)
        if not write:
            print(f"  {path.name:26s} DRIFT")
            continue

        try:
            _verify_equivalent(path.name, source, new_source, len(spec.slots))
        except LibError as exc:
            refused.append((path.name, str(exc)))
            print(f"  {path.name:26s} REFUSED: {exc}")
            continue

        # LF endings, no BOM: matches every existing contract on disk and keeps
        # the digest over the file text stable across platforms.
        path.write_text(new_source, encoding="utf-8", newline="\n")
        migrated += 1
        print(f"  {path.name:26s} migrated ({len(spec.slots)} slots)")

    print(f"\ncanonical:   {migrated + len(drifted) if write else len(drifted)}")
    if write:
        print(f"migrated:    {migrated}")
    print(f"drifted:     {len(drifted)}")
    print(f"bespoke:     {len(skipped)} (own policy rows; never touched)")
    for name, reason in skipped:
        print(f"  {name:26s} {reason[:70]}")
    if refused:
        print(f"REFUSED:     {len(refused)}")
        for name, reason in refused:
            print(f"  {name:26s} {reason[:70]}")

    if not write and drifted:
        print("\nrun with --write to normalise the drifted blocks")
        return 1
    if refused:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
