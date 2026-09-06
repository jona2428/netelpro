"""Zone-rule BUILDER adapter: the Netelpro language writes its own gate rule.

Session 3 of the auto-hosting plan. This module is the host-side frontier:
it canonicalizes + escapes zone roots (frontier work the language cannot do),
invokes the compiled Netelpro builder (`builders/zone_rule_builder.sl`), and
returns the generated `.sl` rule source text. Byte-parity with the reference
Python generator (zone_rule_generator.py) except the provenance line, which
honestly names the builder as author.

Fail-safe: every failure path returns None -> callers keep their Python
fallback (zone_rule_generator.py stays as defense in depth).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = None
_STRAYLIGHT: Path | None = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "workspace" / "straylight").is_dir():
        _STRAYLIGHT = _parent / "workspace" / "straylight"
        _REPO_ROOT = _parent
        break

if _STRAYLIGHT is not None and str(_STRAYLIGHT) not in sys.path:
    sys.path.insert(0, str(_STRAYLIGHT))

try:
    from netelpro.rule_filter import RuleBuilder
except ImportError:  # pragma: no cover - fail-safe
    RuleBuilder = None  # type: ignore[assignment]

_BUILDER_PATH = (_STRAYLIGHT / "builders" / "zone_rule_builder.sl") if _STRAYLIGHT else None
_BUILDER_SLOTS_PER_ZONE = 6


def _escape_sl_string(value: str) -> str:
    """Escape characters for .sl string literals (reference-generator convention)."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def _pack_zone(roots: list[str]) -> list[str]:
    """Pack up to 6 canonicalized+escaped roots into 12 args (exact, prefix pairs)."""
    if len(roots) > _BUILDER_SLOTS_PER_ZONE:
        return []
    args: list[str] = []
    for i in range(_BUILDER_SLOTS_PER_ZONE):
        if i < len(roots):
            r = roots[i]
            exact = _escape_sl_string(r)
            prefix = _escape_sl_string(r if r.endswith("/") else f"{r}/")
            args.extend([exact, prefix])
        else:
            args.extend(["", ""])
    return args


def build_zone_rule_source(
    red_paths: list[str],
    green_roots: list[str],
    yellow_roots: list[str],
) -> str | None:
    """Generate the zone-policy rule .sl source via the Netelpro builder.

    All arguments are ALREADY canonicalized (lowercase, forward slashes,
    deduped, sorted) — mirroring the contract of get_zone_rule_filter.
    Returns None on any failure (caller falls back to Python generator).
    """
    if RuleBuilder is None or _BUILDER_PATH is None or not _BUILDER_PATH.is_file():
        return None

    red_args = _pack_zone(red_paths)
    green_args = _pack_zone(green_roots)
    yellow_args = _pack_zone(yellow_roots)
    if not red_args or not green_args or not yellow_args:
        return None  # capacity exceeded -> caller falls back

    try:
        source_text = _BUILDER_PATH.read_text(encoding="utf-8")
        builder = RuleBuilder(source_text)
        built = builder.build(*red_args, *green_args, *yellow_args)
        return str(built)
    except Exception:
        return None


__all__ = ["build_zone_rule_source"]