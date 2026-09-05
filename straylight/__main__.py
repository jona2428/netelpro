"""Straylight CLI entrypoint -- python -m straylight <file.sl>."""
from __future__ import annotations

import sys
from pathlib import Path

from straylight.evaluator import (
    StrayError,
    StrayHoleError,
    StrayRuntimeError,
    evaluate,
    format_value,
    is_nil,
)
from straylight.caps import check_capabilities, collect_grants
from straylight.parser import parse


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint for evaluating Straylight programs."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or len(argv) != 1:
        print("Usage: python -m straylight <file.sl>", file=sys.stderr)
        return 1

    file_path = Path(argv[0])
    if not file_path.exists():
        print(f"Error: file '{file_path}' not found", file=sys.stderr)
        return 1

    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        return 1

    parse_result = parse(source)
    if not parse_result.ok:
        for err in parse_result.errors:
            print(f"line {err.line}, col {err.col}: {err.message}", file=sys.stderr)
        return 1

    # Phase 3: static capability enforcement — un-granted IO is a COMPILE error.
    granted = collect_grants(parse_result.program)
    cap_errors = check_capabilities(parse_result.program, granted)
    if cap_errors:
        for err in cap_errors:
            print(str(err), file=sys.stderr)
        return 1

    try:
        val = evaluate(parse_result.program)
    except (StrayRuntimeError, StrayHoleError) as e:
        print(str(e), file=sys.stderr)
        return 1
    except StrayError as e:
        print(str(e), file=sys.stderr)
        return 1

    if not is_nil(val):
        print(f"=> {format_value(val)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
