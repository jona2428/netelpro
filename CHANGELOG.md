# Changelog

All notable changes to Netelpro (formerly Straylight) are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/); entries are headed by
commit hash until the first tagged release.

## 110d644 — Rebrand to Netelpro (2026-09-05)

### Changed
- Package renamed `straylight/` → `netelpro/`; all imports, spec references, examples and tests updated (22 files, rename with history preserved via `git mv`).
- GitHub repo renamed to `jona2428/netelpro`.

### Tests
- Suite verified green after rebrand: 304/304.
- CLI verified on both engines: `python -m netelpro --native examples/native_print.sl` → `42`; `python -m netelpro examples/sum_to.sl` → `=> 5000050000`.

## 7d09862 — Spec Consolidated to v0.9 (2026-09-05)

### Changed
- `docs/SPEC.md` (renamed from versioned filename): status no longer "draft" — reflects verified reality (phases 0–6, 304/304 tests, two backends).
- New §15 "Consolidated State": the four-layer honesty stack, deliberate v0.1 limits, and the open decision — the final language name.

## a70d27c — Phase 6: Rule Filter Bridge (2026-09-05)

### Added
- `netelpro/rule_filter.py`: Neuromancer gate rules as compiled pure Netelpro functions — Python calls native code via `ctypes`; differential verification against the interpreter (18 tests).
- `examples/gate_rule.sl`: real gate rule (priority + confidence + escalation → decision).
- Spec §14: the rule-filter bridge.

### Verified
- `decide(3, 80, 0) → True` from Python into native machine code.
- 1001 levels of native recursion in constant stack (structural TCO).
- Zero mismatches between native and interpreter across the bridge suite.

## b640922 — Phase 5: LLVM Native Backend (2026-09-05)

### Added
- `netelpro/codegen.py` (llvmlite 0.49): `i64`/`i1`, structural TCO via back-edge (not optimizer-dependent), bidirectional type inference with unification, caps enforcement at codegen, native `printf` IO, div-by-zero → `exit(1)`.
- `--native` CLI flag (`python -m netelpro --native file.sl`).
- 84 differential tests: every program runs on both engines (interpreter = reference semantics, native = verified implementation); zero mismatches.
- Spec §13, example `native_print.sl`.

### Verified
- `sum-to 100000` → `5000050000` in constant stack, both engines.
- `fib 15` → `610` both engines.
- Native print writes to real fd 1 (subprocess-verified; `capsys` cannot see fd-level output).

## be363b0 — Phase 4: Static Hole Prosecution (2026-09-05)

### Added
- `netelpro/holes.py`: the `sorry` manifest — the only legal unimplemented branch is `(sorry "reason")`; every declared hole is listed with `line:col` + reason on stderr at every compilation. Silent holes are impossible.
- 24 tests; CLI emits the holes manifest.
- Spec §12 corrected to parser law.

### Verified
- The fiscal parser already enforces no-silent-holes at parse time: unknown heads, duplicate top-level definitions, and no first-class calls are rejected before any static pass.

## 4f512d3 — Phase 3: Capabilities as Types (2026-09-05)

### Added
- `netelpro/caps.py`: static capability pass — any capability use requires a top-level `(grant ...)`; IO requires `(grant io)`. Violations are compile-time errors with exact coordinates, aggregated (all uses reported, not just the first), detected even in never-called functions.
- Runtime guard as defense-in-depth (if the API is used bypassing the static pass).
- CLI integration: parse → caps → evaluate.
- 14 tests, spec §11, examples (`hello_io.sl`, `needs_grant.sl`).

### Verified
- `needs_grant.sl` dies at compile time: `line 2, col 1: capability 'io' required by 'print' but not granted`.

## a374237 — Chore: Cleanup & .gitignore (2026-09-05)

### Changed
- Removed stray artifacts (`.pyc`, `__pycache__`, `.bak`); added `.gitignore`.

## 3e17883 — Phases 0–2: Initial Checkpoint (2026-09-05)

### Added
- Grammar spec v0.1: prefix, fixed arity, machine-consumed `spec/arity_table.json` (single source of truth).
- Hand-written lexer; fiscal recursive-descent parser (593 lines) — unknown heads, arity violations, duplicates, unclosed/stray parens: all parse-time errors with exact `line:col`.
- Frozen-dataclass AST.
- TCO evaluator with closures, environment model, iterative tail calls.
- CLI (`python -m netelpro file.sl`).

### Tests
- 25 fiscal-parser tests; full suite 163/163 green.

### Verified
- `fib 15` → `610`; `sum-to 100000` → `5000050000` (interpreter, TCO).