# Changelog

All notable changes to Netelpro (formerly Straylight) are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/); entries are headed by
commit hash until the first tagged release.

## v0.3.0 — Strings at the Native Boundary (2026-09-05)

### Added
- **Str (read-only) at the native boundary**: string literals intern as internal
  constant globals; `filter-rule` params may resolve to `Str` (i8* NUL-terminated
  UTF-8) and cross via `ctypes.c_char_p` (Python `str` encoded UTF-8 in `decide()`,
  `bytes` pass through). Strings are **inputs and comparisons, never products**:
  a bare Str in return position is a compile error (read-only boundary).
- **Type-aware equality**: `==`/`!=` dispatch on the compiled LLVM type — `icmp` for
  i64/i1, libc `strcmp` for pointers. Operands are statically homogeneous
  (TypeVar unification); heterogeneous comparison on concretely-anchored operands
  (Int vs Str) is prosecuted directly with exact coordinates.
- **`prefix?` primitive** (both engines): native = `strncmp(text, prefix,
  strlen(prefix)) == 0` (libc, resolved by the JIT dynamic linker like
  printf/exit); interpreter = `str.startswith` with Str-only prosecution.
  Registered in `spec/arity_table.json` (the parser fiscal consumes it).
- **`print` of strings** (native): `%s` format selected by LLVM type inspection;
  `(grant io)` still required.
- Differential test class `TestDifferentialStrings` (25 cases): strcmp/strncmp edges
  (empty strings, exact-prefix, shared-prefix-byte traps), unicode literals,
  string params through `let`, TCO loop consuming a Str param at 100k levels
  (pointer slot round-trips the back-edge), mixed Str/Int/Bool gates, and all
  v0.3 prosecutions. Bridge suite +7 (`TestStrParamsV03`): the house zone policy
  (8 real paths incl. unicode), per-param ctypes prototype assertions
  (c_char_p/c_bool/c_int64), return-Str rejection.
- `examples/zone_policy.sl`: the Neuromancer zone policy (red/yellow/green) as a
  pure Netelpro gate rule — the first use case that made v0.3 necessary.

### Changed
- `codegen.py`: `==`/`!=` inference no longer anchors to Int (homogeneous
  unification instead — `(== b true)` is legal Bool==Bool); heterogeneous
  concrete operands die with `type mismatch for '==' operands: Int vs Str`.
  `StrLit` in return position raises `cannot be a return value` (was "String
  literals are not supported").
- `rule_filter.py`: per-param audit accepts i64/i1/pointer; return type must be
  i1/i64 (return-Str prosecuted at compile time); per-param ctypes prototype
  (c_char_p for pointer params); `verify()` serializes Python strings as
  quoted literals with the language's escape rules (\\ \\\" \\n \\t).

### Prosecution (unchanged in spirit, extended in scope)
- Arity-2 `or`/`and` law of v0.1 held against the flagship use case (the fiscal
  rejected the 3-operand form of the zone policy; the rule was rewritten with
  nested `or`s — the language does not bend for its star application).
- Mixed-use of a param (Str anchored vs Int demand, or vice versa) remains a
  compile error with exact coordinates.
- Interpreter-only semantics deliberately diverge where documented: dynamic
  cross-type `==` returns False in the interpreter (reference semantics, tested)
  and is rejected at codegen when both sides are concrete (native gates refuse).

### Tests
- Suite: 356/356 green (322 prior + 34 new). Rule-filter bridge: 25 tests.
  House gate suite (consumer of the bridge, repo jona2428/neuromancer-teo):
  27/27 — backward compatibility verified against the production consumer.

## v0.2.0 — Bool Params at the Native Boundary (2026-09-05)

### Added
- **Bool params (i1) in `filter-rule`**: gate rules can now take Python `bool` arguments
  directly. A param used as an `if`/`and`/`or`/`not` operand compiles to an `i1` LLVM
  param and crosses the boundary via `ctypes.c_bool`; unused / Int-context params stay
  `c_int64`. Per-parameter prototypes are built from the compiled LLVM signature.
- Differential test class `TestDifferentialBoolParams` (17 cases): truth tables,
  multi-param mixes, TCO back-edge with an `i1` slot at 500k levels (native == interpreter),
  unused-param default (Int), and mixed-use conflict prosecution with exact coordinates.

### Changed
- `rule_filter.py`: the Int-only heuristics (`_has_non_int_param`, all-i64 audit,
  uniform `c_int64` prototype) were replaced by an audit of the **codegen-resolved**
  parameter types. The AST walk for "Bool-demanding" params is gone: the compiler's own
  inference is now the single source of truth (less duplication, fewer places to lie).
- `verify()` serializes Python bools as Netelpro literals `true`/`false` on the
  interpreter side (previously `str(True)` → `'True'`, not a token of the language —
  unreachable while params were Int-only, now load-bearing).

### Prosecution (unchanged in spirit, sharpened in scope)
- Mixed use of the same param (Bool demanded in one site, Int in another) remains a
  compile error with exact coordinates (`type mismatch`).
- The old test `test_non_int_param_rejected` was updated to the v0.2 contract:
  `(if b 1 0)` is now a legal Bool param; the conflict case takes its place.

### Tests
- Suite: 322/322 green (304 prior + 18 new). Rule-filter bridge: 19 tests.

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