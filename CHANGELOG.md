# Changelog

All notable changes to Netelpro (formerly Straylight) are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/); entries are headed by
commit hash until the first tagged release.

## v0.7.0 — Verification Theater Benchmark & Honesty Guard (2026-09-06)

### Added
- **Formal Whitepaper** (`docs/WHITEPAPER.md`): *Netelpro: Compiler-Enforced Epistemic Honesty for Autonomous LLM Agents*, formalizing cognitive counting grammar, honesty stack, and RLVR grounding.
- **Universal SDK** (`netelpro/guard.py`): `HonestyGuard` interface for Python agent frameworks (LangChain, CrewAI, AutoGen, Ollama) evaluating claims against machine tool evidence in microsecond LLVM native execution.
- **Verification Theater Benchmark (VTB)** (`benchmarks/`): 30 realistic test cases across FileSystem, SystemState, and CodeExecution evaluating false assertion acceptance rate (FAAR: 0.0% on Netelpro vs 100% baseline).
- **Unit test suite expansion** (`netelpro/tests/test_guard.py`): 4 tests validating rejection of false claims, approval of verified turns, and honest silences.

## v0.6.0 — Per-Function Effect Typing (2026-09-05, commit 2489dfd)

### Added
- **Effect inference** (`netelpro/effects.py`): static per-function effect sets via
  monotonic fixpoint over the closed call graph (no first-class calls ⇒ fully static).
  Effects = transitive closure of capability requirements (derived from the same
  `spec/arity_table.json` source of truth as `caps.py`). Direct and mutual recursion
  converge; non-convergence raises `EffectError`.
- **Gate purity law**: `check_gate_purity()` enforces that the decision entry
  (`filter-rule`) has an **empty effect set** — gate rules are pure decisions,
  machine-checkable by any LLM consumer. Impurity reports the exact **call chain**
  (`filter-rule -> helper -> sink -> print`) with source coordinates: the chain is
  the LLM's repair map.
- **Bridge enforcement** (`rule_filter.py`): `RuleFilter.__init__` runs purity as
  step 2b — an impure `filter-rule` is a compile error. `RuleBuilder`
  (`build-rule`) is exempt by design: producers are not judges.
- **MCP exposure** (`mcp_server.py`): `netelpro_compile` returns `effects` per defn
  in every response (success or failure) — an LLM can now read the effect set of a
  rule it is auditing without re-deriving it.

### Design notes
- Dead impure code does **not** pollute a pure entry's effect set: effects flow
  through calls, not through unreachable definitions (verified E2E: pure rule with
  a dead `(print ...)` helper compiles; a rule that *calls* the helper is rejected).
- Semantics decided by Teo under the no-questions mandate; adversarial review by
  antigravity-reasoning caught 3 blind spots pre-integration (Fn/def codegen
  restrictions, RuleBuilder entry name, top-level calls); antigravity-code built
  the self-contained module + 13 tests; Teo integrated, extended E2E (4 cases), and
  fixed a lint auto-fix regression in the BFS chain (tuple semantics corrupted).

## v0.4.0 — MCP Server (2026-09-05)

### Added
- **MCP server** (`netelpro/mcp_server.py`): stdio JSON-RPC 2.0 server exposing the
  compiler/evaluator/verifier to LLM clients — the language's first external
  interface. Line-delimited JSON transport (no Content-Length framing),
  `protocolVersion 2024-11-05`, zero external dependencies. Launched via
  `python -m netelpro --mcp` (new CLI flag) or `python -m netelpro.mcp_server`.
- Four tools: `netelpro_compile` (static parse/caps/holes audit, optional native
  codegen check without execution), `netelpro_eval` (subprocess-isolated execution,
  stdout capture, native opt-in), `netelpro_verify` (differential parity JIT vs
  interpreter across up to 100 cases), `netelpro_spec` (static language knowledge:
  forms, arities, capabilities, sorry-hole semantics). Errors keep the language
  taxonomy (`parse/cap/hole/runtime/limit/codegen`) and ride MCP `isError` with
  structured content, not protocol faults.
- **Fail-closed limits**: `MAX_SOURCE_BYTES=65536`, `PARSE_DEPTH_BUDGET=64`
  (bracket-nesting pre-check), `EVAL_TIMEOUT_S=3.0` (subprocess wall-clock kill →
  phase `limit`), `MAX_CASES=100`, `RESULT_STRING_CAP=65536`, and a stdio frame cap
  (`MAX_LINE_BYTES`) that rejects oversized lines **without buffering them** — a
  hostile 20 MB single line is answered with `-32600` and the session survives.
- **Subprocess isolation as containment**: `netelpro_eval`/`netelpro_verify` run in
  child processes, so TCO infinite loops die at the 3 s wall clock, native JIT
  deaths (non-tail-recursion stack overflow, div-by-zero `exit(1)`) are contained
  in the child, and the server process never executes untrusted source itself.
  Windows process-tree kill (`taskkill /T /F` + fallback), `--out` paths validated
  under the OS temp dir, verify args type/size-validated (int/bool/str, 4096-char
  string cap) before reaching the worker.
- Adversarial corpus `netelpro/tests/test_mcp_adversarial.py` (21 cases, contract-
  first, watchdogs so no case can hang CI): TCO loop → `limit`, exponential
  `str-cat` bomb, 1000-deep nesting, oversize source, 101 verify cases, string args
  with quotes/newlines/escapes, protocol edges (malformed JSON → `-32700`, unknown
  method → `-32601`, `id: null`, wrong version), session statelessness.
- `docs/MCP.md`: wire contract, launch, tool schemas, limits table, client
  integration snippet, and the adversarial suite map.

### Security
- Independent threat model + audit (findings C1/C2/H1–H4/M2/M5/M6 fixed in this
  release): unbounded `readline()` replaced by a chunk-capped line reader; worker
  result paths confined to the temp dir; notification semantics corrected
  (notifications never respond); native JIT result wrapped via `to_json_val` so it
  is always JSON-serializable.

### Notes
- Grammar detail surfaced by the adversarial corpus: `defn` parameter lists use
  parentheses `(defn f (x) ...)` — corpus fixed to match, no language change.
- Repository CI runs pytest only (3.11–3.13 matrix); the MCP corpus runs inside it
  via `pytest netelpro/tests/` and skips cleanly if `llvmlite` is absent (native
  paths are opt-in).

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