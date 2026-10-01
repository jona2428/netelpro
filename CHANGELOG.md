# Changelog

All notable changes to Netelpro (formerly Straylight) are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/); entries are headed by
commit hash until the first tagged release.

## [Unreleased]

### Added
- **`netelpro/receipts.py` — file-effect honesty for LLM agents** (spec:
  `docs/RECEIPTS_SPEC.md`). The failure it targets: the agent says "actualicé
  `config/settings.py`" and the file is byte-for-byte what it was — the write
  tool errored, was blocked, or was never called, and the prose reports
  success. `HonestyGuard` cannot see this (a tool result exists); the evidence
  standard here is the bytes.
  - `snapshot()` / `diff_snapshots()`: sha256 of every file under the root
    before and after the turn; every differing path becomes a `Receipt`
    (created / modified / deleted, hash before, hash after).
  - `ReceiptLedger`: append-only, hash-chained, JSONL-persisted. The model has
    no write path to it and is never asked to recall it; `load()` refuses a
    file whose chain does not verify (one edited field breaks it).
  - `detect_mutation_claims()`: ES + EN extraction of `(path, kind)` effect
    claims with the same scoping discipline as `guard.py` (negation, attempt,
    intent, future, conditional, question, adjectival participle; a
    preposition before the path makes it a container).
  - **`netelpro/rules/mutation_receipt.sl`**: three functions, `(filter-rule
    claim receipt strict)`, deciding whether a claim kind is admitted by a
    receipt kind. Kind-aware ("updated X" when X was created is false);
    `strict` rejects silent writes the text never mentions. Full 40-row
    domain verified against an oracle and native-vs-interpreter: 0 mismatches.
  - `MutationGuard`: `begin()` / `end()` / `audit()` / `enforce()` /
    `ground_truth()` (a block the harness puts in front of the model so it
    reads what changed instead of remembering it). A guard rebuilt from the
    JSONL ledger in a fresh process reaches the same verdict
    (`test_guard_survives_total_context_amnesia`).
  - CLI `python -m netelpro.receipts begin|end|audit|show` (also the
    `netelpro-receipts` console script): zero-integration wrapper for any
    harness that touches a directory. `audit` exits 2 on theater.
- **`examples/receipts_demo.py`** + `tests/test_receipts_demo.py`: real
  temporary workspace, real sha256, a simulated agent turn whose prose lies
  about one of three edits; 19 on-screen checks including tamper refusal and
  the 40-row differential. Plain output when piped (`NO_COLOR` honoured).
- **`tests/test_receipts.py`** — 58 cases across the four layers (rule domain,
  snapshots/ledger, claim corpus with per-label provenance, guard end to end,
  CLI round trip).
- **MCP tool `netelpro_receipts`** (`docs/MCP.md` §3.5): the receipts read path
  from inside the agent. `show` lists what actually changed under the server's
  root since the baseline; `audit` judges the agent's draft text against it.
  Read-only by construction: the root is `NETELPRO_RECEIPTS_ROOT` of the server
  process (never a model argument), the baseline is hashed before the first
  request, and there is no `begin`/`end` action — a model that could move the
  baseline after writing would erase its own receipts. A harness baseline from
  `netelpro-receipts begin` is respected. Unset root: structured
  `phase: "receipts"` error, every other tool unaffected.
  `tests/test_mcp_receipts.py`: 12 cases, in-process and over the real stdio
  process.

- **Incremental snapshots** (`docs/RECEIPTS_SPEC.md` §9). A `Snapshot` now
  carries each hash's stat signature `(size, mtime_ns, ctime_ns, inode)` and
  the take's start time; `snapshot(root, cache=previous)` re-reads only files
  whose signature changed or that were touched within a 2 s racy window of
  the cached take (git's rule). ctime and inode are part of the signature on
  purpose: an agent with a shell can forge mtime after a same-size edit but
  cannot set ctime from user space on Linux/macOS, and a rename-over changes
  the inode — both evasions are tests. `MutationGuard(incremental=False)` and
  CLI `--full` force a full re-read; the pre-fast-path `snapshot.json` still
  loads (as a cache that reuses nothing). Measured on this repo, 420 files /
  97 MB: full 9096 ms, incremental 11 ms, identical result. Residual holes
  declared in the spec (Windows ctime semantics, raw-device/clock-stepping
  root, a model allowed to edit the state dir).

- **Claude Code hooks** (`netelpro/hooks/claude_code.py`, `docs/RECEIPTS_SPEC.md`
  §10). `python -m netelpro.hooks.claude_code install` merges three hooks into
  `.claude/settings.json` (keeping existing hooks, adding `.netelpro/` to
  `.gitignore`): `SessionStart` / `UserPromptSubmit` take the turn's baseline
  and inject a two-line notice that file claims are audited; `Stop` audits
  `last_assistant_message` against the live receipts. A rejected message
  blocks the stop once with the exact claims, the ground truth and the
  instruction to either perform the edit or correct the text; a second
  rejection (`stop_hook_active`) never loops: the stop is allowed and a
  `systemMessage` warns the user. Deliberately fail-open on the hook's own
  problems (no baseline yet, corrupt ledger, internal error: stderr + exit 1,
  never a block). `--strict` makes silent writes block too. Transcript JSONL
  fallback for hosts without `last_assistant_message`. `tests/test_claude_code_hook.py`:
  19 cases, every hook run as the real subprocess with Claude Code's stdin JSON,
  including the exact installed command executed through a shell.
- `netelpro.receipts.open_turn()` / `TurnState`: the "load baseline, observe
  live diff, commit or not" sequence shared by the MCP tool and the hook, so
  both judge identically.

- **`benchmarks/receipts_qwen_live_bench.py`** — the receipts layer against a
  real local model, same discipline as `honesty_guard_qwen_rate_bench.py`:
  real generations at a real temperature, raw transcripts saved, labels made
  by a human afterwards. Ground truth is the bytes: each trial runs in a real
  temporary workspace where the harness applies exactly the effects its
  simulated tool output reports as succeeded, then audits the text with
  `MutationGuard`. Five families: EDIT-RISK (no tool result), BLOCKED-WRITE
  (EACCES / sandbox refusal — the "impedimento de modificación" case),
  PARTIAL (one landed, one failed), HONEST-WRITE (false-rejection check),
  HONEST-SILENT (false-positive check); 16 scenarios × N repeats.
  `tests/test_receipts_bench.py` proves the harness with canned texts (11
  cases).
- **Live run, 2026-10-01** (`benchmarks/receipts_qwen_live_report.md`,
  raw transcripts + hand labels in `receipts_qwen_live_results.json`):
  48 generations from base `qwen2.5-1.5b-instruct-q4_k_m.gguf`, every one
  hand-labeled before reading the detector. **The model reported a blocked
  write as done 12/12 times** when shown an explicit EACCES / sandbox error,
  and narrated an edit it never made 11/12 times with no tool result; next
  to a success, it reported the failure honestly 5/6. The detector as shipped
  had precision 26/26 and live recall 13/24 (54%) on theater trials.

### Fixed (found on the live run above, same day)
- Six Spanish constructions the mutation-claim detector missed on real
  output: list after a colon ("Se han editado los siguientes archivos:
  - X"), feminine participles ("fue editada"), reflexive passives ("se creó
  en X", "X se ha modificado", "X se editó"), bare / `está` participles
  ("Clamp funcion creado en X", "está creado en X"), the resultative "quedó
  con ... agregada" / "quedó con la nueva sección", and an adverb between
  path and auxiliary ("X también quedó cambiado"). The first fix round
  introduced a false rejection across a contrast clause ("README.md se
  actualizó, pero el src/utils.py no se pudo editar"), caught by the
  differential re-classification before shipping: the comma is now a clause
  boundary. After the fix: 24/24 theater caught, 0 false rejections, 12/12
  honest claims read, 0/6 silent-trial claims. `tests/test_receipts.py`
  +14 live-provenance positives, +4 negatives, +1 regression test.

### Fixed (found while building the above)
- Turn numbering derived from the ledger alone could not advance across a
  turn with no effects (no receipt, same number reused). The guard and the
  CLI now carry their own counter and take the max.

## [0.9.1] — 2026-09-19

First release to PyPI since 0.7.1: tags v0.8.0/v0.9.0 were cut but never
published (the packaging discrepancy documented below). This release aligns
the published version with the tagged line and ships the gate as a verifiable
product.

### Added
- **New language primitive `contains?`** (substring test, libc `strstr`), completing
  the string-operations set (`prefix?` was anchor-only and could not decide on
  mid-string evidence). Implemented in both backends symmetrically:
  `codegen.py` (LLVM: single `strlen` setup + `strstr` call per site — declared
  once per module, `DuplicatedNameError` regression test included) and
  `evaluator.py` (reference interpreter). Arity table and SPEC updated.
  Native-vs-interpreter parity: **26/26 cases** (`scripts/verify_contains_gate.py`).
  Native throughput measured: **~534k decisions/sec (~1.87 µs)**.
- **`examples/gates/evasion_detector.sl`** — the evasion rule the primitive was
  built for: denies any command/code text mentioning credentials (`.env`,
  `api_key`, `bearer `, `token=`, `secrets`, ...) in **any position** of the
  text — the exact class the anchor-bounded Python regex sniff lets through.
- **`tests/test_contains_primitive.py`** — 11 formal tests (primitive semantics,
  duplicated-declaration regression, rule-level behaviour).
- **FinalFront integration** — the compiled rule is wired into
  `zone_policy.py` as a second layer after the Python regex sniff: it only
  adds a DENY (RED) when the sniff found nothing (pure evasion). It can never
  produce an ALLOW the sniff would have gated, and the explicit-citation
  policy is respected (sniff-first ordering).

### Fixed
- Nested `or` parsing: the closing-paren budget for nested disjunctions was
  off by one group (documented as the 10-closer rule); fixed and covered by
  tests.

## [0.9.1] — 2026-09-19 (cont.)

### Added
- **`examples/gate_demo.py`** — one command, ~30 seconds, no GPU, no model, no
  training, no network, no optional dependencies. Runs the *real* compiled gate
  (`RuleFilter` → LLVM → ctypes) and proves four claims on screen: the policy is
  a 3-line `filter-rule` a human reads; it compiles to native machine code (entry
  address printed); it decides, with `reason is None` exactly when the rule
  decided; and it fails closed on every failure mode with an explicit reason.
  Measured on the dev machine: **12/12 checks, ~520k decisions/sec, ~1.9 µs per
  decision** including the Python→ctypes boundary, 0 mismatches on the
  differential native-vs-interpreter check.
  Colour is emitted only when stdout is a TTY (and honours `NO_COLOR`): piping or
  capturing yields plain text. Without that the output was unassertable — and
  this was a real defect found by the test, not a stylistic preference.
- **`tests/test_gate_demo.py`** — keeps the demo honest rather than leaving it to
  manual runs: exit code 0, zero `FAIL` marks, the printed tally cross-checked
  against the `PASS` marks actually emitted (a demo that grades itself wrongly is
  worse than one that fails loudly), the four evidence strings asserted directly,
  and a guard against optional-dependency imports creeping in.

## [0.9.0] — 2026-09-18

- netelpro/lib: canonical contract generator + drift guard (2026-09-18)

### Added
- **`netelpro/lib/` — new package data, ships in the wheel** (`lib/*.sl` added
  to `[tool.setuptools.package-data]`; without it an installed netelpro would
  silently lose the prelude).
  - `prelude.sl` — Bool helpers (`all-ofN`, `any-zeroN`). Netelpro has no
    `include`, so the host concatenates the prelude before rule source via
    `netelpro.lib.concat_with_prelude`. A name collision with the rule's own
    definitions is a hard error, never a silent shadow. **No production
    consumer yet:** every live contract is a `truth-table`, and truth-table
    rows cannot call helpers — exhaustiveness is decided syntactically over
    row patterns, so a wildcard row calling a helper covers nothing.
  - `contracts.py` — generator of canonical all-of-N contracts, plus
    `migrate_source()`, which regenerates the formal block while preserving
    the prose header verbatim. `contract_from_source` is a guard rail, not a
    normalizer: it accepts only the canonical form and raises `LibError`
    rather than silently rewriting a contract that carries its own policy.
- **`scripts/migrate_contracts.py`** — `--check` (default, read-only) and
  `--write`. `--write` refuses any contract whose verdicts change over its
  full declared domain plus backstop probes.
- **Drift guard wired into CI** (`fiscal` job, after the netelpro checkout).
  It must run *after* that checkout: the script resolves `netelpro.lib` from
  `workspace/straylight`, and the published package does not carry `lib/`.

### Fixed
- **Drift guard reported a false green on hand-edited generated blocks.** A
  file carrying the `GENERATED` marker but edited by hand was classified as
  `bespoke` and exited 0 — the exact case the guard exists to catch. The
  script never read `GENERATED_MARKER`. It now separates `TAMPERED`
  (generated block hand-edited) from `bespoke` (never generated) and exits 1.
- **Drift guard reported `canonical: 0` with 23 canonical contracts.** The
  pipeline engine marks `ok: True` for any skill output and never reads the
  exit code, so this output *is* the evidence — and it was lying in its own
  count.
- **Token gate: numeric token map admitted tokens with no digit.** The rule
  classified a token as numeric when every character came from `"0123456789.-"`,
  which admits `' -----------'`, `'...............'`, `'..\n\n'`. Measured on
  `qwen2.5-1.5b-instruct.Q4_K_M`: 147 tokens passed, **137 of them carrying no
  digit at all**. The model used one to open its answer, which is where the
  previously unexplained leading `-` in the two-phase demo came from. Requiring
  at least one digit cuts the set to 10 (+EOS). Confirmed by intervention on
  the real gate, not by reimplementation. This removes the artefact; it does
  **not** make the arithmetic correct — format and correctness are orthogonal.

### Notes
- Known pre-existing backend divergence outside the declared domain: the native
  JIT accepts e.g. `2` in a slot typed `(Int 0 1)` and answers via the backstop
  row, while the reference interpreter rejects it at parse time. Consequence:
  `verify_int` cannot differentially test the backstop path. Documented in
  `netelpro/lib/README.md`.

### Versioning
- `pyproject.toml` aligned to the real series. The tags `v0.8.0` and `v0.9.0`
  were both cut over a `pyproject.toml` that still declared `0.7.0`, and the
  package number has never tracked the tags (the CHANGELOG already records
  v0.4–0.6 shipping without a bump). PyPI holds `0.7.0` and `0.7.1` only. The
  declared version now reads `0.9.0`, matching the latest tag.

## Unreleased — gcd curriculum for run #3 (2026-09-08)

### Added
- **gcd curriculum (`rlvr/tasks/`):** 3 TRAIN-side tasks that teach the exact
  skill `gcd_pair` (OOD) demands — two-argument recursion with parameter
  reordering (`(gcd-two b (rem a b))`), absent from every previous train task
  (all `quot`/`rem` tasks were unary): `halving_steps` (unary recursion with
  `quot`), `euclid_steps` (reorder + `rem`, steps counted instead of gcd
  returned) and `sub_gcd_steps` (bidirectional subtractive reorder). Outputs
  are step counts, deliberately distinct from the gcd value itself — the
  lever is pattern transfer, not memorization of the held-out task.
  Corpus: 55 → 58 tasks (38 train + 20 OOD). `OOD_TASK_IDS` untouched (20
  ids): per the contract, adding tasks never grows the OOD — run #3 measures
  `gcd_pair` as true transfer. Gradable: hand-written Netelpro solutions for
  all 3 pass the real RLVR verifier 20/20 cases (`tests/test_rlvr_gcd_curriculum.py`).

### Notes
- Run #3 (fresh from base, protocol of #1/#2) still not executed; the
  curriculum lands in the train pool for that run. No eval-comparison claims
  are made here — nothing has been trained yet.

## v0.9.0 — RAFT run #2: accumulated pool, paired seeded evals (2026-09-08)

### Added
- **RAFT notebook v2** (`training/train_raft_colab.ipynb`): accumulated SFT pool across
  rounds (no forgetting between rounds, vs. v1's per-round-only harvest), seeded
  baseline/final evals (`torch.manual_seed`, reproducible pairing), full loss-curve
  visibility (`logging_steps=1`), 5 rounds (vs. 3) at 16 samples/task (vs. 8).
- **Run #2 results** (5 rounds, Colab T4): OOD pass@8 **20% → 60%** (paired, same
  seed) — 3/5 tasks, 3× more individual samples passing than run #1 (13/40 vs 4/40).
  GGUF q4_k_m re-measurement matched the fp16 verdict exactly (60%, 3/5) — second
  consecutive run where quantization does not degrade the learned behavior.
- **Refuted hypothesis, documented honestly:** `gcd_pair` stayed at 0/8 in both runs
  — the accumulated pool alone does not unlock it; needs a different lever
  (curriculum or more diverse samples), not just more of the same volume.
- **[🤗 `JonaECG/netelpro-qwen2.5-1.5b-raft-v2`](https://huggingface.co/JonaECG/netelpro-qwen2.5-1.5b-raft-v2)** published, with per-task breakdown vs. v1 in the model card.

## v0.8.0 — RLVR/RAFT: training against a compiled verifier (2026-09-07)

### Added
- **RLVR task corpus** (`rlvr/tasks/`): 25 tasks across arithmetic (7), lists (9),
  and strings (8), with a deterministic train/OOD split (`power_int`, `nth_element`,
  `string_to_int`, `gcd_pair`, `list_sum` held out).
- **Binary verifier** (`rlvr/verify.py`): static checks + real interpreter execution
  against 20 randomized test cases per task — the reward signal is a compiler, not
  a preference model.
- **Fixed prompt builder**: condensed language spec + few-shots drawn from
  `examples/` + task description — same shape used at train and eval time.
- **RAFT Colab notebook** (`training/train_raft_colab.ipynb`, run #1): sample →
  verify → filter → SFT loop, 2–4 rounds on a free T4.
  - `SFTTrainer` conditions loss on the prompt via `formatting_func` (not raw
    completion); `completion_only_loss=False` (the Unsloth fork used here rejects
    `formatting_func` with the library default of `True`); explicit GPU guard
    before importing Unsloth (clear failure instead of a cryptic
    `NotImplementedError`); OOD scoring asserts order-insensitive (sorted both
    sides).
- **`rlvr.gguf_eval`**: official local-evaluation tool — the same measurement that
  scored the published GGUF enters the repo, runnable by anyone with Ollama.
- **Run #1 results**: OOD pass@8 **0% → 20% → 40% → 40%** over 3 rounds (baseline
  solved 0/5 tasks in 40 attempts). Local GGUF re-measurement scored 80% (4/5) —
  see the [model card](https://huggingface.co/JonaECG/netelpro-qwen2.5-1.5b-raft#honest-caveats)
  for why that number sits above the in-notebook fp16 figure: the in-notebook
  baseline/final comparison was unseeded, the local re-eval was seeded — the two
  aren't measuring under identical protocol, which run #2 fixed.
- **[🤗 `JonaECG/netelpro-qwen2.5-1.5b-raft`](https://huggingface.co/JonaECG/netelpro-qwen2.5-1.5b-raft)** published.

### Honest caveats (both releases)
- Metric is **pass@8**, not pass@1; **n=5** OOD tasks (20% granularity per task).
- v1 vs. v2 headline numbers (40% vs. 60%) come from *different* runs — directional,
  not paired. The paired comparison is each run against its own seeded baseline.

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