# Netelpro

**A programming language for LLMs — honest, semantic, universal.**

[![CI](https://github.com/jona2428/netelpro/actions/workflows/ci.yml/badge.svg)](https://github.com/jona2428/netelpro/actions/workflows/ci.yml)

*(formerly Straylight — Netelpro: **NE**uron **TEO** **L**anguage **PRO**gramming)*

Netelpro is a programming language written by LLMs and audited by a compiler that behaves as a prosecutor. Its grammar is engineered so that an LLM can verify its own syntax mechanically: every form is `(head arg1 arg2 ...)` with a declared arity, so checking a form means counting the operands between the head and the closing parenthesis. Counting is a mechanical operation an LLM performs reliably; simulating a recursive-descent parser is not.

The thesis: **an LLM can verify its own syntax by counting, not simulating** — and nothing unverifiable passes silently. Unknown heads, arity violations, silent holes, and ungranted IO are all compile-time errors reported with exact `line:col` coordinates. A Netelpro program runs only after surviving every layer of prosecution.

## The Honesty Stack

Four verified layers. Each failure class dies at the earliest layer, with exact coordinates. There is no stage where dishonesty passes silently.

| # | Layer | Stage | Kills |
|---|-------|-------|-------|
| 1 | **Fiscal parser** | parse time | Unknown heads, duplicate top-level definitions, arity violations — every form audited against `spec/arity_table.json`, the machine-consumed single source of truth |
| 2 | **Capabilities as types** | static pass | Any capability use without a top-level `(grant ...)` — IO requires `(grant io)`, enforced statically, even when buried in unexercised branches |
| 3 | **Sorry manifest** | static pass | Silent holes are impossible: the only legal unimplemented branch is `(sorry "reason")`, and every declared hole is listed with `line:col` + reason on stderr at every compilation |
| 4 | **LLVM native backend** | codegen | Non-representable types at use (`Float`/`List`/fn-as-value) and boundary type violations; `llvmlite 0.49`, `i64`/`i1`/`i8*`, structural TCO |

Gate rules (the Phase 6 bridge) take `Int` (i64), `Bool` (i1) and — since v0.3 —
`Str` (i8*) params: a Bool-used param crosses as a native 1-bit flag
(`ctypes.c_bool`); a Str-used param crosses as a **read-only NUL-terminated UTF-8
pointer** (`ctypes.c_char_p`), comparable with type-aware `==`/`!=` (libc `strcmp`)
and `prefix?` (libc `strncmp`), printable with `%s`. Strings are inputs and
comparisons, never products: return-Str is a compile error. Mixed use of the same
param is a compile error with exact coordinates.

The prosecutor's voice is a product feature. Real output from `examples/broken_arity.sl`:

```text
line 3, col 2: '+' expects 2 operand(s), found 3
line 4, col 2: 'if' expects 3 operand(s), found 2
line 6, col 2: 'add' expects 2 operand(s) (declared by defn), found 1
line 7, col 2: unknown head 'unknown-op' (not in the arity table and not a declared defn)
line 8, col 14: duplicate parameter 'p'
line 9, col 1: 'sorry' requires a string literal reason
```

## Quick start

```bash
git clone https://github.com/jona2428/netelpro.git
cd netelpro
pip install -e ".[dev]"    # llvmlite 0.49 included; `dev` extra adds pytest
```

Requires Python 3. The interpreter needs nothing beyond CPython; the native backend needs `llvmlite 0.49`.

```bash
python -m netelpro file.sl           # interpreter — reference semantics
python -m netelpro --native file.sl  # compiled native — LLVM JIT, same static passes
```

`examples/fib.sl`:

```netelpro
; Netelpro v0.1 -- Fibonacci demonstration

(defn fib (n)
  (if (< n 2)
      n
      (+ (fib (- n 1)) (fib (- n 2)))))

(fib 15)
```

Both engines agree:

```text
$ python -m netelpro examples/fib.sl
=> 610
$ python -m netelpro --native examples/fib.sl
=> 610
```

IO is a capability, granted file-wide and top-level only (`examples/hello_io.sl`):

```netelpro
(grant io)
(print "hello, netelpro")
```

Tail calls compile to structural loops in the native backend: recursion verified at 1001+ levels in constant stack — no stack growth.

## Differential testing

Every program runs on **both engines**, by contract:

- **Python interpreter** — the reference semantics.
- **LLVM native backend** — the verified implementation.

The native backend is a strict subset compiler: it accepts only programs whose values are representable in machine words and rejects everything else with a prosecutorial compile error — never a silent fallback, never a silent divergence. The test suite runs every program through both engines and compares them: zero mismatches (see the CI badge above for the current pass count — kept out of this prose so it can't go stale). The same principle is exposed programmatically by the Phase 6 bridge: `RuleFilter.verify(cases)` returns any `(args, expected, interpreted, native)` mismatches; an empty list means full agreement.

## Phase history

| Phase | Delivered |
|-------|-----------|
| **Fase 0** | Grammar spec + embryonic fiscal: operand counting against the arity table proves full structural validity without simulating a parser |
| **Fase 1** | Full frontend: hand-written lexer, frozen typed AST, recursive-descent parser with first-class positional diagnostics |
| **Fase 2** | Tree-walking evaluator with tail-call optimization and closures; strict `Bool`/`Int` discipline; `python -m netelpro` CLI |
| **Fase 3** | Capabilities as types: static capability pass — IO requires `(grant io)`, ungranted IO is a compile error |
| **Fase 4** | Sorry prosecution: declared holes enumerated in a manifest (`line:col` + reason on stderr); silent holes impossible |
| **Fase 5** | LLVM native backend: `llvmlite 0.49`, `i64`/`i1`, structural TCO, JIT invocation via ctypes |
| **Fase 6** | Neuromancer rule-filter bridge: `compile_filter` compiles real Neuromancer gate rules to native code, called from Python via ctypes |

The Phase 6 use case, `examples/zone_policy.sl` (v0.3) — the Neuromancer zone policy as a compiled pure function over real path strings:

```netelpro
(defn filter-rule (path approved mode)
  (if (or (== path ".env") (or (== path "routes.py") (== path "container.py")))
      false
      (if (or (prefix? path "src/") (or (prefix? path "tests/") (prefix? path "skills/")))
          (and approved (== mode 1))
          true)))
```

## Verification Theater Benchmark (VTB) & Empirical Alignment

Netelpro includes a native benchmark measuring **Verification Theater** (agents claiming empirical verification without executing tools) across 30 real-world deceptive scenarios covering FileSystem, SystemState, and CodeExecution:

```bash
python -m benchmarks.vtb_runner
```

### Empirical Results: Base vs. Netelpro Post-DPO (Ollama Local)

Evaluated under identical local execution environments across the 30 standardized VTB scenarios:

| Architecture | Model ID | Epistemic Honesty | Verification Theater (FAAR) | Primary Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Transformer** | `qwen2.5:1.5b` (Base) | 46.7% | 10.0% (3/30 false claims) | Baseline |
| **Transformer** | [🤗 `JonaECG/netelpro-qwen2.5-1.5b-honest`](https://huggingface.co/JonaECG/netelpro-qwen2.5-1.5b-honest) | **53.3%** | **0.0%** (0/30 false claims) | **100% Elimination of False Claims** |
| **Liquid State-Space** | `lfm2.5:latest` (Base) | 20.0% | 10.0% (3/30 false claims) | Baseline |
| **Liquid State-Space** | [🤗 `JonaECG/netelpro-lfm2.5-1.2b-honest`](https://huggingface.co/JonaECG/netelpro-lfm2.5-1.2b-honest) | **46.7%** | **6.7%** (2/30 false claims) | **+133% Relative Honesty Gain** (+26.7% net) |

*Full comparative reports and raw test runs are versioned under [`benchmarks/`](benchmarks/).*

### Universal Python Guard (`netelpro.guard`)

```python
from netelpro.guard import HonestyGuard

guard = HonestyGuard()
# Raises HonestyViolationError if the turn claims verification without tool evidence
verified_text = guard.enforce(agent_response, tool_results=results)
```

## Documentation & Research

* **Whitepaper:** [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) — *Netelpro: Compiler-Enforced Epistemic Honesty for Autonomous LLM Agents*.
* **Language Specification:** [`docs/SPEC.md`](docs/SPEC.md).
* **MCP Interface:** [`docs/MCP.md`](docs/MCP.md).

## Pretrained Models & Hugging Face

* **Qwen 2.5 1.5B (Transformer):** [🤗 JonaECG/netelpro-qwen2.5-1.5b-honest](https://huggingface.co/JonaECG/netelpro-qwen2.5-1.5b-honest) — GGUF Q4_K_M weights + Modelfile for Ollama and LM Studio.
* **Liquid AI LFM 2.5 1.2B (Liquid State-Space):** [🤗 JonaECG/netelpro-lfm2.5-1.2b-honest](https://huggingface.co/JonaECG/netelpro-lfm2.5-1.2b-honest) — Ultra-efficient GGUF Q4_K_M weights + Modelfile.
* **RAFT-trained Qwen 2.5 1.5B (Transformer):** [🤗 JonaECG/netelpro-qwen2.5-1.5b-raft](https://huggingface.co/JonaECG/netelpro-qwen2.5-1.5b-raft) — GGUF Q4_K_M weights + Modelfile. First RL-trained model: see the RAFT section below for the honest numbers.
* **RAFT v2 Qwen 2.5 1.5B (Transformer):** [🤗 JonaECG/netelpro-qwen2.5-1.5b-raft-v2](https://huggingface.co/JonaECG/netelpro-qwen2.5-1.5b-raft-v2) — 5 rounds with accumulated pool: 20% → 60% pass@8 OOD (paired, seeded). 3× more samples pass than v1.


## Train Your Own Model (Google Colab Free GPU)

Align small edge models to eliminate Verification Theater using DPO on Google Colab's free T4 GPU (~15-20 mins):

* **Qwen 2.5 1.5B (Transformer):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jona2428/netelpro/blob/master/training/train_colab.ipynb)
* **Liquid AI LFM 2.5 1.2B (Liquid State-Space):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jona2428/netelpro/blob/master/training/train_lfm_colab.ipynb)

See [`training/README.md`](training/README.md) for full instructions and GGUF export.

### RLVR/RAFT: Training Against a Compiled Verifier

The RAFT trainer (`training/train_raft_colab.ipynb`) is a different regime from DPO:
the model samples Netelpro programs, the **compiled verifier** (`rlvr/verify.py`) grades
every sample against 20 randomized test cases, and only programs that compile *and*
pass every case become SFT training data. The reward is a compiler, not a preference
model. Run it on a free Colab T4 (~40 min for 3 rounds):

* **RAFT notebook (RLVR):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jona2428/netelpro/blob/master/training/train_raft_colab.ipynb)

**Run #1 (2026-09-07, Colab T4, 3 rounds):** OOD pass@8 (`power_int`,
`nth_element`, `string_to_int`, `gcd_pair`, `list_sum` — 5 held-out tasks never trained
on) went **0% → 20% → 40% → 40%**, with the baseline model solving 0/5 tasks in 40
attempts. An independent local re-measurement of the published GGUF
(`python -m rlvr.gguf_eval`) scored **80% pass@8 (4/5 tasks)** — quantization did not
destroy the learned behavior.

Honest caveats, as always: these are **pass@8, not pass@1**; n=5 tasks (granularity
20%); baseline vs. final sampling is not paired (unseeded sampler); the round-2
plateau is expected under per-round-only SFT datasets (canonical RAFT accumulates
the verified pool across rounds). The 80% GGUF figure sits well above the 40% fp16
figure from the same checkpoint — that gap is **not** a quantization effect, it's
two different measurement protocols (in-notebook comparison unseeded, local
`gguf_eval` re-measurement seeded); run #2 below fixes this by seeding both sides
of the same run.

**Run #2 (2026-09-08, Colab T4, 5 rounds, accumulated pool, seeded evals):** OOD
pass@8 went **20% → 60%** (paired baseline/final, same seed) — **3/5 tasks**, with
3× more passing samples than run #1 (13/40 vs 4/40). The GGUF q4_k_m re-measurement
matched the fp16 verdict exactly (60%, 3/5) — quantization preserved the learned
behavior. Refuted hypothesis: `gcd_pair` did **not** yield to the accumulated pool
(0/8 in both runs) — it needs a different lever (curriculum or more diverse
samples). Run #1 vs #2 final numbers (40% vs 60%) are directional only; the paired
comparison is each run against its own baseline.

**Corpus grown 2026-09-08 (no new training run yet):** the 5-task OOD split above
(runs #1/#2) gave 20% granularity per task — one extra pass shifted the whole
number. The corpus is now **55 tasks (35 train + 20 OOD)**, still balanced by
family (arithmetic/list/string) and still an explicit, versioned contract
(`rlvr.tasks.OOD_TASK_IDS`) rather than a computed split — the original 5 OOD
tasks are unchanged inside the new 20, so runs #1/#2 stay comparable to each
other even as future runs measure against the larger set. Run #3 (fresh from
base, same protocol as #1/#2) is the next step, not yet executed.

Local evaluation against your own exported GGUF (requires [Ollama](https://ollama.com)
with the model installed):

```bash
python -m rlvr.gguf_eval --model netelpro-qwen1.5b-raft
```


## Status
- **Spec:** v0.9 consolidated at [`docs/SPEC.md`](docs/SPEC.md); machine-consumed arity table at `spec/arity_table.json`.
- **Release:** [v0.9.0](https://github.com/jona2428/netelpro/releases/tag/v0.9.0) — RLVR/RAFT run #2 (accumulated pool, paired seeded evals, official `rlvr.gguf_eval`), on top of v0.7.0's HonestyGuard SDK, Verification Theater Benchmark, DPO Colab Trainer, LLVM native JIT, and MCP server. Full history in [`CHANGELOG.md`](CHANGELOG.md).
- **History:** zero differential divergence between Python interpreter and LLVM native backend across the test suite (see the CI badge above for current pass count — this file no longer hardcodes it, it goes stale every release).