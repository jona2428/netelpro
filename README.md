# Netelpro

**A programming language for LLMs — honest, semantic, universal.**
*(formerly Straylight; final name decision pending — see [Status](#status))*

Netelpro is a programming language written by LLMs and audited by a compiler that behaves as a prosecutor. Its grammar is engineered so that an LLM can verify its own syntax mechanically: every form is `(head arg1 arg2 ...)` with a declared arity, so checking a form means counting the operands between the head and the closing parenthesis. Counting is a mechanical operation an LLM performs reliably; simulating a recursive-descent parser is not.

The thesis: **an LLM can verify its own syntax by counting, not simulating** — and nothing unverifiable passes silently. Unknown heads, arity violations, silent holes, and ungranted IO are all compile-time errors reported with exact `line:col` coordinates. A Netelpro program runs only after surviving every layer of prosecution.

## The Honesty Stack

Four verified layers. Each failure class dies at the earliest layer, with exact coordinates. There is no stage where dishonesty passes silently.

| # | Layer | Stage | Kills |
|---|-------|-------|-------|
| 1 | **Fiscal parser** | parse time | Unknown heads, duplicate top-level definitions, arity violations — every form audited against `spec/arity_table.json`, the machine-consumed single source of truth |
| 2 | **Capabilities as types** | static pass | Any capability use without a top-level `(grant ...)` — IO requires `(grant io)`, enforced statically, even when buried in unexercised branches |
| 3 | **Sorry manifest** | static pass | Silent holes are impossible: the only legal unimplemented branch is `(sorry "reason")`, and every declared hole is listed with `line:col` + reason on stderr at every compilation |
| 4 | **LLVM native backend** | codegen | Non-representable types at use (`Float`/`Str`/`List`/fn-as-value) and Bool-demand violations at the machine boundary; `llvmlite 0.49`, `i64`/`i1`, structural TCO |

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

Requires Python 3. The interpreter needs nothing beyond CPython; the native backend needs `llvmlite 0.49`.

```bash
python -m netelpro file.sl           # interpreter — reference semantics
python -m netelpro --native file.sl  # compiled native — LLVM JIT, same static passes
```

`examples/fib.sl`:

```netelpro
; Straylight v0.1 -- Fibonacci demonstration

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

The native backend is a strict subset compiler: it accepts only programs whose values are representable in machine words and rejects everything else with a prosecutorial compile error — never a silent fallback, never a silent divergence. The test suite runs every program through both engines and compares: **304 tests passing, zero mismatches**. The same principle is exposed programmatically by the Phase 6 bridge: `RuleFilter.verify(cases)` returns any `(args, expected, interpreted, native)` mismatches; an empty list means full agreement.

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

The Phase 6 use case, `examples/gate_rule.sl` — a Neuromancer gate rule as a compiled pure function:

```netelpro
(defn filter-rule (priority confidence approved)
  (or (and (>= priority 3) (< confidence 90))
      (== approved 1)))
```

## Status

- **Spec:** v0.9 consolidated at [`docs/SPEC.md`](docs/SPEC.md); machine-consumed arity table at `spec/arity_table.json`.
- **Privacy:** private repository until the first alpha — public release is a decision of the repo owner.
- **History:** 8 commits of verified history; every claim in the spec is backed by the test suite (`tests/`, 304 tests) and the examples (`examples/`).
- **Deliberate v0.1 limits** (documented, not accidental): the compiled subset is `Int`/`Bool`; recursion must be tail-recursive to compile; no first-class functions; capabilities are file-scoped (`{io}`).