# Zone Rule Builder — Architecture & Contract

**Netelpro auto-hosting plan, Session 3 — the gate generates its own law.**

Status: **IMPLEMENTED & VERIFIED**. `builders/zone_rule_builder.sl` (arity 36 = 3 zones × 6 slots × 2 forms) is compiled and verified: byte-parity vs the reference generator (except the honest provenance line), `netelpro__compile` clean, `netelpro__verify` 7/7 differential on production-zone vectors. Host frontier adapter: `builders/zone_builder_adapter.py` (capacity >6 roots/zone → reject host-side; fail-safe None on any error). Integration suite: `netelpro/tests/test_zone_builder_integration.py` (6 tests). Status machine-verified: 59/59 language suite green. This document states the obligations any conforming builder must satisfy, as
verified against the host bridge (`netelpro/rule_filter.py`, `RuleBuilder`, v0.5) and the Python
oracle (`zone_rule_generator.py`, Neuromancer host, `src/neuromancer/domains/agentic/application/`).

Related: [`docs/MCP.md`](MCP.md) (audit tools and limits), [`docs/SPEC.md`](SPEC.md) (language
semantics), [`../README.md`](../README.md) (honesty stack, phase history).

---

## 1. Motivation: the guardian writes its own law

Until Session 3, the zone law of the Neuromancer agent — which paths are RED (blocked),
GREEN (autonomous), YELLOW (confirmation required) — was authored by a Python module:
`zone_rule_generator.py` emitted the `.sl` text, and the host compiled it natively through the
Netelpro bridge. The law ran as native machine code, but its *author* was external: a Python
preprocessor stood between the guardian and its own constitution.

Auto-hosting removes that intermediary. The gate now generates its own rules through Netelpro
itself (Session 3 bitácora, commit `a5020dfa`): a Netelpro **builder program** —
`(defn build-rule ...)` — produces the zone-policy rule text as a native string product, and the
host's role shrinks to the frontier work Netelpro cannot do: filesystem canonicalization,
escaping, and audit orchestration.

Why this matters beyond convenience:

- **Self-hosting loop.** The system that enforces the law can re-author it in its own language.
  Changing zones no longer touches Python: roots are *data* crossing the builder's parameter
  list, not code. The compiled builder is zone-agnostic and never needs recompilation for a
  zone change.
- **The product is still prosecuted.** A generated rule is not trusted because the guardian
  wrote it. It deploys only after the same static prosecution and differential verification as
  hand-written rules (§4). Generation changes *who writes*; it does not relax *what must pass*.
- **One oracle, two authors.** During migration the Python generator remains the byte-level
  oracle; the builder must match it byte for byte (§4). Parity, not faith, retires the Python
  path — and the oracle stays available for rollback until the migration is declared complete.

The v0.5 language capability that makes this possible is narrow by design: `build-rule` is the
*only* top-level definition permitted to return `Str` (an `i8*` into a per-call-reset 64 KB
arena). The gate law stands unchanged: `filter-rule` **decides and never produces** — a gate
rule returning a string is a compile-time prosecution. Builders produce; filters decide.

## 2. Architecture & boundary

The migration is a *responsibility split*, not a rewrite. Everything that requires host
semantics (filesystem, case folding, escaping) stays on the Python frontier; everything that is
pure text composition moves into the native builder.

| Concern | Owner | Rationale |
|---|---|---|
| Path canonicalization: `~` expansion, resolve to absolute, `\` → `/`, lowercase | **Host** (Python frontier) | Native `==`/`prefix?` compare case-sensitively; Windows paths are not |
| Deduplication + deterministic sort of root lists | **Host** | Byte-stable products: identical inputs → identical bytes |
| Escaping for `.sl` string literals: `\`→`\\`, `"`→`\"`, newline→`\n`, tab→`\t` | **Host** (frontier) | The builder splices arguments verbatim; it never re-escapes |
| Zone-rank convention (1 = RED, 2 = GREEN, 3 = YELLOW) | **Host** (call convention) | Fixed contract shared by rule, callers, and test vectors |
| Composition of the `.sl` structure: header, `defn`, nested `if`, `or` chains, fallback | **Builder** (native) | The guardian writes its own law |
| String production (`str-cat` composition into the 64 KB arena) | **Builder** (native, v0.5) | Only `build-rule` may return `Str` |
| Byte-parity oracle vs the Python generator | **Host** (test) | Migration acceptance gate (§4) |
| Audit pipeline: compile + verify + deploy | **Host** (MCP orchestration) | A generated law is signed, never assumed |

Data flow, one generation cycle:

```
zone config (raw paths)
  → HOST: canonicalize → dedup → sort → escape          (frontier)
  → RuleBuilder.build(*args)  [native JIT call]          (builder composes .sl text)
  → byte-parity check vs zone_rule_generator.py          (oracle, §4)
  → netelpro__compile + netelpro__verify on the product  (signing, §4)
  → deploy: compile_filter() into the live gate
```

The component-wise containment law is preserved end to end: every root is embedded in the
product as **both** an exact match `(== path "<root>")` **and** a child prefix
`(prefix? path "<root>/")`, so sibling names (`src_backup`) never match root `src` — the trap
that motivated the original generator, and a mandatory verify vector (§4).

## 3. Contract of `build-rule`

> **Contract as fixed by the orchestrator; see `builders/zone_rule_builder.sl` for its final
> form.** The builder is being written in parallel with this document; this section pins the
> obligations, not incidental implementation details.

1. **Entry point.** Exactly one `(defn build-rule ...)`. `RuleBuilder.__init__` prosecutes the
   source at compile time: the definition must exist, and its resolved return type must be a
   pointer (`Str`) — a builder returning `Int`/`Bool` is rejected ("it exists to produce rule
   text").
2. **Arity and argument order.** `N` parameters, all resolving to `Str`, in a fixed
   deterministic order fixed by the orchestrator: zone ranks in ascending order
   (1 = RED, 2 = GREEN, 3 = YELLOW); within a zone, roots in the host-sorted order; per root,
   two arguments — the **exact-match literal** first, then the **trailing-slash prefix
   literal** (`root + "/"`). The sentinel convention and final arity used for empty zones are
   orchestrator-fixed; the builder file is the source of truth for the exact signature.
   `RuleBuilder.build` enforces arity at the bridge: a mismatch raises `RuleFilterError` with
   the `defn`'s exact `line:col`.
3. **Argument semantics (frontier law).** Each argument is *final literal text* to be spliced
   verbatim between quotes in the product. The builder performs no canonicalization and no
   escaping — both are host-frontier responsibilities (§2). Double-escaping at either side is a
   parity bug, not a style choice.
4. **Product.** The complete `.sl` source of the zone rule: the header comment block, the
   `(defn filter-rule (path zone) ...)` definition, nested zone `if`s over ranks 1/2/3, the
   per-root `(== path "...")` / `(prefix? path "...")` pairs joined by nested `or`, and the
   Str-binding fallback `(if (== path "") false false)` that guarantees `path` infers as `Str`
   even when every zone list is empty.
5. **Determinism.** Same arguments → byte-identical product. No clocks, no randomness, no
   iteration-order dependence (the host pre-sorts; the builder only composes).
6. **Purity.** The builder performs no IO; no capabilities are granted. Helper definitions may
   produce strings (legitimate composition under the v0.5 law), but the product is a pure
   function of its arguments.

## 4. Audit pipeline

Every generated `.sl` is **signed** by the MCP audit tools before deploy. A generation cycle
that skips any stage does not produce a law; it produces unverified text.

| # | Stage | Tool / mechanism | Failure meaning |
|---|---|---|---|
| 1 | Builder prosecution | `netelpro__compile` on the builder source (wire name `netelpro_compile` per `docs/MCP.md`): parse, capabilities, hole manifest, native codegen | No product is built at all |
| 2 | Native build | `RuleBuilder.build(*args)` — JIT call; product read NUL-terminated UTF-8 from the per-call arena | `NULL` product → `RuleFilterError` ("arena overflow") |
| 3 | Byte-parity | Product compared byte-for-byte against `generate_zone_rule_source(...)` for identical zone inputs | Migration gate fails; Python path stays authoritative |
| 4 | Product signing | `netelpro__compile` + `netelpro__verify` on the generated `.sl` with zone test vectors | Unsigned text never reaches the gate |
| 5 | Deploy | `compile_filter()` / `RuleFilter` compiles the signed source into the live gate | Compile failure → keep previous law |

Notes on the two verification layers, which must not be confused:

- **`netelpro__verify` verifies the *product*** — a `(defn filter-rule ...)` — by differential
  parity between the native JIT and the reference interpreter (max 100 vectors; each mismatch
  reports both engines' decisions). Mandatory vectors: rank fallback (unknown zone → `false`),
  empty zones, the `src_backup` component-containment trap, and escape round-trips (quotes,
  backslashes, whitespace in paths).
- **The builder itself** is differentially checked by `RuleBuilder.verify_build(cases)`: the
  native `build-rule` product is compared against the reference interpreter evaluating the
  appended call form `(build-rule ...)` — native builder vs interpreted builder, before the
  product is ever compared to the Python oracle.

**Byte-parity, precisely.** The builder's product and the Python generator's output must be
identical for identical zone inputs **except the provenance line** — the first header line,
which names the generating program (Python: `; Netelpro zone policy rule generated by
zone_rule_generator.py.`; the builder names itself). Every other byte — remaining header
comments, the `defn`, all literals, indentation, trailing newline — must match. Parity is the
acceptance gate of the whole migration: until it holds for the full zone corpus, the Python
generator remains the source of truth and the builder's product is treated as a candidate.

## 5. Operational notes

**Arena limits (v0.5).** The builder's string product lives in a static 64 KB arena with a bump
pointer, reset per call — no `free()`, no leaks, fully deterministic. The product must fit;
overflow yields `NULL`, which the bridge prosecutes as `RuleFilterError` ("arena overflow").
There is no truncation and no partial product: a zone configuration too large for the arena is
a hard failure at generation time. The host frontier should reject such configurations
fail-closed rather than shrink them silently.

**MCP containment limits** (`docs/MCP.md` §4) bound every pipeline stage: source ≤ 65 536 bytes
(`MAX_SOURCE_BYTES`), bracket nesting depth ≤ 64, `netelpro__verify` ≤ 100 cases, evaluation
timeout 3.0 s. Generated zone rules are far below these ceilings; the builder *source* must
respect the same limits, and deep `str-cat` nesting counts against the depth budget.

**`str-cat` is binary.** Composition is nested calls, not a variadic fold — a constraint the
builder author works *with* (nesting is the form), not around. This is a fixed language trap
from v0.5, not a builder choice.

**Regenerating when zones change.** Zones are data, not code. A zone change means: new
canonicalized + escaped argument vector → `build()` → full pipeline re-run (§4, stages 2–5).
The compiled builder is never recompiled for a zone change; only its inputs move. Because the
host dedups and sorts before the call, identical zone sets regenerate byte-identical laws —
regeneration is diff-stable by construction.

**Fail-closed rollback.** If any audit stage fails, the previously signed law stays deployed;
unverified text never reaches the gate. The Python generator remains in place as oracle and
fallback until byte-parity is proven and the migration is explicitly declared complete — the
same semantics-preserving fallback discipline established when the gate first went native.

**Where things live.** Builder: `builders/zone_rule_builder.sl` (this repo, in progress —
contract above, final form in the file). Bridge: `netelpro/rule_filter.py` (`RuleBuilder`,
`build`, `verify_build`). Python oracle: `zone_rule_generator.py` (Neuromancer host,
`src/neuromancer/domains/agentic/application/`). Audit tools: `netelpro__compile`,
`netelpro__verify` (wire names `netelpro_compile` / `netelpro_verify`, `docs/MCP.md` §3).