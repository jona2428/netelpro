# Implementation Spec — Phase 1 Combined: `truth-table` + Finite Range Types

**Date:** 2026-09-08
**Scope:** Phase 1 ONLY (`truth-table` + `Int{...}` literal-enumeration types).
Explicitly excluded: comparison-predicate refinements (`!= 0`, `> 0`) — Phase 2;
effect rows — Phase 3 (= Session 4 per-function effects); `prove`/`evidence` — Phase 4.
Source design doc: [2026-09-08-lenguaje-verificacion-nativa-design.md](2026-09-08-lenguaje-verificacion-nativa-design.md) (§1 and §4-range-literals only).
Style: contract spec, `gate_contract.md` convention — every claim below is checkable against the named modules once implemented; strictness is labeled, not excused.

---

## 0. Deliverable definition

Phase 1 adds to the language, in one coherent change set:

1. **Typed parameters** — first type-annotation syntax in Netelpro: `(name : Bool)` and `(name : (Int lit+))` (finite literal enumeration, the written form of the design doc's `Int{0,1,2}`).
2. **The `truth-table` special form** — a table-defined pure function with compile-time exhaustiveness prosecution and a mandatory explicit default row.
3. **Generated parity artifacts** — the compiler emits (a) the Python fallback decision function and (b) the differential test matrix, both derived from the single `.sl` source. Nothing is hand-written twice anymore.

The bug this phase exists to kill: the 2026-09-07 aliasing bug in the hand-written Python fallback of the gate (`apply()` aliasing `audit_paths` with the input decision), detectable only because `test_netelpro_gate_fallback_parity.py` compared both paths exhaustively. After Phase 1, there is exactly one source of truth (the `.sl` file) and the fallback is emitted, not authored.

---

## 1. Surface syntax

### 1.1 Typed parameter annotations (defn and truth-table share one mechanism)

```netelpro
(defn f (critical : Bool) (zone-rank : (Int 0 1 2)) ...)
```

- `Bool` — the type, no domain annotation needed (domain is always `{true, false}`).
- `(Int 0 1 2)` — finite Int enumeration. Head must be the symbol `Int`; tail must be ≥ 1 integer literals. Written `Int{0,1,2}` in prose.
- Annotations are **declarations, not refinements**: no predicates, no comparisons, no inference obligations at call sites for non-literal arguments (Phase 2 work). A non-literal argument to an annotated param compiles fine in v1; only literal arguments are type-checked against the annotation (bool/int strictness as in v0.6: `1` where `Bool` is declared ⇒ compile error with exact coordinates).
- A truth-table param list uses the same annotation syntax and it is **mandatory there** (exhaustiveness is undecidable without declared finite domains).

### 1.2 The `truth-table` form

```netelpro
(truth-table filter-rule
  (critical : Bool)
  (zone-rank : (Int 0 1 2))
  (auto-confirm : Bool)
  (((true false true) -> true)
   ((_ _ false)      -> false)
   ((_ true _)       -> false)      ; example rows — shape only
   (_ -> false)))                   ; mandatory explicit default (see D1)
```

Grammar (s-expression level):

```
(truth-table NAME PARAM+ ROW+)
PARAM  := (name : TYPE)          TYPE := Bool | (Int LIT+)
ROW    := ((SLOT+) -> EXPR)      ; arity of SLOT+ == number of PARAMs
SLOT   := true | false | INT-LIT | _
EXPR   := any Netelpro expression; every row's EXPR must type-unify
```

- The last row MUST be an all-wildcard row `(_ ... _ -> EXPR)`. A table without it is a **compile error** (`missing default row`) — the design doc's "nada implícito" rule, unchanged.
- **First-match-wins evaluation, top to bottom.** Rows may overlap; the first row whose slots match the input decides. Overlap is legal and un-warned (documented; see D3).
- Slots are literal-only: no variables, no expressions, no negation. A `Bool` slot accepts only `true`/`false`/`_`; an `Int` slot accepts only literals from that param's declared enumeration plus `_`. Type-strict: `1` in a `Bool` slot is a compile error (this pins the v0.6 bool/int conflation lesson into the new form).
- Result type must be uniform across all rows (mixed `Int`/`Bool` rows ⇒ compile error). The defined function is callable like any `defn` result, with call-site literal checking against the declared param types.

### 1.3 New lexer tokens

Two new tokens, nothing else: `:` (annotation separator) and `->` (row arrow). Verify during implementation whether the current symbol scanner of `netelpro/lexer.py` already tokenizes them; expected outcome is two explicit additions in the lexer and a matching entry in the special-forms registry (alongside the existing special forms surfaced by `netelpro__netelpro_spec`; `spec/arity_table.json` records primitives — `truth-table` is a special form, not a primitive, and must NOT be added to the arity table).

---

## 2. The exhaustiveness prosecutor (compile-time)

### 2.1 Coverage algorithm

The design doc states two requirements that literally conflict if read naively — "exhaustive coverage of the declared product" AND "a default row is mandatory" (a mandatory catch-all makes every table trivially covered, so the check would be vacuous). Resolution, pinned as **Decision D1**:

> Coverage is computed over the rows **excluding** the mandatory default row. Every combination of the declared cartesian product must match at least one non-default row. The default row is required **and** is expected to be unreachable in a well-formed table — it exists as the explicit statement "nothing falls through implicitly", in the same spirit as `(sorry "razón")`.

Algorithm:
1. Compute the declared product `D = d(p1) × ... × d(pn)` where `d(Bool) = {true, false}` and `d((Int l1...lk)) = {l1..lk}`.
2. Reject at compile time if `|D| > 256` (protects the generated differential matrix from combinatorial explosion). Error names the offending parameter ranges.
3. For each combination, test membership against each non-default row's slots (a slot matches if it equals the value or is `_`). If no row matches → **compile error** listing the uncovered combinations (print up to 8, then `… +N more`).
4. Wildcards in non-default rows are fine — coverage means *matched by some row*, not *enumerated row-by-row*.

### 2.2 Unreachable default: no warning

If non-default rows already cover all of `D`, the default row is dead by construction. The compiler emits **no** dead-code warning for it — it is contractual strictness, not an accident (§2.1). Documented limit: nothing distinguishes "default as safety net" from "default as oversight"; the differential matrix (§4) makes the oversight visible anyway, because every combination's expected value comes from the rows.

### 2.3 Out-of-domain inputs are not UB

Annotated params constrain literals at compile time only. A host (FFI/ctypes bridge) passing an `Int` outside the declared enumeration to a truth-table-defined function has **defined behavior**: no slot can match an out-of-enum value, so control reaches the default row deterministically. No runtime range tags, no runtime validation. This is a boundary contract, stated in §5.

---

## 3. Codegen — desugaring, both backends

### 3.1 The table IS sugar; the if-chain IS the code

Parsing a `truth-table` produces:

1. A `TruthTableSpec` metadata object attached to the function: declared param types, ordered rows, slot literals. Retained in the AST — it is the input for the prosecutor (§2) and for artifact generation (§4).
2. A desugared body: nested `if` chain, one level per row (top to bottom), each level a conjunction of per-slot comparisons against literals (equality per slot; wildcard slots contribute nothing). Both the reference interpreter (`netelpro/evaluator.py`) and the LLVM backend (`netelpro/codegen.py` → `netelpro/rule_filter.py` path) consume **the same desugared body**.

Consequences:
- **Interpreter/native parity is by construction** — same desugared AST, no divergent implementations to reconcile. This is the structural kill of the dual-source-of-truth class.
- **Zero new LLVM constructs**: the branch chain is existing `icmp`/`br` machinery; enum-typed params remain plain `i64`, `Bool` remains `i1` at the boundary. TC0 register machine unaffected.
- Out-of-domain input routing (§2.3) falls out of the chain: every comparison fails, final branch lands on the default row's expression.

### 3.2 Call-site checking

The defined function participates in the existing call-checking pipeline (arity prosecutor, type-strict equality from v0.3/v0.6). Literal arguments are checked against declared annotations (§1.1). Nothing else changes at call sites.

---

## 4. Generated artifacts — the fallback and the matrix

### 4.1 Generated Python fallback

The builder (existing `builders/` machinery — "builder writes the law", Session 3) gains one emission step: for every top-level `truth-table`, emit into the host fallback module a pure-Python decision function generated **from the parsed rows**:

- A module-level tuple of `(slots_tuple, result_expr_lambda_or_value)` rows in declared order, a linear first-match scan, ending in the default row.
- Pure function: no imports beyond stdlib, no state, no consultation of the compiled backend (a fallback that requires the backend is not a fallback).
- File header, machine-emitted: `# GENERATED FROM <sl-path> AT <commit-sha> — DO NOT EDIT BY HAND`.
- The generated function replaces the hand-written `apply()` in the host integration (Neuromancer-side `netelpro_gate.py` fallback path). The aliasing bug class dies here: the fallback no longer has an author who can alias anything; it is a table scan.

### 4.2 Generated differential matrix (pytest)

The builder also emits a pytest module enumerating, for each truth-table:

1. **Every combination in `D`** — asserted equal across (a) native LLVM decision and (b) reference interpreter, with the expected value taken from row semantics (first match).
2. **Out-of-domain probes** — for each `Int`-enum param, values adjacent-outside the declared set (e.g. `min-1`, `max+1`) asserted to route to the default row's result on both backends.

This is the design doc's "differential gratis", made precise: the compiler generates the test FROM the declared rows; the pre-existing hand-written parity suite (`test_netelpro_gate_fallback_parity.py`) stays in place until the cutover (§6) retires the hand-written fallback, after which it pins only the *generator's* output contract (fallback file structure + header), not hand-copied rows.

---

## 5. Boundary contracts (exact)

| # | Contract | Owner | Enforcement |
|---|---|---|---|
| B1 | `.sl` file is the single source of truth for a table's semantics | `truth-table` author | Parser; no parallel hand-written decision code may exist for a table-defined function (cutover deletes the last one, §6) |
| B2 | Enum-typed params cross FFI as plain `i64`; **no runtime range check** | Host bridge | Compile-time-only discipline; out-of-domain → default row (§2.3). Labeled strictness gap, accepted for v1 |
| B3 | Slots are literals, type-strict | Compiler | Compile error with `line:col`, existing prosecutorial format |
| B4 | Default row mandatory, all-wildcard, last | Compiler | Compile error otherwise |
| B5 | Coverage over non-default rows, product ≤ 256 | Compiler | Compile error with uncovered combinations listed |
| B6 | Generated artifacts carry provenance header and are never hand-edited | Builder + reviewer discipline | Header check in generated-artifact contract test |
| B7 | Generated fallback is pure, stdlib-only, backend-independent | Builder | Reviewed emission; contract test asserts no `netelpro` imports in emitted module |

---

## 6. Cutover (the payoff step)

1. Rewrite `data/netelpro_gate.sl` (gate of the house, auto-confirmation rule — lives at the Neuromancer repo root `C:\...\FinalFront\data\`, OUTSIDE the straylight tree) as a `truth-table` over its three flags — the rule is exactly the shape the form exists for.
2. Builder regenerates the gate fallback (§4.1) and the matrix (§4.2).
3. Delete the hand-written fallback decision code from the host integration; keep the parity test as generator-contract pin (§4.2).
4. Full preflight (`ruff` + `mypy` + suite), gate suite 27/27 must stay green and retrocompatible, push immediately after green (house rule: no local commit accumulation).

Note: the ledger item "gate .sl regenerated by builder if demand appears" is satisfied transitively by this cutover — the table form makes regeneration the default path, not a conditional one.

---

## 7. Implementation steps (ordered, sized)

| # | Step | Size | Touchpoints |
|---|---|---|---|
| 1 | Lexer: `:` and `->` tokens | S | `netelpro/lexer.py` |
| 2 | Parser: typed params in `defn` + `truth-table` special form, AST node + `TruthTableSpec` retention | M | `netelpro/parser.py`, `netelpro/ast_nodes.py` |
| 3 | Prosecutor: coverage algorithm, product cap, error messages with coordinates | M | `netelpro/parser.py` (post-parse validation pass) |
| 4 | Desugaring to `if`-chain; evaluator consumes desugared AST | M | `netelpro/evaluator.py` (desugar likely lives in parser output) |
| 5 | LLVM codegen for the desugared body (reuse `icmp`/`br`) | S–M | `netelpro/codegen.py` |
| 6 | Builder emission: fallback + differential matrix + provenance headers | M | `builders/` |
| 7 | Contract tests: §8 list | M | `tests/` |
| 8 | Cutover §6 + preflight + push | M | `../data/netelpro_gate.sl` (Neuromancer root), host fallback |

---

## 8. Contract tests to pin (minimum set)

1. `test_truth_table_missing_default_row_is_compile_error` — table without all-wildcard last row rejected with coordinates.
2. `test_truth_table_uncovered_combination_is_compile_error` — remove one explicit row from a Bool×Bool table; prosecutor lists the missing combination.
3. `test_truth_table_slot_type_strictness` — `1` in a `Bool` slot; out-of-enum literal in an `Int` slot; both compile errors.
4. `test_truth_table_mixed_row_result_types_rejected` — one `Bool` row among `Int` rows.
5. `test_truth_table_first_match_wins` — overlapping rows; both backends pick the earlier row.
6. `test_truth_table_out_of_domain_routes_to_default` — native and interpreter, value outside declared enum.
7. `test_truth_table_product_cap_256` — declared product 512 rejected; 256 accepted.
8. `test_generated_fallback_parity` — generated matrix runs green for every declared combination and out-of-domain probe (the emitted test itself, executed in CI).
9. `test_generated_fallback_is_pure` — emitted module imports nothing from `netelpro`, carries the provenance header.
10. `test_typed_param_literal_checking` — literal `1` at a `: Bool` call site rejected (extends v0.6 bool/int discipline to annotations).
11. Regression pin of the original 2026-09-07 bug class: the aliasing scenario reconstructed against the generated fallback must be structurally impossible (the emitted code contains no variable aliasing — assert by AST inspection of the emitted module).

---

## 9. What this spec does NOT cover, on purpose

- Comparison-predicate refinements (`b != 0` before call-site) — Phase 2; the soundness argument (immutability ⇒ dominating guard is sound without dataflow) is recorded in the design review of 2026-09-08 and lands with Phase 2.
- Effect rows — Phase 3; identical infrastructure to Session 4 per-function effects (`spec/caps.py` + codegen/evaluator bridge), designed as ONE thing when that phase starts.
- `prove`/`evidence` — Phase 4; opaque non-forgeable `evidence` type via FFI-only constructor (border-law pattern of `Str` in `shutdown_rule.sl`), "this turn" and "real tool" semantics stay host-side.
- Runtime range tags, `Str`-typed table params, dynamic row conditions.

## 10. Open decisions for Jona (blocking none of step 1–3)

- **D1 (pinned here, ratify or veto):** default row mandatory AND coverage checked over non-default rows only — the only coherent reading of the design doc's two requirements; consequence is a structurally-unreachable default row in well-formed tables (accepted strictness, no warning).
- **D2 (pinned here):** enumeration syntax is the s-expression form `(Int 0 1 2)` — avoids adding `{`/`}` lexemes for one feature; `Int{0,1,2}` remains the prose spelling.
- **D3 (pinned here):** overlapping rows are legal, first-match-wins — matches the wildcard-heavy style of the design doc's own example.
- **D4 (pinned here):** `truth-table` registers as a special form (NOT in `spec/arity_table.json`, which is for primitives) and appears in the special-forms surface of `netelpro__netelpro_spec`.