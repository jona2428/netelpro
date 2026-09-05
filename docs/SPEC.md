# Netelpro — Language Specification

Status: **v0.9 consolidated** (2026-09-05). Phases 0–6 implemented and verified: 322/322 tests green on this machine (interpreted + native backends, differential-tested; v0.2 adds Bool params at the native boundary). Single source of truth for arities: `spec/arity_table.json` (the spec and the table are kept in sync; the table is machine-consumed). Name: **Netelpro** (NEuron Teo Language PROgramming; formerly Straylight) — decided 2026-09-05.

## 1. Design thesis (why this grammar exists)

Netelpro is a programming language written **by** LLMs and **audited by** a compiler-as-prosecutor. The grammar is engineered so that an LLM can verify its own syntax **mechanically, without simulating a parser**:

1. **Prefixed, fixed arity.** Every form is `(head arg1 arg2 ...)`. Every head has a declared arity. To check a form, the writer counts operands between the head and the closing parenthesis. Counting is a mechanical operation an LLM performs reliably; simulating a recursive-descent parser is not.
2. **One exception, declared, still mechanical.** `list` is the only open-arity form. Its arity is closed by `)` itself, so counting still terminates the form. No other variadic head exists in v0.1.
3. **Zero human ceremony.** No imports, no class boilerplate, no type annotations at use sites. Every token carries meaning.
4. **Reserved grammar for the prosecutor.** `sorry` and `grant` are part of the grammar from v0.1; their static enforcement is established in Phase 3 (capabilities) and Phase 4 (holes). The syntax cannot be invented later; the honesty model must fit from day one.

## 2. Lexical grammar

| Token    | Pattern                                             | Notes |
|----------|-----------------------------------------------------|-------|
| `(` `)`  | literal                                             | the only delimiters |
| symbol   | `[A-Za-z_][A-Za-z0-9_?->]*` or operator heads       | `-` and `>` may appear inside symbols (`int->str`, `is-nil`) |
| int      | `-?[0-9]+`                                          | `-` only leads a number when followed by a digit at token start |
| float    | `-?[0-9]+\.[0-9]+`                                  | must have digits on both sides of `.` |
| string   | `"` ... `"` with escapes `\n \t \" \\`              | no multi-line strings in v0.1 |
| bool     | `true` `false`                                      | |
| nil      | `nil`                                               | the empty list |
| comment  | `;` to end of line                                  | ignored by the reader |

Whitespace separates tokens; `(` and `)` self-delimit.

## 3. Core forms

All control flow and binding are special forms with fixed arity:

```netelpro
(def x 10)                      ; constant binding
(defn add (x y) (+ x y))        ; named function: (defn name (params...) body) — body is exactly one expression
(fn (x) (+ x 1))                ; anonymous function
(let y 20 (+ x y))              ; single binding, chain lets for multiple bindings
(if (< x 10) 0 1)               ; if has NO else-less form; both branches mandatory
(and (> x 0) (< x 100))         ; short-circuit and/or, arity 2
```

Notes:
- `let` binds one name; nested `let`s compose. No destructuring in v0.1.
- Function bodies are exactly one expression. Sequencing is achieved by nesting `let`/`if`; no `do` in v0.1 (will be added with effects in Phase 3+).
- No `elif`, no variadic `+`. Chaining arithmetic is explicit nesting: `(+ (+ 1 2) 3)`.

## 4. Data model (v0.1)

Types: `Int`, `Float`, `Str`, `Bool`, `List<T>`. That is all.

- `(/ a b)` always returns `Float` (promotes Ints). Integer division is `quot`/`rem`.
- Lists are singly linked (`cons`/`head`/`tail`/`nil`). `(list 1 2 3)` desugars conceptually to `(cons 1 (cons 2 (cons 3 nil)))`.
- No maps, no records, no unions yet — added when the first real use case (Neuromancer rule filter, Phase 6) demands them, not before.

## 5. Prosecutor reservations

```netelpro
(sorry "reason as a string literal")     ; yields a value of whatever type context demands;
                                         ; Phase 4: silent holes = compile error, reason mandatory
(grant io)                               ; top-level only; declares file-wide capabilities;
                                         ; Phase 3: un-granted capability use = compile error, not runtime
```

## 6. Implementation decision: hand-written lexer vs generator — DECIDED

**Decision: hand-written lexer + recursive-descent parser driven by the arity table. No parser generator.**

Reasons:
1. The grammar is ~30 heads with declared arities. A hand-written lexer is ~100 lines; a generator adds a dependency and a layer of indirection that contradicts the thesis (the tooling should be as legible as the promise "you can count it yourself").
2. Prosecutor error messages need exact positions and mechanical counting ("`+` expects 2 operands, found 3"). Owning the lexer/parser gives first-class control of diagnostics — the prosecutor's voice is a product feature.
3. No external dependency between lexer and llvmlite backend (Phase 5): the whole compiler stays auditable by reading it.
4. The arity table JSON is the executable specification; parser and fiscal both consume it. One source of truth, not a grammar file duplicated in generator DSL.

## 7. Mechanical verification of the thesis (Phase 0 evidence)

`tools/check_arity.py` implements the embryo of the prosecutor: it reads a program, counts operands per form against the arity table, checks parens balance with exact positions, and checks calls to user `defn`s against their declared parameter counts — without any semantic analysis. Its test suite (`tests/test_check_arity.py`) proves: valid examples pass, every arity violation produces a counting-based error, unknown heads are rejected, and user-defined arities are enforced. This demonstrates that "count parentheses against a table" is sufficient for full structural validity — the core claim of the design.

## 8. Out of scope for v0.1

Macros, modules/imports, records/maps, pattern matching, generics, effects, lazy evaluation, `do`. Each enters only when a phase requires it, and each must be expressible with fixed arity or an explicitly declared open-arity form.

## 9. Phase 1 implementation notes (prosecutor tightened)

Phase 1 delivered the full frontend: `netelpro/lexer.py` (tokenizer), `netelpro/ast_nodes.py` (frozen typed AST), `netelpro/parser.py` (recursive-descent parser + mechanical arity validation). 94 tests green.

The parser is deliberately STRICTER than `tools/check_arity.py`. Documented divergences (parser enforces, Phase 0 fiscal does not):

1. **Reserved heads cannot be bound by `let`** — `(let if 5 ...)` and `(let + 10 ...)` are compile errors. Rationale: with no first-class functions in v0.1, rebinding special forms or primitives in local scope creates ambiguity for every downstream pass.
2. **`def`/`defn` are top-level only** — nested `(def a (def b 2))` or `(+ 1 (defn f (x) x))` are compile errors. Rationale: nested defs would silently miss the defn registry and produce misleading "unknown head" errors at call sites.
3. **Duplicate top-level `defn` is a compile error** — no silent registry overwrite. The last declaration still wins the arity registry (mechanical determinism for call checks), but the collision is reported.
4. **Symbols cannot start with `-`** — the Phase 1 lexer restored the spec regex `[A-Za-z_][A-Za-z0-9_?->]*`; `-foo`, `->`, `--` are invalid tokens (Phase 0 fiscal already rejected them; a worker regression briefly allowed them and the reviewer caught it).
5. **`sorry` requires a STRING literal; `grant` is top-level only with symbol operands** — both reserved since Phase 0.

Deferred to Phase 1.5 (reviewer findings, non-blocking): composite AST nodes use mutable `list` fields — unhashable, fine for current passes; will become tuples if/when AST sets are needed. `let` re-binding of a *user* defn name in an inner scope is currently accepted (lexical shadowing) — revisit when the evaluator lands (Phase 2).

Design call of record: literal nodes are separate subclasses (IntLit/FloatLit/StrLit/BoolLit/NilLit) instead of a kind-tagged Lit — `bool` is a subclass of `int` in Python and a single Lit node would need runtime value inspection; separate classes keep every pass type-safe by construction.

## 10. Phase 2 implementation notes (evaluator semantics — design decisions of record)

Phase 2 delivered the tree-walking evaluator (`netelpro/evaluator.py`), CLI (`netelpro/__main__.py`), and a 163-test suite (94 frontend + 69 evaluator). All green.

Value model of record:
- Int -> host int, Float -> host float, Str -> host str, Bool -> host bool, List -> `StrayList` (frozen, tuple-backed). `nil` is the empty StrayList. NO raw Python list/tuple ever acts as a list value.
- Python bool-is-int trap: every numeric path discriminates with `type(x) is bool` BEFORE `isinstance(x, (int, float))`. `(== true 1)` is False; `(+ true 1)` is a type error; `(nth xs true)` is a type error. Verified adversarially for nested list elements.
- `(/ a b)` always returns Float and rejects zero divisors (int 0 and float 0.0). `quot`/`rem` are Int-only and truncate toward zero.
- Strict booleans: `if`/`and`/`or` conditions MUST be Bool — no truthiness anywhere. `and`/`or` short-circuit and their right operands must also produce Bool (enforced positionally via the TCO bool stack).
- TCO: user calls in tail position (fn/defn body, if branches, let body, and/or right operand) iterate in `eval_loop` instead of recursing. `sum-to` at 100000 depth verified.
- `sorry` raises `StrayHoleError(reason, line, col)` — reaching an unimplemented hole stops the program loudly. `grant` registers capabilities on the evaluator state (enforcement lands in Phase 3).
- Fail-fast evaluation: the first runtime error stops the program (deliberate contrast with the parser, which collects all structural errors).

Prosecutor rules added in Phase 2 (verified adversarially):
1. Raw Python `RecursionError` NEVER reaches the user: deep NON-tail recursion is translated into `StrayRuntimeError("non-tail recursion exceeded host stack depth")`. The host stack limit is a runtime resource, not a Python traceback.
2. `str->int` is strict: `re.fullmatch(r"-?[0-9]+", s)` — no whitespace stripping, no `+` sign. `"+5"` and `" 5"` are runtime errors, not silent parses.

Known limit of record (accepted for v0.1): static-position attribution for host-stack exhaustion reports the program entry position (1,1), not the exact call site — an exact-site fix requires a depth-tracking parameter in `eval_loop` and lands with the capability/typing pass in Phase 3.

No-first-class-calls note: v0.1 has no first-class function calls; closures exist (capturing env) and are returned/bound via `def`, but the head of a call must be a table head or a registered `defn`. Lifting this requires an AST-level call-node rewrite (Phase 3+ design decision), not an evaluator patch.

## 11. Phase 3 implementation notes (capabilities as types)

Phase 3 delivered static capability enforcement (`netelpro/caps.py`), compiler CLI integration (`netelpro/__main__.py`), and runtime defense-in-depth (`netelpro/evaluator.py`).

Design decisions of record:
1. **Capability set v0.1 = `{"io"}`**: The only capability-requiring primitive in v0.1 is `print` (requires `io`). Known capabilities and per-primitive requirements are declared in `spec/arity_table.json` (`known_capabilities` object + `capabilities` arrays on primitives) and derived at runtime by `netelpro/caps.py` (`_derive_capabilities_from_table`). Declaring an unknown capability in a `(grant ...)` form is a compile error.
2. **Enforcement is a SEPARATE static compiler pass (`netelpro/caps.py`)**: Full recursive AST walk (`check_capabilities`), aggregating parser-style `CapError` diagnostics with exact line/col coordinates rather than failing fast. Integrated in `netelpro/__main__.py` within the compilation pipeline (`parse -> caps check -> holes check -> evaluate`): strictly AFTER `parse` and BEFORE `holes check`: un-granted IO is a COMPILE error — the program never starts.
3. **File-wide grants**: The granted set is the union of all top-level `(grant ...)` forms across the translation unit (`collect_grants`); no per-function effect tracking in v0.1 (upgrade path: per-function effect typing when first needed).
4. **No-first-class-calls makes the analysis fully static**: `Call.head` is a plain string, so every capability-requiring site is known at compile time without evaluating anything.
5. **Defense-in-depth**: The evaluator also carries the capability set (threaded through `eval_loop` and `_exec_primitive`), and `print` raises `StrayRuntimeError` if `io` is not granted — protects direct-API users who skip the static pass. The static pass remains THE enforcement.
6. **`sorry` semantics & static verification**: Reaching a `sorry` form continues to evaluate as a runtime hole (`StrayHoleError`); static hole verification is now enforced in Phase 4 (`netelpro/holes.py`, Section 12), ensuring all callable heads resolve statically while explicit `(sorry "reason")` forms compile clean and are recorded in the compile holes manifest.

## 12. Static Hole Prosecution (Phase 4)

Phase 4 delivers static hole prosecution (`netelpro/holes.py`), eliminating silent unimplemented stubs and unresolved call targets before evaluation begins. The compiler acts as a prosecutor: code cannot silently reference undefined identifiers or leave missing logic without explicit, machine-auditable declaration.

### 12.1 The Law of Resolution (No Silent Holes)

Every callable head in an invocation `(head arg1 arg2 ...)` must resolve to one of:
1. A built-in primitive or special form declared in `spec/arity_table.json`.
2. A top-level function definition (`defn`) — the ONLY user-defined legal head.

Anything else is a **silent hole**, rejected at PARSE time by the parser fiscal with exact
coordinates (not by `holes.py` — by parse itself):

```
line 4, col 16: unknown head 'missing-fn' (not in the arity table and not a declared defn)
```

Verified 2026-09-05 (parser probes): heads that are `def`-bound values, `fn`/`defn` parameters,
or `let` bindings are ALSO rejected ("unknown head 'f'"). No first-class calls in v0.1 (§9.1) —
the Phase 2 closure suite exercises closures only via `defn` bodies capturing the global
environment, never as call heads.

### 12.2 Explicit Holes: `(sorry "reason")`

The special form `(sorry "reason")` is the **only legal way** to leave an expression or branch unimplemented in Netelpro:
- **Compile-time validity**: A `sorry` form compiles clean and satisfies all static checks.
- **Holes manifest**: All explicit `sorry` holes are collected during static analysis into a manifest recording `(line, col, reason)`. When holes are present, the CLI emits this manifest to `stderr`, making incomplete implementations transparent and mechanically auditable by external tooling and LLM supervisors.
- **Runtime semantics**: If execution reaches a `sorry` form at runtime, evaluation immediately halts and raises `StrayHoleError(reason, line, col)`.

### 12.3 Scoping, Declarations, and Forward References

1. **Two-pass declaration collection**: Top-level declaration collection strictly precedes expression inspection. Consequently, forward references and mutual recursion among top-level `defn` forms are fully legal and resolve cleanly regardless of declaration order in the source file.
2. **Duplicate top-level declarations**: Duplicate top-level declarations (`def` or `defn` collisions) are rejected as compile-time errors. No top-level definition may silently overwrite or shadow another.
3. **Lexical scoping**: Local `let` bindings and `fn` parameter lists shadow outer bindings within their lexical scope, adhering to standard lexical scoping rules.

### 12.4 Limitation of Record (v0.1)

Documented honestly, verified 2026-09-05: the limitation drafted earlier
("(def x 5) then (x 1) passes the static check, fails at runtime") is FALSE —
the parser fiscal rejects it at PARSE time ("unknown head 'x'"). Phase 1 was already
harder than planned: the silent-hole law was de facto enforced from the start,
with `holes.py` contributing the sorry manifest on top. What remains for future
phases: first-class function heads (fn params / let bindings as callable heads —
currently impossible by design, §9.1) and full type-level callability checking.

### 12.5 Compiler Pipeline Order

Static passes are arranged sequentially in `netelpro/__main__.py` to guarantee deterministic, fail-fast verification before evaluation:

$$\text{parse} \longrightarrow \text{caps check} \longrightarrow \text{holes check} \longrightarrow \text{evaluate}$$

1. **`parse` (`netelpro/parser.py`)**: Tokenization, AST construction, syntax verification, and mechanical arity validation against `spec/arity_table.json`. Fails on syntax or arity errors.
2. **`caps check` (`netelpro/caps.py`)**: Static capability verification. Ensures effectful primitives (e.g. `print` requiring `io`) have matching top-level `(grant ...)` declarations. Un-granted capabilities trigger compile-time `CapError`.
3. **`holes check` (`netelpro/holes.py`)**: Static hole prosecution. Verifies that all callable heads resolve to known primitives, top-level definitions, or lexical bindings; rejects duplicate top-level declarations; and extracts explicit `(sorry "reason")` holes into the compilation manifest emitted on `stderr`. Unresolved heads trigger compile-time `HoleError`.
4. **`evaluate` (`netelpro/evaluator.py`)**: Tree-walking evaluation with tail-call optimization and runtime defense-in-depth. Executed only if all preceding static passes succeed.


## 13. Native Backend (Phase 5): LLVM via llvmlite

Status: verified 2026-09-05 on Windows x86-64 (llvmlite 0.49.0). The native backend is a
**strict subset compiler**: it accepts only programs whose values are representable in
machine words and rejects everything else with prosecutorial `CodegenError` diagnostics
carrying exact source coordinates. No silent fallbacks, no dynamic reinterpretation.

### 13.1 Compiled Subset

| Netelpro | Native representation |
|---|---|
| `Int` | `i64` (LLVM signed 64-bit) |
| `Bool` | `i1` (zext'd to `i64` only at `main` return) |
| `Str` | `i8*` NUL-terminated UTF-8: literals intern as internal constant globals; params cross read-only (`c_char_p`) — strings are inputs and comparisons, never products (return-Str is a compile error) |
| `+`, `-`, `*` | `i64` add/sub/mul (Int×Int→Int) |
| `quot`, `rem` | `sdiv`/`srem` — truncation semantics VERIFIED to match the interpreter in all 4 sign combinations (differential tests) |
| `<`, `<=`, `>`, `>=` | `icmp signed` → `i1` |
| `==`, `!=` | type-aware: `icmp` for i64/i1, `strcmp` (libc) for `i8*` → `i1` — operands are statically homogeneous |
| `prefix?` | `strncmp(text, prefix, strlen(prefix)) == 0` (libc) → `i1` |
| `print` | call to `printf` — `%lld` for Int, `%s` for Str (internal constant globals, selected by LLVM type) — requires `(grant io)` enforced at compile time |
| `not` | `xor i1 1` |
| `if`, `and`, `or`, `let` | SSA branches/phis; and/or short-circuit with merge blocks |
| `defn` | native LLVM function (name-collision-safe: `main` becomes `__sl_user_main__`) |
| self tail-call | **TCO back-edge**: params re-stored into alloca slots + branch to loop header — `sum-to 100000` and `count 500000` run in constant stack space |
| non-tail call | direct native `call` (recursion compiles; depth bounded by the machine stack, as in the interpreter) |
| division by zero | explicit zero check → prints `runtime error: division by zero` and `exit(1)` |

### 13.2 Rejected at Codegen (Compile Errors, Exact Coordinates)

`FloatLit`, `NilLit`, `ListLit`, `Fn` (anonymous), `Sorry`, `/` (returns Float),
List primitives (`cons head tail is-nil len nth`) and string PRODUCTION (`str-cat
str-len int->str str->int int->float` — native code decides over strings, it does not
build them), a bare `Str` in return position (read-only boundary), `def` of a non-literal
value, undefined symbols, and any type conflict found by the type-inference pass —
including heterogeneous `==` on concretely-typed operands (`Int vs Str`). **Type
inference is bidirectional** (TypeVar unification): parameter types are inferred from
use, so `(defn f (n) (if n 1 2))` compiles alone (n: Bool) and the conflict fires at the
incompatible call-site — the prosecutor reports it there, with coordinates. A bare Str
in return position is prosecuted ("cannot be a return value"): strings cross the
boundary read-only.

### 13.3 Prosecution Stages (the honest map)

Errors surface at the EARLIEST stage that can see them — verified, not assumed:

```
parse  →  caps  →  holes  →  codegen  →  JIT run
```

- **parse**: first-class calls, primitive/defn arity mismatches, reserved-head redefinitions.
- **caps**: un-granted `io` for `print` (both backends).
- **holes**: sorry manifest; executing a hole raises `StrayHoleError` (interpreter) —
  in the native backend a reached `sorry` is a compile error (a hole cannot produce a
  native artifact; the manifest already warned).
- **codegen**: everything in §13.2.

### 13.4 Differential Testing Contract

`tests/test_codegen.py` (80 tests) implements the language's thesis on itself: every
compiled program is executed in BOTH engines and the results must agree exactly
(`assert_agree`). The interpreter is the reference semantics; the native backend is
the verified implementation. Coverage: arithmetic (incl. all sign combinations of
quot/rem), comparisons, short-circuit logic, let-chains, def-literals, defn calls,
forward refs, mutual recursion, fib(15), TCO at 100k/500k depth, print via real fd 1
(subprocess), JIT artifact reuse, and every rejection of §13.2 with stage assertions.

### 13.5 CLI

```
python -m netelpro <file.sl>           # interpreter (default)
python -m netelpro --native <file.sl>  # LLVM native backend (same static passes)
```

`--native` runs parse → caps → holes → codegen → JIT. Static passes are IDENTICAL for
both backends — only the execution engine differs. A program that runs interpreted
but cannot compile natively is a documented v0.1 boundary (see §13.2), reported as a
compile error, never a silent divergence.

### 13.6 Implementation Notes (verified on this machine, llvmlite 0.49.0)

- LLVM target registration is explicit: `initialize_native_target()` +
  `initialize_native_asmprinter()` (the "automatic initialization" of 0.49 covers the
  core only, NOT the backends).
- JIT construction: `Target.from_default_triple().create_target_machine(opt=2)` →
  `create_mcjit_compiler(module, tm)`; native calls via `ctypes.CFUNCTYPE` on the raw
  address (`FunctionPointer`, `get_host_triple`, `new_engine`, `TargetMachine.from_triple`
  are all removed in 0.49).
- printf/exit are declared (`declare`) and resolved by the JIT's dynamic linker;
  format strings are internal constant global variables.---

## 14. Phase 6 — The Rule Filter Bridge (rule_filter.py)

The first real use case: Neuromancer gate rules (priority, confidence, escalation flags)
as **compiled pure Netelpro functions**. A host defines a rule, Netelpro prosecutes it
and compiles it to native code, and Python calls it directly.

### 14.1 Contract

```lisp
; The rule must define `filter-rule`. Params resolve statically to Int (i64),
; Bool (i1) or Str (i8* NUL-terminated, read-only). Strings cross via
; c_char_p (UTF-8); the return must be i1/i64 -- strings are inputs, never products.
(defn filter-rule (path approved mode)
  (if (or (== path ".env") (or (== path "routes.py") (== path "container.py")))
      false
      (if (or (prefix? path "src/") (or (prefix? path "tests/") (prefix? path "skills/")))
          (and approved (== mode 1))
          true)))
```

### 14.2 Pipeline and prosecution layers

`compile_filter(source)` runs the full static pipeline: parse → caps → holes → codegen.
Failures raise `RuleFilterError` with `.message/.line/.col` provenance. Additional
bridge-level prosecution:

- **missing `filter-rule`** → error (line 0, col 0) at construction;
- **param statically demanding Bool** → LEGAL since v0.2: the param compiles to an `i1`
  boundary param and crosses via `ctypes.c_bool` (only the low byte is read, per §14.3);
- **mixed use of one param** (Bool demanded in one site, Int in another) → rejected:
  `type mismatch` with exact coordinates (bidirectional inference conflict);
- **`print` without `(grant io)`** → rejected at compile (caps layer, unchanged);
- **declared `sorry` holes are legal** and listed in `.manifest()` — enumerated, never hidden.

### 14.3 Calling convention (verified finding, worker + orchestrator agreement)

User `defn` returns keep native types: Bool-returning functions emit `i1`. On x86_64
(Windows x64 and SysV alike) a 1-bit return value arrives in **AL**, with no guarantee of
upper-bit hygiene; the bridge uses `ctypes.c_bool` so only AL is read. Int rules return
i64. `decide(*args)` builds the `CFUNCTYPE` with `c_int64` per param from the parsed
arity — wrong arity raises TypeError, not UB.

### 14.4 Differential verification

`RuleFilter.verify(cases)` runs the **same rule in both engines**: native `decide()` vs
the reference interpreter (`run_source` with the call appended as final top-level form).
The interpreter is the reference semantics; the native backend is the verified
implementation. `verify()` returns mismatches as `(args, expected, interpreted, native)`
tuples; empty list = full agreement.

### 14.5 Verified evidence (this machine)

```
decide(3,80,0) = True   decide(2,80,0) = False   decide(3,95,0) = False
decide(2,95,1) = True   decide(4,10,0) = True    decide(1,99,0) = False
parity probe: decide(1001) = True   (1001 native recursion levels, TCO, no stack growth)
differential: verify() == [] on all cases; 304/304 suite green
```

### 14.6 Boundary (v0.2 → v0.3)

Params resolve per-param to `Int` (i64, `ctypes.c_int64`), `Bool` (i1,
`ctypes.c_bool`) or `Str` (i8*, `ctypes.c_char_p`, UTF-8) at the native boundary
(v0.2 Bool, v0.3 Str): an unused or Int-context param binds to Int; a param used as
`if`/`and`/`or`/`not` operand binds to Bool; a param compared to string literals
(`==`/`prefix?`) or printed binds to Str (verified: TCO back-edge round-trips both
the i1 slot at 500k levels and the pointer slot at 100k levels, native ==
interpreter). Mixed use is a compile error with exact coordinates. Heterogeneous
`==` on concretely-anchored operands (Int vs Str) is prosecuted directly with
coordinates. Strings cross READ-ONLY: the rule may compare and print them, never
return them (return-Str is a compile error). List production and string production
(`str-cat`, `int->str`, ...) remain interpreter-only. Future: multi-rule modules,
and host callback plumbing if a use case demands it.---

## 15. Consolidated State (v0.9, post-Phase 6)

What the language **is** on this machine, all mechanically verified:

### 15.1 The honesty stack (what kills what, where)

A Netelpro program passes four prosecution layers; each failure class dies at the earliest layer, with exact line/column:

| Stage | Layer | Kills |
|---|---|---|
| 1 | Parser (fiscal) | Unknown heads, arity violations, duplicate top-level defs, nested `def`/`defn`/`grant`, non-first-class heads, broken literals |
| 2 | Capabilities (`caps.py`) | Any capability use without a top-level `(grant ...)`, incl. buried in unexercised paths |
| 3 | Holes (`holes.py`) | Emits the declared-hole manifest (`sorry` with line/col/reason); unknown symbols in unexercised code |
| 4 | Codegen (`codegen.py`, `--native`) | Non-representable types at use (Float/Str/List/fn-as-value), Bool-demand violations at the boundary |

The program runs only after surviving all four. There is no stage where dishonesty passes silently.

### 15.2 Two backends, one semantics, mechanically compared

- Interpreter (`evaluate`) = reference semantics.
- Native backend (`compile_program` → MCJIT, i64/i1, TCO as structural loop) = verified implementation.
- Contract: every compiled program's results must equal the interpreter's (`verify()` differential testing; §13.4, §14.4).

### 15.3 Deliberate limits (documented, not accidental)

- Compiled subset: Int/Bool/Str (read-only), `+ - * / quot rem`, comparisons (incl. string `==`/`prefix?`), `not`, `if/and/or`, `let`, `def`, `defn`, calls, `print` (%lld/%s). String/list PRODUCTION (`str-cat`, `int->str`, `cons`, ...) remains interpreter-only: native code decides, it does not build data.
- Heads are primitives/special forms or top-level `defn` only — no first-class functions in v0.1.
- Params resolve per-param to Int (i64), Bool (i1) or Str (i8*) at the native boundary (v0.2/v0.3); unused params bind to Int by default.
- Capabilities are file-scoped grants (`{io}`); per-function effects deferred.
- `sorry` compiles as a declared hole with runtime tripwire — the manifest is emitted at every compilation.

### 15.4 What remains open for v1.0 (decision of Jona)

- **Final language name**: Netelpro — decided 2026-09-05 (formerly Straylight, working title).
- Naming of the remaining deferred features above (order and priority).

*Every claim in this section is backed by the test suite (`tests/`, 356 tests) and the examples (`examples/`) at the v0.3.0 tree.*