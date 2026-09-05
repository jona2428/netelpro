# Straylight — Grammar Specification v0.1

Status: draft, Phase 0 deliverable. Single source of truth for arities: `spec/arity_table.json` (the spec and the table are kept in sync; the table is machine-consumed).

## 1. Design thesis (why this grammar exists)

Straylight is a programming language written **by** LLMs and **audited by** a compiler-as-prosecutor. The grammar is engineered so that an LLM can verify its own syntax **mechanically, without simulating a parser**:

1. **Prefixed, fixed arity.** Every form is `(head arg1 arg2 ...)`. Every head has a declared arity. To check a form, the writer counts operands between the head and the closing parenthesis. Counting is a mechanical operation an LLM performs reliably; simulating a recursive-descent parser is not.
2. **One exception, declared, still mechanical.** `list` is the only open-arity form. Its arity is closed by `)` itself, so counting still terminates the form. No other variadic head exists in v0.1.
3. **Zero human ceremony.** No imports, no class boilerplate, no type annotations at use sites. Every token carries meaning.
4. **Reserved grammar for the prosecutor.** `sorry` and `grant` are part of the grammar from v0.1 even though their enforcement lands in Phases 4 and 3. The syntax cannot be invented later; the honesty model must fit from day one.

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

```straylight
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

```straylight
(sorry "reason as a string literal")     ; yields a value of whatever type context demands;
                                         ; Phase 4: silent holes = compile error, reason mandatory
(grant io net)                           ; top-level only; declares file-wide capabilities;
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

Macros, modules/imports, records/maps, pattern matching, generics, effects, lazy evaluation, `do`. Each enters only when a phase requires it, and each must be expressible with fixed arity or an explicitly declared open-arity form.## 9. Phase 1 implementation notes (prosecutor tightened)

Phase 1 delivered the full frontend: `straylight/lexer.py` (tokenizer), `straylight/ast_nodes.py` (frozen typed AST), `straylight/parser.py` (recursive-descent parser + mechanical arity validation). 94 tests green.

The parser is deliberately STRICTER than `tools/check_arity.py`. Documented divergences (parser enforces, Phase 0 fiscal does not):

1. **Reserved heads cannot be bound by `let`** — `(let if 5 ...)` and `(let + 10 ...)` are compile errors. Rationale: with no first-class functions in v0.1, rebinding special forms or primitives in local scope creates ambiguity for every downstream pass.
2. **`def`/`defn` are top-level only** — nested `(def a (def b 2))` or `(+ 1 (defn f (x) x))` are compile errors. Rationale: nested defs would silently miss the defn registry and produce misleading "unknown head" errors at call sites.
3. **Duplicate top-level `defn` is a compile error** — no silent registry overwrite. The last declaration still wins the arity registry (mechanical determinism for call checks), but the collision is reported.
4. **Symbols cannot start with `-`** — the Phase 1 lexer restored the spec regex `[A-Za-z_][A-Za-z0-9_?->]*`; `-foo`, `->`, `--` are invalid tokens (Phase 0 fiscal already rejected them; a worker regression briefly allowed them and the reviewer caught it).
5. **`sorry` requires a STRING literal; `grant` is top-level only with symbol operands** — both reserved since Phase 0.

Deferred to Phase 1.5 (reviewer findings, non-blocking): composite AST nodes use mutable `list` fields — unhashable, fine for current passes; will become tuples if/when AST sets are needed. `let` re-binding of a *user* defn name in an inner scope is currently accepted (lexical shadowing) — revisit when the evaluator lands (Phase 2).

Design call of record: literal nodes are separate subclasses (IntLit/FloatLit/StrLit/BoolLit/NilLit) instead of a kind-tagged Lit — `bool` is a subclass of `int` in Python and a single Lit node would need runtime value inspection; separate classes keep every pass type-safe by construction.## 10. Phase 2 implementation notes (evaluator semantics — design decisions of record)

Phase 2 delivered the tree-walking evaluator (`straylight/evaluator.py`), CLI (`straylight/__main__.py`), and a 163-test suite (94 frontend + 69 evaluator). All green.

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