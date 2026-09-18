# netelpro/lib — standard library

Two pieces, both built from measured evidence about what the language can
actually do. Neither is speculative.

## `prelude.sl` — shared Bool helpers

`all-ofN` / `any-zeroN` for `defn` rule bodies.

**Netelpro has no `include`.** `SPEC.md` §8 keeps modules/imports out of scope,
and `(include "lib/core.sl")` is a parse error (`unknown head 'include'`). So
the host **concatenates** the prelude before the rule source:

```python
from netelpro.lib import concat_with_prelude
from netelpro.rule_filter import RuleFilter

filter = RuleFilter(concat_with_prelude(rule_source))
```

Verified against the real compiler (native JIT + reference interpreter):

| Property | Result |
|---|---|
| Concatenation preserves native/interpreter parity | ✅ |
| A prelude name colliding with the contract | **hard error** `duplicate defn 'x'` — never a silent shadow |
| An impure helper the rule never calls | gate purity stays GREEN |
| A `sorry` anywhere in the prelude | breaks **every** rule that concatenates it |
| The prelude alone | no `filter-rule` → `RuleFilterError` (it is a library, not a rule) |

That fourth row is the operational constraint: **the prelude must stay closed.**
`test_prelude_parses_and_has_no_holes` enforces it.

### Dialect law

The prelude is the **Bool** dialect. These helpers compose inside `defn`
bodies. They **cannot** be called from `truth-table` rows: a row slot is
`Int 0 1` and the row body must return Int, so a Bool helper is a type
mismatch. Do not add Int-dialect twins — see below for why they buy nothing.

## `contracts.py` — canonical contract generation

Renders the shape that dominates the host: *N declared slots, one
zero-rejection row per slot, an all-ones admit row, and the mandatory
all-wildcard backstop.*

```bash
python -m netelpro.lib --slots no-secrets,size-ok,no-nan-flood \
    --title "Contrato formal de postcondicion para file_rw."
python -m netelpro.lib --prelude
python -m netelpro.lib --from path/to/existing_contract.sl
```

### Why generation and not a helper

This is the finding that decided the design, and it is worth stating plainly
because the obvious approach does not work.

Twenty-three of the live host contracts are the same shape written by hand.
A prelude helper cannot factor them out, for two independent reasons:

1. **Exhaustiveness is syntactic over row *patterns*.** A wildcard row whose
   body calls a helper covers nothing:
   ```
   ((_ _ _) -> (all-of3-int x y z))
   ```
   → `does not cover the declared product; uncovered: (0,0,0); (0,0,1); …`
   — all eight. The compiler does not evaluate the helper to decide coverage;
   it reads the patterns. A partial wildcard buys back **one row**, not five.
2. **The Int dialect does not help either.** Int-returning helpers compile and
   verify fine, but they eliminate zero of the duplicated rows.

The duplication lives exactly where a helper cannot enter. Generation is the
mechanism that removes it — the same one this project already uses in
`builders/truth_table_builder.py` and `zone_rule_generator.py`: one source,
derived artifacts.

### The extractor is a guard rail, not a normalizer

`contract_from_source` parses a real `.sl` contract with the **real parser and
AST** (no regex) and returns its spec. It accepts *only* the canonical
all-of-N shape. Anything else — bespoke policy rows, a computed row body —
raises `LibError` rather than being quietly reshaped into something that
compiles but means something else.

Verdicts are read back from the real rows, so a canonical-shaped contract
using non-default verdicts (say `3`/`7`) round-trips with its own verdicts.

### Current coverage

`scripts/audit_contract_canonicality.py` (read-only) measures the live set:

```
CANONICAL (all-of-N, generatable): 23
BESPOKE (own policy, not generatable): 0

row lines currently hand-written in canonical set: 109
every canonical contract round-tripped with identical verdicts
over its full declared domain
```

The round-trip proof is `tests/test_netelpro_lib.py::test_roundtrip_matches_every_live_contract`:
for each live contract, extract → re-render → assert identical verdicts over
the whole declared domain on both the native and interpreter paths.

## Migration (done)

The 23 live contracts were migrated: their formal block is now generated and
carries a marker line naming the generator. The prose header of each contract
is untouched — slot names and their meanings still live in the file, and that
remains the source of truth for *what the contract means*.

```bash
python workspace/straylight/scripts/migrate_contracts.py --check   # read-only, default
python workspace/straylight/scripts/migrate_contracts.py --write
```

`--check` exits non-zero if any contract's formal block drifts from what the
generator would emit, so hand-editing a generated block fails loudly. The same
check runs in the test suite as
`test_live_contracts_have_no_generator_drift`.

What actually changed on disk, measured with `git diff`:

| Change | Files |
|---|---|
| Marker line added | 23 |
| Stray UTF-8 BOM removed (inconsistent between files) | 3 |
| Missing trailing newline added | 2 |
| Row semantics changed | **0** |

`--write` refuses to touch a contract unless both the old and new source
compile and decide identically over the full declared domain plus
out-of-domain backstop probes. The first `--check` run before writing reported
23/23 drifted, all by marker only.

### On the provenance digest

`epistemic_edge.record_verified_edge` hashes the `rule_source` string it is
given and stores it verbatim. It does **not** read these files from disk, and
no test hashes their text (verified by grep over `src/` and `tests/`). So the
migration does not change what that digest certifies — the earlier concern
that it would cover "generator + spec" instead of the contract applies only if
a caller starts feeding it generated sources. None does today.

## Not done yet, on purpose

- **No `include` in the language.** Concatenation works today with parity
  intact and zero compiler changes. `include` would require touching the
  parser, the audit path and provenance. Concatenate first; add `include` only
  if the usage actually demands it.
- **The prelude is not used by any live contract.** The 23 migrated contracts
  are truth-tables, and truth-table rows cannot call a Bool helper. The
  prelude is there for `defn`-style rules; it has no production caller yet.

## Packaging

`pyproject.toml` ships `lib/*.sl` in `package-data`. Without that line the
prelude would not travel in the wheel and an installed netelpro would lose it.

## One language-level observation (not a defect of this library)

On out-of-domain inputs the two backends behave differently:

- **Native (FFI):** accepts e.g. `2` in an `(Int 0 1)` slot and answers via
  the backstop row.
- **Reference interpreter:** rejects it at parse time —
  `literal 2 outside declared enumeration (0, 1) of parameter #1 of 'filter-rule'`.

So `verify_int` cannot be used to differential-test the backstop path; the
tests assert native-vs-native agreement there instead. This is preexisting
behaviour of the language, documented here because it is easy to trip over.
