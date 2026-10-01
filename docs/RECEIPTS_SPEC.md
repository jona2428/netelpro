# Receipts — File-Effect Honesty, Specification v0.1

**Status:** v0.1 implemented — `netelpro/receipts.py`,
`netelpro/rules/mutation_receipt.sl`, `examples/receipts_demo.py`,
`tests/test_receipts.py` (58 cases), `tests/test_receipts_demo.py`.
**Origin:** 2026-10-01 conversation — "algo orientado a combatir las
alucinaciones y el impedimento de modificación de archivos en ejecuciones
de las LLM".
**House precedent:** `STATE_TRACKING_GATE_SPEC.md` (state the model must not
misremember lives outside its context; the model reads it, never recalls
it) and `guard.py` (a claim in prose requires evidence outside the prose).
This spec applies both to the one kind of state an agent lies about most
often without meaning to: **the files it says it wrote.**

---

## 1. The problem this targets

An agent harness (Claude Code, OpenCode, Cursor, a LangChain loop) lets a
model call a write tool and then narrate. Four things go wrong, and the
prose looks identical in all four:

1. The write tool was never called. The model described an edit it planned
   or imagined ("Actualicé `config/settings.py`").
2. The tool was called and **failed** — permission denied, read-only mount,
   sandbox policy, path outside the allowed root — and the model reported
   success anyway. This is the "impedimento de modificación" case: the
   harness blocked the write, the model did not notice, or noticed and
   kept narrating.
3. The tool was called on the **wrong path** (basename collision, a path
   relative to the wrong directory) and the prose names the intended one.
4. The write happened, the description is false in kind: "updated X" when
   X did not exist before (the model's belief about the starting state was
   wrong — write drift, in `STATE_TRACKING_GATE_SPEC.md` §1 terms).

`HonestyGuard` cannot see any of these: a non-empty `tool_results` is its
evidence, and in cases 2–4 there *is* a tool result. The evidence standard
has to be the bytes.

---

## 2. Core idea

```
begin()   -> hash every file under the workspace root (sha256)      [harness]
            agent turn runs; the model writes, or thinks it writes
end()     -> hash again; every differing path becomes a RECEIPT:
             (turn, kind ∈ {created, modified, deleted}, path,
              sha256 before, sha256 after), appended to a hash-chained
              append-only LEDGER the model has no write path to
audit()   -> extract MUTATION CLAIMS from the model's text (path, kind)
          -> for each claim, look up the receipt for that path
          -> a compiled Netelpro rule decides: is this claim kind admitted
             by this receipt kind? (fail-closed: no receipt = reject)
          -> strict mode also asks the rule about every receipt no claim
             mentions (a silent write)
```

The asymmetry, same as every other gate in this repo: a receipt is not
"the model said it wrote the file". It is "the bytes differ between two
observations the model did not make." Belief about an effect requires a
receipt; a claim without one is verification theater on file state.

The ledger is also a **read path**: `ground_truth()` renders the turn's
receipts as a block the harness puts in front of the model, so the model
*reads* what changed instead of recalling it. A fresh process with no
memory of the turn reloads the JSONL ledger and reaches the same verdict —
`tests/test_receipts.py::test_guard_survives_total_context_amnesia` is the
falsifiable form of that claim.

---

## 3. The rule

`netelpro/rules/mutation_receipt.sl`, three functions, Int/Int/Bool:

```netelpro
(defn receipt-is-write (receipt)
  (or (== receipt 1) (== receipt 2)))

(defn claim-matches (claim receipt)
  (if (== claim 4)
      (receipt-is-write receipt)
      (== claim receipt)))

(defn filter-rule (claim receipt strict)
  (if (== claim 0)
      (or (not strict) (== receipt 0))
      (claim-matches claim receipt)))
```

| `claim` | meaning | | `receipt` | meaning |
|---|---|---|---|---|
| 0 | no claim about the path | | 0 | sha256 unchanged (no receipt) |
| 1 | "created" | | 1 | created |
| 2 | "modified / updated / edited / fixed" | | 2 | modified |
| 3 | "deleted / removed" | | 3 | deleted |
| 4 | "wrote / saved / added / generated" (lenient: 1 or 2) | | | |

Laws, as written in the rule header: (1) a claim of effect needs a receipt
of the same effect; (2) kind matters — "updated X" when X was created is a
false statement about the world, only the verb class that does not assert
prior existence (4) is lenient; (3) no claim + receipt is the strict-mode
question.

The full domain is 5 × 4 × 2 = 40 rows. All 40 are checked against a
Python oracle and differentially (native LLVM vs reference interpreter) in
`tests/test_receipts.py`. Zero mismatches, zero sorry holes.

---

## 4. Claim detection

Regex over the finished turn, ES + EN, same posture as `guard.py`:

- **Verb classes** are past / perfective / participle only — completed
  effects. Infinitives, imperatives and progressives ("modificar",
  "update", "updating") are not claims.
- **Active**: optional subject, verb, up to 60 chars of filler that never
  crosses a clause boundary, then a path-like token (has a letter-initial
  extension or a `/`; backticks and quotes optional). Conjunctions extend
  to further paths ("modifiqué `a.py` y `b.py`").
- **Passive / resultative**: path first ("`a.py` has been updated", "el
  archivo a.py fue modificado", "a.py quedó actualizado").
- **Container preposition** between verb and path ("agregué la función a
  `x.py`", "removed the import from x.py") makes the path a container:
  claim kind 4, whatever the verb said.
- **Scoped out** (40-char window before the verb, cut by a clause
  boundary, same mechanism as `HonestyGuard._scope_blocks_claim`):
  negation (no / not / sin / without / n't), attempt (intenté / tried),
  intent and future (voy a / going to / will), obligation and ability
  (debería / should / could / can't), conditionals (si / if), questions,
  and adjectival participles ("the updated config.py").
- A path inside a URL is not a path.

Over-matching is the safe direction (a detected claim with a matching
receipt costs nothing); the scoping exists so it does not become noise.
Labeled corpus: 15 positive / 17 negative cases in `tests/test_receipts.py`,
each with the reason for its label.

---

## 5. Relationship to what already exists

| Piece | Reused as |
|---|---|
| `netelpro.gate.Gate` / `RuleFilter` | The decision: compiled once, `check()` per claim, fail-closed with reason |
| `guard.py` scoping (negation window + clause boundary + question span) | Same mechanism for mutation verbs, extended with attempt / intent / modal blockers |
| `STATE_TRACKING_GATE_SPEC.md` §2–3 | Store outside the context, read path instead of recall, write path only through a contract — here the "write path" is observation, not a model proposal |
| `aletheic.py` file_exists / file_content | Orthogonal: aletheic judges *assertions about the world*; receipts judge *assertions about one's own effects on it* |
| Layer B (`NetelproStreamProcessor`) | Not wired. A future step could mask the claim tokens for paths with no receipt during generation, so the false sentence is never produced |

---

## 6. What this is not (declared limits)

- **Attribution.** A receipt says the bytes changed, not who changed them.
  `origin="observed"` is a diff; a harness that intercepts tool calls can
  append `origin="tool"` receipts, but the v0.1 guard treats them alike.
  A concurrent user edit or a build step shows up as a receipt.
- **Correctness.** A receipt proves the file was written, not that the
  edit is right. That is the inference-repair loop's job
  (`INFERENCE_REPAIR_LOOP_SPEC.md`), not this one's.
- **Prevention.** Post-hoc, like `HonestyGuard` Layer A. The false
  sentence is rejected after generation, not prevented during it.
- **Renames.** Observed as delete + create. A rename claim ("renombré X a
  Y") is not a verb class yet.
- **Cost.** `snapshot()` hashes every byte under the root, twice per turn.
  Fine for a repository; wrong for a 50 GB tree. Hole: an mtime+size fast
  path that re-hashes only candidates, or inotify. Deliberately not built
  until a real tree demands it.
- **Detection recall.** Regex. Some phrasing will get through ("the change
  landed in X"). Each miss found in live generation should be added to the
  labeled corpus with provenance, as `guard.py` does.

---

## 7. Honest novelty assessment

Each piece exists somewhere: `git diff` after an agent run, harnesses that
list touched files, hash-chained logs, claim-detection regexes. What did
not exist in one place: a harness-agnostic observer (no tool API to hook —
it works on any agent that touches a directory), a ledger the model reads
but cannot write, a *kind-aware* claim-vs-receipt judgment (created ≠
modified ≠ written), decided by a rule small enough to verify over its
entire input domain, with the silent-write direction covered too. The
novelty is the composition and the verification standard, not any single
part. Saying more than that would itself be theater.

---

## 8. Integration surfaces

**Zero-integration CLI** (any harness, any language):

```bash
python -m netelpro.receipts begin                      # before the agent runs
# ... agent runs ...
python -m netelpro.receipts end                        # record receipts
python -m netelpro.receipts audit --text answer.md     # exit 0 ok, 2 theater
python -m netelpro.receipts audit --strict --json --text -   # from stdin
python -m netelpro.receipts show                       # ground truth + chain check
```

State lives in `<root>/.netelpro/` (`snapshot.json`, `receipts.jsonl`);
the directory is in `DEFAULT_IGNORE` so it never receipts itself.

**In-process:**

```python
from netelpro.receipts import MutationGuard, MutationTheaterError

guard = MutationGuard(repo_root, strict=False)
guard.begin()
# ... agent turn ...
audit = guard.audit(final_text)          # end() implied
if not audit.approved:
    for reason in audit.reasons: ...     # exact path, exact unchanged hash
prompt_block = guard.ground_truth()      # feed the next turn the facts
```

Next step, not built: a `netelpro_receipts` MCP tool so the model itself
can call "what did I actually change this turn" through the existing
stdio server (`docs/MCP.md`), closing the read path from inside the
agent instead of only from the harness.
