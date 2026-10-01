# Receipts — File-Effect Honesty, Specification v0.1

**Status:** v0.1 implemented, incremental snapshots added (§9) — `netelpro/receipts.py`,
`netelpro/rules/mutation_receipt.sl`, `examples/receipts_demo.py`,
`tests/test_receipts.py` (58 cases), `tests/test_receipts_demo.py`,
`netelpro/mcp_server.py` tool `netelpro_receipts` + `tests/test_mcp_receipts.py`.
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
- **Cost.** Resolved by the incremental fast path (§9): with a cache the
  take is one `stat` per file plus the bytes that actually changed. The
  first take of a root is still linear in its bytes.
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

**From inside the agent (MCP):** `netelpro_receipts` on the stdio server
(`docs/MCP.md` §3.5). `show` is "what did I actually change", `audit` is
"is my draft honest about it", both read-only against a baseline the
server hashed before the first request. The root is `NETELPRO_RECEIPTS_ROOT`
of the server process, never a model argument; there is no `begin`/`end`
action, so the model cannot move the baseline to erase its own receipts.

```bash
NETELPRO_RECEIPTS_ROOT=/path/to/repo python -m netelpro --mcp
```

---

## 9. Incremental snapshots (fast path)

Each `Snapshot` carries, per path, the **stat signature** the hash was taken
under: `(size, mtime_ns, ctime_ns, inode)`, plus `taken_ns`, the instant the
walk started. `snapshot(root, cache=previous)` reuses the cached hash of a
file only when all of these hold:

1. its signature equals the cached one;
2. its mtime **and** ctime are both older than `previous.taken_ns -
   RACY_WINDOW_NS` (2 s). A write that lands inside the same timestamp tick
   as the hash leaves size and mtime unchanged — git's "racy" case — so
   anything touched near the cached take is never trusted by that cache;
3. the cached value is a real digest, not the unreadable sentinel.

Everything else is re-read. A plain `dict` cache (no stats) reuses nothing
and gives the same result; `incremental=False` on the guard and `--full`
on the CLI force a full re-read.

**Why not size + mtime.** The threat model is an agent with a shell. After
a same-size edit it can run `touch -d` / `os.utime` and restore the exact
mtime; size+mtime would then reuse the stale hash and the receipt would
vanish. `ctime` is set by the kernel on every inode change, utime included,
and cannot be chosen from user space on Linux/macOS; a rename-over gets a
new inode. Both evasions are tests
(`test_incremental_detects_same_size_edit_with_forged_mtime`,
`test_incremental_detects_rename_over_with_forged_mtime`).

**Residual holes, declared:** Windows, where `st_ctime` is the creation
time and only the racy window and size+mtime defend; a root that writes
the raw device or steps the clock; a `.netelpro/snapshot.json` the model
is allowed to edit (keep the state dir out of its write scope, or run
`--full`).

**Measured on this repository** (420 files, 97 MB tracked, cloud container
disk, 2026-10-01): full take 9096 ms with 420 files hashed; incremental take
11 ms with 0 hashed and 420 reused, identical result. The MCP tool and the
CLI `end` use the baseline as cache automatically; `begin` uses the previous
turn's snapshot.

---

## 10. Claude Code hooks (zero-wrap integration)

```bash
pip install netelpro
python -m netelpro.hooks.claude_code install            # in the repo you work in
python -m netelpro.hooks.claude_code install --strict   # silent writes block too
```

`install` writes the absolute interpreter path into three hooks in
`.claude/settings.json` (`--local` for `settings.local.json`), merging with
whatever is already there, and adds `.netelpro/` to `.gitignore`. Restart
Claude Code in the repo to activate.

| Event | Hook does | Output to Claude Code |
|---|---|---|
| `SessionStart`, `UserPromptSubmit` | baseline of the turn (incremental) | `additionalContext`: "turn N, M files hashed; file claims are checked against receipts when you stop" |
| `Stop`, approved | commit receipts | nothing |
| `Stop`, rejected, first time | keep the turn open | `{"decision":"block","reason":...}`: each claim with no receipt, the ground truth, "perform the edit or correct the message" |
| `Stop`, rejected again (`stop_hook_active`) | commit, allow the stop | `systemMessage` to the user naming the claims still unsupported |

The second row of the rejection path is the important design choice: the
model gets exactly one correction round. Netelpro's job is to make the
false sentence visible and costly, not to hold a session hostage.

**Fail-open, deliberately and only for the hook's own problems:** no
baseline for this turn (installed mid-session) takes one now and lets the
stop through with a `systemMessage`; a corrupt ledger or an internal error
prints to stderr and exits 1, which Claude Code shows as a non-blocking
hook error. The only thing that ever blocks is a real verdict from the
compiled rule.

Residual limits: the hook audits the final text only (intermediate
narration during tool use is not judged); `SubagentStop` is not wired;
attribution (model vs user vs build writes) is still observational.
