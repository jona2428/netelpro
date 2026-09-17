# Inference-Time Repair Loop — Specification v0.1 (DRAFT)

**Status:** DRAFT — spec approved by Jona in conversation ("armemos esto...
arma spect y veamos qué sale de esto", 2026-09-17). Pilot implementation to
follow immediately after this file, same session.
**Origin:** 2026-09-17 conversation, following the llama-cpp-python gate fix
(`5286c73`) and `STATE_TRACKING_GATE_SPEC.md`'s rate-limiting pilot.
**House precedent:** same spec-first protocol as `EPISTEMIC_GATE_SPEC.md` and
`STATE_TRACKING_GATE_SPEC.md`.

---

## 1. What this is

Layer B (`NetelproStreamProcessor`) masks individual tokens in real time —
cheap, and it prevents an out-of-contract token from ever being sampled. But
it can only check what a *single token id* is allowed to be. It cannot check
whether a *finished answer* is actually correct — whether a generated
Netelpro program compiles and passes its test cases, whether a claim matches
real state, whether the math is right. That's a different, stronger kind of
check, and it can only run after there's a complete candidate to check.

Today, when that stronger check fails, the honest thing the system does is
reject and stop — fail-closed, exact reason, same as everything else in this
repo. This spec is about the step after that: instead of stopping at the
first rejection, feed the exact rejection reason back to the model and
retry, bounded, until a real verifier says the candidate is correct or the
retry budget runs out. This is the same idea RAFT already uses to build
training data offline (`training/train_raft_colab.ipynb`, `rlvr/verify.py`:
sample, verify against a compiled oracle, keep only what passes) — applied
live, at inference time, on a single request, instead of accumulating a
training pool.

**What this targets, concretely (from the conversation that requested it):**
an agent that has drifted after long context, or a generative model
producing something that looks plausible but fails a real check the moment
it's run against ground truth (a compiler, a test suite, a state
invariant). The fix under this spec is not "mask the token and move on" —
it's "tell the model exactly why it was wrong and let it try again," bounded
so it can't retry forever.

---

## 2. Core idea

```
GENERATE  -> model produces a full candidate (not token-by-token gating --
             Layer B still runs underneath if attached, this is a separate,
             stronger, whole-output check)
VERIFY    -> a real oracle checks the candidate (rlvr.verify for the v0.1
             pilot: compiles? passes N test cases? exact error if not)
PASS?     -> yes: return the candidate. done.
          -> no: was this the last allowed attempt (RetryLimiter)? 
             yes: fail closed, return the LAST verifier's exact reason --
                  never a silently-wrong answer.
             no: build the next prompt = original prompt + this attempt +
                 the exact verifier error, and GENERATE again.
```

The loop is bounded by `netelpro.state_gate.RetryLimiter` — already built
for exactly "how many attempts have been made, when to stop," reused as-is,
not reinvented. Exhausting the budget is not a special case to design: it's
the same fail-closed contract `RetryLimiter.attempt()` already returns
(`allowed=False, reason=...`).

---

## 3. Relationship to what already exists

| Piece | Reused as |
|---|---|
| `netelpro.state_gate.RetryLimiter` | Attempt budget and cooldown for the whole loop — unmodified |
| `rlvr.verify.VerifyResult` / verifier | The v0.1 oracle: compiled/passed/cases_passed/error, already exists, already used by RAFT for the exact same sample-then-verify shape |
| `netelpro.gate.Gate` / `RuleFilter` | Same fail-closed pattern (explicit reason, never silent) the loop's termination follows |
| Layer B (`NetelproStreamProcessor`) | Orthogonal, not replaced — can run underneath each generation attempt to prevent individual out-of-contract tokens, while this loop handles whole-candidate correctness on top |
| `contract_gate_llama_cpp_demo.py` | The real local model wiring (llama-cpp-python, in-process) this pilot generates from |

This is not a new mechanism grafted onto the project — every piece it needs
already exists for a different purpose (RAFT's offline training loop,
RetryLimiter's budget tracking, the compiler's exact-reason diagnostics). The
new part is *closing the loop live, in one request*, and feeding the reason
back into the next prompt instead of only into a training log.

---

## 4. Open questions (holes)

1. **Does feedback-in-prompt actually help, or does the model repeat the
   same mistake?** This is empirical, not assumed — the pilot must measure
   pass rate across retry attempts on real generations, not claim
   improvement without running it.
2. **What varies between attempts, beyond the fed-back error?** Candidates,
   not mutually exclusive: (a) nothing else — the error text alone is the
   signal; (b) escalate temperature/top_k on each retry (more exploration
   after a failure); (c) different seed only. v0.1 pilot uses (a) alone
   first — simplest, and isolates whether feedback itself does anything
   before adding more variables.
3. **Partial credit** — `VerifyResult.cases_passed / cases_total` gives a
   score, not just pass/fail. Does the loop ever accept a "close enough"
   candidate, or does it always require a full pass? v0.1: always requires
   full pass — partial credit is a threshold decision with real consequences
   (a wrong answer is not "60% honest"), not something to default into
   without deciding it deliberately.
4. **Cost** — each retry is a full regeneration (seconds), not a masked
   token (microseconds). How many retries before this stops being usable
   in a live path vs. only in an offline/batch context? The pilot's
   `RetryLimiter` cap makes this a tunable number, not an unbounded loop,
   but the actual ceiling that's still "live-usable" needs measuring, not
   guessing.
5. **Prompt template for feedback** — how the previous attempt and its
   exact error get phrased back to the model matters and isn't designed
   yet beyond "include the error verbatim." Left to the pilot to draft and
   iterate on empirically.

---

## 5. v0.1 pilot scope

Reuse a task already defined in `rlvr/tasks/` (a train task, not one of the
`OOD_TASK_IDS` contract tasks — reusing an OOD task as a repeated benchmark
target would erode its held-out status). Generate candidates from a real
local model via `llama-cpp-python` (same wiring as
`contract_gate_llama_cpp_demo.py`). Loop bounded by `RetryLimiter` (a small
cap, e.g. 3-5 attempts, tunable). On each failed attempt, verify with
`rlvr.verify`, log the exact `VerifyResult`, and build the next prompt with
the previous candidate + its exact error appended. Report, honestly: did
pass rate improve attempt-over-attempt on the tasks tried, or not — this is
the one falsifiable claim the pilot has to answer, same standard as every
other benchmark in this repo (real numbers, not promised ones).

No code in this spec itself. Implementation follows immediately, same
session, per the go-ahead already given.

---

## 6. v0.1 pilot results (2026-09-17)

Implemented as `examples/inference_repair_loop_demo.py`. Run against
`qwen2.5-1.5b-instruct.Q4_K_M.gguf` — a real, general-purpose local model
with **zero training exposure to Netelpro syntax**, chosen deliberately so
first-attempt success isn't a given.

**`sum_range` (easy task): passed on attempt 1.** No retry needed — not
informative about the loop itself, but confirms the harness (prompt, fence
extraction, `verify_program` wiring) works end to end on a real model.

**`gcd_pair` (hard task, documented elsewhere in this repo as OOD-difficult):
failed all 5 attempts, retry budget exhausted, failed closed with the real
compiler error.** This directly answers open question 1, for this
task/model/strategy combination: **feedback-in-prompt alone (open question
2's option (a)) was not enough.** The model oscillated between two distinct
wrong patterns across the 5 attempts — first calling `-` with one operand
(Netelpro has no unary minus), then, after that error was fed back,
inventing `zero?`, a head that doesn't exist in Netelpro's arity table at
all — never converging toward a working answer. It did not learn from the
specific error text; it substituted one plausible-sounding mistake for
another.

One concrete, actionable lesson from this negative result: the primer given
to the model never listed which heads actually exist — the model had to
guess, and guessed a real Scheme/Lisp idiom (`zero?`) that Netelpro doesn't
have. A primer that includes the actual allowed head list (from
`spec/arity_table.json`, the same machine-consumed source of truth the
compiler itself checks against) rather than two example programs would
likely remove this entire failure class — worth testing before concluding
feedback-in-prompt doesn't work at all, since this run conflates "doesn't
know Netelpro's real vocabulary" with "can't use error feedback to repair a
known-vocabulary mistake."

Not yet tried (open question 2's options (b)/(c) — temperature escalation,
seed variation) and not yet decided whether they would have helped here
instead of, or in addition to, a better primer. Both are the natural next
experiment, not run in this pass — the honest result of this pilot is "the
simplest version of the idea doesn't reliably work yet," not "the idea
doesn't work."

## 7. Follow-up (2026-09-17, same day): real head list closes the syntax gap, exposes a semantic one

Implemented the lesson from §6: `_build_syntax_primer()` in
`examples/inference_repair_loop_demo.py` now renders the primer from
`spec/arity_table.json` directly (every special form and primitive that
actually exists) instead of a hand-picked subset plus two examples.

Re-ran `gcd_pair` with the real primer, same model, same 5-attempt budget:

- **All 5 attempts now compile** (was 0/5). The syntax-guessing failure
  class (`zero?` and similar invented heads) is gone — confirms the §6
  hypothesis.
- **All 5 attempts still fail, and fail the SAME way**: `caso (299, 457):
  esperado 1, obtuvo 299` — identical error, every attempt. The model wrote
  semantically the same broken function five times in a row (it returns `a`
  unconditionally instead of actually recursing toward the base case),
  despite the exact failing case being fed back verbatim each time.

This is a different, more specific answer to open question 1 than §6's:
feedback-in-prompt did its job for the class of error it can actually fix
(syntax — the model now emits legal Netelpro every time), but for a
*semantic* bug it did nothing observable here, because attempts at
`temperature=0.2` are similar enough to each other that the model reproduces
close to the same program regardless of the error text. The loop isn't
exploring alternative logic; it's re-emitting the same logic. Open question
2's option (b) — escalating temperature or otherwise varying sampling
across retries, not just the fed-back text — is now the concrete next
experiment, not a hypothetical one: this run is the evidence that it's
needed, at least for semantic (not syntactic) failures.

Not yet run. Left for the next session on this spec.

## 8. Follow-up: temperature escalation breaks the frozen repeat, doesn't reach a pass

Implemented §7's next experiment: `temperature_for_attempt()` escalates
linearly (default: 0.2 base, +0.25/attempt, capped at 1.3) instead of a
fixed 0.2 for every retry.

Re-ran `gcd_pair`, same model, 6-attempt budget:

- Attempts 1-2 (temp 0.20, 0.45): same frozen wrong answer as §7 —
  low temperature is still too similar to itself to escape the repeat.
- Attempts 3-5 (temp 0.70-1.20): stopped repeating the same wrong logic —
  but instead produced **runaway recursion** (hit the 1,000,000-step
  execution budget, `verify_program`'s interpreter correctly killed it
  rather than hanging) three attempts running, a different failure mode,
  not progress toward correctness.
- Attempt 6 (temp 1.30, the cap): degenerated into **invalid syntax** — an
  unclosed paren. Too much randomness broke output that was reliably
  well-formed at lower temperatures.

Honest reading: escalation did what it was supposed to do narrowly — it
stopped the model from reproducing the exact same wrong program — but
traded "confidently wrong" for "runaway or malformed," and never converged
to a pass within 6 attempts on this task. This is evidence against a naive
linear escalation being sufficient on its own for `gcd_pair` specifically,
not evidence that escalation is worthless — it visibly changed behavior,
just not toward the target. Two candidate next moves, neither tried yet:
(a) a narrower escalation range (this run may have jumped too far, too
fast, skipping over a temperature band that might actually help), or (b)
a stronger primer for this specific task shape — mutual two-argument
recursion with the base case on the SECOND argument — since gcd's harder
part may be structural (which argument shrinks, and how) rather than
vocabulary, which is a different kind of hole than either fix so far
addressed.
