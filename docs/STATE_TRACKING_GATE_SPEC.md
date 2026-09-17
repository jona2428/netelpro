# State-Tracking Gate — Specification v0.1 (DRAFT)

**Status:** v0.1 pilot implemented (§5) — `netelpro/state_gate.py`,
`examples/gates/rate_limit.sl`, `tests/test_state_gate.py`. Sections 1-4
(the general architecture) remain DRAFT beyond this one pilot rule.
**Origin:** 2026-09-17 conversation — "cómo abarcar área que ahora cubren los
mismos LLM que fallan en predicción y alucinan a lo largo del contexto".
**House precedent:** `EPISTEMIC_GATE_SPEC.md` (spec-first, live compiled rule
inside, node/edge/hole concept model, Jona's approval before any code). This
spec reuses that model rather than inventing a new one.

---

## 1. The problem this targets

An LLM's belief about "what's currently true" comes from re-reading its own
context window. Over a long conversation or a long agent run, that recall is
lossy — attention degrades over distance, summarization drops detail,
contradictory statements from 40k tokens ago don't get reconciled. Two
concrete failure modes:

- **Read hallucination**: the agent asserts a fact about current state
  ("budget has $340 left", "the user already declined a refund", "step 3 of
  the plan succeeded") that is stale or was never true. It is not lying on
  purpose — it is recalling from a lossy medium (its own context) instead of
  from a source of truth.
- **Write drift**: the agent's internal model of "what just changed"
  diverges from what actually changed, and the error compounds silently
  because nothing ever checks the agent's belief against ground truth.

Every enforcement layer Netelpro already ships (the compiled decision gate in
`netelpro.guard`, the per-token gate in `netelpro.neuro`) operates on a single
turn or a single generation step. None of them hold state *across* turns. This
spec is about that gap.

---

## 2. Core idea

State that matters (budgets, retry counters, plan progress, declared consent,
anything an agent must not misremember) lives in a small structured store
**outside the LLM's context window** — not something the model is ever asked
to recall. Every read of "what is the current state of X" is answered by
querying the store directly. Every write (the agent proposing a state change)
must pass a compiled Netelpro contract before being committed — same
fail-closed pattern as `Gate.check()` today, just applied to state mutation
instead of text claims or token IDs.

This reuses `EPISTEMIC_GATE_SPEC.md`'s concept model directly instead of
inventing a new one:

| Concept (from EPISTEMIC_GATE_SPEC) | Reused here as |
|---|---|
| Node | A tracked state variable (`retries`, `budget_cents`, `plan_step_3_done`) |
| Edge | A verified transition: compiled contract that approved a specific state change, with what evidence backed it |
| Hole | A state variable the agent is asking about that has no verified value yet (must be fetched from a real source, not assumed) |

The key asymmetry, same as the epistemic gate: an edge is not "the agent said
so" — it is "a compiled contract approved this transition against the prior
verified state." Belief about current state requires a passing contract
evaluation; everything else is a hole the agent must resolve before acting on
it.

---

## 3. Architecture sketch

1. **State store** — append-only log of verified facts: `(key, value,
   evidence, timestamp, verifying_source)`. Not writable by the LLM directly;
   only by a passing contract evaluation.
2. **Read path** — when the agent needs "current value of X", the harness
   answers from the store, never from asking the model to recall it from
   context. This alone removes a large share of long-context hallucination:
   the model is never the source of truth for its own past.
3. **Write path** — the agent proposes a transition:
   `(state_key, old_value_claimed, new_value_claimed, evidence)`. A compiled
   `.sl` rule checks, in order:
   - `old_value_claimed == store.get(state_key)` — catches drift where the
     agent's belief about the *starting* value was already wrong, before even
     considering the new value.
   - The domain invariant holds for the transition itself (e.g.
     `new_value >= 0`, `retries + 1 <= max_retries`, `step_n` cannot complete
     before `step_(n-1)`).
   - The evidence requirement is met — reuses `HonestyGuard`'s
     claimed/verified pattern: a write claiming "confirmed via tool" must
     carry real `tool_results`, not just an assertion.

   Any failed check: fail-closed, the transition is rejected, the store is
   unchanged, and the agent gets the exact reason (same prosecutorial style
   as the rest of Netelpro — coordinates and cause, not a generic denial).
4. **Per-domain contracts** — the invariant is a small `.sl` rule per state
   family, same shape as `zone_policy.sl` / `action_boundary.sl` already in
   the repo, not one large rule for everything.

### Relationship to what already exists

- Reuses `Gate` / `RuleFilter` as-is for the write-path evaluation — no new
  compiler or runtime machinery.
- Reuses the node/edge/hole model from `EPISTEMIC_GATE_SPEC.md` for the store
  shape, so the two specs can eventually share storage rather than each
  inventing their own.
- Extends `HonestyGuard`'s claimed/verified distinction from auditing prose
  claims to auditing state-mutation claims.
- Composable with the Layer B token gate (`NetelproStreamProcessor`): a
  transition the store would reject could also have its *claim tokens* masked
  at generation time, so the agent is structurally prevented from asserting
  the change in the first place, not just corrected after the fact. Proactive
  masking is strictly more work to wire (needs the state check available
  *during* decoding, not just after); reactive rejection (this spec's
  baseline) is the cheaper first cut.

---

## 4. Open questions (holes, following house convention)

1. **Store backend** — `EPISTEMIC_GATE_SPEC.md` already picked DuckDB for its
   entity_graph. Same backend here (share the database) or a separate,
   simpler store (sqlite, or even an in-memory dict for the v0.1 pilot)?
2. **Granularity** — one `.sl` rule per state key, or one rule per domain
   covering several related keys? Netelpro's arity-checked contract model
   favors small single-purpose rules; a shared rule risks becoming another
   `action_boundary.sl`-style generic gate that doesn't actually encode the
   domain.
3. **Who calls the read/write path** — the agent harness (Neuromancer)
   wrapping every tool call transparently, or does the LLM emit explicit
   structured "state operations" that the harness intercepts and evaluates?
   The first is more transparent to the agent but requires harness-level
   instrumentation; the second is easier to build first but relies on the
   model reliably emitting the op format.
4. **Interaction with summarization** — if the harness ever summarizes or
   truncates context (as `examples/mini_llm_chat.py`'s sliding window already
   does), does the state store need to survive that independently, or is
   surviving it the entire point being tested?

---

## 5. Suggested v0.1 scope (minimal, testable)

Reuse the rate-limiting/backoff pilot rule already drafted and compiled in
`EPISTEMIC_GATE_SPEC.md` (`filter-rule (retries max-retries elapsed-ms
cooldown-ms)`), but wire it as state-tracked instead of context-tracked:

- `retries` and `elapsed_ms` live in the store, never in the prompt.
- Every retry attempt is a write through the compiled rule.
- Test: truncate/replace the agent's context entirely (simulate "50k tokens
  later, this is not even in the window anymore") and confirm the retry limit
  is *still* enforced correctly, because enforcement never depended on the
  model recalling anything. This is the concrete, demoable proof that state
  tracking survives what context recall does not — the same "live evidence,
  not a mock" standard as `examples/contract_gate_demo.py`.

**Implemented 2026-09-17.** `netelpro/state_gate.py`'s `RetryLimiter` is the
store (in-memory dict for v0.1 — open question 1 resolved as "simplest thing
that's still a real external store"; DuckDB/sqlite are a later swap, not a
redesign, since nothing outside this class knows the backend). The rule is
`examples/gates/rate_limit.sl`, gated through `netelpro.gate.Gate` exactly as
described above. `tests/test_state_gate.py::test_survives_total_context_amnesia`
is the falsifiable proof: a fake agent that always believes it's on attempt 1
(genuinely stateless, no memory of its own) still gets correctly rate-limited
after 3 attempts, because the limiter never asked it.

This also resolves two of EPISTEMIC_GATE_SPEC.md's original holes for this
rule specifically: the ceiling is a hard deny before cooldown (not a soft
warning), and a fully-elapsed cooldown resets the counter rather than
carrying accumulated history — both documented in `state_gate.py`'s
docstring and covered by dedicated tests.

Not yet done: sections 2-4's general node/edge/hole store and the harness
wiring (open question 3) are still just this spec's prose — only the single
rate-limiting resource type is real. Generalizing beyond "retries/cooldown"
to arbitrary state keys is the next step, not part of this pilot.

---

## Appendix: other fronts discussed, not designed yet (backlog)

Noted from the same conversation, in case any of these gets picked up next.
None of these has a design pass yet — they're one-line pointers, not specs.

1. **Function-calling with a real schema contract.** Today's
   `NetelproLogitsProcessor` gates a numeric `[allowed_min, allowed_max]`
   range. Extending it to gate structured function calls (declared args, arg
   value ranges, required auth/grant) against a compiled `.sl` contract per
   tool is the direct generalization of the corporate "stop the agent from
   calling `delete_prod_db` without a grant" story — broader than the current
   `action_boundary.sl` demo.
2. **Multi-step plan verification.** An agent plans N steps ahead and
   hallucinates intermediate state ("after step 3, X will be true"). A
   compiled contract could check each step's precondition against real
   observed state (tool results) before allowing the next step to execute —
   this is a specific application of the state-tracking gate above (§5's
   pilot generalizes to arbitrary plan steps, not just retry counters).
3. **RAG/retrieval citation grounding.** `HonestyGuard.count_citations()`
   today counts citation-shaped text via regex; it does not verify a cited
   claim actually appears in the retrieved context. A rule that checks each
   claim maps to a real span in the retrieved documents would close actual
   citation hallucination, not just detect the shape of a citation.
4. **Numerical/financial invariant checking.** The costliest domain to
   hallucinate in — "balance is now $X", "transaction succeeded". A compiled
   arithmetic contract checking the claimed delta against the real ledger
   delta, fail-closed, is a narrower, higher-stakes instance of the
   write-path check this spec already describes for state in general.
