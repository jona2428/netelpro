# Epistemic Gate — Specification v0.1 (DRAFT)

**Status:** DRAFT — awaiting Jona's approval. No implementation code written.
**Origin:** "La idea de los dos" — Jona's statistical bed + Teo's epistemic planner,
co-designed and ratified 2026-09-12/13. Two consumers of ONE map.
**House precedent:** `MEMORY_JUDGE_SPEC.md` (spec-first, live compiled rule inside,
Jona's approval before any code).

---

## 1. What this is

Netelpro promoted from arbiter to **epistemic director**: it interrogates an LLM,
verifies each answer as a compiled contract, maintains a concept map of what is
proven / unproven / missing, and **selects the next best question** in real time.

It does not train anything. It does not modify the interrogated model. The
change is in the *method of interrogation*: systematic, verified, self-directing.

**The map has two consumers:**

| Consumer | What it reads | What it drives |
|---|---|---|
| **A. Epistemic planner** | Holes (undetermined pairs the meta needs) | Which question to ask next |
| **B. Statistical bed** (Jona's) | Per-area error rates from the arbiter | Data generation / training toward weak areas |

Same edges, same store. Consumer B was Jona's original idea; consumer A is the
extension. They share the foundation.

---

## 2. Core loop

```
PREGUNTAR  -> planner selects the question with max hole-closure per token
ANSWER     -> interrogated LLM (Gemini via antigravity agents, Pro quota)
COMPILAR   -> answer expressed as netelpro contract; compile + differential verify
             (native backend, microseconds, fail-closed)
DEDUCIR    -> a passing rule DEDUCES for free: covered edge cases, consequences
MAPEAR     -> verified contract recorded as an EDGE in entity_graph (DuckDB);
              holes recomputed
ELEGIR     -> next question chosen from the updated map. Loop closes.
```

The key asymmetry vs plain RAG: an edge in this map is not "the model said so".
It is "it compiled and passed differential verification". Belief requires
compilation; everything else is a hole.

---

## 3. Concept model (ratified by Jona, 2026-09-13)

- **Node** = domain concept (e.g. `retry_count`, `cooldown_ms`, `stagnation`).
- **Edge** = verified relation: a compiled, differentially-verified netelpro
  contract connecting two concepts. Carries verification metadata.
- **Hole** = a concept pair the current meta needs but no verified edge connects.
- **Best question** = the one closing the most hole-weight per token spent.

---

## 4. Pilot meta: rate limiting / backoff rule

The only gate-able domain still pending with no candidate. Live draft rule,
**compiled and verified with the real compiler this turn (10/10 differential,
native vs interpreter, `verify` ok, zero mismatches)**:

```netelpro
;; Epistemic gate pilot meta: rate limiting / backoff for netelpro (DRAFT HYPOTHESIS).
;; allow := (retries < max-retries) OR (elapsed-ms >= cooldown-ms)
(defn filter-rule (retries max-retries elapsed-ms cooldown-ms)
  (or (< retries max-retries) (>= elapsed-ms cooldown-ms)))
```

### Holes this meta must close (first interrogation batch)

1. **Boundary semantics** — what happens at `retries == max-retries` AND
   `elapsed-ms < cooldown-ms`? (Draft says deny; is that the desired behavior?)
2. **Cooldown expiry** — when the cooldown passes: does the counter reset, or
   does it carry accumulated history?
3. **Counter ownership** — who owns `retries` and `elapsed-ms`? The caller,
   the rule, or shared state? (Netelpro rules are pure; state lives outside.)
4. **Backoff shape** — fixed cooldown or exponential? Parameter or constant?
   Should the rule expose `backoff-base` as a parameter like `loop_gate.sl`
   exposes its thresholds?
5. **Interaction with the loop gate** — how does retry-counting relate to the
   existing loop detector's repeats/stagnation? Two gates, one behavior, or
   orthogonal domains?
6. **Success reset** — does a success inside the window reset the retry counter?

Each hole above is a candidate edge: `(retries, max-retries)` semantics already
has a verified form; `cooldown-expiry x retries` has none, etc.

---

## 5. The organ that exists (correction of 2026-09-12)

Teo initially called the relational store "the missing cement". **Wrong: it
exists.** `entity_graph` (DuckDB) already stores the project's real history as
typed relations (releases, decisions, repos, skills). What is missing is only
the **wiring**: compiled-and-verified contract → `upsert_relation` with
verification metadata. Contabilidad, no construcción.

**Edge record (proposed):**
- source/target: the two concepts
- relation: `verificada_por` (or domain-specific type)
- context: compile checksum, test-vector count, verification date
- provenance: originating question + answer hash + interrogated model id

---

## 6. Interrogated model

**Gemini via antigravity agents** (Pro quota, ratified by Jona: "los geminosos
me salen gratis mientras estén en la cuota"). One delegation per question
batch. Antigravity's historical failure to grasp this concept ("le pedí eso y
no me entendió ni una wea") is irrelevant here: agents are used as *answer
sources*, never as planners. The planning, the compilation, the map and the
question selection all live in this house. An agent answers; the arbiter
decides if the answer becomes knowledge.

---

## 7. Honesty constraints (non-negotiable)

1. **Fail-closed epistemology.** An answer NEVER becomes an edge until
   compile + differential verify pass. An unverifiable answer stays a hole,
   with the prose stored as candidate context, not as knowledge.
2. **Contradictions mark contested holes.** If two answers conflict, the hole
   is marked contested; neither edge exists until a discriminating contract
   compiles.
3. **Auditable selection.** Every planner choice is logged with its
   hole-weights: why THIS question, what it should close. The question log is
   reviewable post-hoc.
4. **No self-promotion of knowledge.** The map may never claim more than its
   verified edges show. Hole count is public within the loop.
5. **Scope honesty.** Only the formalizable is domino-verifiable. The map
   covers gates, policies, protocols, decisions. Everything else is
   explicitly out of the map's jurisdiction.

---

## 8. Phases (approval gate between each)

- **Phase A — Wiring:** compiled contract → edge in entity_graph with
  verification metadata. Small, testable, no new language surface.
- **Phase B — Hole enumeration:** the pilot meta's hole list (§4) loaded as
  the first live map content.
- **Phase C — Question selection:** weight function + question templates;
  first live interrogation of Gemini against the rate-limiting holes.
- **Phase D — Pilot rule closed:** `data/rl_gate.sl` candidate compiled,
  differentially verified, wired with semantic-preserving fallback
  (exact pattern of `loop_gate.sl`, commit ff54375).
- **Phase E — Statistical bed (consumer B):** benchmark error tables (run #4
  baseline included) feed the same map as per-area weakness data, driving
  targeted dataset generation for the next training run.

## 9. Out of scope (v0.1)

- No training, no fine-tuning, no new netelpro primitives.
- No auto-modification of zone/permission rules (RED zone stays RED).
- No multi-meta planning: one meta (rate limiting) until it closes.

## 10. Success criteria

1. `rl_gate.sl` compiled, differentially verified, wired, with fallback.
2. Every edge traceable to a verified answer (provenance intact).
3. The question log auditable end-to-end.
4. Zero unverifiable edges in the map. Zero claimed-but-unverified knowledge.