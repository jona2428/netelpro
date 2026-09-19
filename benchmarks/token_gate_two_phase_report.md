# Two-Phase Token Gate — Reason Free, Gate Only the Final Answer

**Date**: 2026-09-17
**Model**: `qwen2.5-1.5b-instruct.Q4_K_M.gguf` (local, base)
**Script**: [`examples/token_gate_two_phase_demo.py`](../examples/token_gate_two_phase_demo.py)

## Why this run

[`token_gate_numeric_output_report.md`](token_gate_numeric_output_report.md) found that gating every token from the first one (digits-only, no reasoning room) got the format right but the arithmetic wrong (47+89 → -82). Hypothesis here: let the model reason freely in natural language first, switch the gate on only for the final answer span (two separate `llm()` calls — llama.cpp has no mid-generation logits_processor swap), and check whether that recovers correctness while keeping the structural format guarantee.

## A real bug found in this script's own evaluation, before trusting any result

The first version of this script checked correctness with `str(ground_truth) in answer_text` — a substring check. On the `9+6` trial, the gated answer was `"-15.9999"`, and `"15" in "-15.9999"` is `True` in Python even though **-15.9999 is not 15** — the script reported a wrong answer as correct. Caught by reading the raw output, not by assuming the check was right. Fixed: `extract_number()` parses the first signed number from the text with a regex and compares it numerically (`abs(parsed - ground_truth) < 0.5`), not as a string. All numbers in this report use the fixed check, and the original substring bug is left in the script's own docstring as a note, not silently corrected away.

## Results — 4 trials

| a + b | ground truth | Phase 1 reasoning (truncated) | Phase 2 answer | numeric format | correct |
|---|---|---|---|---|---|
| 47+89 | 136 | "Comencemos: 47 más 89 es igual a..." | `136.0000` | ✓ | ✓ |
| 23+58 | 81 | "...23 + 58 = 81." (stated explicitly) | `81. . . . . .` | ✓ | ✓ |
| 104+267 | 371 | "Paso 1: Suma 104 y 200. Paso 2: Suma el resultado con 67." (cut off before stating the sum) | `-311.000` | ✓ | ✗ |
| 9+6 | 15 | "Paso 1: 9 + 6 = 15" (stated explicitly, correctly) | `-15.9999` | ✓ | ✗ |

**Format guarantee: 4/4 (100%), as designed — every answer span is structurally digits-only, no exceptions.**
**Correctness: 2/4 (50%)** — real improvement over the single-phase demo's implicit 0/1, but not a fix, and not claimed as one from a 4-trial sample.

## Honest reading

- **The hypothesis is partially supported, not confirmed.** Letting the model reason first clearly *can* recover correctness (47+89, 23+58) — the format guarantee didn't force a wrong answer the way single-phase gating did. But it didn't reliably fix it: two of four trials were still wrong.
- **The two failure modes are different, and only one is explained.** `104+267`'s reasoning genuinely never finished the arithmetic (cut off mid-decomposition, likely `max_reasoning_tokens=100` or the stop marker firing before the model reached a sum) — phase 2 had nothing correct to draw from, so a wrong answer isn't surprising. `9+6` is the harder case: the reasoning phase **stated the correct answer explicitly** ("9 + 6 = 15"), yet phase 2 still generated `-15.9999` — wrong, and with a spurious sign and trailing digits neither phase 1 nor the question suggested. This means having the right answer already in context doesn't guarantee phase 2 reproduces it once heavy vocabulary masking (only ~148/151,936 tokens allowed) and greedy sampling are combined. Not explained here — a real, open question, not swept under a "needs more reasoning" story that the `9+6` case already contradicts.
- **n=4 is not a measured rate.** Same caveat as every other live-generation result this session: enough to show the two-phase approach is a real, partial improvement and to surface a genuine unexplained failure mode, not enough to claim "50% correct" as a stable number. A larger sweep, and specifically investigating the `9+6`-shaped failure (right answer in context, still generates wrong under the gate), would be the next real step — not attempted here.

## Cross-references

- Single-phase demo (format right, arithmetic wrong): [`benchmarks/token_gate_numeric_output_report.md`](token_gate_numeric_output_report.md)
- Discrete-map fast path both demos exercise at real vocab scale: [`docs/GATE_KERNEL_FUSION_SPEC.md`](../docs/GATE_KERNEL_FUSION_SPEC.md) §11

## Follow-up (2026-09-17, later the same day) — the `9+6` failure mode is now explained

The "Honest reading" section above left the `9+6` case as an open question: the reasoning phase stated `15` explicitly, yet phase 2 emitted `-15.9999` under the gate. That is now diagnosed, and the cause was in this demo's own token map, not in the model.

**Root cause.** `build_numeric_token_action_map()` classified a token as numeric if its decoded text consisted only of characters from `"0123456789.-"`. That test admits tokens with **no digit at all** — a run of hyphens (`' -----------'`), a run of dots (`'...............'`), `'..\n\n\n\n'`. Counted on the real vocabulary (`qwen2.5-1.5b-instruct.Q4_K_M`):

| Rule | Tokens allowed | Of which carry no digit |
|---|---|---|
| Characters only (original) | 148 (147 + EOS) | **137** |
| Requires ≥1 digit (fixed) | 11 (10 + EOS) | 0 |

The model was therefore free to open its answer with a bare `-`, which is a legal token under the old map. That is the spurious sign — no model mystery required.

**Confirmed by intervention on the real gate, not by reimplementation.** Re-running this demo with the digit requirement added, same model, same question, same seed path:

- before: `'-15.9999'`
- after: `'15961515'`

The sign is gone. The answer is still wrong, and that distinction matters: **the format guarantee and arithmetic correctness are orthogonal.** Masking 151,925 of 151,936 tokens leaves the model with 11 candidates, and a 1.5B model picking greedily among 11 digit fragments will happily emit `15961515`. The old map's 148 candidates at least contained more of the right pieces. Fixing the map removes the artefact; it does not make the arithmetic work, and this report should not be read as claiming it does.

**What this says about the demo's fitness as evidence.** The arithmetic framing conflates two questions — *did the gate hold?* and *is the number right?* — so a fully working gate reads as a failure. For demonstrating what the token gate actually guarantees, a boundary case is the honest showcase: the model is about to emit a forbidden token and that logit goes to `-inf` before sampling. No correct-or-incorrect answer to muddy the result. The single-phase demo shows the same effect from the same cause: `47+89` previously produced `-82`, and after the fix it produces `'8600000000000000'` — numeric format held, value meaningless.

**Not measured here:** whether the digit rule changes outcomes across a sweep (n=1 per configuration; this identifies mechanism, not rate), and whether a larger model narrows the 11-candidate gap. Both are open.
