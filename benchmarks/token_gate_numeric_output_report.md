# Token Gate — Semantic Constraint on Real Qwen2.5 Output

**Date**: 2026-09-17
**Model**: `qwen2.5-1.5b-instruct.Q4_K_M.gguf` (local, base)
**Script**: [`examples/token_gate_numeric_output_demo.py`](../examples/token_gate_numeric_output_demo.py)

## What this demonstrates

Extends `contract_gate_llama_cpp_demo.py` (which gates arbitrary token-ID ranges — a mechanism demo, not a use case) to a real semantic constraint: force Qwen2.5 to emit **only digits**, structurally, by building a real `token_to_action_map` (action=1 for any vocab token whose decoded text is pure digits/`.`/`-`, action=0 otherwise, EOS force-allowed so generation can still stop). This is also the best possible case for the discrete-map fast path fixed the same session (`docs/GATE_KERNEL_FUSION_SPEC.md` §11): **2 unique actions across a 151,936-token vocabulary**.

## Real results

- `token_to_action_map` built in **0.58s** (one-time, 151,936 `detokenize()` calls) — 148/151,936 tokens are numeric.
- **Gate overhead: 7,108 µs/token** (~7.1ms) on the real vocab, 2-unique-action case — this is the fixed discrete-map path doing O(2) native gate calls per step instead of O(151,936); the old per-token-native-call loop this replaced would have cost on the order of seconds per token at this vocab size (see `benchmarks/discrete_action_map_bench.py`'s 152,000-vocab measurement: 371ms/step at 200 unique actions — this case has 100x fewer unique actions and correspondingly lower overhead).
- **Output format: 100% numeric, verified** (`" -82 -82.000000000"`) — the gate did exactly what it was built to do. Not one non-digit token in the output, by construction — every non-numeric logit was `-inf` before every single sampling step, not filtered after the fact.
- **Output content: wrong.** Prompt: "¿Cuánto es 47 más 89? Respondé solo con el resultado." (47+89=136). Ungated (Phase 1), the model got it right: `" 136"`. Gated (Phase 2), forced to emit only digits from the very first token, the model produced `" -82 -82.000000000"` — incorrect.

## Honest reading

**The mechanism works exactly as designed. The demonstration also surfaced a real cost, not hidden here:** constraining the *format* of every token from the first one removes the model's ability to work through the arithmetic in natural-language tokens before committing to a digit — there's no room for anything resembling "47 + 89 = " scratch reasoning when every token must already be a digit. Structural format guarantees and answer correctness are not the same property, and this run is direct evidence they can trade off against each other, not just a theoretical concern.

This isn't a flaw in the gate — it did exactly what `[allowed_min, allowed_max]` + `token_to_action_map` are specified to do (`gate_contract.md`'s contract has no correctness claim, only a format/boundary one). It's a real design consideration for anyone using this mechanism: digits-only output is only free if the answer doesn't need reasoning tokens to get right. Not attempted here, a natural next step if this matters: a two-phase generation (unconstrained reasoning, then switch to the numeric-only gate only for a final tagged answer span) — genuinely more engineering than this demo, not assumed to work, not built.

## Cross-references

- Mechanism-only demo (arbitrary token-ID ranges): [`examples/contract_gate_llama_cpp_demo.py`](../examples/contract_gate_llama_cpp_demo.py)
- Discrete-map fast path this exercises at real scale: [`docs/GATE_KERNEL_FUSION_SPEC.md`](../docs/GATE_KERNEL_FUSION_SPEC.md) §11
