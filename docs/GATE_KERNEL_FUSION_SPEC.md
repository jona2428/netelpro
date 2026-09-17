# Token-Gate Kernel Fusion — Specification v0.1 (DRAFT)

**Status:** DRAFT — spec approved by Jona in conversation ("seguí con netelpro,
armemos el kernel Triton/CUDA pa fusionar el gate en el forward pass,
corriendo en Kaggle", 2026-09-17). Scope confirmed same session: contiguous
range first (not the discrete `token_to_action_map` path), on
`NetelproTransformer` (Teo v2) directly, not an external HF/llama.cpp model.
**Origin:** follow-up to `5286c73` (`llama_cpp_processor` vectorization,
370ms/token → microseconds) and the standing question from an earlier
session — "¿esto se podrá incorporar dentro de la LLM, fusionado con
torch?" — answered here with a concrete design.
**House precedent:** same spec-first protocol as `EPISTEMIC_GATE_SPEC.md`,
`STATE_TRACKING_GATE_SPEC.md`, `INFERENCE_REPAIR_LOOP_SPEC.md`.
**Hardware constraint (binding):** Triton has no native Windows build
(`ROADMAP_TEO_7B_MOE_SDS.md` §4, already documented as a house rule). The
kernel module in this spec is therefore written with a guarded import
(`HAS_TRITON`, mirrors the existing `HAS_TORCH` pattern in `ste.py`) and is
never exercised on the Windows dev machine — it compiles, runs, and is
benchmarked only on Kaggle (Linux, real GPU: T4 or P100).

---

## 1. What this is

Today the token gate and the model's own forward pass are two separate
things that have never been wired together:

- `NetelproTransformer.forward()` ([`netelpro/neuro/transformer.py:226`](../netelpro/neuro/transformer.py)) computes
  `logits = self.lm_head(x)` — a plain `nn.Linear` matmul over the full
  vocabulary — and returns it ungated. `generate()` samples directly from
  these logits. No gate involved.
- `NetelproStreamProcessor` / `NetelproVectorKernel.filter_logits_tensor`
  ([`netelpro/neuro/native_kernel.py:56`](../netelpro/neuro/native_kernel.py)) gates logits, but only as a
  **second, separate step** bolted onto *external* inference engines
  (HuggingFace `logits_processor=[...]`, llama.cpp's callback). It is never
  called from `NetelproTransformer.generate()` itself.

Even where the gate *is* wired in (HF path), it is a second kernel launch
over the full logits tensor: `lm_head` writes `(vocab_size,)` floats to
memory, then a second op (`masked[..., :lo] = mask_value`, `masked[...,
hi+1:] = mask_value`) reads part of that tensor back and rewrites it. For
the default contiguous-range rule (`action_boundary.sl`, reproduced below)
this second pass is pure overhead — the values it writes are a static
`-inf` over a range known before the matmul even runs.

```netelpro
; netelpro/neuro/rules/action_boundary.sl
(defn filter-rule (action_id allowed_min allowed_max safety_state)
  (and (== safety_state 1)
       (and (>= action_id allowed_min)
            (<= action_id allowed_max))))
```

This spec fuses the two: one Triton kernel that computes
`logits = x @ W^T` (the `lm_head` matmul) and writes `-inf` directly into
the output tile for any column outside `[allowed_min, allowed_max]` — in
the same kernel, same memory pass, instead of matmul-then-mask as two
launches. `safety_state == 0` (full freeze) short-circuits to writing
`-inf` everywhere without reading `W` at all, same short-circuit
`filter_logits_tensor` already takes.

---

## 2. Core idea

```
BASELINE (today, HF path only -- Teo v2's own generate() has no gate at all):
  x @ W^T -> logits  [kernel 1, full vocab_size write]
  mask logits in [0,lo) and (hi,vocab_size)  [kernel 2, partial read+write]

FUSED (this spec):
  for each output tile of logits:
      if tile fully outside [allowed_min, allowed_max]:
          write -inf, skip the matmul for this tile entirely
      elif tile fully inside [allowed_min, allowed_max]:
          compute x @ W^T for this tile normally
      else (tile straddles a boundary):
          compute x @ W^T for this tile, then mask the out-of-range columns
          within the tile before writing
  -> one kernel, one memory pass, and the fully-outside case never even
     does the matmul for masked columns (not just "compute then discard" --
     genuinely skip the FLOPs, since Triton kernels are launched per tile
     and a masked tile's program instance can return before the dot product)
```

`safety_state == 0`: skip straight to "write `-inf` everywhere, skip every
tile's matmul" — the existing full-freeze short-circuit, now also skipping
the FLOPs it previously still paid for (today's `filter_logits_tensor`
still runs the *whole* `lm_head` matmul before `filter_logits_tensor`
throws the result away via `torch.full_like`; this spec's freeze path
never launches the matmul).

---

## 3. Relationship to what already exists

| Piece | Reused as |
|---|---|
| `netelpro.neuro.native_kernel.NetelproVectorKernel.filter_logits_tensor` | The reference implementation this kernel must match bit-for-bit on masked positions (differential test, §5) and the fallback path when Triton/CUDA isn't available (CPU, or Windows dev) |
| `netelpro.gate.Gate` / `action_boundary.sl` | The rule being fused stays the source of truth for semantics — the kernel is a hand-written GPU implementation of *this one rule's shape* (contiguous range + binary freeze flag), not a general compiler backend. A different rule shape is out of scope (§6). |
| `NetelproTransformer.lm_head` | The `nn.Linear` this kernel replaces at inference time — same weight tensor (`self.lm_head.weight`, tied to `self.wte.weight`), same output shape. Training/backprop path is untouched (§4). |
| House pattern from `gate_contract.md` §5.2 (differential verification) | Same discipline applied to the kernel: replay test vectors through both the fused kernel and `filter_logits_tensor`, empty mismatch list required before trusting it |

This is not a new gate mechanism — the rule (`action_boundary.sl`) and its
semantics are unchanged. What's new is *where* the rule's effect gets
computed: inside the matmul kernel instead of after it.

---

## 4. Scope boundaries (what this does NOT do)

1. **Inference only.** The fused kernel replaces `self.lm_head(x)` inside
   `generate()` / an explicit `gated_forward()` path. `forward()`'s
   training path (`logits = self.lm_head(x)` feeding `F.cross_entropy`)
   is untouched — no custom backward pass is written for v0.1. Masked
   positions never need gradients at inference time; at training time the
   gate isn't applied at all today, and this spec doesn't change that.
2. **One rule shape.** Only `action_boundary.sl`'s exact contract
   (`safety_state ∈ {0,1}`, contiguous `[allowed_min, allowed_max]`) gets a
   fused kernel. A custom `.sl` rule with different semantics falls back to
   `filter_logits_tensor` (unfused) automatically — never silently wrong,
   same fail-closed posture as the rest of the gate stack.
3. **The discrete `token_to_action_map` path is explicitly out of scope**
   here (scope decision made 2026-09-17, see open questions). It's a
   different fusion shape (gather, not matmul-epilogue masking) and a
   different, currently-unfixed bottleneck — worth its own spec section
   later, not conflated with this one.
4. **Batch size 1, decode step, single GPU.** No tensor-parallel, no
   multi-GPU, no prefill-time fusion (prefill computes logits for every
   position, not just the last one — masking there is a different tiling
   problem). v0.1 targets the actual hot path: one new token, one gate
   check, per decode step.

---

## 5. v0.1 pilot scope

1. **Kernel module:** `netelpro/neuro/gate_kernel.py`. Triton kernel
   `fused_gated_lm_head_kernel` implementing tiled `x @ W^T` with the
   range-mask epilogue described in §2. Guarded import: `HAS_TRITON`
   (mirrors `HAS_TORCH` in `ste.py`) so importing this module on a machine
   without Triton (the Windows dev box) doesn't break anything that
   imports `netelpro.neuro` — it degrades to "kernel unavailable," not an
   ImportError at package load time.
2. **Reference/fallback path:** a plain-torch function with identical
   signature that does `lm_head(x)` then `filter_logits_tensor` — used (a)
   as the differential-test oracle, (b) as the automatic fallback whenever
   `HAS_TRITON` is `False` or the module is running on CPU.
3. **Wiring:** a `gated_forward()` method on `NetelproTransformer` (does
   not replace `forward()` — additive, so the training path is
   byte-identical to before) that calls the fused kernel when available,
   otherwise the reference path. `generate()` gains an optional
   `gate: NetelproStreamProcessor | None = None` parameter; when set, decode
   steps call `gated_forward()` instead of `forward()` + separate masking.
4. **Correctness gate before any perf claim:** `tests/test_gate_kernel.py`
   runs the differential check (fused output vs. reference output,
   exact match on masked positions — `-inf` where `filter_logits_tensor`
   puts `-inf`, and floating-point-identical elsewhere within matmul
   tolerance) on CPU using the reference path only (no GPU in CI). The
   actual Triton kernel launch is exercised only in the Kaggle notebook
   (§7), gated on `torch.cuda.is_available()`.
5. **Benchmark:** `benchmarks/gate_kernel_fusion_kaggle.ipynb` on Kaggle
   T4/P100. Loads or constructs a `NetelproTransformer` at Teo v2's real
   config (vocab 32768, `d_model` 768, 12 layers — same as
   `docs/ROADMAP_TEO_7B_MOE_SDS.md` §2), runs N decode steps with a
   representative `[allowed_min, allowed_max]` window three ways:
   - baseline: `lm_head(x)` then `filter_logits_tensor` (today's actual
     unfused path, two kernel launches)
   - fused: `gated_forward()` via the Triton kernel (one kernel launch)
   - ungated: `lm_head(x)` alone (lower bound — cost of the matmul with
     zero masking overhead, to see how much of the baseline's cost is the
     gate at all vs. just the matmul)
   Report real `torch.cuda.Event`-timed latency per decode step, averaged
   over enough steps to be stable, honestly — same standard as every other
   benchmark in this repo (real numbers, not promised ones). If the fused
   kernel is not faster, that is a valid, reportable result, same as
   §8 of `INFERENCE_REPAIR_LOOP_SPEC.md`.

No code in this spec itself. Implementation follows immediately, same
session, per the go-ahead already given.

---

## 6. Open questions (holes)

1. **Does tile-skip actually save FLOPs in practice, or does Triton's
   scheduler pay for the skipped tiles anyway?** Depends on tile size vs.
   `[allowed_min, allowed_max]` width — if the allowed range is most of the
   vocab (common case: broad safety envelope, narrow denial), few tiles are
   fully-outside and there's little matmul to skip; the win shrinks to "one
   less kernel launch + no redundant memory pass," which is still real but
   smaller than the pilot's framing suggests. Needs the real Kaggle numbers
   before claiming a specific speedup.
2. **Discrete `token_to_action_map` fusion (deferred by this session's
   scope decision).** The actual documented-as-slow path
   (`logits_processor.py:82`, a Python loop calling `gate.check()` once per
   vocab entry, every decode step) is *not* what this pilot fixes. Worth
   its own follow-up: likely a gather kernel over a precomputed
   per-action allow bitmask (computed once per `set_context()` call, since
   the map is static but which actions are allowed changes per turn) rather
   than a matmul-epilogue fusion — different shape, different spec section.
3. **Training-time fusion.** Not attempted here (§4.1) — if gating ever
   needs to apply during training (e.g., to teach the model the boundary
   is real, not just enforced at decode time), the backward pass through a
   custom Triton kernel is new work, not a small extension of this one.
4. **Prefill-time masking.** Also not attempted (§4.4) — today nothing
   masks prefill logits either (only the last position matters for
   sampling), so this isn't a regression, but a future "mask the whole
   sequence's logits" use case (e.g., scoring) would need a different
   tiling scheme.

---

## 7. Kaggle notebook plan

`benchmarks/gate_kernel_fusion_kaggle.ipynb`, structure mirrors the
existing Kaggle notebooks in this repo (`training/train_teo_v2_t4_kaggle.ipynb`
style: one install cell, one hardware-diagnostic cell, then the actual
pilot):

1. Install/verify: confirm `torch.cuda.is_available()` and `triton`
   importable (Kaggle's PyTorch image ships Triton on Linux by default;
   diagnostic cell prints versions and fails loudly, not silently, if
   either is missing).
2. Build a `NetelproTransformer` at Teo v2 config on GPU (random init is
   fine — this benchmarks kernel latency, not model quality; no checkpoint
   download required).
3. Differential correctness check on real GPU tensors (not just the CPU
   reference test from §5.4) — belt and suspenders before trusting any
   timing number that follows.
4. The three-way timed comparison from §5.5, reported as a table:
   attempt/config, mean/min/max latency, honestly, whatever the numbers
   turn out to be.

## 8. Results

*(empty — filled in after the Kaggle run. Not claimed before it's measured,
same discipline as every other spec in this repo.)*
