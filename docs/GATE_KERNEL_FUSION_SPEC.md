# Token-Gate Kernel Fusion — Specification v0.1

**Status:** CLOSED, positive, same day (2026-09-17). v0.1 (§8): naive
kernel, honest negative (7x slower than baseline). v0.2 (§9): `tl.dot`
rewrite, honest positive but boundary-dependent (3.1x faster narrow,
1.27x slower broad). v0.3 (§10): range-width dispatch wired into
`gated_lm_head` (what `generate(gate=...)` actually calls) and validated
on real GPU — **no regression in either tested regime, 3.0x win where
there's real work to skip.** The token gate is genuinely fused into
`NetelproTransformer`'s forward pass now, not just specced. §11 (same
day): the other bottleneck §6 deferred — the discrete
`token_to_action_map` path — fixed too, no GPU needed, measured
**~35-37x on real workloads, honest tie (not a regression) in the
adversarial case.**
Scope confirmed same session: contiguous
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

## 8. Results (2026-09-17, Kaggle T4/P100, real run)

Differential correctness check (§7 step 3) passed on real CUDA tensors
before any timing was trusted. Full three-way benchmark, `torch.cuda.Event`
timing, 200 iterations after 20 warmup, `NetelproTransformer` at Teo v2
config (vocab 32768, 12 layers, `d_model` 768):

| range | method | mean_ms | min_ms | p50_ms |
|---|---|---|---|---|
| broad `[100, 30000]` (mostly allowed) | ungated | 0.4325 | 0.4219 | 0.4282 |
| broad `[100, 30000]` | baseline_unfused | 0.4529 | 0.4430 | 0.4502 |
| broad `[100, 30000]` | **fused_triton** | **3.1362** | 3.0724 | 3.1315 |
| narrow `[5000, 5200]` (mostly masked) | ungated | 0.4221 | 0.4140 | 0.4202 |
| narrow `[5000, 5200]` | baseline_unfused | 0.4433 | 0.4331 | 0.4397 |
| narrow `[5000, 5200]` | **fused_triton** | **0.6470** | 0.6288 | 0.6430 |

**Honest reading: the fused kernel lost in both configs.** Broad:
~7x slower than baseline (3.14ms vs. 0.45ms). Narrow: still slower
(0.65ms vs. 0.44ms), though the gap shrank a lot from the broad case —
tile-skip visibly did something (open question 1, §6, is answered: yes,
skip amount matters, exactly as predicted), it just wasn't enough to close
the gap, let alone open a lead.

**Root cause, diagnosed from the shape of the numbers, not guessed:**
`ungated` ≈ `baseline_unfused` (0.43ms vs. 0.45ms) — today's unfused path
is *already* close to free relative to the bare matmul. That means the
premise in §1 ("the second masking pass is pure overhead worth fusing
away") was true in kind but small in magnitude: for a batch-1 decode-step
GEMV, `nn.Linear` dispatches to a cuBLAS GEMV kernel that's already close
to the T4's memory-bandwidth floor for reading the ~100MB `lm_head.weight`
matrix once — there was never much daylight for a fused kernel to win by
just removing one cheap elementwise pass. The v0.1 kernel's actual loss
is bigger than that gap, though, which points at a second, separate
problem: `_fused_gated_lm_head_kernel`'s inner loop
(`tl.sum(w_chunk * x_chunk[None, :], axis=1)`, a manual broadcast-multiply-
then-reduce) is a naive GEMV implementation that doesn't use `tl.dot`
(Triton's tensor-core-mapped matmul primitive) — it's a slower kernel than
cuBLAS's tuned GEMV on its own merits, independent of masking. That's
consistent with 7x slower in the broad config (almost no tiles skipped,
so it's nearly a pure "naive Triton dot vs. cuBLAS" comparison) and with
narrow still losing despite skipping ~31/32 tiles (skip removes most of
the *work*, but per-launch/per-program overhead plus the remaining tile's
inefficient reduction still costs more than the entire baseline path).

**This does not fix or contradict the spec's premise (§1) about the
masking pass being separable overhead — it shows that overhead was smaller
than assumed, and that a hand-rolled reduction is the wrong tool to spend
effort closing a gap that small.** Naive fusion doesn't win. If this is
worth pursuing further, the next experiment is not "tune the tile sizes"
but a structurally different kernel body — `tl.dot`-based tiling (real
tensor-core matmul, not manual multiply-reduce) — which is new kernel-
design work, not an incremental fix to this one. Not attempted in v0.1;
left as a candidate for a future session, same as the untried moves at the
end of `INFERENCE_REPAIR_LOOP_SPEC.md` §8.

**v0.1 verdict: negative result, honestly measured.** The idea ("fuse the
gate into the forward pass") is not disproven — the *implementation*
(naive Triton GEMV) loses to what already exists (`nn.Linear` + tensor
slicing). Same standard as the rest of this repo: a real number that says
"no" is worth more than an assumed "yes."

---

## 9. v0.2 — `tl.dot` rewrite (design, 2026-09-17 same day)

Following §8's diagnosis directly: the v0.1 kernel's inner loop
(`tl.sum(w_chunk * x_chunk[None, :], axis=1)`) never touches tensor cores.
v0.2 replaces it with `tl.dot`, Triton's tensor-core-mapped matmul
primitive, in `_fused_gated_lm_head_kernel_dot`
(`netelpro/neuro/gate_kernel.py`). The v0.1 kernel is kept, renamed
`_fused_gated_lm_head_kernel_naive` / `gated_lm_head_naive`, specifically
so the Kaggle notebook can benchmark both in the same run — the comparison
needs to isolate the kernel-design change, not rely on comparing today's
numbers against §8's numbers from a different session/run.

**The M=1 problem and its standard fix.** A decode step is exactly one
token — `x` is a single `(n_embd,)` row, so the GEMV is M=1. Tensor-core
`mma` instructions require M ≥ 16; there is no way to hand a 1-row operand
to `tl.dot` and get tensor-core execution. The fix used here (a known
pattern, not invented for this repo): pad the left operand to
`BLOCK_M=16` with 15 all-zero rows, run a real 16×`BLOCK_K`×`BLOCK_N`
tensor-core matmul, then discard rows 1–15 of the result
(`tl.sum(tl.where(row_idx[:, None] == 0, acc, 0.0), axis=0)`). This
computes 16x the FLOPs a true GEMV needs — but §8 already established the
kernel is memory-bandwidth-bound on reading `weight`, not compute-bound,
so the extra FLOPs are expected to be close to free relative to that read.
This is the falsifiable part of v0.2: if it's *not* close to free, that's
itself informative (it would mean the memory-bound diagnosis in §8 was
incomplete).

**Precision.** `tl.dot`'s tensor-core path needs fp16 (T4/Turing, sm_75)
or tf32 (Ampere+, sm_80+ — not available on T4 or P100). This spec targets
fp16 inputs with an explicit fp32 accumulator (`acc = tl.dot(xb, wt, acc)`)
— the summation itself stays fp32-precise; only the per-element fp16
rounding of `x` and `weight` before each multiply introduces error. The
notebook's correctness check (§7) accordingly uses a looser tolerance for
v2 than v1's exact fp32 comparison (`max_abs_diff < 0.5`,
`max_rel_diff < 0.05`, checked explicitly and printed, not assumed) —
masked positions still must be exactly `-inf`, only the unmasked logit
values get the fp16 tolerance.

**Hardware caveat, stated up front rather than discovered after the
fact:** P100 (Pascal, sm_60) has no tensor cores at all. On a P100,
`tl.dot` cannot get a tensor-core speedup over v1 — any win there would
have to come from better Triton-generated codegen for the same CUDA-core
path, which is a real possibility but a different, weaker claim than "used
tensor cores." The notebook's hardware-check cell (§7 step 1) prints the
actual device name so results are never misattributed to the wrong GPU.

**Open question for this section:** does the 16x FLOP inflation from
M-padding actually stay free, or does it show up in the numbers? Not
answered here — v0.1's mistake was claiming a mechanism worked before
measuring it; v0.2 doesn't repeat that. Results land in the next Kaggle
run, appended below, not assumed.

**Correction found on the first real run (2026-09-17): the correctness
check's tolerance formula was wrong, not the kernel.** First Kaggle run
of the correctness cell reported `dot_match=False` with `max_abs_diff`
0.0004–0.0007 but `max_rel_diff` up to 0.3998 (40%). The absolute error
matches the predicted fp16-rounding magnitude almost exactly (§9's own
estimate: term magnitude ≈ `0.02 × 1`, summed over 768 terms with
independent fp16 rounding ≈ `sqrt(768) × 0.02 × 0.001 ≈ 0.0004`) — the
kernel's actual numerics were fine. The bug was `rel_diff = abs_diff /
ref.abs().clamp_min(1e-3)`: dividing a tiny absolute error by a
reference value that happens to be near zero (e.g. `ref ≈ 0.0018`) turns
a 0.0007 rounding error into a nonsense "40% relative error" — a standard
relative-error trap, not a masking or indexing bug in
`_fused_gated_lm_head_kernel_dot`. Fixed by switching the notebook's
check to the combined `torch.allclose` criterion
(`abs_diff <= atol + rtol·|ref|`, `atol=1e-2, rtol=0.05`), which doesn't
blow up near zero. Recorded here because it's a real, reportable mistake
in this pilot's own test harness, not swept past silently — same
standard the spec holds the kernel to.

### Results (2026-09-17, Kaggle T4/P100, real run, post-tolerance-fix)

Differential correctness passed (both kernels, combined `atol+rtol`
criterion for v2). Four-way benchmark, same methodology as §8:

| range | method | mean_ms | min_ms | p50_ms |
|---|---|---|---|---|
| broad `[100, 30000]` | ungated | 0.4292 | 0.4206 | 0.4260 |
| broad `[100, 30000]` | baseline_unfused | 0.4496 | 0.4421 | 0.4485 |
| broad `[100, 30000]` | fused_triton_naive (v1) | 3.1347 | 3.0638 | 3.1315 |
| broad `[100, 30000]` | **fused_triton_dot (v2)** | **0.5724** | 0.5373 | 0.5726 |
| narrow `[5000, 5200]` | ungated | 0.4217 | 0.4156 | 0.4204 |
| narrow `[5000, 5200]` | baseline_unfused | 0.4412 | 0.4321 | 0.4390 |
| narrow `[5000, 5200]` | fused_triton_naive (v1) | 0.6706 | 0.6451 | 0.6672 |
| narrow `[5000, 5200]` | **fused_triton_dot (v2)** | **0.1425** | 0.1381 | 0.1404 |

**Honest reading: v2 beats v1 decisively (confirms §9's tensor-core
diagnosis), and v2 beats the real baseline in exactly the regime §6 open
question 1 predicted it would.**

- **v2 vs. v1:** ~5.5x faster broad (3.13ms → 0.57ms), ~4.7x faster narrow
  (0.67ms → 0.14ms). The `tl.dot` rewrite is unambiguously the right
  design over the manual reduce — this is now established, not
  hypothesized.
- **v2 vs. baseline, narrow (mostly masked):** **fused wins, 3.1x faster**
  (0.1425ms vs. 0.4412ms) — and beats even `ungated` (0.42ms), the bare
  unmasked matmul, because skipping ~31/32 tiles means the kernel reads a
  small fraction of `lm_head.weight` instead of the whole ~100MB matrix,
  where `ungated` still has to read all of it. This is the fusion premise
  from §1 working as designed, in the regime where there's real work to
  skip.
- **v2 vs. baseline, broad (mostly allowed):** fused still loses, ~1.27x
  slower (0.5724ms vs. 0.4496ms) — better than v1's 7x loss, but not a
  win. Consistent with §8's diagnosis: cuBLAS's GEMV is already close to
  the memory-bandwidth floor when there's little to skip, and this
  kernel's fixed overhead (16x FLOP inflation from M-padding, 128 program
  launches for `BLOCK_N=256` over a 32768 vocab) has nothing to amortize
  against when almost no tile gets to skip its matmul.

**v0.2 verdict: real, positive, and boundary-dependent.** The fused
kernel is the better choice exactly when the gate's allowed range is
narrow relative to the vocabulary — the safety-critical case (tight
action boundaries) rather than the permissive one (broad, mostly-open
generation). It is not a strict improvement over today's code in every
configuration, and this spec does not claim it is.

---

## 10. v0.3 — range-width dispatch, wired and validated (2026-09-17, same day)

§9's mixed result forced a real choice: dispatch by range width, or
accept the broad-case regression. Chosen: dispatch. `gated_lm_head` in
`netelpro/neuro/gate_kernel.py` now computes `_touched_tile_fraction`
(how much of `lm_head.weight` the v2 kernel would have to read for a
given `[allowed_min, allowed_max]`) and only launches the kernel when
that's below `DOT_KERNEL_TOUCHED_TILE_THRESHOLD` (0.5, a linear
interpolation between §9's two measured points — explicitly not a swept
number). Above the threshold, it calls `gated_lm_head_reference` instead
— never a regression by construction, not by assumption. The raw v2
kernel, unconditional, moved to `gated_lm_head_dot` (what the benchmark
notebook times directly, so the kernel-vs-baseline comparison stays
honest and isn't laundered through the dispatcher).

`NetelproTransformer.gated_forward()` / `generate(gate=...)` already
called `gated_lm_head` — no wiring changes needed. The dispatch is live
in the actual generation path as of this commit.

### Results (2026-09-17, Kaggle T4, real run, correctly labeled)

Five-way benchmark: `ungated`, `baseline_unfused` (today's real path pre-
this-spec), `fused_triton_naive` (v1), `fused_triton_dot` (v2 kernel,
unconditional — what §9 measured), `fused_dispatch` (`gated_lm_head`,
what `generate()` actually calls now):

| range | method | mean_ms | min_ms | p50_ms |
|---|---|---|---|---|
| broad `[100, 30000]` | ungated | 0.4321 | 0.4203 | 0.4261 |
| broad `[100, 30000]` | baseline_unfused | 0.4440 | 0.4352 | 0.4418 |
| broad `[100, 30000]` | fused_triton_naive (v1) | 3.1264 | 3.0420 | 3.1212 |
| broad `[100, 30000]` | fused_triton_dot (v2, raw) | 0.6060 | 0.5566 | 0.6048 |
| broad `[100, 30000]` | **fused_dispatch (v3)** | **0.4455** | 0.4345 | 0.4425 |
| narrow `[5000, 5200]` | ungated | 0.4211 | 0.4137 | 0.4192 |
| narrow `[5000, 5200]` | baseline_unfused | 0.4392 | 0.4315 | 0.4366 |
| narrow `[5000, 5200]` | fused_triton_naive (v1) | 0.6856 | 0.6561 | 0.6756 |
| narrow `[5000, 5200]` | fused_triton_dot (v2, raw) | 0.1441 | 0.1377 | 0.1417 |
| narrow `[5000, 5200]` | **fused_dispatch (v3)** | **0.1468** | 0.1398 | 0.1446 |

**Confirmed exactly as designed, no surprises:**

- **Broad:** `fused_dispatch` (0.4455ms) ≈ `baseline_unfused` (0.4440ms)
  — the dispatcher correctly avoided the raw kernel's 0.606ms and landed
  within 0.3% of the unfused baseline. **No regression**, where an
  un-dispatched fused-everywhere design would have cost ~36% extra on
  every broad-range decode step.
- **Narrow:** `fused_dispatch` (0.1468ms) ≈ `fused_triton_dot`
  (0.1441ms) — the dispatcher correctly picked the kernel, losing only
  ~0.003ms (~2%) to the `_touched_tile_fraction` check itself. **3.0x
  faster than baseline** (0.4392ms → 0.1468ms), the real win carried all
  the way through to what `generate()` actually calls.

**v0.3 / pilot verdict: closed, positive.** The token gate is now
genuinely fused into `NetelproTransformer`'s forward pass, wired through
`generate(gate=...)`, real measured GPU numbers on both sides of the
dispatch boundary, and it is a strict improvement over the pre-this-spec
baseline in both tested regimes — no configuration measured here makes
generation slower than it already was. What's still open, honestly: only
two range widths were tested; the 0.5 threshold is an interpolation, not
a sweep, so a range near that boundary hasn't been measured directly and
could in principle land on the wrong side of the dispatch. A future
session could sweep more widths to tighten the threshold, or measure on
a P100 to check the no-tensor-cores caveat from §9 — neither blocks using
what's here today.

---

## 11. Discrete `token_to_action_map` fast path (2026-09-17, same day)

§6 open question 2 deferred this on purpose: the contiguous-range work
above (§1–§10) is a GPU-kernel problem. This one isn't. It's an
algorithmic fix, runs and is measured entirely on CPU, no Triton, no
Kaggle needed.

**The bottleneck.** `NetelproLogitsProcessor.__call__`'s discrete branch
and `NetelproStreamProcessor.llama_cpp_processor`'s discrete branch (used
whenever a `token_to_action_map` is supplied — the arbitrary token→action
correspondence path, not the default contiguous-range rule) both looped
over every vocab token, calling the compiled gate once per token:

```python
for token_id in range(vocab_size):
    action_id = token_to_action_map.get(token_id, token_id)
    allow, _ = self.gate.check(action_id, allowed_min, allowed_max, safety_state)
    if not allow:
        scores[token_id] = mask_value
```

This is the same *class* of bug the contiguous-range fix (`5286c73`)
already killed for the default rule — a Python loop of native ctypes
calls, once per decoding step, over a 32k–152k vocab — just never fixed
for this branch, because a discrete map doesn't have an obvious tensor-
slicing shortcut the way a contiguous range does.

**The fix: evaluate the gate once per unique action, not once per token.**
`token_to_action_map` is set once at construction and never mutated
after (no setter exists). So its `(unique actions, token→unique-action
index)` decomposition can be computed once, cached, and reused across
every decode step — only the per-unique-action gate evaluation needs to
re-run when `allowed_min`/`allowed_max`/`safety_state` change via
`set_context()`. Implemented in
`NetelproLogitsProcessor._action_map_decomposition()` (pure Python dict
grouping, O(vocab_size) once) and `._unique_action_allowed()` (calls
`self.gate.check()` — the same fail-closed wrapper the old per-token loop
used, deliberately not `NetelproVectorKernel.evaluate_batch`'s raw native
call, which has no exception handling and would have silently weakened
the fail-closed contract `gate_contract.md` §3.2 documents). Both the HF
`__call__` path (torch gather) and the llama.cpp `llama_cpp_processor`
path (numpy gather when available, a plain-Python index lookup — no
native calls, still a real win — when it isn't) share this one cache, via
`self.processor`.

**Correctness:** `tests/test_neuro_streaming.py` — six new tests, all new
coverage (nothing exercised this branch before this fix, at all). Brute-
force comparison against the actual compiled gate (`sp.gate.check()` per
token, same pattern `test_llama_cpp_processor_boundary_consistent_with_native_gate`
already used for the contiguous path), the HF and llama.cpp adapters
checked to agree with each other (they share the same cache), the no-numpy
fallback exercised via monkeypatch, and a `set_context()` reactivity test
confirming the cached decomposition doesn't go stale when the allowed
range changes.

**Results (2026-09-17, local CPU run, `benchmarks/discrete_action_map_bench.py`):**

| scenario | old (ms/step) | new (ms/step) | speedup |
|---|---|---|---|
| vocab 32,768, 64 unique actions (tokens collapse 512:1) | 79.4 | 2.13 | **37.3x** |
| vocab 152,000, 200 unique actions (this file's original motivating scale) | 371.5 | 10.41 | **35.7x** |
| worst case: vocab 32,768, ~all-identity map (unique ≈ vocab_size) | 79.4 | 78.6 | **~1.0x (honest tie, not a win)** |

**Honest reading.** When a `token_to_action_map` actually collapses many
tokens onto a smaller action space — the reason such a map exists at all
(constraining a large LLM vocabulary down to a small legal action surface
for an agent/system, per this file's own module docstring) — this is a
real, large, unconditional win: ~35-37x measured, no dispatch heuristic
needed (unlike §10's kernel work), no downside. In the pathological case
where the map barely collapses anything (each token maps to its own
distinct action, explicitly or via the identity fallback), the fix is a
tie, not a regression: the per-step cost is still bounded by
`num_unique_actions` native calls, which in that case is ~vocab_size,
same as before. The worst case was measured, not assumed, and the
benchmark script's own comment originally overclaimed a win there before
this run corrected it — left in as a small, real example of the same
discipline this whole spec has tried to hold to: check before you claim.
