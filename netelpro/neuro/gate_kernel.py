"""Fused token-gate kernel: lm_head matmul + action_boundary.sl range mask,
in one GPU kernel instead of matmul-then-mask as two.

See docs/GATE_KERNEL_FUSION_SPEC.md for the full design and
docs/GATE_KERNEL_FUSION_SPEC.md Section 9 for why v1 (a hand-rolled
broadcast-multiply-then-reduce) lost to cuBLAS by ~7x and why v2 below
uses tl.dot (tensor-core matmul) instead. Guarded import, same pattern as
HAS_TORCH in ste.py: Triton has no native Windows build
(docs/ROADMAP_TEO_7B_MOE_SDS.md Section 4), so importing this module on the
Windows dev machine must not raise -- it must degrade to "kernel
unavailable" and let callers fall back to netelpro.neuro.native_kernel's
unfused torch path.
"""

from __future__ import annotations

from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_gated_lm_head_kernel_naive(
        x_ptr, w_ptr, out_ptr,
        n_embd, vocab_size,
        allowed_min, allowed_max,
        mask_value,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """v1: manual broadcast-multiply-then-reduce GEMV. Kept only as the
        losing side of the v1-vs-v2 comparison in
        benchmarks/gate_kernel_fusion_kaggle.ipynb -- superseded by
        _fused_gated_lm_head_kernel_dot below (GATE_KERNEL_FUSION_SPEC.md
        Section 8: measured ~7x slower than the unfused cuBLAS baseline on
        a T4, because this inner loop never touches tensor cores).
        """
        pid = tl.program_id(axis=0)
        col_start = pid * BLOCK_N
        cols = col_start + tl.arange(0, BLOCK_N)
        col_mask = cols < vocab_size

        tile_max = col_start + BLOCK_N - 1
        tile_fully_outside = (tile_max < allowed_min) | (col_start > allowed_max)

        if tile_fully_outside:
            tl.store(out_ptr + cols, mask_value, mask=col_mask)
            return

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in range(0, n_embd, BLOCK_K):
            k_idx = k + tl.arange(0, BLOCK_K)
            k_mask = k_idx < n_embd
            x_chunk = tl.load(x_ptr + k_idx, mask=k_mask, other=0.0)
            w_ptrs = w_ptr + cols[:, None] * n_embd + k_idx[None, :]
            w_mask = col_mask[:, None] & k_mask[None, :]
            w_chunk = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.sum(w_chunk * x_chunk[None, :], axis=1)

        in_range = (cols >= allowed_min) & (cols <= allowed_max)
        result = tl.where(in_range, acc, mask_value)
        tl.store(out_ptr + cols, result, mask=col_mask)

    @triton.jit
    def _fused_gated_lm_head_kernel_dot(
        x_ptr, w_ptr, out_ptr,
        n_embd, vocab_size,
        allowed_min, allowed_max,
        mask_value,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """v2: tl.dot (tensor-core matmul) GEMV, one row real + 15 zero-padded.

        A decode step is M=1 (one token's hidden state). Tensor-core mma
        instructions need M >= 16, so a real M=1 GEMV can't hit tensor
        cores directly -- the standard trick (used here) is to pad the
        left operand to BLOCK_M=16 with 15 all-zero rows, run a real
        16xBLOCK_Kx BLOCK_N tile matmul on tensor cores, then discard the
        14 zero rows of the result. This computes 16x the FLOPs of a true
        GEMV, but the kernel is memory-bandwidth-bound on reading
        `weight` regardless (GATE_KERNEL_FUSION_SPEC.md Section 8's
        ungated ~= baseline_unfused finding), so the extra FLOPs are free
        relative to that read -- what changes is using tensor-core
        throughput for the multiply-add instead of CUDA-core scalar ops.

        fp16 inputs, fp32 accumulator: T4 (Turing, sm_75) tensor cores do
        fp16, not tf32 (that needs Ampere+) -- P100 (Pascal, sm_60) has no
        tensor cores at all, so this kernel is only expected to beat v1
        there by better codegen, not by tensor-core throughput; measure,
        don't assume (spec Section 9).
        """
        BLOCK_M: tl.constexpr = 16

        pid = tl.program_id(axis=0)
        col_start = pid * BLOCK_N
        cols = col_start + tl.arange(0, BLOCK_N)
        col_mask = cols < vocab_size

        tile_max = col_start + BLOCK_N - 1
        tile_fully_outside = (tile_max < allowed_min) | (col_start > allowed_max)

        if tile_fully_outside:
            tl.store(out_ptr + cols, mask_value, mask=col_mask)
            return

        row_idx = tl.arange(0, BLOCK_M)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, n_embd, BLOCK_K):
            k_idx = k + tl.arange(0, BLOCK_K)
            k_mask = k_idx < n_embd

            x_chunk = tl.load(x_ptr + k_idx, mask=k_mask, other=0.0).to(tl.float16)
            # Row 0 = the real token vector, rows 1..15 = zero padding.
            xb = tl.where(row_idx[:, None] == 0, x_chunk[None, :], 0.0).to(tl.float16)

            w_ptrs = w_ptr + cols[None, :] * n_embd + k_idx[:, None]
            w_mask = k_mask[:, None] & col_mask[None, :]
            wt = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float16)

            acc = tl.dot(xb, wt, acc)

        # Discard the 15 padding rows -- row 0 is the only real GEMV result.
        result_row = tl.sum(tl.where(row_idx[:, None] == 0, acc, 0.0), axis=0)
        in_range = (cols >= allowed_min) & (cols <= allowed_max)
        result = tl.where(in_range, result_row, mask_value)
        tl.store(out_ptr + cols, result, mask=col_mask)


def gated_lm_head_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    allowed_min: int,
    allowed_max: int,
    safety_state: int = 1,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """Plain-torch reference: lm_head matmul, then the same masking
    native_kernel.NetelproVectorKernel.filter_logits_tensor applies.

    This is the oracle for the differential test (tests/test_gate_kernel.py)
    and the automatic fallback whenever HAS_TRITON is False or there's no
    CUDA device -- never a silently-skipped gate, same fail-closed posture
    as the rest of the gate stack.
    """
    if safety_state == 0:
        vocab_size = weight.size(0)
        return torch.full((vocab_size,), mask_value, dtype=x.dtype, device=x.device)

    logits = torch.nn.functional.linear(x, weight)
    vocab_size = logits.size(-1)
    masked = logits.clone()
    if allowed_min > 0:
        masked[: min(allowed_min, vocab_size)] = mask_value
    if allowed_max + 1 < vocab_size:
        masked[max(0, allowed_max + 1) :] = mask_value
    return masked


def _launch(
    kernel,
    x: torch.Tensor,
    weight: torch.Tensor,
    allowed_min: int,
    allowed_max: int,
    safety_state: int,
    mask_value: float,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    if safety_state == 0:
        vocab_size = weight.size(0)
        return torch.full((vocab_size,), mask_value, dtype=x.dtype, device=x.device)

    n_embd = x.size(-1)
    vocab_size = weight.size(0)
    out = torch.empty((vocab_size,), dtype=torch.float32, device=x.device)
    x = x.contiguous().to(torch.float32)
    weight = weight.contiguous().to(torch.float32)

    grid = (triton.cdiv(vocab_size, block_n),)
    kernel[grid](
        x, weight, out,
        n_embd, vocab_size,
        allowed_min, allowed_max,
        mask_value,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out.to(x.dtype)


def gated_lm_head_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    allowed_min: int,
    allowed_max: int,
    safety_state: int = 1,
    mask_value: float = float("-inf"),
    block_n: int = 1024,
    block_k: int = 128,
) -> torch.Tensor:
    """v1 kernel (manual reduce, no tensor cores) -- kept for the
    benchmark comparison in benchmarks/gate_kernel_fusion_kaggle.ipynb,
    not used by gated_lm_head's default dispatch anymore."""
    if not HAS_TRITON or not x.is_cuda:
        return gated_lm_head_reference(
            x, weight, allowed_min, allowed_max, safety_state, mask_value
        )
    return _launch(
        _fused_gated_lm_head_kernel_naive, x, weight, allowed_min, allowed_max,
        safety_state, mask_value, block_n, block_k,
    )


DOT_KERNEL_TOUCHED_TILE_THRESHOLD = 0.5
"""Fraction of lm_head.weight tiles the v2 kernel would actually have to
compute (not skip) for a given [allowed_min, allowed_max]. Above this,
gated_lm_head dispatches straight to the unfused reference path instead of
launching the kernel at all.

GATE_KERNEL_FUSION_SPEC.md Section 9 measured two real points on a T4:
~91% of tiles touched (broad range) -> fused 0.57ms, lost to baseline's
0.45ms; ~1/128 tiles touched (narrow range) -> fused 0.14ms, beat baseline
3.1x. 0.5 is a linear interpolation between those two points (see the
spec's "crossover" note), not a swept-and-confirmed number -- it has never
been measured directly. Override via gated_lm_head(...,
dispatch_threshold=...) if you have real numbers for your own range
distribution; refining this default needs a real sweep, not another guess.
"""


def _touched_tile_fraction(allowed_min: int, allowed_max: int, vocab_size: int, block_n: int) -> float:
    """Fraction of BLOCK_N-wide column tiles that overlap [allowed_min,
    allowed_max] at all -- these are the tiles the v2 kernel cannot skip
    (GATE_KERNEL_FUSION_SPEC.md Section 2's tile-skip early return only
    fires for tiles fully outside the range)."""
    lo = max(0, allowed_min)
    hi = min(vocab_size - 1, allowed_max)
    if hi < lo:
        return 0.0
    first_tile = lo // block_n
    last_tile = hi // block_n
    touched = last_tile - first_tile + 1
    total_tiles = -(-vocab_size // block_n)  # ceil division
    return touched / total_tiles


def gated_lm_head_dot(
    x: torch.Tensor,
    weight: torch.Tensor,
    allowed_min: int,
    allowed_max: int,
    safety_state: int = 1,
    mask_value: float = float("-inf"),
    block_n: int = 256,
    block_k: int = 32,
) -> torch.Tensor:
    """v2 kernel (tl.dot / tensor-core), unconditional -- always launches
    the kernel when Triton+CUDA are available, no range-width dispatch.

    This is the raw kernel entry point used by
    benchmarks/gate_kernel_fusion_kaggle.ipynb to measure the kernel
    itself (including on ranges where gated_lm_head's dispatch would skip
    it) -- gated_lm_head below is what production code (gated_forward)
    should call instead, since it picks the faster path per Section 9's
    measurements.
    """
    if not HAS_TRITON or not x.is_cuda:
        return gated_lm_head_reference(
            x, weight, allowed_min, allowed_max, safety_state, mask_value
        )
    return _launch(
        _fused_gated_lm_head_kernel_dot, x, weight, allowed_min, allowed_max,
        safety_state, mask_value, block_n, block_k,
    )


def gated_lm_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    allowed_min: int,
    allowed_max: int,
    safety_state: int = 1,
    mask_value: float = float("-inf"),
    block_n: int = 256,
    block_k: int = 32,
    dispatch_threshold: float = DOT_KERNEL_TOUCHED_TILE_THRESHOLD,
) -> torch.Tensor:
    """Fused gate + lm_head for one decode step (x is a single (n_embd,) row).

    This is the production entry point (what
    NetelproTransformer.gated_forward calls) -- three-way dispatch, all
    internal, callers never see a difference in output (modulo
    fp16-accumulation tolerance on the kernel path, see spec Section 9),
    only in latency:
      1. No Triton, or x isn't a CUDA tensor -> gated_lm_head_reference.
      2. Triton + CUDA, but the allowed range would force the v2 kernel to
         touch more than `dispatch_threshold` of the vocab's tiles ->
         gated_lm_head_reference too. Measured (Section 9): the fused
         kernel loses to the unfused path on broad ranges, so launching it
         there would be a regression, not a fusion.
      3. Triton + CUDA + a narrow enough range -> gated_lm_head_dot, where
         Section 9 measured a real 3.1x win.

    To benchmark or test the kernel itself regardless of range width, call
    gated_lm_head_dot directly -- this function's whole point is to *not*
    always do that.

    Default block_n/block_k (256/32) are multiples of 16 to satisfy
    tensor-core tiling; both divide Teo v2's n_embd=768 and
    vocab_size=32768 exactly, but the mask logic handles remainders for
    other configs too.
    """
    if not HAS_TRITON or not x.is_cuda:
        return gated_lm_head_reference(
            x, weight, allowed_min, allowed_max, safety_state, mask_value
        )

    if safety_state == 1:
        vocab_size = weight.size(0)
        touched = _touched_tile_fraction(allowed_min, allowed_max, vocab_size, block_n)
        if touched > dispatch_threshold:
            return gated_lm_head_reference(
                x, weight, allowed_min, allowed_max, safety_state, mask_value
            )

    return gated_lm_head_dot(
        x, weight, allowed_min, allowed_max, safety_state, mask_value, block_n, block_k,
    )
