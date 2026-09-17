"""Fused token-gate kernel: lm_head matmul + action_boundary.sl range mask,
in one GPU kernel instead of matmul-then-mask as two.

See docs/GATE_KERNEL_FUSION_SPEC.md for the full design. Guarded import,
same pattern as HAS_TORCH in ste.py: Triton has no native Windows build
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
    def _fused_gated_lm_head_kernel(
        x_ptr, w_ptr, out_ptr,
        n_embd, vocab_size,
        allowed_min, allowed_max,
        mask_value,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One program instance computes BLOCK_N output columns of
        `logits = x @ W^T` for a single row `x` (batch=1, one decode step).

        x:   (n_embd,)               contiguous fp32/fp16
        w:   (vocab_size, n_embd)    lm_head.weight, row `j` is the weight
                                     vector for vocab entry `j` (nn.Linear
                                     layout: out_features x in_features)
        out: (vocab_size,)

        Tiles fully outside [allowed_min, allowed_max] skip the dot product
        entirely and write mask_value -- the FLOP-skip described in
        GATE_KERNEL_FUSION_SPEC.md Section 2.
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


def gated_lm_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    allowed_min: int,
    allowed_max: int,
    safety_state: int = 1,
    mask_value: float = float("-inf"),
    block_n: int = 1024,
    block_k: int = 128,
) -> torch.Tensor:
    """Fused gate + lm_head for one decode step (x is a single (n_embd,) row).

    Dispatches to the Triton kernel when available and running on a CUDA
    tensor; otherwise falls back to gated_lm_head_reference. Callers never
    see the difference in output -- only in latency (measured honestly in
    benchmarks/gate_kernel_fusion_kaggle.ipynb, not assumed here).
    """
    if not HAS_TRITON or not x.is_cuda:
        return gated_lm_head_reference(
            x, weight, allowed_min, allowed_max, safety_state, mask_value
        )

    if safety_state == 0:
        vocab_size = weight.size(0)
        return torch.full((vocab_size,), mask_value, dtype=x.dtype, device=x.device)

    n_embd = x.size(-1)
    vocab_size = weight.size(0)
    out = torch.empty((vocab_size,), dtype=torch.float32, device=x.device)
    x = x.contiguous().to(torch.float32)
    weight = weight.contiguous().to(torch.float32)

    grid = (triton.cdiv(vocab_size, block_n),)
    _fused_gated_lm_head_kernel[grid](
        x, weight, out,
        n_embd, vocab_size,
        allowed_min, allowed_max,
        mask_value,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out.to(x.dtype)
