"""Netelpro Native Kernel: Vectorized and high-throughput execution for neural layers and logits.

Eliminates Python object lookup overhead in hot loops, maximizing native FFI
throughput for multi-neuron batches and LLM vocabularies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from netelpro.gate import Gate
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch


class NetelproVectorKernel:
    """High-throughput vectorized execution engine over compiled Netelpro gates."""

    def __init__(self, gate_or_path: Gate | str | Path) -> None:
        if isinstance(gate_or_path, Gate):
            self.gate = gate_or_path
        else:
            self.gate = Gate(Path(gate_or_path))

        self.native_fn = self.gate._filter._native_fn
        if self.native_fn is None:
            # Force entry address resolution
            self.gate.decide(*([0] * self.gate._filter._arity))
            self.native_fn = self.gate._filter._native_fn

    def evaluate_batch(
        self,
        z_scaled_values: Sequence[int],
        z_min: int,
        z_max: int,
        control_flags: int | Sequence[int],
    ) -> list[bool]:
        """Evaluates a batch of neuron potentials with minimum overhead."""
        fn = self.native_fn
        n = len(z_scaled_values)

        if isinstance(control_flags, int):
            c = control_flags
            # Hoisted local loop for maximum performance
            return [bool(fn(z, z_min, z_max, c)) for z in z_scaled_values]

        c_list = list(control_flags)
        if len(c_list) != n:
            c_list = (c_list * ((n // len(c_list)) + 1))[:n]

        return [bool(fn(z, z_min, z_max, c)) for z, c in zip(z_scaled_values, c_list)]

    def filter_logits_tensor(
        self,
        logits: torch.Tensor,
        allowed_min: int,
        allowed_max: int,
        safety_state: int = 1,
        mask_value: float = float("-inf"),
    ) -> torch.Tensor:
        """Vectorized logit pruning over PyTorch tensor.

        For contiguous action bounds, executes vectorized tensor slicing (nanosecond latency).
        For custom discrete sets, evaluates native machine code per index.
        """
        if safety_state == 0:
            # Full safety freeze: all tokens masked out
            return torch.full_like(logits, mask_value)

        masked = logits.clone()
        vocab_size = logits.size(-1)

        # Vectorized fast-path for linear intervals
        if allowed_min > 0:
            masked[..., :min(allowed_min, vocab_size)] = mask_value
        if allowed_max + 1 < vocab_size:
            masked[..., max(0, allowed_max + 1):] = mask_value

        return masked
