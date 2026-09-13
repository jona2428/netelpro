"""Straight-Through Estimator (STE) for Netelpro formal activation gates.

Enables differentiable gradient propagation through non-differentiable
formal compiled gates in PyTorch.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
    from torch.autograd import Function
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Function = object  # type: ignore


class NetelproActivationSTE(Function):
    """Straight-Through Estimator (STE) for Netelpro formal gates.

    Forward:
        y = g(z) if gate_allow else 0.0 (fail-closed)

    Backward:
        Propagates gradients grad_output * g'(z) through allowed regions,
        enabling neural network weights to train via backpropagation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        z: torch.Tensor,
        gate_mask: torch.Tensor,
        activation_fn: str = "relu",
    ) -> torch.Tensor:
        ctx.save_for_backward(z, gate_mask)
        ctx.activation_fn = activation_fn

        if activation_fn == "relu":
            base = torch.relu(z)
        elif activation_fn == "gelu":
            base = torch.nn.functional.gelu(z)
        elif activation_fn == "sigmoid":
            base = torch.sigmoid(z)
        elif activation_fn == "identity":
            base = z
        else:
            base = torch.relu(z)

        return base * gate_mask

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        z, gate_mask = ctx.saved_tensors
        activation_fn = ctx.activation_fn

        if activation_fn == "relu":
            d_base = (z > 0).to(z.dtype)
        elif activation_fn == "gelu":
            d_base = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))) + (
                z / math.sqrt(2.0 * math.pi)
            ) * torch.exp(-0.5 * z**2)
        elif activation_fn == "sigmoid":
            sig = torch.sigmoid(z)
            d_base = sig * (1.0 - sig)
        elif activation_fn == "identity":
            d_base = torch.ones_like(z)
        else:
            d_base = (z > 0).to(z.dtype)

        grad_z = grad_output * d_base * gate_mask
        return grad_z, None, None
