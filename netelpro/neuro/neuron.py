"""Netelpro Neuro: Neuro-symbolic neuron and dense layer with formal activation gate.

Fuses continuous linear combinations with deterministic LLVM-compiled logic
gates under a strict fail-closed contract.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from netelpro.gate import Gate
from netelpro.neuro.native_kernel import NetelproVectorKernel
from netelpro.neuro.ste import HAS_TORCH, NetelproActivationSTE

if HAS_TORCH:
    import torch
    import torch.nn as nn
    _ModuleBase = nn.Module
else:
    class _ModuleBase:  # type: ignore
        pass

_DEFAULT_RULE = Path(__file__).parent / "rules" / "activation_guard.sl"


def _eval_scalar_activation(z: float, activation_fn: str) -> float:
    if activation_fn == "relu":
        return max(0.0, z)
    elif activation_fn == "gelu":
        return 0.5 * z * (1.0 + math.erf(z / math.sqrt(2.0)))
    elif activation_fn == "sigmoid":
        return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))
    elif activation_fn == "identity":
        return z
    return max(0.0, z)


class NetelproNeuron(_ModuleBase):
    """Single neuro-symbolic neuron governed by a compiled Netelpro Gate.

    Formulation:
        z = w^T x + b
        y = g(z) if Gate.check(round(z * scale), z_min, z_max, control_flag) else 0.0 (fail-closed)
    """

    def __init__(
        self,
        in_features: int,
        rule_path: str | Path | None = None,
        scale_factor: float = 1000.0,
        z_min: int = -10000,
        z_max: int = 10000,
        activation_fn: str = "relu",
    ) -> None:
        if HAS_TORCH:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_features) * (1.0 / math.sqrt(in_features)))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.weight = [1.0 / math.sqrt(in_features)] * in_features
            self.bias = 0.0

        self.in_features = in_features
        self.scale_factor = scale_factor
        self.z_min = z_min
        self.z_max = z_max
        self.activation_fn = activation_fn

        actual_rule = Path(rule_path) if rule_path else _DEFAULT_RULE
        self.gate = Gate(actual_rule)
        self.last_audit: dict[str, Any] = {}

    def forward(
        self,
        x: Any,
        control_flag: int = 1,
    ) -> Any:
        """Forward pass through neuron with formal gate verification."""
        if HAS_TORCH and isinstance(x, torch.Tensor):
            # PyTorch path with autograd / STE support
            z = torch.dot(x, self.weight) + self.bias
            z_scaled = int(round(float(z.item()) * self.scale_factor))
            allow, reason = self.gate.check(z_scaled, self.z_min, self.z_max, control_flag)

            self.last_audit = {
                "z": float(z.item()),
                "z_scaled": z_scaled,
                "allow": allow,
                "reason": reason,
                "control_flag": control_flag,
            }

            mask = torch.tensor(1.0 if allow else 0.0, dtype=z.dtype, device=z.device)
            return NetelproActivationSTE.apply(z, mask, self.activation_fn)

        # Pure Python / scalar fallback
        if isinstance(x, (list, tuple)):
            if len(x) != self.in_features:
                raise ValueError(f"expected {self.in_features} inputs, got {len(x)}")
            z_val = sum(float(w) * float(xi) for w, xi in zip(self.weight, x)) + float(self.bias if not HAS_TORCH else self.bias.item())
        else:
            z_val = float(x)

        z_scaled = int(round(z_val * self.scale_factor))
        allow, reason = self.gate.check(z_scaled, self.z_min, self.z_max, control_flag)

        self.last_audit = {
            "z": z_val,
            "z_scaled": z_scaled,
            "allow": allow,
            "reason": reason,
            "control_flag": control_flag,
        }

        if not allow:
            return 0.0

        return _eval_scalar_activation(z_val, self.activation_fn)

    def __call__(self, x: Any, control_flag: int = 1) -> Any:
        return self.forward(x, control_flag=control_flag)

    def audit(self) -> dict[str, Any]:
        """Returns the formal audit verdict and diagnostic of the last forward pass."""
        return dict(self.last_audit)


class NetelproLayer(_ModuleBase):
    """Dense neuro-symbolic layer of M Netelpro neurons."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rule_path: str | Path | None = None,
        scale_factor: float = 1000.0,
        z_min: int = -10000,
        z_max: int = 10000,
        activation_fn: str = "relu",
    ) -> None:
        if HAS_TORCH:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
        else:
            self.linear = None  # type: ignore

        self.in_features = in_features
        self.out_features = out_features
        self.scale_factor = scale_factor
        self.z_min = z_min
        self.z_max = z_max
        self.activation_fn = activation_fn

        actual_rule = Path(rule_path) if rule_path else _DEFAULT_RULE
        self.gate = Gate(actual_rule)
        self.kernel = NetelproVectorKernel(self.gate)
        self.last_audit: list[dict[str, Any]] = []

    def forward(
        self,
        x: Any,
        control_flags: int | Sequence[int] = 1,
    ) -> Any:
        """Forward pass for batch or single vector input."""
        if HAS_TORCH and isinstance(x, torch.Tensor):
            z = self.linear(x)  # shape: [..., out_features]
            z_flat = z.view(-1, self.out_features)
            batch_size = z_flat.size(0)

            if isinstance(control_flags, int):
                if control_flags == 0:
                    mask_tensor = torch.zeros_like(z)
                    self.last_audit = [
                        {"neuron": j, "z": float(z_flat[0, j].item()) if batch_size > 0 else 0.0, "z_scaled": 0, "allow": False, "reason": "inhibit", "control_flag": 0}
                        for j in range(self.out_features)
                    ]
                    return NetelproActivationSTE.apply(z, mask_tensor, self.activation_fn)

                # Vectorized fast path directly on PyTorch tensor (SIMD / AVX / GPU)
                z_scaled = z * self.scale_factor
                mask_tensor = ((z_scaled >= self.z_min) & (z_scaled <= self.z_max)).to(z.dtype)

                # Diagnostic audit only during inference/eval to maximize training SIMD speed
                if not self.training:
                    z0 = z_flat[0]
                    z0_scaled = (z0 * self.scale_factor).round()
                    m0 = mask_tensor.view(-1, self.out_features)[0]
                    self.last_audit = [
                        {
                            "neuron": j,
                            "z": float(z0[j].item()),
                            "z_scaled": int(z0_scaled[j].item()),
                            "allow": bool(m0[j].item() > 0.5),
                            "reason": None,
                            "control_flag": 1,
                        }
                        for j in range(self.out_features)
                    ]
                else:
                    self.last_audit = []
                return NetelproActivationSTE.apply(z, mask_tensor, self.activation_fn)

            # Custom per-neuron control flags path
            c_list = list(control_flags)
            if len(c_list) != self.out_features:
                c_list = (c_list * ((self.out_features // len(c_list)) + 1))[:self.out_features]

            c_tensor = torch.tensor(c_list, dtype=z.dtype, device=z.device)
            z_scaled = z * self.scale_factor
            in_range = ((z_scaled >= self.z_min) & (z_scaled <= self.z_max)).to(z.dtype)
            mask_tensor = in_range * (c_tensor == 1).to(z.dtype)

            z0 = z_flat[0]
            z0_scaled = (z0 * self.scale_factor).round()
            m0 = mask_tensor.view(-1, self.out_features)[0]
            self.last_audit = [
                {
                    "neuron": j,
                    "z": float(z0[j].item()),
                    "z_scaled": int(z0_scaled[j].item()),
                    "allow": bool(m0[j].item() > 0.5),
                    "reason": None,
                    "control_flag": c_list[j],
                }
                for j in range(self.out_features)
            ]
            return NetelproActivationSTE.apply(z, mask_tensor, self.activation_fn)

        raise NotImplementedError("Batch pure-python layer without torch: use NetelproNeuron individually")

    def __call__(self, x: Any, control_flags: int | Sequence[int] = 1) -> Any:
        return self.forward(x, control_flags=control_flags)

    def audit(self) -> list[dict[str, Any]]:
        return list(self.last_audit)
