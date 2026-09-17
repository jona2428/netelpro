"""Netelpro Deep Network: Multilayer neuro-symbolic architecture with real-time audit certificates.

Stacks NetelproLayers to form end-to-end differentiable networks where every
hidden representation is guaranteed by compiled formal gates.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from netelpro.neuro.certificate import AuditCertificate, LayerAuditRecord
from netelpro.neuro.neuron import NetelproLayer
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn as nn
    _ModuleBase = nn.Module
else:
    class _ModuleBase:  # type: ignore
        pass


class NetelproDeepNetwork(_ModuleBase):
    """Deep Neuro-Symbolic Network governed by compiled Netelpro gates."""

    def __init__(
        self,
        layer_dims: Sequence[int],
        rule_path: str | Path | None = None,
        scale_factor: float = 1000.0,
        z_min: int = -5000,
        z_max: int = 5000,
        activation_fn: str = "relu",
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("NetelproDeepNetwork requires PyTorch.")

        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must specify at least input and output dimensions (>= 2 elements)")

        self.layer_dims = list(layer_dims)
        self.scale_factor = scale_factor
        self.z_min = z_min
        self.z_max = z_max
        self.activation_fn = activation_fn

        self.layers = nn.ModuleList()
        num_layers = len(layer_dims) - 1

        for i in range(num_layers - 1):
            # Hidden neuro-symbolic layers
            layer = NetelproLayer(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                rule_path=rule_path,
                scale_factor=scale_factor,
                z_min=z_min,
                z_max=z_max,
                activation_fn=activation_fn,
            )
            self.layers.append(layer)

        # Output projection layer (linear)
        self.output_layer = nn.Linear(layer_dims[-2], layer_dims[-1])

    def forward(
        self,
        x: torch.Tensor,
        control_flags: int | Sequence[int] = 1,
    ) -> tuple[torch.Tensor, AuditCertificate]:
        """Forward pass emitting predictions and formal audit certificate."""
        t0 = time.perf_counter_ns()
        curr = x
        records: list[LayerAuditRecord] = []

        for idx, layer in enumerate(self.layers):
            t_layer_0 = time.perf_counter_ns()
            curr = layer(curr, control_flags=control_flags)
            t_layer_1 = time.perf_counter_ns()
            layer_latency_us = (t_layer_1 - t_layer_0) / 1000.0

            # Extract layer audit statistics
            audit_list = layer.audit()
            total_n = layer.out_features
            if audit_list:
                active_n = sum(1 for a in audit_list if a["allow"])
                suppressed_n = total_n - active_n
                mean_pot = sum(a["z"] for a in audit_list) / max(1, total_n)
            else:
                active_n = total_n
                suppressed_n = 0
                mean_pot = 0.0

            records.append(
                LayerAuditRecord(
                    layer_index=idx + 1,
                    total_neurons=total_n,
                    active_neurons=active_n,
                    suppressed_neurons=suppressed_n,
                    mean_potential=mean_pot,
                    latency_us=layer_latency_us,
                    details=audit_list,
                )
            )

        logits = self.output_layer(curr)
        t1 = time.perf_counter_ns()
        total_latency_us = (t1 - t0) / 1000.0

        certificate = AuditCertificate(
            records=records,
            total_latency_us=total_latency_us,
        )

        return logits, certificate
