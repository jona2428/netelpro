"""Netelpro Nano-Transformer: Autoregressive transformer architecture with formal silicon gating.

Integrates compiled Netelpro gates into the feed-forward (MLP) layers and attention projections,
constraining intermediate representations and autoregressive generation with formal guarantees.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from netelpro.neuro.certificate import AuditCertificate, LayerAuditRecord
from netelpro.neuro.gate_kernel import gated_lm_head
from netelpro.neuro.neuron import NetelproLayer
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from torch import nn
    _ModuleBase = nn.Module
    _no_grad = torch.no_grad
else:
    class _ModuleBase:  # type: ignore
        pass
    def _no_grad():  # type: ignore
        def decorator(fn):
            return fn
        return decorator


@dataclass
class NetelproTransformerConfig:
    """Hyperparameter configuration for the Netelpro Nano-Transformer."""

    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    scale_factor: float = 1000.0
    z_min: int = -5000
    z_max: int = 5000
    rule_path: str | Path | None = None
    activation_fn: str = "relu"


class NetelproCausalSelfAttention(_ModuleBase):
    """Causal multi-head self-attention mechanism."""

    def __init__(self, config: NetelproTransformerConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Key, Query, Value projections combined in a single linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask to ensure attention is only applied to the left in input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class NetelproMLP(_ModuleBase):
    """Feed-Forward Network governed by compiled Netelpro formal silicon gates."""

    def __init__(self, config: NetelproTransformerConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.c_fc = NetelproLayer(
            in_features=config.n_embd,
            out_features=hidden_dim,
            rule_path=config.rule_path,
            scale_factor=config.scale_factor,
            z_min=config.z_min,
            z_max=config.z_max,
            activation_fn=config.activation_fn,
        )
        self.c_proj = nn.Linear(hidden_dim, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, control_flags: int | Sequence[int] = 1) -> torch.Tensor:
        h = self.c_fc(x, control_flags=control_flags)
        h2 = self.c_proj(h)
        return self.dropout(h2)

    def audit(self) -> list[dict[str, Any]]:
        return self.c_fc.audit()


class NetelproBlock(_ModuleBase):
    """Transformer block with causal self-attention and Netelpro formal MLP."""

    def __init__(self, config: NetelproTransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = NetelproCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = NetelproMLP(config)

    def forward(
        self, x: torch.Tensor, control_flags: int | Sequence[int] = 1
    ) -> tuple[torch.Tensor, list[dict[str, Any]], float]:
        t0 = time.perf_counter_ns()
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), control_flags=control_flags)
        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1000.0
        audit_info = self.mlp.audit()
        return x, audit_info, latency_us


class NetelproTransformer(_ModuleBase):
    """Full Netelpro Nano-Transformer with formal silicon gating in every hidden block."""

    def __init__(self, config: NetelproTransformerConfig) -> None:
        if not HAS_TORCH:
            raise RuntimeError("NetelproTransformer requires PyTorch.")

        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([NetelproBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme between token embeddings and final projection head
        self.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        control_flags: int | Sequence[int] = 1,
    ) -> tuple[torch.Tensor, torch.Tensor | None, AuditCertificate]:
        """Forward pass with causal generation and formal silicon auditing."""
        t0 = time.perf_counter_ns()
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward the transformer blocks
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        records: list[LayerAuditRecord] = []
        for l_idx, block in enumerate(self.blocks):
            x, audit_list, layer_lat_us = block(x, control_flags=control_flags)
            total_n = block.mlp.c_fc.out_features
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
                    layer_index=l_idx + 1,
                    total_neurons=total_n,
                    active_neurons=active_n,
                    suppressed_neurons=suppressed_n,
                    mean_potential=mean_pot,
                    latency_us=layer_lat_us,
                    details=audit_list,
                )
            )

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # If we are given targets, calculate cross-entropy loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        t1 = time.perf_counter_ns()
        certificate = AuditCertificate(
            records=records,
            total_latency_us=(t1 - t0) / 1000.0,
        )

        return logits, loss, certificate

    @_no_grad()
    def gated_forward(
        self,
        idx: torch.Tensor,
        allowed_min: int,
        allowed_max: int,
        safety_state: int = 1,
        mask_value: float = float("-inf"),
        control_flags: int | Sequence[int] = 1,
    ) -> torch.Tensor:
        """Last-position logits with the token gate fused into the lm_head
        matmul (docs/GATE_KERNEL_FUSION_SPEC.md), instead of computing the
        full lm_head then masking as a separate pass.

        Batch size 1 only (single decode step -- see spec Section 4.4;
        prefill-time / batched gating is out of scope for v0.1). Runs the
        transformer body identically to forward(), then replaces
        `self.lm_head(x[:, -1, :])` with gated_lm_head, which dispatches to
        the Triton kernel on CUDA or the plain-torch reference otherwise
        (netelpro/neuro/gate_kernel.py) -- same masked output either way,
        only latency differs.
        """
        batch, t = idx.size()
        assert batch == 1, "gated_forward is decode-step only (batch size 1); see spec Section 4.4"
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        device = idx.device
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _, _ = block(x, control_flags=control_flags)

        x = self.ln_f(x)
        x_last = x[0, -1, :]  # (n_embd,) -- the one row this decode step needs

        logits = gated_lm_head(
            x_last, self.lm_head.weight, allowed_min, allowed_max, safety_state, mask_value
        )
        return logits.unsqueeze(0)  # (1, vocab_size), matches forward()'s [:, -1, :] shape

    @_no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        control_flags: int = 1,
        gate: Any | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate new tokens governed by the formal transformer.

        gate: an optional netelpro.neuro.stream.NetelproStreamProcessor. When
        given, each decode step calls gated_forward() (fused gate + lm_head,
        one kernel) instead of forward() + a separate masking pass -- the
        wiring GATE_KERNEL_FUSION_SPEC.md describes. When None, behavior is
        byte-identical to before this parameter existed.
        """
        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens if context is too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            if gate is not None:
                p = gate.processor
                logits = self.gated_forward(
                    idx_cond, p.allowed_min, p.allowed_max, p.safety_state, p.mask_value,
                    control_flags=control_flags,
                )
            else:
                logits, _, _ = self(idx_cond, control_flags=control_flags)
                logits = logits[:, -1, :]
            logits = logits / max(1e-5, temperature)

            # Optionally crop the probabilities to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
