"""Netelpro SDS (Silicon Dynamic State): Non-Transformer linear state-space architecture.

Fuses continuous-time dynamic state recurrence with formal compiled Netelpro silicon bounds.
Achieves O(1) inference memory complexity (Zero KV-Cache) while guaranteeing numerical
stability through hardware-bounded state gating.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from netelpro.neuro.certificate import AuditCertificate, LayerAuditRecord
from netelpro.neuro.neuron import NetelproLayer
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Function
    _ModuleBase = nn.Module
    _FunctionBase = Function
    _no_grad = torch.no_grad
else:
    class _ModuleBase:  # type: ignore
        pass
    class _FunctionBase:  # type: ignore
        pass
    def _no_grad():  # type: ignore
        def decorator(fn):
            return fn
        return decorator


@dataclass
class NetelproSDSConfig:
    """Hyperparameter configuration for Netelpro Silicon Dynamic State models."""

    vocab_size: int = 256
    n_layer: int = 4
    d_model: int = 128
    d_state: int = 16            # State expansion dimensionality per channel
    d_conv: int = 4              # Local causal 1D convolution receptive field
    expand: int = 2              # Expansion factor for inner dimension
    scale_factor: float = 1000.0
    z_min: int = -5000           # Bound: -5000 / 1000 = -5.0
    z_max: int = 5000            # Bound: +5000 / 1000 = +5.0
    rule_path: str | Path | None = None
    dropout: float = 0.0
    activation_fn: str = "relu"

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model


class NetelproSiliconStateSTE(_FunctionBase):
    """Straight-Through Estimator (STE) for state-space recurrence clipping.

    Guarantees state tensor elements remain strictly within [z_min / scale, z_max / scale],
    eliminating numeric state explosion (NaNs) while preserving backpropagation gradients.
    """

    @staticmethod
    def forward(
        ctx: Any,
        h: torch.Tensor,
        bound_min: float,
        bound_max: float,
        control_flag: int = 1,
    ) -> torch.Tensor:
        ctx.save_for_backward(h)
        ctx.bound_min = bound_min
        ctx.bound_max = bound_max
        ctx.control_flag = control_flag

        if control_flag == 0:
            # Full fail-closed safety state: state is zeroed out
            return torch.zeros_like(h)

        return torch.clamp(h, min=bound_min, max=bound_max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        (h,) = ctx.saved_tensors
        bound_min = ctx.bound_min
        bound_max = ctx.bound_max
        control_flag = ctx.control_flag

        if control_flag == 0:
            return torch.zeros_like(grad_output), None, None, None

        # Straight-Through Estimator: pass gradients where state was within silicon limits
        mask = (h >= bound_min) & (h <= bound_max)
        grad_h = grad_output * mask.to(grad_output.dtype)
        return grad_h, None, None, None


class NetelproSDSCell(_ModuleBase):
    """Core Silicon Dynamic State Cell with selective gating and bounded recurrence."""

    def __init__(self, config: NetelproSDSConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_state = config.d_state

        self.bound_min = float(config.z_min) / float(config.scale_factor)
        self.bound_max = float(config.z_max) / float(config.scale_factor)

        # Input projection: projects input to 2 * d_inner (one branch for SSM, one for gate)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)

        # Short 1D Causal Convolution for capturing immediate token n-grams
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )

        # Selective projection: input-dependent Delta, B, C
        self.dt_rank = max(16, self.d_model // 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize log(A): strictly negative to guarantee exponential decay
        # A has shape (d_inner, d_state)
        A_init = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _discretize(
        self, delta: torch.Tensor, B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretizes continuous parameters (A, B) using time step delta.

        Args:
            delta: Shape (B, T, d_inner) or (B, d_inner)
            B: Shape (B, T, d_state) or (B, d_state)

        Returns:
            dA: Shape (B, T, d_inner, d_state) or (B, d_inner, d_state)
            dB: Shape (B, T, d_inner, d_state) or (B, d_inner, d_state)
        """
        # A is negative: -exp(A_log)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        if delta.dim() == 3:
            # (B, T, d_inner, 1) * (1, 1, d_inner, d_state) -> (B, T, d_inner, d_state)
            dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            # dB = delta * B
            # (B, T, d_inner, 1) * (B, T, 1, d_state) -> (B, T, d_inner, d_state)
            dB = delta.unsqueeze(-1) * B.unsqueeze(-2)
        else:
            # (B, d_inner, 1) * (1, d_inner, d_state) -> (B, d_inner, d_state)
            dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
            dB = delta.unsqueeze(-1) * B.unsqueeze(-2)

        return dA, dB

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        conv_state: torch.Tensor,
        control_flag: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a single O(1) recurrent step for token-by-token inference.

        Args:
            x_t: Shape (B, d_model)
            state: Hidden state tensor of shape (B, d_inner, d_state)
            conv_state: Convolution buffer of shape (B, d_inner, d_conv)
            control_flag: Silicon safety flag (1 = allow, 0 = fail-closed)

        Returns:
            y_t: Output of shape (B, d_model)
            new_state: Updated bounded state of shape (B, d_inner, d_state)
            new_conv_state: Updated conv buffer of shape (B, d_inner, d_conv)
        """
        B = x_t.size(0)

        # 1. Project input
        xz = self.in_proj(x_t)  # (B, 2 * d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # (B, d_inner), (B, d_inner)

        # 2. Update 1D Conv buffer and compute causal conv output for this step
        # Shift buffer left and append current x_inner
        new_conv_state = torch.cat([conv_state[:, :, 1:], x_inner.unsqueeze(-1)], dim=-1)
        # Compute 1D dot product with conv1d kernel weights
        conv_out = (new_conv_state * self.conv1d.weight.squeeze(1)).sum(dim=-1)
        if self.conv1d.bias is not None:
            conv_out = conv_out + self.conv1d.bias
        x_conv = F.silu(conv_out)  # (B, d_inner)

        # 3. Dynamic Selection: derive Delta, B, C from x_conv
        ssm_params = self.x_proj(x_conv)  # (B, dt_rank + 2 * d_state)
        dt_raw = ssm_params[:, : self.dt_rank]
        B_ssm = ssm_params[:, self.dt_rank : self.dt_rank + self.d_state]
        C_ssm = ssm_params[:, self.dt_rank + self.d_state :]

        delta = F.softplus(self.dt_proj(dt_raw))  # (B, d_inner)

        # 4. Discretize
        dA, dB = self._discretize(delta, B_ssm)  # (B, d_inner, d_state)

        # 5. Silicon-Bounded State Transition: h_t = dA * h_{t-1} + dB * x
        h_next = dA * state + dB * x_conv.unsqueeze(-1)

        # Apply Formal Silicon Bound STE: [-5.0, 5.0]
        bounded_state = NetelproSiliconStateSTE.apply(
            h_next, self.bound_min, self.bound_max, control_flag
        )

        # 6. Readout: y = (bounded_state * C).sum(dim=-1) + D * x_conv
        y_ssm = (bounded_state * C_ssm.unsqueeze(-2)).sum(dim=-1)  # (B, d_inner)
        y_ssm = y_ssm + self.D * x_conv

        # 7. Multiplicative gate with z branch (SwiGLU-style gating)
        y_inner = y_ssm * F.silu(z)

        # 8. Output projection
        y_t = self.out_proj(y_inner)  # (B, d_model)

        return y_t, bounded_state, new_conv_state

    def forward(
        self,
        x: torch.Tensor,
        control_flags: int | Sequence[int] = 1,
    ) -> torch.Tensor:
        """Sequential/Parallel forward pass over full sequence (B, T, d_model)."""
        B, T, D = x.size()

        c_flag = control_flags if isinstance(control_flags, int) else control_flags[0]

        # 1. Project input
        xz = self.in_proj(x)  # (B, T, 2 * d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # (B, T, d_inner), (B, T, d_inner)

        # 2. Causal 1D convolution over sequence
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)  # (B, T, d_inner)

        # 3. Dynamic Selection: derive Delta, B, C for all tokens
        ssm_params = self.x_proj(x_conv)  # (B, T, dt_rank + 2 * d_state)
        dt_raw = ssm_params[:, :, : self.dt_rank]
        B_ssm = ssm_params[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C_ssm = ssm_params[:, :, self.dt_rank + self.d_state :]

        delta = F.softplus(self.dt_proj(dt_raw))  # (B, T, d_inner)

        # 4. Discretize
        dA, dB = self._discretize(delta, B_ssm)  # (B, T, d_inner, d_state)

        # 5. Silicon-Bounded Dynamic Recurrence across sequence length T
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []

        for t in range(T):
            dA_t = dA[:, t]      # (B, d_inner, d_state)
            dB_t = dB[:, t]      # (B, d_inner, d_state)
            x_t = x_conv[:, t]   # (B, d_inner)
            C_t = C_ssm[:, t]    # (B, d_state)

            h = dA_t * h + dB_t * x_t.unsqueeze(-1)
            h = NetelproSiliconStateSTE.apply(h, self.bound_min, self.bound_max, c_flag)

            y_t = (h * C_t.unsqueeze(-2)).sum(dim=-1) + self.D * x_t
            y_list.append(y_t)

        y_ssm = torch.stack(y_list, dim=1)  # (B, T, d_inner)

        # 6. Gated output
        y_inner = y_ssm * F.silu(z)

        # 7. Project back to d_model
        return self.out_proj(y_inner)


class NetelproSDSBlock(_ModuleBase):
    """Full Silicon Dynamic State transformer-alternative block."""

    def __init__(self, config: NetelproSDSConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.sds = NetelproSDSCell(config)
        self.ln_2 = nn.LayerNorm(config.d_model)

        from netelpro.neuro.transformer import NetelproMLP, NetelproTransformerConfig
        mlp_config = NetelproTransformerConfig(
            vocab_size=config.vocab_size,
            n_embd=config.d_model,
            dropout=config.dropout,
            scale_factor=config.scale_factor,
            z_min=config.z_min,
            z_max=config.z_max,
            rule_path=config.rule_path,
            activation_fn=config.activation_fn,
        )
        self.mlp = NetelproMLP(mlp_config)

    def forward(
        self, x: torch.Tensor, control_flags: int | Sequence[int] = 1
    ) -> Tuple[torch.Tensor, list[dict[str, Any]], float]:
        t0 = time.perf_counter_ns()
        x = x + self.sds(self.ln_1(x), control_flags=control_flags)
        x = x + self.mlp(self.ln_2(x), control_flags=control_flags)
        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1000.0
        audit_info = self.mlp.audit()
        return x, audit_info, latency_us

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        conv_state: torch.Tensor,
        control_flag: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token recurrent step."""
        sds_out, new_state, new_conv_state = self.sds.step(
            self.ln_1(x_t), state, conv_state, control_flag=control_flag
        )
        x_next = x_t + sds_out
        x_next = x_next + self.mlp(self.ln_2(x_next), control_flags=control_flag)
        return x_next, new_state, new_conv_state


class NetelproSDSModel(_ModuleBase):
    """Full Silicon Dynamic State Autoregressive Language Model.

    Provides dual-mode execution:
    - Training mode: Sequence forward pass with associative recurrence.
    - Inference mode: Pure O(1) state step (Zero KV-Cache, constant microsecond latency).
    """

    def __init__(self, config: NetelproSDSConfig) -> None:
        if not HAS_TORCH:
            raise RuntimeError("NetelproSDSModel requires PyTorch.")

        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([NetelproSDSBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying scheme
        self.wte.weight = self.lm_head.weight

        # Weight initialization
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
        targets: Optional[torch.Tensor] = None,
        control_flags: int | Sequence[int] = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], AuditCertificate]:
        """Sequence forward pass for training."""
        B, T = idx.size()
        x = self.drop(self.wte(idx))

        total_latency = 0.0
        layer_records: list[LayerAuditRecord] = []

        for layer_idx, block in enumerate(self.blocks):
            x, audits, block_latency = block(x, control_flags=control_flags)
            total_latency += block_latency

            for rec in audits:
                layer_records.append(
                    LayerAuditRecord(
                        layer_index=layer_idx,
                        neuron_index=rec.get("neuron_index", 0),
                        pre_activation=rec.get("z", 0.0),
                        scaled_value=rec.get("z_scaled", 0),
                        gate_allowed=rec.get("allow", True),
                        gate_reason=rec.get("reason", "OK"),
                        control_flag=rec.get("control_flag", 1),
                        latency_us=block_latency,
                    )
                )

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        certificate = AuditCertificate(
            records=layer_records,
            total_latency_us=total_latency,
            verified=all(r.gate_allowed for r in layer_records),
        )

        return logits, loss, certificate

    def init_inference_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Initializes constant O(1) state buffers for pure recurrent inference."""
        if device is None:
            device = next(self.parameters()).device

        states = [
            torch.zeros(
                batch_size,
                self.config.d_inner,
                self.config.d_state,
                device=device,
                dtype=self.wte.weight.dtype,
            )
            for _ in range(self.config.n_layer)
        ]
        conv_states = [
            torch.zeros(
                batch_size,
                self.config.d_inner,
                self.config.d_conv,
                device=device,
                dtype=self.wte.weight.dtype,
            )
            for _ in range(self.config.n_layer)
        ]
        return states, conv_states

    def step(
        self,
        token_t: torch.Tensor,
        states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
        control_flag: int = 1,
    ) -> Tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Generates the next logit vector using O(1) constant-memory recurrence.

        Zero KV-Cache: Memory consumption is strictly invariant with sequence length.
        """
        x = self.wte(token_t)  # (B, d_model)

        new_states = []
        new_conv_states = []

        for block, state, c_state in zip(self.blocks, states, conv_states):
            x, next_st, next_c_st = block.step(
                x, state, c_state, control_flag=control_flag
            )
            new_states.append(next_st)
            new_conv_states.append(next_c_st)

        x = self.ln_f(x)
        logits_t = self.lm_head(x)  # (B, vocab_size)

        return logits_t, new_states, new_conv_states

    @_no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        control_flag: int = 1,
    ) -> torch.Tensor:
        """Autoregressive text generation via pure O(1) recurrent stepping."""
        B, T = prompt_tokens.size()
        states, conv_states = self.init_inference_state(B, device=prompt_tokens.device)

        logits = None
        for t in range(T):
            token_t = prompt_tokens[:, t]
            logits, states, conv_states = self.step(
                token_t, states, conv_states, control_flag=control_flag
            )

        generated = []
        next_token = logits.argmax(dim=-1)

        for _ in range(max_new_tokens):
            generated.append(next_token)
            logits, states, conv_states = self.step(
                next_token, states, conv_states, control_flag=control_flag
            )
            if temperature > 0.0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)

        gen_tensor = torch.stack(generated, dim=1)
        return torch.cat([prompt_tokens, gen_tensor], dim=1)
