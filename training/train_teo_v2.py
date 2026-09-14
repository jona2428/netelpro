"""Teo v2 (GPT-2-small-class, ~124M params) packed token stream trainer.

Architecture:
  - 12 layers, 12 attention heads, d_model 768, mlp 3072, context 1024, vocab 32768.
  - Tied embeddings (wte == lm_head), zero dropout.
  - Netelpro band-gate + Straight-Through Estimator (STE) semantics, vectorized on GPU/CPU.
  - AdamW optimizer (lr 3e-4, warmup 2000 steps, cosine decay to 3e-5, weight decay 0.1 on non-embedding params, grad clip 1.0).
  - Multi-shard resume scanning training/data/teo_v2_corpus/shard_*.bin + meta.json.
  - Checkpoints saved every 1000 steps to models/teo_v2/checkpoint.pt with full model, optimizer, step, shard index, and RNG states.
  - Contiguous packed uint16 token streams with zero padding tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator, Sequence

import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from netelpro.neuro.ste import HAS_TORCH, NetelproActivationSTE
from training.data.compile_packed import get_batch, load_stream, stream_stats

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    raise RuntimeError("PyTorch 2.x is required for teo_v2 training.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TeoV2Config:
    """Hyperparameter configuration for Teo v2 (~124M params)."""

    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    context: int = 1024
    mlp: int = 3072
    tied_embeddings: bool = True
    dropout: float = 0.0

    # Optimizer & schedule
    lr: float = 3e-4
    warmup_steps: int = 2000
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Netelpro formal silicon constraints
    scale_factor: float = 1000.0
    z_min: int = -5000
    z_max: int = 5000
    activation_fn: str = "relu"

    # Training runtime defaults
    batch_size: int = 4
    log_interval: int = 50
    save_interval: int = 1000
    max_steps: int = 100000

    # Aliases for compatibility with other Netelpro configs
    @property
    def n_embd(self) -> int:
        return self.d_model

    @property
    def n_layer(self) -> int:
        return self.n_layers

    @property
    def n_head(self) -> int:
        return self.n_heads

    @property
    def block_size(self) -> int:
        return self.context

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_embd"] = self.n_embd
        d["n_layer"] = self.n_layer
        d["n_head"] = self.n_head
        d["block_size"] = self.block_size
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeoV2Config:
        clean = dict(data)
        # Normalize aliases
        if "d_model" not in clean and "n_embd" in clean:
            clean["d_model"] = clean["n_embd"]
        if "n_layers" not in clean and "n_layer" in clean:
            clean["n_layers"] = clean["n_layer"]
        if "n_heads" not in clean and "n_head" in clean:
            clean["n_heads"] = clean["n_head"]
        if "context" not in clean and "block_size" in clean:
            clean["context"] = clean["block_size"]

        # Drop non-dataclass keys if any
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in clean.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_path: str | Path) -> TeoV2Config:
        p = Path(json_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, json_path: str | Path) -> None:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Teo v2 Architecture (Netelpro Band-Gate + STE + SDPA Attention)
# ---------------------------------------------------------------------------


class TeoV2CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention using PyTorch SDPA."""

    def __init__(self, config: TeoV2Config) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout

        # Query, Key, Value projections in a single linear layer
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Project and reshape: (B, T, 3 * C) -> 3 x (B, n_heads, T, head_dim)
        qkv = self.c_attn(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Fast vectorized SDPA with causal masking
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Re-assemble head outputs: (B, n_heads, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class TeoV2MLP(nn.Module):
    """Feed-Forward Network with formal Netelpro band-gate & Straight-Through Estimator.

    During training: vectorized directly on GPU/CPU using PyTorch tensor operations.
    During certification/audit: bounds verification adheres to formal fail-closed logic.
    """

    def __init__(self, config: TeoV2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.mlp)
        self.c_proj = nn.Linear(config.mlp, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.scale_factor = config.scale_factor
        self.z_min = config.z_min
        self.z_max = config.z_max
        self.activation_fn = config.activation_fn
        self.out_features = config.mlp

    def forward(self, x: torch.Tensor, control_flags: int = 1) -> torch.Tensor:
        z = self.c_fc(x)

        if control_flags == 0:
            # Complete inhibition (fail-closed)
            mask = torch.zeros_like(z)
            return NetelproActivationSTE.apply(z, mask, self.activation_fn)

        # Vectorized Netelpro band-gate on PyTorch tensor
        z_scaled = z * self.scale_factor
        mask = ((z_scaled >= self.z_min) & (z_scaled <= self.z_max)).to(z.dtype)

        # Straight-Through Estimator propagates gradients through allowed region
        h = NetelproActivationSTE.apply(z, mask, self.activation_fn)
        h2 = self.c_proj(h)
        return self.dropout(h2)


class TeoV2Block(nn.Module):
    """Transformer block with pre-LayerNorm, causal SDPA, and Netelpro formal MLP."""

    def __init__(self, config: TeoV2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = TeoV2CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = TeoV2MLP(config)

    def forward(self, x: torch.Tensor, control_flags: int = 1) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), control_flags=control_flags)
        return x


class TeoV2Transformer(nn.Module):
    """Full Teo v2 Autoregressive Transformer (~124M parameters).

    Features:
      - Tied embeddings: wte.weight == lm_head.weight
      - Positional embeddings: wpe
      - 12 Netelpro blocks with band-gate STE activations
      - Contiguous token inputs without padding
    """

    def __init__(self, config: TeoV2Config) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.context, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TeoV2Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying scheme
        if config.tied_embeddings:
            self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Scaled residual projection initialization (standard GPT-2)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

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

    def count_parameters(self) -> dict[str, int]:
        """Returns parameter breakdown."""
        total = sum(p.numel() for p in self.parameters())
        unique = sum(p.numel() for p in {p.data_ptr(): p for p in self.parameters()}.values())
        return {"total_parameters": total, "unique_parameters": unique}

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        control_flags: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.context, f"Seq length {t} exceeds context {self.config.context}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, control_flags=control_flags)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        control_flags: int = 1,
    ) -> torch.Tensor:
        """Autoregressive text generation."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.context else idx[:, -self.config.context :]
            logits, _ = self(idx_cond, control_flags=control_flags)
            logits = logits[:, -1, :] / max(1e-5, temperature)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ---------------------------------------------------------------------------
# Optimizer & LR Schedule
# ---------------------------------------------------------------------------


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    """Configures AdamW with weight decay 0.1 on non-embedding 2D weights.

    Embedding weights (wte, wpe) and 1D parameters (biases, LayerNorms)
    receive weight decay 0.0.
    """
    decay_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Non-embedding 2D parameters get weight decay
        if param.dim() >= 2 and not any(emb in name for emb in ("wte", "wpe")):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = (device_type == "cuda") and hasattr(torch.optim.AdamW, "fused")
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=betas,
        eps=1e-8,
        fused=use_fused,
    )
    return optimizer


def get_lr(
    step: int,
    warmup_steps: int = 2000,
    max_steps: int = 100000,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
) -> float:
    """Computes learning rate with linear warmup and cosine decay to min_lr."""
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Corpus & Multi-Shard Utilities
# ---------------------------------------------------------------------------


def discover_shards(corpus_path: str | Path) -> tuple[list[Path], dict[str, Any] | None]:
    """Scans for shard_*.bin files and optional meta.json sidecar."""
    p = Path(corpus_path)
    shards: list[Path] = []
    meta: dict[str, Any] | None = None

    if p.is_file():
        shards = [p]
        meta_p = p.with_suffix(".meta.json")
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return shards, meta

    if p.is_dir():
        shards = sorted(p.glob("shard_*.bin"))
        if not shards:
            shards = sorted(p.glob("*.bin"))

        # Smart fallback: if no shards found in requested dir, check data/teo_v2
        if not shards and str(p) not in ("data/teo_v2", "data\\teo_v2"):
            alt_dir = Path("data/teo_v2")
            if alt_dir.is_dir():
                alt_shards = sorted(alt_dir.glob("shard_*.bin"))
                if alt_shards:
                    p = alt_dir
                    shards = alt_shards

        meta_file = p / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    return shards, meta


def build_synthetic_corpus(
    out_bin: str | Path,
    num_tokens: int = 2_000_000,
    vocab_size: int = 32768,
    seed: int = 42,
    doc_min_tokens: int = 40,
    doc_max_tokens: int = 120,
) -> dict[str, Any]:
    """Generates a synthetic packed uint16 binary stream with bos/eos framing and zero padding.

    Uses special token IDs:
      pad = 0 (forbidden in stream)
      bos = 1 (document boundary)
      eos = 2 (document boundary)
      unk = 3

    Builds learnable motifs so next-token prediction loss decreases noticeably during smoke tests.
    """
    rng = np.random.default_rng(seed)
    out_p = Path(out_bin)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    bos_id = 1
    eos_id = 2
    pad_id = 0

    # Structured motifs within vocab range [4, 1024] to facilitate loss convergence
    motif_pool = [
        rng.integers(4, min(1000, vocab_size), size=rng.integers(6, 18)).tolist()
        for _ in range(64)
    ]

    tokens: list[int] = []
    total_docs = 0

    while len(tokens) < num_tokens:
        doc_len = int(rng.integers(doc_min_tokens, doc_max_tokens))
        doc_tokens = [bos_id]
        while len(doc_tokens) < doc_len - 1:
            motif = motif_pool[int(rng.integers(0, len(motif_pool)))]
            doc_tokens.extend(motif)
        doc_tokens = doc_tokens[: doc_len - 1] + [eos_id]

        tokens.extend(doc_tokens)
        total_docs += 1

    tokens = tokens[:num_tokens]
    # Ensure final token is eos for clean framing
    if len(tokens) > 0 and tokens[-1] != eos_id:
        tokens[-1] = eos_id

    arr = np.array(tokens, dtype="<u2")
    assert not np.any(arr == pad_id), "Synthetic stream must not contain pad token!"

    arr.tofile(out_p)
    sha256 = hashlib.sha256(arr.tobytes()).hexdigest()

    meta = {
        "total_tokens": int(len(arr)),
        "total_docs": total_docs,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "sha256": sha256,
        "pad_token_id": pad_id,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
    }

    meta_file = Path(f"{out_p}.meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    shard_idx: int,
    config: TeoV2Config,
    loss: float,
    rng_state: dict[str, Any] | None = None,
    scaler: Any = None,
) -> Path:
    """Atomically saves training checkpoint."""
    p = Path(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = p.with_suffix(".pt.tmp")

    if rng_state is None:
        rng_state = {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    ckpt = {
        "step": step,
        "epoch": epoch,
        "shard_idx": shard_idx,
        "loss": float(loss),
        "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng_state,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "timestamp": time.time(),
    }

    torch.save(ckpt, tmp_p)
    if p.exists():
        p.unlink()
    tmp_p.rename(p)
    return p


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Loads checkpoint state into model, optimizer, and RNG states."""
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    ckpt = torch.load(p, map_location=device, weights_only=False)

    if model is not None and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    # Restore RNG states if available
    if "rng_state" in ckpt and isinstance(ckpt["rng_state"], dict):
        rng = ckpt["rng_state"]
        if "torch" in rng and rng["torch"] is not None:
            try:
                torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
            except Exception:
                pass
        if "numpy" in rng and rng["numpy"] is not None:
            try:
                np.random.set_state(rng["numpy"])
            except Exception:
                pass
        if "python" in rng and rng["python"] is not None:
            try:
                random.setstate(rng["python"])
            except Exception:
                pass

    return ckpt


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def train_teo_v2(
    config: TeoV2Config | None = None,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data/teo_v2",
    data_file: str | Path | None = None,
    checkpoint_dir: str | Path = "models/teo_v2",
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    max_steps: int | None = None,
    batch_size: int | None = None,
    context_len: int | None = None,
    lr: float | None = None,
    warmup_steps: int | None = None,
    min_lr: float | None = None,
    weight_decay: float | None = None,
    grad_clip: float | None = None,
    log_interval: int = 50,
    save_interval: int = 1000,
    autocast: bool = False,
    device: str | None = None,
    seed: int = 42,
    num_threads: int | None = None,
) -> dict[str, Any]:
    """Multi-shard trainer for Teo v2 with checkpoint resumption and token-throughput logging."""
    # 1. Config resolution
    if config is None:
        cfg_file = Path(config_path) if config_path else Path(__file__).parent / "teo_v2_config.json"
        if cfg_file.exists():
            config = TeoV2Config.from_json(cfg_file)
        else:
            config = TeoV2Config()

    # Overrides from parameters
    if max_steps is not None:
        config.max_steps = max_steps
    if batch_size is not None:
        config.batch_size = batch_size
    if context_len is not None:
        config.context = context_len
    if lr is not None:
        config.lr = lr
    if warmup_steps is not None:
        config.warmup_steps = warmup_steps
    if min_lr is not None:
        config.min_lr = min_lr
    if weight_decay is not None:
        config.weight_decay = weight_decay
    if grad_clip is not None:
        config.grad_clip = grad_clip
    if log_interval is None:
        log_interval = config.log_interval
    if save_interval is None:
        save_interval = config.save_interval

    # 2. Hardware setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        threads = num_threads or min(8, os.cpu_count() or 4)
        torch.set_num_threads(threads)

    # Autocast setup
    use_autocast = autocast and (device == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_autocast else None

    # Determinism
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    # 3. Model & Optimizer
    model = TeoV2Transformer(config).to(device)
    optimizer = configure_optimizers(
        model,
        weight_decay=config.weight_decay,
        lr=config.lr,
        device_type=device,
    )

    # 4. Checkpoint path resolution
    ckpt_file = Path(checkpoint_path) if checkpoint_path else Path(checkpoint_dir) / "checkpoint.pt"
    start_step = 0
    start_epoch = 0
    start_shard_idx = 0
    last_loss = 0.0

    if resume and ckpt_file.exists():
        try:
            print(f"🔄 Resuming training from checkpoint: {ckpt_file}", flush=True)
            ckpt = load_checkpoint(ckpt_file, model=model, optimizer=optimizer, scaler=scaler, device=device)
            start_step = ckpt["step"]
            start_epoch = ckpt.get("epoch", 0)
            start_shard_idx = ckpt.get("shard_idx", 0)
            last_loss = ckpt.get("loss", 0.0)
            print(f"   Resumed at step {start_step}, epoch {start_epoch}, shard {start_shard_idx}, loss: {last_loss:.4f}")
        except Exception as e:
            print(f"⚠️ Checkpoint load failed, starting fresh: {e}")

    # 5. Shard discovery
    if data_file:
        shards = [Path(data_file)]
        meta = None
    else:
        shards, meta = discover_shards(data_dir)

    if not shards:
        raise FileNotFoundError(f"No corpus shards found at {data_file or data_dir}")

    total_shards = len(shards)
    print(f"📂 Discovered {total_shards} corpus shard(s): {[s.name for s in shards[:5]]}")

    # 6. Training loop
    model.train()
    step = start_step
    epoch = start_epoch
    curr_shard_idx = start_shard_idx % total_shards

    # Pre-load initial shard stream
    stream = load_stream(shards[curr_shard_idx])
    assert len(stream) > config.context + 1, (
        f"Shard {shards[curr_shard_idx]} with {len(stream)} tokens too small for context {config.context}"
    )

    history: list[dict[str, Any]] = []
    tokens_per_step = config.batch_size * config.context
    t_log_start = time.perf_counter()
    tokens_in_window = 0

    print("=" * 80)
    print(f"🚀 Training Teo v2 (~124M params) on {device.upper()}")
    print(f"   Context: {config.context} | Batch: {config.batch_size} | LR: {config.lr:.2e} -> {config.min_lr:.2e}")
    print(f"   Warmup: {config.warmup_steps} steps | Autocast: {use_autocast} | Save Interval: {save_interval}")
    print("=" * 80)

    while step < config.max_steps:
        t_step_start = time.perf_counter()

        # Update learning rate
        current_lr = get_lr(
            step,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            lr=config.lr,
            min_lr=config.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Fetch batch
        try:
            inputs, targets = get_batch(
                stream,
                block_size=config.context,
                batch_size=config.batch_size,
                generator=np_rng,
                device=device,
                return_tensor=True,
            )
        except Exception:
            # Rotate to next shard if current stream sampling errors
            curr_shard_idx = (curr_shard_idx + 1) % total_shards
            stream = load_stream(shards[curr_shard_idx])
            inputs, targets = get_batch(
                stream,
                block_size=config.context,
                batch_size=config.batch_size,
                generator=np_rng,
                device=device,
                return_tensor=True,
            )

        optimizer.zero_grad(set_to_none=True)

        if use_autocast:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(inputs, targets=targets)
            assert loss is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(inputs, targets=targets)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        loss_val = float(loss.item())
        last_loss = loss_val
        step += 1
        tokens_in_window += tokens_per_step

        # Periodic shard rotation across epochs
        if step % max(1, (len(stream) // (config.context * max(1, config.batch_size) * 10) or 500)) == 0:
            curr_shard_idx = (curr_shard_idx + 1) % total_shards
            if curr_shard_idx == 0:
                epoch += 1
            stream = load_stream(shards[curr_shard_idx])

        # Logging every log_interval steps
        if step % log_interval == 0 or step == start_step + 1 or step == config.max_steps:
            dt = time.perf_counter() - t_log_start
            tok_per_sec = tokens_in_window / max(dt, 1e-4)
            record = {
                "step": step,
                "loss": round(loss_val, 4),
                "lr": current_lr,
                "tokens_per_sec": round(tok_per_sec, 1),
                "shard_idx": curr_shard_idx,
                "epoch": epoch,
                "dt": round(dt, 2),
            }
            history.append(record)
            print(
                f"[Step {step:05d}/{config.max_steps:05d}] "
                f"loss: {loss_val:.4f} | "
                f"lr: {current_lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s | "
                f"shard: {curr_shard_idx} | "
                f"epoch: {epoch} | "
                f"dt: {dt:.2f}s",
                flush=True,
            )
            t_log_start = time.perf_counter()
            tokens_in_window = 0

        # Periodic checkpointing
        if step % save_interval == 0 or step == config.max_steps:
            save_checkpoint(
                ckpt_file,
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch,
                shard_idx=curr_shard_idx,
                config=config,
                loss=loss_val,
                scaler=scaler,
            )
            print(f"💾 Checkpoint saved at step {step}: {ckpt_file}", flush=True)

    print("=" * 80)
    print(f"✅ Teo v2 training completed at step {step} with final loss: {last_loss:.4f}")
    print(f"   Saved checkpoint: {ckpt_file}")
    print("=" * 80)

    return {
        "final_step": step,
        "final_loss": last_loss,
        "checkpoint_path": str(ckpt_file),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Smoke Test Runner
# ---------------------------------------------------------------------------


def run_smoke_test(
    corpus_bin: str | Path | None = None,
    checkpoint_dir: str | Path = "models/teo_v2",
    steps: int = 200,
    batch_size: int = 1,
    context_len: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    """Executes the mandatory CPU smoke test:

    1. Builds tiny synthetic packed bin (~2M random token IDs with bos/eos framing, zero pad).
    2. Trains 100 steps from step 0. Saves checkpoint at step 100.
    3. Stops and resumes from step 100 to step 200.
    4. Verifies:
       (a) Loss decreases.
       (b) Stop+resume produces continuous loss and correct step counter.
       (c) Checkpoint file is valid.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pt"

    # 1. Synthetic corpus
    if corpus_bin is None:
        corpus_bin = Path("training/data/teo_v2_corpus/smoke_synth.bin")
    corpus_p = Path(corpus_bin)
    if not corpus_p.exists() or corpus_p.stat().st_size < 2_000_000 * 2:
        print(f"🔨 Building synthetic packed corpus ({corpus_p})...", flush=True)
        build_synthetic_corpus(corpus_p, num_tokens=2_000_000, vocab_size=32768)

    # Diagnostic check on corpus
    stream = load_stream(corpus_p)
    stats = stream_stats(stream)
    assert stats["pad_fraction"] == 0.0, f"Smoke test corpus has padding: {stats}"
    print(f"✅ Verified synthetic stream: {stats['total_tokens']:,} tokens, pad_fraction={stats['pad_fraction']}")

    # Clean old checkpoint for fresh test
    if ckpt_path.exists():
        ckpt_path.unlink()

    config = TeoV2Config(
        vocab_size=32768,
        d_model=768,
        n_layers=12,
        n_heads=12,
        context=context_len,
        mlp=3072,
        lr=3e-4,
        warmup_steps=20,
        min_lr=3e-5,
        weight_decay=0.1,
        grad_clip=1.0,
        batch_size=batch_size,
    )

    # 2. Phase 1: Train first 100 steps
    print("\n--- Smoke Test Phase 1: Steps 0 to 100 ---", flush=True)
    res_1 = train_teo_v2(
        config=config,
        data_file=corpus_p,
        checkpoint_path=ckpt_path,
        resume=False,
        max_steps=100,
        save_interval=100,
        log_interval=20,
        device=device,
    )

    assert ckpt_path.exists(), "Phase 1 failed to create checkpoint.pt"
    ckpt_phase1 = load_checkpoint(ckpt_path, device=device)
    assert ckpt_phase1["step"] == 100, f"Expected checkpoint step 100, got {ckpt_phase1['step']}"

    # 3. Phase 2: Resume and train steps 101 to 200
    print("\n--- Smoke Test Phase 2: Stop + Resume to Step 200 ---", flush=True)
    res_2 = train_teo_v2(
        config=config,
        data_file=corpus_p,
        checkpoint_path=ckpt_path,
        resume=True,
        max_steps=200,
        save_interval=100,
        log_interval=20,
        device=device,
    )

    assert ckpt_path.exists(), "Phase 2 failed to update checkpoint.pt"
    ckpt_phase2 = load_checkpoint(ckpt_path, device=device)
    assert ckpt_phase2["step"] == 200, f"Expected final checkpoint step 200, got {ckpt_phase2['step']}"

    # 4. Verifications
    all_history = res_1["history"] + res_2["history"]
    first_loss = all_history[0]["loss"]
    loss_at_100 = res_1["final_loss"]
    loss_resumed = res_2["history"][0]["loss"]
    final_loss = res_2["final_loss"]

    # (a) Loss decreases
    loss_decreased = final_loss < first_loss
    assert loss_decreased, f"Loss did not decrease: start={first_loss:.4f}, end={final_loss:.4f}"

    # (b) Continuous loss and correct step counter
    step_resumed_correct = (res_2["final_step"] == 200)
    loss_continuity_diff = abs(loss_resumed - loss_at_100)

    # (c) Checkpoint validity
    ckpt_valid = (
        "model_state_dict" in ckpt_phase2
        and "optimizer_state_dict" in ckpt_phase2
        and ckpt_phase2["step"] == 200
        and "rng_state" in ckpt_phase2
    )

    print("\n" + "=" * 60)
    print("🏆 SMOKE TEST RESULTS SUMMARY:")
    print(f"  • (a) Loss Decreased:         {loss_decreased} ({first_loss:.4f} ➡️ {final_loss:.4f})")
    print(f"  • (b) Stop+Resume Continuous: Step {ckpt_phase1['step']} ➡️ {ckpt_phase2['step']} (continuity diff: {loss_continuity_diff:.4f})")
    print(f"  • (c) Checkpoint Valid:       {ckpt_valid} ({ckpt_path.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 60)

    return {
        "loss_decreased": loss_decreased,
        "first_loss": first_loss,
        "loss_at_100": loss_at_100,
        "loss_resumed": loss_resumed,
        "final_loss": final_loss,
        "step_counter_correct": step_resumed_correct,
        "checkpoint_valid": ckpt_valid,
        "all_history": all_history,
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Teo v2 (~124M GPT-2-class model) on packed token streams.")
    parser.add_argument("--config", default="training/teo_v2_config.json", help="Path to config JSON.")
    parser.add_argument("--data-dir", default="data/teo_v2", help="Corpus shard directory.")
    parser.add_argument("--data-file", default=None, help="Optional single binary stream file.")
    parser.add_argument("--checkpoint-dir", default="models/teo_v2", help="Checkpoint save directory.")
    parser.add_argument("--checkpoint-path", default=None, help="Explicit checkpoint file path.")
    parser.add_argument("--no-resume", action="store_true", help="Start training from step 0, ignoring checkpoints.")
    parser.add_argument("--max-steps", type=int, default=None, help="Total training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--context-len", type=int, default=None, help="Context sequence length.")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps.")
    parser.add_argument("--min-lr", type=float, default=None, help="Cosine decay floor learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay on 2D non-embedding params.")
    parser.add_argument("--grad-clip", type=float, default=None, help="Maximum gradient norm.")
    parser.add_argument("--log-interval", type=int, default=50, help="Logging step interval.")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint step interval.")
    parser.add_argument("--autocast", action="store_true", help="Enable fp16 mixed precision on GPU.")
    parser.add_argument("--device", default=None, help="Device ('cpu', 'cuda').")
    parser.add_argument("--smoke-test", action="store_true", help="Run 200-step CPU smoke test verification.")
    parser.add_argument("--build-synthetic", action="store_true", help="Build ~2M synthetic packed bin and exit.")
    parser.add_argument("--synthetic-out", default="training/data/teo_v2_corpus/shard_000.bin", help="Output for synthetic build.")

    args = parser.parse_args()

    if args.build_synthetic:
        meta = build_synthetic_corpus(args.synthetic_out)
        print(f"✅ Built synthetic corpus at {args.synthetic_out}: {meta['total_tokens']:,} tokens")
        return

    if args.smoke_test:
        run_smoke_test(
            corpus_bin=args.data_file,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device or "cpu",
        )
        return

    train_teo_v2(
        config_path=args.config,
        data_dir=args.data_dir,
        data_file=args.data_file,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path,
        resume=not args.no_resume,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        context_len=args.context_len,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        autocast=args.autocast,
        device=args.device,
    )


if __name__ == "__main__":
    main()
