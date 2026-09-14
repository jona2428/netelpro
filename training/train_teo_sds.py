"""Teo-SDS (Netelpro Silicon Dynamic State, ~300-350M params) packed token stream trainer.

Fase 3 of the Netelpro roadmap: replaces the Teo v2 Transformer core with the
non-transformer Netelpro SDS engine (netelpro/neuro/dynamic_state.py) — zero
KV-Cache, O(1) inference memory via causal convolution + dynamic state recurrence.

Architecture (roadmap default):
  - 16 layers, d_model 1024, d_state 16, expand 2 (d_inner 2048), vocab 32768.
  - Tied embeddings (wte == lm_head), zero dropout.
  - Netelpro band-gate + Straight-Through Estimator (STE) semantics in the MLP sublayer.
  - AdamW optimizer (lr 3e-4, warmup 2000 steps, cosine decay to 3e-5, weight decay 0.1
    on non-embedding params, grad clip 1.0).
  - Multi-shard resume scanning training/data/teo_v2_massive/shard_*.bin + meta.json.
  - Checkpoints saved every 1000 steps to models/teo_sds/checkpoint.pt with full model,
    optimizer, step, shard index, and RNG states.
  - Contiguous packed uint16 token streams with zero padding tokens (same corpus format
    as Teo v2 — reuses training/data/compile_packed.py loaders).

Unlike train_teo_v2.py, NetelproSDSModel.forward returns (logits, loss, AuditCertificate);
the certificate is ignored during training (it exists for inference-time audit trails).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from netelpro.neuro.ste import HAS_TORCH
from netelpro.neuro.dynamic_state import NetelproSDSConfig, NetelproSDSModel
from training.data.compile_packed import get_batch, load_stream
from training.train_teo_v2 import build_synthetic_corpus, discover_shards, get_lr

if HAS_TORCH:
    import torch
    import torch.nn as nn
else:
    raise RuntimeError("PyTorch 2.x is required for teo_sds training.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TeoSDSConfig:
    """Hyperparameter configuration for Teo-SDS (~300-350M params)."""

    # Architecture (-> netelpro.neuro.dynamic_state.NetelproSDSConfig)
    vocab_size: int = 32768
    n_layer: int = 16
    d_model: int = 1024
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    scale_factor: float = 1000.0
    z_min: int = -5000
    z_max: int = 5000
    rule_path: str | None = None
    dropout: float = 0.0
    activation_fn: str = "relu"

    # Training-only (sequence chunk length for parallel training; the SDS model
    # itself has no hard context limit — inference is O(1) recurrent stepping).
    context: int = 512

    # Optimizer & schedule
    lr: float = 3e-4
    warmup_steps: int = 2000
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Training runtime defaults
    batch_size: int = 8
    log_interval: int = 50
    save_interval: int = 1000
    max_steps: int = 100000

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    def to_sds_config(self) -> NetelproSDSConfig:
        return NetelproSDSConfig(
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            scale_factor=self.scale_factor,
            z_min=self.z_min,
            z_max=self.z_max,
            rule_path=self.rule_path,
            dropout=self.dropout,
            activation_fn=self.activation_fn,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeoSDSConfig:
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_path: str | Path) -> TeoSDSConfig:
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
# Optimizer
# ---------------------------------------------------------------------------


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    """Configures AdamW with weight decay 0.1 on non-embedding 2D weights.

    The token embedding (wte, tied to lm_head) and 1D parameters (biases,
    LayerNorms, SDS gate/conv vectors) receive weight decay 0.0.
    """
    decay_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and "wte" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = (device_type == "cuda") and hasattr(torch.optim.AdamW, "fused")
    return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=1e-8, fused=use_fused)


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
    config: TeoSDSConfig,
    loss: float,
) -> Path:
    """Atomically saves training checkpoint."""
    p = Path(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = p.with_suffix(".pt.tmp")

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

    rng = ckpt.get("rng_state")
    if isinstance(rng, dict):
        if rng.get("torch") is not None:
            try:
                torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
            except Exception:
                pass
        if rng.get("numpy") is not None:
            try:
                np.random.set_state(rng["numpy"])
            except Exception:
                pass
        if rng.get("python") is not None:
            try:
                random.setstate(rng["python"])
            except Exception:
                pass

    return ckpt


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def train_teo_sds(
    config: TeoSDSConfig | None = None,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data/teo_v2_massive",
    data_file: str | Path | None = None,
    checkpoint_dir: str | Path = "models/teo_sds",
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
    device: str | None = None,
    seed: int = 42,
    num_threads: int | None = None,
) -> dict[str, Any]:
    """Multi-shard trainer for Teo-SDS with checkpoint resumption and token-throughput logging."""
    # 1. Config resolution
    if config is None:
        cfg_file = Path(config_path) if config_path else Path(__file__).parent / "teo_sds_config.json"
        config = TeoSDSConfig.from_json(cfg_file) if cfg_file.exists() else TeoSDSConfig()

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

    # 2. Hardware setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        threads = num_threads or min(8, os.cpu_count() or 4)
        torch.set_num_threads(threads)

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    # 3. Model & Optimizer
    model = NetelproSDSModel(config.to_sds_config()).to(device)
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
            ckpt = load_checkpoint(ckpt_file, model=model, optimizer=optimizer, device=device)
            start_step = ckpt["step"]
            start_epoch = ckpt.get("epoch", 0)
            start_shard_idx = ckpt.get("shard_idx", 0)
            last_loss = ckpt.get("loss", 0.0)
            print(f"   Resumed at step {start_step}, epoch {start_epoch}, shard {start_shard_idx}, loss: {last_loss:.4f}")
        except Exception as e:
            print(f"⚠️ Checkpoint load failed, starting fresh: {e}")

    # 5. Shard discovery (reuses Teo v2 shard scanner — same packed uint16 format)
    if data_file:
        shards = [Path(data_file)]
    else:
        shards, _meta = discover_shards(data_dir)

    if not shards:
        raise FileNotFoundError(f"No corpus shards found at {data_file or data_dir}")

    total_shards = len(shards)
    print(f"📂 Discovered {total_shards} corpus shard(s): {[s.name for s in shards[:5]]}")

    # 6. Training loop
    model.train()
    step = start_step
    epoch = start_epoch
    curr_shard_idx = start_shard_idx % total_shards

    stream = load_stream(shards[curr_shard_idx])
    assert len(stream) > config.context + 1, (
        f"Shard {shards[curr_shard_idx]} with {len(stream)} tokens too small for context {config.context}"
    )

    history: list[dict[str, Any]] = []
    tokens_per_step = config.batch_size * config.context
    t_log_start = time.perf_counter()
    tokens_in_window = 0

    param_count = sum(p.numel() for p in model.parameters())

    print("=" * 80)
    print(f"🚀 Training Teo-SDS (~{param_count / 1e6:.1f}M params) on {device.upper()}")
    print(f"   Layers: {config.n_layer} | d_model: {config.d_model} | d_state: {config.d_state} | d_inner: {config.d_inner}")
    print(f"   Context: {config.context} | Batch: {config.batch_size} | LR: {config.lr:.2e} -> {config.min_lr:.2e}")
    print(f"   Warmup: {config.warmup_steps} steps | Save Interval: {save_interval} | Zero KV-Cache (O(1) inference)")
    print("=" * 80)

    while step < config.max_steps:
        t_step_start = time.perf_counter()

        current_lr = get_lr(
            step,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            lr=config.lr,
            min_lr=config.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

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

        _, loss, _certificate = model(inputs, targets=targets, control_flags=1)
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
            )
            print(f"💾 Checkpoint saved at step {step}: {ckpt_file}", flush=True)

    print("=" * 80)
    print(f"✅ Teo-SDS training completed at step {step} with final loss: {last_loss:.4f}")
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
    checkpoint_dir: str | Path = "models/teo_sds",
    context_len: int = 64,
    batch_size: int = 2,
    device: str = "cpu",
) -> dict[str, Any]:
    """Mandatory CPU smoke test:

    1. Builds a tiny synthetic packed bin (~2M random token IDs, bos/eos framing, zero pad).
    2. Trains 100 steps from step 0, saves checkpoint at step 100.
    3. Stops and resumes from step 100 to step 200.
    4. Verifies loss decreases and stop+resume is continuous with a valid checkpoint.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pt"

    if corpus_bin is None:
        corpus_bin = Path("training/data/teo_sds_corpus/smoke_synth.bin")
    corpus_p = Path(corpus_bin)
    if not corpus_p.exists() or corpus_p.stat().st_size < 2_000_000 * 2:
        print(f"🔨 Building synthetic packed corpus ({corpus_p})...", flush=True)
        build_synthetic_corpus(corpus_p, num_tokens=2_000_000, vocab_size=32768)

    if ckpt_path.exists():
        ckpt_path.unlink()

    config = TeoSDSConfig(
        vocab_size=32768,
        n_layer=4,
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        context=context_len,
        lr=3e-4,
        warmup_steps=20,
        min_lr=3e-5,
        weight_decay=0.1,
        grad_clip=1.0,
        batch_size=batch_size,
    )

    print("\n--- Smoke Test Phase 1: Steps 0 to 100 ---", flush=True)
    res_1 = train_teo_sds(
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

    print("\n--- Smoke Test Phase 2: Stop + Resume to Step 200 ---", flush=True)
    res_2 = train_teo_sds(
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

    all_history = res_1["history"] + res_2["history"]
    first_loss = all_history[0]["loss"]
    final_loss = res_2["final_loss"]
    loss_decreased = final_loss < first_loss
    assert loss_decreased, f"Loss did not decrease: start={first_loss:.4f}, end={final_loss:.4f}"

    ckpt_valid = (
        "model_state_dict" in ckpt_phase2
        and "optimizer_state_dict" in ckpt_phase2
        and ckpt_phase2["step"] == 200
        and "rng_state" in ckpt_phase2
    )

    print("\n" + "=" * 60)
    print("🏆 SMOKE TEST RESULTS SUMMARY:")
    print(f"  • Loss Decreased:         {loss_decreased} ({first_loss:.4f} ➡️ {final_loss:.4f})")
    print(f"  • Stop+Resume Continuous: Step {ckpt_phase1['step']} ➡️ {ckpt_phase2['step']}")
    print(f"  • Checkpoint Valid:       {ckpt_valid} ({ckpt_path.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 60)

    return {
        "loss_decreased": loss_decreased,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "checkpoint_valid": ckpt_valid,
        "all_history": all_history,
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Teo-SDS (~300-350M Netelpro Silicon Dynamic State model).")
    parser.add_argument("--config", default="training/teo_sds_config.json", help="Path to config JSON.")
    parser.add_argument("--data-dir", default="data/teo_v2_massive", help="Corpus shard directory.")
    parser.add_argument("--data-file", default=None, help="Optional single binary stream file.")
    parser.add_argument("--checkpoint-dir", default="models/teo_sds", help="Checkpoint save directory.")
    parser.add_argument("--checkpoint-path", default=None, help="Explicit checkpoint file path.")
    parser.add_argument("--no-resume", action="store_true", help="Start training from step 0, ignoring checkpoints.")
    parser.add_argument("--max-steps", type=int, default=None, help="Total training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--context-len", type=int, default=None, help="Training sequence chunk length.")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps.")
    parser.add_argument("--min-lr", type=float, default=None, help="Cosine decay floor learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay on 2D non-embedding params.")
    parser.add_argument("--grad-clip", type=float, default=None, help="Maximum gradient norm.")
    parser.add_argument("--log-interval", type=int, default=50, help="Logging step interval.")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint step interval.")
    parser.add_argument("--device", default=None, help="Device ('cpu', 'cuda').")
    parser.add_argument("--smoke-test", action="store_true", help="Run 200-step CPU smoke test verification.")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(
            corpus_bin=args.data_file,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device or "cpu",
        )
        return

    train_teo_sds(
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
        device=args.device,
    )


if __name__ == "__main__":
    main()
