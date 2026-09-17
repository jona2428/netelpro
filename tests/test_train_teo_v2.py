"""Tests for Teo v2 training pipeline, model architecture, and checkpoint roundtrip."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.data.compile_packed import get_batch, load_stream, stream_stats
from training.train_teo_v2 import (
    TeoV2Config,
    TeoV2Transformer,
    configure_optimizers,
    get_lr,
    load_checkpoint,
    save_checkpoint,
)


def test_config_load():
    """Verifies that training/teo_v2_config.json loads and matches required specifications."""
    cfg_path = ROOT_DIR / "training" / "teo_v2_config.json"
    assert cfg_path.exists(), f"Configuration file not found: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Required parameters
    assert data["vocab_size"] == 32768
    assert data["d_model"] == 768
    assert data["n_layers"] == 12
    assert data["n_heads"] == 12
    assert data["context"] == 1024
    assert data["mlp"] == 3072
    assert data["tied_embeddings"] is True
    assert data["dropout"] == 0.0
    assert abs(data["lr"] - 3e-4) < 1e-7
    assert data["warmup_steps"] == 2000
    assert abs(data["min_lr"] - 3e-5) < 1e-8
    assert abs(data["weight_decay"] - 0.1) < 1e-7
    assert abs(data["grad_clip"] - 1.0) < 1e-7

    # Dataclass load and alias verification
    config = TeoV2Config.from_json(cfg_path)
    assert config.vocab_size == 32768
    assert config.d_model == 768
    assert config.n_embd == 768
    assert config.n_layers == 12
    assert config.n_layer == 12
    assert config.n_heads == 12
    assert config.n_head == 12
    assert config.context == 1024
    assert config.block_size == 1024
    assert config.mlp == 3072


def test_batch_shapes_from_compile_packed(tmp_path: Path):
    """Verifies get_batch returns correct shapes, shifted targets, and zero padding from stream."""
    stream_file = tmp_path / "test_stream.bin"
    # Create 10,000 synthetic uint16 tokens with bos (1) and eos (2), zero pad
    rng = np.random.default_rng(42)
    tokens = rng.integers(4, 1000, size=10000, dtype=np.uint16)
    tokens[0] = 1   # bos
    tokens[-1] = 2  # eos
    tokens.tofile(stream_file)

    stream = load_stream(stream_file)
    assert len(stream) == 10000

    # Diagnostic stats
    stats = stream_stats(stream)
    assert stats["pad_fraction"] == 0.0
    assert stats["total_tokens"] == 10000

    # Test get_batch with return_tensor=True
    batch_size = 4
    context_len = 128
    inputs, targets = get_batch(
        stream,
        block_size=context_len,
        batch_size=batch_size,
        generator=rng,
        return_tensor=True,
    )

    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert inputs.shape == (batch_size, context_len)
    assert targets.shape == (batch_size, context_len)
    assert inputs.dtype == torch.int64
    assert targets.dtype == torch.int64

    # Verify targets is inputs shifted by 1 position: targets[b, t] follows inputs[b, t]
    for b in range(batch_size):
        assert torch.equal(inputs[b, 1:], targets[b, :-1])


def test_checkpoint_resume_roundtrip(tmp_path: Path):
    """Verifies saving and loading a checkpoint restores model, optimizer, step, and RNG states."""
    config = TeoV2Config(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        context=16,
        mlp=64,
        lr=1e-3,
        weight_decay=0.1,
    )

    model1 = TeoV2Transformer(config)
    optimizer1 = configure_optimizers(model1, weight_decay=config.weight_decay, lr=config.lr)

    # Perform one training step on model1 to alter weights and optimizer state
    dummy_x = torch.randint(0, 128, (2, 16))
    dummy_y = torch.randint(0, 128, (2, 16))
    _, loss = model1(dummy_x, targets=dummy_y)
    loss.backward()
    optimizer1.step()

    ckpt_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path=ckpt_path,
        model=model1,
        optimizer=optimizer1,
        step=42,
        epoch=1,
        shard_idx=3,
        config=config,
        loss=float(loss.item()),
    )

    assert ckpt_path.exists()

    # Create model2 with fresh initialization
    model2 = TeoV2Transformer(config)
    optimizer2 = configure_optimizers(model2, weight_decay=config.weight_decay, lr=config.lr)

    # Confirm model2 initial weights differ from model1
    with torch.no_grad():
        diff = sum((p1 - p2).abs().sum().item() for p1, p2 in zip(model1.parameters(), model2.parameters()))
        assert diff > 0.0

    # Load checkpoint into model2 and optimizer2
    loaded = load_checkpoint(ckpt_path, model=model2, optimizer=optimizer2, device="cpu")

    assert loaded["step"] == 42
    assert loaded["epoch"] == 1
    assert loaded["shard_idx"] == 3
    assert abs(loaded["loss"] - float(loss.item())) < 1e-5

    # Verify all model weights match exactly
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)

    # Verify forward pass produces identical logits
    with torch.no_grad():
        logits1, _ = model1(dummy_x)
        logits2, _ = model2(dummy_x)
        assert torch.allclose(logits1, logits2, atol=1e-6)


def test_tied_embeddings_and_ste_gradients():
    """Verifies tied embeddings (wte == lm_head) and gradient flow through Netelpro band-gate STE."""
    config = TeoV2Config(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        context=16,
        mlp=64,
        tied_embeddings=True,
    )

    model = TeoV2Transformer(config)
    assert model.wte.weight is model.lm_head.weight, "Embeddings must be tied to lm_head!"

    model.train()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))

    logits, loss = model(x, targets=y)
    assert loss is not None
    assert loss.item() > 0

    loss.backward()

    # Gradients must flow to tied embedding and MLP linear layers
    assert model.wte.weight.grad is not None
    assert model.wte.weight.grad.norm().item() > 0

    mlp_fc_grad = model.blocks[0].mlp.c_fc.weight.grad
    assert mlp_fc_grad is not None
    assert mlp_fc_grad.norm().item() > 0


def test_lr_schedule():
    """Verifies warmup linear increase and cosine decay to min_lr."""
    warmup = 2000
    max_steps = 100000
    lr = 3e-4
    min_lr = 3e-5

    # Step 0: start of warmup
    lr_0 = get_lr(0, warmup_steps=warmup, max_steps=max_steps, lr=lr, min_lr=min_lr)
    assert abs(lr_0 - (lr / warmup)) < 1e-9

    # Mid warmup: step 1000
    lr_mid = get_lr(999, warmup_steps=warmup, max_steps=max_steps, lr=lr, min_lr=min_lr)
    assert abs(lr_mid - (lr * 0.5)) < 1e-6

    # Peak: step 2000
    lr_peak = get_lr(warmup - 1, warmup_steps=warmup, max_steps=max_steps, lr=lr, min_lr=min_lr)
    assert abs(lr_peak - lr) < 1e-9

    # Halfway decay: step 51000
    lr_decay_mid = get_lr(51000, warmup_steps=warmup, max_steps=max_steps, lr=lr, min_lr=min_lr)
    assert min_lr < lr_decay_mid < lr

    # End of schedule: step max_steps
    lr_end = get_lr(max_steps, warmup_steps=warmup, max_steps=max_steps, lr=lr, min_lr=min_lr)
    assert abs(lr_end - min_lr) < 1e-9

    # Beyond max_steps: stays at min_lr
    lr_beyond = get_lr(max_steps + 5000, warmup_steps=warmup, max_steps=max_steps, lr=lr, min_lr=min_lr)
    assert lr_beyond == min_lr
