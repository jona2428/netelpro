"""Unit tests for the Netelpro Mini LLM and Tokenizer."""

from __future__ import annotations

import pytest

from netelpro.neuro.minillm import NetelproMiniLLM
from netelpro.neuro.tokenizer import NetelproTokenizer
from netelpro.neuro.transformer import NetelproTransformerConfig
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch


def test_tokenizer_encode_decode_roundtrip():
    tok = NetelproTokenizer()
    text = "Hola mundo, esto es Netelpro 2026. <|thought|> prueba <|endthought|>"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_tokenizer_chat_formatting():
    tok = NetelproTokenizer()
    messages = [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Hola mundo"},
    ]
    formatted = tok.format_chat(messages, add_generation_prompt=True)
    assert "<|user|>\nHola\n" in formatted
    assert "<|assistant|>\nHola mundo<|eos|>\n" in formatted
    assert formatted.endswith("<|assistant|>\n")


def test_tokenizer_save_and_load(tmp_path):
    tok = NetelproTokenizer()
    save_file = tmp_path / "vocab.json"
    tok.save(save_file)
    assert save_file.exists()

    loaded_tok = NetelproTokenizer.load(save_file)
    assert loaded_tok.vocab_size == tok.vocab_size
    assert loaded_tok.encode("Netelpro") == tok.encode("Netelpro")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_minillm_initialization():
    tok = NetelproTokenizer()
    cfg = NetelproTransformerConfig(
        vocab_size=tok.vocab_size,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    model = NetelproMiniLLM(config=cfg, tokenizer=tok)
    assert model.config.vocab_size == tok.vocab_size
    assert len(model.model.blocks) == 2


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_minillm_forward_and_loss():
    tok = NetelproTokenizer()
    cfg = NetelproTransformerConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    model = NetelproMiniLLM(config=cfg, tokenizer=tok)
    x = torch.randint(0, tok.vocab_size, (2, 8))
    y = torch.randint(0, tok.vocab_size, (2, 8))

    logits, loss, cert = model(x, targets=y)
    assert logits.shape == (2, 8, tok.vocab_size)
    assert loss is not None
    assert loss.item() > 0
    assert cert is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_minillm_generation_and_streaming():
    tok = NetelproTokenizer()
    cfg = NetelproTransformerConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=16,
    )
    model = NetelproMiniLLM(config=cfg, tokenizer=tok)

    # 1. generate_text
    prompt = "<|user|>\nHola\n<|assistant|>\n"
    text, cert = model.generate_text(prompt, max_new_tokens=5, temperature=0.8)
    assert isinstance(text, str)
    assert cert.total_latency_us > 0

    # 2. stream_chat
    tokens_streamed = []
    audits_streamed = []
    for t_str, audit in model.stream_chat(prompt, max_new_tokens=4, temperature=0.8):
        tokens_streamed.append(t_str)
        audits_streamed.append(audit)

    assert len(tokens_streamed) == 4
    assert len(audits_streamed) == 4
    assert "active_neurons" in audits_streamed[0]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_minillm_save_and_load_pretrained(tmp_path):
    tok = NetelproTokenizer()
    cfg = NetelproTransformerConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=16,
    )
    model = NetelproMiniLLM(config=cfg, tokenizer=tok)
    save_dir = tmp_path / "minillm_ckpt"
    model.save_pretrained(save_dir)

    assert (save_dir / "config.json").exists()
    assert (save_dir / "vocab.json").exists()
    assert (save_dir / "model.pt").exists()

    loaded = NetelproMiniLLM.from_pretrained(save_dir)
    assert loaded.config.vocab_size == model.config.vocab_size
    assert loaded.config.n_embd == model.config.n_embd

    # Test inference on loaded model
    out_text, cert = loaded.generate_text("Test", max_new_tokens=3)
    assert isinstance(out_text, str)
    assert cert is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_minillm_fail_closed_inhibition():
    tok = NetelproTokenizer()
    cfg = NetelproTransformerConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=16,
    )
    model = NetelproMiniLLM(config=cfg, tokenizer=tok)

    # Inhibit all neurons in silicio
    prompt = "Test"
    _, cert = model.generate_text(prompt, max_new_tokens=2, control_flags=0)
    for r in cert.records:
        assert r.active_neurons == 0
        assert r.suppressed_neurons == r.total_neurons
