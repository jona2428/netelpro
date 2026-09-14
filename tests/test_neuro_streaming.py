"""Tests for Netelpro Real-Time Streaming Inference Engine."""

from __future__ import annotations

import pytest

from netelpro.neuro.stream import NetelproStreamProcessor, stream_generate
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch


def test_stream_processor_initialization():
    sp = NetelproStreamProcessor(allowed_min=10, allowed_max=20, safety_state=1)
    assert sp.processor.allowed_min == 10
    assert sp.processor.allowed_max == 20
    assert sp.token_count == 0
    assert len(sp.history) == 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_stream_processor_hf_logits():
    sp = NetelproStreamProcessor(allowed_min=2, allowed_max=5, safety_state=1)
    vocab_size = 10
    input_ids = torch.tensor([[1, 2]])
    scores = torch.zeros((1, vocab_size))

    out_scores = sp.process_hf_logits(input_ids, scores)
    assert sp.token_count == 1
    assert len(sp.history) == 1
    assert sp.history[0]["step"] == 1
    assert sp.history[0]["latency_us"] > 0

    # Tokens 2..5 should remain 0.0, tokens 0, 1, 6..9 should be -inf
    for t in range(vocab_size):
        if 2 <= t <= 5:
            assert out_scores[0, t] == 0.0
        else:
            assert out_scores[0, t] == float("-inf")


def test_stream_processor_llama_cpp_callback():
    sp = NetelproStreamProcessor(allowed_min=1, allowed_max=3, safety_state=1)
    input_ids = [100, 101]
    scores = [0.0] * 6

    out_scores = sp.llama_cpp_processor(input_ids, scores)
    assert sp.token_count == 1
    assert out_scores[0] == float("-inf")
    assert out_scores[1] == 0.0
    assert out_scores[2] == 0.0
    assert out_scores[3] == 0.0
    assert out_scores[4] == float("-inf")
    assert out_scores[5] == float("-inf")


def test_stream_processor_dynamic_context():
    sp = NetelproStreamProcessor(allowed_min=0, allowed_max=10, safety_state=1)
    scores1 = [0.0] * 5
    sp.llama_cpp_processor([1], scores1)
    assert all(s == 0.0 for s in scores1)

    # Inhibit completely
    sp.set_context(allowed_min=0, allowed_max=10, safety_state=0)
    scores2 = [0.0] * 5
    sp.llama_cpp_processor([1], scores2)
    assert all(s == float("-inf") for s in scores2)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_stream_generate_loop_torch():
    vocab_size = 10
    sp = NetelproStreamProcessor(allowed_min=3, allowed_max=4, safety_state=1)

    def dummy_logits_fn(context_tokens: list[int]) -> torch.Tensor:
        # Give higher preference to token 3
        scores = torch.zeros(vocab_size)
        scores[3] = 10.0
        scores[7] = 50.0  # forbidden token with high logit
        return scores

    generated_tokens = []
    for token_id, token_str, audit in stream_generate(
        logits_fn=dummy_logits_fn,
        stream_processor=sp,
        initial_tokens=[1],
        max_tokens=4,
        eos_token_id=0,
        temperature=0.0,
    ):
        generated_tokens.append(token_id)
        assert token_id in (3, 4)  # Must never select token 7 despite high logit
        assert audit["latency_us"] > 0

    assert len(generated_tokens) == 4
    summary = sp.get_summary()
    assert summary["total_tokens_processed"] == 4
    assert summary["avg_pruning_latency_us"] < 5000.0  # sub-millisecond execution


def test_stream_generate_loop_native_list():
    sp = NetelproStreamProcessor(allowed_min=2, allowed_max=2, safety_state=1)

    def dummy_logits_fn(context_tokens: list[int]) -> list[float]:
        # Token 2 is allowed; token 5 is forbidden but higher raw score
        scores = [0.0] * 6
        scores[2] = 5.0
        scores[5] = 99.0
        return scores

    tokens = []
    for token_id, token_str, audit in stream_generate(
        logits_fn=dummy_logits_fn,
        stream_processor=sp,
        initial_tokens=[1],
        max_tokens=3,
        eos_token_id=0,
    ):
        tokens.append(token_id)
        assert token_id == 2

    assert len(tokens) == 3
