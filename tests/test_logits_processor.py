"""Tests for NetelproLogitsProcessor (Hugging Face / PyTorch autoregressive masking)."""

from __future__ import annotations

import torch

from netelpro.neuro import NetelproLogitsProcessor


def test_logits_processor_prunes_out_of_bound_tokens():
    processor = NetelproLogitsProcessor(
        allowed_min=10,
        allowed_max=15,
        safety_state=1,
    )

    batch_size = 2
    vocab_size = 20
    # Create fake logits
    scores = torch.zeros(batch_size, vocab_size)

    input_ids = torch.zeros(batch_size, 5, dtype=torch.long)
    masked_scores = processor(input_ids, scores)

    # Tokens < 10 or > 15 must be -inf
    for token_id in range(vocab_size):
        for b in range(batch_size):
            val = masked_scores[b, token_id].item()
            if 10 <= token_id <= 15:
                assert val == 0.0, f"Token {token_id} in allowed range was unexpectedly masked"
            else:
                assert val == float("-inf"), f"Token {token_id} outside allowed range was not masked"

    assert processor.total_steps == 1
    assert processor.last_audit["pruned_count"] == (vocab_size - 6)


def test_logits_processor_dynamic_context_update():
    processor = NetelproLogitsProcessor(
        allowed_min=0,
        allowed_max=10,
        safety_state=1,
    )

    scores = torch.zeros(1, 10)
    input_ids = torch.zeros(1, 1, dtype=torch.long)

    # Step 1: Normal
    out1 = processor(input_ids, scores.clone())
    assert out1[0, 5].item() == 0.0

    # Step 2: Emergency lockdown
    processor.set_context(allowed_min=0, allowed_max=10, safety_state=0)
    out2 = processor(input_ids, scores.clone())
    assert all(out2[0, i].item() == float("-inf") for i in range(10))
