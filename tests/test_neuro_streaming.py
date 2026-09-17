"""Tests for Netelpro Real-Time Streaming Inference Engine."""

from __future__ import annotations

import pytest

from netelpro.neuro.ste import HAS_TORCH
from netelpro.neuro.stream import NetelproStreamProcessor, stream_generate

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


# ---------------------------------------------------------------------------
# Regression coverage for the vectorized llama_cpp_processor rewrite
# (netelpro/neuro/stream.py) -- the original per-token Python loop over a
# 152k-token vocab measured at ~370ms/token overhead against a real
# llama-cpp-python model; the vectorized fast path brings that to ~3us/token.
# Existing tests above only ever used plain Python lists at vocab_size<=6,
# which never exercised numpy (llama-cpp-python's actual LogitsProcessorList
# signature is `(input_ids: np.intc[], scores: np.single[]) -> np.single[]`)
# or a vocab large enough for the min/max boundary slicing to matter.
# ---------------------------------------------------------------------------


def test_llama_cpp_processor_numpy_array_wide_range():
    """llama-cpp-python passes numpy arrays, not lists -- the vectorized
    slice-assign path must work on the real type, not just lists."""
    np = pytest.importorskip("numpy")
    vocab_size = 1000
    sp = NetelproStreamProcessor(allowed_min=100, allowed_max=200, safety_state=1)
    scores = np.zeros(vocab_size, dtype=np.float32)

    out = sp.llama_cpp_processor(np.array([1, 2], dtype=np.intc), scores)

    assert out[99] == float("-inf")
    assert out[100] == 0.0
    assert out[200] == 0.0
    assert out[201] == float("-inf")
    # Every boundary, not just a handful of samples -- this is exactly the
    # off-by-one surface the slicing rewrite could get wrong.
    for t in range(vocab_size):
        expected = 0.0 if 100 <= t <= 200 else float("-inf")
        assert out[t] == expected, f"token {t}: expected {expected}, got {out[t]}"


def test_llama_cpp_processor_numpy_emergency_freeze():
    np = pytest.importorskip("numpy")
    vocab_size = 500
    sp = NetelproStreamProcessor(allowed_min=0, allowed_max=vocab_size - 1, safety_state=0)
    scores = np.zeros(vocab_size, dtype=np.float32)

    out = sp.llama_cpp_processor(np.array([1], dtype=np.intc), scores)

    assert bool(np.all(np.isneginf(out)))


def test_llama_cpp_processor_reports_total_pruned():
    """Regression: get_summary()['total_pruned_tokens'] reads
    processor.total_pruned, which only the HF __call__ path used to update.
    The llama.cpp adapter bypasses that path entirely (mutates scores
    directly), so it was silently always 0 through this route until fixed."""
    sp = NetelproStreamProcessor(allowed_min=1, allowed_max=3, safety_state=1)
    sp.llama_cpp_processor([1], [0.0] * 6)  # 3 of 6 tokens pruned

    summary = sp.get_summary()
    assert summary["total_pruned_tokens"] == 3


def test_llama_cpp_processor_boundary_consistent_with_native_gate():
    """The vectorized fast path assumes the default rule IS a contiguous
    [min, max] + safety_state check -- verify it agrees with the actual
    compiled gate at every boundary for a representative range, not just
    that it "looks fast"."""
    sp = NetelproStreamProcessor(allowed_min=7, allowed_max=7, safety_state=1)
    vocab_size = 20
    scores = [0.0] * vocab_size

    out = sp.llama_cpp_processor([1], scores)

    for t in range(vocab_size):
        allow, _ = sp.gate.check(t, 7, 7, 1)
        expected = 0.0 if allow else float("-inf")
        assert out[t] == expected, f"token {t}: vectorized path disagrees with compiled gate"


# ---------------------------------------------------------------------------
# Regression coverage for the discrete token_to_action_map fast path
# (docs/GATE_KERNEL_FUSION_SPEC.md Section 11) -- the original code called
# the compiled gate once per vocab token (a Python loop of native ctypes
# calls, the documented remaining bottleneck after the contiguous-range
# fix above). The rewrite evaluates the gate once per UNIQUE action and
# expands via a cached index -- nothing exercised this branch at all
# before this change, so every test below is new coverage, not a
# regression check against a prior test.
# ---------------------------------------------------------------------------

_DISCRETE_MAP = {0: 100, 1: 100, 2: 100, 5: 200, 6: 200, 10: 300}
_DISCRETE_VOCAB = 20
_DISCRETE_ALLOWED_MIN, _DISCRETE_ALLOWED_MAX = 100, 250
# Expected: tokens 0,1,2 (action 100) and 5,6 (action 200) allowed; token 10
# (action 300) denied; every other token defaults to action_id == token_id
# (all < 20, well below allowed_min=100) -- denied.
_DISCRETE_EXPECTED_ALLOWED = {0, 1, 2, 5, 6}


def _brute_force_allowed(sp: NetelproStreamProcessor, token_map: dict[int, int], vocab_size: int) -> set[int]:
    allowed = set()
    for t in range(vocab_size):
        action_id = token_map.get(t, t)
        allow, _ = sp.gate.check(action_id, _DISCRETE_ALLOWED_MIN, _DISCRETE_ALLOWED_MAX, 1)
        if allow:
            allowed.add(t)
    return allowed


def test_discrete_action_map_brute_force_sanity():
    """The hand-picked expected set above must actually match what the
    compiled gate says -- a sanity check on the fixture itself, not the
    fast path, so a wrong fixture can't make the fast-path tests below
    pass for the wrong reason."""
    sp = NetelproStreamProcessor(
        allowed_min=_DISCRETE_ALLOWED_MIN, allowed_max=_DISCRETE_ALLOWED_MAX, safety_state=1,
        token_to_action_map=_DISCRETE_MAP,
    )
    assert _brute_force_allowed(sp, _DISCRETE_MAP, _DISCRETE_VOCAB) == _DISCRETE_EXPECTED_ALLOWED


def test_llama_cpp_processor_discrete_action_map_plain_list():
    sp = NetelproStreamProcessor(
        allowed_min=_DISCRETE_ALLOWED_MIN, allowed_max=_DISCRETE_ALLOWED_MAX, safety_state=1,
        token_to_action_map=_DISCRETE_MAP,
    )
    scores = [0.0] * _DISCRETE_VOCAB
    out = sp.llama_cpp_processor([1], scores)

    for t in range(_DISCRETE_VOCAB):
        expected = 0.0 if t in _DISCRETE_EXPECTED_ALLOWED else float("-inf")
        assert out[t] == expected, f"token {t}: expected {expected}, got {out[t]}"

    summary = sp.get_summary()
    assert summary["total_pruned_tokens"] == _DISCRETE_VOCAB - len(_DISCRETE_EXPECTED_ALLOWED)


def test_llama_cpp_processor_discrete_action_map_numpy():
    np = pytest.importorskip("numpy")
    sp = NetelproStreamProcessor(
        allowed_min=_DISCRETE_ALLOWED_MIN, allowed_max=_DISCRETE_ALLOWED_MAX, safety_state=1,
        token_to_action_map=_DISCRETE_MAP,
    )
    scores = np.zeros(_DISCRETE_VOCAB, dtype=np.float32)
    out = sp.llama_cpp_processor(np.array([1], dtype=np.intc), scores)

    for t in range(_DISCRETE_VOCAB):
        expected = 0.0 if t in _DISCRETE_EXPECTED_ALLOWED else float("-inf")
        assert out[t] == expected, f"token {t}: expected {expected}, got {out[t]}"


def test_llama_cpp_processor_discrete_action_map_no_numpy_fallback(monkeypatch):
    """Forces the plain-Python fallback branch (no native calls per token,
    but no numpy vectorization either) -- must produce identical output to
    the numpy-accelerated path, just without the extra speedup."""
    import netelpro.neuro.stream as stream_mod

    monkeypatch.setattr(stream_mod, "HAS_NUMPY", False)

    sp = NetelproStreamProcessor(
        allowed_min=_DISCRETE_ALLOWED_MIN, allowed_max=_DISCRETE_ALLOWED_MAX, safety_state=1,
        token_to_action_map=_DISCRETE_MAP,
    )
    scores = [0.0] * _DISCRETE_VOCAB
    out = sp.llama_cpp_processor([1], scores)

    for t in range(_DISCRETE_VOCAB):
        expected = 0.0 if t in _DISCRETE_EXPECTED_ALLOWED else float("-inf")
        assert out[t] == expected, f"token {t}: expected {expected}, got {out[t]}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_logits_processor_hf_discrete_action_map_matches_llama_cpp_path():
    """Both adapters (HF __call__ and llama_cpp_processor) share the same
    NetelproLogitsProcessor instance and its cached decomposition -- they
    must agree on every token, not just each look internally consistent."""
    sp = NetelproStreamProcessor(
        allowed_min=_DISCRETE_ALLOWED_MIN, allowed_max=_DISCRETE_ALLOWED_MAX, safety_state=1,
        token_to_action_map=_DISCRETE_MAP,
    )
    input_ids = torch.tensor([[1, 2]])
    scores = torch.zeros((1, _DISCRETE_VOCAB))

    out = sp.process_hf_logits(input_ids, scores)

    for t in range(_DISCRETE_VOCAB):
        expected = 0.0 if t in _DISCRETE_EXPECTED_ALLOWED else float("-inf")
        assert out[0, t].item() == expected, f"token {t}: expected {expected}, got {out[0, t].item()}"


def test_discrete_action_map_reacts_to_set_context():
    """The (unique_actions, token_to_unique_index) decomposition is cached
    per vocab_size and must stay valid, but the allow/deny decision for
    each unique action must still react to a changed allowed range --
    only the expensive per-vocab-token work is cached, not the decision
    itself."""
    sp = NetelproStreamProcessor(
        allowed_min=_DISCRETE_ALLOWED_MIN, allowed_max=_DISCRETE_ALLOWED_MAX, safety_state=1,
        token_to_action_map=_DISCRETE_MAP,
    )
    scores1 = [0.0] * _DISCRETE_VOCAB
    sp.llama_cpp_processor([1], scores1)
    assert scores1[0] == 0.0  # action 100 allowed under the initial range

    sp.set_context(allowed_min=300, allowed_max=300, safety_state=1)
    scores2 = [0.0] * _DISCRETE_VOCAB
    sp.llama_cpp_processor([1], scores2)
    assert scores2[0] == float("-inf")  # action 100 no longer in range
    assert scores2[10] == 0.0  # action 300 now allowed


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
