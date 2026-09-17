"""Netelpro Real-Time Streaming Inference Engine.

Integrates compiled Netelpro formal gates into real-time token streaming pipelines,
enabling microsecond-level pruning of illegal next-token candidates across
HuggingFace Transformers and llama.cpp/GGUF engines without interrupting user-facing latency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Sequence

from netelpro.gate import Gate
from netelpro.neuro.logits_processor import NetelproLogitsProcessor
from netelpro.neuro.native_kernel import NetelproVectorKernel
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _fill_range(scores: Any, value: float, start: int, stop: int) -> None:
    """Set scores[start:stop] = value for numpy arrays, plain lists, or any
    other __setitem__-supporting sequence -- whichever `scores` turns out to
    be at runtime. numpy supports a scalar broadcast slice assign directly;
    a plain list needs a same-length fill; anything else falls back to a
    per-element loop (still O(range), just not a 152k-iteration one)."""
    if stop <= start:
        return
    try:
        scores[start:stop] = value
    except (TypeError, ValueError):
        try:
            scores[start:stop] = [value] * (stop - start)
        except TypeError:
            for i in range(start, stop):
                scores[i] = value


class NetelproStreamProcessor:
    """Universal real-time streaming processor for neuro-symbolic gating.

    Provides adapters for:
    1. HuggingFace `generate(..., logits_processor=[...], streamer=...)`.
    2. `llama-cpp-python` / GGUF logits_processor callbacks: `(input_ids, scores) -> scores`.
    3. Standalone step-by-step decoding loop with per-token microsecond audit trails.
    """

    def __init__(
        self,
        rule_path: str | Path | None = None,
        allowed_min: int = 0,
        allowed_max: int = 1000000,
        safety_state: int = 1,
        token_to_action_map: Optional[dict[int, int]] = None,
        mask_value: float = float("-inf"),
    ) -> None:
        self.processor = NetelproLogitsProcessor(
            rule_path=rule_path,
            allowed_min=allowed_min,
            allowed_max=allowed_max,
            safety_state=safety_state,
            token_to_action_map=token_to_action_map,
            mask_value=mask_value,
        )
        self.history: list[dict[str, Any]] = []
        self.total_stream_latency_ns: int = 0
        self.token_count: int = 0

    @property
    def gate(self) -> Gate:
        return self.processor.gate

    @property
    def kernel(self) -> NetelproVectorKernel:
        return self.processor.kernel

    def set_context(self, allowed_min: int, allowed_max: int, safety_state: int = 1) -> None:
        """Dynamically update safety boundaries or state during streaming."""
        self.processor.set_context(allowed_min, allowed_max, safety_state)

    def process_hf_logits(self, input_ids: Any, scores: Any) -> Any:
        """Adapter for HuggingFace `LogitsProcessor`."""
        t0 = time.perf_counter_ns()
        pruned_scores = self.processor(input_ids, scores)
        dt_ns = time.perf_counter_ns() - t0

        self.total_stream_latency_ns += dt_ns
        self.token_count += 1
        record = {
            "step": self.token_count,
            "latency_us": dt_ns / 1_000.0,
            "pruned_count": self.processor.last_audit.get("pruned_count", 0),
            "safety_state": self.processor.safety_state,
        }
        self.history.append(record)
        return pruned_scores

    def llama_cpp_processor(self, input_ids: Sequence[int], scores: Any) -> Any:
        """Adapter for llama.cpp / llama-cpp-python logits_processor signature:

        logits_processor(input_ids: Sequence[int], scores: list[float] | np.ndarray)

        Vectorized for the default contiguous-range contract: measured at
        ~370ms/token overhead on a 152k-token vocab before this fix (a plain
        Python loop calling the native gate once per candidate token, every
        decoding step -- the exact bottleneck this file's own docstring
        promises "without interrupting user-facing latency"). Mirrors
        NetelproVectorKernel.filter_logits_tensor's torch slicing, for
        numpy arrays / plain lists instead of tensors.
        """
        t0 = time.perf_counter_ns()
        vocab_size = len(scores)
        allowed_min = self.processor.allowed_min
        allowed_max = self.processor.allowed_max
        safety_state = self.processor.safety_state
        mask_value = self.processor.mask_value
        token_to_action_map = self.processor.token_to_action_map

        pruned_count = 0
        if safety_state == 0:
            # Emergency freeze: every candidate denied. No need to consult
            # the compiled rule per token when the answer is "everything" --
            # same short-circuit NetelproVectorKernel takes.
            _fill_range(scores, mask_value, 0, vocab_size)
            pruned_count = vocab_size
        elif token_to_action_map is None:
            # Only valid when the loaded rule IS a [allowed_min, allowed_max]
            # + safety_state contract (true for the default
            # action_boundary.sl). A custom rule_path with different
            # semantics needs the per-token path below -- same limitation
            # NetelproVectorKernel.filter_logits_tensor already has for the
            # torch path; this mirrors it rather than introducing a new one.
            lo = max(0, allowed_min)
            hi = min(vocab_size - 1, allowed_max)
            if lo > 0:
                _fill_range(scores, mask_value, 0, lo)
                pruned_count += lo
            if hi + 1 < vocab_size:
                _fill_range(scores, mask_value, hi + 1, vocab_size)
                pruned_count += vocab_size - (hi + 1)
        else:
            # Discrete action map: the compiled gate is evaluated once per
            # UNIQUE action reachable from the map (cached on self.processor,
            # shared with the HF __call__ path), not once per vocab token --
            # see docs/GATE_KERNEL_FUSION_SPEC.md Section 11 for why the old
            # per-token native-call loop was the documented bottleneck this
            # replaces.
            unique_actions, token_to_unique_index = self.processor._action_map_decomposition(vocab_size)
            unique_allowed = self.processor._unique_action_allowed(
                unique_actions, allowed_min, allowed_max, safety_state
            )
            if HAS_NUMPY:
                unique_allowed_arr = np.asarray(unique_allowed, dtype=bool)
                index_arr = np.asarray(token_to_unique_index, dtype=np.int64)
                denied_mask = ~unique_allowed_arr[index_arr]
                pruned_count = int(denied_mask.sum())
                if isinstance(scores, np.ndarray):
                    scores[denied_mask] = mask_value
                else:
                    for idx in np.flatnonzero(denied_mask).tolist():
                        scores[idx] = mask_value
            else:
                # No numpy available: still no per-token native calls (the
                # actual bottleneck) -- just a plain Python list-index
                # lookup per token instead of a ctypes call per token.
                for token_id in range(vocab_size):
                    if not unique_allowed[token_to_unique_index[token_id]]:
                        scores[token_id] = mask_value
                        pruned_count += 1

        # get_summary()'s total_pruned_tokens reads self.processor.total_pruned,
        # which the HF __call__ path updates itself -- this adapter bypasses
        # that path entirely (it mutates scores directly), so it has to do
        # the same bookkeeping or the reported total is silently always 0.
        self.processor.total_pruned += pruned_count

        dt_ns = time.perf_counter_ns() - t0
        self.total_stream_latency_ns += dt_ns
        self.token_count += 1
        record = {
            "step": self.token_count,
            "latency_us": dt_ns / 1_000.0,
            "pruned_count": pruned_count,
            "safety_state": self.processor.safety_state,
        }
        self.history.append(record)
        return scores

    def get_summary(self) -> dict[str, Any]:
        """Return streaming audit and performance metrics."""
        avg_us = (self.total_stream_latency_ns / max(1, self.token_count)) / 1_000.0
        return {
            "total_tokens_processed": self.token_count,
            "total_pruned_tokens": self.processor.total_pruned,
            "total_latency_ms": self.total_stream_latency_ns / 1_000_000.0,
            "avg_pruning_latency_us": round(avg_us, 2),
            "last_step_audit": self.history[-1] if self.history else {},
        }


def stream_generate(
    logits_fn: Callable[[list[int]], Any],
    stream_processor: NetelproStreamProcessor,
    initial_tokens: Optional[list[int]] = None,
    max_tokens: int = 32,
    eos_token_id: int = 0,
    tokenizer_decode: Optional[Callable[[int], str]] = None,
    temperature: float = 1.0,
) -> Generator[tuple[int, str, dict[str, Any]], None, None]:
    """Decoupled streaming token generation loop with formal silicon gating.

    Yields:
        tuple of (token_id: int, token_str: str, step_audit: dict)
        for each emitted token in real time.
    """
    current_tokens = list(initial_tokens or [1])

    for step in range(max_tokens):
        # 1. Get raw logits from the backend model
        raw_scores = logits_fn(current_tokens)

        # 2. Apply silicon formal filter
        if HAS_TORCH and isinstance(raw_scores, torch.Tensor):
            if raw_scores.dim() == 1:
                batched = raw_scores.unsqueeze(0)
                filtered = stream_processor.process_hf_logits(
                    torch.tensor([current_tokens]), batched
                ).squeeze(0)
            else:
                filtered = stream_processor.process_hf_logits(
                    torch.tensor([current_tokens]), raw_scores
                )

            # Softmax & Sampling
            if temperature <= 0.0:
                next_token_id = int(torch.argmax(filtered).item())
            else:
                probs = torch.softmax(filtered / max(1e-5, temperature), dim=-1)
                # Ensure no NaN / inf issues if everything was pruned
                if torch.isnan(probs).any() or probs.sum() <= 0:
                    next_token_id = eos_token_id
                else:
                    next_token_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            # Native list / numpy path
            scores_copy = list(raw_scores)
            filtered = stream_processor.llama_cpp_processor(current_tokens, scores_copy)
            max_val = float("-inf")
            next_token_id = eos_token_id
            for idx, val in enumerate(filtered):
                if val > max_val:
                    max_val = val
                    next_token_id = idx

        audit_entry = stream_processor.history[-1] if stream_processor.history else {}

        # 3. Decode token text
        token_str = (
            tokenizer_decode(next_token_id)
            if tokenizer_decode
            else f"tok_{next_token_id}"
        )

        current_tokens.append(next_token_id)

        yield next_token_id, token_str, audit_entry

        if next_token_id == eos_token_id:
            break
