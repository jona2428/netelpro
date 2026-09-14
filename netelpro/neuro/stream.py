"""Netelpro Real-Time Streaming Inference Engine.

Integrates compiled Netelpro formal gates into real-time token streaming pipelines,
enabling microsecond-level pruning of illegal next-token candidates across
HuggingFace Transformers and llama.cpp/GGUF engines without interrupting user-facing latency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Optional, Sequence

from netelpro.gate import Gate
from netelpro.neuro.logits_processor import NetelproLogitsProcessor
from netelpro.neuro.native_kernel import NetelproVectorKernel
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch


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
        """
        t0 = time.perf_counter_ns()
        vocab_size = len(scores)

        pruned_count = 0
        if hasattr(scores, "__getitem__") and hasattr(scores, "__setitem__"):
            for token_id in range(vocab_size):
                action_id = (
                    self.processor.token_to_action_map.get(token_id, token_id)
                    if self.processor.token_to_action_map
                    else token_id
                )
                allow, _ = self.gate.check(
                    action_id,
                    self.processor.allowed_min,
                    self.processor.allowed_max,
                    self.processor.safety_state,
                )
                if not allow:
                    scores[token_id] = self.processor.mask_value
                    pruned_count += 1

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
