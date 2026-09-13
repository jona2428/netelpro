"""Netelpro Logits Processor for HuggingFace Transformers and PyTorch inference.

Prunes illegal next-token candidates at every autoregressive decoding step
using a deterministic compiled Netelpro Gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from netelpro.gate import Gate
from netelpro.neuro.native_kernel import NetelproVectorKernel
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch
    try:
        from transformers import LogitsProcessor
    except ImportError:
        class LogitsProcessor:  # type: ignore
            pass
else:
    class LogitsProcessor:  # type: ignore
        pass

_DEFAULT_ACTION_RULE = Path(__file__).parent / "rules" / "action_boundary.sl"


class NetelproLogitsProcessor(LogitsProcessor):
    """Autoregressive LogitsProcessor powered by Netelpro formal gates.

    Conforms to the standard Hugging Face LogitsProcessor API:
        processor(input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor
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
        actual_rule = Path(rule_path) if rule_path else _DEFAULT_ACTION_RULE
        self.gate = Gate(actual_rule)
        self.kernel = NetelproVectorKernel(self.gate)
        self.allowed_min = allowed_min
        self.allowed_max = allowed_max
        self.safety_state = safety_state
        self.token_to_action_map = token_to_action_map
        self.mask_value = mask_value

        self.total_steps: int = 0
        self.total_pruned: int = 0
        self.last_audit: dict[str, Any] = {}

    def set_context(self, allowed_min: int, allowed_max: int, safety_state: int = 1) -> None:
        """Update dynamic environmental constraints for subsequent token steps."""
        self.allowed_min = allowed_min
        self.allowed_max = allowed_max
        self.safety_state = safety_state

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Prunes scores (logits) for the next token based on compiled contract."""
        if not HAS_TORCH or not isinstance(scores, torch.Tensor):
            return scores

        self.total_steps += 1
        vocab_size = scores.size(-1)

        if self.token_to_action_map is None:
            # Vectorized fast path
            scores = self.kernel.filter_logits_tensor(
                scores, self.allowed_min, self.allowed_max, self.safety_state, self.mask_value
            )
            pruned_this_step = vocab_size if self.safety_state == 0 else max(0, vocab_size - (min(self.allowed_max, vocab_size - 1) - max(0, self.allowed_min) + 1))
        else:
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=scores.device)
            pruned_this_step = 0
            for token_id in range(vocab_size):
                action_id = self.token_to_action_map.get(token_id, token_id)
                allow, _ = self.gate.check(action_id, self.allowed_min, self.allowed_max, self.safety_state)
                if not allow:
                    mask[token_id] = True
                    pruned_this_step += 1
            scores[:, mask] = self.mask_value

        self.total_pruned += pruned_this_step

        self.last_audit = {
            "step": self.total_steps,
            "pruned_count": pruned_this_step,
            "total_tokens": vocab_size,
            "allowed_min": self.allowed_min,
            "allowed_max": self.allowed_max,
            "safety_state": self.safety_state,
        }

        return scores
