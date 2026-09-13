"""Netelpro Logits Gate: Neuro-symbolic candidate pruning for LLMs and discrete policies.

Applies deterministic compiled Netelpro rules to logit vectors, setting
violating tokens or action indices to -inf (probability 0.0) before sampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from netelpro.gate import Gate
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch

_DEFAULT_ACTION_RULE = Path(__file__).parent / "rules" / "action_boundary.sl"


class NetelproLogitsGate:
    """Deterministic logit pruning gate over a compiled Netelpro rule."""

    def __init__(
        self,
        rule_path: str | Path | None = None,
        defn_name: str = "filter-rule",
    ) -> None:
        actual_rule = Path(rule_path) if rule_path else _DEFAULT_ACTION_RULE
        self.gate = Gate(actual_rule, defn_name=defn_name)

    def check_action(
        self,
        action_id: int,
        allowed_min: int,
        allowed_max: int,
        safety_state: int = 1,
    ) -> tuple[bool, str | None]:
        """Check a single action ID against the gate rule. Never raises."""
        return self.gate.check(action_id, allowed_min, allowed_max, safety_state)

    def filter_logits(
        self,
        logits: Any,
        allowed_min: int,
        allowed_max: int,
        safety_state: int = 1,
        mask_value: float = float("-inf"),
    ) -> Any:
        """Mask out non-conforming tokens with mask_value (-inf)."""
        if HAS_TORCH and isinstance(logits, torch.Tensor):
            masked = logits.clone()
            is_1d = (logits.dim() == 1)
            vocab_size = logits.size(-1)

            # Evaluate mask for vocabulary indices
            valid_indices = []
            for idx in range(vocab_size):
                allow, _ = self.gate.check(idx, allowed_min, allowed_max, safety_state)
                if not allow:
                    if is_1d:
                        masked[idx] = mask_value
                    else:
                        masked[..., idx] = mask_value
            return masked

        if isinstance(logits, (list, tuple)):
            filtered = list(logits)
            for idx in range(len(filtered)):
                allow, _ = self.gate.check(idx, allowed_min, allowed_max, safety_state)
                if not allow:
                    filtered[idx] = mask_value
            return filtered

        raise TypeError(f"unsupported logits type: {type(logits)}")
