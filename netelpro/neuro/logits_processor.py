"""Netelpro Logits Processor for HuggingFace Transformers and PyTorch inference.

Prunes illegal next-token candidates at every autoregressive decoding step
using a deterministic compiled Netelpro Gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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

        # token_to_action_map is set once at construction and never mutated
        # after (no setter exists) -- keyed by vocab_size since that's the
        # only other thing the decomposition depends on. See
        # _action_map_decomposition's docstring for what's cached and why.
        self._action_decomposition_cache: dict[int, tuple[list[int], list[int]]] = {}
        # Separate from the Python-list cache above: torch.tensor(list) is
        # itself an O(vocab_size) bulk copy, cheap once but wasteful if
        # rebuilt every decode step -- cached per (vocab_size, device).
        self._action_index_tensor_cache: dict[tuple[int, str], Any] = {}

    def _action_map_decomposition(self, vocab_size: int) -> tuple[list[int], list[int]]:
        """(unique_actions, token_to_unique_index) for token_to_action_map
        at this vocab_size, computed once and cached.

        unique_actions[i] is the i-th distinct action id reachable from any
        token (via the map, or via the `.get(token_id, token_id)` identity
        fallback for tokens the map doesn't mention). token_to_unique_index
        has length vocab_size; token_to_unique_index[t] is the position in
        unique_actions of token t's action.

        This is what lets the per-step gate evaluation cost O(unique
        actions) native calls instead of O(vocab_size) -- see
        docs/GATE_KERNEL_FUSION_SPEC.md Section 11. Building the
        decomposition itself is still O(vocab_size) (pure Python dict
        grouping, no native calls), but it happens once per vocab_size, not
        once per decode step, which is the entire point: today's per-step
        cost was O(vocab_size) *native ctypes calls*, not O(vocab_size)
        Python-level bookkeeping.
        """
        cached = self._action_decomposition_cache.get(vocab_size)
        if cached is not None:
            return cached

        action_to_index: dict[int, int] = {}
        unique_actions: list[int] = []
        token_to_unique_index: list[int] = [0] * vocab_size
        tam = self.token_to_action_map
        for token_id in range(vocab_size):
            action_id = tam.get(token_id, token_id) if tam else token_id
            idx = action_to_index.get(action_id)
            if idx is None:
                idx = len(unique_actions)
                action_to_index[action_id] = idx
                unique_actions.append(action_id)
            token_to_unique_index[token_id] = idx

        result = (unique_actions, token_to_unique_index)
        self._action_decomposition_cache[vocab_size] = result
        return result

    def _unique_action_allowed(
        self, unique_actions: list[int], allowed_min: int, allowed_max: int, safety_state: int
    ) -> list[bool]:
        """Evaluate the compiled gate once per unique action (not per
        token) -- via self.gate.check(), same fail-closed wrapper the
        per-token loop this replaces used (NOT
        NetelproVectorKernel.evaluate_batch, which calls the raw native
        function pointer with no exception handling -- that's fine for its
        existing neuron-potential use case, but this path keeps the exact
        fail-closed contract §3.2 of gate_contract.md documents: a
        misbehaving rule denies, it never raises past this call)."""
        return [
            self.gate.check(action_id, allowed_min, allowed_max, safety_state)[0]
            for action_id in unique_actions
        ]

    def _action_index_tensor(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        """torch.LongTensor version of _action_map_decomposition's
        token_to_unique_index, cached per (vocab_size, device) so the
        O(vocab_size) list->tensor conversion happens once, not every
        decode step."""
        key = (vocab_size, str(device))
        cached = self._action_index_tensor_cache.get(key)
        if cached is not None:
            return cached
        _, token_to_unique_index = self._action_map_decomposition(vocab_size)
        index_t = torch.tensor(token_to_unique_index, dtype=torch.long, device=device)
        self._action_index_tensor_cache[key] = index_t
        return index_t

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
            unique_actions, _ = self._action_map_decomposition(vocab_size)
            unique_allowed = self._unique_action_allowed(
                unique_actions, self.allowed_min, self.allowed_max, self.safety_state
            )
            # Gate evaluated once per unique action (above), expanded back
            # to per-token via a single vectorized gather -- not a
            # vocab_size-length Python loop of native calls, which is what
            # made this branch the documented bottleneck (see docs/
            # GATE_KERNEL_FUSION_SPEC.md Section 11).
            unique_allowed_t = torch.tensor(unique_allowed, dtype=torch.bool, device=scores.device)
            index_t = self._action_index_tensor(vocab_size, scores.device)
            token_allowed = unique_allowed_t[index_t]
            mask = ~token_allowed
            pruned_this_step = int(mask.sum().item())
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
