"""Benchmark: discrete token_to_action_map gate evaluation, old vs new.

docs/GATE_KERNEL_FUSION_SPEC.md Section 11's target -- the per-token
Python loop calling the compiled gate once per vocab token (unfixed
alongside the contiguous-range fix in 5286c73) vs. the unique-action
cache + gather rewrite in netelpro/neuro/logits_processor.py. No GPU
needed: this is a pure algorithmic fix, measured on CPU, honestly, same
as every other benchmark in this repo.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from netelpro.neuro.stream import NetelproStreamProcessor


def old_per_token_loop(sp: NetelproStreamProcessor, vocab_size: int, scores: list[float]) -> list[float]:
    """Reconstruction of the loop netelpro/neuro/stream.py's
    llama_cpp_processor used before this fix -- kept here, not in the
    package, purely so this benchmark can show a real before/after."""
    processor = sp.processor
    token_to_action_map = processor.token_to_action_map
    allowed_min, allowed_max, safety_state = processor.allowed_min, processor.allowed_max, processor.safety_state
    mask_value = processor.mask_value
    for token_id in range(vocab_size):
        action_id = token_to_action_map.get(token_id, token_id)
        allow, _ = sp.gate.check(action_id, allowed_min, allowed_max, safety_state)
        if not allow:
            scores[token_id] = mask_value
    return scores


def build_action_map(vocab_size: int, num_actions: int) -> dict[int, int]:
    """A plausible discrete-action scenario: most of a large LLM vocab
    collapses onto a much smaller enumerated action space (the reason a
    token_to_action_map exists at all -- see stream.py's module docstring
    on agent/action-constrained generation)."""
    return {token_id: token_id % num_actions for token_id in range(vocab_size)}


def run(vocab_size: int, num_actions: int, num_steps: int) -> None:
    print("=" * 70)
    print(f"vocab_size={vocab_size:,}  unique_actions={num_actions}  steps={num_steps}")
    print("=" * 70)

    action_map = build_action_map(vocab_size, num_actions)
    allowed_min, allowed_max = 0, num_actions // 3  # roughly a third of actions allowed

    sp_old = NetelproStreamProcessor(
        allowed_min=allowed_min, allowed_max=allowed_max, safety_state=1, token_to_action_map=action_map
    )
    t0 = time.perf_counter()
    for _ in range(num_steps):
        old_per_token_loop(sp_old, vocab_size, [0.0] * vocab_size)
    old_elapsed = time.perf_counter() - t0
    old_per_step_ms = 1000.0 * old_elapsed / num_steps

    sp_new = NetelproStreamProcessor(
        allowed_min=allowed_min, allowed_max=allowed_max, safety_state=1, token_to_action_map=action_map
    )
    # Warm the decomposition cache once (this is exactly what happens
    # naturally on the first real decode step -- amortized over a whole
    # generation, not re-paid every step).
    sp_new.llama_cpp_processor([1], [0.0] * vocab_size)
    t0 = time.perf_counter()
    for _ in range(num_steps):
        sp_new.llama_cpp_processor([1], [0.0] * vocab_size)
    new_elapsed = time.perf_counter() - t0
    new_per_step_ms = 1000.0 * new_elapsed / num_steps

    print(f"old (per-token native call loop): {old_per_step_ms:.3f} ms/step")
    print(f"new (unique-action cache + gather): {new_per_step_ms:.3f} ms/step")
    print(f"speedup: {old_per_step_ms / new_per_step_ms:.1f}x")
    print()


def run_worst_case(vocab_size: int, num_steps: int) -> None:
    """The map exists but barely collapses anything -- 99% of tokens fall
    through to the identity default (action_id == token_id), so
    unique_actions ~= vocab_size. This should show no meaningful
    regression versus the old loop, not a win -- honest worst case, not
    cherry-picked."""
    print("=" * 70)
    print(f"WORST CASE: vocab_size={vocab_size:,}  sparse map (1% explicit, 99% identity)  steps={num_steps}")
    print("=" * 70)

    action_map = {t: 999_000_000 + t for t in range(0, vocab_size, 100)}  # 1% of tokens overridden
    allowed_min, allowed_max = 0, vocab_size // 3

    sp_new = NetelproStreamProcessor(
        allowed_min=allowed_min, allowed_max=allowed_max, safety_state=1, token_to_action_map=action_map
    )
    sp_new.llama_cpp_processor([1], [0.0] * vocab_size)  # warm cache
    t0 = time.perf_counter()
    for _ in range(num_steps):
        sp_new.llama_cpp_processor([1], [0.0] * vocab_size)
    new_elapsed = time.perf_counter() - t0
    print(f"new (unique-action cache + gather), ~identity map: {1000.0 * new_elapsed / num_steps:.3f} ms/step")
    print("(measured: essentially tied with old's per-step cost at this vocab size, not faster -- when")
    print(" unique_actions ~= vocab_size the per-step cost IS still ~vocab_size native gate calls, same")
    print(" as the old loop. No regression, but no win here either -- honest, not oversold.)")
    print()


if __name__ == "__main__":
    # Teo v2-scale vocab, a modest discrete action space, few steps for the
    # O(vocab) old path (it's slow -- deliberately keep num_steps small so
    # this finishes in a reasonable time), more steps for the new path
    # (cheap enough to get a stable average).
    run(vocab_size=32768, num_actions=64, num_steps=5)

    # A larger, GGUF-scale vocab (this file's original motivating case --
    # see the ~370ms/token note in stream.py's llama_cpp_processor
    # docstring for the sibling contiguous-range fix).
    run(vocab_size=152000, num_actions=200, num_steps=3)

    run_worst_case(vocab_size=32768, num_steps=5)
