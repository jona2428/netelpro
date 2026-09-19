"""Live evidence: a semantically meaningful token gate on a real model, not
an arbitrary token-ID range. Forces Qwen2.5 to answer with digits ONLY --
structurally, at every sampling step, not by asking nicely in the prompt.

Builds a real token_to_action_map: every vocab token gets action 1 (allowed)
if its decoded text is purely numeric AND contains at least one real digit
('.'/'-' are allowed only as part of a number), action 0 (denied) otherwise
-- except the EOS token, always allowed so generation can actually stop.
Two unique actions across a 151,936-token
vocab (Qwen2.5's real vocab size) is the best possible case for the
discrete-map fast path fixed the same session
(docs/GATE_KERNEL_FUSION_SPEC.md Section 11, netelpro/guard's discrete
token_to_action_map cache): O(2) native gate calls per decode step instead
of O(vocab_size), regardless of how large the vocabulary is.

Usage:
    python -m examples.token_gate_numeric_output_demo --model "C:/path/to/qwen2.5-1.5b-instruct.gguf"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llama_cpp import Llama, LogitsProcessorList
except ImportError:
    print("This demo needs llama-cpp-python: pip install llama-cpp-python")
    sys.exit(1)

from netelpro.neuro.stream import NetelproStreamProcessor

_NUMERIC_CHARS = set("0123456789.-")


def build_numeric_token_action_map(llm: Llama, vocab_size: int) -> dict[int, int]:
    """action=1 for tokens that decode to pure digits (+ '.'/'-'), action=0
    otherwise. EOS is force-allowed (action=1) so the model can still stop
    generating -- without this it would run to max_tokens every time, since
    the EOS token never decodes to digits on its own.

    A token counts as numeric only if it also contains at least one real digit
    (0-9). Without that check the map admits tokens whose text is made only of
    '.'/'-' -- e.g. ' -----------', '...............' -- which pass the
    character-set test while carrying no number at all. Measured 2026-09-17 on
    Qwen2.5-1.5B: loose map = 147 allowed tokens, digit-requiring map = 10,
    i.e. 137 tokens that were "numeric" with zero digits in them. That gap is
    where the spurious sign in the two-phase demo came from: the model was
    free to open its answer with a bare '-' (see
    benchmarks/token_gate_numeric_output_report.md)."""
    eos_id = llm.token_eos()
    action_map: dict[int, int] = {}
    for token_id in range(vocab_size):
        if token_id == eos_id:
            action_map[token_id] = 1
            continue
        text = llm.detokenize([token_id]).decode("utf-8", errors="ignore").strip()
        has_digit = any(c.isdigit() for c in text)
        is_numeric = bool(text) and has_digit and all(c in _NUMERIC_CHARS for c in text)
        action_map[token_id] = 1 if is_numeric else 0
    return action_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="¿Cuánto es 47 más 89? Respondé solo con el resultado.")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No file at {model_path}")
        sys.exit(1)

    print("=" * 78)
    print("TOKEN GATE -- SEMANTIC CONSTRAINT ON A REAL MODEL (digits-only output)")
    print("=" * 78)

    llm = Llama(model_path=str(model_path), n_ctx=512, verbose=False)
    vocab_size = llm.n_vocab()
    print(f"Model loaded. vocab_size={vocab_size:,}")

    t0 = time.perf_counter()
    action_map = build_numeric_token_action_map(llm, vocab_size)
    build_time = time.perf_counter() - t0
    numeric_count = sum(1 for v in action_map.values() if v == 1)
    print(f"Built token_to_action_map in {build_time:.2f}s -- {numeric_count:,}/{vocab_size:,} tokens are numeric-allowed")

    # --- Phase 1: baseline, no gate --------------------------------------
    t0 = time.perf_counter()
    out = llm(args.prompt, max_tokens=args.max_tokens, temperature=0.0)
    dt = time.perf_counter() - t0
    print("\nPHASE 1 -- no gate (model free to explain, hedge, use words)")
    print(f"  output: {out['choices'][0]['text']!r}")
    print(f"  wall time: {dt:.2f}s")

    # --- Phase 2: gated, digits-only structurally enforced ----------------
    sp = NetelproStreamProcessor(
        allowed_min=1, allowed_max=1, safety_state=1, token_to_action_map=action_map,
    )
    lp = LogitsProcessorList([sp.llama_cpp_processor])
    t0 = time.perf_counter()
    out = llm(args.prompt, max_tokens=args.max_tokens, temperature=0.0, logits_processor=lp)
    dt = time.perf_counter() - t0
    text = out["choices"][0]["text"]
    print("\nPHASE 2 -- gated (structurally cannot emit a non-numeric token)")
    print(f"  output: {text!r}")
    print(f"  wall time: {dt:.2f}s ({dt / args.max_tokens * 1000:.2f}ms/token, model+gate combined)")

    is_all_numeric = bool(text.strip()) and all(c in _NUMERIC_CHARS or c.isspace() for c in text)
    print(f"  output is purely numeric: {is_all_numeric}")

    summary = sp.get_summary()
    print(f"  gate overhead: {summary['avg_pruning_latency_us']:.2f} us/token (2 unique actions, {vocab_size:,}-token vocab)")
    print(f"  tokens pruned per step (avg): {summary['total_pruned_tokens'] / max(1, summary['total_tokens_processed']):.0f} of {vocab_size:,}")

    print("\n" + "=" * 78)
    print("Phase 2's output cannot contain a non-numeric token by construction --")
    print("not because the prompt asked nicely, because every non-numeric logit")
    print("was set to -inf before sampling, every single step.")


if __name__ == "__main__":
    main()
