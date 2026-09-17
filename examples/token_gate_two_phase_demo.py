"""Two-phase generation: unconstrained reasoning, THEN switch the token gate
on for the final answer only -- follow-up to
benchmarks/token_gate_numeric_output_report.md, which found that gating
every token from the first one (digits-only, no room for reasoning tokens)
produced a wrong answer (47+89 -> -82 instead of 136).

Hypothesis this tests, not assumed: if the model gets to reason freely in
natural language first, and the gate only turns on for the final answer
span, does it get the arithmetic right AND keep the structural
digits-only guarantee on the answer itself? Two real llama-cpp-python
calls, not one -- llama.cpp has no mid-generation logits_processor swap,
so phase 1 (ungated) and phase 2 (gated) are separate calls, phase 2's
prompt built from phase 1's own output plus an explicit marker.

Usage:
    python -m examples.token_gate_two_phase_demo --model "C:/path/to/qwen2.5-1.5b-instruct.gguf"
"""

from __future__ import annotations

import argparse
import re
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
_MARKER = "Respuesta final:"


def extract_number(text: str) -> float | None:
    """First signed number in text, or None. Used instead of a substring
    check -- this script's first version used `str(ground_truth) in
    answer_text`, which called a wrong answer "correct" because "15" is a
    literal substring of "-15.9999" even though they're different numbers.
    Caught by hand-checking real output, not assumed correct; see
    benchmarks/token_gate_two_phase_report.md."""
    m = re.search(r"-?\d+(\.\d+)?", text)
    return float(m.group()) if m else None


def build_numeric_token_action_map(llm: Llama, vocab_size: int) -> dict[int, int]:
    eos_id = llm.token_eos()
    action_map: dict[int, int] = {}
    for token_id in range(vocab_size):
        if token_id == eos_id:
            action_map[token_id] = 1
            continue
        text = llm.detokenize([token_id]).decode("utf-8", errors="ignore").strip()
        is_numeric = bool(text) and all(c in _NUMERIC_CHARS for c in text)
        action_map[token_id] = 1 if is_numeric else 0
    return action_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--a", type=int, default=47)
    parser.add_argument("--b", type=int, default=89)
    parser.add_argument("--max-reasoning-tokens", type=int, default=100)
    parser.add_argument("--max-answer-tokens", type=int, default=8)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No file at {model_path}")
        sys.exit(1)

    ground_truth = args.a + args.b

    prompt = (
        f"¿Cuánto es {args.a} más {args.b}? "
        "Pensá el cálculo paso a paso en una oración breve, sin escribir el "
        "número final todavía. Cuando termines de pensar, escribí "
        f"exactamente la frase '{_MARKER}' y recién ahí vas a decir el número."
    )

    print("=" * 78)
    print("TWO-PHASE TOKEN GATE -- reason free, gate only the final answer")
    print("=" * 78)
    print(f"Question: {args.a} + {args.b} = {ground_truth} (ground truth, computed in Python)")

    llm = Llama(model_path=str(model_path), n_ctx=512, verbose=False)
    vocab_size = llm.n_vocab()

    action_map = build_numeric_token_action_map(llm, vocab_size)
    sp = NetelproStreamProcessor(
        allowed_min=1, allowed_max=1, safety_state=1, token_to_action_map=action_map,
    )
    lp = LogitsProcessorList([sp.llama_cpp_processor])

    # --- Phase 1: unconstrained reasoning, stop at the marker -------------
    t0 = time.perf_counter()
    out1 = llm(prompt, max_tokens=args.max_reasoning_tokens, temperature=0.0, stop=[_MARKER])
    dt1 = time.perf_counter() - t0
    reasoning = out1["choices"][0]["text"]
    print("\nPHASE 1 -- unconstrained reasoning (no gate)")
    print(f"  output: {reasoning!r}")
    print(f"  wall time: {dt1:.2f}s")
    reasoning_leaked_answer = bool(re.search(rf"\b{re.escape(str(ground_truth))}\b", reasoning))
    print(f"  ground-truth number already appears in reasoning: {reasoning_leaked_answer}")

    # --- Phase 2: gated, digits-only, continuing from phase 1's own text --
    phase2_prompt = prompt + reasoning + _MARKER
    t0 = time.perf_counter()
    out2 = llm(phase2_prompt, max_tokens=args.max_answer_tokens, temperature=0.0, logits_processor=lp)
    dt2 = time.perf_counter() - t0
    answer_text = out2["choices"][0]["text"]
    print(f"\nPHASE 2 -- gated (structurally digits-only, continuing after '{_MARKER}')")
    print(f"  output: {answer_text!r}")
    print(f"  wall time: {dt2:.2f}s")

    is_all_numeric = bool(answer_text.strip()) and all(
        c in _NUMERIC_CHARS or c.isspace() for c in answer_text
    )
    parsed_answer = extract_number(answer_text)
    correct = parsed_answer is not None and abs(parsed_answer - ground_truth) < 0.5
    print(f"  answer is purely numeric: {is_all_numeric}")
    print(f"  parsed answer value: {parsed_answer}")
    print(f"  answer equals the ground-truth number ({ground_truth}): {correct}")

    print("\n" + "-" * 78)
    print("SUMMARY (honest, not assumed):")
    print(f"  final answer structurally digits-only: {is_all_numeric}")
    print(f"  final answer arithmetically correct: {correct}")
    if is_all_numeric and correct:
        print("  -> two-phase generation kept BOTH properties this run: format guarantee AND correctness.")
    elif is_all_numeric and not correct:
        print("  -> format guarantee held, but the answer is still wrong -- reasoning access alone didn't fix it here.")
    else:
        print("  -> format guarantee itself failed to hold -- investigate before trusting either number.")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
