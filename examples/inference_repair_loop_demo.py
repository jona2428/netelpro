"""Live evidence for docs/INFERENCE_REPAIR_LOOP_SPEC.md: does feeding a
real compiler's exact rejection reason back into the next prompt actually
improve pass rate, on a real (non-finetuned) local model?

Generate -> verify (real oracle: rlvr.verify.verify_program, the same one
RAFT uses to build training data offline) -> if it fails, build the next
prompt from the previous candidate + its EXACT error -> retry, bounded by
netelpro.state_gate.RetryLimiter (reused as-is, not reinvented) -> on
exhaustion, fail closed with the last real error, never a silently-wrong
answer.

The model used here (any local instruct GGUF, e.g. qwen2.5-1.5b-instruct)
was never trained on Netelpro syntax -- that's deliberate. First-attempt
success is unlikely, which is exactly what makes the repair loop's effect
observable: does the exact `line:col` compiler diagnostic teach a generic
model the grammar within a few retries, or not. This prints the honest
answer, attempt by attempt -- it does not assume the loop helps.

Requires llama-cpp-python (pip install ".[llama-cpp]"):

    python -m examples.inference_repair_loop_demo --model "C:/path/to/model.gguf"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llama_cpp import Llama
except ImportError:
    print("This demo needs llama-cpp-python: pip install llama-cpp-python")
    sys.exit(1)

import importlib

from benchmarks.run5_gate import extract_block
from netelpro.state_gate import RetryLimiter
from rlvr.verify import VerifyResult, verify_program

SYNTAX_PRIMER = """Netelpro is a small Lisp-like language. Every form is
(head arg1 arg2 ...). Define a function with defn:

(defn fib (n)
  (if (< n 2)
      n
      (+ (fib (- n 1)) (fib (- n 2)))))

Operators: + - * < > <= >= == and or if. No loops -- use recursion.
Write ONLY the function definition inside a ```netelpro fenced block, and
nothing else."""


def build_prompt(task_desc: str, signature: str, prior_attempt: str | None, prior_error: str | None) -> str:
    base = (
        f"{SYNTAX_PRIMER}\n\nTask: {task_desc}\nSignature: {signature}\n\n"
        f"```netelpro\n"
    )
    if prior_attempt is None:
        return base
    return (
        f"{SYNTAX_PRIMER}\n\nTask: {task_desc}\nSignature: {signature}\n\n"
        f"Your previous attempt:\n```netelpro\n{prior_attempt}\n```\n"
        f"It failed with this exact compiler error:\n{prior_error}\n\n"
        f"Fix it. Write the corrected function inside a ```netelpro fenced block.\n\n"
        f"```netelpro\n"
    )


def format_verdict(attempt: int, result: VerifyResult) -> str:
    if not result.compiled:
        return f"  attempt {attempt}: DID NOT COMPILE -- {result.error}"
    if result.passed:
        return f"  attempt {attempt}: PASSED all {result.cases_total} cases"
    return (
        f"  attempt {attempt}: compiled, but {result.cases_passed}/{result.cases_total} "
        f"cases passed -- {result.error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="sum_range", help="Module name under rlvr.tasks")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No file at {model_path}")
        sys.exit(1)

    task = importlib.import_module(f"rlvr.tasks.{args.task}")

    print("=" * 78)
    print("INFERENCE-TIME REPAIR LOOP -- LIVE EVIDENCE")
    print(f"Model: {model_path.name} | Task: {task.TASK_ID}")
    print("=" * 78)

    llm = Llama(model_path=str(model_path), n_ctx=1024, verbose=False)
    limiter = RetryLimiter()
    resource = f"repair-loop::{task.TASK_ID}"
    # cooldown effectively disabled: we want a hard cap of max_retries
    # attempts total for this single request, not a time-windowed limit.
    cooldown_ms = 10**9

    prior_attempt: str | None = None
    prior_error: str | None = None
    results: list[VerifyResult] = []
    final_candidate: str | None = None

    attempt_n = 0
    while True:
        allowed, reason = limiter.attempt(resource, max_retries=args.max_retries, cooldown_ms=cooldown_ms)
        if not allowed:
            print(f"\nRetry budget exhausted: {reason}")
            break

        attempt_n += 1
        prompt = build_prompt(task.DESCRIPTION_ES, task.SIGNATURE, prior_attempt, prior_error)
        out = llm(prompt, max_tokens=args.max_tokens, temperature=0.2, stop=["```"])
        raw = out["choices"][0]["text"]
        candidate = (extract_block(raw, fence="netelpro") or raw).strip()

        result = verify_program(candidate, task, num_cases=10, seed=0)
        results.append(result)
        print(format_verdict(attempt_n, result))

        if result.passed:
            final_candidate = candidate
            break

        prior_attempt = candidate
        prior_error = result.error or "compiled but produced wrong results on some cases"

    print("\n" + "-" * 78)
    n_passed_attempts = sum(1 for r in results if r.passed)
    n_compiled = sum(1 for r in results if r.compiled)
    print(f"Attempts made: {len(results)} | compiled: {n_compiled} | passed: {n_passed_attempts}")

    if final_candidate is not None:
        print("\nFinal verified candidate (passed all cases via the real compiler):")
        print(final_candidate)
    else:
        print("\nNo candidate passed within the retry budget. Failing closed --")
        print(f"last real error: {results[-1].error if results else 'no attempts recorded'}")

    print("\n" + "=" * 78)
    print("Every verdict above came from rlvr.verify.verify_program -- the real")
    print("parser, capability/hole checker, and interpreter -- not a guess about")
    print("whether the candidate looked plausible.")


if __name__ == "__main__":
    main()
