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
import json

from benchmarks.run5_gate import extract_block
from netelpro.state_gate import RetryLimiter
from rlvr.verify import VerifyResult, verify_program

_ARITY_TABLE_PATH = Path(__file__).parent.parent / "netelpro" / "spec" / "arity_table.json"


def _build_syntax_primer() -> str:
    """Renders the exhaustive head list from spec/arity_table.json -- the
    same machine-consumed source of truth the compiler itself checks
    against -- instead of a hand-picked example subset.

    Root cause #1 this fixes: the v0.1 pilot's hand-written primer only
    mentioned "+ - * < > <= >= == and or if" and two example programs. On
    the gcd_pair task the model invented `zero?` (a real Scheme/Lisp idiom
    Netelpro does not have) because nothing told it what actually exists.
    Listing every legal head removes the guessing, not just the specific
    mistake that got fed back last time.

    Root cause #2 (INFERENCE_REPAIR_LOOP_SPEC.md section 8's second untried
    move, tried now): after root cause #1 was fixed, gcd_pair still failed
    all 5 attempts the same way -- the model wrote `(defn gcd-two (a b) a)`
    every time (returns `a` unconditionally, never recurses). The only
    worked example in the primer (`fib`) recurses on a SINGLE argument
    (`n`); it never shows a function whose base case depends on the SECOND
    argument, or whose recursive call transforms BOTH arguments together.
    That's a plausible reason the model never tried it for gcd_two: nothing
    in the primer demonstrated the shape at all. The second example below
    (`cuenta-pasos`) teaches exactly that shape -- two arguments, base case
    on the second one, both arguments change in the recursive call -- on a
    deliberately different, much simpler problem (counts down both
    arguments by 1 until the second hits zero), so it's a structural
    template, not the gcd answer smuggled in.
    """
    table = json.loads(_ARITY_TABLE_PATH.read_text(encoding="utf-8"))
    forms = sorted(table["special_forms"].keys())
    prims = sorted(table["primitives"].keys())
    return (
        "Netelpro is a small Lisp-like language. Every form is "
        "(head arg1 arg2 ...). Define a function with defn:\n\n"
        "(defn fib (n)\n"
        "  (if (< n 2)\n"
        "      n\n"
        "      (+ (fib (- n 1)) (fib (- n 2)))))\n\n"
        "Functions can take more than one argument, and the recursive call "
        "can change ANY of them, not just the first:\n\n"
        "(defn cuenta-pasos (a b)\n"
        "  (if (== b 0)\n"
        "      a\n"
        "      (cuenta-pasos (- a 1) (- b 1))))\n\n"
        f"ALL special forms that exist (nothing else is legal): {', '.join(forms)}\n"
        f"ALL primitives that exist (nothing else is legal): {', '.join(prims)}\n"
        "There is no `zero?`, no `cond`, no loops, no let* -- only what's listed "
        "above. Use recursion, not iteration.\n"
        "Write ONLY the function definition inside a ```netelpro fenced block, "
        "and nothing else."
    )


SYNTAX_PRIMER = _build_syntax_primer()


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


def temperature_for_attempt(attempt_n: int, base: float, step: float, cap: float) -> float:
    """Escalates sampling temperature across retries.

    Finding this answers (INFERENCE_REPAIR_LOOP_SPEC.md section 7): at a
    fixed low temperature, consecutive attempts were similar enough to each
    other that feedback-in-prompt fixed syntax errors but never moved the
    model off a repeated SEMANTIC mistake -- 5 attempts, same wrong logic,
    same failing case, verbatim. Escalating temperature gives each retry a
    genuinely different sample to check, instead of a near-repeat of the
    last one; if this doesn't help either, it's still a fact worth having,
    not a fact to avoid finding out.
    """
    return min(cap, base + (attempt_n - 1) * step)


def format_verdict(attempt: int, temperature: float, result: VerifyResult) -> str:
    if not result.compiled:
        return f"  attempt {attempt} (temp={temperature:.2f}): DID NOT COMPILE -- {result.error}"
    if result.passed:
        return f"  attempt {attempt} (temp={temperature:.2f}): PASSED all {result.cases_total} cases"
    return (
        f"  attempt {attempt} (temp={temperature:.2f}): compiled, but "
        f"{result.cases_passed}/{result.cases_total} cases passed -- {result.error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="sum_range", help="Module name under rlvr.tasks")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--base-temperature", type=float, default=0.2)
    parser.add_argument("--temperature-step", type=float, default=0.25,
                         help="Added to base-temperature per retry attempt (0 = disabled, matches section 6/7 behavior)")
    parser.add_argument("--temperature-cap", type=float, default=1.3)
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
        temperature = temperature_for_attempt(
            attempt_n, args.base_temperature, args.temperature_step, args.temperature_cap
        )
        prompt = build_prompt(task.DESCRIPTION_ES, task.SIGNATURE, prior_attempt, prior_error)
        out = llm(prompt, max_tokens=args.max_tokens, temperature=temperature, stop=["```"])
        raw = out["choices"][0]["text"]
        candidate = (extract_block(raw, fence="netelpro") or raw).strip()

        result = verify_program(candidate, task, num_cases=10, seed=0)
        results.append(result)
        print(format_verdict(attempt_n, temperature, result))

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
