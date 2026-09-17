"""Live evidence: the same compiled Netelpro contract gate, but hooked into
a REAL local model through llama-cpp-python -- the actual deployment path
for a GGUF file loaded in-process (LM Studio's own HTTP server does not
expose per-token logits, so it cannot be gated this way; see the session
notes in docs/STATE_TRACKING_GATE_SPEC.md's sibling conversation. This
script is the path that DOES work: load the .gguf directly, attach
NetelproStreamProcessor.llama_cpp_processor as a real logits_processor).

Requires the optional `llama-cpp-python` dependency (not installed by
default -- it's a real, separate native build):

    pip install llama-cpp-python

Run against any local GGUF, e.g. a small instruct model:

    python -m examples.contract_gate_llama_cpp_demo --model "C:/path/to/model.gguf"

What changed vs examples/contract_gate_demo.py (the toy 182K-param model):
this is a real, capable local model (1-1.5B+ params), and the gate's
llama.cpp adapter used to be a plain Python loop evaluating the compiled
rule once per vocabulary token, every decoding step -- measured at ~370ms of
overhead PER TOKEN against a 152k-token vocabulary (netelpro/neuro/stream.py
git history has the numbers). That defeated the entire point of a
low-latency native gate. It's now vectorized (same slicing
NetelproVectorKernel already used for the PyTorch path, applied to the
numpy arrays llama-cpp-python actually hands the processor) -- this script
prints the per-token overhead live so the fix is evidence, not a claim.
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


def run_phase(llm: Llama, sp: NetelproStreamProcessor, prompt: str, max_tokens: int, label: str) -> None:
    lp = LogitsProcessorList([sp.llama_cpp_processor])
    t0 = time.perf_counter()
    out = llm(prompt, max_tokens=max_tokens, temperature=0.0, logits_processor=lp)
    dt = time.perf_counter() - t0
    text = out["choices"][0]["text"]
    per_token_ms = (dt / max(1, max_tokens)) * 1000.0
    print(f"\n{label}")
    print(f"  output: {text!r}")
    print(f"  wall time: {dt:.2f}s ({per_token_ms:.1f}ms/token, model+gate combined)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a local .gguf file")
    parser.add_argument("--prompt", default="Q: What is 2+2?\nA:")
    parser.add_argument("--max-tokens", type=int, default=12)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No file at {model_path}")
        sys.exit(1)

    print("=" * 78)
    print("NETELPRO CONTRACT GATE -- LIVE EVIDENCE ON A REAL LOCAL MODEL")
    print(f"Model: {model_path.name}")
    print("=" * 78)

    t0 = time.perf_counter()
    llm = Llama(model_path=str(model_path), n_ctx=512, verbose=False)
    vocab_size = llm.n_vocab()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s. vocab_size={vocab_size:,}\n")

    # --- Phase 1: baseline, no gate at all -----------------------------
    t0 = time.perf_counter()
    out = llm(args.prompt, max_tokens=args.max_tokens, temperature=0.0)
    dt = time.perf_counter() - t0
    print("PHASE 1 -- no gate attached (raw model speed, baseline)")
    print(f"  output: {out['choices'][0]['text']!r}")
    print(f"  wall time: {dt:.2f}s ({dt / args.max_tokens * 1000:.1f}ms/token)")

    # --- Phase 2: wide-open contract, gate attached ------------------------
    sp = NetelproStreamProcessor(allowed_min=0, allowed_max=vocab_size - 1, safety_state=1)
    run_phase(
        llm, sp, args.prompt, args.max_tokens,
        f"PHASE 2 -- gate attached, wide-open contract [0, {vocab_size - 1}]\n"
        f"           (should track Phase 1's speed -- this is the vectorized fast path)",
    )
    print(f"  gate overhead: {sp.get_summary()['avg_pruning_latency_us']:.2f} us/token")

    # --- Phase 3: narrow contract, live evidence of blocking ------------
    narrow_max = vocab_size // 20
    sp2 = NetelproStreamProcessor(allowed_min=0, allowed_max=narrow_max, safety_state=1)
    run_phase(
        llm, sp2, args.prompt, args.max_tokens,
        f"PHASE 3 -- contract narrowed to [0, {narrow_max}] ({narrow_max / vocab_size * 100:.1f}% of vocab).\n"
        f"           Every token this model would prefer outside that range is\n"
        f"           masked to -inf before sampling -- structurally impossible\n"
        f"           for the model to emit it, not filtered after the fact.",
    )
    summary2 = sp2.get_summary()
    print(f"  gate overhead: {summary2['avg_pruning_latency_us']:.2f} us/token")
    print(f"  total tokens pruned across the run: {summary2['total_pruned_tokens']:,}")

    # --- Phase 4: emergency freeze ------------------------------------------
    sp3 = NetelproStreamProcessor(allowed_min=0, allowed_max=vocab_size - 1, safety_state=0)
    run_phase(
        llm, sp3, args.prompt, min(4, args.max_tokens),
        "PHASE 4 -- safety_state=0: emergency freeze, every candidate denied",
    )

    print("\n" + "=" * 78)
    print("Every number above came from a real model loaded in-process and the")
    print("real compiled gate (netelpro.gate.Gate -> LLVM native code via ctypes) --")
    print("no mock logits, no HTTP layer in between.")


if __name__ == "__main__":
    main()
