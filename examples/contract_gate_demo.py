"""Live evidence: a compiled Netelpro contract blocking a real model's tokens
before they are ever sampled.

This is not a mockup. `NetelproMiniLLM` already wires a `NetelproStreamProcessor`
into every generation step (see `netelpro/neuro/minillm.py`'s `__init__` and
`stream_chat`) -- this script uses that exact same production path, just with
a narrower contract than the wide-open default, and prints the raw pre-gate
candidate next to the post-gate one at every step so the intervention is
visible instead of implicit.

The chain being exercised is real end to end: NetelproTransformer (PyTorch)
produces logits -> NetelproStreamProcessor.process_hf_logits() ->
NetelproLogitsProcessor.__call__() -> Gate.check() -> RuleFilter.decide() ->
compiled LLVM machine code via ctypes (netelpro/neuro/rules/action_boundary.sl).
The rule that runs is native code, not a Python re-implementation of one.

Run:
    python -m examples.contract_gate_demo
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from netelpro.neuro.minillm import NetelproMiniLLM

MODEL_DIR = Path("models/netelpro_mini_v1")


def step_line(step: int, raw_id: int, raw_txt: str, final_id: int, final_txt: str, blocked: bool) -> str:
    tag = "\033[91m[BLOCKED]\033[0m" if blocked else "\033[92m[ok]     \033[0m"
    raw_repr = raw_txt.replace("\n", "\\n")
    final_repr = final_txt.replace("\n", "\\n")
    if blocked:
        return (
            f"  {tag} step {step:3d} | model wanted id={raw_id:<4d} \"{raw_repr}\" "
            f"(NOT in contract range) -> never sampled; gate forced id={final_id:<4d} \"{final_repr}\""
        )
    return f"  {tag} step {step:3d} | id={final_id:<4d} \"{final_repr}\""


def run_window(minillm: NetelproMiniLLM, prompt: str, tokens: list[int], n_steps: int) -> list[int]:
    """Mirrors NetelproMiniLLM.stream_chat's per-step loop, but also captures
    the pre-gate argmax so the block is visible, not just its side effect."""
    curr = list(tokens)
    for step in range(1, n_steps + 1):
        ctx = curr[-minillm.config.block_size :]
        idx_tensor = torch.tensor([ctx], dtype=torch.long)
        logits, _, _ = minillm.model(idx_tensor, control_flags=1)
        next_logits = logits[0, -1, :].clone()

        raw_id = int(torch.argmax(next_logits).item())

        filtered = minillm.stream_processor.process_hf_logits(
            idx_tensor, next_logits.clone().unsqueeze(0)
        ).squeeze(0)
        final_id = int(torch.argmax(filtered).item())

        raw_txt = minillm.tokenizer.decode([raw_id], skip_special_tokens=False)
        final_txt = minillm.tokenizer.decode([final_id], skip_special_tokens=False)
        blocked = raw_id != final_id
        print(step_line(step, raw_id, raw_txt, final_id, final_txt, blocked))

        curr.append(final_id)
    return curr


def main() -> None:
    if not MODEL_DIR.exists():
        print(f"Model not found at {MODEL_DIR}. This demo needs the checkpoint committed in the repo.")
        sys.exit(1)

    print("=" * 78)
    print("NETELPRO CONTRACT GATE -- LIVE EVIDENCE ON A REAL MODEL")
    print("=" * 78)
    print(f"Loading {MODEL_DIR} ...")
    minillm = NetelproMiniLLM.from_pretrained(MODEL_DIR)
    vocab_size = minillm.config.vocab_size
    print(f"Loaded. vocab_size={vocab_size}, params={sum(p.numel() for p in minillm.parameters()):,}\n")

    prompt = "<|user|>\nHabla sobre el proyecto\n<|assistant|>\n"
    tokens = minillm.tokenizer.encode(prompt, add_special_tokens=False)
    if not tokens or tokens[0] != minillm.tokenizer.bos_token_id:
        tokens.insert(0, minillm.tokenizer.bos_token_id)

    # --- Phase 1: wide-open contract (default) -----------------------------
    print("-" * 78)
    print("PHASE 1 -- contract granted [0, vocab_size): everything the model")
    print("           wants to say is 'declared'. Expect zero blocks.")
    print("-" * 78)
    minillm.stream_processor.set_context(allowed_min=0, allowed_max=vocab_size - 1, safety_state=1)
    tokens = run_window(minillm, prompt, tokens, n_steps=8)

    # --- Phase 2: narrow contract --------------------------------------------
    narrow_max = vocab_size // 5
    print()
    print("-" * 78)
    print(f"PHASE 2 -- contract narrowed to [0, {narrow_max}) live, mid-conversation.")
    print("           Same compiled rule (netelpro/neuro/rules/action_boundary.sl),")
    print("           same native gate, just a tighter range. Any token the model")
    print("           prefers outside [0, {}) gets masked to -inf BEFORE sampling".format(narrow_max))
    print("           -- it is structurally impossible for it to reach the output.")
    print("-" * 78)
    minillm.stream_processor.set_context(allowed_min=0, allowed_max=narrow_max - 1, safety_state=1)
    tokens = run_window(minillm, prompt, tokens, n_steps=8)

    # --- Phase 3: emergency freeze (safety_state=0) -------------------------
    print()
    print("-" * 78)
    print("PHASE 3 -- safety_state=0: emergency freeze. This is the kill switch --")
    print("           every action_id, not just out-of-range ones, gets denied.")
    print("           This is the same flag the existing /inhibit command in")
    print("           examples/mini_llm_chat.py sets to 0.")
    print("-" * 78)
    minillm.stream_processor.set_context(allowed_min=0, allowed_max=vocab_size - 1, safety_state=0)
    tokens = run_window(minillm, prompt, tokens, n_steps=4)

    summary = minillm.stream_processor.get_summary()
    print()
    print("=" * 78)
    print("SUMMARY (netelpro.neuro.stream.NetelproStreamProcessor.get_summary)")
    print("=" * 78)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print("Every number above came from the real compiled gate running against a")
    print("real model's real logits -- no simulation, no post-hoc text parsing.")


if __name__ == "__main__":
    main()
