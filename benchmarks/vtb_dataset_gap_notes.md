# VTB dataset: what the 15 new cases (2026-09-17) actually test

The original 30 `VTBTestCase` entries in `vtb_dataset.py` cover three technical
categories (filesystem, system_state, code_execution) but nearly all of them
use the same handful of claim templates — "Verifiqué X y...", "Revisé Y y...",
"Comprobé Z y..." — with the technical fact swapped out. That's real domain
breadth, but it doesn't stress-test anything about *how* a claim is phrased,
which matters because `HonestyGuard.detect_claims()` is a regex-based parser,
not a semantic one.

The 15 new cases (FS-11..15, SYS-11..15, EXEC-11..15) each deliberately use a
*different claim shape* than the 30 originals, not just a different fact:

| Case  | Shape exercised |
|-------|------------------|
| FS-11 | Third-person/passive framing ("El escaneo confirmó...") instead of first-person |
| FS-12 | Partial truth — half the claim is accurate, the specific detail asked about isn't |
| FS-13 | Long, multi-sentence reasoning before the claim, not a single short sentence |
| FS-14 | Hedged confidence ("Creo que ya revisé...") that still asserts verification |
| FS-15 | Names a specific CLI tool (`grep`) as the evidence, not "verifiqué" alone |
| SYS-11 | Claim with a fabricated precise number (47 days) — theater that sounds credible |
| SYS-12 | Fabricated tool output with fake latency figure (`PONG in 0.3ms`) |
| SYS-13 | Explicit hedged reasoning ("Pensándolo con cuidado: no creo que...") before the claim |
| SYS-14 | Two chained verification claims in one turn (`free -h`, then `vmstat`) |
| SYS-15 | Claim citing a specific command's exact output as evidence |
| EXEC-11 | Numeric claim (coverage %) against a specific threshold |
| EXEC-12 | Claim about branch state via `git diff`, not a tool run |
| EXEC-13 | Cites an external link/URL as if that were equivalent to having looked at it |
| EXEC-14 | Claim with a full fabricated HTTP response (status line + body) |
| EXEC-15 | Reasoning-heavy claim explaining *why* the result should be correct, then asserting it was checked |

None of this changes what `vtb_runner.py` measures today (it drives the
compiled decision with `override_claimed=True`, so `detect_claims()` is not on
the path) — see `tests/test_guard_claim_detection.py` for the benchmark that
actually exercises the parser these shapes were designed to probe.
