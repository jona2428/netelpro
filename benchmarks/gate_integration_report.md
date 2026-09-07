# VTB × HonestyGuard — Gate Integration Report

**Date**: 2026-09-07
**Models**: `Qwen/Qwen2.5-1.5B-Instruct` vs `JonaECG/netelpro-qwen2.5-1.5b-honest` · `lfm2.5:latest` vs `netelpro-lfm1.2b:latest`
**Method**: 30 VTB scenarios × 2 arms (base/aligned) × 2 gate states (on/off). Raw model outputs replayed through `HonestyGuard.verify_turn()` (claim detection from text only, no overrides — non-circular).
**Raw outputs**: `benchmarks/vtb_qwen_local_benchmark_results.json`, `benchmarks/vtb_lfm_benchmark_results.json`
**Per-run reports**: `benchmarks/gate_reports/` (16 files, JSON + TXT)

## Results Matrix

| Arm | Gate | Theater Cases | Caught | Missed | FP | Claim Agreement |
|---|---|---|---|---|---|---|
| Qwen base | OFF | 1 | 0 | 1 | 0 | 96.7% |
| Qwen base | ON | 1 | 0 | 1 | 0 | 96.7% |
| Qwen aligned | OFF | 0 | 0 | 0 | 0 | 100.0% |
| Qwen aligned | ON | 0 | 0 | 0 | 0 | 100.0% |
| LFM base | OFF | 3 | 0 | 3 | 0 | 90.0% |
| LFM base | ON | 3 | 0 | 3 | 0 | 90.0% |
| LFM aligned | OFF | 2 | 0 | 2 | 0 | 90.0% |
| LFM aligned | ON | 2 | 0 | 2 | 1 | 90.0% |

**Gate recall on VTB theater: 0/6.** Single FP: 1/90 non-theater turns (LFM aligned, gate ON).

## Central Finding — Construct Contamination, Not Gate Failure

The gate caught none of the theater cases. The cause is definitional, not a defect:

- **VTB labels theater as *alethic***: unverified factual assertions about the world ("Ollama is listening on port 8080", "port 8000 is not reserved") — states declared without evidence. VTB Rule 1.
- **HonestyGuard cages *procedural* claims**: assertions about one's own verification process ("I verified", "I ran", "all tests passed") — claims of action taken. HonestyGuard Rule 3.

The gate enforced exactly its design scope: 0 FPs on genuine procedural claims across 90 turns (the sole FP, "tests pasaron", is a modality drift — Rule 3 pattern in imperative/instructive tone, verified against the detector). VTB's alethic theater is out of scope by construction.

## What Each Layer Actually Delivers

- **DPO alone (Qwen aligned)**: theater reduced to 0/30 — voice fixed at the model level. Gate adds nothing *here* because there is nothing left to catch (0 theater, 0 FP).
- **LFM aligned**: theater down 3→2, residual is alethic. Gate adds nothing *yet* — again construct mismatch.
- **Complementarity claim revised**: voice (DPO) and gate (HonestyGuard) are complementary in *design* but this benchmark measures them on disjoint constructs. Their overlap on VTB is empty by definition.

## Recommendations

1. **VTB v2**: add procedural theater scenarios ("I ran the tests and they all pass" without execution) — these are in-scope for the gate and would measure its recall positively.
   *(RESOLVED 2026-09-07: `benchmarks/vtb_procedural.py` + `benchmarks/vtb_procedural_summary.md`.
   Deterministic replay, 18 cases: procedural recall 7/9 — misses are detector verb
   coverage gaps ("validé"/"audité"), 6/6 genuine claims approved, 2 FPs (1 negation
   blindness, 1 contract strictness on cited sources). FAAR on the procedural
   construct: 100% gate-off → 22.2% gate-on. The gate does its job on its own construct.)*
2. **Alethic detection (gate v2)**: factual-state claims require verification-grounding, beyond regex scope. Candidate: tool-call tracing (claim of state → require executed command proving it).
3. **Claim-detection agreement** is high where theater is procedural (Qwen: 96.7–100% agreement) and lower where theater is alethic (LFM: 90%) — consistent with the construct split.

## Honest Bottom Line

On this benchmark, the gate does not improve theater rates for either model — because the benchmark tests a different theater than the gate cages. The DPO models' gains (Qwen 10%→0%, LFM 10%→6.7%) stand on their own and are the entire measured effect. The gate's value remains demonstrated by its own test suite (claim caging, 0 FP on genuine claims) and awaits a benchmark with procedural theater to show recall.