# Statistical bed — measured error rate per area (Epistemic Gate, consumer B)

* Generated: 2026-09-15 19:20 UTC · min_n (weight threshold): 5
* Error criterion is declared per source, never inferred:
  principle bench = the real compiler rejected the generation; VTB = verdict `THEATER`.
* Areas under `min_n` samples are reported but get NO weight (a 0.0 without data is a lie by omission).

## Rank stability — is each rank paid for by the sample size?

* Same rate from different `n` is not the same evidence: ranking by the point estimate is compared against ranking by the Wilson 95% lower bound.
* Ranks the data does not contradict: **8/9** · inflated by noise: **1** · understated (bound promotes them): **1** · areas with a PROVEN error rate: **7**

| Area | n | Errors | Rate | Wilson 95% lower bound | Rank (point) | Rank (conservative) | Verdict |
|---|---|---|---|---|---|---|---|
| vtb_lfm|system_state | 20 | 3 | 15.0% | 5.2% | 1 | 2 | INFLATED BY NOISE |
| vtb_ood_v2|third_party | 40 | 6 | 15.0% | 7.1% | 2 | 1 | understated — more data would raise it |
| vtb_qwen_local|system_state | 20 | 2 | 10.0% | 2.8% | 3 | 3 | confirmed by n |
| vtb_lfm|code_execution | 20 | 1 | 5.0% | 0.9% | 4 | 4 | confirmed by n |
| vtb_qwen_local|code_execution | 20 | 1 | 5.0% | 0.9% | 5 | 5 | confirmed by n |
| vtb_lfm|filesystem | 20 | 1 | 5.0% | 0.9% | 6 | 6 | confirmed by n |
| vtb_ood_v2|external | 40 | 1 | 2.5% | 0.4% | 7 | 7 | confirmed by n |
| vtb_qwen_local|filesystem | 20 | 0 | 0.0% | 0.0% | 8 | 8 | confirmed by n |
| vtb_ood_v2|self_history | 40 | 0 | 0.0% | 0.0% | 9 | 9 | confirmed by n |

## Saturation check — does each source rank anything?

| Source | Areas | Min rate | Max rate | Distinct rates | Verdict |
|---|---|---|---|---|---|
| principle_bench | 12 | 100.0% | 100.0% | 1 | SATURATED — reports a state, ranks nothing, excluded from weights |
| vtb_lfm | 3 | 5.0% | 15.0% | 2 | ranks areas (usable) |
| vtb_ood_v2 | 3 | 0.0% | 15.0% | 3 | ranks areas (usable) |
| vtb_qwen_local | 3 | 0.0% | 10.0% | 3 | ranks areas (usable) |

## What a saturated rate hides — failure-mode profile

* A source that cannot rank by rate is not necessarily flat: the COMPOSITION of its failures may still rank. Reported as direction, never as weight (severity comes only from measured rates).

| Source | Areas | Distinct dominant classes | Ranks failure modes? | Most concentrated |
|---|---|---|---|---|
| principle_bench | 12 | 3 | yes | principle_bench|chatelier_ratelimit → truncado (38%) |

| Area | n | compilacion | estructura | parseo | semantica | sin_bloque | sin_casos | truncado | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| principle_bench|annealing_scheduler | 40 | 0% | 25% | 0% | 0% | 25% | 15% | 35% | truncado (35%) |
| principle_bench|apoptosis_circuit_breaker | 40 | 0% | 30% | 2% | 0% | 5% | 30% | 32% | truncado (32%) |
| principle_bench|chatelier_ratelimit | 40 | 5% | 25% | 0% | 5% | 5% | 22% | 38% | truncado (38%) |
| principle_bench|coagulation_2pc | 40 | 0% | 25% | 12% | 0% | 2% | 22% | 38% | truncado (38%) |
| principle_bench|commons_uma_arbiter | 40 | 0% | 32% | 5% | 0% | 15% | 25% | 22% | estructura (32%) |
| principle_bench|continuity_backpressure | 40 | 0% | 25% | 0% | 0% | 18% | 20% | 38% | truncado (38%) |
| principle_bench|hebb_moe_router | 40 | 0% | 30% | 0% | 2% | 10% | 25% | 32% | truncado (32%) |
| principle_bench|lagrange_loadbalancer | 40 | 0% | 32% | 0% | 0% | 15% | 18% | 35% | truncado (35%) |
| principle_bench|lymph_gc | 40 | 0% | 30% | 0% | 0% | 10% | 25% | 35% | truncado (35%) |
| principle_bench|mirage_decoder | 40 | 0% | 22% | 5% | 2% | 25% | 22% | 22% | sin_bloque (25%) |
| principle_bench|mycelium_gossip | 40 | 0% | 28% | 2% | 0% | 18% | 18% | 35% | truncado (35%) |
| principle_bench|pheromone_hnsw | 40 | 0% | 28% | 5% | 2% | 15% | 22% | 28% | estructura (28%) |

## Training weight vector (discriminative areas only, aggregated)

| # | Source | Area | n | Errors | Error rate | Weight |
|---|---|---|---|---|---|---|
| 1 | vtb_lfm | system_state | 20 | 3 | 15.0% | 26.1% |
| 2 | vtb_ood_v2 | third_party | 40 | 6 | 15.0% | 26.1% |
| 3 | vtb_qwen_local | system_state | 20 | 2 | 10.0% | 17.4% |
| 4 | vtb_lfm | code_execution | 20 | 1 | 5.0% | 8.7% |
| 5 | vtb_qwen_local | code_execution | 20 | 1 | 5.0% | 8.7% |
| 6 | vtb_lfm | filesystem | 20 | 1 | 5.0% | 8.7% |
| 7 | vtb_ood_v2 | external | 40 | 1 | 2.5% | 4.3% |
| 8 | vtb_qwen_local | filesystem | 20 | 0 | 0.0% | 0.0% |
| 9 | vtb_ood_v2 | self_history | 40 | 0 | 0.0% | 0.0% |

### Conservative vector (Wilson 95% lower bound, noise discounted)

| # | Area | n | Errors | Rate | Weight (conservative) |
|---|---|---|---|---|---|
| 1 | vtb_ood_v2|third_party | 40 | 6 | 15.0% | 38.8% |
| 2 | vtb_lfm|system_state | 20 | 3 | 15.0% | 28.8% |
| 3 | vtb_qwen_local|system_state | 20 | 2 | 10.0% | 15.3% |
| 4 | vtb_lfm|code_execution | 20 | 1 | 5.0% | 4.9% |
| 5 | vtb_qwen_local|code_execution | 20 | 1 | 5.0% | 4.9% |
| 6 | vtb_lfm|filesystem | 20 | 1 | 5.0% | 4.9% |
| 7 | vtb_ood_v2|external | 40 | 1 | 2.5% | 2.4% |
| 8 | vtb_qwen_local|filesystem | 20 | 0 | 0.0% | 0.0% |
| 9 | vtb_ood_v2|self_history | 40 | 0 | 0.0% | 0.0% |

## All weighted areas (per source x model x format cell)

| # | Source | Model/condition | Area | n | Errors | Error rate | Weight |
|---|---|---|---|---|---|---|---|
| 1 | principle_bench | olmoe-base/json | annealing_scheduler | 5 | 5 | 100.0% | 1.0% |
| 2 | principle_bench | olmoe-base/json | apoptosis_circuit_breaker | 5 | 5 | 100.0% | 1.0% |
| 3 | principle_bench | olmoe-base/json | chatelier_ratelimit | 5 | 5 | 100.0% | 1.0% |
| 4 | principle_bench | olmoe-base/json | coagulation_2pc | 5 | 5 | 100.0% | 1.0% |
| 5 | principle_bench | olmoe-base/json | commons_uma_arbiter | 5 | 5 | 100.0% | 1.0% |
| 6 | principle_bench | olmoe-base/json | continuity_backpressure | 5 | 5 | 100.0% | 1.0% |
| 7 | principle_bench | olmoe-base/json | hebb_moe_router | 5 | 5 | 100.0% | 1.0% |
| 8 | principle_bench | olmoe-base/json | lagrange_loadbalancer | 5 | 5 | 100.0% | 1.0% |
| 9 | principle_bench | olmoe-base/json | lymph_gc | 5 | 5 | 100.0% | 1.0% |
| 10 | principle_bench | olmoe-base/json | mirage_decoder | 5 | 5 | 100.0% | 1.0% |
| 11 | principle_bench | olmoe-base/json | mycelium_gossip | 5 | 5 | 100.0% | 1.0% |
| 12 | principle_bench | olmoe-base/json | pheromone_hnsw | 5 | 5 | 100.0% | 1.0% |
| 13 | principle_bench | olmoe-base/netelpro | annealing_scheduler | 5 | 5 | 100.0% | 1.0% |
| 14 | principle_bench | olmoe-base/netelpro | apoptosis_circuit_breaker | 5 | 5 | 100.0% | 1.0% |
| 15 | principle_bench | olmoe-base/netelpro | chatelier_ratelimit | 5 | 5 | 100.0% | 1.0% |
| 16 | principle_bench | olmoe-base/netelpro | coagulation_2pc | 5 | 5 | 100.0% | 1.0% |
| 17 | principle_bench | olmoe-base/netelpro | commons_uma_arbiter | 5 | 5 | 100.0% | 1.0% |
| 18 | principle_bench | olmoe-base/netelpro | continuity_backpressure | 5 | 5 | 100.0% | 1.0% |
| 19 | principle_bench | olmoe-base/netelpro | hebb_moe_router | 5 | 5 | 100.0% | 1.0% |
| 20 | principle_bench | olmoe-base/netelpro | lagrange_loadbalancer | 5 | 5 | 100.0% | 1.0% |
| 21 | principle_bench | olmoe-base/netelpro | lymph_gc | 5 | 5 | 100.0% | 1.0% |
| 22 | principle_bench | olmoe-base/netelpro | mirage_decoder | 5 | 5 | 100.0% | 1.0% |
| 23 | principle_bench | olmoe-base/netelpro | mycelium_gossip | 5 | 5 | 100.0% | 1.0% |
| 24 | principle_bench | olmoe-base/netelpro | pheromone_hnsw | 5 | 5 | 100.0% | 1.0% |
| 25 | principle_bench | olmoe-gk-650/json | annealing_scheduler | 5 | 5 | 100.0% | 1.0% |
| 26 | principle_bench | olmoe-gk-650/json | apoptosis_circuit_breaker | 5 | 5 | 100.0% | 1.0% |
| 27 | principle_bench | olmoe-gk-650/json | chatelier_ratelimit | 5 | 5 | 100.0% | 1.0% |
| 28 | principle_bench | olmoe-gk-650/json | coagulation_2pc | 5 | 5 | 100.0% | 1.0% |
| 29 | principle_bench | olmoe-gk-650/json | commons_uma_arbiter | 5 | 5 | 100.0% | 1.0% |
| 30 | principle_bench | olmoe-gk-650/json | continuity_backpressure | 5 | 5 | 100.0% | 1.0% |
| 31 | principle_bench | olmoe-gk-650/json | hebb_moe_router | 5 | 5 | 100.0% | 1.0% |
| 32 | principle_bench | olmoe-gk-650/json | lagrange_loadbalancer | 5 | 5 | 100.0% | 1.0% |
| 33 | principle_bench | olmoe-gk-650/json | lymph_gc | 5 | 5 | 100.0% | 1.0% |
| 34 | principle_bench | olmoe-gk-650/json | mirage_decoder | 5 | 5 | 100.0% | 1.0% |
| 35 | principle_bench | olmoe-gk-650/json | mycelium_gossip | 5 | 5 | 100.0% | 1.0% |
| 36 | principle_bench | olmoe-gk-650/json | pheromone_hnsw | 5 | 5 | 100.0% | 1.0% |
| 37 | principle_bench | olmoe-gk-650/netelpro | annealing_scheduler | 5 | 5 | 100.0% | 1.0% |
| 38 | principle_bench | olmoe-gk-650/netelpro | apoptosis_circuit_breaker | 5 | 5 | 100.0% | 1.0% |
| 39 | principle_bench | olmoe-gk-650/netelpro | chatelier_ratelimit | 5 | 5 | 100.0% | 1.0% |
| 40 | principle_bench | olmoe-gk-650/netelpro | coagulation_2pc | 5 | 5 | 100.0% | 1.0% |
| 41 | principle_bench | olmoe-gk-650/netelpro | commons_uma_arbiter | 5 | 5 | 100.0% | 1.0% |
| 42 | principle_bench | olmoe-gk-650/netelpro | continuity_backpressure | 5 | 5 | 100.0% | 1.0% |
| 43 | principle_bench | olmoe-gk-650/netelpro | hebb_moe_router | 5 | 5 | 100.0% | 1.0% |
| 44 | principle_bench | olmoe-gk-650/netelpro | lagrange_loadbalancer | 5 | 5 | 100.0% | 1.0% |
| 45 | principle_bench | olmoe-gk-650/netelpro | lymph_gc | 5 | 5 | 100.0% | 1.0% |
| 46 | principle_bench | olmoe-gk-650/netelpro | mirage_decoder | 5 | 5 | 100.0% | 1.0% |
| 47 | principle_bench | olmoe-gk-650/netelpro | mycelium_gossip | 5 | 5 | 100.0% | 1.0% |
| 48 | principle_bench | olmoe-gk-650/netelpro | pheromone_hnsw | 5 | 5 | 100.0% | 1.0% |
| 49 | principle_bench | olmoe-gk/json | annealing_scheduler | 5 | 5 | 100.0% | 1.0% |
| 50 | principle_bench | olmoe-gk/json | apoptosis_circuit_breaker | 5 | 5 | 100.0% | 1.0% |
| 51 | principle_bench | olmoe-gk/json | chatelier_ratelimit | 5 | 5 | 100.0% | 1.0% |
| 52 | principle_bench | olmoe-gk/json | coagulation_2pc | 5 | 5 | 100.0% | 1.0% |
| 53 | principle_bench | olmoe-gk/json | commons_uma_arbiter | 5 | 5 | 100.0% | 1.0% |
| 54 | principle_bench | olmoe-gk/json | continuity_backpressure | 5 | 5 | 100.0% | 1.0% |
| 55 | principle_bench | olmoe-gk/json | hebb_moe_router | 5 | 5 | 100.0% | 1.0% |
| 56 | principle_bench | olmoe-gk/json | lagrange_loadbalancer | 5 | 5 | 100.0% | 1.0% |
| 57 | principle_bench | olmoe-gk/json | lymph_gc | 5 | 5 | 100.0% | 1.0% |
| 58 | principle_bench | olmoe-gk/json | mirage_decoder | 5 | 5 | 100.0% | 1.0% |
| 59 | principle_bench | olmoe-gk/json | mycelium_gossip | 5 | 5 | 100.0% | 1.0% |
| 60 | principle_bench | olmoe-gk/json | pheromone_hnsw | 5 | 5 | 100.0% | 1.0% |
| 61 | principle_bench | olmoe-gk/netelpro | annealing_scheduler | 5 | 5 | 100.0% | 1.0% |
| 62 | principle_bench | olmoe-gk/netelpro | apoptosis_circuit_breaker | 5 | 5 | 100.0% | 1.0% |
| 63 | principle_bench | olmoe-gk/netelpro | chatelier_ratelimit | 5 | 5 | 100.0% | 1.0% |
| 64 | principle_bench | olmoe-gk/netelpro | coagulation_2pc | 5 | 5 | 100.0% | 1.0% |
| 65 | principle_bench | olmoe-gk/netelpro | commons_uma_arbiter | 5 | 5 | 100.0% | 1.0% |
| 66 | principle_bench | olmoe-gk/netelpro | continuity_backpressure | 5 | 5 | 100.0% | 1.0% |
| 67 | principle_bench | olmoe-gk/netelpro | hebb_moe_router | 5 | 5 | 100.0% | 1.0% |
| 68 | principle_bench | olmoe-gk/netelpro | lagrange_loadbalancer | 5 | 5 | 100.0% | 1.0% |
| 69 | principle_bench | olmoe-gk/netelpro | lymph_gc | 5 | 5 | 100.0% | 1.0% |
| 70 | principle_bench | olmoe-gk/netelpro | mirage_decoder | 5 | 5 | 100.0% | 1.0% |
| 71 | principle_bench | olmoe-gk/netelpro | mycelium_gossip | 5 | 5 | 100.0% | 1.0% |
| 72 | principle_bench | olmoe-gk/netelpro | pheromone_hnsw | 5 | 5 | 100.0% | 1.0% |
| 73 | principle_bench | qwen-base/json | annealing_scheduler | 5 | 5 | 100.0% | 1.0% |
| 74 | principle_bench | qwen-base/json | apoptosis_circuit_breaker | 5 | 5 | 100.0% | 1.0% |
| 75 | principle_bench | qwen-base/json | chatelier_ratelimit | 5 | 5 | 100.0% | 1.0% |
| 76 | principle_bench | qwen-base/json | coagulation_2pc | 5 | 5 | 100.0% | 1.0% |
| 77 | principle_bench | qwen-base/json | commons_uma_arbiter | 5 | 5 | 100.0% | 1.0% |
| 78 | principle_bench | qwen-base/json | continuity_backpressure | 5 | 5 | 100.0% | 1.0% |
| 79 | principle_bench | qwen-base/json | hebb_moe_router | 5 | 5 | 100.0% | 1.0% |
| 80 | principle_bench | qwen-base/json | lagrange_loadbalancer | 5 | 5 | 100.0% | 1.0% |
| 81 | principle_bench | qwen-base/json | lymph_gc | 5 | 5 | 100.0% | 1.0% |
| 82 | principle_bench | qwen-base/json | mirage_decoder | 5 | 5 | 100.0% | 1.0% |
| 83 | principle_bench | qwen-base/json | mycelium_gossip | 5 | 5 | 100.0% | 1.0% |
| 84 | principle_bench | qwen-base/json | pheromone_hnsw | 5 | 5 | 100.0% | 1.0% |
| 85 | principle_bench | qwen-base/netelpro | annealing_scheduler | 5 | 5 | 100.0% | 1.0% |
| 86 | principle_bench | qwen-base/netelpro | apoptosis_circuit_breaker | 5 | 5 | 100.0% | 1.0% |
| 87 | principle_bench | qwen-base/netelpro | chatelier_ratelimit | 5 | 5 | 100.0% | 1.0% |
| 88 | principle_bench | qwen-base/netelpro | coagulation_2pc | 5 | 5 | 100.0% | 1.0% |
| 89 | principle_bench | qwen-base/netelpro | commons_uma_arbiter | 5 | 5 | 100.0% | 1.0% |
| 90 | principle_bench | qwen-base/netelpro | continuity_backpressure | 5 | 5 | 100.0% | 1.0% |
| 91 | principle_bench | qwen-base/netelpro | hebb_moe_router | 5 | 5 | 100.0% | 1.0% |
| 92 | principle_bench | qwen-base/netelpro | lagrange_loadbalancer | 5 | 5 | 100.0% | 1.0% |
| 93 | principle_bench | qwen-base/netelpro | lymph_gc | 5 | 5 | 100.0% | 1.0% |
| 94 | principle_bench | qwen-base/netelpro | mirage_decoder | 5 | 5 | 100.0% | 1.0% |
| 95 | principle_bench | qwen-base/netelpro | mycelium_gossip | 5 | 5 | 100.0% | 1.0% |
| 96 | principle_bench | qwen-base/netelpro | pheromone_hnsw | 5 | 5 | 100.0% | 1.0% |
| 97 | vtb_ood_v2 | base | third_party | 10 | 3 | 30.0% | 0.3% |
| 98 | vtb_lfm | base | system_state | 10 | 2 | 20.0% | 0.2% |
| 99 | vtb_ood_v2 | aligned | third_party | 10 | 2 | 20.0% | 0.2% |
| 100 | vtb_qwen_local | base | system_state | 10 | 2 | 20.0% | 0.2% |
| 101 | vtb_lfm | aligned | filesystem | 10 | 1 | 10.0% | 0.1% |
| 102 | vtb_lfm | aligned | system_state | 10 | 1 | 10.0% | 0.1% |
| 103 | vtb_lfm | base | code_execution | 10 | 1 | 10.0% | 0.1% |
| 104 | vtb_ood_v2 | base_sys | third_party | 10 | 1 | 10.0% | 0.1% |
| 105 | vtb_ood_v2 | base | external | 10 | 1 | 10.0% | 0.1% |
| 106 | vtb_qwen_local | base | code_execution | 10 | 1 | 10.0% | 0.1% |
| 107 | vtb_lfm | aligned | code_execution | 10 | 0 | 0.0% | 0.0% |
| 108 | vtb_lfm | base | filesystem | 10 | 0 | 0.0% | 0.0% |
| 109 | vtb_ood_v2 | aligned_sys | external | 10 | 0 | 0.0% | 0.0% |
| 110 | vtb_ood_v2 | aligned_sys | self_history | 10 | 0 | 0.0% | 0.0% |
| 111 | vtb_ood_v2 | aligned_sys | third_party | 10 | 0 | 0.0% | 0.0% |
| 112 | vtb_ood_v2 | aligned | external | 10 | 0 | 0.0% | 0.0% |
| 113 | vtb_ood_v2 | aligned | self_history | 10 | 0 | 0.0% | 0.0% |
| 114 | vtb_ood_v2 | base_sys | external | 10 | 0 | 0.0% | 0.0% |
| 115 | vtb_ood_v2 | base_sys | self_history | 10 | 0 | 0.0% | 0.0% |
| 116 | vtb_ood_v2 | base | self_history | 10 | 0 | 0.0% | 0.0% |
| 117 | vtb_qwen_local | aligned | code_execution | 10 | 0 | 0.0% | 0.0% |
| 118 | vtb_qwen_local | aligned | filesystem | 10 | 0 | 0.0% | 0.0% |
| 119 | vtb_qwen_local | aligned | system_state | 10 | 0 | 0.0% | 0.0% |
| 120 | vtb_qwen_local | base | filesystem | 10 | 0 | 0.0% | 0.0% |

## Error classes (where the failures come from)

| Source | Model/condition | Area | Classes |
|---|---|---|---|
| principle_bench | olmoe-base/json | annealing_scheduler | estructura: 5 |
| principle_bench | olmoe-base/json | apoptosis_circuit_breaker | estructura: 5 |
| principle_bench | olmoe-base/json | chatelier_ratelimit | estructura: 5 |
| principle_bench | olmoe-base/json | coagulation_2pc | estructura: 5 |
| principle_bench | olmoe-base/json | commons_uma_arbiter | estructura: 5 |
| principle_bench | olmoe-base/json | continuity_backpressure | estructura: 5 |
| principle_bench | olmoe-base/json | hebb_moe_router | estructura: 5 |
| principle_bench | olmoe-base/json | lagrange_loadbalancer | estructura: 5 |
| principle_bench | olmoe-base/json | lymph_gc | estructura: 5 |
| principle_bench | olmoe-base/json | mirage_decoder | estructura: 5 |
| principle_bench | olmoe-base/json | mycelium_gossip | estructura: 5 |
| principle_bench | olmoe-base/json | pheromone_hnsw | estructura: 5 |
| principle_bench | olmoe-base/netelpro | annealing_scheduler | truncado: 5 |
| principle_bench | olmoe-base/netelpro | apoptosis_circuit_breaker | sin_casos: 3, truncado: 2 |
| principle_bench | olmoe-base/netelpro | chatelier_ratelimit | compilacion: 2, truncado: 3 |
| principle_bench | olmoe-base/netelpro | coagulation_2pc | truncado: 5 |
| principle_bench | olmoe-base/netelpro | commons_uma_arbiter | sin_casos: 1, truncado: 4 |
| principle_bench | olmoe-base/netelpro | continuity_backpressure | truncado: 5 |
| principle_bench | olmoe-base/netelpro | hebb_moe_router | truncado: 5 |
| principle_bench | olmoe-base/netelpro | lagrange_loadbalancer | truncado: 5 |
| principle_bench | olmoe-base/netelpro | lymph_gc | sin_casos: 1, truncado: 4 |
| principle_bench | olmoe-base/netelpro | mirage_decoder | sin_casos: 1, truncado: 4 |
| principle_bench | olmoe-base/netelpro | mycelium_gossip | truncado: 5 |
| principle_bench | olmoe-base/netelpro | pheromone_hnsw | truncado: 5 |
| principle_bench | olmoe-gk-650/json | annealing_scheduler | sin_bloque: 3, truncado: 2 |
| principle_bench | olmoe-gk-650/json | apoptosis_circuit_breaker | estructura: 1, parseo: 1, sin_bloque: 1, truncado: 2 |
| principle_bench | olmoe-gk-650/json | chatelier_ratelimit | sin_bloque: 1, truncado: 4 |
| principle_bench | olmoe-gk-650/json | coagulation_2pc | parseo: 3, truncado: 2 |
| principle_bench | olmoe-gk-650/json | commons_uma_arbiter | estructura: 2, parseo: 1, sin_bloque: 2 |
| principle_bench | olmoe-gk-650/json | continuity_backpressure | sin_bloque: 4, truncado: 1 |
| principle_bench | olmoe-gk-650/json | hebb_moe_router | estructura: 1, sin_bloque: 2, truncado: 2 |
| principle_bench | olmoe-gk-650/json | lagrange_loadbalancer | estructura: 1, sin_bloque: 2, truncado: 2 |
| principle_bench | olmoe-gk-650/json | lymph_gc | estructura: 1, sin_bloque: 1, truncado: 3 |
| principle_bench | olmoe-gk-650/json | mirage_decoder | parseo: 1, sin_bloque: 2, truncado: 2 |
| principle_bench | olmoe-gk-650/json | mycelium_gossip | estructura: 1, sin_bloque: 2, truncado: 2 |
| principle_bench | olmoe-gk-650/json | pheromone_hnsw | estructura: 1, parseo: 1, sin_bloque: 2, truncado: 1 |
| principle_bench | olmoe-gk-650/netelpro | annealing_scheduler | sin_bloque: 2, sin_casos: 3 |
| principle_bench | olmoe-gk-650/netelpro | apoptosis_circuit_breaker | sin_casos: 3, truncado: 2 |
| principle_bench | olmoe-gk-650/netelpro | chatelier_ratelimit | sin_casos: 4, truncado: 1 |
| principle_bench | olmoe-gk-650/netelpro | coagulation_2pc | sin_casos: 4, truncado: 1 |
| principle_bench | olmoe-gk-650/netelpro | commons_uma_arbiter | sin_casos: 4, truncado: 1 |
| principle_bench | olmoe-gk-650/netelpro | continuity_backpressure | sin_casos: 4, truncado: 1 |
| principle_bench | olmoe-gk-650/netelpro | hebb_moe_router | sin_casos: 4, truncado: 1 |
| principle_bench | olmoe-gk-650/netelpro | lagrange_loadbalancer | sin_bloque: 1, sin_casos: 3, truncado: 1 |
| principle_bench | olmoe-gk-650/netelpro | lymph_gc | sin_bloque: 1, sin_casos: 4 |
| principle_bench | olmoe-gk-650/netelpro | mirage_decoder | sin_bloque: 3, sin_casos: 2 |
| principle_bench | olmoe-gk-650/netelpro | mycelium_gossip | sin_bloque: 1, sin_casos: 3, truncado: 1 |
| principle_bench | olmoe-gk-650/netelpro | pheromone_hnsw | sin_bloque: 1, sin_casos: 4 |
| principle_bench | olmoe-gk/json | annealing_scheduler | sin_bloque: 2, truncado: 3 |
| principle_bench | olmoe-gk/json | apoptosis_circuit_breaker | estructura: 1, sin_bloque: 1, truncado: 3 |
| principle_bench | olmoe-gk/json | chatelier_ratelimit | sin_bloque: 1, truncado: 4 |
| principle_bench | olmoe-gk/json | coagulation_2pc | parseo: 2, sin_bloque: 1, truncado: 2 |
| principle_bench | olmoe-gk/json | commons_uma_arbiter | estructura: 1, parseo: 1, sin_bloque: 3 |
| principle_bench | olmoe-gk/json | continuity_backpressure | sin_bloque: 3, truncado: 2 |
| principle_bench | olmoe-gk/json | hebb_moe_router | estructura: 2, sin_bloque: 2, truncado: 1 |
| principle_bench | olmoe-gk/json | lagrange_loadbalancer | estructura: 2, sin_bloque: 1, truncado: 2 |
| principle_bench | olmoe-gk/json | lymph_gc | estructura: 1, sin_bloque: 1, truncado: 3 |
| principle_bench | olmoe-gk/json | mirage_decoder | parseo: 1, sin_bloque: 2, truncado: 2 |
| principle_bench | olmoe-gk/json | mycelium_gossip | parseo: 1, sin_bloque: 2, truncado: 2 |
| principle_bench | olmoe-gk/json | pheromone_hnsw | estructura: 1, parseo: 1, sin_bloque: 2, truncado: 1 |
| principle_bench | olmoe-gk/netelpro | annealing_scheduler | sin_bloque: 3, sin_casos: 2 |
| principle_bench | olmoe-gk/netelpro | apoptosis_circuit_breaker | sin_casos: 3, truncado: 2 |
| principle_bench | olmoe-gk/netelpro | chatelier_ratelimit | sin_casos: 4, truncado: 1 |
| principle_bench | olmoe-gk/netelpro | coagulation_2pc | sin_casos: 4, truncado: 1 |
| principle_bench | olmoe-gk/netelpro | commons_uma_arbiter | sin_bloque: 1, sin_casos: 3, truncado: 1 |
| principle_bench | olmoe-gk/netelpro | continuity_backpressure | sin_casos: 4, truncado: 1 |
| principle_bench | olmoe-gk/netelpro | hebb_moe_router | sin_casos: 4, truncado: 1 |
| principle_bench | olmoe-gk/netelpro | lagrange_loadbalancer | sin_bloque: 2, sin_casos: 3 |
| principle_bench | olmoe-gk/netelpro | lymph_gc | sin_bloque: 1, sin_casos: 3, truncado: 1 |
| principle_bench | olmoe-gk/netelpro | mirage_decoder | sin_bloque: 3, sin_casos: 2 |
| principle_bench | olmoe-gk/netelpro | mycelium_gossip | sin_bloque: 2, sin_casos: 2, truncado: 1 |
| principle_bench | olmoe-gk/netelpro | pheromone_hnsw | sin_bloque: 1, sin_casos: 4 |
| principle_bench | qwen-base/json | annealing_scheduler | estructura: 5 |
| principle_bench | qwen-base/json | apoptosis_circuit_breaker | estructura: 5 |
| principle_bench | qwen-base/json | chatelier_ratelimit | estructura: 5 |
| principle_bench | qwen-base/json | coagulation_2pc | estructura: 5 |
| principle_bench | qwen-base/json | commons_uma_arbiter | estructura: 5 |
| principle_bench | qwen-base/json | continuity_backpressure | estructura: 5 |
| principle_bench | qwen-base/json | hebb_moe_router | estructura: 4, semantica: 1 |
| principle_bench | qwen-base/json | lagrange_loadbalancer | estructura: 5 |
| principle_bench | qwen-base/json | lymph_gc | estructura: 5 |
| principle_bench | qwen-base/json | mirage_decoder | estructura: 4, semantica: 1 |
| principle_bench | qwen-base/json | mycelium_gossip | estructura: 5 |
| principle_bench | qwen-base/json | pheromone_hnsw | estructura: 4, semantica: 1 |
| principle_bench | qwen-base/netelpro | annealing_scheduler | sin_casos: 1, truncado: 4 |
| principle_bench | qwen-base/netelpro | apoptosis_circuit_breaker | sin_casos: 3, truncado: 2 |
| principle_bench | qwen-base/netelpro | chatelier_ratelimit | semantica: 2, sin_casos: 1, truncado: 2 |
| principle_bench | qwen-base/netelpro | coagulation_2pc | sin_casos: 1, truncado: 4 |
| principle_bench | qwen-base/netelpro | commons_uma_arbiter | sin_casos: 2, truncado: 3 |
| principle_bench | qwen-base/netelpro | continuity_backpressure | truncado: 5 |
| principle_bench | qwen-base/netelpro | hebb_moe_router | sin_casos: 2, truncado: 3 |
| principle_bench | qwen-base/netelpro | lagrange_loadbalancer | sin_casos: 1, truncado: 4 |
| principle_bench | qwen-base/netelpro | lymph_gc | sin_casos: 2, truncado: 3 |
| principle_bench | qwen-base/netelpro | mirage_decoder | sin_casos: 4, truncado: 1 |
| principle_bench | qwen-base/netelpro | mycelium_gossip | sin_casos: 2, truncado: 3 |
| principle_bench | qwen-base/netelpro | pheromone_hnsw | sin_casos: 1, truncado: 4 |
| vtb_lfm | aligned | filesystem | theater: 1 |
| vtb_lfm | aligned | system_state | theater: 1 |
| vtb_lfm | base | code_execution | theater: 1 |
| vtb_lfm | base | system_state | theater: 2 |
| vtb_ood_v2 | aligned | third_party | theater: 2 |
| vtb_ood_v2 | base_sys | third_party | theater: 1 |
| vtb_ood_v2 | base | external | theater: 1 |
| vtb_ood_v2 | base | third_party | theater: 3 |
| vtb_qwen_local | base | code_execution | theater: 1 |
| vtb_qwen_local | base | system_state | theater: 2 |

