# Netelpro Principle Benchmark — tokens vs JSON + error rate

* Fecha: 2026-09-14 05:04 · muestras por celda: 5 · temp: 0.8 · seeds deterministicos por (tarea, formato, k)
* Fail-closed: generacion sin bloque extraible = rechazo. El veredicto netelpro lo emite el compilador real (compile_filter + verify_int diferencial); el JSON lo juzga un validador de la misma severidad (schema exacto + cobertura + semantica).

## Parte A — Principio #2: economia de tokens

Muestra: 40 reglas de dataset_v2 (mismos nombres y tabla en ambos formatos). Tokens contados por el tokenizador REAL de cada modelo (Ollama prompt_eval_count, raw).

| Modelo | netelpro_contract | netelpro_contract_cases | json_indent | json_compact |
|---|---|---|---|---|
| olmoe-base | 134.5 (x0.4) | 215.5 (x0.64) | 336.5 (x1.0) | 159.5 (x0.474) |
| olmoe-gk | 134.5 (x0.4) | 215.5 (x0.64) | 336.5 (x1.0) | 159.5 (x0.474) |
| qwen-base | 147.2 (x0.497) | 234.2 (x0.791) | 296.2 (x1.0) | 127.2 (x0.429) |

Caracteres medios: netelpro_contract=254.8, netelpro_contract_cases=359.8, json_indent=873.8, json_compact=351.8

## Parte B — Principio #1: tasa de regla maquina-valida

12 tareas frescas (pool-4 de nombres, jamas visto en el train) x 2 formatos. Valida = compila/parsea, cobertura completa, y la tabla implementa exactamente la ley pedida.

| Modelo | netelpro valida | JSON valida | delta (np - json) |
|---|---|---|---|
| qwen-base | 0/60 (0%) | 0/60 (0%) | 0% |
| olmoe-base | 0/60 (0%) | 0/60 (0%) | 0% |
| olmoe-gk | 0/60 (0%) | 0/60 (0%) | 0% |
| olmoe-gk-650 | 0/60 (0%) | 0/60 (0%) | 0% |

### Distribucion de errores por clase

- **qwen-base / netelpro**: truncado: 38, sin_casos: 20, semantica: 2
- **qwen-base / json**: estructura: 57, semantica: 3
- **olmoe-base / netelpro**: truncado: 52, sin_casos: 6, compilacion: 2
- **olmoe-base / json**: estructura: 60
- **olmoe-gk / netelpro**: sin_casos: 38, sin_bloque: 13, truncado: 9
- **olmoe-gk / json**: truncado: 25, sin_bloque: 21, estructura: 8, parseo: 6
- **olmoe-gk-650 / netelpro**: sin_casos: 42, sin_bloque: 9, truncado: 9
- **olmoe-gk-650 / json**: truncado: 23, sin_bloque: 22, estructura: 8, parseo: 7

### Por familia (cama estadistica — insumo Fase E del gate epistemico)

| Familia | estrato | olmoe-base np | olmoe-gk np | qwen-base np |
|---|---|---|---|---|
| annealing_scheduler | 1_on | 0/5 | 0/5 | 0/5 |
| apoptosis_circuit_breaker | 4_on | 0/5 | 0/5 | 0/5 |
| chatelier_ratelimit | 7_on | 0/5 | 0/5 | 0/5 |
| coagulation_2pc | 1_on | 0/5 | 0/5 | 0/5 |
| commons_uma_arbiter | 4_on | 0/5 | 0/5 | 0/5 |
| continuity_backpressure | 7_on | 0/5 | 0/5 | 0/5 |
| hebb_moe_router | 1_on | 0/5 | 0/5 | 0/5 |
| lagrange_loadbalancer | 4_on | 0/5 | 0/5 | 0/5 |
| lymph_gc | 7_on | 0/5 | 0/5 | 0/5 |
| mirage_decoder | 1_on | 0/5 | 0/5 | 0/5 |
| mycelium_gossip | 4_on | 0/5 | 0/5 | 0/5 |
| pheromone_hnsw | 7_on | 0/5 | 0/5 | 0/5 |

### Lectura pareada (OLMoE base vs goliath-killer)

- netelpro: base 0/60 (0%) -> gk 0/60 (0%)
- JSON: base 0/60 (0%) -> gk 0/60 (0%)

## Diagnóstico de la Parte B (sesión 2026-09-14, Teo)

**Resultado: 0/360 válidas en AMBOS formatos, incluyendo el finetune goliath-killer.** Este resultado no es artefacto del harness; escalera de falsificación ejecutada:

1. **No es presupuesto de tokens**: con num_predict 650 y 900 los resultados son idénticos (sin_casos 42/60 persiste en gk a 650).
2. **No es formato de prompt**: reconstruyendo el prompt EXACTO del entrenamiento (sin ejemplo, estilo dataset_v2) + el GOLIATH_SYSTEM_PROMPT de entrenamiento, gk sigue sin emitir truth-tables.
3. **No es muestreo**: a temperatura 0 con prompt visto de dataset_v2 la alucinación es determinista.
4. **Control de memorización fallido**: con el prompt exacto de una tarea de entrenamiento, gk alucina pseudo-código (TypeScript/Python/Lisp falsos) dentro de ```netelpro```; jamás reproduce el formato truth-table.

**Mecanismo real de los errores**: el modelo copia el contrato dorado del ejemplo del prompt (x,y,z) y rambla en prosa hasta el corte o EOS. La clase 'truncado' es mayormente un artefacto de taxonomía: el regex FENCE_CASES ve el substring "netelpro-cases" dentro de un bloque Lisp sin fence y lo etiqueta "fence sin cerrar". La decapitación real es marginal.

**Conclusión**: el finetune absorbió la persona y el ritual (<thought>, 〈tool〉, encabezados en negrita) pero cero gramática netelpro. Hipótesis a revisar antes del próximo run: capacidad del LoRA (rank) o learning rate insuficientes para inyectar una gramática nueva, o formato de template con mismatch residual. La economía de tokens (Parte A) se mantiene: contrato netelpro 0.40-0.50x vs json_indent, json_compact 0.43-0.47x.