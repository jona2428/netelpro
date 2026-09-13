# Netelpro Principle Benchmark — tokens vs JSON + error rate

* Fecha: 2026-09-13 04:25 · muestras por celda: 5 · temp: 0.8 · seeds deterministicos por (tarea, formato, k)
* Fail-closed: generacion sin bloque extraible = rechazo. El veredicto netelpro lo emite el compilador real (compile_filter + verify_int diferencial); el JSON lo juzga un validador de la misma severidad (schema exacto + cobertura + semantica).

## Parte A — Principio #2: economia de tokens

Muestra: 40 reglas de dataset_v2 (mismos nombres y tabla en ambos formatos). Tokens contados por el tokenizador REAL de cada modelo (Ollama prompt_eval_count, raw).

| Modelo | netelpro_contract | netelpro_contract_cases | json_indent | json_compact |
|---|---|---|---|---|
| olmoe-base | 134.5 (x0.4) | 215.5 (x0.64) | 336.5 (x1.0) | 159.5 (x0.474) |
| olmoe-gk | 134.5 (x0.4) | 215.5 (x0.64) | 336.5 (x1.0) | 159.5 (x0.474) |
| qwen-base | 147.2 (x0.497) | 234.2 (x0.791) | 296.2 (x1.0) | 127.2 (x0.429) |

Caracteres medios: netelpro_contract=254.8, netelpro_contract_cases=359.8, json_indent=873.8, json_compact=351.8
