# Fase 3: Effect rows — spec de implementación

**Fecha:** 2026-09-08 · **Base:** design doc §3 (2026-09-08) + Sesión 4 del plan del proyecto (per-function effects)
**Draft:** arquitecto (nemotron-3-ultra) · **Revisión y correcciones:** Teo
**Estado:** esperando ratificación D1-D12 por Jona

---

## §0 Resumen

Efectos declarados **por función**: cada `defn` puede declarar `: (effects (VERBO "patrón")+)` — el conjunto de efectos IO que su cuerpo está autorizado a realizar. El compilador rechaza el programa si el cuerpo (o un callee) excede lo declarado. **Esta fase ES la Sesión 4 del plan del proyecto** (per-function effects) con sintaxis declarada — no es trabajo nuevo paralelo.

**Hallazgo de review (cambia el diseño del draft):** `effects.py` ya implementa inferencia de efectos por función (fixpoint monotónico, cap 100, sobre el call graph estático — sin first-class calls, el análisis es completo). Hoy trabaja sobre **strings de capability** (`"io"`). La Fase 3 NO crea un sistema nuevo: **extiende el existente** de capability-strings a pares estructurados `(verb, pattern)`. La coexistencia con `(grant io)` top-level ya existe como mecanismo (`collect_grants`) y se conserva.

**Alcance v1:** verbo + patrón de path con **igualdad literal** (sin álgebra de globs). Sin condiciones dinámicas ("solo si el usuario confirmó" sigue siendo trabajo del host/gate). Costo runtime: **cero** — chequeo solo en compilación, sin representación en codegen.

## §1 Sintaxis exacta

### 1.1 Posición en defn

```
(defn nombre ((p : Tipo) ...)
  : (effects (VERBO "patrón")+)
  cuerpo)
```

La cláusula va **después de params, antes del cuerpo**. El token `:` ya existe (Fase 1) — se reutiliza. **Cero tokens nuevos.**

### 1.2 Gramática

```
effects_clause ::= ":" "(" "effects" effect_row+ ")"
effect_row     ::= "(" VERB STRING ")"
VERB           ::= "read" | "write" | "delete" | "network" | "io"
STRING         ::= patrón literal (p.ej. "./config/*", "*")
```

**VERIFICACIÓN EN DISCO (Teo):** el sistema vigente (`caps.py`) tiene la capability `"io"` como único verbo, requerido por `print` (`CAPABILITY_REQUIREMENTS = {"print": {"io"}}`, derivable de `arity_table.json`). El verbo `io` se conserva como el "verbo general" existente; `read/write/delete/network` son los refinados nuevos. Un primitivo `read-file` declarará `(read "patrón")` cuando exista en `arity_table.json`.

### 1.3 Ejemplos

```lisp
(defn leer-config (path : Str)
  : (effects (read "./config/*"))
  (print "leyendo"))            ; print requiere io — ver D10

(defn borrar-todo ()
  : (effects (write "*") (delete "*"))
  (print "borrando"))

(defn pura (x : Int)            ; SIN cláusula = conjunto vacío = pura
  (+ x 1))
```

### 1.4 defn sin cláusula

Efectos = ∅ (pura por declaración). Llamar a una función con efectos desde una pura = **error de compilación** (composición, D2). Nada implícito.

## §2 Decisiones semánticas D1-D12

**D1 — Chequeo solo compile-time.** Costo runtime 0, sin representación en codegen. El host (zone_policy) mantiene el enforcement dinámico; Netelpro verifica lo estático. *Rechazado:* effect tracking en runtime (overhead, rompe el modelo de costo cero).

**D2 — Composición: `effects(caller) ⊇ effects(callee)` para cada llamado directo.** Chequeo por par `(verb, pattern)` con **igualdad literal** en v1. Evita la evasión trivial (envolver IO en helper sin declarar). *Rechazado:* subconjunto semántico de globs (`"./config/*" ⊆ "./*"`) — requiere motor de matching de globs, ambigüedad, se pospone a v2 con el vocabulario de `zone_rule_generator.py`.

**D3 — Primitivas effectuales vía tabla existente.** `caps.py` deriva de `arity_table.json` (`primitives[*].capabilities`) — fuente única de verdad, no duplicación. Hoy: `print → io`. Futuro: `read-file → read`, `write-file → write`, etc. *Rechazado:* hardcodear la lista en el parser.

**D4 — `fn` y truth-table: effectless por construcción.** `fn` anónimo no tiene sintaxis para declarar efectos (sus efectos se atribuyen a la defn que lo encierra — decisión YA vigente en `effects.py`, confirmada en disco). Truth-table desugar a if-chain puro: efectos prohibidos. *Rechazado:* efectos en lambdas (requeriría effect polymorphism).

**D5 — Errores con coordenadas del call ofensivo + efecto faltante + hint accionable.** Formato: `line X, col Y: call to 'read-file' requires effect (read "./data/*") not declared in function 'foo' — add : (effects (read "./data/*"))`.

**D6 — Filas duplicadas en la misma cláusula: warning + deduplicación.** `(effects (read "a") (read "a"))` compila con warning. *Rechazado:* error duro (demasiado estricto para edición interactiva).

**D7 — Verbo desconocido: error de compilación** con lista de verbos válidos. Fail-fast en la declaración.

**D8 — Patrón vacío `""`: error.** Siendo literal equality, un patrón vacío jamás matchearía nada — typo probable.

**D9 — Recursión mutua: cada participante declara sus efectos explícitamente.** El fixpoint existente propaga y verifica consistencia; si una fn del ciclo no declara → error. *Rechazado:* inferir en ciclos (rompe la declaratividad).

**D10 — `(grant io)` top-level coexiste.** `grant` = "este archivo puede usar estas capabilities" (mecanismo vigente, verificado); effect rows = "esta función declara exactamente estos efectos". Chequeo vigente de caps NO se elimina — migración gradual sin breaking change. *Rechazado:* eliminar `grant` (innecesario).

**D11 — `def` constantes: sin cláusula de efectos.** Es binding de valor, no función con cuerpo ejecutable de calls.

**D12 — Integración gate/host: metadata export, no enforcement.** En v1 Netelpro exporta las effect rows como metadata (para que zone_policy/policy-as-code las consuma después). No bloquea compilación ni acopla el compilador a policy runtime.

## §3 Estrategia de compilación

```
Lexer → Parser (defn con cláusula effects → Defn.effects)
      → Effect Registry (single pass sobre defns)
      → Fixpoint existente en effects.py (cap 100) — extendido a pares (verb, pattern)
      → Composition check: caller ⊇ callee por arista del call graph
      → Reporte agregado prosecutorial (estilo CapError/EffectError existentes)
      → Codegen SIN cambios (los efectos no existen en runtime)
```

Estructuras: `EffectRow` (frozen dataclass: verb, pattern, line, col); `Defn.effects: list[EffectRow]`; registry `dict[str, frozenset[EffectRow]]`.

**Impacto LLVM: ninguno.** `codegen.py` sin tocar.

## §4 Touchpoints exactos

| Archivo | Cambio | Estado |
|---|---|---|
| `netelpro/caps.py` | MENOR: verbos (`VERBS = {"read","write","delete","network","io"}`), mapeo primitiva→EffectRow en la derivación de `arity_table.json` | verificado en disco (estructura real confirmada) |
| `parser.py` | MAYOR: parsear cláusula en `build_node` case `defn`, poblar `Defn.effects`, registrar en `collect_defns` | touchpoint principal |
| `ast_nodes.py` | MENOR: `EffectRow` + campo `effects` en `Defn` | nuevo |
| `netelpro/effects.py` | MAYOR: `infer_effects` extendido de strings a pares `(verb, pattern)`, composition check caller⊇callee por arista | **verificado en disco**: fixpoint cap-100 ya existe |
| `evaluator.py` | NINGUNO | effects son compile-only |
| `codegen.py` | NINGUNO | sin representación runtime |
| `netelpro/__main__.py` | MENOR: invocar effect checking tras parse, antes de eval/codegen | integración CLI |
| `spec/arity_table.json` | MENOR (v2): campo `effects` para primitivas IO futuras | hoy solo `print` con `io` |

## §5 Contratos de frontera B1-B8

| ID | Contrato | Comportamiento |
|---|---|---|
| B1 | defn con cláusula válida, IO dentro de lo declarado | ACCEPT |
| B2 | defn sin cláusula llamando primitiva effectual | REJECT: set vacío |
| B3 | Call a primitiva fuera de lo declarado | REJECT: `line X, col Y: call to 'read-file' requires effect (read "./data/*") not declared in function 'foo' — add : (effects (read "./data/*"))` |
| B4 | defn effectual llamado desde pura | REJECT: `call to 'leer-config' brings effects (read "./config/*") but caller 'main' declares none` |
| B5 | caller con subconjunto estricto | REJECT: `... declares only (write "*") — missing (delete "*")` |
| B6 | verbo desconocido | REJECT: `unknown effect verb 'execute' — valid verbs: read, write, delete, network, io` |
| B7 | patrón vacío | REJECT: `effect pattern cannot be empty string` |
| B8 | filas duplicadas | WARNING + deduplicación |

## §6 Integración con gate/host (informacional v1)

`netelpro_gate.py` y `zone_policy.classify()` hacen enforcement dinámico en runtime (tripwire de texto). En v1 Netelpro exporta las effect rows como metadata para: (1) validar coherencia con lo que `zone_rule_generator.py` permite (globs verdes/amarillas/rojas), (2) policy-as-code futuro (rechazar deployment si effect rows piden zona roja sin approval). No bloquea compilación (D12).## §7 Plan de implementación (7 pasos)

| # | Paso | Verificación |
|---|---|---|
| 1 | AST: `EffectRow` + `Defn.effects` | round-trip |
| 2 | Parser: cláusula `: (effects ...)` en defn (validación verbos/patrón/duplicados) | tests de sintaxis |
| 3 | caps.py: `VERBS`, derivación primitiva→EffectRow desde arity_table | tests de tabla |
| 4 | effects.py: infer fixpoint extendido a pares + check `declared ⊇ inferred` | tests de inferencia |
| 5 | effects.py: composition check `caller ⊇ callee` por arista | tests de composición |
| 6 | `__main__.py`: integración del checking en pipeline CLI | end-to-end |
| 7 | Suite completa: 12 tests nuevos + regresión Fase 1 cero | pytest verde |

## §8 Contrato de tests (12 nombrados)

| Test | Descripción | Esperado |
|---|---|---|
| `test_effect_declared_allows_io` | defn con `(io "stdout")` llama `print` | PASS |
| `test_effect_missing_rejects_io` | defn pura llama `print` | REJECT con coords |
| `test_composition_caller_superset` | caller declara menos que callee | REJECT |
| `test_composition_exact_match_ok` | caller ⊇ callee exacto | PASS |
| `test_unknown_verb_rejected` | `(execute "*")` | REJECT |
| `test_empty_pattern_rejected` | `(read "")` | REJECT |
| `test_duplicate_effect_warning` | filas duplicadas | PASS + warning |
| `test_effect_in_truth_table_forbidden` | print en rama de truth-table | REJECT |
| `test_lambda_no_effects_syntax` | `fn` con cláusula effects | REJECT de sintaxis |
| `test_recursive_mutual_effects_declared` | even/odd mutuas con efectos declarados | PASS |
| `test_recursive_undeclared_error` | recursión con print sin declarar | REJECT |
| `test_regression_phase1_unchanged` | suite Fase 1 completa | PASS |

## §9 Riesgos y deuda

| Riesgo | Mitigación / deuda |
|---|---|
| Igualdad literal vs globs reales (v1) | Deuda documentada; v2: vocabulario de zone_rule_generator.py para matching semántico |
| Sin condiciones dinámicas | Sigue siendo runtime/host (tripwire vigente); effect rows cubre solo lo estáticamente decidible |
| Host sniffing como tripwire se conserva | Defense in depth — effect rows reduce superficie, no elimina el runtime check |
| Primitivas IO futuras ausentes hoy | arity_table.json crece cuando existan (`read-file`, etc.); derivación ya preparada |
| Fixpoint no converge | Cap 100 ya existente en effects.py; error claro si excede |

## §10 Ratificación — D1-D12 (una línea cada una)

1. **D1**: chequeo solo compile-time, costo runtime cero, sin codegen.
2. **D2**: composición `caller ⊇ callee` con igualdad literal de `(verb, pattern)` en v1.
3. **D3**: primitivas effectuales vía caps.py/arity_table.json (fuente única).
4. **D4**: `fn` y truth-table effectless por construcción.
5. **D5**: errores con coordenadas del call + efecto faltante + hint accionable.
6. **D6**: duplicados = warning + deduplicación.
7. **D7**: verbo desconocido = error con lista válida.
8. **D8**: patrón vacío = error.
9. **D9**: recursión mutua exige efectos declarados en cada participante.
10. **D10**: `(grant io)` top-level coexiste con effect rows (chequeo vigente no se elimina).
11. **D11**: `def` constantes sin cláusula de efectos.
12. **D12**: integración host = metadata export, no enforcement, en v1.