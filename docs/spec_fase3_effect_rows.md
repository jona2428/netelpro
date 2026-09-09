# Fase 3: Effect Rows — Typed Per-Function Effects

**Estado**: Especificación completa — lista para implementación  
**Versión**: 1.0  
**Autor**: Software Architect (Neuromancer agent)  
**Baseline**: Fase 1 verificada (typed params, enum ranges, truth-table, file-level capabilities)

---

## §0 Resumen

Esta especificación define **effect rows declarados por función** para Netelpro v0.1. Cada `defn` puede declarar opcionalmente un conjunto de efectos permitidos (`: (effects ...)`). El compilador rechaza el programa si el cuerpo invoca primitivas de I/O fuera del conjunto declarado. La regla de composición exige que el llamador declare un **superconjunto** de los efectos del llamado — de lo contrario, el chequeo estático sería trivialmente evadible envolviendo I/O en helpers.

**Alcance v1**: verbo + patrón de ruta (literal string equality). Sin condiciones dinámicas, sin álgebra de globs. Costo en tiempo de ejecución: **cero** (solo chequeo en tiempo de compilación). Sin cambios en codegen LLVM.

---

## §1 Sintaxis Exacta

### 1.1 Posición en `defn`

```
(defn nombre ((param : Tipo) ...)
  : (effects (VERB "pattern")+)
  cuerpo)
```

La cláusula `: (effects ...)` va **después de la lista de parámetros y antes del cuerpo**. El token `:` **ya existe** desde Fase 1 (usado en anotaciones de tipo en parámetros); se reutiliza — **no se añaden tokens nuevos al lexer**.

### 1.2 Gramática de la cláusula effects

```
effects_clause  ::= ":" "(" "effects" effect_row+ ")"
effect_row      ::= "(" VERB STRING ")"
VERB            ::= "read" | "write" | "delete" | "network"
```

- `VERB`: conjunto cerrado de 4 verbos. Mayúsculas/minúsculas: **solo minúsculas** (coherente con símbolos Netelpro).
- `STRING`: patrón de ruta literal (p.ej. `"./config/*"`, `"/tmp/data.log"`, `"*"`). En v1: **igualdad literal exacta** — sin expansión de globs, sin normalización de rutas.

### 1.3 Ejemplos válidos

```lisp
(defn leer-config (path : Str)
  : (effects (read "./config/*"))
  (print "leyendo config"))

(defn borrar-todo ()
  : (effects (write "*") (delete "*"))
  (print "borrando"))

(defn solo-red ()
  : (effects (network "api.example.com"))
  (print "llamada HTTP"))

(defn pura (x : Int)  ; SIN cláusula effects = conjunto vacío = pura
  (+ x 1))
```

### 1.4 Semántica de `defn` SIN cláusula effects

- **Conjunto de efectos declarado = ∅ (vacío)**.
- La función es **pura por declaración**.
- Llamar a una función con efectos desde una función pura es **error de compilación** (regla de composición: el llamador debe declarar superconjunto).
- Nada es implícito: **estrictez total**.

---

## §2 Decisiones Semánticas Numeradas (D1–D12)

| ID | Decisión | Rationale | Alternativa Rechazada |
|----|----------|-----------|----------------------|
| **D1** | Effect checking es **solo compile-time**. Costo runtime = 0. Sin representación en codegen LLVM. | Coherente con "capabilities as types" de Fase 1 (file-level boolean). El host (zone_policy) hace enforcement dinámico; Netelpro hace verificación estática. | Runtime effect tracking (tags en closures, checks en calls) — añade overhead, rompe zero-cost abstraction. |
| **D2** | **Regla de composición**: `effects(caller) ⊇ effects(callee)` para **cada** llamado directo. Chequeo por par `(verb, pattern)` con **igualdad literal de strings** en v1. | Evita evasión trivial envolviendo I/O en helpers. Igualdad literal es decidible, simple, sin falsos positivos/negativos por semántica de globs. | Subset semántico de globs (p.ej. `"./config/*" ⊆ "./*"`) — requiere motor de matching de globs, zona de ambigüedad, posponer a v2 con vocabulario de `zone_rule_generator`. |
| **D3** | Primitivas de I/O identificadas vía **tabla de capacidades existente** (`spec/caps.py` → `CAPABILITY_REQUIREMENTS`). Enumeración actual (VERIFICAR EN DISCO): `print` (requiere `io`). Futuro: `read-file`, `write-file`, `delete-file`, `http-get` etc. | Single source of truth en `arity_table.json` + `caps.py`. No duplicar conocimiento. | Hardcodear lista de primitivas effectful en parser — frágil, duplicación. |
| **D4** | `fn` (lambdas anónimas) y `truth-table` son **effectless by construction**. `truth-table` desugarea a cadena pura de `if` (Fase 1). Efectos en lambdas **prohibidos** (no hay syntax para declararlos). | Netelpro v0.1 no tiene first-class calls; lambdas solo se usan inline. Efectos en lambdas requerirían effect polymorphism — fuera de scope v1. | Permitir efectos en `fn` con inferencia — rompe regla de composición declarativa, añade complejidad de inferencia. |
| **D5** | **Error reporting**: coordenadas exactas del `Call` ofensivo + efecto faltante. Formato: `line X, col Y: call to 'print' requires effect (io "stdout") not declared in function 'foo' — add : (effects (io "stdout"))`. | Estilo prosecutorial consistente con parser y caps.py. Actionable: dice qué agregar. | Solo nombre de función sin coordenadas — inacciónable para archivos grandes. |
| **D6** | **Efectos duplicados** en la misma cláusula: **warning** (no error), se deduplican. `(effects (read "a") (read "a"))` → warning "duplicate effect row (read "a")". | Permite refactoring mecánico sin romper build. | Error duro — demasiado estricto para edición interactiva. |
| **D7** | **Verbos desconocidos** en cláusula effects: **error de compilación** con lista de verbos válidos. | Fail-fast en declaración, no en uso. | Silenciar y ignorar — oculta typos. |
| **D8** | **Patrones de ruta vacíos** (`""`): **error** — patrón debe ser string no vacío. | Patrón vacío no tiene sentido semántico; probable typo. | Permitir como "cualquier ruta" — ambiguo, mejor `*` explícito. |
| **D9** | **Mutual recursion con efectos**: cada participante debe declarar sus efectos explícitamente. El fixpoint de inferencia (ya existe en `effects.py`) propaga y verifica consistencia. | Ya hay infraestructura de fixpoint en `effects.py:infer_effects`. No declarar efectos en recursión mutua = error "effects not declared for recursive function X". | Inferir efectos en recursión mutua — rompe declaratividad, ciclos de inferencia complejos. |
| **D10** | **`grant` top-level** (Fase 1 file-level capability) **coexiste** con effect rows per-function. `grant` = "este archivo puede usar estas capabilities"; effect rows = "esta función usa exactamente estos efectos". Ambos se chequean. | Migración gradual: código legacy con `grant io` sigue compilando; nuevo código usa effect rows. | Eliminar `grant` — breaking change innecesario. |
| **D11** | **Efectos en `def` (constantes top-level)**: **prohibidos**. `def` no tiene cuerpo ejecutable con calls; solo valor inicial. | `def` es binding de valor, no función. | Permitir — sin utilidad, confunde modelo. |
| **D12** | **Integración con `netelpro_gate.py` / `zone_policy`**: **informacional only en v1**. Effect rows exportados como metadata para que el host pueda hacer policy checks más precisos en futuro. No bloquea compilación. | Separación de responsabilidades: Netelpro = verificación estática; Host = enforcement dinámico. | Gate rechace basándose en effect rows — acopla compilador a policy runtime. |

---

## §3 Estrategia de Compilación

### 3.1 Pipeline estático (solo compile-time)

```
Source → Lexer → Parser (collect_defns + parse effects clause)
         → AST (Defn con campo effects: list[(verb, pattern)])
         → Effect Registry Construction (single pass over defns)
         → Call-Graph Propagation (O(calls) fixpoint, max 100 iter)
         → Composition Check (caller ⊇ callee per call edge)
         → Error Report (aggregated, prosecutorial)
         → Codegen (SIN cambios — effects no existen en runtime)
```

### 3.2 Estructuras de datos nuevas

- **En `ast_nodes.py`**: `Defn` gana campo opcional `effects: list[EffectRow] = field(default_factory=list)`
- **`EffectRow`** (nuevo dataclass frozen): `verb: str`, `pattern: str`, `line: int`, `col: int`
- **En `parser.py` / `spec/caps.py`**: `EffectRegistry: dict[str, frozenset[EffectRow]]` — mapea nombre de función → conjunto declarado.

### 3.3 Manejo de recursión

- Recursión directa/mutua: **cada `defn` en el ciclo debe declarar sus efectos**.
- El fixpoint existente en `effects.py:infer_effects` (100 iteraciones max) propaga efectos transitivamente y verifica que lo declarado ⊇ lo inferido.
- Si un `defn` recursivo no declara effects → error: `"recursive function 'foo' must declare effects explicitly"`.

### 3.4 Impacto LLVM

**Ninguno**. Effects no tienen representación en runtime. `codegen.py` no se toca.

---

## §4 Touchpoints Exactos

| Archivo | Cambio | Estado |
|---------|--------|--------|
| `spec/caps.py` | **MAYOR**: Añadir `VERBS = {"read","write","delete","network"}`, `EFFECT_PRIMITIVES: dict[str, set[EffectRow]]` mapeando primitiva → efectos requeridos (p.ej. `read-file` → `{(read, "*")}`). Derivar de `arity_table.json` v2. | **VERIFICAR EN DISCO** — `caps.py` hoy solo tiene `io` para `print`. |
| `parser.py` | **MAYOR**: En `build_node` case `"defn"`: parsear cláusula `: (effects ...)` opcional entre params y body. Validar verbos, patrones no vacíos, duplicados (warning). Poblar `Defn.effects`. En `collect_defns`: registrar effects declarados en registry. | Confirmado: parser ya maneja `:` token. |
| `ast_nodes.py` | **MENOR**: Añadir `EffectRow` dataclass + campo `effects` en `Defn`. | Listo para editar. |
| `effects.py` | **MAYOR**: `infer_effects` ya existe. Adaptar para: (a) leer `Defn.effects` declarados, (b) chequear `declared ⊇ inferred`, (c) composition check caller⊇callee en call graph. | Infraestructura fixpoint ya existe. |
| `evaluator.py` | **NINGUNO** — effects son compile-only. | Confirmado. |
| `codegen.py` | **NINGUNO** — effects no existen en runtime. | Confirmado. |
| `spec/arity_table.json` | **MENOR**: Añadir primitivas de I/O futuras con campo `"effects": [["read", "./*"]]` (v2). En v1: solo documentar `print` como `io`. | **VERIFICAR EN DISCO** — hoy solo `capabilities: ["io"]`. |
| `netelpro/__main__.py` | **MENOR**: Integrar effect checking en pipeline CLI (después de parse, antes de eval/codegen). | Touchpoint de integración. |

---

## §5 Contratos de Frontera (B1–B8)

| ID | Contrato | Comportamiento Exacto |
|----|----------|----------------------|
| **B1** | `defn` **con** effects clause válida | Acepta. Body puede llamar primitivas cuyos efectos ⊆ declarados. |
| **B2** | `defn` **sin** effects clause | Equivale a `: (effects )` — conjunto vacío. Body **no puede** llamar primitivas effectful. |
| **B3** | Call a primitiva effectual **fuera** de effects declarados | **RECHAZA**. Error: `line X, col Y: call to 'read-file' requires effect (read "./data/*") not declared in function 'foo'`. |
| **B4** | Call a `defn` effectual desde `defn` pura (sin effects) | **RECHAZA**. Error: `line X, col Y: call to 'leer-config' brings effects (read "./config/*") but caller 'main' declares none — add : (effects (read "./config/*"))`. |
| **B5** | Call a `defn` effectual desde `defn` con effects **subconjunto** | **RECHAZA**. Error: `line X, col Y: call to 'borrar-todo' brings effects (write "*") (delete "*") but caller 'wrapper' declares only (write "*") — missing (delete "*")`. |
| **B6** | Verbo desconocido en effects clause | **RECHAZA**. Error: `line X, col Y: unknown effect verb 'execute' — valid verbs: read, write, delete, network`. |
| **B7** | Patrón vacío en effects clause | **RECHAZA**. Error: `line X, col Y: effect pattern cannot be empty string`. |
| **B8** | Efectos duplicados en misma cláusula | **WARNING** (no error). `line X, col Y: warning: duplicate effect row (read "./config/*") — deduplicated`. |

---

## §6 Integración con Gate/Host (Informacional v1)

`netelpro_gate.py` y `zone_policy.classify()` consumen hoy **solo heurística de texto** en runtime (tripwire). En v1, Netelpro exporta **metadata de effect rows** (JSON o atributo en módulo compilado) para que el host pueda:

1. **Validar coherencia**: que lo declarado en Netelpro coincida con lo que `zone_rule_generator.py` permite (vocabulario de globs verdes/amarillas/rojas).
2. **Policy-as-code futuro**: host rechace deployment si effect rows piden rutas en zona roja sin approval.

**No bloquea compilación en v1**. Es contrato de intercambio de datos para v2.

---

## §7 Plan de Implementación (8 Pasos Verificables Independientemente)

| Paso | Descripción | Verificación |
|------|-------------|--------------|
| **1** | **AST**: Añadir `EffectRow` dataclass + campo `effects` en `Defn` (`ast_nodes.py`). | `pytest tests/test_ast.py -k effect` — nodos se construyen/serializan. |
| **2** | **Parser**: Parsear `: (effects ...)` en `build_node` case `"defn"`. Validar verbos, patrones, duplicados. Poblar `Defn.effects`. | `pytest tests/test_parser.py -k effect` — parse ok/reject cases. |
| **3** | **Caps/Registry**: En `caps.py`, definir `VERBS`, `EFFECT_PRIMITIVES` (mapeo primitiva→effect rows). Función `primitive_effects(head) -> frozenset[EffectRow]`. | `pytest tests/test_caps.py -k effect` — tabla correcta. |
| **4** | **Effect Inference**: En `effects.py`, `infer_effects` lee `Defn.effects` declarados, computa inferidos, chequea `declared ⊇ inferred`. | `pytest tests/test_effects.py -k infer` — inferencia + check. |
| **5** | **Composition Check**: En `effects.py`, nueva `check_composition(program, declared_effects)` recorre call graph y verifica `caller ⊇ callee` por edge. | `pytest tests/test_effects.py -k composition` — caller/callee rules. |
| **6** | **Pipeline Integration**: En `__main__.py` / entrypoint, invocar effect checking después de parse, antes de eval/codegen. Agregar flag `--no-effect-check` para debug. | `pytest tests/test_cli.py -k effect` — end-to-end. |
| **7** | **Arity Table v2**: Extender `spec/arity_table.json` con campo `"effects"` en primitivas I/O futuras. `caps.py` lo lee. | `pytest tests/test_spec.py -k arity` — tabla parseable. |
| **8** | **Test Suite Completa**: 12 tests nombrados (ver §8). Ejecutar suite completa `pytest tests/ -x` — cero regresiones Fase 1. | CI verde. |

---

## §8 Contrato de Tests (12 Tests Nombrados)

| Test ID | Descripción | Expectativa |
|---------|-------------|-------------|
| **T1** `test_effect_declared_allows_io` | `defn` con `: (effects (io "stdout"))` llama `print` | **PASS** — compila y evalúa. |
| **T2** `test_effect_missing_rejects_io` | `defn` pura (sin effects) llama `print` | **REJECT** — error compile-time con coords exactas. |
| **T3** `test_composition_caller_superset` | `foo` declara `(read "a")`, llama `bar` que declara `(read "a") (write "b")` | **REJECT** — caller missing `(write "b")`. |
| **T4** `test_composition_exact_match_ok` | `foo` declara `(read "a") (write "b")`, llama `bar` que declara `(read "a")` | **PASS** — caller ⊇ callee. |
| **T5** `test_unknown_verb_rejected` | `: (effects (execute "*"))` | **REJECT** — unknown verb error. |
| **T6** `test_empty_pattern_rejected` | `: (effects (read ""))` | **REJECT** — empty pattern error. |
| **T7** `test_duplicate_effect_warning` | `: (effects (read "a") (read "a"))` | **PASS + WARNING** — deduplicado. |
| **T8** `test_effect_in_truth_table_forbidden` | `truth-table` con `print` en rama | **REJECT** — truth-table effectless by construction. |
| **T9** `test_lambda_no_effects_syntax` | `(fn () : (effects (io "stdout")) (print "x"))` | **REJECT** — syntax error: fn no acepta effects clause. |
| **T10** `test_recursive_mutual_effects_declared` | `defn even/odd` mutuamente recursivas, ambas declaran effects | **PASS** — fixpoint converge. |
| **T11** `test_recursive_undeclared_error` | `defn fact` recursiva sin effects clause llama `print` | **REJECT** — recursive function must declare effects. |
| **T12** `test_regression_phase1_unchanged` | Suite completa Fase 1 (typed params, enum, truth-table) | **PASS** — cero regresiones. |

---

## §9 Riesgos y Deuda Técnica

| Riesgo | Mitigación / Deuda |
|--------|-------------------|
| **Literal pattern equality** (v1) vs glob semantics real | Deuda conocida: v2 usará vocabulario de `zone_rule_generator.py` (globs verdes/amarillas/rojas) para matching semántico. Documentar como limitación v1. |
| **Sin condiciones dinámicas** ("only if user confirmed") | Fuera de scope: sigue siendo runtime/host work (tripwire actual). Effect rows = parte estáticamente decidible. |
| **Host sniffing sigue como tripwire** | No se elimina: defense in depth. Effect rows reducen superficie; host sigue validando paths reales en runtime. |
| **Primtivas I/O futuras no en arity_table.json hoy** | `caps.py:_derive_capabilities_from_table` ya lee `capabilities` y `effects` del JSON. Extender JSON cuando existan primitivas `read-file` etc. |
| **Mutual recursion + effects + fixpoint non-convergence** | Límite 100 iteraciones (ya en `effects.py`). Error claro si no converge. |

---

## §10 Lista Final D1–D12 para Ratificación de Jona

1. **D1**: Effect checking solo compile-time, cero runtime cost, sin codegen changes. ✓
2. **D2**: Composición `caller ⊇ callee` con igualdad literal de `(verb, pattern)` en v1. ✓
3. **D3**: Primitivas effectuales vía `caps.py` / `arity_table.json` (single source of truth). ✓
4. **D4**: `fn` y `truth-table` effectless by construction; efectos en lambdas prohibidos. ✓
5. **D5**: Error reporting con coords exactas del call ofensivo + efecto faltante + hint actionable. ✓
6. **D6**: Efectos duplicados = warning, se deduplican. ✓
7. **D7**: Verbo desconocido = error compile-time con lista válida. ✓
8. **D8**: Patrón vacío = error. ✓
9. **D9**: Recursión mutua requiere effects declarados en cada participante. ✓
10. **D10**: `grant` top-level coexiste con effect rows per-function (ambos se chequean). ✓
11. **D11**: `def` (constantes) no acepta effects clause. ✓
12. **D12**: Integración gate/host = metadata export only en v1, no bloquea compilación. ✓

---

**Fin de especificación**. Listo para implementación en 8 pasos (§7).