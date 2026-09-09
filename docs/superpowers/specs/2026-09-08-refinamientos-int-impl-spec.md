# Fase 2: Refinamientos de Int/Bool — spec de implementación

**Fecha:** 2026-09-08 · **Base:** design doc §4 (2026-09-08) + Fase 1 implementada (a2513e9)
**Draft:** arquitecto (nemotron-3-ultra) · **Revisión y correcciones:** Teo
**Estado:** esperando ratificación D1-D10 por Jona

---

## §0 Resumen

Tipos de refinamiento para `Int`: predicados de comparación contra constantes, adjuntos al tipo. El compilador exige prueba en cada call-site donde el argumento no sea un literal verificable — la prueba es una **guardia dominante** (`if`/`and`) que precede a la llamada. La inmutabilidad total del lenguaje (sin asignación) hace que la dominancia sintáctica sea **sound sin análisis de flujo**: si la guardia domina, el valor no puede cambiar después. Erasure total tras type-check: costo runtime cero, cero cambios en codegen LLVM.

## §1 Sintaxis exacta

### 1.1 Gramática (extiende Fase 1)

```
type      ::= "Int" | "Bool" | "Str" | "Nil"
            | "(" "Int" int_const+ ")"                ; enum Fase 1 (sin cambios)
            | "(" "Ref" "Int" predicate+ ")"          ; refinamiento v1
predicate ::= "(" cmp_op int_const ")"
cmp_op    ::= ">" | ">=" | "<" | "<=" | "!=" | "=="
```

### 1.2 Ejemplos (todos parseables)

```lisp
(defn reciproco (x : (Ref Int (> 0))) x)                 ; un predicado
(defn clamp100 (x : (Ref Int (>= 0) (<= 100))) x)        ; conjunción de predicados
(defn safe-div (a : Int) (b : (Ref Int (!= 0))) a)       ; != 0 para divisores
(defn pick (c : (Int 0 1 2)) c)                          ; enum Fase 1 coexiste (D3)
```

### 1.3 Dónde aparecen (v1)

| Posición | Permitido | Razón |
|---|---|---|
| Params de `defn`/`fn` | SÍ | caso de uso principal |
| Tipo de retorno | NO (D8) | probar el cuerpo requiere ejecución simbólica |
| `let` bindings | NO | v1 solo params |
| Params de truth-table | NO (D5) | cobertura requiere dominio enumerable |
| `def` constantes | NO | v1 solo params de función |

### 1.4 Impacto lexer/parser

**Cero tokens nuevos**: reutiliza paréntesis, símbolos de comparación (ya lexean como SYMBOL u operador — VERIFICAR EN DISCO cómo lexean hoy `>` `>=` dentro de s-expr) y literales INT. `:` y `->` existen desde Fase 1. El parser extiende la resolución de tipos (`build_node` en posición de anotación) para reconocer `(Ref ...)` y validar predicados.

## §2 Decisiones semánticas D1-D10

**D1 — Prueba estática: literal vs variable.** Literal constante: se evalúa el predicado en compilación (`(safe-div 10 2)` ✓ porque `2 != 0`). Variable o expresión no-literal: pasa SOLO con guardia dominante (D2); sin ella → error de compilación con coordenadas del call-site. *Rechazado:* confianza en el programador (violación silenciosa = el bug que la fase mata).

**D2 — Dominancia sintáctica, definición exacta.** Una guardia `G` domina un call-site `C` para variable `v` y predicado `P` si:
1. `G` es un nodo `If` o `And` en el camino AST desde la raíz de la función hasta `C`.
2. `G` contiene una comparación **idéntica** a `P` sobre la **misma variable** `v` (misma ligadura léxica, mismo nombre).
3. Direccionalidad: en `If (cmp v k) then else` — la rama `then` hereda `cmp v k`; la rama `else` hereda la negación. En `And (cmp v k) rest` — `rest` hereda `cmp v k` (short-circuit).
4. TODAS las rutas de ejecución hasta `C` pasan por una guardia que prueba `P` sobre `v`.
5. Inmutabilidad: una sola verificación por variable basta — no hay invariantes de bucle ni dataflow. *Rechazado:* análisis de flujo completo (overkill v1, rompe el modelo O(size(body))).

**D3 — Enum y refinamiento coexisten, separados.** Enum `(Int 0 1 2)`: membresía en conjunto finito, alimenta la persecución de cobertura de truth-table. Refinamiento: predicados sobre dominio infinito, alimenta dominancia sintáctica. Sintaxis distinta, mecanismo distinto — mezclarlos confundiría la fiscalía de cobertura. *Rechazado:* unificar enum como azúcar de refinamiento (perdería la fiscalía D1 de Fase 1).

**D4 — Erasure total.** Los refinamientos desaparecen tras prosecution; codegen LLVM ve solo `Int` → `i64`. Cero instrucciones nuevas. Las guardias son código del programador, no insertadas por el compilador. *Rechazado:* checks en runtime (rompe el modelo de costo cero).

**D5 — Truth-table sin refinamientos.** La fiscalía de cobertura necesita dominios finitos enumerables; `(Ref Int (> 0))` es infinito. Error claro si se intenta. Enum sigue siendo el mecanismo de truth-table.

**D6 — Operandos: solo constantes literales.** `(Ref Int (> y))` con variable → error. Predicados entre variables requieren resolución de constraints (v2+).

**D7 — Recursión: el self-call es un call-site más.** Debe satisfacer dominancia como cualquier llamada:
```lisp
; RECHAZADO v1: (- n 1) no prueba (>= 0)
(defn fact (n : (Ref Int (>= 0)))
  (if (== n 0) 1 (* n (fact (- n 1)))))
; VÁLIDO: guardia explícita para el self-call
(defn fact (n : (Ref Int (>= 0)))
  (if (== n 0)
      1
      (if (>= n 1) (* n (fact (- n 1))) (sorry "unreachable"))))
```
v2 podrá razonar aritmética simple (`n >= 1 => n-1 >= 0`). *Rechazado v1:* razonamiento aritmético (frágil, abierto).

**D8 — Retorno refinado: diferido a v2.** Se acepta sintácticamente, no se verifica (probar el cuerpo = weakest precondition, fuera de alcance).

**D9 — Frontera FFI: trust boundary.** Erasure ⇒ el host ve `i64` plano; un host puede pasar valores violatorios. Documentado como límite de confianza. Para validación en frontera existe el enum de Fase 1 (out-of-domain → default).

**D10 — Errores con coordenadas, formato Fase 1.** `line N, col M: refinement predicate '(> 0)' not proven for argument 'x' at call to 'reciproco'` / `line N, col M: literal 0 violates refinement '(!= 0)' for parameter 'b' of 'safe-div'` / `line N, col M: refinement types not allowed in truth-table parameters (use enum)`.

## §3 Estrategia de compilación

Todo estático, erasure tras type-check. La prosecution vive en la **misma caminata que la fiscalía de truth-table** (`walk_and_validate` en parser.py, extendida) — una sola pasada post-`collect_defns`.

Algoritmo por call-site (single-pass O(size(body))):
```
para cada Call en el cuerpo de cada defn/fn:
  para cada argumento i con tipo formal T_i:
    si T_i es RefType con predicados P_1..P_k:
      si argumento es IntLit: evaluar P_j(literal) ∀j; si alguna falla → ERROR (D10)
      sino: recolectar guardias dominantes en el path AST raíz→call
            para cada P_j: si no hay guardia que la pruebe sobre esa variable → ERROR
```

Recolección de guardias (DFS desde la raíz de la fn):
- Mapa por variable: `name -> Set<Predicate>` (stack de guardias activas).
- Entrar a `If (cmp v k)`: push `cmp v k` para rama then, push negación para else. Entrar a `And (cmp v k) rest`: push `cmp v k` para rest. Salir del nodo: pop.
- En `Call`: consultar el mapa para cada argumento con tipo refinado.

**Impacto LLVM: ninguno.** `codegen.py` consume AST ya validado; `RefType` se desvanece antes.

## §4 Touchpoints exactos

| Archivo | Cambio | Estado |
|---|---|---|
| `lexer.py` | NINGUNO | sin tokens nuevos |
| `parser.py` | SÍ: type grammar (`build_node` en anotaciones) + prosecution en `walk_and_validate` | verificado (Fase 1 vive ahí) |
| `ast_nodes.py` | SÍ: `RefType(base, predicates)` + `Predicate(op, const)` | nuevo |
| `evaluator.py` | NINGUNO | erasure total |
| `codegen.py` | NINGUNO | erasure total |
| `spec/arity_table.json` | NINGUNO | sintaxis de tipo, no de head |
| `netelpro/caps.py` | NINGUNO en v1 | **verificado en disco**: caps.py maneja capabilities de primitivas ({"print": {"io"}}), ortogonal a refinamientos de tipo |

## §5 Contratos de frontera B1-B12

| ID | Entrada | Esperado |
|---|---|---|
| B1 | `(defn f (x : (Ref Int (> 0))) x)` + `(f 5)` | ACCEPT |
| B2 | mismo + `(f 0)` | REJECT: `literal 0 violates refinement '(> 0)'` |
| B3 | mismo + `(f y)` sin guardia (y : Int en scope) | REJECT: `refinement '(> 0)' not proven for 'y'` |
| B4 | cuerpo `(if (> x 0) (f x) 0)` | ACCEPT: guardia domina en then |
| B5 | params `(x : (Ref Int (> 0))) (y : Int)`, cuerpo `(if (> y 0) (f x) 0)` | REJECT: guardia en variable distinta |
| B6 | cuerpo `(if (> x 0) 0 (f x))` | REJECT: call en else (la guardia prueba `<= 0`) |
| B7 | cuerpo `(and (> x 0) (f x))` | ACCEPT: And short-circuit domina |
| B8 | cuerpo `(if (> x 0) (if (< x 10) (f x) 0) 0)` | ACCEPT: guardias anidadas conjuntas |
| B9 | cuerpo `(if (> x 0) (f x) (f x))` | REJECT: rama else sin guardia para x |
| B10 | tipo `(Ref Int (> y))` | REJECT: `predicate operand must be constant literal` |
| B11 | truth-table con param `(Ref Int ...)` | REJECT (D5) |
| B12 | recursión sin guardia para el self-call | REJECT (D7) |## §6 Plan de implementación (7 pasos)

| # | Paso | Verificación |
|---|---|---|
| 1 | AST: `RefType` + `Predicate` en `ast_nodes.py` | round-trip parse → AST |
| 2 | Parser: type grammar `(Ref ...)`, validar ops/constantes, rechazar en truth-table | tests de sintaxis |
| 3 | Prosecutor: recolección de guardias (DFS con stack por variable) | unit tests de recolección |
| 4 | Prosecutor: chequeo de call-sites (integrado en `walk_and_validate`) | tests B1-B12 |
| 5 | Evaluación de literales en compilación (reusa comparaciones primitivas) | tests de literales |
| 6 | Erasure en codegen: `RefType` → tipo base (`i64`) | IR generado sin trazas |
| 7 | Regresión: suite Fase 1 completa + nuevos tests | pytest verde |

## §7 Contrato de tests (14 nombrados)

| Test | Descripción | Esperado |
|---|---|---|
| `test_ref_literal_satisfies` | `(f 5)` con `(Ref Int (> 0))` | PASS |
| `test_ref_literal_violates` | `(f 0)` con `(Ref Int (> 0))` | FAIL: literal violation |
| `test_ref_guarded_variable_if_then` | guardia en then | PASS |
| `test_ref_guarded_variable_and` | guardia por And short-circuit | PASS |
| `test_ref_unguarded_variable` | variable sin guardia | FAIL: missing guard |
| `test_ref_guard_wrong_variable` | guardia en `y`, call con `x` | FAIL: wrong variable |
| `test_ref_partial_guard_some_paths` | solo una rama guardada | FAIL: else path unguarded |
| `test_ref_nested_guards` | guardias anidadas conjuntas | PASS |
| `test_ref_predicate_variable_operand` | `(Ref Int (> y))` | FAIL: operand must be constant |
| `test_ref_in_truth_table` | refinamiento en truth-table | FAIL: not allowed |
| `test_ref_recursive_call_guarded` | recursión con guardia explícita | PASS |
| `test_ref_recursive_call_unguarded` | recursión sin guardia | FAIL |
| `test_fase1_regression_enum` | enum + truth-table intactos | PASS |
| `test_fase1_regression_typed_params` | params tipados + call-site checking intactos | PASS |

## §8 Riesgos y deuda

| Riesgo | Mitigación / deuda v2 |
|---|---|
| Falsos negativos por dominancia sintáctica (casos semánticamente probables pero sintácticamente no dominantes) | Documentado; v2: análisis de flujo acotado para eliminar redundancia |
| Host bridge puede violar refinamientos vía FFI | Trust boundary documentado (D9); validación en frontera = enum Fase 1 con default |
| Composición refinamiento + enum | Mutuamente excluyentes en sintaxis v1; v2 evalúa composición |
| Predicados aritméticos compuestos `(> (+ x y) 0)` | Fuera de alcance v1; v2: expresiones lineales (Presburger acotado) |
| Retorno refinado | Diferido v2 (weakest precondition) |

## §9 Ratificación — D1-D10 (una línea cada una)

1. **D1**: literal ⇒ evalúa en compilación; variable ⇒ requiere guardia dominante.
2. **D2**: dominancia = toda ruta AST raíz→call pasa por If/And que prueba el predicado idéntico sobre la misma variable; sound por inmutabilidad.
3. **D3**: enum y refinamiento coexisten separados (enum = truth-table; ref = contratos numéricos).
4. **D4**: erasure total tras prosecution; codegen ve tipo base; costo runtime cero.
5. **D5**: refinamientos prohibidos en truth-table params.
6. **D6**: predicados solo contra constantes literales.
7. **D7**: self-call recursivo tratado como call-site normal (requiere guardia).
8. **D8**: retorno refinado aceptado sintácticamente, no verificado (v2).
9. **D9**: frontera FFI = trust boundary documentada.
10. **D10**: errores con coordenadas exactas, formato Fase 1.