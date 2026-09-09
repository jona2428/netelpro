# Fase 2: Refinamientos de Int/Bool — Contratos en el Tipo

**Estado**: Borrador para ratificación  
**Versión**: 0.1  
**Fecha**: 2026-09-08  
**Autor**: Software Architect (Neuromancer agent)  
**Contexto**: Post-Fase 1 (todo verificado, no re-inventar). Netelpro = lenguaje S-expression, intérprete Python + backend LLVM (llvmlite). Tipado estricto Bool/Int (nunca confluidos, errores en call-sites con coordenadas). Fase 1 ya añadió: params tipados `(p : Bool)`, rangos enum `(Int 0 1 2)` en posición de tipo, truth-table con persecución de cobertura en tiempo de compilación, desugar a if-chain. Chequeo de literales en call-site ya existe para Bool vs Int.

---

## §0 Resumen

Esta fase introduce **tipos de refinamiento** (refinement types) para `Int`: predicados adjuntos al tipo que el compilador exige probar en cada call-site donde el argumento **no** sea un literal constante verificable estáticamente. La prueba se realiza mediante **guardias sintácticas dominantes** (`if`, `and`) que preceden a la llamada. Gracias a la inmutabilidad total del lenguaje (StrayList inmutable, sin asignación), la dominancia sintáctica es **sound** sin análisis de flujo de datos: si una guardia domina el call-site, el valor no puede cambiar después.

**Alcance v1**: Solo predicados de comparación simples contra **CONSTANTES** (`> 0`, `!= 0`, rango `0..n`). Sin predicados con llamadas a funciones (indecidible, fuera de alcance). Los refinamientos se **borran** tras el type-check: costo cero en runtime, cero cambios en codegen LLVM.

---

## §1 Sintaxis Exacta

### 1.1 Gramática de Tipos (extiende Fase 1)

```
type ::= "Int" | "Bool" | "Str" | "Nil"
       | "(" "Int" int_const+ ")"                    ; enum Fase 1
       | "(" "Ref" base_type predicate+ ")"          ; REFINAMIENTO v1
       | "(" "Ref" "Int" predicate+ ")"              ; azúcar: base_type implícito Int

base_type ::= "Int" | "Bool"                          ; v1: solo Int tiene refinamientos

predicate ::= "(" cmp_op int_const ")"                ; comparación binaria vs constante
cmp_op    ::= ">" | ">=" | "<" | "<=" | "!=" | "=="
int_const ::= INT_LITERAL                             ; literal entero (negativo permitido)
```

### 1.2 Ejemplos Parseables

```lisp
; Parámetro refinado: x debe ser > 0
(defn reciproco (x : (Ref Int (> 0))) (quot 1 x))

; Múltiples predicados = conjunción (AND): 0 <= x <= 100
(defn clamp100 (x : (Ref Int (>= 0) (<= 100))) x)

; Predicado != 0 (común para divisores)
(defn safe-div (a : Int) (b : (Ref Int (!= 0))) (quot a b))

; Enum Fase 1 sigue válido y coexiste (ver D3)
(defn pick (c : (Int 0 1 2)) c)
```

### 1.3 Dónde Aparecen (v1)

| Posición | Permitido | Nota |
|----------|-----------|------|
| Parámetros `defn` / `fn` | **SÍ** | Principal use-case |
| Tipo de retorno | **NO** | Diferido a v2 (requiere ejecución simbólica del cuerpo) |
| `let` bindings | **NO** | v1: solo params; `let` infiere del valor |
| Truth-table params | **NO** | Ver D5 |
| `def` constants | **NO** | v1: solo params de función |

### 1.4 Impacto Lexer/Parser

- **NO nuevos tokens**: toda la sintaxis reutiliza paréntesis, símbolos (`Ref`, `>`, `>=`, `<`, `<=`, `!=`, `==`), y `INT` literals — ya existentes en Fase 1.
- Tokens `COLON` (`:`) y `ARROW` (`->`) ya existen en Fase 1 para params tipados; sin cambios.
- Parser: extender `parse_type` (o equivalente en `build_node`) para reconocer forma `(Ref ...)` y validar predicados.

---

## §2 Decisiones Semánticas Numeradas (D1–D10)

### D1 — Prueba Estática: Literal vs Variable
- **Literal constante**: pasa si el predicado se cumple evaluando el literal en tiempo de compilación. Ej: `(safe-div 10 2)` ✓ porque `2 != 0`.
- **Variable / expresión no-literal**: pasa **SOLO** si existe una **guardia dominante** que prueba el predicado sintácticamente (ver D2). Sin guardia → error de compilación con coordenadas del call-site.

### D2 — Dominancia Sintáctica (Definición Exacta)
Una guardia `G` **domina** un call-site `C` para variable `v` y predicado `P` si:
1. `G` es un nodo `If` o `And` en el camino desde la raíz de la función hasta `C` (AST path).
2. `G` contiene una comparación **idéntica** a `P` sobre la **misma variable** `v` (misma binding lexical, mismo nombre).
3. Para `If`: la rama que contiene `C` es la rama **then** (condición verdadera) o **else** (condición falsa) según el operador:
   - `If (cmp v k) then else`: `then` domina para `cmp v k`; `else` domina para `not (cmp v k)`.
   - `And (cmp v k) rest`: domina para `cmp v k` en `rest` (short-circuit).
4. **Todas** las rutas de ejecución desde el entry de la función hasta `C` pasan por al menos una guardia que domina para `P` sobre `v`.
5. Como el lenguaje es **inmutable** (sin asignación, StrayList frozen), una sola verificación por variable basta; no se necesitan invariantes de bucle ni análisis de flujo.

**Rechazado**: Análisis de flujo de datos completo / interpretación abstracta — overkill para v1, rompe el modelo de costos O(size(body)).

### D3 — Coexistencia Enum (Fase 1) vs Refinamiento
**Decisión**: Mantener **ambos** con mecanismos de chequeo distintos.
- **Enum** `(Int 0 1 2)`: membresía en conjunto finito. Usado por truth-table para persecución de cobertura exhaustiva. Chequeo: valor ∈ {0,1,2}.
- **Refinamiento** `(Ref Int (>= 0) (<= 2))`: predicados sobre dominio potencialmente infinito. Chequeo: dominancia sintáctica.
- **Rationale**: Enum sirve a truth-table (dominios finitos enumerables); refinamiento sirve a contratos numéricos (rangos, ≠0). Mezclarlos confundiría la persecución de cobertura. Sintaxis distinta evita ambigüedad.

### D4 — Borrado (Erasure) Total en Compilación
- Los refinamientos **desaparecen** tras la fase de type-check/prosecution.
- Codegen LLVM ve **solo el tipo base** (`Int` → `i64`). Cero instrucciones extra, cero checks en runtime.
- El desugar de truth-table (Fase 1) ya genera if-chains; las guardias de dominancia son **código fuente escrito por el programador**, no insertadas por el compilador.
- **Rechazado**: Checks en runtime — rompe el modelo de costo cero y la semántica de "contrato estático".

### D5 — Truth-Table + Refinamientos: NO en v1
- Truth-table requiere **dominios finitos enumerables** para perseguir cobertura exhaustiva (cartesiano de params).
- Refinamientos tienen dominios **infinitos/dispersos** (ej: `> 0` = infinitos enteros). No se puede generar tabla de verdad finita.
- **Decisión**: Rechazar refinamientos en params de truth-table con error claro. Enum sigue siendo el mecanismo para truth-table.

### D6 — Operandos de Predicado: Solo Constantes Literales
- Gramática: `predicate ::= "(" cmp_op INT_LITERAL ")"`
- **Rechazado**: Variables en predicados (ej: `(> x y)`). Introduce dependencias entre variables, requiere resolución de constraints, sale del alcance v1 (decidibilidad).

### D7 — Parámetros Refinados en Funciones Recursivas
- La llamada recursiva **es un call-site más** y debe satisfacer dominancia igual que cualquier otra.
- Ejemplo válido:
  ```lisp
  (defn fact (n : (Ref Int (>= 0)))
    (if (== n 0)
        1
        (* n (fact (- n 1)))))  ; (- n 1) no prueba (>= 0) → RECHAZADO v1
  ```
  Para que pase, el programador debe escribir guardia explícita:
  ```lisp
  (defn fact (n : (Ref Int (>= 0)))
    (if (== n 0)
        1
        (if (>= n 1)                 ; guardia dominante para llamada recursiva
            (* n (fact (- n 1)))
            (sorry "unreachable"))))
  ```
- **Rationale**: Conservador pero sound. v2 puede añadir reasoning aritmético simple (n-1 ≥ 0 si n ≥ 1).

### D8 — Tipos de Retorno Refinados: Diferidos a v2
- v1 **no** verifica que el cuerpo cumpla refinamiento de retorno.
- Anotación de retorno refinada se acepta sintácticamente pero se ignora en prosecution (warning o silent ignore).
- **Rationale**: Probar que una expresión arbitraria satisface un predicado requiere ejecución simbólica / weakest-precondition — fuera de alcance v1.

### D9 — Interacción con FFI / Host Bridge
- Los refinamientos se borran en codegen → en el boundary FFI el host ve `i64` plano.
- Un host hostil puede pasar valores que violan el refinamiento.
- **Mitigación**: Documentar trust boundary. Fase 1 enum tiene comportamiento out-of-domain → default (ya implementado). Para refinamientos, el host bridge es el límite de confianza; no hay runtime check.

### D10 — Errores con Coordenadas Exactas (Formato Fase 1)
Todos los errores de refinamiento usan `ParseError` / `CodegenError` existentes con `line`, `col`, mensaje prosecutorial:
```
line 12, col 5: refinement predicate '(> 0)' not proven for argument 'x' at call to 'reciproco'; missing dominating guard
line 8, col 15: literal 0 violates refinement '(!= 0)' for parameter 'b' of 'safe-div'
line 20, col 3: refinement types not allowed in truth-table parameters (use enum (Int ...) for coverage)
```

---

## §3 Estrategia de Compilación

### 3.1 Fase de Prosecution (Post `collect_defns`, Pre Codegen)
Ubicación: misma caminata que hace truth-table prosecution (extender `walk_and_validate` o nueva pasada `prosecute_refinements` en `parser.py` o módulo dedicado `prosecutor.py`).

**Algoritmo por call-site** (single-pass O(tamaño cuerpo)):
```
para cada Call node en el cuerpo de cada defn/fn:
  para cada argumento i con tipo formal T_i:
    si T_i es RefType con predicados P_1..P_k:
      si argumento es IntLit:
        evaluar P_j(literal) ∀j; si alguna falla → ERROR
      sino (variable / expresión compleja):
        recolectar guardias dominantes en path AST desde fn-root hasta Call
        para cada predicado P_j:
          si NO existe guardia dominante que pruebe P_j sobre esa variable → ERROR
```

**Recolección de guardias dominantes** (DFS desde fn-root):
- Mantener stack de guardias activas por variable: `Map<var_name, Set<Predicate>>`
- Al entrar `If (cmp v k) then else`:
  - `then`: añadir `cmp v k` al set de `v`
  - `else`: añadir `not (cmp v k)` al set de `v`
- Al entrar `And (cmp v k) rest`: añadir `cmp v k` al set de `v` para `rest`
- Al salir nodo: pop guardias introducidas en ese nodo
- En Call: consultar set actual para cada variable de argumento refinado

### 3.2 Impacto LLVM: **Ninguno**
- `codegen.py` consume AST ya validado. Tipos base (`Int` → `i64`) sin cambios.
- `RefType` nodes se ignoran en `_to_llvm_type` (o se eliminan en build_node).

### 3.3 Archivos Touchpoints (Marcados "VERIFICAR EN DISCO")

| Archivo | Cambio Esperado | Notas |
|---------|-----------------|-------|
| `lexer.py` | **NINGUNO** | No nuevos tokens |
| `parser.py` | **SÍ** | Extender type grammar + prosecution walk |
| `ast_nodes.py` | **SÍ** | Añadir `RefType` node + `Predicate` node |
| `evaluator.py` | **NINGUNO** | Erasure total |
| `codegen.py` | **NINGUNO** | Erasure total |
| `spec/arity_table.json` | **NINGUNO** | Sintaxis en tipo, no en head |
| `spec/caps.py` | **VERIFICAR EN DISCO** | Si existe capability gating |

---

## §4 Contratos de Frontera (B1–B12)

| ID | Entrada | Esperado | Formato Error |
|----|---------|----------|---------------|
| B1 | `(defn f (x : (Ref Int (> 0))) x)` + `(f 5)` | **ACCEPT** | — |
| B2 | `(defn f (x : (Ref Int (> 0))) x)` + `(f 0)` | **REJECT** | `literal 0 violates refinement '(> 0)'` |
| B3 | `(defn f (x : (Ref Int (> 0))) x)` + `(f y)` sin guardia | **REJECT** | `refinement '(> 0)' not proven for argument 'y'` |
| B4 | `(defn f (x : (Ref Int (> 0))) (if (> x 0) (f x) 0))` | **ACCEPT** | guardia `If` domina en then-branch |
| B5 | `(defn f (x : (Ref Int (> 0))) (if (> y 0) (f x) 0))` | **REJECT** | guardia en variable distinta `y` |
| B6 | `(defn f (x : (Ref Int (> 0))) (if (> x 0) 0 (f x)))` | **REJECT** | call en else-branch (guarda `≤ 0`) |
| B7 | `(defn f (x : (Ref Int (> 0))) (and (> x 0) (f x)))` | **ACCEPT** | `And` short-circuit domina |
| B8 | `(defn f (x : (Ref Int (> 0))) (if (> x 0) (if (< x 10) (f x) 0) 0))` | **ACCEPT** | guardia anidada conjunta |
| B9 | `(defn f (x : (Ref Int (> 0))) (if (> x 0) (f x) (f x)))` | **REJECT** | else-branch sin guardia para `x` |
| B10 | `(defn f (x : (Ref Int (> y))) x)` | **REJECT** | `predicate operand must be constant literal, found variable 'y'` |
| B11 | `(truth-table (x : (Ref Int (> 0))) ...)` | **REJECT** | `refinement types not allowed in truth-table parameters` |
| B12 | `(defn f (x : (Ref Int (>= 0))) (f (- x 1)))` | **REJECT** v1 | `recursive call lacks dominating guard for '(>= 0)'` |

---

## §5 Plan de Implementación (7 Pasos Independientes)

| Paso | Descripción | Verificación |
|------|-------------|--------------|
| 1 | **AST nodes**: Añadir `RefType(base_type, predicates[])` y `Predicate(op, const)` en `ast_nodes.py`. Test: round-trip parse → AST → repr. | `test_ast_ref_type.py` |
| 2 | **Parser type grammar**: Extender `build_node` / `parse_type` para reconocer `(Ref ...)` y validar predicados (solo ops permitidos, solo INT const). Rechazar en truth-table params. | `test_parser_ref_syntax.py` |
| 3 | **Prosecutor: recolección guardias**: Implementar DFS con stack de guardias por variable en nueva función `collect_dominating_guards(fn_body) -> Map<var, Set<Predicate>>`. | `test_prosecutor_guard_collection.py` |
| 4 | **Prosecutor: chequeo call-sites**: Integrar en `walk_and_validate` o pasada separada: para cada Call con args refinados, aplicar reglas B1–B12. | `test_prosecutor_callsite.py` |
| 5 | **Literal evaluation**: Evaluar predicados sobre `IntLit` en tiempo de compilación (reutilizar lógica de comparaciones primitivas). | `test_prosecutor_literals.py` |
| 6 | **Erasure en codegen**: Asegurar `_to_llvm_type` ignora `RefType` y usa `base_type`. Test: compilar función con param refinado → LLVM IR usa `i64`. | `test_codegen_ref_erasure.py` |
| 7 | **Integración + regresión**: Suite completa Fase 1 + nuevos tests Fase 2. Zero regressions. | `pytest tests/` |

---

## §6 Contrato de Tests (12 Tests Nombrados)

| Test ID | Descripción | Expectativa |
|---------|-------------|-------------|
| `test_ref_literal_satisfies` | `(defn f (x:(Ref Int (>0))) x)` + `(f 5)` | PASS |
| `test_ref_literal_violates` | `(defn f (x:(Ref Int (>0))) x)` + `(f 0)` | FAIL: literal violation |
| `test_ref_guarded_variable_if_then` | Guardia `If (> x 0)` en then-branch | PASS |
| `test_ref_guarded_variable_and` | Guardia `And (> x 0) (f x)` | PASS |
| `test_ref_unguarded_variable` | Variable sin guardia | FAIL: missing guard |
| `test_ref_guard_wrong_variable` | Guardia en `y`, call con `x` | FAIL: wrong variable |
| `test_ref_partial_guard_some_paths` | If con call en ambas ramas, solo then guardado | FAIL: else path unguarded |
| `test_ref_nested_guards` | `If (> x 0) (If (< x 10) (f x) 0) 0` | PASS |
| `test_ref_predicate_variable_operand` | `(Ref Int (> y))` | FAIL: operand must be constant |
| `test_ref_in_truth_table` | Truth-table con param refinado | FAIL: not allowed |
| `test_ref_recursive_call_guarded` | Recursión con guardia explícita | PASS |
| `test_ref_recursive_call_unguarded` | Recursión sin guardia | FAIL |
| `test_fase1_regression_enum` | Enum `(Int 0 1 2)` + truth-table sigue funcionando | PASS |
| `test_fase1_regression_typed_params` | Params `(x : Bool)` + call-site Bool/Int checking | PASS |

---

## §7 Riesgos y Deuda Técnica

| Riesgo | Descripción | Mitigación / Deuda v2 |
|--------|-------------|----------------------|
| **Falsos negativos por dominancia sintáctica** | Casos semánticamente probables pero sintácticamente no dominantes (ej: `if (> x 0) then (if (> x 0) (f x) 0) else 0` — la guardia interna es redundante pero requerida). | Documentar como limitación v1. v2: análisis de flujo de datos / abstract interpretation para eliminar redundancias. |
| **Host bridge trust boundary** | Refinamientos borrados → host puede pasar valores inválidos vía FFI. | Documentar: "Refinamientos son contratos estáticos internos; boundary FFI es zona de confianza. Para validación en boundary, usar enum Fase 1 con default behavior." |
| **Composición Refinamiento + Enum** | `(Ref (Int 0 1 2) (> 0))` — ¿sentido? | Diferido v2. v1: mutuamente excluyentes en sintaxis (uno o el otro). |
| **Predicados aritméticos compuestos** | `(> (+ x y) 0)` — fuera de alcance v1. | v2: permitir expresiones lineales simples (Presburger). |
| **Retorno refinado** | Verificar que body cumple postcondición. | v2: weakest-precondition / symbolic execution acotada. |

---

## §8 Lista Final D1–D10 para Ratificación de Jona

| ID | Decisión (una línea) |
|----|----------------------|
| D1 | Literal constante: evalúa predicado en compile-time; variable: requiere guardia dominante sintáctica. |
| D2 | Dominancia = toda ruta AST desde fn-root al call-site pasa por If/And que prueba predicado idéntico sobre misma variable; inmutabilidad hace sound sin dataflow. |
| D3 | Enum y Refinamiento coexisten separados: enum para truth-table (dominio finito), refinamiento para contratos numéricos (dominio infinito). |
| D4 | Erasure total: refinamientos desaparecen tras prosecution; codegen ve solo tipo base; cero costo runtime. |
| D5 | Refinamientos prohibidos en truth-table params (dominio no enumerable); error claro. |
| D6 | Predicados solo contra literales Int constantes; variables en predicado rechazadas v1. |
| D7 | Llamada recursiva tratada como call-site normal: requiere guardia dominante explícita. |
| D8 | Tipos de retorno refinados aceptados sintácticamente pero no verificados v1 (diferido a v2). |
| D9 | FFI boundary: host ve i64 plano; trust boundary documentado; mitigación = enum con default para validación en boundary. |
| D10 | Errores con coordenadas exactas formato Fase 1: `line N, col M: refinement predicate '...' not proven for argument '...' at call to '...'`. |

---

**FIN DEL SPEC** — Listo para ratificación y implementación por pasos.