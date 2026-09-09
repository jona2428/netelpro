# Fase 4: prove/evidence — spec de implementación

**Fecha:** 2026-09-08 · **Base:** design doc §2 (2026-09-08) + experimento verified_write.py + VerificationGuard del host
**Draft:** reasoning specialist (qwen3.5) · **Revisión y correcciones:** Teo
**Estado:** esperando ratificación D1-D8 por Jona

---

## §0 Resumen

`prove` (forma especial) + `Evidence` (tipo opaco no-fabricable). El compilador garantiza que una afirmación booleana esté respaldada por evidencia que **solo puede originarse en la frontera FFI** — nadie puede fabricarla dentro del lenguaje pasando `true` a mano. Violación en runtime ⇒ `StrayHoleError` (mismo mecanismo que `(sorry ...)` alcanzado). v1: una evidencia por `prove`, tipo `Evidence` con subyacente `Bool`.

**Principio pre-aprobado (del owner del lenguaje):** `evidence` es opaco y no-fabricable — sin constructor visible en la superficie del lenguaje; solo el puente FFI/ctypes lo crea (mismo patrón que la ley de frontera de `Str` en `shutdown_rule.sl`). El lenguaje garantiza **origen no-fabricable**; "este turno" y "herramienta real" son semántica del host.

## §1 Sintaxis exacta

### 1.1 Gramática

```
Type   ::= ... | "Evidence"                       ; tipo opaco nuevo
Expr   ::= ... | "(prove" ClaimExpr "(evidence" VarName ": Evidence)" ")"
```

### 1.2 Ejemplo válido

```lisp
(defn reportar-estado (status : Bool) (ev : Evidence)
  (prove status (evidence ev : Evidence))
  status)
```

### 1.3 Decisión de binding — D-clave

**Los valores `Evidence` son parámetros de función** (`(ev : Evidence)` en la firma), inyectados por el host vía FFI. La forma `(evidence ev : Evidence)` dentro de `prove` **no crea binding: valida** que `ev` es un parámetro de tipo `Evidence` en scope. *Rechazado:* binder tipo `let` local dentro de `prove` — imposible garantizar origen FFI si el lenguaje permite crear valores localmente; un let fabricaría la evidencia que debe ser no-fabricable.

`prove` retorna el valor de la claim si pasa (componible en posición de expresión).

## §2 Decisiones semánticas D1-D8

**D1 — Disciplina de tipos (Evidence ≠ Bool).** Sin coerción. `Bool` donde se espera `Evidence` ⇒ error de compilación. `Evidence` donde se espera `Bool` (fuera de `prove`) ⇒ error de compilación. El link claim-evidencia es **estructural**: `prove` exige que su segundo argumento sea cualquier expresión de tipo `Evidence` en scope (el origen está garantizado por el tipo — no hace falta restringir a un NAME directo).

**D2 — Semántica runtime (evaluación estricta).**
1. Evaluar claim (debe ser `Bool`).
2. Evaluar la evidencia (debe ser `Evidence`, subyacente `Bool`).
3. `claim=true ∧ evidence=false` ⇒ **StrayHoleError** con coordenadas.
4. `claim=false ∧ evidence=false` ⇒ OK (**negación honesta permitida** — el hole solo castiga el mentir).
5. `claim=true ∧ evidence=true` ⇒ OK.

**D3 — Representación LLVM.** `Evidence` = `i1` (igual que Bool — verificado en disco: codegen compila Bool a `i1` nativo). La distinción es solo estática (tipo del AST + prosecution). El fallo se compila a: `and claim, not evidence` → `cond_br` → bloque `unreachable`. Sin helper externo en v1 (cero overhead).

**D4 — Restricción estática de posición.** `prove` solo dentro del cuerpo de `defn`. Global o dentro de truth-table ⇒ error de compilación (truth-table es dispatch puro — coherente con D4/Fase 3).

**D5 — Uso obligatorio.** Parámetro `Evidence` declarado y no consumido por exactamente un `prove` en el cuerpo ⇒ error de compilación (evidencia "muerta" sugiere verificación faltante — nada implícito).

**D6 — Sintaxis `(evidence NAME : Evidence)`: valida origen, no crea valor.** El segundo argumento de `prove` debe ser EXACTAMENTE esa forma nombrando un parámetro `Evidence` en scope; una expresión computada como `(not y)` ⇒ error (fabricación por estructura).

**D7 — Prohibition en truth-table.** Los cuerpos de truth-table no pueden contener `prove` (dispatch finito puro, v1).

**D8 — El bridge del host es el límite de confianza.** La garantía de no-fabricabilidad es tan fuerte como la clase `Evidence` Python del host. Documentado; mitigación: `__slots__`, constructor de un solo módulo, documentar como trust boundary.

## §3 Estrategia de compilación

**Intérprete (`evaluator.py`):** evaluar claim y evidencia; `claim ∧ ¬evidence` ⇒ `StrayHoleError("Proof violation: claim true, evidence false", line, col)`. VERIFICAR EN DISCO: acceso a coordenadas del nodo en el evaluator.

**Codegen (`codegen.py`):** bajar `prove` a IR: `and claim, not evidence` → `cond_br` → `unreachable` en el bloque de violación. Sin helper externo (rendimiento, v1).

**Puente FFI (host, `verification_guard.py` — informativo):**
```python
class Evidence:
    __slots__ = ("_val",)
    def __init__(self, val: bool) -> None:   # solo el módulo del bridge lo instancia
        self._val = val
    def __bool__(self) -> bool:
        return self._val
# Host: ev = Evidence(tool_return_bool); netelpro_fn(claim, ev)
```
Si el host olvida envolver, el type-check de la frontera (param tipado `: Evidence`) falla antes de entrar al código compilado.

## §4 Touchpoints exactos

| Archivo | Cambio | Estado |
|---|---|---|
| `lexer.py` | NINGUNO — `prove`/`evidence` son símbolos head-dispatched (como `truth-table` en Fase 1), no keywords globales | patrón verificado |
| `parser.py` | SÍ: parseo de forma especial `prove`, validación estructura, prosecution de D1/D4/D5/D6/D7 | touchpoint principal |
| `ast_nodes.py` | SÍ: nodo `Prove` + tipo `Evidence` en la gramática de tipos | nuevo |
| `netelpro/evaluator.py` | SÍ: semántica runtime D2 + StrayHoleError | VERIFICAR EN DISCO coordenadas |
| `netelpro/codegen.py` | SÍ: bajada a `unreachable` | nuevo caso |
| `spec/caps.py` | SÍ (tipo): registrar `Evidence` como tipo válido en anotaciones | menor |
| `spec/arity_table.json` | **NO** — `prove` es forma especial, no primitiva | confirmado |

## §5 Contratos de frontera B1-B6

| ID | Nivel | Contrato |
|---|---|---|
| B1 | compile | `Bool` donde se espera `Evidence` ⇒ REJECT con coords |
| B2 | compile | `Evidence` donde se espera `Bool` (claim) ⇒ REJECT con coords |
| B3 | compile | segundo arg de `prove` no es `(evidence NAME : Evidence)` ⇒ REJECT (`SyntaxError: prove form must contain evidence binder`) |
| B4 | compile | parámetro `Evidence` sin `prove` que lo consuma ⇒ REJECT |
| B5 | compile | `prove` en global o truth-table ⇒ REJECT |
| B6 | runtime | `StrayHoleError: Proof violation at line X, col Y. Claim=True, Evidence=False` — formato exacto para auditoría |

## §6 Host integration (informativo v1)

Endpoint de Neuromancer: `evidence_obj = Evidence(tool_return_bool)` → invoca la función Netelpro compilada pasando `evidence_obj` en el parámetro `: Evidence`. El guard deja de ser responsable de la verificación post-ejecución: la garantía es pre-ejecución (tipo en frontera) e intra-ejecución (trap). El patrón probado en `verified_write.py` (2026-09-08) migra del protocolo opcional del host a gramática obligatoria del lenguaje.

## §7 Plan de implementación (7 pasos)

| # | Paso | Verificación |
|---|---|---|
| 1 | AST: tipo `Evidence` + nodo `Prove` | round-trip |
| 2 | Parser: forma especial + prosecution (D1/D4/D5/D6/D7) | tests de sintaxis |
| 3 | Type discipline: Evidence ≠ Bool en params y args (call-site checking Fase 1 extendido) | tests de tipos |
| 4 | Evaluator: semántica D2 + StrayHoleError | tests runtime |
| 5 | Codegen: bajada a unreachable + paridad nativa | netelpro_eval nativo |
| 6 | Bridge FFI host (clase Evidence) — informativo, puede diferirse | integración host |
| 7 | Suite: 10 tests nuevos + regresión cero | pytest verde |

## §8 Contrato de tests (10 nombrados)

| Test | Descripción | Esperado |
|---|---|---|
| `test_prove_success` | claim=true, evidence=true | ejecuta OK |
| `test_prove_honest_negative` | claim=false, evidence=false | ejecuta OK |
| `test_prove_hole_fired` | claim=true, evidence=false | StrayHoleError |
| `test_compile_bool_for_evidence` | `true` donde va `Evidence` | REJECT compile |
| `test_compile_evidence_for_bool` | `Evidence` donde va claim Bool | REJECT compile |
| `test_compile_fabricated_evidence` | `(evidence (not y) : Evidence)` | REJECT compile |
| `test_compile_unused_evidence` | param Evidence sin prove | REJECT compile |
| `test_compile_prove_global` | prove fuera de defn | REJECT compile |
| `test_compile_prove_truth_table` | prove en truth-table | REJECT compile |
| `test_regression_fase1` | suite existente | PASS sin cambios |## §9 Riesgos y deuda

| Riesgo | Mitigación / deuda |
|---|---|
| Trust boundary: la no-fabricabilidad depende de la clase Evidence del host (un host comprometido instancia a mano) | Documentado como límite de confianza; D8; __slots__ + constructor de módulo único |
| "Este turno" no lo verifica el lenguaje | Responsabilidad del host; contrato host-lenguaje documentado |
| Composición de evidencias (múltiples por prove) | v2 — requiere resolver cómo se combinan sin perder garantía de origen |
| StrayHoleError en LLVM (unreachable) no produce traceback estructurado | v1 acepta el trap; v2 evalúa helper de runtime con mensaje |

## §10 Ratificación — D1-D8 (una línea cada una)

1. **D1**: `Evidence` tipo opaco ≠ `Bool`, sin coerción en ninguna dirección.
2. **D2**: StrayHoleError solo con claim=true + evidence=false (negación honesta OK).
3. **D3**: `Evidence` = `i1` en LLVM, fallo → `unreachable`, cero overhead vs Bool.
4. **D4**: `prove` solo dentro de `defn` (global y truth-table prohibidos).
5. **D5**: parámetro `Evidence` no usado en ningún `prove` = error de compilación.
6. **D6**: `(evidence NAME : Evidence)` valida origen; no crea valor; expresiones computadas rechazadas.
7. **D7**: truth-table no puede contener `prove` (v1).
8. **D8**: el bridge host es el límite de confianza documentado.