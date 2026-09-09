# Spec: cuatro primitivos de lenguaje para Netelpro

**Fecha:** 2026-09-08
**Autor de la spec:** Claude (Sonnet 5), en conversación con Jona
**Para implementar:** Teo (glm-5.3-flash)
**Contexto:** no son aplicaciones nuevas sobre Netelpro — son primitivos que
faltan en el lenguaje mismo. La línea que conecta los cuatro: mover reglas que
hoy viven en convención humana (un comentario, un protocolo que un host tiene
que acordarse de aplicar) hacia algo que el compilador obliga.

Evidencia real detrás de cada uno, no invención abstracta — ver la sesión del
2026-09-07/08 en `CLAUDE.md` de Neuromancer, secciones 45-51 y el experimento
`verified_write.py` (scratchpad, mismo día).

---

## 1. `truth-table` — tabla de verdad como forma del lenguaje, no comentario

### Motivación (bug real que esto habría prevenido)

`netelpro_gate.sl` tiene DOS implementaciones de la misma regla: el binario
compilado, y un fallback en Python dentro de `netelpro_gate.py` (`apply()`)
que corre cuando el compilador no está disponible. El 2026-09-07 se encontró
que el fallback aliasaba `audit_paths` con la decisión de entrada — un bug
real, sólo detectado porque `test_netelpro_gate_fallback_parity.py` comparó
ambos caminos exhaustivamente. Hasta ese momento, la tabla de verdad vivía
como comentario en el header del `.sl` y como código Python separado — dos
fuentes de verdad que podían divergir sin que nadie se enterara.

### Diseño propuesto

```
(truth-table filter-rule (zone-rank critical auto-confirm)
  ((1 0 1) -> 1)   ; YELLOW, no-critical, auto-confirm -> aprobado
  ((_ _ 0) -> 0)   ; sin auto-confirm -> nunca
  ((2 _ _) -> 0)   ; RED -> nunca
  (_ -> 0))        ; default explícito, obligatorio (ver "Requisitos")
```

Compila a una función pura `filter-rule` idéntica en semántica a un `defn`
con `if`/`and` anidados — pero la tabla ES el código, no una paráfrasis en
comentario de otro código.

### Requisitos de la fiscalía del parser

- **Exhaustividad obligatoria en tipos finitos.** Si un parámetro es `Bool`,
  el compilador exige que las filas cubran las 2 combinaciones (o un `_`
  comodín) antes de aceptar la tabla — error de compilación, no runtime, si
  falta un caso.
- **Sin fila default silenciosa.** El `(_ -> 0)` final debe estar presente
  explícito; el compilador rechaza una tabla sin default (mismo espíritu que
  `(sorry "razón")` — nada implícito).
- **Diferencial gratis:** dado que `truth-table` es sintaxis, el intérprete y
  el backend LLVM DEBEN producir el mismo resultado para cada fila por
  construcción — el test diferencial que hoy se escribe a mano
  (`test_netelpro_gate_fallback_parity.py`) pasa a ser generado por el propio
  compilador a partir de las filas declaradas.

### Alcance de la v1

Solo parámetros `Bool` e `Int` de rango declarado explícitamente (ej.
`zone-rank : Int{0,1,2}`) — sin eso, "exhaustivo" no se puede verificar en
compilación. `Str` queda fuera de v1 (dominio infinito, no exhaustivo).

---

## 2. `prove` / `evidence` — la afirmación exige evidencia, en la gramática

### Motivación (patrón ya probado hoy, fuera del lenguaje)

`verification_rule.sl` + `verification_guard.py` implementan exactamente
"una afirmación (`claimed`) necesita evidencia real (`verified`)" — pero como
protocolo que un HOST (Neuromancer) tiene que acordarse de invocar en cada
punto donde un agente reporta un resultado. Si un desarrollador olvida
llamarlo en un nuevo endpoint, la protección desaparece silenciosamente. El
experimento `verified_write.py` (2026-09-08) demostró el patrón funcionando
end-to-end, reusando el guardián existente sin código Netelpro nuevo — la
prueba de que el CONCEPTO es sólido. Lo que falta es que deje de ser opcional.

### Diseño propuesto

```
(defn reportar-resultado (claimed-success : Bool)
  (prove claimed-success
    (evidence tool-return-value : Bool)))
```

`prove` es una forma especial, no una función: el compilador exige que el
segundo argumento sea literalmente una `evidence` — un valor que en runtime
viene marcado como "retornado por una herramienta real este turno", no un
valor arbitrario que el programa podría fabricar. Si `claimed-success` es
`true` y la `evidence` asociada es `false` (o no hay evidencia), el programa
NO COMPILA-EJECUTA esa rama — es un `StrayHoleError` en tiempo de ejecución,
igual que un `(sorry ...)` alcanzado, no una advertencia.

### Requisitos

- `evidence` es un tipo nuevo, no un `Bool` disfrazado — el compilador debe
  poder distinguir "un booleano que vino de una herramienta verificada" de
  "un booleano que el programa calculó". Sin esa distinción, `prove` es
  decorativo (cualquiera podría fabricar una evidencia falsa pasándole
  `true` a mano).
- El puente FFI/ctypes (mismo patrón que `netelpro_gate.py`/
  `verification_guard.py`) debe marcar explícitamente qué valores cruzan la
  frontera como `evidence` — analogía directa con la "ley de frontera" que
  ya rige `Str` en `shutdown_rule.sl`.

### Alcance de la v1

Un solo `evidence` por `prove`, tipo `Bool`. Composición de múltiples
evidencias (`prove claim (and evidence1 evidence2)`) queda para v2 —
necesita resolver primero cómo se combinan sin perder la garantía de origen.

---

## 3. Effect rows — capacidades tipadas, no solo booleanas

### Motivación (mover el gate de runtime a compile-time)

`zone_policy.classify()` decide en cada llamada, en runtime, si una skill
puede tocar un path (verde/amarillo/rojo). Es necesario porque hoy Neuromancer
no tiene forma de saber ANTES de ejecutar qué va a tocar una función — solo
lo audita con heurísticas de texto (§7 de esta spec, "sniff" documentado como
tripwire no muro en `CLAUDE.md`). Netelpro ya tiene "capabilities as types"
(un booleano: tiene IO o no). El salto es declarar QUÉ IO, no solo SI hay IO.

### Diseño propuesto

```
(defn leer-config (path : Str)
  : (effects (read "./config/*"))
  ...)

(defn borrar-todo ()
  : (effects (write "*") (delete "*"))
  ...)
```

La firma de la función declara el conjunto de efectos permitidos (patrón de
paths + verbo: read/write/delete/network). El compilador rechaza el programa
si el cuerpo invoca una primitiva de IO fuera del conjunto declarado — igual
que la fiscalía del parser hoy rechaza una llamada a una forma no declarada.

### Requisitos

- Reusa el vocabulario de patrones que YA vive en `zone_rule_generator.py`
  (roots verdes/amarillas/rojas) — no inventar una sintaxis de globbing
  nueva, portar la que Neuromancer ya usa y ya probó.
- Composición: una función que llama a otra hereda (o debe declarar
  explícitamente un subconjunto de) los efectos de la llamada — sin esto, el
  chequeo estático es trivialmente evadible envolviendo la IO en una función
  auxiliar.

### Alcance de la v1

Solo el verbo y el patrón de path — sin condiciones dinámicas (ej. "solo si
el usuario confirmó"). Esa parte sigue siendo trabajo del host/gate en
runtime, como hoy; effect rows cubre la parte ESTÁTICAMENTE decidible.

---

## 4. Refinamiento de `Int`/`Bool` — contratos en el tipo

### Motivación

Extensión directa de la disciplina estricta que el lenguaje ya tiene (sin
casteo implícito). Reduce cuánto tiene que cubrir el test diferencial a mano,
porque una franja entera de invariantes (positividad, rango) deja de ser
responsabilidad del test y pasa a ser responsabilidad del tipo.

### Diseño propuesto

```
(defn dividir (a : Int) (b : Int{b != 0})
  (/ a b))
```

`Int{predicado}` es un tipo refinado: el compilador exige, en cada punto de
llamada donde el argumento no sea una constante literal verificable
estáticamente, una prueba de que el predicado se cumple — vía `if`/`and`
guardas antes de la llamada, o rechaza en compilación si el valor viene de
una fuente no acotada.

### Alcance de la v1

Solo predicados de comparación simple contra constantes (`> 0`, `!= 0`,
rango `{0,1,2}`) — nada de predicados arbitrarios con llamadas a función
(eso abre la puerta a undecidability real, fuera de alcance).

---

## Orden de implementación sugerido

1. **`truth-table`** primero — el más chico, el que ya tiene el bug real
   documentado que lo justifica, y no depende de los otros tres.
2. **Refinamiento `Int`/`Bool`** — extensión natural del sistema de tipos
   existente, base útil para el punto 3.
3. **Effect rows** — el de mayor impacto arquitectónico, pero depende de que
   el sistema de tipos ya soporte anotaciones más ricas (se beneficia de 2
   estando hecho primero).
4. **`prove`/`evidence`** al final — es el que más deuda de diseño abierta
   tiene (el tipo `evidence` como concepto de frontera nunca se probó en el
   lenguaje, solo en el host vía FFI) y el que menos urge: el patrón YA
   funciona hoy vía `VerificationGuard`, solo vive un nivel más afuera de lo
   ideal.

## Lo que esta spec NO resuelve, a propósito

No incluye sintaxis final revisada por el propio equipo del lenguaje, no
incluye implementación de referencia, no incluye el impacto en el backend
LLVM de cada primitivo nuevo. Es el diseño y la motivación — la
especificación de implementación detallada (spec §7 style, con contratos de
frontera exactos) es trabajo de la siguiente fase, con Teo.
