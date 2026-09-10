# SPEC — El Juez de Memoria (Netelpro como política de ingesta)

Estado: **propuesto** (fase spec, previo a implementación)
Versión del lenguaje objetivo: netelpro ≥ v0.7 (strings read-only, Bool params, effects tracking)
Fecha: 2026-09-10

---

## 1. Problema

El índice de memoria semántica de la casa (RAG) hoy decide qué indexar por criterio
difuso del agente y por defecto del código de ingesta. Evidencia verificada
(2026-09-10):

1. Un query sin contenido semántico ("Alo?") devolvió 5 vecinos flojos — el RAG
   no sabe decir "no encontré nada".
2. Data con PII de terceros (dataset FAE-CAP: RUTs, teléfonos, correos de
   postulantes reales) quedó indexada en varias pasadas, duplicada.
3. Fragmentos duplicados conviven en el índice (mismo bloque indexado 3 veces).
4. No existe herramienta de purga ni metadatos de vigencia: la memoria solo crece.

La decisión de qué entra, qué se duplica, qué se purga y qué se expone vive
dispersa entre juicio no-auditable y código ad-hoc.

## 2. Tesis

La política de memoria es **el tercer dominio** del mismo shape ya probado en
producción por la casa:

| Dominio | Regla | Estado |
|---|---|---|
| Seguridad de zonas | `zone_policy` (paths rojos/amarillos/verdes) | **en producción** (gate Phase I) |
| Honestidad del agente | `step_admit` / `step_close` / fiscal | **en producción** (HonestyGuard + capa alética) |
| **Memoria** | `memory_policy` (este spec) | **propuesto** |

Un juez escrito en Netelpro es una función pura compilada a código de máquina:
versionada en git, auditable por lectura, con fiscalía de aridad y tipos, y con
differential testing (intérprete vs LLVM) que impide divergencias silenciosas.

## 3. El contrato del juez

El juez recibe los metadatos de un candidato a ingesta y devuelve **una decisión
de cuatro valores**. No ejecuta: decide. La ejecución queda en el servicio de
memoria; el juez solo emite el veredicto.

### 3.1 Entradas (todas strings/bools — nada más cruza la frontera)

| Param | Tipo | Semántica |
|---|---|---|
| `source_kind` | Str | clase de origen: `"user_upload"`, `"session_log"`, `"doc_ingest"`, `"code_context"`, `"external_fetch"` |
| `contains_pii` | Bool | el detector de PII marcó el fragmento (RUT/telefono/email/identificación de tercero) |
| `is_duplicate` | Bool | hash del fragmento ya existe en el índice |
| `size_class` | Int | 0 = chico (<2KB), 1 = medio (<64KB), 2 = grande (≥64KB) |
| `approved` | Bool | el dueño aprobó explícitamente la ingesta (flag de conversación) |

### 3.2 Decisiones (retorno: Int por contrato del gate actual)

- `0` → **REJECT** (no indexar; responder al remitente con la razón)
- `1` → **INDEX** (indexar con metadatos: source_kind, fecha, hash)
- `2` → **INDEX_QUARANTINE** (indexar con flag PII: solo recuperable por el
  dueño, jamás expuesto por `memory_search` a clientes externos)
- `3` → **MERGE** (fragmento duplicado: actualizar vigencia del existente, no
  crear copia)

### 3.3 Regla de ejemplo (v0.7, compila hoy)

```netelpro
(defn filter-rule (source_kind contains_pii is_duplicate size_class approved)
  (if is_duplicate
      3
      (if contains_pii
          (if approved 2 0)
          (if (== source_kind "user_upload")
              (if approved 1 0)
              (if (== source_kind "session_log") 1
                  (if (== source_kind "external_fetch")
                      (if (>= size_class 1) 0 1)
                      1))))))
```

Ley expresada: duplicados se fusionan; PII sin aprobación explícita se rechaza,
con aprobación se cuarentena; lo que el usuario sube para *preguntar* (no para
que se recuerde) requiere aprobación explícita; fetches externos grandes se
rechazan; lo demás se indexa.

## 4. Invariantes que el juez debe satisfacer (verificables)

1. **Determinismo**: misma tupla de entrada → misma decisión (función pura).
2. **Fail-closed ante PII**: sin `approved`, `contains_pii=true` jamás produce `1`.
3. **Fail-closed ante exposición**: `INDEX_QUARANTINE` nunca alimenta el bridge
   externo; eso es invariantes del consumidor, no del juez, pero el juez no
   produce nunca un `1` para PII.
4. **No-inflación**: `is_duplicate=true` jamás produce `1` (solo `3`).
5. **Legibilidad**: la regla completa cabe en una pantalla; toda excepción es
   una rama visible, no un default oculto.

## 5. Integración (fases)

- **Fase A (spec)** — este documento + regla compilada y verificación
  differential en el repo del lenguaje. *Esta fase.*
- **Fase B (guard del RAG)** — threshold floor + respuesta honesta de
  "no encontré nada" en `memory_search` (cambio en el servicio, previo al juez).
- **Fase C (detector)** — `contains_pii` e `is_duplicate` como pre-paso barato
  (regex PII + hash de simhash) antes de llamar al juez.
- **Fase D (cutover)** — el servicio de ingesta llama al juez compilado; el
  criterio difuso actual queda como fallback de paridad (mismo patrón del gate
  Phase I), luego se retira.
- **Fase E (purga con juez)** — batch de revisión del índice existente: el juez
  re-decide por fragmento; los `0`/`3` de la re-evaluación definen la lista de
  purga, **con aprobación explícita del dueño antes de borrar**.

## 6. Fuera de alcance (deliberado)

- Retención temporal (TTL por documento): requiere timestamps en el índice; se
  propone después de la Fase E, como segundo juez (purga por vigencia).
- Borrado automático sin dueño: nunca. La casa no borra memoria por su cuenta.
- Clasificación semántica de calidad del contenido: el juez decide por
  metadatos verificables, no por juicio de valor.