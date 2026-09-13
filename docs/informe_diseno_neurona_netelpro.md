# Informe Técnico: Arquitectura Neuro-Simbólica y Función de Activación Determinista en Netelpro

**Autor del proyecto:** Jonathan (Neuroteo)  
**Fecha:** 13 de Septiembre de 2026  
**Clasificación:** Confidencial / Propietario — Todos los derechos reservados  

---

## 1. Resumen Ejecutivo de la Sesión

Durante esta sesión se completaron dos hitos fundamentales para el proyecto:

1. **Blindaje de Propiedad Intelectual:**
   - Se formalizó el cambio de licencia de código abierto (MIT / Apache-2.0) a **Licencia Propietaria estricta (Todos los derechos reservados)** en los dos repositorios clave:
     - `jona2428/netelpro`: Archivo `LICENSE` y `pyproject.toml` actualizados y confirmados en `master`.
     - `jona2428/neuromancer-teo`: Archivos `LICENSE`, `pyproject.toml`, `README.md`, `INSTALL.txt`, `Cargo.toml` y `src/netelpro-core/LICENSE` actualizados y confirmados en `master`.
   - Se verificó que ambos repositorios se encuentran configurados como **Privados** en GitHub, asegurando la reserva total de los derechos de autor para su creador.

2. **Exploración Arquitectónica de la Neurona Netelpro:**
   - Se debatió la ubicación del motor formal determinista dentro de la unidad de procesamiento biológica/artificial, concluyendo que el punto crítico de inserción es la **función de activación $f(z)$**.
   - Se analizó la tesis de compensación paramétrica: el uso de lógica determinista estricta en modelos pequeños (1B–3B) para lograr una confiabilidad y rigor procedimental comparables a modelos de gran escala (70B) en tareas delimitadas.

---

## 2. Marco Conceptual: La Neurona con Activación Formal

En el paradigma de redes neuronales artificiales, la formulación clásica de una neurona se divide en dos fases:

$$\begin{aligned}
\text{1. Combinación lineal (Suma ponderada):} \quad & z = \sum_{i=1}^n w_i x_i + b \\
\text{2. Función de activación no lineal:} \quad & y = f(z)
\end{aligned}$$

### El problema de las funciones de activación continuas
Las funciones tradicionales como **Sigmoid**, **ReLU**, **GELU** o **SiLU** son curvas suaves y diferenciables diseñadas para el cálculo de gradientes por retropropagación (*backpropagation*). Sin embargo:
- **No toman decisiones categóricas:** Dejan pasar valores intermedios que se propagan como incertidumbre y ruido.
- **Son ciegas a las reglas del sistema:** Una activación biológica o continua no evalúa precondiciones lógicas, límites de memoria ni contratos de invariantes.
- **Fomentan la alucinación probabilística:** El muestreo continuo no cuenta con un mecanismo de corte que fuerce el silencio cuando el potencial no cumple con una ley estricta.

### La propuesta: Activación gobernada por Netelpro
En lugar de una curva puramente flotante, la función de activación incorpora una **compuerta formal verificada en código nativo (LLVM)**. 

La suma ponderada $z$ y un vector de señales de control $\mathbf{c}$ entran a una compuerta lógica discreta compilada:

$$y = \begin{cases} 
g(z) & \text{si } \mathcal{F}_{\text{netelpro}}(z, \mathbf{c}) = 1 \quad \text{(Admitido / Disparo)} \\
0 & \text{si } \mathcal{F}_{\text{netelpro}}(z, \mathbf{c}) = 0 \quad \text{(Inhibición / Corte Fail-Closed)}
\end{cases}$$

---

## 3. Análisis de la Hipótesis: ¿Puede un modelo de 1B–3B igualar a un 70B con lógica determinista?

### Fundamento teórico
Un modelo de 70 mil millones de parámetros resuelve tareas lógicas mediante **fuerza bruta estadística**: ha visto tantos millones de ejemplos durante el preentrenamiento que sus pesos flotantes aproximan la lógica booleana y el razonamiento procedural de manera probabilística. No obstante, esa aproximación sigue siendo no determinista y propensa a fallar en casos límite (*edge cases*).

Por el contrario, un modelo de 1B a 3B parámetros:
- Dispone de una capacidad de memoria paramétrica significativamente menor.
- Gasta una porción desproporcionada de su capacidad intentando recordar reglas fijas, formatos de datos y árboles de decisión.

Al descargar las reglas lógicas a un compilador determinista como Netelpro, se libera al modelo pequeño de esa carga: el LLM se enfoca en lo que mejor hace (asociación semántica y procesamiento de lenguaje), mientras que el silicio garantiza la invariante lógica en microsegundos.

---

## 4. Dónde SÍ y Dónde NO: Límites y Viabilidad Técnica

Para construir sobre cimientos científicos sólidos, es imprescindible delimitar con precisión qué problemas se resuelven y cuáles no mediante este enfoque.

### DÓNDE SÍ (Ventajas competitivas claras)

1. **Consistencia en protocolos y llamadas a herramientas (Tool Calling & API Contracts):**
   - *Por qué:* Los modelos de 1B a menudo fallan al generar JSONs estrictos o al respetar precondiciones de herramientas. Netelpro actúa como un validador que impide cualquier ejecución ilegal en tiempo constante.
2. **Poda de espacio de búsqueda y eliminación de alucinaciones en decisiones discretas:**
   - *Por qué:* Al aplicar máscaras de logits de $-\infty$ antes del muestreo, opciones sintácticamente o lógicamente inválidas se reducen a probabilidad exacta cero ($P=0$).
3. **Eficiencia en hardware limitado (Arquitectura UMA / CPU local):**
   - *Por qué:* Evaluar una regla compilada en LLVM toma nanosegundos y consume unos pocos bytes de memoria. Permite obtener garantías que en la nube requerirían inferir un modelo masivo con costo energético y de latencia prohibitivos.
4. **Entrenamiento guiado por compilador (RLVR / RAFT):**
   - *Por qué:* El compilador sirve como función de recompensa exacta (*ground truth reward*). El modelo aprende a converger hacia sintaxis válida sin necesidad de anotación humana.

---

### DÓNDE NO (Límites estructurales que la lógica pura no puede suplir)

1. **Conocimiento enciclopédico de mundo abierto:**
   - *Por qué:* Una regla lógica no puede inventar hechos históricos, vocabulario especializado ni contexto cultural amplio que no fue visto en el preentrenamiento del modelo pequeño. La lógica estructura la verdad, pero no genera datos fácticos de la nada.
2. **Ambigüedad y extracción inicial ("Garbage In, Garbage Out"):**
   - *Por qué:* Si la entrada del usuario es altamente ambigua o requiere desentrañar sutilezas retóricas complejas, el modelo de 1B puede fallar en extraer correctamente los parámetros $(x_1, x_2, \dots)$ antes de pasarlos a la regla Netelpro. Si la entrada a la regla es errónea, la salida será un rechazo o una decisión incorrecta.
3. **Entrenamiento clásico por descenso de gradiente estándar (*Backpropagation*):**
   - *Por qué:* Las funciones de activación deterministas basadas en condiciones booleanas o tablas de verdad escalonadas tienen derivada cero en casi todo su dominio ($\frac{\partial y}{\partial z} = 0$). No se pueden entrenar directamente con el algoritmo tradicional de *backprop* continuo, requiriendo métodos alternativos como estimadores de gradiente directo (*Straight-Through Estimators*) o aprendizaje por refuerzo con vericadores (RLVR).

---

## 5. Arquitectura de Implementación: Formas de Llevarlo al Código

Existen dos niveles complementarios para implementar este concepto:

### Nivel 1: Capa de Inferencia Neuro-Simbólica (`NetelproNeuroGate`)
*Es el enfoque actualmente implementado y validado en el repositorio.*
- El LLM genera los candidatos y sus logits continuos.
- La compuerta Netelpro en LLVM evalúa la tabla de verdad y reasigna los logits no conformes a $-\infty$.
- **Ventaja:** Cero reentrenamiento; funciona de inmediato con cualquier modelo local (Ollama, llama.cpp, vLLM).

### Nivel 2: Función de Activación Personalizada en el Núcleo de Rust (`netelpro-core`)
Para un módulo o red de perceptrones que opere de forma nativa:

```rust
// Implementación conceptual en netelpro-core (Rust FFI)
#[pyfunction]
pub fn netelpro_activation(z: i64, z_min: i64, z_max: i64, control_flag: bool) -> i64 {
    // Si la compuerta de control está inactiva o z sale de la banda de estabilidad: corte a 0
    if !control_flag || z < z_min || z > z_max {
        0
    } else {
        // Disparo normalizado
        1
    }
}
```

Y su correspondiente regla formal en lenguaje Netelpro (`.sl`):

```netelpro
; Contrato formal de activación por banda de estabilidad y control
(truth-table activation-rule
  (in-range : (Int 0 1))
  (control-valid : (Int 0 1))
  ((0 _) -> 0)
  ((_ 0) -> 0)
  ((1 1) -> 1)
  ((_ _) -> 0))
```

---

## 6. Plan de Trabajo e Hitos Recomendados

Para abordar este desarrollo con método científico y evitar dispersión:

1. **Hito 1: Formalización del Banco de Casos (Evaluación Controlada)**
   - Diseñar un conjunto de 30 escenarios sintéticos donde un modelo de 1.5B falle de forma sistemática por alucinación lógica o desborde de parámetros.
2. **Hito 2: Medición de Línea Base vs. Compuerta Netelpro**
   - Correr la evaluación en frío (sin compuerta) y registrar la tasa de error.
   - Acoplar `NetelproNeuroGate` y verificar la reducción exacta de violaciones a cero en las invariantes evaluadas.
3. **Hito 3: Optimización FFI de Latencia**
   - Asegurar que la llamada de frontera entre la inferencia y el microkernel Rust/LLVM se mantenga en el orden de sub-microsegundos por decisión.

---

## 7. Conclusión

El camino de integrar lógica determinista formal en la toma de decisiones neuronales es una dirección sólida en la investigación moderna de sistemas fiables. Permite resolver el talón de Aquiles de los modelos pequeños —la inconsistencia y la alucinación en tareas de decisión estructurada— sin exigir infraestructura de hardware inalcanzable.

El avance requiere rigor, medición paso a paso y la tranquilidad de construir ladrillo sobre ladrillo, asegurando que cada componente funcione de manera verificable antes de pasar al siguiente.
