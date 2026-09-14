# 🏛️ Teo v1: Manifiesto y Manual de Arquitectura en Silicio

**Teo v1** es el modelo neuro-simbólico personal desarrollado por **Jona** dentro del ecosistema **Netelpro**.

---

## 1. Principios Fundamentales

1. **Pensamiento Previo en Silicio (`<|thought|>`):** Ninguna afirmación técnica o fragmento de código se emite de forma impulsiva; el modelo evalúa precondiciones, análisis asintótico ($O$) y consecuencias lógicas en una fase de reflexión previa.
2. **Multi-Programador Políglota:** Dominio de 8 lenguajes primarios:
   - **Python** (Estructuras de datos, decoradores, generadores de bajo consumo de RAM, PyTorch).
   - **JavaScript / TypeScript** (Uniones discriminadas, `async`/`await`, `Promise.allSettled`, debounce).
   - **C & C++** (Gestión manual estricta con `malloc`/`free`, prevención de fugas y paradigma RAII con `std::unique_ptr`).
   - **Rust** (Sistema de posesión *ownership*, préstamos `&`/`&mut`, propagación con `Result<T, E>` y operador `?`).
   - **Go** (Concurrencia canónica con *worker pools*, goroutines, canales y manejo explícito `if err != nil`).
   - **SQL** (Consultas avanzadas, Common Table Expressions y funciones de ventana `ROW_NUMBER() OVER`).
   - **Bash & PowerShell** (Automatización robusta con `set -euo pipefail` y filtrado administrativo seguro).
   - **HTML5 & CSS3** (Maquetación semántica moderna con Flexbox y CSS Grid).
3. **Reivindicación de la iGPU UMA:** Integración nativa con **DirectML** para aprovechar los 448 núcleos gráficos de la **AMD Radeon Vega 7**, alcanzando más de **22 tokens por segundo** con latencia sub-50ms.
4. **Verificación Lógica del Trívium:** Auditoría formal integrada para detectar falacias retóricas y garantizar razonamiento válido.

---

## 2. Especificaciones Técnicas

* **Neuronas Activas:** 6,565,376 parámetros
* **Profundidad:** 6 Capas Transformer Causales
* **Atención:** 8 Cabezales con proyección causal
* **Dimensión Oculta:** 256 canales latentes
* **Contexto:** 384 tokens
* **Vocabulario:** 6,750 tokens
* **Aceleración Hardware:** DirectML (DirectX 12 Compute) + SIMD AVX2 (12 Hilos)

---

## 3. Comandos de Operación

Iniciar la terminal de Teo:
```powershell
C:\Python314\python.exe examples/teo_chat.py
```

Comandos en sesión:
* `/device` : Diagnóstico del acelerador en silicio activo (DirectML iGPU vs CPU).
* `/remember <hecho>` : Grabar conocimiento persistente en el Banco de Memoria Binaria Viva.
* `/memories` : Consultar las ranuras de memoria activas.
* `/trivium <texto>` : Auditar falacias dialécticas.
* `/clear` : Reiniciar el contexto conversacional.
* `/exit` : Cerrar sesión.
