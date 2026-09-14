# 🚀 Plan Maestro Netelpro: De Teo v2 (124M) a Teo-MoE SDS (7B)
**Documento Maestro de Arquitectura, Continuidad y Hoja de Ruta**  
*Autor: Jonathan (Arquitecto) & Antigravity (Pair Programmer)*  
*Repositorio:* `https://github.com/jona2428/netelpro.git` (Rama `master`)  
*Último Commit Base:* `66d5850`

---

## 1. Misión y Filosofía del Proyecto

Netelpro nació para desafiar el dogma de la industria de la inteligencia artificial:
1. **La industria gasta billones en fuerza bruta:** Apilan capas de Transformers y aumentan el tamaño del KV-Cache hasta que solo servidores de $50,000 USD de Nvidia pueden correr modelos largos.
2. **La tesis de Netelpro:** Inteligencia en el silicio de usuario (**Edge AI**) mediante:
   - **Compuertas de Silicio Acotadas:** Activaciones en el intervalo formal `[-5000, 5000]` con factor de escala `1000.0` (rango real `[-5.0, 5.0]`) y Estimador Recto (STE).
   - **Memoria de Contexto $O(1)$ (Netelpro SDS):** Muerte del KV-Cache mediante Espacios de Estado Dinámicos (SSM) y convoluciones causales (tipo Liquid AI / Mamba).
   - **Mezcla Dispersa de Expertos (Sparse MoE):** Un cerebro total de **7 a 8 Billones de neuronas**, pero activando únicamente **~1 Billón por palabra**, permitiendo que corra a alta velocidad en computadores cotidianos (APUs AMD Ryzen con memoria DDR4).

---

## 2. Estado Actual del Código (Commit `66d5850`)

### Componentes Activos en el Repositorio:
* 🧠 **Teo v2 (Silicon-Bounded Transformer):** [`netelpro/neuro/transformer.py`](file:///c:/Users/Jona/Documents/netelpro/netelpro/neuro/transformer.py)
  - 12 capas, 12 cabezas, $d_{model}=768$, contexto 512, vocabulario BPE 32.768.
  - Entrenando en vivo en Kaggle GPU T4 con pérdida bajando de 10.48 a ~3.50.
* ⚡ **Netelpro SDS (Silicon Dynamic State):** [`netelpro/neuro/dynamic_state.py`](file:///c:/Users/Jona/Documents/netelpro/netelpro/neuro/dynamic_state.py)
  - Motor no-transformer con memoria $O(1)$ estricta (~1.1 MB de estado fijo para 12 capas).
  - Cero KV-Cache. Confinamiento de estado con `NetelproSiliconStateSTE`.
  - Pruebas y benchmark: [`tests/test_dynamic_state.py`](file:///c:/Users/Jona/Documents/netelpro/tests/test_dynamic_state.py) y [`examples/sds_benchmark.py`](file:///c:/Users/Jona/Documents/netelpro/examples/sds_benchmark.py).
* 📚 **Generadores de Datos:**
  - `training/data/build_balanced_dataset.py`: 4 dominios balanceados (Español SFT, Inglés FineWeb-Edu, Código de Sistemas Rust/C++/C#/Python, Filosofía).
  - `training/data/build_massive_corpus.py`: Pipeline de corpus masivo (250M a 500M tokens) con Cosmopedia v2, OpenAssistant, Deep Systems y trazas `<|thought|>`.
* 📓 **Notebooks de Kaggle:**
  - `training/train_teo_v2_t4_kaggle.ipynb`: Entrenamiento inicial de Teo v2 (en ejecución activa).
  - `training/train_teo_v2_continuation_kaggle.ipynb`: Notebook de reanudación y corpus masivo para la segunda tanda.
* 🔤 **Tokenizador:** `data/teo_v2/tokenizer.json` (32.768 tokens, congelado y sincronizado).

---

## 3. Hoja de Ruta Fase por Fase

### Fase 1: Cosecha de Teo v2 (124M Transformer)
* **Objetivo:** Obtener el primer modelo conversacional funcional y validado.
* **Acción:**
  1. Esperar el término de la tanda de entrenamiento de 5 horas en Kaggle (o cuando el usuario decida frenar).
  2. Ejecutar la **Celda 5** del notebook de Kaggle para chatear en vivo con Teo v2 y comprobar su coherencia en español y código.
  3. Ejecutar la **Celda 6** del notebook para descargar el archivo `teo_v2_final_checkpoint.tar.gz`.
  4. En el PC local: extraer el archivo y mover `checkpoint.pt` a `models/teo_v2/checkpoint.pt`.
  5. Probarlo localmente en la terminal ejecutando:
     ```bash
     python examples/teo_chat.py
     ```

### Fase 2: Escalamiento del Corpus y Afinamiento (Kaggle Cuenta 2)
* **Objetivo:** Entrenar con el corpus masivo de 500M tokens utilizando las 30 horas semanales de la segunda cuenta de Kaggle.
* **Acción:**
  1. Abrir `training/train_teo_v2_continuation_kaggle.ipynb` en Kaggle.
  2. Si se desea continuar desde el checkpoint de la Fase 1, subir el `checkpoint.pt` como Kaggle Dataset privado y apuntar el script para reanudar el optimizador AdamW.
  3. Dejar corriendo para alcanzar >250M-500M tokens procesados.

### Fase 3: Transición al Núcleo Teo-SDS (250M - 350M parámetros)
* **Objetivo:** Pasar del Transformer clásico al motor **Netelpro SDS (Non-Transformer)**.
* **Configuración del Modelo:**
  - `vocab_size = 32768`
  - `n_layer = 16`
  - `d_model = 1024`
  - `d_state = 16`
  - `expand = 2` ($d_{inner}=2048$)
  - Parámetros totales: ~300M a 350M.
* **Beneficio en el PC Ryzen:** 
  - Ocupa apenas **~175 MB** en RAM (cuantizado INT4).
  - **Cero KV-Cache** (velocidad fija de 50-70 tok/s sin importar lo largo de la conversación).

### Fase 4: La Gran Frontera — *Sparse Upcycling* a Teo-MoE 7B
* **Objetivo:** Construir un modelo con la sabiduría de 7 Billones de neuronas, pero liviano para el silicio local.
* **Estrategia (Sparse Upcycling):**
  1. **No entrenar 7B desde cero:** Tomar el checkpoint entrenado de Teo-SDS (350M).
  2. **Clonar las capas de estado dinámico** en 8 o 16 expertos paralelos.
  3. **Añadir el Router Disperso (`NetelproMoERouter`):**
     - 1 Experto Compartido (siempre activo para lenguaje base).
     - 8 Expertos Especializados (solo el Top-1 o Top-2 se activan por palabra).
  4. **Entrenamiento de Especialización:** Entrenar únicamente los routers y expertos en dominios específicos (Rust, C++, Filosofía, Razonamiento).
  5. **Cuantización de Silicio INT4/INT3:** El archivo final de 7B se reduce de 14 GB a **~2.8 a 3.2 GB**, corriendo perfectamente en los 16 GB de RAM del Ryzen 5 PRO 4650GE.

---

## 4. Guía para el Siguiente Asistente de IA (Claude / Neuromancer)

Si continúas este trabajo con **Claude 3.5/3.7 Sonnet** o con **Neuromancer**:

> ### 📌 Instrucciones para el Agente Sucesor:
> 1. **Inspecciona el repo:** Lee este archivo (`docs/ROADMAP_TEO_7B_MOE_SDS.md`), el archivo [`walkthrough.md`](file:///c:/Users/Jona/.gemini/antigravity/brain/ef4a6e1c-6a59-402c-94a6-6a267471e707/walkthrough.md) y [`netelpro/neuro/dynamic_state.py`](file:///c:/Users/Jona/Documents/netelpro/netelpro/neuro/dynamic_state.py).
> 2. **Respeta las leyes de silicio de Netelpro:**
>    - Todas las activaciones y estados neuronales DEBEN estar acotados por las reglas de silicio: `[-5000, 5000]` con escala `1000.0` y Estimador Recto (STE).
>    - Fail-closed: si `control_flag == 0`, el estado colapsa a 0.0 (seguridad crítica).
> 3. **Hardware del usuario:**
>    - CPU: AMD Ryzen 5 PRO 4650GE (6 cores / 12 threads), 16 GB DDR4-3200 (memoria UMA compartida con Vega 7 iGPU).
>    - Nube: 2 cuentas de Kaggle con GPU T4 (16 GB VRAM) y 30 horas semanales cada una.
> 4. **No inventes dependencias pesadas:** No uses Triton (no corre nativo en Windows), usa PyTorch puro con operadores vectorizados compatibles con CPU/DirectML y GPU T4.
> 5. **Seguridad:** NUNCA quites `*token*.txt` de `.gitignore` ni hagas públicos tokens de GitHub o HuggingFace.

---

## 5. Recomendación de Jonathan sobre Herramientas de Continuidad

* **Claude 3.5 Sonnet / 3.7 Sonnet (Altamente Recomendado para Arquitectura):**
  - Es actualmente el modelo con mayor precisión en arquitectura de bajo nivel, compiladores, autograd de PyTorch y diseño de sistemas complejos. Te mantendrá el rigor matemático sin alucinar.
* **Neuromancer / Modelos Locales:**
  - Excelente para experimentación libre de cuotas y hackeo privado en tu máquina, pero para escribir código de kernels y redes neuronales de vanguardia, apóyate en Claude pasándole este archivo como contexto inicial.

---
*¡El silicio tiene memoria, y Teo está vivo en el repo!*
