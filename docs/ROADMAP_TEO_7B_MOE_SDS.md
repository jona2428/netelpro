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

## 6. Bitácora de Ejecución (Claude Sonnet 5)

### 2026-09-14 — Cancelación Fase 1 + arranque Fase 3 (SDS)

* **Fase 1 (Teo v2 124M) cancelada por el usuario.** Diagnóstico del run en Kaggle: loss cayó normal 10.48→~3.5 en los primeros ~3000 steps, pero se estancó (oscilando 3.5-4.2) durante 32.000+ steps más porque el corpus solo tenía **2 shards** — el modelo llegó a **epoch 57** sobre el mismo dataset chico (riesgo de sobreajuste, sin ganancia real). Decisión: saltar directo a Fase 3 (SDS) con el corpus masivo, en vez de seguir escalando Teo v2 Transformer.
* **Trainer nuevo para Netelpro SDS:** [`training/train_teo_sds.py`](file:///c:/Users/Jona/Documents/netelpro/training/train_teo_sds.py). Espeja la arquitectura de `train_teo_v2.py` (mismo formato de corpus packed uint16, mismo loop de checkpointing/resume/LR schedule) pero instancia `NetelproSDSModel` (16 capas, d_model 1024, d_state 16, expand 2 → d_inner 2048, vocab 32768 por default — Fase 3 del plan). Incluye smoke test CPU (`--smoke-test`) que verifica loss decreciente y resume correcto.
* **Bug real encontrado y corregido en `netelpro/neuro/dynamic_state.py`** (`NetelproSDSModel.forward`, no relacionado al script nuevo): construía `LayerAuditRecord(...)` con campos que no existen en el dataclass (`neuron_index`, `pre_activation`, `scaled_value`, `gate_allowed`, `gate_reason`, `control_flag`) y pasaba `verified=...` a `AuditCertificate`, kwarg inexistente — el forward pass crasheaba con `targets` siempre, en cualquier llamada de entrenamiento. Se corrigió replicando el patrón correcto ya usado en `netelpro/neuro/transformer.py` (agregación por capa: `total_neurons`, `active_neurons`, `suppressed_neurons`, `mean_potential`, `details`). Verificado con smoke test: loss 10.44→6.21 en 200 steps, checkpoint válido, resume continuo.
* **Pendiente para cerrar Fase 3:**
  1. Generar corpus masivo real (250-500M tokens) con `training/data/build_massive_corpus.py` — requiere acceso a HuggingFace datasets (Cosmopedia v2, fineweb-2, codeparrot), mejor correrlo en Kaggle por ancho de banda/tiempo.
  2. Entrenar SDS con `train_teo_sds.py` sobre ese corpus (Kaggle T4, igual que Teo v2).
  3. Validar benchmark de memoria O(1) real con [`examples/sds_benchmark.py`](file:///c:/Users/Jona/Documents/netelpro/examples/sds_benchmark.py) sobre el checkpoint entrenado.
* **Aún no existe (bloqueante para Fase 4):** `NetelproMoERouter` — el Sparse Upcycling a MoE 3B-7B no tiene código todavía. `training/create_moe_notebook.py` es otra cosa (finetune LoRA/DPO de OLMoE de HuggingFace, no el upcycling custom del roadmap).
* **Rebalanceo bilingüe del corpus** (`training/data/build_massive_corpus.py`): el usuario detectó que el corpus original sesgaba fuerte hacia prosa cruda en inglés (Cosmopedia) vs. diálogo instrucción-respuesta en español (Alpaca-ES) — un modelo chico entrenado poco tiempo nunca fija el patrón Q&A en ambos idiomas así, y responde incoherente (ya lo habían visto en una corrida anterior del proyecto). Se agregaron dos streams nuevos:
  - `stream_english_instructions`: Alpaca inglés (`tatsu-lab/alpaca`), simétrico al de Alpaca-ES — enseña el mismo patrón pregunta→respuesta en inglés, no solo prosa suelta.
  - `generate_translation_stream` + banco `TRANSLATION_PAIRS`: 15 pares ES↔EN curados a mano (identidad Teo, conceptos Netelpro, código, física, conversación), generan pares "Traduce al inglés/español: ..." en ambas direcciones — enseña la correspondencia directa entre idiomas, que antes no existía en el corpus.
  - Schedule reponderado: `cosmopedia_stem` bajó de peso 3→2, se sumaron `translation_pairs` (peso 2) y `english_instructions` (peso 2). Los dos streams nuevos están exceptuados de deduplicación (como `reasoning_lore`) porque son bancos chicos intencionalmente repetidos vía `persona_multiplier`.
  - Verificado con smoke test offline (`offline=True`, corpus chico) — compila sin errores. No invalida shards ya generados en la corrida de Kaggle en curso; solo aplica a partir del próximo `git pull` + reinicio de la celda de compilación.
* **3 bugs reales encontrados en la corrida real de Kaggle** (log del usuario: solo 3/20 shards, 50M/500M tokens, terminó con "All available document streams reached completion" y un crash tipo `Fatal Python error: PyGILState_Release` al final):
  1. **Cosmopedia roto:** `load_dataset("HuggingFaceTB/cosmopedia-v2", split="train", streaming=True)` faltaba el nombre del config (`Config name is missing`) — caía siempre al banco local de 4 ejemplos. Arreglado pasando `"cosmopedia-v2"` como config.
  2. **Código roto (segundo intento):** `codeparrot/github-code` Y `codeparrot/github-code-clean` **ambos** usan un loading script (`github-code.py` / `github-code-clean.py` respectivamente) que `datasets >= 4.0` no ejecuta bajo ningún nombre de config — no hay fix de parámetros, el dataset en sí no es Parquet-nativo. Cambiado a **`bigcode/the-stack-smol`** (Parquet real, una carpeta por lenguaje: `data/python`, `data/rust`, `data/cpp`, `data/c-sharp`). **Limitación conocida:** solo ~10k muestras por lenguaje (~40k documentos total) — se agota rápido en una corrida de 500M tokens y el dominio `systems_code` puede quedar sin aportar nada durante buena parte de la compilación. Aceptable por ahora (mejor que 0 docs reales); si hace falta más volumen de código real más adelante, evaluar `bigcode/the-stack-dedup` (requiere aceptar términos de acceso en HuggingFace) o convertir `github-code` a Parquet manualmente vía `convert_to_parquet`.
  3. **Todo detectado como duplicado:** el `--out-dir data/teo_v2_massive` reusado ya tenía `dedup_state.json` con 96.788 hashes de una corrida anterior (Teo v2, mismas fuentes HF). Como los streams de HuggingFace siempre reinician desde el principio del dataset en cada ejecución del script (no hay cursor persistente entre sesiones), la mayoría de los documentos nuevos coincidían con hashes ya vistos → deduplicador los descartaba casi todos → "all streams reached completion" prematuro. **Recomendación: usar un `--out-dir` nuevo y limpio** (ej. `data/teo_sds_massive`) para la corrida real de 500M tokens, no reusar el directorio viejo.
  4. El `Fatal Python error: PyGILState_Release` al final es un crash de un hilo de reintento en background de la librería `datasets`/`huggingface_hub` durante el cierre del proceso — ocurre DESPUÉS de que el manifest ya se cerró y los shards ya se flushearon a disco (dentro del `finally`), así que no pierde datos ya escritos; es ruido cosmético del teardown, no bloqueante.
* **Auditoría de investigación (GPT, prompt de investigador) confirmó un segundo bug real en `dynamic_state.py`:** el término de skip `self.D * x_conv` (dentro de `NetelproSDSCell.step` y `.forward`) se sumaba al output de la celda **sin pasar por el gate de `control_flag`** — el estado recurrente `h` sí colapsaba a cero con `control_flag=0` (vía `NetelproSiliconStateSTE`), pero el output final de la celda seguía filtrando señal no-nula por ese skip. Esto contradecía la ley documentada en este mismo roadmap ("Fail-closed: si control_flag == 0, el estado colapsa a 0.0"). **Corregido:** ahora `y_t`/`out` colapsan a `torch.zeros_like(...)` explícitamente cuando `control_flag == 0`, en ambos paths (`step()` recurrente e inferencia, y `forward()` secuencial de entrenamiento). Verificado con test aislado a nivel de celda: `flag=1` → output no-nulo (0.048 max abs); `flag=0` → exactamente `0.0`. Smoke test de `train_teo_sds.py` sigue pasando igual (loss 10.44→6.21).
  - Investigación completa de GPT (Líneas 2: cuantización BitNet/GPTQ/AWQ vs. plan actual; Línea 1: Product-Key Memory Layers) concluyó: no cambiar el plan de Fase 4 (INT4/INT3 post-entrenamiento) ahora, PKM queda para después — ninguna evidencia obliga a adoptarlas de inmediato. Ver conversación para el reporte completo con fuentes (BitNet, BitNet b1.58, GPTQ, AWQ, Lample et al. 2019, Berges/Oğuz et al. 2024).
* **Diseño de `NetelproMoERouter` (Fase 4) — investigación GPT resuelve una ambigüedad real del roadmap:** la frase original "clonar las capas de estado dinámico en 8-16 expertos" es incorrecta según Sparse Upcycling estándar (Komatsuzaki et al. 2022). Cada `NetelproSDSBlock` tiene DOS sub-capas distintas: `self.sds` (stateful, recurrente) y `self.mlp` (stateless, FFN). Sparse Upcycling clona el FFN (equivalente a `self.mlp`), no la mezcla temporal (equivalente a `self.sds`) — clonar la celda SDS completa no tiene precedente limpio: no está resuelto qué estado (`h`, `conv_state`) hereda un experto cuando el router lo selecciona después de que otro experto procesó el token anterior. Precedentes SSM+MoE reales (MoE-Mamba, Routing Mamba, Swimba) mantienen una única trayectoria de estado compartida, no clonan la celda recurrente entera.
  - **Recomendación adoptada:** primera variante de Fase 4 clona `NetelproMLP` en expertos (top-1/top-2 + 1 compartido, patrón DeepSeekMoE), mantiene `NetelproSDSCell` compartida sin cambios. Clonar SDS queda como experimento separado, falsable, no como el plan por defecto.
  - Balanceo de carga: pérdida auxiliar estilo Switch Transformer (Fedus et al. 2021); router en FP32; top-k no es diferenciable per se, el aprendizaje pasa por los pesos continuos.
  - **Pendiente de decisión del usuario:** GPT también detectó que el fix de fail-closed (ver arriba) solo cubre la celda SDS individual — el bloque completo (`NetelproSDSBlock`) con `control_flag=0` deja pasar el residual sin cambios (esa capa se "salta", el resto del modelo sigue funcionando normal). Falta confirmar si la intención real es esa (suprimir solo esa capa) o un killswitch en cascada (todo el modelo colapsa). No implementado hasta confirmar.
  - Fuentes: Komatsuzaki et al. 2022 (Sparse Upcycling), MoE-Mamba (Pióro et al. 2024), Routing Mamba (Zhan et al. 2025), Swimba (Du et al. 2026 preprint), Switch Transformer (Fedus et al. 2021), DeepSeekMoE (Dai et al. 2024), DEMix (Gururangan et al. 2021).
* **Corpus masivo Fase 2 — completado en la práctica:** 175.4M tokens (7 shards completos + 1 parcial, 217,680 docs, cero padding), generado en Kaggle tras resolver 5 bugs reales (Cosmopedia sin config, `github-code`/`github-code-clean` con loading script deprecado → `bigcode/the-stack-smol` gated + HF_TOKEN, crash de `hf_transfer`/PyGILState en el sandbox de red de Kaggle, y la celda de empaquetado apuntando a la carpeta vieja `teo_v2_massive` en vez de `teo_sds_massive`). Decisión de cortar en 175M en vez de perseguir los 500M del plan original: retorno decreciente peleando contra una librería inestable, 175M ya es un corpus real y sano para arrancar Fase 3. Backup local en `data/teo_sds_massive_corpus_175M.tar.gz` (gitignorado, no se sube — supera el límite de 100MB de GitHub).
* **Notebook Kaggle de entrenamiento real de Teo-SDS (Fase 3):** [`training/train_teo_sds_kaggle.ipynb`](file:///c:/Users/Jona/Documents/netelpro/training/train_teo_sds_kaggle.ipynb) (generador: [`training/create_teo_sds_kaggle_notebook.py`](file:///c:/Users/Jona/Documents/netelpro/training/create_teo_sds_kaggle_notebook.py)). Monta el corpus como Kaggle Dataset (no vuelve a golpear HuggingFace, no debería toparse con el crash de `datasets`), corre el smoke test opcional, entrena con `train_teo_sds.py` en GPU T4 con auto-resume, prueba inferencia O(1) real (stepping recurrente, cero KV-Cache) con `model.generate()`, y empaqueta el checkpoint final.
* **Notebook Kaggle del corpus masivo generado:** [`training/build_massive_corpus_kaggle.ipynb`](file:///c:/Users/Jona/Documents/netelpro/training/build_massive_corpus_kaggle.ipynb) (generador: [`training/create_massive_corpus_kaggle_notebook.py`](file:///c:/Users/Jona/Documents/netelpro/training/create_massive_corpus_kaggle_notebook.py)). Solo compila corpus (no entrena): target 20 shards × 25M tokens = 500M tokens, streaming Cosmopedia v2 + Alpaca-ES + FineWeb-2 español + código de sistemas + lore Netelpro, con `--resume` activo. Soporta reanudar entre sesiones de Kaggle (subiendo el `.tar.gz` parcial como Dataset) y al final empaqueta todo para subir como Kaggle Dataset reutilizable en `train_teo_sds.py` sin re-descargar de HuggingFace.

---
*¡El silicio tiene memoria, y Teo está vivo en el repo!*
