# 🧠 Netelpro DPO Trainer (Google Colab Gratis)

Entrena tu propio modelo de lenguaje de **1.5B parámetros** (`Qwen/Qwen2.5-1.5B-Instruct`) para **eliminar el Teatro de Verificación** usando **DPO (Direct Preference Optimization)** y el dataset auditado por el compilador **Netelpro**.

---

## ⚡ Enlace Rápido
Puedes abrir el notebook directamente en Google Colab con este enlace:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jona2428/netelpro/blob/master/training/train_colab.ipynb)

---

## 📋 Requisitos (100% Gratuito)
* Una cuenta de Google (Gmail).
* Cero GPUs locales: el entrenamiento corre en la nube en una GPU **Nvidia Tesla T4 (16 GB VRAM)** proporcionada gratuitamente por Google Colab.
* Tiempo estimado de entrenamiento: **~15 a 20 minutos**.

---

## 🚀 Paso a Paso

### 1. Abrir el Notebook
1. Entra a [Google Colab](https://colab.research.google.com).
2. Haz clic en la pestaña **GitHub**.
3. En el buscador escribe: `jona2428/netelpro`.
4. Selecciona el archivo: `training/train_colab.ipynb`.
*(Alternativa: descárgalo de tu carpeta local `training/train_colab.ipynb` y súbelo en la pestaña "Subir" de Colab).*

### 2. Activar la GPU T4 Gratis
1. En el menú superior de Colab, ve a:  
   **Entorno de ejecución (Runtime) ➔ Cambiar tipo de entorno de ejecución (Change runtime type)**.
2. En *Acelerador de hardware*, selecciona **T4 GPU**.
3. Haz clic en **Guardar**.

### 3. Ejecutar el Entrenamiento
1. Haz clic en **Entorno de ejecución ➔ Ejecutar todo (Run all)** o presiona `Ctrl + F9`.
2. El notebook hará todo automáticamente:
   * Instalará **Unsloth** (que acelera el entrenamiento y reduce la VRAM a < 2 GB).
   * Descargará `Qwen/Qwen2.5-1.5B-Instruct` en 4-bit.
   * Clonará el repositorio de Netelpro y cargará el dataset DPO (`netelpro_dpo_train.jsonl`).
   * Entrenará los adaptadores LoRA durante 3 épocas con `DPOTrainer`.
   * Evaluará el modelo con una pregunta trampa en vivo.
   * Exportará el modelo a formato **GGUF** (`q4_k_m`) listo para correr en local.

### 4. Descargar y Usar en tu PC (Ollama / Neuromancer)
Al finalizar la última celda:
1. Abre el panel de archivos a la izquierda en Colab (icono de carpeta 📁).
2. Entra a la carpeta `netelpro_qwen1.5b_honest`.
3. Descarga el archivo `.gguf` a tu PC.
4. Para correrlo en **Ollama**:
   Crea un archivo `Modelfile`:
   ```dockerfile
   FROM ./netelpro-qwen-1.5b-q4_k_m.gguf
   PARAMETER temperature 0.2
   ```
   Y crea el modelo en tu terminal:
   ```bash
   ollama create netelpro-honest -f Modelfile
   ollama run netelpro-honest
   ```

---

## 📊 Dataset Auditado por Netelpro
El dataset se encuentra en `training/data/`:
* `netelpro_dpo_train.jsonl` (106 pares de entrenamiento).
* `netelpro_dpo_eval.jsonl` (19 pares de evaluación).

Cada par fue verificado mediante el compilador nativo LLVM de Netelpro (`verification_rule.sl`):
* **Rejected:** Afirmaciones de verificación sin evidencia de máquina (penalizadas por DPO).
* **Chosen:** Reconocimiento honesto de la necesidad de herramientas o respuestas con ejecución válida (recompensadas por DPO).
