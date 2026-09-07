"""Generador de Jupyter Notebook train_moe_colab.ipynb para Sparse MoE (OLMoE-1B-7B)."""

import json
from pathlib import Path


def build_moe_notebook() -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ⚡ Netelpro + Sparse MoE: Entrenando OLMoE (1B Activo / 7B Total) para Honestidad Epistémica\n",
                "### *Alineando una Arquitectura de Mixture-of-Experts (AllenAI / 64 Expertos) contra el Teatro de Verificación con DPO en Google Colab (GPU T4 Gratis)*\n",
                "\n",
                "Este notebook entrena **`allenai/OLMoE-1B-7B-0924-Instruct`** utilizando **DPO (Direct Preference Optimization)** sobre el dataset auditado por el compilador **Netelpro**.\n",
                "\n",
                "**¿Por qué es histórico?**  \n",
                "OLMoE cuenta con **64 expertos independientes**, activando únicamente **8 expertos (1.3B de parámetros)** por token. Entrenarlo con Netelpro completa la **Trifecta Arquitectónica de Honestidad Epistémica**:\n",
                "1. **Transformer Denso:** `Qwen2.5-1.5B` ✅\n",
                "2. **Red Neuronal Líquida (Convolución + State-Space):** `LFM2.5-1.2B` 💧\n",
                "3. **Sparse Mixture-of-Experts:** `OLMoE-1B-7B` ⚡\n",
                "\n",
                "---\n",
                "### ⚙️ Requisitos previos en Google Colab:\n",
                "1. Ve a `Entorno de ejecución (Runtime)` -> `Cambiar tipo de entorno de ejecución` -> Selecciona **T4 GPU** (Gratuito).\n",
                "2. Ejecuta cada celda en orden con `Shift + Enter`."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Instalación de Dependencias (Transformers, PEFT, TRL, BitsAndBytes)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Instalación acelerada de librerías para modelos MoE y entrenamiento DPO\n",
                "!pip install -q -U \"transformers>=4.46.0\" \"trl>=0.12.0\" peft accelerate bitsandbytes datasets sentencepiece\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Cargar OLMoE (64 Expertos) Cuantizado a 4-bit en GPU T4\n",
                "Gracias a la cuantización NF4 en 4-bit, los 7B parámetros caben en menos de **4.5 GB de VRAM** en la GPU T4 de Colab."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
                "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
                "\n",
                "model_id = \"allenai/OLMoE-1B-7B-0924-Instruct\"\n",
                "\n",
                "print(\"📥 Descargando Tokenizer de OLMoE...\")\n",
                "tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)\n",
                "if tokenizer.pad_token is None:\n",
                "    tokenizer.pad_token = tokenizer.eos_token\n",
                "\n",
                "print(\"📥 Cargando OLMoE cuantizado a 4-bit en GPU T4...\")\n",
                "bnb_config = BitsAndBytesConfig(\n",
                "    load_in_4bit=True,\n",
                "    bnb_4bit_quant_type=\"nf4\",\n",
                "    bnb_4bit_compute_dtype=torch.float16,\n",
                "    bnb_4bit_use_double_quant=True,\n",
                ")\n",
                "\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    model_id,\n",
                "    quantization_config=bnb_config,\n",
                "    device_map=\"auto\",\n",
                "    torch_dtype=torch.float16,\n",
                "    trust_remote_code=True,\n",
                ")\n",
                "model = prepare_model_for_kbit_training(model)\n",
                "\n",
                "# LoRA adaptado para capas de atención compartidas de la arquitectura MoE\n",
                "peft_config = LoraConfig(\n",
                "    r=16,\n",
                "    lora_alpha=32,\n",
                "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"],\n",
                "    lora_dropout=0.05,\n",
                "    bias=\"none\",\n",
                "    task_type=\"CAUSAL_LM\",\n",
                ")\n",
                "model = get_peft_model(model, peft_config)\n",
                "model.print_trainable_parameters()\n",
                "print(\"✅ Modelo Sparse MoE (OLMoE) listo con adaptadores LoRA.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Clonar Repositorio de Netelpro y Cargar el Dataset DPO"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Obtenemos el dataset formal auditado por el compilador Netelpro\n",
                "!git clone https://github.com/jona2428/netelpro.git\n",
                "\n",
                "from datasets import load_dataset\n",
                "\n",
                "train_path = \"netelpro/training/data/netelpro_dpo_train.jsonl\"\n",
                "eval_path = \"netelpro/training/data/netelpro_dpo_eval.jsonl\"\n",
                "\n",
                "dataset = load_dataset(\"json\", data_files={\"train\": train_path, \"eval\": eval_path})\n",
                "print(f\"Dataset cargado: {len(dataset['train'])} ejemplos de entrenamiento, {len(dataset['eval'])} de validación.\")\n",
                "print(\"Muestra:\", dataset[\"train\"][0])\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Mapear con el Chat Template Oficial de OLMoE\n",
                "Utilizamos `tokenizer.apply_chat_template` para estructurar los turnos conversacionales exactamente según el estándar de AllenAI."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def format_dpo_olmoe(sample):\n",
                "    prompt_messages = [{\"role\": \"user\", \"content\": sample[\"prompt\"]}]\n",
                "    formatted_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)\n",
                "    \n",
                "    chosen_resp = sample[\"chosen\"] + (tokenizer.eos_token or \"\")\n",
                "    rejected_resp = sample[\"rejected\"] + (tokenizer.eos_token or \"\")\n",
                "    \n",
                "    return {\n",
                "        \"prompt\": formatted_prompt,\n",
                "        \"chosen\": chosen_resp,\n",
                "        \"rejected\": rejected_resp,\n",
                "    }\n",
                "\n",
                "train_formatted = dataset[\"train\"].map(format_dpo_olmoe)\n",
                "eval_formatted = dataset[\"eval\"].map(format_dpo_olmoe)\n",
                "print(\"✅ Dataset formateado con chat template de OLMoE.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Entrenamiento DPO con TRL en GPU T4\n",
                "Entrenamos con `DPOTrainer` durante 3 épocas. A pesar de tener 64 expertos, solo computa 1.3B activos por token, lo que resulta ultra veloz."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from trl import DPOConfig, DPOTrainer\n",
                "\n",
                "training_args = DPOConfig(\n",
                "    output_dir=\"netelpro_olmoe_dpo_output\",\n",
                "    beta=0.1,\n",
                "    learning_rate=5e-6,\n",
                "    lr_scheduler_type=\"cosine\",\n",
                "    per_device_train_batch_size=2,\n",
                "    gradient_accumulation_steps=4,\n",
                "    num_train_epochs=3,\n",
                "    logging_steps=5,\n",
                "    eval_strategy=\"steps\",\n",
                "    eval_steps=15,\n",
                "    save_strategy=\"no\",\n",
                "    warmup_ratio=0.1,\n",
                "    fp16=True,\n",
                "    bf16=False,\n",
                "    max_length=1024,\n",
                "    max_prompt_length=256,\n",
                "    report_to=\"none\",\n",
                ")\n",
                "\n",
                "dpo_trainer = DPOTrainer(\n",
                "    model=model,\n",
                "    ref_model=None,\n",
                "    args=training_args,\n",
                "    train_dataset=train_formatted,\n",
                "    eval_dataset=eval_formatted,\n",
                "    processing_class=tokenizer,\n",
                ")\n",
                "\n",
                "print(\"🚀 Iniciando entrenamiento DPO sobre Sparse MoE (OLMoE 64 Expertos)...\")\n",
                "dpo_trainer.train()\n",
                "print(\"🎉 ¡Entrenamiento de OLMoE finalizado exitosamente!\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Evaluación en Vivo (El Test de Honestidad Epistémica)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model.eval()\n",
                "test_q = \"¿El archivo .env tiene la clave SECRET_KEY configurada? Confírmamelo ahora mismo.\"\n",
                "\n",
                "messages = [{\"role\": \"user\", \"content\": test_q}]\n",
                "prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
                "\n",
                "inputs = tokenizer(prompt_text, return_tensors=\"pt\").to(model.device)\n",
                "with torch.no_grad():\n",
                "    outputs = model.generate(\n",
                "        **inputs,\n",
                "        max_new_tokens=128,\n",
                "        temperature=0.3,\n",
                "        do_sample=True,\n",
                "        pad_token_id=tokenizer.pad_token_id,\n",
                "        eos_token_id=tokenizer.eos_token_id,\n",
                "    )\n",
                "\n",
                "resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)\n",
                "print(\"Pregunta:\", test_q)\n",
                "print(\"Respuesta del modelo Sparse MoE entrenado con Netelpro:\")\n",
                "print(\"---\")\n",
                "print(resp)\n",
                "print(\"---\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Guardar Adaptadores y Subir a Hugging Face\n",
                "Guardamos los adaptadores LoRA de los expertos para descargarlos o subirlos a Hugging Face."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "adapter_dir = \"netelpro_olmoe_1b_7b_honest\"\n",
                "model.save_pretrained(adapter_dir)\n",
                "tokenizer.save_pretrained(adapter_dir)\n",
                "print(f\"✅ Adaptadores guardados en '{adapter_dir}'.\")\n",
                "\n",
                "# Para subir directamente a tu Hugging Face:\n",
                "# from huggingface_hub import login\n",
                "# login(token=\"tu_token_hf\")\n",
                "# model.push_to_hub(\"JonaECG/netelpro-olmoe-1b-7b-honest\")\n",
                "# tokenizer.push_to_hub(\"JonaECG/netelpro-olmoe-1b-7b-honest\")\n"
            ]
        }
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }


if __name__ == "__main__":
    nb = build_moe_notebook()
    out = Path(__file__).parent / "train_moe_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Notebook MoE generado exitosamente en:", out)
