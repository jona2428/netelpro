"""Generador de Jupyter Notebook train_raft_colab.ipynb -- RAFT
(rejection-sampling fine-tuning) para que Qwen2.5-1.5B-Instruct aprenda a
escribir programas Netelpro, usando "compiló + pasó los tests" del corpus
rlvr/tasks/ como señal de recompensa.

Ver docs/superpowers/specs/2026-09-07-rlvr-netelpro-raft-design.md §6 para
el diseño del loop. Mismo patrón de generador que create_moe_notebook.py --
el notebook no se edita a mano, se regenera corriendo este script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def build_raft_notebook() -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎯 Netelpro + RAFT: Entrenando Qwen2.5-1.5B a escribir programas Netelpro correctos\n",
                "### *Rejection-sampling fine-tuning (RAFT/STaR): la recompensa es \"compiló + pasó los tests\", no preferencia humana*\n",
                "\n",
                "Ver `docs/superpowers/specs/2026-09-07-rlvr-netelpro-raft-design.md` para el diseño completo.\n",
                "\n",
                "**Loop por ronda:** muestrear N candidatos por tarea de train -> verificar (compiló + pasó todos los casos) "
                "-> quedarse con los que pasan -> SFT LoRA sobre esos -> repetir con el modelo mejorado. "
                "2-4 rondas, medido contra un split held-out (OOD) que el loop nunca entrena.\n",
                "\n",
                "---\n",
                "### ⚙️ Requisitos previos en Google Colab:\n",
                "1. `Entorno de ejecución` -> `Cambiar tipo de entorno de ejecución` -> **T4 GPU** (gratis).\n",
                "2. Ejecutar cada celda en orden con `Shift + Enter`.",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Instalación de dependencias"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Unsloth + TRL para fine-tune LoRA rápido en T4, llvmlite para netelpro\n",
                '!pip install --no-deps "xformers<0.0.29" "trl<0.15.0" peft accelerate bitsandbytes triton\n',
                '!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n',
                "!pip install -q llvmlite>=0.49\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Clonar el repo y cargar el corpus de tareas RLVR"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clonamos el repo para tener netelpro/ y rlvr/ disponibles como paquetes\n",
                "!git clone https://github.com/jona2428/netelpro.git\n",
                "\n",
                "import sys\n",
                'sys.path.insert(0, "netelpro")\n',
                "\n",
                "from rlvr.tasks import load_all_tasks, split_train_ood\n",
                "from rlvr.prompting import build_prompt\n",
                "from rlvr.verify import verify_program\n",
                "\n",
                "all_tasks = load_all_tasks()\n",
                "train_ids, ood_ids = split_train_ood(list(all_tasks.keys()), ood_fraction=0.2)\n",
                'print(f"Corpus: {len(all_tasks)} tareas -- {len(train_ids)} train, {len(ood_ids)} OOD (held-out)")\n',
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Configuración del experimento"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "NUM_ROUNDS = 3\n",
                "SAMPLES_PER_TASK = 8\n",
                "MAX_KEEP_PER_TASK = 2  # tope de candidatos que pasan por tarea, evita desbalancear el SFT\n",
                "MAX_NEW_TOKENS = 256\n",
                "SAMPLING_TEMPERATURE = 0.8\n",
                "NUM_TEST_CASES = 20\n",
                "EVAL_SEED = 0\n",
                "MODEL_NAME = \"Qwen/Qwen2.5-1.5B-Instruct\"  # base limpio, NO el checkpoint DPO de honestidad (spec Sec 1, Sec 9)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Cargar el modelo base + LoRA\n",
                "Mismo patrón que `train_colab.ipynb` (Unsloth 4-bit + LoRA r=16)."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from unsloth import FastLanguageModel\n",
                "import torch\n",
                "\n",
                "max_seq_length = 1024\n",
                "\n",
                "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                "    model_name=MODEL_NAME,\n",
                "    max_seq_length=max_seq_length,\n",
                "    load_in_4bit=True,\n",
                "    dtype=None,\n",
                ")\n",
                "\n",
                "model = FastLanguageModel.get_peft_model(\n",
                "    model,\n",
                "    r=16,\n",
                '    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],\n',
                "    lora_alpha=16,\n",
                "    lora_dropout=0,\n",
                '    bias="none",\n',
                '    use_gradient_checkpointing="unsloth",\n',
                "    random_state=42,\n",
                ")\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Muestreo y extracción de código Netelpro"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def extract_sl_code(raw_text: str) -> str:\n",
                '    """El modelo puede envolver el programa en un bloque de código markdown --\n',
                "    si hay un fence ``` lo extraemos, si no devolvemos el texto tal cual.\"\"\"\n",
                '    if "```" in raw_text:\n',
                '        parts = raw_text.split("```")\n',
                "        if len(parts) >= 2:\n",
                "            candidate = parts[1]\n",
                '            candidate = candidate.removeprefix("netelpro").removeprefix("lisp").strip()\n',
                "            return candidate\n",
                "    return raw_text.strip()\n",
                "\n",
                "\n",
                "def sample_completions(prompt: str, n: int, temperature: float) -> list[str]:\n",
                "    FastLanguageModel.for_inference(model)\n",
                '    chat = tokenizer.apply_chat_template(\n',
                '        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True\n',
                "    )\n",
                '    inputs = tokenizer([chat] * n, return_tensors="pt", padding=True).to("cuda")\n',
                "    outputs = model.generate(\n",
                "        **inputs,\n",
                "        max_new_tokens=MAX_NEW_TOKENS,\n",
                "        do_sample=True,\n",
                "        temperature=temperature,\n",
                "    )\n",
                "    texts = tokenizer.batch_decode(\n",
                "        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True\n",
                "    )\n",
                "    return [extract_sl_code(t) for t in texts]\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Pass rate en el split OOD (criterio de éxito, spec §7)"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluate_pass_rate(task_ids: list[str], num_samples: int) -> float:\n",
                "    # task_id ES la clave de all_tasks (load_all_tasks() carga por\n",
                "    # nombre de módulo, y por construcción TASK_ID == nombre de módulo\n",
                "    # en todo el corpus -- ver rlvr/tasks/*.py). Sin indirección.\n",
                "    passed_tasks = 0\n",
                "    for task_id in task_ids:\n",
                "        task_module = all_tasks[task_id]\n",
                "        prompt = build_prompt(task_module)\n",
                "        candidates = sample_completions(prompt, num_samples, SAMPLING_TEMPERATURE)\n",
                "        if any(\n",
                "            verify_program(c, task_module, num_cases=NUM_TEST_CASES, seed=EVAL_SEED).passed\n",
                "            for c in candidates\n",
                "        ):\n",
                "            passed_tasks += 1\n",
                "    return passed_tasks / len(task_ids)\n",
                "\n",
                "\n",
                'baseline_pass_rate = evaluate_pass_rate(ood_ids, num_samples=SAMPLES_PER_TASK)\n',
                'print(f"[baseline, sin entrenar] pass rate OOD: {baseline_pass_rate:.1%}")\n',
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Loop RAFT iterativo"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from trl import SFTConfig, SFTTrainer\n",
                "from datasets import Dataset\n",
                "\n",
                "for round_num in range(NUM_ROUNDS):\n",
                '    print(f"\\n=== Ronda {round_num} ===")\n',
                "    sft_examples = []\n",
                "    for task_id in train_ids:\n",
                "        task_module = all_tasks[task_id]\n",
                "        prompt = build_prompt(task_module)\n",
                "        candidates = sample_completions(prompt, SAMPLES_PER_TASK, SAMPLING_TEMPERATURE)\n",
                "        kept = 0\n",
                "        for candidate in candidates:\n",
                "            if kept >= MAX_KEEP_PER_TASK:\n",
                "                break\n",
                "            result = verify_program(candidate, task_module, num_cases=NUM_TEST_CASES, seed=EVAL_SEED)\n",
                "            if result.passed:\n",
                '                sft_examples.append({"prompt": prompt, "completion": candidate})\n',
                "                kept += 1\n",
                '    print(f"Ronda {round_num}: {len(sft_examples)} ejemplos SFT recolectados de {len(train_ids)} tareas")\n',
                "\n",
                "    if not sft_examples:\n",
                '        print("Ninguna tarea de train produjo un candidato que pasara -- se aborta esta ronda")\n',
                "        continue\n",
                "\n",
                "    round_dataset = Dataset.from_list(sft_examples)\n",
                "    FastLanguageModel.for_training(model)\n",
                "    sft_args = SFTConfig(\n",
                '        output_dir=f"netelpro_raft_round_{round_num}",\n',
                "        per_device_train_batch_size=2,\n",
                "        gradient_accumulation_steps=4,\n",
                "        num_train_epochs=2,\n",
                "        logging_steps=5,\n",
                '        save_strategy="no",\n',
                "        warmup_ratio=0.1,\n",
                "        fp16=not torch.cuda.is_bf16_supported(),\n",
                "        bf16=torch.cuda.is_bf16_supported(),\n",
                '        report_to="none",\n',
                '        dataset_text_field="completion",\n',
                "    )\n",
                "    sft_trainer = SFTTrainer(model=model, args=sft_args, train_dataset=round_dataset)\n",
                "    sft_trainer.train()\n",
                "\n",
                "    round_pass_rate = evaluate_pass_rate(ood_ids, num_samples=SAMPLES_PER_TASK)\n",
                '    print(f"[ronda {round_num}] pass rate OOD: {round_pass_rate:.1%}")\n',
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Criterio de éxito (spec §7)\n",
                "El RAFT tiene que superar CLARAMENTE al baseline sin entrenar -- si no, no está enseñando nada que el prompt no diera gratis."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "final_pass_rate = evaluate_pass_rate(ood_ids, num_samples=SAMPLES_PER_TASK)\n",
                'print(f"Baseline (sin entrenar): {baseline_pass_rate:.1%}")\n',
                'print(f"Final (tras {NUM_ROUNDS} rondas de RAFT): {final_pass_rate:.1%}")\n',
                "if final_pass_rate > baseline_pass_rate + 0.1:\n",
                '    print("RAFT superó claramente al baseline -- la señal de recompensa enseñó algo real.")\n',
                "else:\n",
                '    print("RAFT NO superó claramente al baseline -- no declarar éxito, reportar el número tal cual.")\n',
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 9. Exportar a GGUF"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model.save_pretrained_gguf(\n",
                '    "netelpro_qwen1.5b_raft", tokenizer, quantization_method="q4_k_m"\n',
                ")\n",
                'print("✅ Modelo GGUF exportado en la carpeta \'netelpro_qwen1.5b_raft\'.")\n',
            ],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(output_path: Path | str = "train_raft_colab.ipynb") -> None:
    notebook = build_raft_notebook()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "train_raft_colab.ipynb"
    main(target)
