"""Generador de Jupyter Notebook train_unified_colab.ipynb -- Port a Google Colab
del reentreno unificado (RAFT -> DPO) sobre Qwen2.5-1.5B-Instruct.

Port de create_unified_notebook.py (Kaggle) a Colab con tres correcciones:

1. Receta de dependencias PROBADA en Colab T4 (la de train_colab.ipynb que
   produjo JonaECG/netelpro-qwen2.5-1.5b-honest): pins --no-deps +
   unsloth[colab-new] (NO kaggle-new, extra nunca probado en ninguna corrida
   exitosa).
2. Dependencias del export GGUF que faltaban en la version Kaggle:
   gguf + sentencepiece + tiktoken (save_pretrained_gguf las exige).
3. Perfil FAST para que ninguna ronda dure ~1h: 3 rondas x 8 muestras x 192
   tokens (v1 probado) + MAX_VERIFY con early-exit. Estimado total ~2h en T4.

La ciencia no se toco: protocolo oficial de eval (system prompt importado de
vtb_ood_runner + seed por muestra + pass@8 + scorer compartido) en cada punto
de la trayectoria: vanilla -> raft-round-{k} -> post-dpo. La corrida produce
la fila base (ancla) que falta en la tabla publica eval_all_20260909-0320 y la
trayectoria completa del unificado, ambas comparables con esa tabla.

Mismo patron de generador que create_unified_notebook.py -- el notebook no se
edita a mano, se regenera corriendo este script. Celdas como raw strings
triple-comilla simple no funcionan con f-strings dobles adentro; se usa
triple-comilla doble con f-strings de comilla simple adentro (patron original).
"""

from __future__ import annotations

import json
from pathlib import Path


def _lines(src: str) -> list[str]:
    """Parte una raw string en lineas con terminador (formato source de nbformat)."""
    body = src.strip("\n")
    return [ln + "\n" for ln in body.split("\n")]


def build_unified_colab_notebook() -> dict:
    cells: list[dict] = []

    def md(source: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(source)})

    def code(source: str) -> None:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _lines(source),
            }
        )

    # ------------------------------------------------------------------
    md(r"""
# 🎯 Netelpro: Reentreno Unificado (RAFT → DPO) en Colab -- Qwen2.5-1.5B
### *Una sola corrida, las dos señales: compilador Netelpro (skill) + preferencia de honestidad (alineación)*

**Por qué existe:** los modelos publicados salieron de dos formas que nunca se combinaron:
- `netelpro-qwen2.5-1.5b-honest` (DPO de honestidad, sin código Netelpro)
- `netelpro-qwen2.5-1.5b-raft` / `raft-v2` (RAFT con señal de compilador, sin dataset de honestidad)

La tabla pública `eval_all_20260909-0320` mostró que NINGUNA tiene ambas habilidades. Este notebook
entrena las dos en secuencia sobre el mismo adaptador y mide la trayectoria completa con el
protocolo oficial del eval.

**Trayectoria medida (comparable 1:1 con la tabla pública):**
1. `vanilla` -- base sin entrenar (la fila que falta en la tabla, ancla absoluta)
2. `raft-round-{k}` -- tras cada ronda de RAFT (curva de aprendizaje en ambos ejes)
3. `post-dpo` -- tras DPO de honestidad sobre el checkpoint RAFT

**Perfil FAST (~2h en T4, ~25-35 min por ronda):** 3 rondas × 8 muestras × 192 tokens.
Los knobs están documentados en la celda de configuración; subirlos restaura la config v2.

---
### ⚙️ Antes de correr (Colab, 30 segundos):
1. **Runtime → Change runtime type → T4 GPU.** (Colab free lo da sin verificar teléfono, a diferencia de Kaggle.)
2. Ejecuta las celdas en orden (`Ctrl+F9` = Run all). La sesión free dura ~12h tope; esta corrida ~2h.
3. Al final, los reportes (`unified_run_*.json/.md`) y el GGUF se descargan solos a tu carpeta de Descargas.
""")

    # ------------------------------------------------------------------
    md("## 1. Instalación de dependencias (receta probada en Colab T4)")
    code(r"""
import os
os.environ["WANDB_DISABLED"] = "true"  # evita el prompt de API key de wandb que cuelga el kernel.

# Receta PROBADA en Colab T4 (train_colab.ipynb que produjo netelpro-qwen2.5-1.5b-honest).
# NO cambiar los pins sin probar: --no-deps evita que pip resuelva torch/cuda desde cero.
!pip install --no-deps "xformers<0.0.29" "trl<0.15.0" peft accelerate bitsandbytes triton
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# llvmlite: runtime del verificador netelpro (eval OOD + señal RAFT).
!pip install -q "llvmlite>=0.49"

# Export GGUF: save_pretrained_gguf los exige -- faltaban en la versión de Kaggle.
!pip install -q gguf sentencepiece tiktoken datasets tabulate pandas
""")

    md("### 1b. Verificar GPU (antes de importar unsloth)")
    code(r"""
# Unsloth falla con NotImplementedError críptico sin CUDA. Fail-loud temprano:
import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "No hay GPU activa. Runtime -> Change runtime type -> T4 GPU -> Guardar. "
        "El runtime se reinicia: corre todo de nuevo desde la celda 1."
    )
print("GPU OK:", torch.cuda.get_device_name(0))
""")

    # ------------------------------------------------------------------
    md(r"""
## 2. Clonar el repo y cargar corpus + instrumentos de evaluación

Importa exactamente lo mismo que `eval_all_models_kaggle.ipynb` (corrida validada):
scorer compartido, VTB-30, system prompt oficial, verificador y corpus RLVR.
`OOD_TASK_IDS` es contrato fijo del repo (`rlvr/tasks/__init__.py`); train = corpus − OOD.
""")
    code(r"""
import sys
from pathlib import Path

if not Path("netelpro").exists():
    !git clone https://github.com/jona2428/netelpro.git
else:
    !cd netelpro && git pull

sys.path.insert(0, "netelpro")

from rlvr.tasks import load_all_tasks, OOD_TASK_IDS
from rlvr.prompting import build_prompt
from rlvr.verify import verify_program
from benchmarks.honesty_scorer import evaluate_response_honesty, faar, honesty_rate
from benchmarks.vtb_ood_runner import HONESTY_SYSTEM_PROMPT
from benchmarks.vtb_dataset import VTB_CASES

all_tasks = load_all_tasks()
train_ids = sorted(set(all_tasks.keys()) - set(OOD_TASK_IDS))
print(f"Corpus: {len(all_tasks)} tareas -- {len(train_ids)} train, {len(OOD_TASK_IDS)} OOD | VTB: {len(VTB_CASES)} casos")
""")

    # ------------------------------------------------------------------
    md("## 3. Configuración del experimento (perfil FAST)")
    code(r"""
# --- Fase A: RAFT (señal de compilador) ---
# Perfil FAST: ronda ~25-35 min en T4. La config v2 (5x16x256) duraba ~1h/ronda.
# Si quieres restaurar v2: NUM_ROUNDS=5, SAMPLES_PER_TASK=16, MAX_NEW_TOKENS=256.
NUM_ROUNDS = 3           # v1 documentó plateau en ronda 2; la ronda 3 confirma la meseta.
SAMPLES_PER_TASK = 8     # valor probado de la v1 (produjo raft-v1); 16 duplica el muestreo.
MAX_KEEP_PER_TASK = 2    # tope de candidatos correctos por tarea (evita desbalance del SFT).
MAX_NEW_TOKENS = 192     # programas netelpro caben en <120 tokens; 256 era colchón caro.
SAMPLING_TEMPERATURE = 0.8
NUM_TEST_CASES = 20
EVAL_SEED = 0
MAX_VERIFY_STEPS = 1_000_000  # presupuesto de pasos del verificador (fix C1)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # base limpio (spec Sec 1, Sec 9)

# --- Fase B: DPO (preferencia de honestidad) ---
DPO_BETA = 0.1            # estrictitud de la penalización hacia Verification Theater
DPO_LR = 5e-6
DPO_EPOCHS = 3
DPO_MAX_PROMPT_LENGTH = 256

# --- Eval integrada (protocolo oficial) ---
PASS_K = 8
VTB_SYSTEM_PROMPT = HONESTY_SYSTEM_PROMPT  # prompt-first: decisión VTB-OOD 2026-09-08

trajectory = []  # filas: vanilla -> raft-round-k -> post-dpo (FAAR/honesty + pass@8)
""")

    # ------------------------------------------------------------------
    md(r"""
## 4. Cargar el modelo base + LoRA

Unsloth 4-bit + LoRA r=16 (patrón de ambas formas probadas). El adaptador recién montado
inicializa en cero: el modelo ES exactamente el vanilla hasta el primer SFT, así que la
eval `vanilla` corre sin contaminación.
""")
    code(r"""
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=(
        ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
        if "lfm" in MODEL_NAME.lower()
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ),
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
""")

    # ------------------------------------------------------------------
    md("## 5. Muestreo, extracción de código y generación con protocolo oficial")
    code(r"""
def extract_sl_code(raw_text: str) -> str:
    # El modelo puede envolver el programa en un fence markdown -- si hay, se extrae.
    if "```" in raw_text:
        parts = raw_text.split("```")
        if len(parts) >= 2:
            candidate = parts[1]
            candidate = candidate.removeprefix("netelpro").removeprefix("lisp").strip()
            return candidate
    return raw_text.strip()


def sample_completions(prompt: str, n: int, temperature: float) -> list[str]:
    # Muestreo del loop RAFT (forma probada): sin system prompt, batch de n candidatos.
    FastLanguageModel.for_inference(model)
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([chat] * n, return_tensors="pt", padding=True).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
    )
    texts = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return [extract_sl_code(t) for t in texts]


def gen_live(prompt: str, seed: int = 0) -> str:
    # Generación única con el protocolo OFICIAL del eval: system prompt importado,
    # misma temperatura, seed fijada por muestra. Comparable 1:1 con eval_all.
    FastLanguageModel.for_inference(model)
    torch.manual_seed(seed)
    msgs = (
        [{"role": "system", "content": VTB_SYSTEM_PROMPT}] if VTB_SYSTEM_PROMPT else []
    ) + [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=SAMPLING_TEMPERATURE,
    )
    return tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0]
""")

    # ------------------------------------------------------------------
    md(r"""
## 6. Eval integrada: VTB-30 (FAAR/honesty) + OOD pass@8

Copia EXACTA del protocolo de `eval_all_models_kaggle.ipynb` (validado en la corrida 0320):
seed=idx para VTB, seed=i para pass@8, scorer compartido, system prompt oficial.
Única diferencia: genera sobre el modelo vivo (unsloth) en vez de GGUF/transformers.
""")
    code(r"""
import time

def eval_vtb30() -> dict:
    statuses, responses = [], []
    for idx, case in enumerate(VTB_CASES):
        r = gen_live(case.prompt, seed=idx)
        statuses.append(evaluate_response_honesty(r)["status"])
        responses.append({"id": case.id, "status": statuses[-1], "response": r})
        # Progreso en vivo: sin esto, 30 generaciones secuenciales parecen un cuelgue.
        print(f"\r  VTB {idx + 1}/{len(VTB_CASES)}", end="", flush=True)
    print(flush=True)
    return {"faar": faar(statuses), "honesty": honesty_rate(statuses), "cases": responses}


def eval_ood_pass8() -> dict:
    passed = []
    ood_sorted = sorted(OOD_TASK_IDS)
    for n, tid in enumerate(ood_sorted, 1):
        task = all_tasks[tid]
        prompt = build_prompt(task)
        ok = False
        tries = 0
        for i in range(PASS_K):
            c = extract_sl_code(gen_live(prompt, seed=i))
            tries = i + 1
            if verify_program(
                c, task, num_cases=NUM_TEST_CASES, seed=EVAL_SEED, max_steps=MAX_VERIFY_STEPS
            ).passed:
                ok = True
                break
        print(
            f"  OOD {n}/{len(ood_sorted)} {tid}: {'PASS' if ok else 'fail'} "
            f"(intentos {tries}/{PASS_K})",
            flush=True,
        )
        if ok:
            passed.append(tid)
    return {"pass_at_k": len(passed) / len(ood_sorted), "passed_ids": passed}


def run_eval(label: str) -> dict:
    print(f"=== EVAL [{label}]: VTB-30 + OOD pass@{PASS_K} ===", flush=True)
    t0 = time.time()
    vtb = eval_vtb30()
    ood = eval_ood_pass8()
    dt = time.time() - t0
    row = {
        "label": label,
        "faar": round(vtb["faar"], 1),
        "honesty": round(vtb["honesty"], 1),
        "pass_at_k_ood": round(ood["pass_at_k"], 3),
        "ood_passed": ood["passed_ids"],
        "vtb_statuses": {c["id"]: c["status"] for c in vtb["cases"]},
        "minutes": round(dt / 60, 1),
    }
    trajectory.append(row)
    print(
        f"[{label}] FAAR={row['faar']}% honesty={row['honesty']}% "
        f"pass@{PASS_K} OOD={row['pass_at_k_ood']:.0%} ({row['minutes']} min)"
    )
    return row
""")

    md("### 6b. Ancla: eval del vanilla (la fila base que falta en la tabla pública, gratis)")
    code(r"""
# LoRA recién montado = identidad: esto ES el base Qwen2.5-1.5B-Instruct + system prompt.
run_eval("vanilla")
""")

    # ------------------------------------------------------------------
    md("## 7. Fase A: loop RAFT iterativo (pool acumulado v2)")
    code(r"""
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

all_sft_examples: list[dict] = []  # pool acumulado -- RAFT canónico (v2)
round_history: list[dict] = []

for round_num in range(NUM_ROUNDS):
    print(f"\n=== Ronda RAFT {round_num} ===")
    t_sample = t_verify = 0.0
    sft_examples = []
    for task_id in train_ids:
        task_module = all_tasks[task_id]
        prompt = build_prompt(task_module)
        t0 = time.time()
        candidates = sample_completions(prompt, SAMPLES_PER_TASK, SAMPLING_TEMPERATURE)
        t_sample += time.time() - t0
        t0 = time.time()
        kept = 0
        for candidate in candidates:
            if kept >= MAX_KEEP_PER_TASK:
                break
            result = verify_program(
                candidate, task_module, num_cases=NUM_TEST_CASES, seed=EVAL_SEED, max_steps=MAX_VERIFY_STEPS
            )
            if result.passed:
                sft_examples.append({"prompt": prompt, "completion": candidate})
                kept += 1
        t_verify += time.time() - t0
    all_sft_examples.extend(sft_examples)
    print(
        f"Ronda {round_num}: {len(sft_examples)} nuevos -- pool acumulado: {len(all_sft_examples)} "
        f"| sample {t_sample/60:.1f}min verify {t_verify/60:.1f}min"
    )

    if not sft_examples:
        print("Ningún candidato pasó -- se aborta esta ronda")
        continue

    round_dataset = Dataset.from_list(all_sft_examples)
    FastLanguageModel.for_training(model)

    def format_sft_example(example):
        # Mismo formato que sample_completions: si entrenar y muestrear divergen,
        # el modelo entrena en un formato y samplea en otro.
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return chat + example["completion"]

    sft_args = SFTConfig(
        output_dir=f"netelpro_unified_round_{round_num}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        logging_steps=1,
        save_strategy="no",
        warmup_ratio=0.1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        # Dataset rows con prompt/completion -> TRL defaultea completion_only_loss=True,
        # que el fork de Unsloth rechaza junto a formatting_func. False = full-sequence
        # loss, objetivo RAFT canónico.
        completion_only_loss=False,
    )
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=round_dataset,
        formatting_func=format_sft_example,
    )
    t0 = time.time()
    sft_trainer.train()
    sft_min = (time.time() - t0) / 60

    # Eval oficial tras cada ronda: curva de aprendizaje en AMBOS ejes
    round_row = run_eval(f"raft-round-{round_num}")
    round_history.append({"round": round_num, "pool_size": len(all_sft_examples), "sft_minutes": round(sft_min, 1), **round_row})
""")

    # ------------------------------------------------------------------
    md(r"""
## 8. Fase B: DPO de honestidad sobre el checkpoint RAFT

El dataset DPO (pares auditados por HonestyGuard) viaja en el repo:
`training/data/netelpro_dpo_train.jsonl` (+ eval). Formato EXACTO de la forma A
(`train_colab.ipynb`) que produjo `honest-qwen`.
""")
    code(r"""
from datasets import load_dataset

train_path = "netelpro/training/data/netelpro_dpo_train.jsonl"
eval_path = "netelpro/training/data/netelpro_dpo_eval.jsonl"
for p in (train_path, eval_path):
    if not Path(p).exists():
        raise FileNotFoundError(f"Dataset DPO no encontrado en el repo clonado: {p}")

dataset = load_dataset("json", data_files={"train": train_path, "eval": eval_path})
print(f"Dataset DPO: {len(dataset['train'])} train, {len(dataset['eval'])} eval")


def format_dpo(sample):
    # Formato EXACTO de la forma A (train_colab.ipynb) que produjo honest-qwen.
    p = f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    c = f"{sample['chosen']}<|im_end|>"
    r = f"{sample['rejected']}<|im_end|>"
    return {"prompt": p, "chosen": c, "rejected": r}

formatted_train = dataset["train"].map(format_dpo)
formatted_eval = dataset["eval"].map(format_dpo)
print("Mapeo DPO completado.")
""")
    code(r"""
from trl import DPOConfig, DPOTrainer

dpo_args = DPOConfig(
    output_dir="netelpro_unified_dpo",
    beta=DPO_BETA,
    learning_rate=DPO_LR,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=DPO_EPOCHS,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=15,
    save_strategy="no",
    warmup_ratio=0.1,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",
)

dpo_trainer = DPOTrainer(
    model=model,  # adaptador LoRA del RAFT sigue montado: DPO sobre el checkpoint RAFT
    ref_model=None,  # Unsloth gestiona el modelo de referencia sin duplicar memoria
    args=dpo_args,
    train_dataset=formatted_train,
    eval_dataset=formatted_eval,
    tokenizer=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=DPO_MAX_PROMPT_LENGTH,
)

print("Iniciando DPO sobre checkpoint RAFT...")
dpo_trainer.train()
print("DPO finalizado.")
""")

    # ------------------------------------------------------------------
    md("## 9. Eval final post-DPO + informe de trayectoria")
    code(r"""
run_eval("post-dpo")

import datetime

stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M")

report = {
    "when": stamp,
    "platform": "colab-t4",
    "model_name": MODEL_NAME,
    "protocol": "official: system prompt vtb_ood_runner + seed per sample + pass@8 + scorer compartido",
    "config": {
        "profile": "fast",
        "num_rounds": NUM_ROUNDS,
        "samples_per_task": SAMPLES_PER_TASK,
        "max_keep_per_task": MAX_KEEP_PER_TASK,
        "max_new_tokens": MAX_NEW_TOKENS,
        "dpo_beta": DPO_BETA,
        "dpo_lr": DPO_LR,
        "dpo_epochs": DPO_EPOCHS,
    },
    "trajectory": trajectory,
    "round_history": round_history,
}

json_path = f"unified_run_{stamp}.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"JSON: {json_path}")

# Informe Markdown con la tabla pública 0320 como referencia citada.
ref_rows = [
    ("JonaECG/netelpro-qwen2.5-1.5b-honest", 3.3, 16.7, 25),
    ("JonaECG/netelpro-lfm2.5-1.2b-honest", 6.7, 30.0, 0),
    ("JonaECG/netelpro-qwen2.5-1.5b-raft", 6.7, 20.0, 25),
    ("JonaECG/netelpro-qwen2.5-1.5b-raft-v2", 3.3, 23.3, 25),
]
md_path = f"unified_run_{stamp}.md"
lines = [
    f"# Trayectoria unificada RAFT->DPO ({stamp}, Colab T4, perfil fast)",
    "",
    "## Esta corrida",
    "| punto | FAAR | honesty | pass@8 OOD | min |",
    "|---|---|---|---|---|",
]
for r in trajectory:
    lines.append(f"| {r['label']} | {r['faar']}% | {r['honesty']}% | {r['pass_at_k_ood']:.0%} | {r['minutes']} |")
lines += [
    "",
    "## Referencia: tabla pública eval_all_20260909-0320 (modelos publicados)",
    "| modelo | FAAR | honesty | pass@8 OOD |",
    "|---|---|---|---|",
]
for name, fa, ho, pa in ref_rows:
    lines.append(f"| {name} | {fa}% | {ho}% | {pa}% |")
lines += [
    "",
    "Protocolo: system prompt de vtb_ood_runner + seed por muestra + pass@8 + scorer compartido (comparable 1:1).",
    "",
]
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"MD: {md_path}")

# Tabla legible en pantalla para pegar en el chat
print(f"\n=== Trayectoria unificada ({stamp}) ===")
print(f"{'punto':<16} {'FAAR':>8} {'honesty':>10} {'pass@8 OOD':>12}")
for r in trajectory:
    print(f"{r['label']:<16} {r['faar']:>7.1f}% {r['honesty']:>9.1f}% {r['pass_at_k_ood']:>11.0%}")
""")

    # ------------------------------------------------------------------
    md("## 10. Exportar a GGUF + descarga de todos los artefactos")
    code(r"""
# Export GGUF (misma vía que produjo los GGUF publicados de raft/raft-v2).
gguf_dir = "netelpro_qwen1.5b_unified"
model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
print("Modelo GGUF exportado en la carpeta", gguf_dir)

# Descarga automática: informe JSON/MD + GGUF.
from google.colab import files
for p in sorted(Path(".").glob("unified_run_*.json")) + sorted(Path(".").glob("unified_run_*.md")):
    files.download(str(p))
for p in sorted(Path(gguf_dir).glob("*.gguf")):
    files.download(str(p))
print("Descargas disparadas -- revisa tu carpeta de Descargas del navegador.")
""")

    # ------------------------------------------------------------------
    md(r"""
## 11. Criterios de decisión (leer ANTES de publicar)

Con la trayectoria en la mano, responder en orden:
1. **Ancla:** `vanilla` + system prompt. Si su FAAR es ~0% y pass@8 es 25%, el prompt gratis
   ya empata a los finetunes en honestidad → el unified solo se justifica si SUBE el pass@8
   sobre 25% sin degradar FAAR.
2. **Ganancia del RAFT:** comparar `vanilla` vs `raft-round-{k}` en pass@8. Curva plana = la
   señal de compilador no mueve la aguja en 3 rondas (subir NUM_ROUNDS a 5 restaura v2).
3. **Costo del DPO:** comparar `raft-round-{N}` vs `post-dpo` en pass@8. Si el DPO destruye el
   código (pass@8 cae), el orden correcto era DPO → RAFT y se re-entrena con ese orden.
4. **Dominancia:** si `post-dpo` domina a `honest` y `raft-v2` en los 3 ejes (FAAR, honesty,
   pass@8), ese es el modelo para publicar como `netelpro-qwen2.5-1.5b-unified`.
""")

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
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


def main() -> None:
    out_path = Path(__file__).parent / "train_unified_colab.ipynb"
    nb = build_unified_colab_notebook()
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"Notebook generado: {out_path}")
    print(f"  {len(nb['cells'])} celdas ({n_md} markdown, {n_code} código)")


if __name__ == "__main__":
    main()