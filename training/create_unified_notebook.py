"""Generador de Jupyter Notebook train_unified_kaggle.ipynb -- Reentreno unificado
(RAFT -> DPO) sobre Qwen2.5-1.5B-Instruct.

Une las dos formas de entrenamiento que existen en el repo:
- Forma A (train_colab.ipynb): DPO de honestidad epistemica, dataset auditado por
  HonestyGuard, beta=0.1, lr=5e-6, 3 epocas.
- Forma B (train_raft_kaggle.ipynb): RAFT con senal de compilador Netelpro,
  pool acumulado (v2), 5 rondas, 16 muestras/tarea.

Orden: RAFT primero (habilidad), DPO despues (alineacion) -- mismo principio que
RLHF canonico (SFT -> preferencia). La eval integrada usa el protocolo OFICIAL
del eval_all_models_kaggle.ipynb (system prompt importado de vtb_ood_runner +
seed por muestra + pass@8 + scorer compartido) en 3 puntos: vanilla (ancla) ->
post-RAFT -> post-DPO. Asi la corrida produce la fila base que falta en la
tabla publica Y la fila unificada, ambas comparables con eval_all_20260909-0320.

Mismo patron de generador que create_raft_notebook.py -- el notebook no se
edita a mano, se regenera corriendo este script.

Patron de celdas: cada celda se define como raw triple-single-quoted string
(r'''...''') partido en lineas reales por _lines(). El raw string deja los
backslashes del codigo de celda literales (el escape del f-string de celda llega
intacto al notebook) y el delimitador de comilla simple evita choque con los
docstrings de doble comilla del codigo de celda.
"""

from __future__ import annotations

import json
from pathlib import Path


def _lines(src: str) -> list[str]:
    """Parte una raw string en lineas con terminador de linea (formato source de nbformat)."""
    body = src.strip("\n")
    return [ln + "\n" for ln in body.split("\n")]


def build_unified_notebook() -> dict:
    cells: list[dict] = []

    def md(source: str) -> None:
        cells.append(
            {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}
        )

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
# 🎯 Netelpro: Reentreno Unificado (RAFT -> DPO) -- Qwen2.5-1.5B
### *Una sola corrida, las dos senales: compilador Netelpro (skill) + preferencia de honestidad (alineacion)*

**Por que existe este notebook:** los modelos publicados salieron de dos formas distintas que nunca se combinaron --
- `netelpro-qwen2.5-1.5b-honest` (DPO sobre 25 templates de honestidad, sin codigo Netelpro)
- `netelpro-qwen2.5-1.5b-raft` / `raft-v2` (RAFT con senal de compilador sobre base limpia, sin dataset de honestidad)

La tabla de `eval_all_20260909-0320` mostro que las cohortes no son comparables entre si (dataset y metodo cambiaron
juntos) y que NINGUN modelo tiene ambas habilidades. Este notebook entrena las dos en secuencia sobre el mismo
adaptador y mide la trayectoria completa con el protocolo oficial del eval.

**Orden de fases (decision de diseno):** RAFT primero, DPO despues -- skill first, alignment last (mismo principio
que RLHF canonico: SFT -> preferencia). El DPO con lr=5e-6 es suave; la eval integrada mide AMBOS ejes antes y
despues de cada fase, asi el costo del DPO sobre el pass@8 OOD es visible en la tabla, no asumido.

**Trayectoria medida (protocolo oficial, comparable con la tabla publica):**
1. `vanilla` -- base sin entrenar (la 5a fila que falta en la tabla, ancla absoluta)
2. `raft-round-{k}` -- tras cada ronda de RAFT (curva de aprendizaje)
3. `post-dpo` -- tras DPO de honestidad sobre el checkpoint RAFT

---

### ⚠️ Antes de correr esto por primera vez en Kaggle (no lo saltes)

1. **Verificacion de telefono, obligatoria.** Kaggle exige verificar tu numero por SMS para activar
   GPU/Internet. Perfil -> Settings -> Phone verification. **Numero real de carrier, no VoIP.**
2. **Accelerator: GPU T4 x2** (o P100). Panel derecho -> Settings -> Accelerator.
3. **Internet: On.** Mismo panel -> Internet. Sin esto, pip y git clone fallan.
4. **Cuota:** ~30h/semana de GPU T4. Sesion individual tope 12h. Esta corrida = RAFT completo
   (5 rondas x 16 muestras) + DPO (3 epocas) + 7 evaluaciones integradas (VTB-30 + pass@8 cada una).
   Estimado 4-7h; si te preocupa, baja NUM_ROUNDS o SAMPLES_PER_TASK en la celda de configuracion.
5. **Persistencia:** "Save Version" -> "Save & Run All (Commit)" para que el output en `/kaggle/working/`
   (reportes + GGUF) sobreviva el cierre de sesion.
""")

    # ------------------------------------------------------------------
    md("## 1. Instalacion de dependencias")
    code(r"""
import os
os.environ["WANDB_DISABLED"] = "true"  # Kaggle a veces pide API key de wandb y cuelga el kernel.

# Unsloth + TRL para fine-tune LoRA rapido en T4, llvmlite para netelpro.
# unsloth[kaggle-new] (no colab-new): el extra correcto para la imagen base de Kaggle.
!pip install --no-deps "xformers<0.0.29" "trl<0.15.0" peft accelerate bitsandbytes triton
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q "llvmlite>=0.49"
""")

    md("### 1b. Verificar GPU (antes de importar unsloth)")
    code(r"""
# Unsloth se niega a importar sin acelerador CUDA, con un NotImplementedError criptico.
# Fallamos temprano y en claro: sin GPU no hay experimento.
import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "No hay GPU activa. En Kaggle: panel derecho -> Settings -> Accelerator -> GPU T4 x2 "
        "(requiere telefono verificado) -> el kernel se reinicia: corre todo de nuevo desde la celda 1."
    )
print("GPU OK:", torch.cuda.get_device_name(0))
""")

    # ------------------------------------------------------------------
    md(r"""
## 2. Clonar el repo y cargar corpus + instrumentos de evaluacion

Importa lo mismo que `eval_all_models_kaggle.ipynb`: scorer compartido, VTB-30, system prompt
oficial y verificador. La eval integrada es comparable con la tabla publica.
""")
    code(r"""
# Requiere 'Internet: On' en Settings -- sin esto este git clone y los pip install fallan.
!git clone https://github.com/jona2428/netelpro.git

import sys
sys.path.insert(0, "netelpro")

from rlvr.tasks import load_all_tasks, split_train_ood, OOD_TASK_IDS
from rlvr.prompting import build_prompt
from rlvr.verify import verify_program
from benchmarks.honesty_scorer import evaluate_response_honesty, faar, honesty_rate
from benchmarks.vtb_ood_runner import HONESTY_SYSTEM_PROMPT
from benchmarks.vtb_dataset import VTB_CASES

all_tasks = load_all_tasks()
train_ids, ood_ids = split_train_ood(list(all_tasks.keys()), ood_fraction=0.2)
assert sorted(ood_ids) == sorted(OOD_TASK_IDS), "held-out explicito: split debe calzar con OOD_TASK_IDS"
print(f"Corpus: {len(all_tasks)} tareas -- {len(train_ids)} train, {len(ood_ids)} OOD | VTB: {len(VTB_CASES)} casos")
""")

    # ------------------------------------------------------------------
    md("## 3. Configuracion del experimento")
    code(r"""
# --- Fase A: RAFT (misma config de train_raft_kaggle.ipynb, forma B validada) ---
NUM_ROUNDS = 5
SAMPLES_PER_TASK = 16
MAX_KEEP_PER_TASK = 2
MAX_NEW_TOKENS = 256
SAMPLING_TEMPERATURE = 0.8
NUM_TEST_CASES = 20
EVAL_SEED = 0
MAX_VERIFY_STEPS = 1_000_000  # presupuesto de pasos del verificador (fix C1)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # base limpio (spec Sec 1, Sec 9)

# --- Fase B: DPO (misma config de train_colab.ipynb, forma A validada) ---
DPO_BETA = 0.1            # estrictitud de la penalizacion hacia Verification Theater
DPO_LR = 5e-6
DPO_EPOCHS = 3
DPO_MAX_PROMPT_LENGTH = 256

# --- Eval integrada (protocolo oficial de eval_all_models_kaggle.ipynb) ---
PASS_K = 8
VTB_SYSTEM_PROMPT = HONESTY_SYSTEM_PROMPT  # prompt-first: decision VTB-OOD 2026-09-08

trajectory = []  # filas: vanilla -> raft-round-k -> post-dpo (FAAR/honesty + pass@8)
""")

    # ------------------------------------------------------------------
    md(r"""
## 4. Cargar el modelo base + LoRA

Mismo patron que las dos formas (Unsloth 4-bit + LoRA r=16). El adaptador LoRA recien montado
inicializa en cero = el modelo es EXACTAMENTE el vanilla hasta el primer SFT, asi que la eval
'vanilla' se corre sin contaminacion.
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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
""")

    # ------------------------------------------------------------------
    md("## 5. Muestreo, extraccion de codigo y generacion con protocolo oficial")
    code(r"""
def extract_sl_code(raw_text: str) -> str:
    # El modelo puede envolver el programa en un bloque markdown -- si hay fence lo extraemos.
    if "```" in raw_text:
        parts = raw_text.split("```")
        if len(parts) >= 2:
            candidate = parts[1]
            candidate = candidate.removeprefix("netelpro").removeprefix("lisp").strip()
            return candidate
    return raw_text.strip()


def sample_completions(prompt: str, n: int, temperature: float) -> list[str]:
    # Muestreo del loop RAFT (forma B): sin system prompt, batch de n candidatos.
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
        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return [extract_sl_code(t) for t in texts]


def gen_live(prompt: str, seed: int = 0) -> str:
    # Generacion unica con el protocolo OFICIAL del eval: system prompt importado de
    # vtb_ood_runner, misma temperatura, seed fijada por muestra. Comparable 1:1 con
    # la gen del eval_all_models_kaggle.ipynb.
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
        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0]
""")

    # ------------------------------------------------------------------
    md(r"""
## 6. Eval integrada: VTB-30 (FAAR/honesty) + OOD pass@8

Copia EXACTA del protocolo de `eval_all_models_kaggle.ipynb` (celda 3): seed=idx para VTB,
seed=i para pass@8, scorer compartido. Unica diferencia: genera sobre el modelo vivo (unsloth)
en vez de GGUF/transformers descargados.
""")
    code(r"""
import time

def eval_vtb30() -> dict:
    statuses, responses = [], []
    for idx, case in enumerate(VTB_CASES):
        r = gen_live(case.prompt, seed=idx)
        statuses.append(evaluate_response_honesty(r)["status"])
        responses.append({"id": case.id, "status": statuses[-1], "response": r})
    return {"faar": faar(statuses), "honesty": honesty_rate(statuses), "cases": responses}


def eval_ood_pass8() -> dict:
    passed = []
    for tid in sorted(OOD_TASK_IDS):
        task = all_tasks[tid]
        prompt = build_prompt(task)
        ok = False
        for i in range(PASS_K):
            c = extract_sl_code(gen_live(prompt, seed=i))
            if verify_program(
                c, task, num_cases=NUM_TEST_CASES, seed=EVAL_SEED, max_steps=MAX_VERIFY_STEPS
            ).passed:
                ok = True
                break
        if ok:
            passed.append(tid)
    return {"pass_at_k": len(passed) / len(OOD_TASK_IDS), "passed_ids": passed}


def run_eval(label: str) -> dict:
    # Corre ambos ejes sobre el modelo vivo y registra la fila en la trayectoria.
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

    md("### 6b. Ancla: eval del vanilla (5a fila de la tabla publica, gratis)")
    code(r"""
# LoRA recien montado = identidad: esto ES el base Qwen2.5-1.5B-Instruct + system prompt.
# La tabla 0320 no tiene esta fila -- esta corrida la cierra.
run_eval("vanilla")
""")

    # ------------------------------------------------------------------
    md("## 7. Fase A: loop RAFT iterativo (forma B, pool acumulado v2)")
    code(r"""
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

all_sft_examples: list[dict] = []  # pool acumulado -- RAFT canonico (v2)
round_history: list[dict] = []

for round_num in range(NUM_ROUNDS):
    print(f"\n=== Ronda RAFT {round_num} ===")
    sft_examples = []
    for task_id in train_ids:
        task_module = all_tasks[task_id]
        prompt = build_prompt(task_module)
        candidates = sample_completions(prompt, SAMPLES_PER_TASK, SAMPLING_TEMPERATURE)
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
    all_sft_examples.extend(sft_examples)
    print(f"Ronda {round_num}: {len(sft_examples)} nuevos -- pool acumulado: {len(all_sft_examples)}")

    if not sft_examples:
        print("Ningun candidato paso -- se aborta esta ronda")
        continue

    round_dataset = Dataset.from_list(all_sft_examples)
    FastLanguageModel.for_training(model)

    def format_sft_example(example):
        # Mismo formato de sample_completions: si entrenar y muestrear divergen, el modelo
        # entrena en un formato y samplea en otro.
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
        # Dataset rows carry prompt/completion keys -> TRL los clasifica prompt-completion y
        # defaultea completion_only_loss=True, que el fork de Unsloth rechaza junto a
        # formatting_func. Explicit False = full-sequence loss, objetivo RAFT canonico.
        completion_only_loss=False,
    )
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=round_dataset,
        formatting_func=format_sft_example,
    )
    sft_trainer.train()

    # Eval oficial tras cada ronda: la curva de aprendizaje en AMBOS ejes
    round_row = run_eval(f"raft-round-{round_num}")
    round_history.append({"round": round_num, "pool_size": len(all_sft_examples), **round_row})
""")

    # ------------------------------------------------------------------
    md(r"""
## 8. Fase B: DPO de honestidad sobre el checkpoint RAFT (forma A)

El dataset DPO es el mismo de `train_colab.ipynb`: pares (prompt, chosen, rejected) auditados
por el HonestyGuard de Netelpro. Si el repo no trae el JSONL, se regenera en runtime con
`generate_dataset.py` (mismo generador, mismo audit).
""")
    code(r"""
from datasets import load_dataset
from pathlib import Path

train_path = "netelpro/training/data/netelpro_dpo_train.jsonl"
eval_path = "netelpro/training/data/netelpro_dpo_eval.jsonl"

if not Path(train_path).exists():
    # Fallback: JSONL no viene en el repo -- regenerar con el generador canonico.
    # "netelpro/" ya esta en sys.path para `from netelpro.guard import HonestyGuard`
    # (import del generador); agregamos su propio directorio para importarlo como modulo plano.
    print("JSONL no encontrado en el repo -- regenerando con generate_dataset.py (audit HonestyGuard)...")
    import sys
    sys.path.insert(0, "netelpro/training")
    from generate_dataset import export_dataset
    export_dataset(Path("netelpro/training/data"))

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
    md("## 9. Eval final post-DPO + reporte de trayectoria")
    code(r"""
run_eval("post-dpo")

import json, datetime

stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M")

report = {
    "when": stamp,
    "model_name": MODEL_NAME,
    "protocol": "official: system prompt vtb_ood_runner + seed per sample + pass@8 + scorer compartido",
    "config": {
        "num_rounds": NUM_ROUNDS,
        "samples_per_task": SAMPLES_PER_TASK,
        "max_keep_per_task": MAX_KEEP_PER_TASK,
        "dpo_beta": DPO_BETA,
        "dpo_lr": DPO_LR,
        "dpo_epochs": DPO_EPOCHS,
    },
    "trajectory": trajectory,
    "round_history": round_history,
}

with open(f"unified_run_{stamp}.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# Tabla legible para pegar en el chat
print(f"\n=== Trayectoria unificada ({stamp}) ===")
header = f"{'punto':<16} {'FAAR':>8} {'honesty':>10} {'pass@8 OOD':>12}"
print(header)
for r in trajectory:
    print(f"{r['label']:<16} {r['faar']:>7.1f}% {r['honesty']:>9.1f}% {r['pass_at_k_ood']:>11.0%}")
print(f"\nReporte: unified_run_{stamp}.json en /kaggle/working/")
""")

    # ------------------------------------------------------------------
    md("## 10. Exportar a GGUF")
    code(r"""
model.save_pretrained_gguf(
    "netelpro_qwen1.5b_unified", tokenizer, quantization_method="q4_k_m"
)
print("Modelo GGUF exportado en la carpeta 'netelpro_qwen1.5b_unified'.")
""")

    # ------------------------------------------------------------------
    md(r"""
## 11. Criterios de decision (leer ANTES de publicar)

Con la trayectoria en la mano, responder en orden:
1. **Ancla:** `vanilla` + system prompt. Si su FAAR es 0% y pass@8 es 25%, el prompt gratis ya
   empata a los finetunes en honestidad -> el unified solo se justifica si SUBE el pass@8 sobre
   25% sin degradar FAAR.
2. **Ganancia del RAFT:** comparar `vanilla` vs `raft-round-{k}` en pass@8. Curva plana = la
   senal de compilador no mueve la aguja en 5 rondas.
3. **Costo del DPO:** comparar `raft-round-{N}` vs `post-dpo` en pass@8. Si el DPO destruye el
   codigo (pass@8 cae), el orden correcto era DPO -> RAFT y se re-entrena con ese orden.
4. **Dominancia:** si `post-dpo` domina a `honest` y `raft-v2` en los 3 ejes (FAAR, honesty,
   pass@8), ese es el modelo para publicar como `netelpro-qwen2.5-1.5b-unified`.
""")

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


def main() -> None:
    out_path = Path(__file__).parent / "train_unified_kaggle.ipynb"
    nb = build_unified_notebook()
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"Notebook generado: {out_path}")
    print(f"  {len(nb['cells'])} celdas ({n_md} markdown, {n_code} codigo)")


if __name__ == "__main__":
    main()
