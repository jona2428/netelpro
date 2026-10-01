"""Generator of training/train_receipts_raft_kaggle.ipynb -- Receipts-RAFT
(spec docs/superpowers/specs/2026-10-01-receipts-raft-design.md v0.2, §4, §8 step 3).

Built from train_raft_kaggle.ipynb (RAFT v2) by deterministic transformation:
the install, GPU-check and model+LoRA cells are carried over verbatim (the
frozen part of the protocol: Unsloth 4-bit, LoRA r=16 alpha=16, max_seq_length
1024, batch 2 x grad-accum 4, 2 epochs); the oracle, the data and the loop are
replaced. If an anchor disappears from the source, this aborts instead of
writing a broken notebook.

The notebook runs ONE stage per Kaggle kernel (see rlvr/receipts_raft.py):
the hand audit of §6 sits between stages, and stage r >= 1 refuses to train
unless round r-1's audit labels are committed in the repo it clones.
ARM and STAGE are set in the config cell; training/push_receipts_raft.py
writes one kernel per (arm, stage) with the previous stage as its input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
SOURCE_NOTEBOOK = TRAINING_DIR / "train_raft_kaggle.ipynb"
OUTPUT_NOTEBOOK = TRAINING_DIR / "train_receipts_raft_kaggle.ipynb"

# Cells carried over verbatim, located by anchor text (not by index).
ANCHORS = {
    "install": 'os.environ["WANDB_DISABLED"] = "true"',
    "gpu": "if not torch.cuda.is_available():",
    "model": "FastLanguageModel.get_peft_model(",
}
FROZEN_IN_MODEL_CELL = ("r=16", "lora_alpha=16", "max_seq_length = 1024", "load_in_4bit=True")


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def _find(cells: list[dict], anchor: str) -> dict:
    hits = [c for c in cells if c["cell_type"] == "code" and anchor in _src(c)]
    if len(hits) != 1:
        sys.exit(f"anchor {anchor!r} matched {len(hits)} cells in {SOURCE_NOTEBOOK.name} -- refusing to generate")
    return hits[0]


INTRO = """
# Receipts-RAFT: RAFT con los bytes como verificador

Spec: `docs/superpowers/specs/2026-10-01-receipts-raft-design.md` (v0.2, aprobada 2026-10-01).

Mismo loop que RAFT v2 (`train_raft_kaggle.ipynb`); cambia **solo el oráculo**: en vez de
"¿compiló y pasó los casos?", la recompensa es "¿lo que dijo coincide con los bytes?"
(`rlvr/receipts_reward.py`: auditoría estricta de recibos, negación explícita en
BLOCKED-WRITE, cero claims sin efectos, mínimo 8 palabras y una ruta).

**Una etapa por kernel.** Entre cosecha y entrenamiento hay una auditoría humana de
48 muestras (spec §6); un kernel de Kaggle no puede pausar para eso. La etapa `r >= 1`
se niega a entrenar si `benchmarks/receipts_raft_audit/arm{ARM}_r{r-1}_labels.json`
no está commiteado en el repo con hacking <= 1/48.

| Etapa | Hace |
|---|---|
| 0 | cosecha ronda 0 con el base -> hoja de auditoría |
| 1-4 | gate de auditoría r-1, re-califica el pool completo, entrena, cosecha ronda r |
| 5 | gate de auditoría 4, re-califica, entrena, exporta GGUF |

Brazo A = SFT sobre R=1. Brazo B = DPO on-policy sobre pares (R=1, R=0) del mismo muestreo.
"""

CONFIG = """
# Lo sobreescribe training/push_receipts_raft.py por kernel.
ARM = "A"  # "A" (SFT sobre R=1) | "B" (DPO on-policy)
STAGE = 0  # 0..5, ver tabla de arriba
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # D14: el brazo D cambia solo esta línea
EXPORT_DIR = f"netelpro_qwen1.5b_receipts_raft_{ARM.lower()}"
STATE_DIR = "receipts_raft_state"  # lo que la etapa siguiente recibe como input
AUDIT_SEED = 1000 + STAGE
"""

CLONE = """
# Requiere 'Internet: On'. Se clona en cada etapa: re-calificar con el detector ACTUAL
# es parte del protocolo (spec §6.2) -- un fix del detector entre etapas se propaga solo.
!git clone --depth 1 https://github.com/jona2428/netelpro.git
import sys, json, glob
from pathlib import Path
sys.path.insert(0, "netelpro")
REPO = Path("netelpro")

from rlvr import receipts_raft as rr
assert ARM in rr.ARMS and 0 <= STAGE <= rr.NUM_ROUNDS

prev_pool, prev_adapter = [], None
if STAGE > 0:
    print(rr.require_audit(REPO, ARM, STAGE - 1))  # D11: aborta si falta o si supera 1/48
    found = glob.glob(f"/kaggle/input/*/{STATE_DIR}/pool.json")
    assert len(found) == 1, f"esperaba exactamente un estado previo en /kaggle/input, hay {found}"
    prev_pool = rr.load_pool(Path(found[0]))
    prev_adapter = str(Path(found[0]).parent / "adapter")
    changed = rr.rescore(prev_pool)
    print(f"pool previo: {len(prev_pool)} muestras, re-calificadas con el detector actual: {changed} cambiaron")
"""

MODEL_PREV = """
# Etapa >= 1: continuar el MISMO LoRA de la etapa anterior (RAFT v2 entrena el mismo
# adaptador ronda tras ronda sobre el pool acumulado).
if prev_adapter is not None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    state = load_file(str(Path(prev_adapter) / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, state)
    assert not result.unexpected_keys, result.unexpected_keys[:5]
    print("adaptador de la etapa anterior cargado:", prev_adapter)
"""

SAMPLER = """
def sample_chat(messages, n, temperature):
    FastLanguageModel.for_inference(model)
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat] * n, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, max_new_tokens=rr.MAX_NEW_TOKENS, do_sample=True, temperature=temperature)
    return tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

def prompt_text(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""

TRAIN = """
from datasets import Dataset
torch.manual_seed(0)

if STAGE > 0:
    FastLanguageModel.for_training(model)
    if ARM == "A":
        from trl import SFTConfig, SFTTrainer
        examples = rr.sft_examples(prev_pool)
        print(f"brazo A: {len(examples)} ejemplos SFT (R=1, <= {rr.MAX_KEEP_PER_SCENARIO} por escenario)")
        ds = Dataset.from_list([{"text": prompt_text(e["messages"]) + e["completion"] + tokenizer.eos_token} for e in examples])
        trainer = SFTTrainer(
            model=model, train_dataset=ds,
            args=SFTConfig(
                output_dir="out_sft", per_device_train_batch_size=2, gradient_accumulation_steps=4,
                num_train_epochs=2, logging_steps=1, save_strategy="no", warmup_ratio=0.1,
                fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
                report_to="none", dataset_text_field="text", completion_only_loss=False,
            ),
        )
    else:
        from unsloth import PatchDPOTrainer
        PatchDPOTrainer()
        from trl import DPOConfig, DPOTrainer
        pairs = rr.dpo_pairs(prev_pool)
        print(f"brazo B: {len(pairs)} pares on-policy (chosen R=1, rejected R=0, mismo prompt)")
        ds = Dataset.from_list([{"prompt": prompt_text(p["messages"]), "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs])
        trainer = DPOTrainer(
            model=model, ref_model=None, tokenizer=tokenizer, train_dataset=ds,
            args=DPOConfig(
                output_dir="out_dpo", per_device_train_batch_size=2, gradient_accumulation_steps=4,
                num_train_epochs=2, logging_steps=1, save_strategy="no", warmup_ratio=0.1, beta=0.1,
                max_length=1024, max_prompt_length=768,
                fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(), report_to="none",
            ),
        )
    trainer.train()
"""

HARVEST = """
state = Path(STATE_DIR)
state.mkdir(exist_ok=True)
pool = list(prev_pool)
stats = {"arm": ARM, "stage": STAGE, "model": MODEL_NAME, "prev_pool": len(prev_pool)}

if STAGE < rr.NUM_ROUNDS:
    torch.manual_seed(STAGE)
    fresh = rr.harvest(STAGE, sample_chat)  # ronda STAGE: los mismos 60 escenarios para A y B (D8)
    pool += fresh
    stats["harvest"] = rr.round_stats(fresh)
    sheet = rr.audit_sheet(fresh, seed=AUDIT_SEED)
    Path(f"audit_arm{ARM}_r{STAGE}_sheet.json").write_text(json.dumps(sheet, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats["harvest"], indent=2, ensure_ascii=False))
    print(f"hoja de auditoría: {len(sheet)} muestras R=1 -> etiquetar a mano, guardar como "
          f"benchmarks/receipts_raft_audit/arm{ARM}_r{STAGE}_labels.json, commit + push, y recién ahí la etapa {STAGE + 1}")

rr.save_pool(pool, state / "pool.json")
model.save_pretrained(str(state / "adapter"))
Path(state / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
"""

EXPORT = """
if STAGE == rr.NUM_ROUNDS:
    model.save_pretrained_gguf(EXPORT_DIR, tokenizer, quantization_method="q4_k_m")
    print(f"GGUF en {EXPORT_DIR}/ -- se evalúa con benchmarks/receipts_qwen_live_bench.py --set bench|ood --repeats 10 (spec §5)")
"""


def build() -> dict:
    src = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = src["cells"]
    install, gpu, model = (_find(cells, ANCHORS[k]) for k in ("install", "gpu", "model"))
    model_src = _src(model)
    for frozen in FROZEN_IN_MODEL_CELL:
        if frozen not in model_src:
            sys.exit(f"frozen setting {frozen!r} missing from the source model cell -- protocol drifted, refusing")
    if "model_name=MODEL_NAME" not in model_src:
        sys.exit("model cell no longer reads MODEL_NAME -- refusing")

    out_cells = [
        _md(INTRO),
        _md("## 1. Dependencias"), install,
        _md("### GPU"), gpu,
        _md("## 2. Configuración de la etapa"), _code(CONFIG),
        _md("## 3. Repo, gate de auditoría y pool previo"), _code(CLONE),
        _md("## 4. Modelo base + LoRA (congelado de RAFT v2)"), model, _code(MODEL_PREV),
        _md("## 5. Muestreo (chat, mismo template para muestrear y entrenar)"), _code(SAMPLER),
        _md("## 6. Entrenamiento sobre el pool acumulado (etapas 1-5)"), _code(TRAIN),
        _md("## 7. Cosecha de la ronda + hoja de auditoría (etapas 0-4)"), _code(HARVEST),
        _md("## 8. Export GGUF (etapa 5)"), _code(EXPORT),
    ]
    return {"cells": out_cells, "metadata": src.get("metadata", {}), "nbformat": src.get("nbformat", 4),
            "nbformat_minor": src.get("nbformat_minor", 5)}


def main() -> int:
    nb = build()
    OUTPUT_NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_NOTEBOOK} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
