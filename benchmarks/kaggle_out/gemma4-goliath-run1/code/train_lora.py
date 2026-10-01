"""SFT LoRA sobre out/<dataset>.jsonl (pérdida solo sobre la respuesta).

  python train_lora.py --data out/verified.jsonl --out out/lora_verified [--r 16 --epochs 3 --lr 2e-4]
  --max_steps N  : corta tras N pasos de optimizador (mini-prueba)
  --mock         : no carga el modelo; escribe un adaptador falso (prueba de cañería)
"""
import argparse, json, math, os, random, re, sys
from pathlib import Path
from config import HF_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--r", type=int, default=16); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--max_steps", type=int, default=0); ap.add_argument("--mock", action="store_true")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
print(f"datos: {len(rows)} ejemplos de {a.data}", flush=True)
if a.mock:
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "MOCK").write_text("mock"); sys.exit(0)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

fp16 = not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
dtype = getattr(torch, os.environ.get("DTYPE", "float16" if fp16 else "bfloat16"))
fp16 = dtype == torch.float16
tok = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=dtype, device_map="auto")
model.config.use_cache = False

# LoRA solo en las proyecciones del modelo de lenguaje (sin torres de visión/audio)
names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
         and re.search(r"(q|k|v|o|gate|up|down)_proj$", n) and not re.search(r"vision|audio|image|video|mm_", n)]
print(f"LoRA sobre {len(names)} módulos, ej: {names[:3]}", flush=True)
assert names, "no encontré proyecciones lineales para LoRA"
if not os.environ.get("NO_GC"):
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=names))
model.print_trainable_parameters()


def encode(r):
    user = [{"role": "user", "content": r["prompt"]}]
    p = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    try:  # la plantilla pone bien los tokens de cierre de turno
        full = tok.apply_chat_template(user + [{"role": "assistant", "content": r["response"]}], tokenize=False, enable_thinking=False)
        assert full.startswith(p)
        tail = full[len(p):]
    except Exception:
        tail = r["response"] + tok.eos_token
    p_ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids
    t_ids = tok(tail, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([p_ids, t_ids], 1)
    labels = ids.clone(); labels[:, : p_ids.shape[1]] = -100
    return ids, labels


data = [encode(r) for r in rows]
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", enabled=fp16 and torch.cuda.is_available())
total_steps = max(1, math.ceil(len(data) * a.epochs / a.accum))
if a.max_steps: total_steps = min(total_steps, a.max_steps)
step = micro = 0
rng = random.Random(0)
model.train()
for ep in range(a.epochs):
    order = list(range(len(data))); rng.shuffle(order); tot = 0.0
    for i in order:
        ids, labels = (t.to(next(model.parameters()).device) for t in data[i])
        loss = model(input_ids=ids, labels=labels).loss
        if not torch.isfinite(loss):
            print("PÉRDIDA NO FINITA: abortando", flush=True); sys.exit(3)
        scaler.scale(loss / a.accum).backward(); tot += loss.item(); micro += 1
        if micro % a.accum == 0:
            for g in opt.param_groups: g["lr"] = a.lr * max(0.05, 1 - step / total_steps)
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); step += 1
            if step % 5 == 0 or a.max_steps: print(f"  paso {step}/{total_steps} loss {loss.item():.4f}", flush=True)
            if a.max_steps and step >= a.max_steps: break
    print(f"epoch {ep + 1}: loss medio {tot / len(order):.4f}", flush=True)
    if a.max_steps and step >= a.max_steps: break
model.save_pretrained(a.out)
print("adaptador guardado en", a.out, flush=True)
