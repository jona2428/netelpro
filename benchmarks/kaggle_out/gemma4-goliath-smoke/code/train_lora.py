"""SFT LoRA mínimo sobre out/<dataset>.jsonl. SIN PROBAR en GPU: primer paso del plan es un smoke test.

  python train_lora.py --data out/verified.jsonl --out out/lora_verified
"""
import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config import HF_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=2e-4)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=getattr(torch, os.environ.get("DTYPE", "bfloat16")), device_map="auto")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
model.train()
for ep in range(a.epochs):
    tot = 0.0
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False)
        p_ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids
        r_ids = tok(r["response"] + tok.eos_token, add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([p_ids, r_ids], 1).to(model.device)
        labels = ids.clone(); labels[:, : p_ids.shape[1]] = -100      # pérdida solo sobre la respuesta
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad(); tot += loss.item()
    print(f"epoch {ep + 1}: loss {tot / len(rows):.4f}", flush=True)
model.save_pretrained(a.out)
