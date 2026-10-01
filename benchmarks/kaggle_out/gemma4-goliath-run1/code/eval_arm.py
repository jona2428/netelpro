"""Mide un brazo (modelo base o base+LoRA): pass@k en Netelpro (train y OOD) y en Python (OOD).

Uso:  python eval_arm.py --name base
      python eval_arm.py --name raft_verified --adapter out/lora_verified
      python eval_arm.py --name mock --mock        # prueba de cañería sin GPU
"""
import argparse, json, os, random
from config import *
from common import *


def make_generate(adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    # bf16 solo en GPU Ampere+ (T4/P100 no lo soportan bien): si no, fp16
    default = "bfloat16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "float16"
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=getattr(torch, os.environ.get("DTYPE", default)), device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def gen_many(prompt, n, seed=SEED):
        """n muestras del mismo prompt en un solo generate (batch): mucho más rápido que n llamadas."""
        torch.manual_seed(seed)
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt", return_dict=True).to(model.device)
        res = []
        for c in range(0, n, 8):     # de a 8 para no reventar la VRAM
            out = model.generate(**ids, do_sample=True, temperature=TEMP, max_new_tokens=MAX_NEW_TOKENS,
                                 num_return_sequences=min(8, n - c))
            res += [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]
        return res

    def gen(prompt, seed):
        return gen_many(prompt, 1, seed)[0]
    gen.many = gen_many
    return gen


def mock_generate(prompt, seed):
    return "```netelpro\n(defn f (x) x)\n```\n```python\ndef f(x):\n    return x\n```"


def pass_at_k(gen, ids, prompt_fn, verify_fn, k):
    per_task, lens = {}, []
    for i, tid in enumerate(ids):
        task = load_task(tid)
        prompt = prompt_fn(task)
        outs = gen.many(prompt, k) if hasattr(gen, "many") else [gen(prompt, seed=SEED + i) for i in range(k)]
        per_task[tid] = sum(bool(verify_fn(o, task)) for o in outs)
        lens += [len(o) for o in outs]
        print(f"  [{i + 1}/{len(ids)}] {tid}: {per_task[tid]}/{k}", flush=True)
    return per_task, sum(lens) / max(len(lens), 1)


def v_sl(out, task):
    src = extract_src(out)
    return src is not None and verify_program(src, task).passed


def v_py(out, task):
    src = extract_py(out)
    return src is not None and verify_python(src, task)


def summarize(res, k):
    per_task, mean_chars = res
    n = len(per_task)
    return {"tasks": n, "mean_chars": round(mean_chars), "pass@k": sum(v > 0 for v in per_task.values()) / n,
            "pass@1_mean": sum(per_task.values()) / (n * k), "per_task": per_task}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--only", default="sl_train,sl_ood,py_ood", help="secciones a medir, separadas por coma")
    ap.add_argument("--train_tasks", default="", help="ids de tareas train separados por coma (anula la lista completa)")
    a = ap.parse_args()
    gen = mock_generate if a.mock else make_generate(a.adapter)
    ids_train, ids_ood = (TRAIN_IDS[:2], OOD_IDS[:2]) if a.mock else (TRAIN_IDS, OOD_IDS)
    if a.train_tasks:
        ids_train = a.train_tasks.split(",")
    sections = {
        "sl_train": (ids_train, build_prompt_sl, v_sl),
        "sl_ood":   (ids_ood,   build_prompt_sl, v_sl),
        "py_ood":   (ids_ood,   build_prompt_py, v_py),
    }
    res = {key: summarize(pass_at_k(gen, *sections[key][:1], *sections[key][1:], a.k), a.k)
           for key in a.only.split(",") if key in sections and sections[key][0]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"eval_{a.name}.json").write_text(json.dumps(res, indent=1))
    for key, v in res.items():
        print(f"{a.name:16s} {key:9s} pass@{a.k}={v['pass@k']:.0%}  pass@1={v['pass@1_mean']:.1%}  ({v['tasks']} tareas)")
