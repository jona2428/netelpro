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
        out = model.generate(**ids, do_sample=True, temperature=TEMP, max_new_tokens=MAX_NEW_TOKENS,
                             num_return_sequences=n)
        return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

    def gen(prompt, seed):
        return gen_many(prompt, 1, seed)[0]
    gen.many = gen_many
    return gen


def mock_generate(prompt, seed):
    return "```netelpro\n(defn f (x) x)\n```\n```python\ndef f(x):\n    return x\n```"


def pass_at_k(gen, ids, prompt_fn, verify_fn, k):
    per_task = {}
    for tid in ids:
        task = load_task(tid)
        prompt = prompt_fn(task)
        outs = gen.many(prompt, k) if hasattr(gen, "many") else [gen(prompt, seed=SEED + i) for i in range(k)]
        per_task[tid] = sum(bool(verify_fn(o, task)) for o in outs)
    return per_task


def v_sl(out, task):
    src = extract_src(out)
    return src is not None and verify_program(src, task).passed


def v_py(out, task):
    src = extract_py(out)
    return src is not None and verify_python(src, task)


def summarize(per_task, k):
    n = len(per_task)
    return {"tasks": n, "pass@k": sum(v > 0 for v in per_task.values()) / n,
            "pass@1_mean": sum(per_task.values()) / (n * k), "per_task": per_task}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    gen = mock_generate if a.mock else make_generate(a.adapter)
    ids_train, ids_ood = (TRAIN_IDS[:2], OOD_IDS[:2]) if a.mock else (TRAIN_IDS, OOD_IDS)
    res = {
        "sl_train": summarize(pass_at_k(gen, ids_train, build_prompt_sl, v_sl, a.k), a.k),
        "sl_ood":   summarize(pass_at_k(gen, ids_ood,   build_prompt_sl, v_sl, a.k), a.k),
        "py_ood":   summarize(pass_at_k(gen, ids_ood,   build_prompt_py, v_py, a.k), a.k),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"eval_{a.name}.json").write_text(json.dumps(res, indent=1))
    for key, v in res.items():
        print(f"{a.name:16s} {key:9s} pass@{a.k}={v['pass@k']:.0%}  pass@1={v['pass@1_mean']:.1%}  ({v['tasks']} tareas)")
