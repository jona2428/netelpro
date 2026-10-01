"""Smoke test (NO entrena): ¿carga Gemma 4 E2B, cabe en memoria, genera y el verificador lo puntúa?

Mide velocidad de generación, VRAM y pass@k sobre unas pocas tareas. Todo el reporte va a out/smoke.json.
"""
import argparse, json, time
from config import *
from common import *
import eval_arm

ap = argparse.ArgumentParser()
ap.add_argument("--mock", action="store_true")
ap.add_argument("--n_tasks", type=int, default=6)
ap.add_argument("--k", type=int, default=4)
a = ap.parse_args()

rep = {"model": HF_MODEL}
t0 = time.time()
gen = eval_arm.mock_generate if a.mock else eval_arm.make_generate(None)
rep["load_s"] = round(time.time() - t0, 1)
if not a.mock:
    import torch
    rep["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    rep["vram_gb_after_load"] = [round(torch.cuda.memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())]

task = load_task(OOD_IDS[0])
for name, prompt_fn in (("sl", build_prompt_sl), ("py", build_prompt_py)):
    t = time.time(); out = gen(prompt_fn(task), seed=0); dt = time.time() - t
    rep[f"sample_{name}"] = {"task": OOD_IDS[0], "seconds": round(dt, 1), "chars": len(out), "output": out[:1200]}
    print(f"--- muestra {name} ({dt:.1f}s) ---\n{out[:800]}\n", flush=True)

ids = OOD_IDS[: a.n_tasks]
t = time.time()
rep["sl_ood"] = eval_arm.summarize(eval_arm.pass_at_k(gen, ids, build_prompt_sl, eval_arm.v_sl, a.k), a.k)
rep["py_ood"] = eval_arm.summarize(eval_arm.pass_at_k(gen, ids, build_prompt_py, eval_arm.v_py, a.k), a.k)
rep["eval_s"] = round(time.time() - t, 1)
rep["n_generations"] = 2 * len(ids) * a.k
rep["s_per_generation"] = round(rep["eval_s"] / rep["n_generations"], 2)
if not a.mock:
    rep["vram_gb_peak"] = [round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "smoke.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in rep.items() if not k.startswith("sample_") and k not in ("sl_ood", "py_ood")}, indent=1))
for k_ in ("sl_ood", "py_ood"):
    print(k_, "pass@k", f"{rep[k_]['pass@k']:.0%}", "pass@1", f"{rep[k_]['pass@1_mean']:.0%}", rep[k_]["per_task"])
