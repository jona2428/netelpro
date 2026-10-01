"""Pipeline completo del experimento. Cada etapa es un subproceso (libera la GPU) y deja constancia en out/status.json.

  A  eval base  ->  datos (verified/unverified)  ->  mini-prueba de entrenamiento
  ->  C: entrenar verificado + PUERTA de memorización (solo tareas train; escala hiperparámetros si no pasa)
  ->  eval C  ->  B: entrenar sin filtrar con la MISMA config  ->  eval B  ->  report.md

Regla de honestidad: los hiperparámetros se ajustan solo mirando la puerta (tareas de ENTRENAMIENTO). Las OOD no se miran hasta el final.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
from config import OUT

HERE = Path(__file__).parent
PREV = HERE / "prev"     # resultados de la corrida anterior (brazo A y 17 ejemplos verificados)

MOCK = "--mock" in sys.argv
MIN_VERIFIED = int(os.environ.get("MIN_VERIFIED", "0" if MOCK else "12"))
GATE = 0.70
LEVELS = [dict(r=16, epochs=3, lr=2e-4), dict(r=64, epochs=8, lr=3e-4)]   # escalera de hiperparámetros
m = ["--mock"] if MOCK else []
OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT / "status.json"
status = {"stages": {}}


def save(): STATUS.write_text(json.dumps(status, indent=1, ensure_ascii=False))


def run(name, args, fatal=False):
    t = time.time(); print(f"\n===== {name}: {' '.join(args)}", flush=True)
    p = subprocess.Popen([sys.executable] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        print(line, end="", flush=True); tail.append(line); tail = tail[-30:]
    rc = p.wait()
    status["stages"][name] = {"rc": rc, "seconds": round(time.time() - t), "tail": "".join(tail)[-1500:] if rc else ""}
    save(); print(f"===== {name}: rc={rc} en {time.time() - t:.0f}s", flush=True)
    if rc and fatal:
        status["aborted_at"] = name; save(); sys.exit(rc)
    return rc


def load(name):
    f = OUT / f"eval_{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


# 0. mini-prueba de entrenamiento PRIMERO (falla rápido si el entorno o la memoria no dan) con datos reales de la corrida anterior
if (PREV / "verified.jsonl").exists():
    rows0 = open(PREV / "verified.jsonl", encoding="utf-8").read().splitlines()[:3]
    (OUT / "mini.jsonl").write_text("\n".join(rows0) + "\n", encoding="utf-8")
    run("train_test", ["train_lora.py", "--data", str(OUT / "mini.jsonl"), "--out", str(OUT / "lora_test"), "--max_steps", "3"] + m, fatal=True)
# 1. brazo A (se reutiliza si ya existe: mismo código, prompt y configuración)
if (PREV / "eval_base.json").exists() and not (OUT / "eval_base.json").exists():
    shutil.copy(PREV / "eval_base.json", OUT / "eval_base.json"); status["stages"]["A_eval_base"] = {"rc": 0, "reused": True}; save()
else:
    run("A_eval_base", ["eval_arm.py", "--name", "base"] + m, fatal=True)
# 2. datos: ronda 2 (48 muestras/tarea, otra semilla) fusionada con la ronda 1
run("datos", ["raft_data.py", "--n", "48", "--seed", "2000", "--prev", str(PREV / "verified.jsonl")] + m, fatal=True)
verified = [json.loads(l) for l in open(OUT / "verified.jsonl", encoding="utf-8")]
tasks_v = sorted({r["task"] for r in verified})
status["verified"] = {"ejemplos": len(verified), "tareas": len(tasks_v)}; save()
if len(verified) < MIN_VERIFIED:
    status["aborted_at"] = f"datos insuficientes: {len(verified)} verificados < {MIN_VERIFIED}"; save()
    print(status["aborted_at"]); sys.exit(2)
# 4. C con puerta
chosen = None
for i, lv in enumerate(LEVELS):
    tag = f"L{i + 1}"
    run(f"C_train_{tag}", ["train_lora.py", "--data", str(OUT / "verified.jsonl"), "--out", str(OUT / "lora_verified"),
                           "--r", str(lv["r"]), "--epochs", str(lv["epochs"]), "--lr", str(lv["lr"])] + m)
    run(f"C_gate_{tag}", ["eval_arm.py", "--name", f"gate_{tag}", "--adapter", str(OUT / "lora_verified"), "--only", "sl_train",
                          "--train_tasks", ",".join(tasks_v)] + m)
    g = load(f"gate_{tag}")
    rate = g["sl_train"]["pass@k"] if g else None
    status["gate_" + tag] = {"config": lv, "pass@8_en_tareas_entrenadas": rate}; save()
    if rate is None or rate >= GATE or not tasks_v:
        chosen = lv; status["gate_passed"] = rate is not None and (rate >= GATE or not tasks_v); break
    chosen = lv; status["gate_passed"] = False
save()
# 5. eval C completo
run("C_eval", ["eval_arm.py", "--name", "raft_verified", "--adapter", str(OUT / "lora_verified")] + m)
# 6. B con la misma config
run("B_train", ["train_lora.py", "--data", str(OUT / "unverified.jsonl"), "--out", str(OUT / "lora_unverified"),
                "--r", str(chosen["r"]), "--epochs", str(chosen["epochs"]), "--lr", str(chosen["lr"])] + m)
run("B_eval", ["eval_arm.py", "--name", "raft_unverified", "--adapter", str(OUT / "lora_unverified")] + m)
# 7. reporte
run("report", ["report.py"])
status["done"] = True; save()
print("PIPELINE TERMINADO")
