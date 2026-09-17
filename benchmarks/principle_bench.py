#!/usr/bin/env python3
"""
Principle Benchmark: netelpro vs JSON (tokens + syntax error rate).
==============================================================

Empirically measures the two core design principles of Netelpro
(docs/WHITEPAPER.md, section 2):

Principle #2 -- token economy. Same decision rule, two representations:
the netelpro truth-table contract vs a semantically equivalent JSON.
Token counts come from the REAL tokenizers of the models (Ollama
prompt_eval_count, raw:true, num_predict=1 -- pure tokenizer, no
template overhead, no sampling).

Principle #1 -- "an LLM can verify its own syntax by counting, not
simulating." Measured as: rate of machine-valid generations of a
decision rule in netelpro (fiscal grammar, judged by the REAL compiler,
fail-closed) vs the same rule in JSON (judged by an equally strict
validator: same fail-closed philosophy, any structural deviation or
semantic mismatch is a rejection).

Experimental design (part B):
- 12 fresh tasks: 4th name-pool per family -- names NEVER used in
  dataset_v2 (which only uses pools 0-2) -- x 3 strata of complexity:
  1 row ON (hardest), 4 rows ON (mid), 7 rows ON (easiest).
- 2 formats (netelpro / JSON), same semantic content, analogous format
  specs with full worked examples in the prompt.
- 3 models: OLMoE base vs goliath-killer (paired: same tokenizer, one
  trained on dataset_v2, one not) + qwen2.5:1.5b base (2nd family).
- Temperature 0.8 (house protocol), deterministic seeds per
  (task, format, sample), generation through the model template
  (same protocol as rlvr/gguf_eval.py).
- Fail-closed: no extractable block = rejection. The compiler decides.

Checkpoint: results are saved to principle_bench_results.json after
every cell and the run is resumable (--resume is the default).

Usage:
    python benchmarks/principle_bench.py --tokens-only
    python benchmarks/principle_bench.py --syntax-only
    python benchmarks/principle_bench.py --full
    python benchmarks/principle_bench.py --reload   (report from saved json)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = REPO_ROOT
DATASET_V2 = REPO_ROOT / "training" / "data" / "dataset_v2.jsonl"
RESULTS_JSON = REPO_ROOT / "benchmarks" / "principle_bench_results.json"
RESULTS_MD = REPO_ROOT / "benchmarks" / "principle_bench_summary.md"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODELS: Dict[str, str] = {
    "olmoe-base": "hf.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF:olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
    "olmoe-gk": "goliath-killer:latest",
    "qwen-base": "qwen2.5:1.5b",
}
TEMP = 0.8
NUM_PREDICT_NP = 350
NUM_PREDICT_JS = 500
N_SAMPLES = 5
TOKEN_SAMPLE_LIMIT = 40
TOKEN_SAMPLE_STEP = 125  # 5000/125 = 40 examples spread across families

if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from netelpro.rule_filter import RuleFilterError, compile_filter  # noqa: E402

# ---------------------------------------------------------------------------
# 12 fresh tasks: 4th name-pool per family (never in dataset_v2 pools 0-2)
# ---------------------------------------------------------------------------

FRESH_POOL_4: Dict[str, List[str]] = {
    "continuity_backpressure": ["feed-ok", "exit-ok", "gate-open"],
    "apoptosis_circuit_breaker": ["mark-ok", "wall-done", "team-ok"],
    "annealing_scheduler": ["chill-ok", "cap-ok", "grab-ok"],
    "hebb_moe_router": ["pair-live", "weight-ok", "span-ok"],
    "coagulation_2pc": ["ask-ok", "lock-ok", "split-clear"],
    "mycelium_gossip": ["seep-ok", "lane-ok", "net-live"],
    "chatelier_ratelimit": ["force-ok", "room-ok", "steer-ok"],
    "commons_uma_arbiter": ["rent-paid", "slack-ok", "slice-ok"],
    "pheromone_hnsw": ["mark-live", "grade-ok", "drop-ok"],
    "lymph_gc": ["wash-ok", "vein-ok", "maid-live"],
    "mirage_decoder": ["truth-ok", "scope-ok", "say-ok"],
    "lagrange_loadbalancer": ["north-ok", "south-ok", "core-ok"],
}

FAMILY_META: Dict[str, Tuple[str, str, str, str]] = {
    "continuity_backpressure": (
        "Dinamica de Fluidos (Hidraulica)", "Colas de Mensajeria y Streaming (Kafka/RabbitMQ)",
        "Ecuacion de Continuidad y Prevencion de Cavitacion vs Backpressure",
        "sistema de absorcion de rafagas en streams distribuidos con valvulas de alivio y flujo laminar"),
    "apoptosis_circuit_breaker": (
        "Biologia Celular (Apoptosis)", "Arquitectura de Microservicios y Circuit Breakers",
        "Aislamiento programado y muerte celular controlada para preservar el organismo",
        "orquestador que destruye instancias degradadas sin afectar el quorum (cascada de caspasas)"),
    "annealing_scheduler": (
        "Termodinamica Clasica y Metalurgia", "Optimizacion Heuristica de Rutas y Recursos",
        "Enfriamiento lento (Simulated Annealing) y minimizacion de entropia libre",
        "balanceo de tareas en cluster UMA con descenso de temperatura de Boltzmann para escapar de minimos locales"),
    "hebb_moe_router": (
        "Neurobiologia (Plasticidad Sinaptica y Poda)", "Enrutamiento en Redes Sparse MoE",
        "Regla de Hebb y poda de conexiones subutilizadas",
        "router de MoE de 64 expertos que desactiva expertos redundantes sin perder cobertura semantica"),
    "coagulation_2pc": (
        "Hematologia (Cascada de Coagulacion)", "Consenso Distribuido (Two-Phase Commit)",
        "Amplificacion enzimatica irreversible y sellado inmediato de brechas",
        "commit atomico distribuido inspirado en polimerizacion de fibrina, anti-deadlock ante particiones"),
    "mycelium_gossip": (
        "Micologia (Redes de Micelio)", "Enrutamiento Descentralizado (P2P Mesh Gossip)",
        "Transporte de nutrientes por gradiente osmotico sin nodo central",
        "protocolo gossip edge de baja latencia que redistribuye memoria y ancho de banda"),
    "chatelier_ratelimit": (
        "Quimica Fisica (Principio de Le Chatelier)", "Autoscaling Reactivo y Rate Limiting",
        "Desplazamiento del equilibrio termodinamico para contrarrestar la fuerza impuesta",
        "middleware HTTP/2 que ajusta cuotas segun la constante de equilibrio Keq"),
    "commons_uma_arbiter": (
        "Teoria de Juegos (Tragedia de los Comunes)", "Gestion de Memoria Unificada (UMA)",
        "Egoismo de procesos locales causando colapso del recurso compartido",
        "arbitro de RAM/VRAM con impuestos pigouvianos a procesos que retienen buffers innecesarios"),
    "pheromone_hnsw": (
        "Entomologia (Senderos de Feromonas)", "Optimizacion de Consultas en Bases Vectoriales (HNSW)",
        "Evaporacion temporal y refuerzo por retroalimentacion positiva de trayectos optimos",
        "busqueda HNSW con decaimiento de feromonas para podar rutas de similitud coseno suboptimas"),
    "lymph_gc": (
        "Fisiologia Humana (Sistema Linfatico)", "Recoleccion de Basura Asincrona (GC)",
        "Drenaje continuo de desechos intersticiales en paralelo sin pausar el flujo arterial",
        "recolector de memoria en segundo plano para caches sin Stop-The-World"),
    "mirage_decoder": (
        "Optica y Meteorologia (Espejismos y Refraccion)", "Mitigacion de Alucinaciones y Calibracion de Confianza en LLMs",
        "Distorsion de la senal por gradientes termicos en medios heterogeneos",
        "decodificador con verificacion epistemica que detecta refraccion de datos y fuerza rechazo de premisas falaces"),
    "lagrange_loadbalancer": (
        "Mecanica Celeste (Puntos de Lagrange)", "Topologia de Balanceadores de Carga Multi-Region",
        "Equilibrio gravitatorio orbital donde fuerzas centrifugas y de atraccion se anulan",
        "replicas de solo lectura multi-region con latencia neta en equilibrio estacionario"),
}

# Strata: rotating by family index -> 4 families x (1 ON, 4 ON, 7 ON).
STRATUM_ON_ROWS: Dict[str, List[int]] = {}
for _i, _fam in enumerate(sorted(FRESH_POOL_4)):
    if _i % 3 == 0:
        STRATUM_ON_ROWS[_fam] = [7]           # hardest: only (1,1,1)
    elif _i % 3 == 1:
        STRATUM_ON_ROWS[_fam] = [4, 5, 6, 7]  # mid: first signal = 1
    else:
        STRATUM_ON_ROWS[_fam] = [1, 2, 3, 4, 5, 6, 7]  # easiest: all but (0,0,0)


def bits_for_row(row: int) -> Tuple[int, int, int]:
    return ((row >> 2) & 1, (row >> 1) & 1, row & 1)


def verdict_for_row(table_id: int, row: int) -> int:
    return (table_id >> (7 - row)) & 1


def table_id_from_on_rows(on_rows: List[int]) -> int:
    t = 0
    for row in on_rows:
        t |= 1 << (7 - row)
    return t


def on_set_desc(table_id: int) -> str:
    on = [f"({a},{b},{c})" for row in range(8)
          if verdict_for_row(table_id, row)
          for a, b, c in [bits_for_row(row)]]
    if not on:
        return "ninguna combinacion activa el paso (fail-closed total)"
    if len(on) == 8:
        return "toda combinacion activa el paso"
    if len(on) == 1:
        return f"el paso se activa unicamente para {on[0]}"
    if len(on) == 7:
        off = [f"({a},{b},{c})" for row in range(8)
               if not verdict_for_row(table_id, row)
               for a, b, c in [bits_for_row(row)]]
        return f"el paso se activa para todo salvo {off[0]}"
    return "el paso se activa exactamente para: " + ", ".join(on)


def task_list() -> List[Dict[str, Any]]:
    tasks = []
    for fam in sorted(FRESH_POOL_4):
        params = FRESH_POOL_4[fam]
        table_id = table_id_from_on_rows(STRATUM_ON_ROWS[fam])
        dom_a, dom_b, law, scenarios = FAMILY_META[fam]
        tasks.append({
            "task_id": f"{fam}#pool4",
            "family": fam,
            "params": params,
            "table_id": table_id,
            "domain_a": dom_a, "domain_b": dom_b,
            "law": law, "scenarios": scenarios,
            "stratum": f"{len(STRATUM_ON_ROWS[fam])}_on",
        })
    return tasks


# ---------------------------------------------------------------------------
# Ollama transport (stdlib only, house protocol of rlvr/gguf_eval.py)
# ---------------------------------------------------------------------------

def _post(payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def token_count(model: str, text: str) -> int:
    """Pure tokenizer count: raw (no template), num_predict=1, temp 0."""
    payload = {"model": model, "prompt": text, "stream": False, "raw": True,
               "options": {"num_predict": 1, "seed": 0, "temperature": 0.0}}
    data = _post(payload, 300)
    n = data.get("prompt_eval_count")
    if n is None:
        raise RuntimeError(f"prompt_eval_count ausente para {model}")
    return int(n)


def ollama_generate(model: str, prompt: str, seed: int, num_predict: int) -> str:
    """One sample via /api/generate (template applied, house protocol)."""
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": TEMP, "num_predict": num_predict,
                           "seed": seed}}
    data = _post(payload, 900)
    return data["response"]


def probe_model(model: str) -> bool:
    try:
        token_count(model, "probe")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Artifacts: netelpro contract / cases / equivalent JSON
# ---------------------------------------------------------------------------

def netelpro_contract(params: List[str], table_id: int) -> str:
    decls = "\n".join(f"  ({p} : (Int 0 1))" for p in params)
    rows = "\n".join(
        f"  (({a} {b} {c}) -> {verdict_for_row(table_id, row)})"
        for row in range(8) for a, b, c in [bits_for_row(row)])
    return f"(truth-table filter-rule\n{decls}\n{rows}\n  ((_ _ _) -> 0))"


def netelpro_cases_block(params: List[str], table_id: int) -> str:
    return "\n".join(
        f"({a},{b},{c}) -> {verdict_for_row(table_id, row)}"
        for row in range(8) for a, b, c in [bits_for_row(row)])


def json_rule(params: List[str], table_id: int, indent: bool = True) -> str:
    rows = [{"signals": list(bits_for_row(row)), "step": verdict_for_row(table_id, row)}
            for row in range(8)]
    obj = {"rule_name": "filter-rule", "signals": params,
           "logic": "decision_table", "rows": rows, "default": 0}
    if indent:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def true_cases(table_id: int) -> List[Tuple[Tuple[int, int, int], int]]:
    return [(bits_for_row(row), verdict_for_row(table_id, row)) for row in range(8)]


# ---------------------------------------------------------------------------
# Part B: prompts
# ---------------------------------------------------------------------------

NETELPRO_SPEC = """Formaliza esta regla en el lenguaje netelpro como contrato truth-table.

Formato del contrato (senales ficticias x, y, z; el paso se activa solo para (1,0,1)):

```netelpro
(truth-table filter-rule
  (x : (Int 0 1))
  (y : (Int 0 1))
  (z : (Int 0 1))
  ((0 0 0) -> 0)
  ((0 0 1) -> 0)
  ((0 1 0) -> 0)
  ((0 1 1) -> 0)
  ((1 0 0) -> 0)
  ((1 0 1) -> 1)
  ((1 1 0) -> 0)
  ((1 1 1) -> 0)
  ((_ _ _) -> 0))
```

Seguido de un bloque de casos de verificacion (una linea por combinacion, en el mismo orden):

```netelpro-cases
(0,0,0) -> 0
(0,0,1) -> 0
(0,1,0) -> 0
(0,1,1) -> 0
(1,0,0) -> 0
(1,0,1) -> 1
(1,1,0) -> 0
(1,1,1) -> 0
```

Reglas del formato: cada senal se declara como (nombre : (Int 0 1)); las 8 combinaciones van explicitas y en orden binario ((0 0 0) primero, (1 1 1) ultimo); la fila default ((_ _ _) -> 0) cierra el contrato; el veredicto 1 activa el paso, 0 no lo activa.

Responde unicamente con el bloque ```netelpro``` (contrato completo) seguido del bloque ```netelpro-cases``` (8 lineas). Sin texto adicional."""

JSON_SPEC = """Formaliza esta regla como JSON con estructura de tabla de decision.

Formato (senales ficticias x, y, z; el paso se activa solo para (1,0,1)):

```json
{
  "rule_name": "filter-rule",
  "signals": ["x", "y", "z"],
  "logic": "decision_table",
  "rows": [
    {"signals": [0, 0, 0], "step": 0},
    {"signals": [0, 0, 1], "step": 0},
    {"signals": [0, 1, 0], "step": 0},
    {"signals": [0, 1, 1], "step": 0},
    {"signals": [1, 0, 0], "step": 0},
    {"signals": [1, 0, 1], "step": 1},
    {"signals": [1, 1, 0], "step": 0},
    {"signals": [1, 1, 1], "step": 0}
  ],
  "default": 0
}
```

Reglas del formato: "signals" lista las tres senales en orden; cada fila de "rows" declara la combinacion (en ese orden) y su veredicto "step" (1 activa el paso, 0 no lo activa); las 8 combinaciones van explicitas y en orden binario; "default" es 0.

Responde unicamente con un bloque ```json``` con la regla completa. Sin texto adicional."""


def build_prompt(task: Dict[str, Any], fmt: str) -> str:
    p = task["params"]
    common = (
        f"Diseña una regla de decision verificable para: {task['scenarios']}. "
        f"La ley invariante compartida entre {task['domain_a']} y {task['domain_b']} es: "
        f"{task['law']}. Las senales de decision son: {p[0]}, {p[1]}, {p[2]} "
        f"(cada una vale 0 o 1). {on_set_desc(task['table_id']).capitalize()}. "
        f"Ante cualquier otra combinacion, el paso NO se activa (fail-closed: "
        f"el sistema permanece en el estado seguro).\n\n")
    if fmt == "netelpro":
        return common + NETELPRO_SPEC
    return common + JSON_SPEC


# ---------------------------------------------------------------------------
# Part B: grading (fail-closed, compiler decides)
# ---------------------------------------------------------------------------

FENCE_NP = re.compile(r"```netelpro\s*\n(.*?)```", re.DOTALL)
FENCE_CASES = re.compile(r"```netelpro-cases\s*\n(.*?)```", re.DOTALL)
FENCE_JSON = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
CASE_LINE = re.compile(r"^\s*\(\s*([01][01,\s]*)\s*\)\s*->\s*([01]|true|false)\s*$")

# Opening fences must be matched SPECIFICALLY, never by substring: "```netelpro"
# is a literal prefix of "```netelpro-cases", so a bare `"```netelpro" in text`
# test cannot distinguish "the model opened the contract fence" from "the model
# opened the cases fence". Measured 2026-09-17 over the 2026-09-14 run artifact:
# 128 of the 137 netelpro `truncado` labels carried detail "fence cases sin
# cerrar", a message that contradicts itself -- it is only reachable AFTER the
# contract fence closed, so no fence was left unterminated. The label came from
# the substring test, and it mislabelled "contract present, cases missing" (the
# eval's largest error class, `sin_casos`) as truncation. The `(?![-\w])`
# lookahead is what keeps the contract fence from matching the cases fence.
FENCE_NP_OPEN = re.compile(r"```netelpro(?![-\w])")
FENCE_CASES_OPEN = re.compile(r"```netelpro-cases")


def grade_netelpro(response: str, params: List[str], table_id: int) -> Tuple[bool, str, str]:
    if not FENCE_NP_OPEN.search(response):
        return False, "sin_bloque", ""
    m = FENCE_NP.search(response)
    if not m:
        return False, "truncado", "fence netelpro sin cerrar"
    contract = m.group(1).strip()
    mc = FENCE_CASES.search(response)
    if not mc:
        # `truncado` means the cases fence was OPENED and never closed. A response
        # that merely names the fence in prose, or omits it, is `sin_casos` -- the
        # same class the run #5 exit gate is held to (see benchmarks/run5_gate.py).
        if FENCE_CASES_OPEN.search(response):
            return False, "truncado", "fence cases sin cerrar"
        return False, "sin_casos", ""
    model_cases: List[Tuple[Tuple[int, ...], int]] = []
    for line in mc.group(1).strip().splitlines():
        mm = CASE_LINE.match(line)
        if mm:
            args = tuple(int(t) for t in re.split(r"[,\s]+", mm.group(1).strip()) if t)
            exp = 1 if mm.group(2) in ("1", "true") else 0
            model_cases.append((args, exp))
    if not model_cases:
        return False, "sin_casos", "bloque cases vacio"
    try:
        rf = compile_filter(contract)
    except RuleFilterError as exc:
        return False, "compilacion", str(exc).splitlines()[0][:160]
    except Exception as exc:  # fail-closed: motor raro = rechazo
        return False, "compilacion", repr(exc)[:160]
    truth = true_cases(table_id)
    try:
        mism = rf.verify_int(truth)
    except Exception as exc:
        return False, "verificacion", repr(exc)[:160]
    if mism:
        return False, "semantica", f"{len(mism)} casos no calzan con la ley pedida"
    truth_map = {args: exp for args, exp in truth}
    wrong = sum(1 for a, e in model_cases if truth_map.get(a) != e)
    if wrong:
        return False, "casos_incorrectos", f"{wrong} casos declarados con veredicto equivocado"
    if len(model_cases) < 8:
        return False, "casos_incompletos", f"{len(model_cases)}/8 casos"
    return True, "ok", ""


def grade_json(response: str, params: List[str], table_id: int) -> Tuple[bool, str, str]:
    if "```json" not in response:
        return False, "sin_bloque", ""
    m = FENCE_JSON.search(response)
    if not m:
        return False, "truncado", "fence json sin cerrar"
    try:
        data = json.loads(m.group(1))
    except Exception as exc:
        return False, "parseo", str(exc)[:160]
    if not isinstance(data, dict):
        return False, "estructura", "la raiz no es un objeto"
    if data.get("rule_name") != "filter-rule":
        return False, "estructura", "rule_name != filter-rule"
    sigs = data.get("signals")
    if not isinstance(sigs, list) or [str(s) for s in sigs] != list(params):
        return False, "estructura", "signals no calzan con las pedidas"
    if data.get("logic") != "decision_table":
        return False, "estructura", "logic != decision_table"
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        n = len(rows) if isinstance(rows, list) else "?"
        return False, "estructura", f"rows != 8 (={n})"
    seen = set()
    for r in rows:
        if not isinstance(r, dict) or set(r.keys()) != {"signals", "step"}:
            return False, "estructura", "fila con claves incorrectas"
        s, st = r.get("signals"), r.get("step")
        if (not isinstance(s, list) or len(s) != 3
                or any(not isinstance(x, int) or x not in (0, 1) for x in s)):
            return False, "estructura", "fila signals invalida"
        if not isinstance(st, int) or st not in (0, 1):
            return False, "estructura", "fila step invalida"
        key = tuple(s)
        if key in seen:
            return False, "estructura", "fila duplicada"
        seen.add(key)
    if seen != {bits_for_row(row) for row in range(8)}:
        return False, "cobertura", "no cubre las 8 combinaciones exactamente"
    if data.get("default") != 0:
        return False, "estructura", "default != 0"
    wrong = 0
    for r in rows:
        s = r["signals"]
        row_idx = (s[0] << 2) | (s[1] << 1) | s[2]
        if r["step"] != verdict_for_row(table_id, row_idx):
            wrong += 1
    if wrong:
        return False, "semantica", f"{wrong} filas no calzan con la ley pedida"
    return True, "ok", ""


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_results() -> Dict[str, Any]:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {}, "part_a": None, "part_b": {"cells": {}}}


def save_results(results: Dict[str, Any]) -> None:
    tmp = RESULTS_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    tmp.replace(RESULTS_JSON)


def seed_for(task_id: str, fmt: str, k: int) -> int:
    return zlib.crc32(f"{task_id}|{fmt}|{k}".encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Part A: token economy
# ---------------------------------------------------------------------------

def run_tokens(models: Dict[str, str], limit: int, step: int,
               results: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 70)
    print("PRINCIPIO #2 -- ECONOMIA DE TOKENS: netelpro vs JSON (tokenizadores reales)")
    print("=" * 70)
    rows = []
    with open(DATASET_V2, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    sample = rows[::step][:limit]
    print(f"muestra: {len(sample)} ejemplos de dataset_v2 (step={step})")
    artifacts = ["netelpro_contract", "netelpro_contract_cases", "json_indent", "json_compact"]
    per_model: Dict[str, Dict[str, List[int]]] = {k: {a: [] for a in artifacts} for k in models}
    chars: Dict[str, List[int]] = {a: [] for a in artifacts}
    t0 = time.time()
    for i, ex in enumerate(sample):
        params, table_id = ex["params"], ex["table_id"]
        texts = {
            "netelpro_contract": ex["netelpro_contract"],
            "netelpro_contract_cases": (ex["netelpro_contract"] + "\n\n"
                                        + "\n".join(
                                            f"({a},{b},{c}) -> {e}"
                                            for a, b, c, e in ex["netelpro_cases"])),
            "json_indent": json_rule(params, table_id, indent=True),
            "json_compact": json_rule(params, table_id, indent=False),
        }
        for a in artifacts:
            chars[a].append(len(texts[a]))
        for label, model in models.items():
            for a in artifacts:
                per_model[label][a].append(token_count(model, texts[a]))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sample)} ejemplos ({time.time() - t0:.0f}s)")
    summary: Dict[str, Any] = {"n_examples": len(sample),
                               "chars_mean": {a: round(statistics.mean(v), 1) for a, v in chars.items()}}
    tok_stats: Dict[str, Any] = {}
    for label in models:
        tok_stats[label] = {}
        for a in artifacts:
            v = per_model[label][a]
            tok_stats[label][a] = {"mean": round(statistics.mean(v), 1),
                                   "median": statistics.median(v)}
        base = tok_stats[label]["json_indent"]["mean"]
        for a in artifacts:
            tok_stats[label][a]["ratio_vs_json_indent"] = round(
                tok_stats[label][a]["mean"] / base, 3)
    summary["tokens"] = tok_stats
    results["part_a"] = summary
    save_results(results)
    for label in models:
        s = tok_stats[label]
        print(f"\n[{label}]")
        print(f"  contrato netelpro          : {s['netelpro_contract']['mean']} tokens "
              f"(x{s['netelpro_contract']['ratio_vs_json_indent']} vs JSON)")
        print(f"  contrato+casos netelpro    : {s['netelpro_contract_cases']['mean']} tokens "
              f"(x{s['netelpro_contract_cases']['ratio_vs_json_indent']} vs JSON)")
        print(f"  JSON indentado             : {s['json_indent']['mean']} tokens")
        print(f"  JSON compacto              : {s['json_compact']['mean']} tokens "
              f"(x{s['json_compact']['ratio_vs_json_indent']} vs JSON indentado)")
    return summary


# ---------------------------------------------------------------------------
# Part B: syntax/semantic error rate
# ---------------------------------------------------------------------------

def run_syntax(models: Dict[str, str], n_samples: int,
               results: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 70)
    print("PRINCIPIO #1 -- ERROR SINTACTICO/SEMANTICO: el compilador decide (fail-closed)")
    print("=" * 70)
    tasks = task_list()
    cells = results["part_b"]["cells"]
    for label, model in models.items():
        if not probe_model(model):
            print(f"[{label}] MODELO NO DISPONIBLE ({model}) -- celda omitida")
            cells.setdefault(f"{label}|__probe__", {"error": "modelo_no_disponible"})
            save_results(results)
            continue
        for task in tasks:
            key = f"{label}|{task['task_id']}"
            existing = cells.get(key)
            if (isinstance(existing, dict) and not existing.get("error")
                    and all(len(existing.get(fmt, {}).get("samples", [])) >= n_samples
                            for fmt in ("netelpro", "json"))):
                print(f"  [skip] {key} ya completo")
                continue
            cell: Dict[str, Any] = {"family": task["family"],
                                    "stratum": task["stratum"],
                                    "table_id": task["table_id"]}
            for fmt in ("netelpro", "json"):
                prompt = build_prompt(task, fmt)
                num_predict = NUM_PREDICT_NP if fmt == "netelpro" else NUM_PREDICT_JS
                samples = []
                for k in range(n_samples):
                    seed = seed_for(task["task_id"], fmt, k)
                    t0 = time.time()
                    try:
                        resp = ollama_generate(model, prompt, seed, num_predict)
                        if fmt == "netelpro":
                            ok, err, detail = grade_netelpro(resp, task["params"], task["table_id"])
                        else:
                            ok, err, detail = grade_json(resp, task["params"], task["table_id"])
                        samples.append({"k": k, "seed": seed, "ok": ok, "err": err,
                                        "detail": detail,
                                        "secs": round(time.time() - t0, 1),
                                        "len": len(resp)})
                    except (urllib.error.URLError, TimeoutError, OSError) as exc:
                        samples.append({"k": k, "seed": seed, "ok": False,
                                        "err": "error_api", "detail": str(exc)[:160],
                                        "secs": round(time.time() - t0, 1), "len": 0})
                    status = "." if ok else ("X" if err in ("semantica", "casos_incorrectos") else "x")
                    print(status, end="", flush=True)
                valid = sum(1 for s in samples if s["ok"])
                cell[fmt] = {"valid": valid, "n": len(samples), "samples": samples}
                print(f"  {key} [{fmt}] {valid}/{len(samples)} "
                      f"({time.time() - t0:.0f}s)")
            cells[key] = cell
            save_results(results)
    return results["part_b"]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def write_report(results: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Netelpro Principle Benchmark — tokens vs JSON + error rate")
    lines.append("")
    meta = results.get("meta", {})
    lines.append(f"* Fecha: {meta.get('date', 'n/d')} · muestras por celda: "
                 f"{meta.get('n_samples', N_SAMPLES)} · temp: {TEMP} · "
                 f"seeds deterministicos por (tarea, formato, k)")
    lines.append("* Fail-closed: generacion sin bloque extraible = rechazo. "
                 "El veredicto netelpro lo emite el compilador real "
                 "(compile_filter + verify_int diferencial); el JSON lo juzga un "
                 "validador de la misma severidad (schema exacto + cobertura + semantica).")
    lines.append("")
    part_a = results.get("part_a")
    if part_a:
        lines.append("## Parte A — Principio #2: economia de tokens")
        lines.append("")
        lines.append(f"Muestra: {part_a['n_examples']} reglas de dataset_v2 "
                     f"(mismos nombres y tabla en ambos formatos). "
                     f"Tokens contados por el tokenizador REAL de cada modelo "
                     f"(Ollama prompt_eval_count, raw).")
        lines.append("")
        arts = ["netelpro_contract", "netelpro_contract_cases",
                "json_indent", "json_compact"]
        header = "| Modelo | " + " | ".join(arts) + " |"
        sep = "|---" * (len(arts) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for label, s in part_a["tokens"].items():
            row = [label]
            for a in arts:
                row.append(f"{s[a]['mean']} (x{s[a]['ratio_vs_json_indent']})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(f"Caracteres medios: " + ", ".join(
            f"{a}={part_a['chars_mean'][a]}" for a in arts))
        lines.append("")
    part_b = results.get("part_b")
    if part_b and part_b.get("cells"):
        cells = {k: v for k, v in part_b["cells"].items()
                 if isinstance(v, dict) and not v.get("error")}
        lines.append("## Parte B — Principio #1: tasa de regla maquina-valida")
        lines.append("")
        lines.append("12 tareas frescas (pool-4 de nombres, jamas visto en el train) "
                     "x 2 formatos. Valida = compila/parsea, cobertura completa, "
                     "y la tabla implementa exactamente la ley pedida.")
        lines.append("")
        lines.append("| Modelo | netelpro valida | JSON valida | delta (np - json) |")
        lines.append("|---|---|---|---|")
        agg: Dict[str, Dict[str, Tuple[int, int]]] = {}
        for key, cell in cells.items():
            label = key.split("|")[0]
            agg.setdefault(label, {"netelpro": (0, 0), "json": (0, 0)})
            for fmt in ("netelpro", "json"):
                v, n = cell[fmt]["valid"], cell[fmt]["n"]
                pv, pn = agg[label][fmt]
                agg[label][fmt] = (pv + v, pn + n)
        for label, s in agg.items():
            np_v, np_n = s["netelpro"]
            js_v, js_n = s["json"]
            np_rate = np_v / np_n if np_n else 0.0
            js_rate = js_v / js_n if js_n else 0.0
            lines.append(f"| {label} | {np_v}/{np_n} ({fmt_pct(np_rate)}) "
                         f"| {js_v}/{js_n} ({fmt_pct(js_rate)}) "
                         f"| {fmt_pct(np_rate - js_rate)} |")
        lines.append("")
        lines.append("### Distribucion de errores por clase")
        lines.append("")
        err_agg: Dict[str, Dict[str, Dict[str, int]]] = {}
        for key, cell in cells.items():
            label = key.split("|")[0]
            for fmt in ("netelpro", "json"):
                for smp in cell[fmt]["samples"]:
                    if not smp["ok"]:
                        err_agg.setdefault(label, {}).setdefault(fmt, {})
                        err_agg[label][fmt][smp["err"]] = \
                            err_agg[label][fmt].get(smp["err"], 0) + 1
        for label in err_agg:
            for fmt in ("netelpro", "json"):
                if err_agg[label].get(fmt):
                    dist = ", ".join(f"{k}: {v}" for k, v in
                                     sorted(err_agg[label][fmt].items(),
                                            key=lambda kv: -kv[1]))
                    lines.append(f"- **{label} / {fmt}**: {dist}")
        lines.append("")
        lines.append("### Por familia (cama estadistica — insumo Fase E del gate epistemico)")
        lines.append("")
        lines.append("| Familia | estrato | olmoe-base np | olmoe-gk np | qwen-base np |")
        lines.append("|---|---|---|---|---|")
        for task in task_list():
            tid = task["task_id"]
            row = [task["family"], task["stratum"]]
            for label in ("olmoe-base", "olmoe-gk", "qwen-base"):
                cell = cells.get(f"{label}|{tid}")
                if cell:
                    row.append(f"{cell['netelpro']['valid']}/{cell['netelpro']['n']}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("### Lectura pareada (OLMoE base vs goliath-killer)")
        lines.append("")
        base = agg.get("olmoe-base", {"netelpro": (0, 0)})["netelpro"]
        gk = agg.get("olmoe-gk", {"netelpro": (0, 0)})["netelpro"]
        if base[1] and gk[1]:
            lines.append(f"- netelpro: base {base[0]}/{base[1]} "
                         f"({fmt_pct(base[0] / base[1])}) -> gk {gk[0]}/{gk[1]} "
                         f"({fmt_pct(gk[0] / gk[1])})")
        base_j = agg.get("olmoe-base", {"json": (0, 0)})["json"]
        gk_j = agg.get("olmoe-gk", {"json": (0, 0)})["json"]
        if base_j[1] and gk_j[1]:
            lines.append(f"- JSON: base {base_j[0]}/{base_j[1]} "
                         f"({fmt_pct(base_j[0] / base_j[1])}) -> gk {gk_j[0]}/{gk_j[1]} "
                         f"({fmt_pct(gk_j[0] / gk_j[1])})")
        lines.append("")
    text = "\n".join(lines)
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Principle benchmark netelpro vs JSON")
    parser.add_argument("--tokens-only", action="store_true")
    parser.add_argument("--syntax-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--reload", action="store_true",
                        help="solo regenerar el reporte desde results json")
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    parser.add_argument("--limit-tokens", type=int, default=TOKEN_SAMPLE_LIMIT)
    parser.add_argument("--models", type=str, default="",
                        help="subset separado por comas (labels)")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="graders contra contratos dorados y corruptos")
    args = parser.parse_args()

    if args.list_tasks:
        for t in task_list():
            print(f"{t['task_id']:40s} stratum={t['stratum']:6s} "
                  f"table={t['table_id']:3d} params={','.join(t['params'])}")
        return 0

    if args.selftest:
        return _selftest()

    models = MODELS
    if args.models:
        wanted = [x.strip() for x in args.models.split(",") if x.strip()]
        models = {k: v for k, v in MODELS.items() if k in wanted}

    results = load_results()
    if args.reload:
        print(write_report(results))
        return 0

    results["meta"] = {"date": time.strftime("%Y-%m-%d %H:%M"),
                       "n_samples": args.samples,
                       "temp": TEMP,
                       "models": dict(models)}
    save_results(results)

    if args.full or (not args.tokens_only and not args.syntax_only):
        run_tokens(models, args.limit_tokens, TOKEN_SAMPLE_STEP, results)
        run_syntax(models, args.samples, results)
    elif args.tokens_only:
        run_tokens(models, args.limit_tokens, TOKEN_SAMPLE_STEP, results)
    else:
        run_syntax(models, args.samples, results)

    print()
    print(write_report(results))
    print(f"\nresultados: {RESULTS_JSON}")
    print(f"reporte  : {RESULTS_MD}")
    return 0


def _selftest() -> int:
    """Golden contracts must pass; corrupted ones must fail with the right class."""
    t = task_list()[0]
    fence_np = "```netelpro\n{}\n```\n```netelpro-cases\n{}\n```"
    golden_np = fence_np.format(netelpro_contract(t["params"], t["table_id"]),
                                netelpro_cases_block(t["params"], t["table_id"]))
    ok, err, d = grade_netelpro(golden_np, t["params"], t["table_id"])
    print(f"GOLDEN_NP  ok={ok} err={err} {d}")
    assert ok, "golden netelpro debe pasar"
    wrong = netelpro_contract(t["params"], t["table_id"]).replace(
        "((1 1 1) -> 1)", "((1 1 1) -> 0)")
    bad_np = fence_np.format(wrong, netelpro_cases_block(t["params"], t["table_id"]))
    ok2, err2, d2 = grade_netelpro(bad_np, t["params"], t["table_id"])
    print(f"WRONG_NP   ok={ok2} err={err2} {d2}")
    assert not ok2 and err2 == "semantica", "ley equivocada debe rebotar por semantica"
    fence_js = "```json\n{}\n```"
    golden_js = fence_js.format(json_rule(t["params"], t["table_id"]))
    ok3, err3, d3 = grade_json(golden_js, t["params"], t["table_id"])
    print(f"GOLDEN_JS  ok={ok3} err={err3} {d3}")
    assert ok3, "golden json debe pasar"
    obj = json.loads(json_rule(t["params"], t["table_id"]))
    obj["rows"][7]["step"] = 0
    bad_js = fence_js.format(json.dumps(obj, ensure_ascii=False, indent=2))
    ok4, err4, d4 = grade_json(bad_js, t["params"], t["table_id"])
    print(f"WRONG_JS   ok={ok4} err={err4} {d4}")
    assert not ok4 and err4 == "semantica", "json con ley equivocada debe rebotar por semantica"
    ok5, err5, d5 = grade_netelpro("no hay bloque aqui", t["params"], t["table_id"])
    print(f"NOBLOCK_NP ok={ok5} err={err5} {d5}")
    assert not ok5 and err5 == "sin_bloque"
    ok6, err6, d6 = grade_json("{}", t["params"], t["table_id"])
    print(f"NOBLOCK_JS ok={ok6} err={err6} {d6}")
    assert not ok6 and err6 == "sin_bloque"
    print("SELFTEST 6/6 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())