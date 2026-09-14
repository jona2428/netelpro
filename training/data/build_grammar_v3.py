#!/usr/bin/env python3
"""Grammar v3 SFT Dataset Generator for Netelpro Truth-Table Syntax.
===================================================================

Builds an amplified SFT dataset for fine-tuning Qwen2.5-1.5B-Instruct to emit
machine-valid Netelpro truth-table rule syntax and verification cases.

Context & Design:
1. Anti-template-mismatch: Task prompts match the evaluation format in
   `benchmarks/principle_bench.py` byte-for-byte (`build_prompt(task, "netelpro")`).
2. Honest targets: Completions are strictly the golden truth-table contract
   and verification cases block without thinking tags or unverified prose.
3. Oracle verification: Targets are computed by `oracle_cases` from
   `builders/truth_table_builder.py` (real Netelpro reference interpreter),
   never hand-written.
4. Native compiler verification: Every target is compiled with `compile_filter`
   (LLVM JIT native engine) and differentially verified with `rf.verify_int`
   and graded with `grade_netelpro`.
5. Anti-forgetting mix: 25% of the final train mix is sampled deterministically
   from `training/data/dataset_v2.jsonl`.
6. Strict disjointness: Held-out rule names and signals NEVER appear in the
   training set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from netelpro.parser import parse
from netelpro.rule_filter import RuleFilterError, compile_filter
from builders.truth_table_builder import oracle_cases
from benchmarks.principle_bench import (
    NETELPRO_SPEC,
    bits_for_row,
    build_prompt,
    grade_netelpro,
    netelpro_cases_block,
    netelpro_contract,
    on_set_desc,
    verdict_for_row,
)
from benchmarks.vtb_ood_runner import HONESTY_SYSTEM_PROMPT

# Official honesty system prompt (prompt-first decision)
SYSTEM_PROMPT: str = HONESTY_SYSTEM_PROMPT

DEFAULT_SEED: int = 42
FIXED_TIMESTAMP: float = 1789300000.0

# 10 diverse systems domains
DOMAIN_SPECS = [
    (
        "ratelimit",
        "Quimica Fisica (Cinetica de Arrhenius)",
        "Sistemas Distribuidos (Rate Limiting y Token Bucket)",
        "Ley de Accion de Masas y Capacidad de Absorcion",
        "regulacion de tasa de peticiones con cubetas de desbordamiento",
    ),
    (
        "backpressure",
        "Hidraulica y Fluidodinamica",
        "Transmision de Flujos y Colas Reactivas",
        "Principio de Continuidad y Amortiguacion Hidraulica",
        "control de presion inversa en canales de streaming distribuido",
    ),
    (
        "circuit_breaker",
        "Ingenieria Electrica de Potencia",
        "Tolerancia a Fallos y Microservicios",
        "Apertura Termomagnetica y Desacople de Redes Sobrecargadas",
        "interruptor de circuito para aislar dependencias degradadas",
    ),
    (
        "arbiter",
        "Teoria de Colas y Asignacion Justa",
        "Sistemas Operativos y Acceso a Buses UMA/PCIe",
        "Invariante de Exclusion Mutua y Progreso Libre de Inanicion",
        "arbitro de bus de memoria compartida con cesion justa de turnos",
    ),
    (
        "scheduler",
        "Mecanica Estadistica de Glauber",
        "Planificacion de Procesos y Computacion Distribuida",
        "Minimizacion de Energia Libre y Templado Simulado",
        "despachador estocastico de tareas con umbrales de enfriamiento",
    ),
    (
        "gc",
        "Biologia Celular (Autofagia)",
        "Gestion de Memoria y Recoleccion de Basura",
        "Reciclaje Homeostatico sin Detencion Vital",
        "colector concurrente de memoria sin detencion del flujo principal",
    ),
    (
        "routing",
        "Neurobiologia (Poda Sinaptica)",
        "Enrutamiento en Redes Sparse MoE",
        "Regla de Hebb y Poda de Conexiones Subutilizadas",
        "enrutador de expertos MoE con desconexion de rutas redundantes",
    ),
    (
        "hnsw",
        "Cristalografia y Redes de Bravais",
        "Indices Espaciales y Grafos de Navegacion HNSW",
        "Minimizacion de Distancia Geodesica y Poda de Aristas",
        "busqueda en grafos jerarquicos con poda de trayectorias suboptimas",
    ),
    (
        "gossip",
        "Epidemiologia Estocastica",
        "Consenso Descentralizado y Protocolos Gossip",
        "Propagacion Exponencial de Informacion por Contacto Aleatorio",
        "protocolo gossip de difusion de estados con amortiguacion de ecos",
    ),
    (
        "apoptosis",
        "Genetica Celular (Cascada de Caspasas)",
        "Orquestacion Resiliente de Contenedores y Pods",
        "Aislamiento Programado y Muerte Celular Controlada",
        "terminacion deliberada de instancias insalubres para preservar el cluster",
    ),
]


def is_valid_netelpro_ident(name: str) -> bool:
    """Netelpro allows kebab-case identifiers mapping '-' to '_' in Python."""
    cand = name.replace("-", "_")
    return cand.isidentifier() and not keyword.iskeyword(cand)


def synthesize_rule_definitions(
    n_per_domain: int = 24,
    n_heldout_per_domain: int = 2,
    reserved_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Synthesize >= 200 distinct filter-rule definitions across domains.

    Args:
        n_per_domain: Number of definitions per domain (default 24 -> 240 total).
        n_heldout_per_domain: How many definitions per domain are designated held-out (default 2 -> 20 held-out).
        reserved_names: Existing names to avoid colliding with (e.g. dataset_v2 or bench pool 4).

    Returns:
        List of rule definition dicts with completely unique, non-overlapping signals.
    """
    if reserved_names is None:
        reserved_names = set()

    definitions: List[Dict[str, Any]] = []
    used_params: Set[str] = set(reserved_names)

    for dom_id, dom_a, dom_b, law, scenario in DOMAIN_SPECS:
        for idx in range(n_per_domain):
            fam_name = f"synth_{dom_id}_{idx+1:02d}"
            p0 = f"syn-{dom_id[:4]}-a{idx+1:02d}-ok"
            p1 = f"syn-{dom_id[:4]}-b{idx+1:02d}-ok"
            p2 = f"syn-{dom_id[:4]}-c{idx+1:02d}-gate"
            params = [p0, p1, p2]

            for p in params:
                if not is_valid_netelpro_ident(p):
                    raise ValueError(f"Generated invalid Netelpro identifier: {p}")
                if p in used_params:
                    raise ValueError(f"Param name collision detected: {p}")
                used_params.add(p)

            is_heldout = idx >= (n_per_domain - n_heldout_per_domain)
            definitions.append({
                "family": fam_name,
                "domain_a": dom_a,
                "domain_b": dom_b,
                "shared_law": f"{law} [Variante {idx+1:02d}]",
                "scenarios": f"{scenario} (modo de operacion {idx+1:02d})",
                "params": params,
                "is_heldout": is_heldout,
            })

    return definitions


# Representative strata of table_ids across difficulty classes
STRATUM_TABLE_CANDIDATES = [
    # 1_on (hardest: only 1 combo ON)
    [1, 2, 4, 8, 16, 32, 64, 128],
    # 2_on (2 combos ON)
    [3, 5, 9, 17, 33, 65, 129, 6],
    # 3_on (3 combos ON)
    [7, 11, 19, 35, 67, 131, 14, 21],
    # 4_on (mid: 4 combos ON, including half-space and parity)
    [15, 23, 240, 170, 85, 51, 102, 204],
    # 5_on (5 combos ON)
    [31, 47, 79, 143, 241, 171, 87, 55],
    # 6_on (6 combos ON)
    [63, 95, 159, 243, 175, 91, 245, 111],
    # 7_on (easiest: all but 1 combo ON)
    [254, 253, 251, 247, 239, 223, 191, 127],
]


def table_id_for_stratum(def_idx: int, stratum_idx: int) -> int:
    """Deterministically select a table_id for a given rule index and stratum."""
    candidates = STRATUM_TABLE_CANDIDATES[stratum_idx % len(STRATUM_TABLE_CANDIDATES)]
    return candidates[def_idx % len(candidates)]


def generate_sample(
    defn: Dict[str, Any],
    table_id: int,
    verify_compiler: bool = True,
) -> Optional[Dict[str, Any]]:
    """Generate a single verified grammar SFT sample.

    1. Renders the byte-exact bench prompt (`build_prompt(task, "netelpro")`).
    2. Renders truth-table contract.
    3. Evaluates reference interpreter oracle (`oracle_cases`).
    4. Compiles natively via LLVM JIT (`compile_filter`) and differentially
       verifies with `verify_int`.
    5. Validates with `grade_netelpro`.
    6. Returns JSONL dict with exact dataset_v2 schema.
    """
    params = defn["params"]
    task = {
        "params": params,
        "scenarios": defn["scenarios"],
        "domain_a": defn["domain_a"],
        "domain_b": defn["domain_b"],
        "law": defn["shared_law"],
        "table_id": table_id,
    }

    # Byte-exact bench prompt
    prompt = build_prompt(task, "netelpro")

    # Contract source
    contract = netelpro_contract(params, table_id)

    # 1. Oracle cases evaluation (reference interpreter)
    prog = parse(contract)
    if prog.errors:
        return None
    ast_defn = prog.program.forms[0]
    oracle_res = oracle_cases(ast_defn, contract)

    # 2. Extract in-domain cases directly from oracle
    in_domain_tuples = [bits_for_row(r) for r in range(8)]
    cases_lines = [
        f"({a},{b},{c}) -> {int(oracle_res[(a, b, c)])}"
        for a, b, c in in_domain_tuples
    ]
    cases_block = "\n".join(cases_lines)

    # Clean target completion: exact fence block matching bench golden response
    completion = f"```netelpro\n{contract}\n```\n```netelpro-cases\n{cases_block}\n```"

    # 3. Native compiler verification (fail-closed)
    if verify_compiler:
        try:
            rf = compile_filter(contract)
            truth_cases = [
                (combo, int(oracle_res[combo]))
                for combo in in_domain_tuples
            ]
            mismatches = rf.verify_int(truth_cases)
            if mismatches:
                return None
        except Exception:
            return None

        # 4. Benchmark evaluator verification
        ok, _, _ = grade_netelpro(completion, params, table_id)
        if not ok:
            return None

    # Exact dataset_v2 field schema
    netelpro_cases_list = [
        [a, b, c, int(oracle_res[(a, b, c)])]
        for a, b, c in in_domain_tuples
    ]

    return {
        "domain_a": defn["domain_a"],
        "domain_b": defn["domain_b"],
        "shared_law": defn["shared_law"],
        "prompt": prompt,
        "completion": completion,
        "netelpro_contract": contract,
        "netelpro_cases": netelpro_cases_list,
        "netelpro_verified": True,
        "family": defn["family"],
        "table_id": table_id,
        "params": params,
        "generator": "build_grammar_v3.py (oracle-verified, LLVM-verified)",
        "timestamp": FIXED_TIMESTAMP,
    }


def to_chat_format(
    row: Dict[str, Any],
    system_prompt: str = HONESTY_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Convert a sample row to standard ChatML messages format."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["completion"]},
    ]


def load_dataset_v2_params(v2_path: Path) -> Set[str]:
    """Extract all parameter names from dataset_v2.jsonl."""
    params: Set[str] = set()
    if not v2_path.exists():
        return params
    with open(v2_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                params.update(d.get("params", []))
    return params


def build_grammar_datasets(
    output_dir: Path,
    n_train_target: int = 1500,
    n_heldout_target: int = 140,
    v2_mix_fraction: float = 0.25,
    seed: int = DEFAULT_SEED,
    v2_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build train.jsonl, heldout.jsonl, and train_mix.jsonl.

    Args:
        output_dir: Directory where the output .jsonl files will be written.
        n_train_target: Minimum number of grammar train samples (>=1500).
        n_heldout_target: Minimum number of heldout samples (>=120).
        v2_mix_fraction: Fraction of the final train mix sampled from dataset_v2 (0.25).
        seed: Fixed seed for determinism.
        v2_path: Path to dataset_v2.jsonl.

    Returns:
        Summary dict containing counts, file paths, compile stats, and pass rates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if v2_path is None:
        v2_path = REPO_ROOT / "training" / "data" / "dataset_v2.jsonl"

    reserved = load_dataset_v2_params(v2_path)
    definitions = synthesize_rule_definitions(
        n_per_domain=24,
        n_heldout_per_domain=2,
        reserved_names=reserved,
    )

    train_defs = [d for d in definitions if not d["is_heldout"]]
    heldout_defs = [d for d in definitions if d["is_heldout"]]

    # Verify disjointness of parameter names
    train_param_names = {p for d in train_defs for p in d["params"]}
    heldout_param_names = {p for d in heldout_defs for p in d["params"]}
    assert train_param_names.isdisjoint(heldout_param_names), "Train and heldout params must be disjoint!"

    t0 = time.time()
    compile_attempts = 0
    compile_passes = 0

    # 1. Generate train samples
    train_samples: List[Dict[str, Any]] = []
    def_idx = 0
    stratum_idx = 0

    while len(train_samples) < n_train_target:
        defn = train_defs[def_idx % len(train_defs)]
        tid = table_id_for_stratum(def_idx, stratum_idx)
        compile_attempts += 1
        sample = generate_sample(defn, tid, verify_compiler=True)
        if sample is not None:
            compile_passes += 1
            train_samples.append(sample)
        def_idx += 1
        if def_idx % len(train_defs) == 0:
            stratum_idx += 1

    # 2. Generate held-out samples
    heldout_samples: List[Dict[str, Any]] = []
    h_def_idx = 0
    h_stratum_idx = 0

    while len(heldout_samples) < n_heldout_target:
        defn = heldout_defs[h_def_idx % len(heldout_defs)]
        tid = table_id_for_stratum(h_def_idx, h_stratum_idx)
        compile_attempts += 1
        sample = generate_sample(defn, tid, verify_compiler=True)
        if sample is not None:
            compile_passes += 1
            heldout_samples.append(sample)
        h_def_idx += 1
        if h_def_idx % len(heldout_defs) == 0:
            h_stratum_idx += 1

    # 3. Sample dataset_v2 for anti-forgetting train mix
    # If v2 is 25% of the final train mix:
    # n_v2 / (n_train + n_v2) = v2_mix_fraction
    # n_v2 = round(n_train * v2_mix_fraction / (1.0 - v2_mix_fraction))
    v2_rows: List[Dict[str, Any]] = []
    if v2_path.exists():
        with open(v2_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    v2_rows.append(json.loads(line))

    n_v2_needed = int(round(len(train_samples) * v2_mix_fraction / (1.0 - v2_mix_fraction)))
    rng = random.Random(seed)
    v2_sampled = rng.sample(v2_rows, n_v2_needed) if len(v2_rows) >= n_v2_needed else v2_rows

    # Combine and shuffle train mix with fixed seed
    train_mix = list(train_samples) + list(v2_sampled)
    rng.shuffle(train_mix)

    # 4. Write output files
    train_file = output_dir / "train.jsonl"
    heldout_file = output_dir / "heldout.jsonl"
    mix_file = output_dir / "train_mix.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(heldout_file, "w", encoding="utf-8") as f:
        for item in heldout_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(mix_file, "w", encoding="utf-8") as f:
        for item in train_mix:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dt = time.time() - t0
    compile_pass_rate = (compile_passes / compile_attempts) * 100.0 if compile_attempts > 0 else 0.0

    return {
        "definitions_count": len(definitions),
        "train_definitions_count": len(train_defs),
        "heldout_definitions_count": len(heldout_defs),
        "train_count": len(train_samples),
        "heldout_count": len(heldout_samples),
        "mix_count": len(train_mix),
        "v2_sampled_count": len(v2_sampled),
        "v2_mix_pct": round(len(v2_sampled) / len(train_mix) * 100.0, 2),
        "compile_attempts": compile_attempts,
        "compile_passes": compile_passes,
        "compile_pass_rate": compile_pass_rate,
        "duration_seconds": round(dt, 2),
        "train_file": str(train_file),
        "heldout_file": str(heldout_file),
        "mix_file": str(mix_file),
        "train_size_bytes": train_file.stat().st_size,
        "heldout_size_bytes": heldout_file.stat().st_size,
        "mix_size_bytes": mix_file.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Netelpro Grammar v3 SFT dataset")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "training" / "data")
    parser.add_argument("--train-samples", type=int, default=1500)
    parser.add_argument("--heldout-samples", type=int, default=140)
    parser.add_argument("--v2-mix-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print("=======================================================================")
    print("NETELPRO GRAMMAR V3 SFT DATASET GENERATOR")
    print("=======================================================================")
    print(f"Output directory    : {args.output_dir}")
    print(f"Target train samples: {args.train_samples}")
    print(f"Target heldout      : {args.heldout_samples}")
    print(f"Dataset v2 replay   : {args.v2_mix_fraction * 100:.1f}% of final mix")
    print(f"Deterministic seed  : {args.seed}")
    print("Generating oracle-verified targets...")

    res = build_grammar_datasets(
        output_dir=args.output_dir,
        n_train_target=args.train_samples,
        n_heldout_target=args.heldout_samples,
        v2_mix_fraction=args.v2_mix_fraction,
        seed=args.seed,
    )

    print("\n-----------------------------------------------------------------------")
    print("GENERATION RESULTS")
    print("-----------------------------------------------------------------------")
    print(f"Rule definitions synthesized : {res['definitions_count']} (train: {res['train_definitions_count']}, heldout: {res['heldout_definitions_count']})")
    print(f"train.jsonl                  : {res['train_count']} samples ({res['train_size_bytes'] / 1024:.1f} KB)")
    print(f"heldout.jsonl                : {res['heldout_count']} samples ({res['heldout_size_bytes'] / 1024:.1f} KB)")
    print(f"train_mix.jsonl              : {res['mix_count']} samples ({res['mix_size_bytes'] / 1024:.1f} KB)")
    print(f"  └─ v2 replay component     : {res['v2_sampled_count']} samples ({res['v2_mix_pct']}%)")
    print(f"Compiler attempts            : {res['compile_attempts']}")
    print(f"Compiler passes              : {res['compile_passes']}")
    print(f"Compiler pass rate           : {res['compile_pass_rate']:.2f}%")
    print(f"Elapsed time                 : {res['duration_seconds']}s")
    print("-----------------------------------------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())
