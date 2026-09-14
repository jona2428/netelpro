"""Fix notebook run5: force 1 GPU (bitsandbytes NF4 breaks under DataParallel)."""
import json

P = "benchmarks/train_qlora_dora_run5_kaggle.ipynb"
nb = json.load(open(P, encoding="utf-8"))

# --- Fix 1: cell 4 (hardware diagnostics) ---
c = nb["cells"][4]
src = "".join(c["source"])

old = 'os.environ["WANDB_DISABLED"] = "true"'
assert src.count(old) == 1, "anchor WANDB not found"
src = src.replace(old, '# WANDB removed: SFTConfig already uses report_to="none"')

old2 = "import torch\nimport bitsandbytes as bnb"
assert src.count(old2) == 1, "anchor torch import not found"
new2 = (
    "# FORCE 1 GPU: bitsandbytes NF4 does not support nn.DataParallel (Kaggle 2xT4\n"
    '# triggers CUBLAS_STATUS_NOT_SUPPORTED in replicas). With 1.5B + QLoRA,\n'
    "# one T4 is plenty. Must be set BEFORE importing torch.\n"
    'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
    "\n"
    "import torch\n"
    "import bitsandbytes as bnb"
)
src = src.replace(old2, new2)
c["source"] = src.splitlines(keepends=True)

# --- Fix 2: cell 10 (training) ---
c10 = nb["cells"][10]
src10 = "".join(c10["source"])
old3 = "from trl import SFTConfig, SFTTrainer"
assert src10.count(old3) == 1, "anchor trl import not found"
src10 = src10.replace(
    old3,
    old3
    + '\nimport os as _os\n_os.environ.pop("WANDB_DISABLED", None)  # kill deprecation warning',
)
c10["source"] = src10.splitlines(keepends=True)

json.dump(nb, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# --- verify from disk ---
nb2 = json.load(open(P, encoding="utf-8"))
src4 = "".join(nb2["cells"][4]["source"])
src10 = "".join(nb2["cells"][10]["source"])
assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in src4
assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in src4.split("import torch")[0], \
    "CUDA_VISIBLE_DEVICES must precede torch import"
assert src4.count("WANDB_DISABLED") == 0
assert src10.count('os.environ.pop("WANDB_DISABLED"') == 1
json.loads(open(P, encoding="utf-8").read())  # valid JSON roundtrip
print("FIX OK: 1-GPU forced before torch import; WANDB env removed; notebook JSON valid")