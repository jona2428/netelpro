"""Build (and optionally push) one Kaggle kernel per Receipts-RAFT stage.

Spec v0.2 §4, §8 step 3. Each (arm, stage) gets its own kernel
`<user>/receipts-raft-<arm>-s<stage>` whose notebook is
train_receipts_raft_kaggle.ipynb with ARM/STAGE fixed, and whose input is the
previous stage's kernel output (kernel_sources), so the pool and the LoRA
travel stage to stage without leaving Kaggle.

Default is a dry build into training/kernels/; nothing is sent. --push sends
the kernel (spends Kaggle GPU quota). Before stage r >= 1 the audit labels of
round r-1 must be committed AND pushed to GitHub: the kernel clones the repo
and refuses to train without them (rlvr.receipts_raft.require_audit).

    python training/push_receipts_raft.py --arm A --stage 0            # build only
    python training/push_receipts_raft.py --arm A --stage 0 --push     # send to Kaggle
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rlvr.receipts_raft import SHARED_STAGE0_ARM, audit_labels_path  # noqa: E402

TRAINING_DIR = Path(__file__).parent
REPO = TRAINING_DIR.parent
NOTEBOOK = TRAINING_DIR / "train_receipts_raft_kaggle.ipynb"
KERNELS_DIR = TRAINING_DIR / "kernels"
KAGGLE_USER = "jonacgerizo"
NUM_ROUNDS = 5


def slug(arm: str, stage: int) -> str:
    return f"receipts-raft-{arm.lower()}-s{stage}"


def build(arm: str, stage: int, *, user: str = KAGGLE_USER) -> Path:
    if arm not in ("A", "B") or not 0 <= stage <= NUM_ROUNDS:
        raise ValueError(f"arm {arm!r} stage {stage}")
    if stage == 0 and arm != SHARED_STAGE0_ARM:
        raise SystemExit(f"stage 0 is shared: it runs once as arm {SHARED_STAGE0_ARM} (rlvr.receipts_raft.SHARED_STAGE0_ARM)")
    if stage > 0:
        labels = audit_labels_path(REPO, arm, stage - 1)
        if not labels.exists():
            raise SystemExit(f"stage {stage} needs {labels.relative_to(REPO)} (hand audit of round {stage - 1}), committed and pushed")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    hits = 0
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and src.lstrip().startswith("# Lo sobreescribe training/push_receipts_raft.py"):
            src, n1 = re.subn(r'^ARM = "[AB]"', f'ARM = "{arm}"', src, flags=re.M)
            src, n2 = re.subn(r"^STAGE = \d+", f"STAGE = {stage}", src, flags=re.M)
            if (n1, n2) != (1, 1):
                raise SystemExit("config cell anchors changed -- regenerate the notebook")
            cell["source"] = src.splitlines(keepends=True)
            hits += 1
    if hits != 1:
        raise SystemExit(f"expected one config cell, found {hits}")

    name = slug(arm, stage)
    out = KERNELS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "id": f"{user}/{name}",
        "title": name,
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "keywords": ["gpu"],
        "dataset_sources": [],
        "kernel_sources": [f"{user}/{slug(SHARED_STAGE0_ARM if stage == 1 else arm, stage - 1)}"] if stage > 0 else [],
        "competition_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, choices=("A", "B"))
    ap.add_argument("--stage", required=True, type=int)
    ap.add_argument("--push", action="store_true", help="send to Kaggle (spends GPU quota)")
    a = ap.parse_args(argv)
    out = build(a.arm, a.stage)
    print(f"built {out}")
    if a.push:
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(out)], capture_output=True, text=True)
        print(r.stdout or r.stderr)
        return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
