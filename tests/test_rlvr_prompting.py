"""Tests de rlvr/prompting.py: el prompt fijo del RAFT (spec resumida +
few-shot + tarea). Ver
docs/superpowers/specs/2026-09-07-rlvr-netelpro-raft-design.md §5 -- el mismo
prompt, palabra por palabra, se usa en baseline y en RAFT (es el control)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlvr.prompting import build_prompt  # noqa: E402
from rlvr.tasks import double_value, reverse_list  # noqa: E402


def test_prompt_includes_spec_summary_core_forms():
    prompt = build_prompt(double_value)
    assert "(defn nombre" in prompt
    assert "(if cond entonces sino)" in prompt


def test_prompt_includes_few_shot_examples_verbatim():
    prompt = build_prompt(double_value)
    assert "(defn fib (n)" in prompt
    assert "(defn sum-to (n acc)" in prompt


def test_prompt_includes_task_description_and_signature():
    prompt = build_prompt(double_value)
    assert double_value.DESCRIPTION_ES in prompt
    assert double_value.SIGNATURE in prompt


def test_prompt_shares_identical_preamble_across_tasks():
    prompt_a = build_prompt(double_value)
    prompt_b = build_prompt(reverse_list)
    preamble_a = prompt_a.split("Ahora escribí")[0]
    preamble_b = prompt_b.split("Ahora escribí")[0]
    assert preamble_a == preamble_b
