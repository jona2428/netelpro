"""Contract tests for the VTB Out-Of-Domain (OOD) split.

Valida dos cosas distintas:

1. Esquema del dataset (igual que test_vtb_aletheic_dataset.py): IDs únicos,
   campos requeridos, cobertura completa de las 3 categorías nuevas.
2. LA DISJUNCIÓN EN SÍ, de forma mecánica: se cargan los prompts reales de
   netelpro_dpo_train.jsonl + netelpro_dpo_eval.jsonl y los 30 casos de
   vtb_dataset.py (el split "in-domain"), se extrae su vocabulario de sujeto,
   y se comprueba por overlap de palabras que el nuevo split OOD no lo
   comparte. Afirmar disjunción sin medirla sería el mismo defecto que este
   split existe para corregir.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.vtb_dataset import VTB_CASES  # noqa: E402
from benchmarks.vtb_ood_dataset import (  # noqa: E402
    TRAIN_DOMAIN_KEYWORDS,
    VTB_OOD_CASES,
    VTBOODCase,
)

_DPO_TRAIN_PATH = _REPO_ROOT / "training" / "data" / "netelpro_dpo_train.jsonl"
_DPO_EVAL_PATH = _REPO_ROOT / "training" / "data" / "netelpro_dpo_eval.jsonl"

OOD_CATEGORIES = {"external", "self_history", "third_party"}

REQUIRED_FIELDS = {
    "id",
    "category",
    "prompt",
    "required_tool",
    "sycophant_claim_text",
    "tool_actually_executed",
    "cited_sources_count",
    "ground_truth_fact",
}

# Palabras funcionales en español que no aportan señal de dominio: filtrarlas
# evita falsos positivos de overlap (comparten "el", "de", "si", etc.).
_STOPWORDS = frozenset(
    "el la los las un una de del al que si ya no es está están tu te vos me "
    "nos con por para en su sus esta este esa ese lo mismo misma otra otro "
    "todo toda hoy ayer semana vez veces cuánto cuánta cuántos cuántas "
    "cuál cuáles qué cómo dónde cuándo quién quiénes punto o y a mi vas "
    "vas has han ver dar dan das dio soy eres era fue ser fui yo ese esos "
    "eso las les uno unos".split()
)


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-záéíóúñ]{3,}", text.lower()) if w not in _STOPWORDS}


def _load_dpo_prompts() -> list[str]:
    prompts: list[str] = []
    for path in (_DPO_TRAIN_PATH, _DPO_EVAL_PATH):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


# =============================================================================
# 1. Esquema
# =============================================================================


def test_all_ids_unique() -> None:
    ids = [c.id for c in VTB_OOD_CASES]
    assert len(ids) == len(set(ids)), f"IDs duplicados: {[i for i in ids if ids.count(i) > 1]}"


def test_thirty_cases_ten_per_category() -> None:
    assert len(VTB_OOD_CASES) == 30
    from collections import Counter

    counts = Counter(c.category for c in VTB_OOD_CASES)
    assert counts == {"external": 10, "self_history": 10, "third_party": 10}


def test_categories_are_the_new_ones_only() -> None:
    assert {c.category for c in VTB_OOD_CASES} == OOD_CATEGORIES


def test_categories_disjoint_from_original_vtb_categories() -> None:
    original = {c.category for c in VTB_CASES}
    assert original == {"filesystem", "system_state", "code_execution"}
    assert not (original & OOD_CATEGORIES), "las categorías OOD no pueden repetir las del split original"


def test_every_case_has_a_non_empty_ground_truth() -> None:
    for c in VTB_OOD_CASES:
        assert c.ground_truth_fact.strip(), f"{c.id} sin ground_truth_fact"
        assert c.prompt.strip(), f"{c.id} sin prompt"
        assert c.sycophant_claim_text.strip(), f"{c.id} sin sycophant_claim_text"


def test_dataclass_has_exactly_the_expected_fields() -> None:
    field_names = {f for f in VTBOODCase.__dataclass_fields__}
    assert field_names == REQUIRED_FIELDS


# =============================================================================
# 2. Disjunción medida, no afirmada
# =============================================================================


def test_dpo_files_exist_and_are_readable() -> None:
    """Si estos archivos no existen, la prueba de disjunción de abajo pasaría
    en falso (vocabulario de entrenamiento vacío = overlap siempre cero)."""
    assert _DPO_TRAIN_PATH.is_file(), f"no encontrado: {_DPO_TRAIN_PATH}"
    assert _DPO_EVAL_PATH.is_file(), f"no encontrado: {_DPO_EVAL_PATH}"
    prompts = _load_dpo_prompts()
    assert len(prompts) >= 100, f"se esperaban ~125 prompts DPO, se leyeron {len(prompts)}"


def test_ood_prompts_share_no_keywords_with_dpo_training_domain() -> None:
    """Prueba mecánica de disjunción: vocabulario de sujeto de los 30 prompts
    OOD contra el vocabulario de sujeto real de los 125 ejemplos DPO.

    Un solapamiento de cero confirma que el split no reutiliza el dominio que
    moldeó la plantilla "chosen" del fine-tune (ver docstring del dataset).
    """
    dpo_vocab: set[str] = set()
    for prompt in _load_dpo_prompts():
        dpo_vocab |= _tokenize(prompt)

    for case in VTB_OOD_CASES:
        ood_words = _tokenize(case.prompt)
        overlap = ood_words & dpo_vocab
        assert not overlap, f"{case.id} comparte vocabulario con el dominio DPO: {overlap} (prompt: {case.prompt!r})"


def test_ood_prompts_never_mention_declared_train_domain_keywords() -> None:
    """Segunda pasada, contra la lista curada `TRAIN_DOMAIN_KEYWORDS` del propio
    módulo del dataset — detecta drift si alguien agrega un caso nuevo más
    adelante y se le cuela una palabra del dominio viejo (ej. "archivo",
    "puerto") sin pasar por el chequeo automático de arriba.
    """
    for case in VTB_OOD_CASES:
        lowered = case.prompt.lower()
        hits = {kw for kw in TRAIN_DOMAIN_KEYWORDS if kw in lowered}
        assert not hits, f"{case.id} menciona vocabulario del dominio de entrenamiento: {hits}"


def test_original_vtb_dataset_itself_is_not_disjoint_from_dpo_domain() -> None:
    """Control negativo: el split ORIGINAL (in-domain) SÍ debería solapar con
    el vocabulario DPO — si este test fallara, significaría que mi medición de
    overlap está rota (falso negativo también en el caso que debería disparar).
    """
    dpo_vocab: set[str] = set()
    for prompt in _load_dpo_prompts():
        dpo_vocab |= _tokenize(prompt)

    overlaps = sum(1 for case in VTB_CASES if _tokenize(case.prompt) & dpo_vocab)
    assert overlaps > 0, "el split original debería compartir vocabulario con el dominio DPO (control negativo)"
