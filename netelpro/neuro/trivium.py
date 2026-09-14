"""Netelpro Trivium Engine: Rhetoric, Dialectics, and Formal Fallacy Detection.

Integrates classical rhetoric (Ethos, Pathos, Logos) and social science argumentation
with compiled LLVM formal gates to verify logical soundness and diagnose fallacies.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from netelpro.gate import Gate

_DEFAULT_FALLACY_RULE = Path(__file__).parent / "rules" / "fallacy_detector.sl"


@dataclass
class TriviumAuditRecord:
    """Audit record certifying the rhetorical and logical soundness of an argument."""

    is_valid: bool
    fallacies_detected: list[str]
    logos_score: float
    ethos_score: float
    pathos_score: float
    latency_us: float
    formal_proof: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "fallacies_detected": self.fallacies_detected,
            "logos": round(self.logos_score, 2),
            "ethos": round(self.ethos_score, 2),
            "pathos": round(self.pathos_score, 2),
            "latency_us": round(self.latency_us, 2),
            "formal_proof": self.formal_proof,
        }


class NetelproTriviumEngine:
    """Rhetorical and logical analyzer governed by Netelpro formal silicon gates."""

    def __init__(self, rule_path: str | Path | None = None) -> None:
        actual_path = Path(rule_path) if rule_path else _DEFAULT_FALLACY_RULE
        self.gate = Gate(actual_path)

        # Common linguistic markers for rhetorical analysis
        self._re_ad_hominem = re.compile(
            r"(no le crean|es un (incompetente|ignorante|tonto|mentiroso|corrupto)|"
            r"eres (incompetente|ignorante|tonto|mentiroso)|por ser (joven|viejo|pobre|rico))",
            re.IGNORECASE,
        )
        self._re_dichotomy = re.compile(
            r"(o estás conmigo o|o apoyas .* o estás en contra|o es blanco o negro|"
            r"la única opción es|o todo o nada)",
            re.IGNORECASE,
        )
        self._re_straw_man = re.compile(
            r"(dices que no importa nada|pretendes que regalemos|quieres destruir|afirmas que todo es fácil)",
            re.IGNORECASE,
        )
        self._re_grounding = re.compile(
            r"(porque|dado que|la evidencia|según los datos|por lo tanto|en consecuencia|debido a)",
            re.IGNORECASE,
        )

    def analyze_argument(self, text: str) -> TriviumAuditRecord:
        """Analyzes an argument for rhetorical balance and evaluates fallacy gates in silicon."""
        t0 = time.perf_counter_ns()

        fallacies: list[str] = []

        # 1. Evaluate indicators
        ad_hominem = 1 if self._re_ad_hominem.search(text) else 0
        if ad_hominem:
            fallacies.append("ad_hominem")

        false_dichotomy = 1 if self._re_dichotomy.search(text) else 0
        if false_dichotomy:
            fallacies.append("false_dichotomy")

        straw_man = 1 if self._re_straw_man.search(text) else 0
        if straw_man:
            fallacies.append("straw_man")

        premise_grounded = 1 if self._re_grounding.search(text) else 0
        if not premise_grounded and not fallacies and len(text.split()) > 4:
            # Short assertion without premise
            premise_grounded = 0
            fallacies.append("unsupported_assertion")
        elif not fallacies:
            premise_grounded = 1

        # 2. Query compiled LLVM silicon gate
        is_valid, _ = self.gate.check(ad_hominem, false_dichotomy, straw_man, premise_grounded)

        # 3. Compute rhetorical balance (Ethos, Pathos, Logos)
        logos = 0.9 if is_valid else max(0.1, 0.8 - len(fallacies) * 0.3)
        ethos = 0.3 if ad_hominem else 0.85
        pathos = 0.7 if (false_dichotomy or straw_man) else 0.5

        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1000.0

        verdict_str = "VÁLIDO (Admitido)" if is_valid else f"FALAZ (Inhibido: {', '.join(fallacies)})"
        proof = f"Gate[ad_hom={ad_hominem}, dich={false_dichotomy}, straw={straw_man}, ground={premise_grounded}] => {verdict_str}"

        return TriviumAuditRecord(
            is_valid=is_valid,
            fallacies_detected=fallacies,
            logos_score=logos,
            ethos_score=ethos,
            pathos_score=pathos,
            latency_us=latency_us,
            formal_proof=proof,
        )

    def format_dialectical_prompt(self, topic: str, user_claim: str) -> str:
        """Formats a dialectical inquiry prompt for the Netelpro Mini LLM."""
        return (
            f"<|user|>\n"
            f"Analiza dialécticamente el siguiente planteamiento sobre '{topic}':\n"
            f"\"{user_claim}\"\n"
            f"<|assistant|>\n"
            f"<|thought|>\n"
            f"Desglosar tesis, antítesis, evaluar solidez lógica y sintetizar con perspectiva humanista.\n"
            f"<|endthought|>\n"
        )
