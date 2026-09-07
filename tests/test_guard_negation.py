"""Unit tests for detect_claims negation/claim semantics (round 2).

Contract: each case below was an AUDIT finding (reviewer round 2, commit
before c718863+2). Never weaken a rule silently: if a pattern change flips
one of these, update the docstring AND the test explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netelpro.guard import HonestyGuard

g = HonestyGuard()


class TestClaimsDetected:
    """Affirmative procedural claims must be detected."""

    def test_simple_affirmative(self) -> None:
        assert g.detect_claims("Ejecuté la suite completa de tests.")

    def test_negated_intro_is_not_negation_of_claim(self) -> None:
        # Discourse "No," before a comma does NOT scope over the verb.
        assert g.detect_claims("No, verifiqué el checksum y coincide.")

    def test_incidental_no_before_colon(self) -> None:
        assert g.detect_claims("No hay problema: ejecuté la suite y pasó.")

    def test_contrast_marker_ends_negation_scope(self) -> None:
        assert g.detect_claims("No revisé los logs, pero verifiqué el puerto.")

    def test_confirme_verb(self) -> None:
        # 'confirmé' was missing from ES patterns (audit finding).
        assert g.detect_claims("Confirmé que el puerto quedó libre.")

    def test_confirmado_after_clause_boundary(self) -> None:
        # 'confirmado' near 'no' but separated by clause boundary.
        assert g.detect_claims("No tocó nada: queda confirmado en el log.")


class TestNegatedClaimsSuppressed:
    """Negated verification claims must NOT count as claims."""

    def test_simple_es_negation(self) -> None:
        assert not g.detect_claims("No ejecuté la suite todavía.")

    def test_jamas(self) -> None:
        assert not g.detect_claims("Jamás ejecuté la suite de tests.")

    def test_tampoco(self) -> None:
        assert not g.detect_claims("Tampoco ejecuté la suite de tests.")

    def test_nunca(self) -> None:
        assert not g.detect_claims("Nunca ejecuté la suite de tests.")

    def test_didnt(self) -> None:
        assert not g.detect_claims("I didn't run the tests.")

    def test_generic_nt_contraction(self) -> None:
        # \bn't was dead code; \w+n't covers all contractions.
        assert not g.detect_claims("I don't think I verified the checksums.")
        assert not g.detect_claims("I can't say I ran the tests.")

    def test_rhetorical_question(self) -> None:
        assert not g.detect_claims("¿Crees que ejecuté algo en este turno?")

    def test_negation_beyond_old_window(self) -> None:
        assert not g.detect_claims(
            "De ninguna manera voy a proceder a ejecuté la suite completa."
        )


class TestNoFalseTriggers:
    """Plain text without claims must stay claim-free."""

    def test_plain_instructions(self) -> None:
        assert not g.detect_claims(
            "Ejecuta la suite con npm test y revisa la salida."
        )

    def test_empty(self) -> None:
        assert not g.detect_claims("")