"""Labeled corpus for HonestyGuard.detect_claims -- the regex-based NLP layer.

VTB (benchmarks/vtb_dataset.py) exercises the *compiled decision* over
(claimed, verified, sources) triples with `override_claimed` always set, so it
never actually calls `detect_claims()` on real text. This is the only
regression coverage for the regex heuristic itself: negation scope, clause
boundaries, question spans, English contractions -- the exact edge cases
`guard.py`'s own docstrings describe but that nothing previously checked.

Each case is (text, expected_is_claim, note). `note` records *why* the label
is what it is -- for false positives/negatives caused by a real, documented
gap in the current regex (not a test bug), the case is marked
`xfail_known_gap` instead of silently asserting the wrong thing.
"""

from __future__ import annotations

import pytest

from netelpro.guard import HonestyGuard

guard = HonestyGuard()

# (text, expected_is_claim, note)
POSITIVE_ES = [
    ("He revisado el archivo .env y todo está en orden.", True, "he revisado"),
    ("Verifiqué los puertos y el 8000 está libre.", True, "verifiqué"),
    ("Comprobé la config y no hay errores.", True, "comprobé"),
    ("Confirmé que el servicio está activo.", True, "confirmé"),
    ("Ejecuté la suite completa y todo pasó.", True, "ejecuté"),
    ("Corrí los tests de integración anoche.", True, "corrí"),
    ("Pasé el linter sobre el módulo.", True, "pasé"),
    ("Medí la latencia p99 del endpoint.", True, "medí"),
    ("Validé el esquema contra el JSON de entrada.", True, "validé"),
    ("Audité los permisos del bucket.", True, "audité"),
    ("Escaneé las dependencias con pip-audit.", True, "escaneé"),
    ("Testeé el flujo de login manualmente.", True, "testeé"),
    ("Analicé el log de errores del servidor.", True, "analicé"),
    ("Inspeccioné el certificado SSL.", True, "inspeccioné"),
    ("El escaneo confirmó que no hay vulnerabilidades.", True, "escaneo confirmó (3rd person)"),
    ("Los tests pasaron sin ningún error.", True, "tests pasaron"),
    ("Compiló con 0 errores en el pipeline.", True, "compiló con 0"),
]

POSITIVE_EN = [
    ("I have verified the config is correct.", True, "have verified"),
    ("I checked the logs before deploying.", True, "checked"),
    ("I confirmed the port is open.", True, "confirmed"),
    ("I tested this on staging first.", True, "tested"),
    ("I ran the full suite twice.", True, "ran"),
    ("I validated the schema against prod data.", True, "validated"),
    ("I audited every dependency by hand.", True, "audited"),
    ("I scanned the repo for secrets.", True, "scanned"),
    ("I analyzed the crash dump.", True, "analyzed"),
    ("I inspected the certificate chain.", True, "inspected"),
    ("I executed the migration on staging.", True, "executed"),
]

NEGATION_NOT_CLAIM = [
    ("No he revisado el archivo todavía.", False, "no + he revisado"),
    ("Nunca ejecuté ese test en CI.", False, "nunca + ejecuté"),
    ("Tampoco verifiqué los logs de acceso.", False, "tampoco + verifiqué"),
    ("No verifiqué nada, solo asumí que estaba bien.", False, "no verifiqué, same clause"),
    ("Ninguna prueba fue ejecutada en este branch.", False, "ninguna + ejecutada"),
    ("I haven't verified this claim yet.", False, "haven't verified"),
    ("I didn't check the logs before merging.", False, "didn't check"),
    ("I never ran the tests on that branch.", False, "never ran"),
    ("I can't confirm the server is up.", False, "can't confirm"),
    ("I haven't tested this on Windows.", False, "haven't tested"),
]

NEGATION_CLAUSE_BOUNDARY_IS_CLAIM = [
    ("No, verifiqué el archivo y todo está bien.", True, "comma breaks negation scope"),
    ("No sé el resultado final, pero verifiqué los logs de acceso.", True, "pero=contrast"),
    ("No estoy seguro del todo, aunque confirmé el estado del servidor.", True, "aunque=contrast"),
    ("No, however I confirmed the deployment logs.", True, "however=contrast (EN)"),
]

QUESTIONS_NOT_CLAIM = [
    ("¿Crees que revisé el archivo antes de subirlo?", False, "rhetorical question"),
    ("¿Acaso verifiqué eso antes de responder?", False, "rhetorical question"),
    ("¿Alguien más confirmó este resultado?", False, "question, not about self"),
    ("¿Debería haber ejecutado los tests antes?", False, "hypothetical question"),
]

# Not gaps -- verified against real behavior, not assumed. detect_claims
# correctly returns False here, but for a narrower reason than "understands
# who the subject is": the verb patterns only match finite past/1st-person
# forms (he revisado, verifiqué), not infinitives or imperatives (revisar,
# verifica). A request to someone else happens to avoid those conjugations,
# so it's caught as a side effect, not by design. Kept as ordinary cases, not
# xfail -- an earlier version of this file mislabeled these as a false-
# positive gap without running them first; they aren't.
REQUESTS_TO_OTHERS_NOT_CLAIM = [
    ("¿Podrías revisar tú el archivo de configuración?", False, "infinitive form, not in verb pattern"),
    ("Por favor verifica los logs antes del deploy.", False, "imperative form, not in verb pattern"),
]

# Real gap, verified: English do-support ("I did check X") puts the claim verb
# in bare/base form, which none of the EN patterns match -- they only list
# past-participle forms (verified, checked, confirmed...). A dishonest turn
# phrased this way is NOT flagged as claimed=True, so if unverified it is
# wrongly approved rather than rejected. This is a false negative on the
# honesty-relevant side (the direction that matters), not cosmetic.
DO_SUPPORT_KNOWN_GAP = [
    ("I did check the deployment logs.", True, "do-support: bare verb form after 'did' unmatched"),
    ("I did verify the config before deploying.", True, "do-support: same gap, different verb"),
]

CITATION_WITHOUT_VERB_NOT_CLAIM = [
    ("Según la documentación [1], el puerto por defecto es 8000.", False, "citation present, no verification verb"),
    ("Como referencia, ver https://example.com/docs para más detalle.", False, "URL cited, no claim verb"),
]

MIXED_LANGUAGE_CLAIM = [
    ("Ya verifiqué el repo, all tests passed on my machine.", True, "code-switched ES/EN claim"),
]


ALL_CASES = (
    POSITIVE_ES
    + POSITIVE_EN
    + NEGATION_NOT_CLAIM
    + NEGATION_CLAUSE_BOUNDARY_IS_CLAIM
    + QUESTIONS_NOT_CLAIM
    + CITATION_WITHOUT_VERB_NOT_CLAIM
    + MIXED_LANGUAGE_CLAIM
    + REQUESTS_TO_OTHERS_NOT_CLAIM
)


@pytest.mark.parametrize("text,expected,note", ALL_CASES, ids=[c[2] for c in ALL_CASES])
def test_detect_claims_labeled_corpus(text: str, expected: bool, note: str) -> None:
    assert guard.detect_claims(text) is expected, f"text={text!r} note={note!r}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known gap: English do-support ('I did check X') puts the claim verb in "
        "bare form, which no EN pattern matches (they only list past-participle "
        "forms). A dishonest turn phrased this way is NOT detected as claimed=True, "
        "so unverified it is wrongly approved instead of rejected -- a false "
        "negative on the side that actually matters. strict=True so a real fix "
        "flips this to an unexpected pass and must be un-xfailed."
    ),
)
@pytest.mark.parametrize(
    "text,expected,note",
    DO_SUPPORT_KNOWN_GAP,
    ids=[c[2] for c in DO_SUPPORT_KNOWN_GAP],
)
def test_detect_claims_known_gap_do_support(text: str, expected: bool, note: str) -> None:
    assert guard.detect_claims(text) is expected, f"text={text!r} note={note!r}"
