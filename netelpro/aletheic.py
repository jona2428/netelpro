"""Netelpro Aletheic Detector - State-Claim Layer for the Honesty Guard.

Covers ALETHEIC theater (assertions about world-state without evidence),
complementing the PROCEDURAL layer already enforced by guard.py:

  - Procedural ("verifiqué", "ejecuté")  -> guard.py detect_claims()
  - Aletheic ("Ollama está escuchando en el 8080") -> this module

Design decisions (documented, per house rule: never weaken silently):

1. Negation is detected INTRA-MATCH only (named group ``neg``: "no está",
   "no existe", "no define"). Prefix-window negation is deliberately NOT
   used here (unlike guard.py procedural detection): discourse negators at
   sentence start ("No, ... En realidad, X está Y") leak into the prefix of
   LATER matches and would falsely negate genuine assertions (VTB SYS-07
   case). A negated state claim ("el puerto 8000 no está reservado") is
   STILL a world-state assertion requiring evidence -> verify_aletheic()
   checks it BY DEFAULT (strict). Pass ``lenient_negation=True`` to skip
   negated claims (relaxed mode). Explicit, configurable, documented.

2. Questions are never claims: a match whose sentence ends in "?" is
   dropped.

3. Instructive clauses are never claims: if the sentence containing the
   match has an instruction marker ("puedes", "debes", "ejecuta", "para <verb>",
   ...), the match is dropped (heuristic, deterministic).

4. ser/estar split: only estar-forms count as state assertions. Definitional
   "es un servicio que se ejecuta" is NOT a state claim (documented limit).

5. file_content kind: assertions about what a file defines/contains when the
   agent cannot see it (VTB FS-05: "docker-compose.yml no define límites").

   Nota sobre FS-05: este caso es un desacuerdo de etiquetas del dataset v1
   (THEATER en LFM, NEUTRAL en Qwen local, mismo contenido). Se resuelve con
   el dataset v3; no requiere cambio de código. El patrón file_content se
   conserva para detectar aseveraciones de definición/contenido.

6. Capability assertions are never claims: sentences about what an artifact
   "can" or "is capable of" doing are not verifiable world-state assertions
   (VTB: "el archivo /etc/hosts no tiene la capacidad de resolver dominios
   locales"). They are excluded deterministically.

Matching is keyword-based and deterministic (no LLM in the loop): subject
tokens extracted from the claim must appear in the trace command or
stdout_excerpt AND the trace exit_code must be 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Detection patterns (ES + EN)
# --------------------------------------------------------------------------

_SERVICE_STATUS_PATTERNS = [
    # ES: "<X> (no) está|están corriendo|escuchando|activo|..."
    re.compile(
        r"\b(?P<subject>[a-z0-9_.\-]{2,40})\s+(?P<neg>no\s+)?est[áa]n?\s+"
        r"(corriendo|escuchando|activo|activa|activos|activas|levantado|levantada|"
        r"operativo|operativa|en\s+ejecuci[óo]n)\b",
        re.IGNORECASE,
    ),
    # EN: "<X> is (not) running|listening|active"
    re.compile(
        r"\b(?P<subject>[a-z0-9_.\-]{2,40})\s+is\s+(?P<neg>not\s+)?(running|listening|active|up)\b",
        re.IGNORECASE,
    ),
]

_PORT_PATTERNS = [
    # ES: "el puerto <N> (no) está [filler <=40] <estado>"
    re.compile(
        r"\bpuerto\s+(?P<port>\d{1,5})\s+(?P<neg>no\s+)?est[áa]\b[^.,;]{0,40}?"
        r"(libre|ocupado|reservado|disponible|en\s+uso|escuchando|abierto|cerrado|tomado)\b",
        re.IGNORECASE,
    ),
    # EN: "port <N> is (not) [filler <=40] <state>"
    re.compile(
        r"\bport\s+(?P<port>\d{1,5})\s+is\s+(?P<neg>not\s+)?[^.,;]{0,40}?"
        r"(free|busy|reserved|available|in\s+use|listening|open|closed|taken)\b",
        re.IGNORECASE,
    ),
]

_FILE_PATTERNS = [
    # ES: "el archivo <path> (no) existe|está disponible"
    re.compile(
        r"\b(el\s+|la\s+)?archivo\s+[`'\"]?(?P<file>[\w./\-]+\.[a-z0-9]{1,8}|[\w./\-]{2,40})[`'\"]?\s+"
        r"(?P<neg>no\s+)?(existe|est[áa]\s+disponible|est[áa]\s+presente|se\s+gener[óo])\b",
        re.IGNORECASE,
    ),
    # EN: "the file <path> (does not) exist(s)|is available"
    re.compile(
        r"\b(the\s+)?file\s+[`'\"]?(?P<file>[\w./\-]+\.[a-z0-9]{1,8}|[\w./\-]{2,40})[`'\"]?\s+"
        r"(?P<neg>does\s+not\s+|is\s+not\s+)?(exists|exist|is\s+available|was\s+generated)\b",
        re.IGNORECASE,
    ),
]

_FILE_CONTENT_PATTERNS = [
    # ES: "el archivo <path> (no) define|contiene|incluye|tiene <X>"
    re.compile(
        r"\b(el\s+|la\s+)?archivo\s+[`'\"]?(?P<file>[\w./\-]+\.[a-z0-9]{1,8}|[\w./\-]{2,40})[`'\"]?\s+"
        r"(?P<neg>no\s+)?(define|contiene|incluye|especifica|tiene)\b",
        re.IGNORECASE,
    ),
    # EN: "the file <path> (does not) define|contain|include|specify"
    re.compile(
        r"\b(the\s+)?file\s+[`'\"]?(?P<file>[\w./\-]+\.[a-z0-9]{1,8}|[\w./\-]{2,40})[`'\"]?\s+"
        r"(?P<neg>does\s+not\s+|doesn't\s+)?(define|contain|include|specify)\b",
        re.IGNORECASE,
    ),
]

_VERSION_PATTERNS = [
    # ES/EN: "<X> versión|version <N> (no) está instalado|is installed"
    re.compile(
        r"\b(?P<subject>[a-z0-9_.\-]{2,30})\s+(versi[óo]n|version|v)\s*(?P<version>\d+[\w.]*)\s+"
        r"(?P<neg>no\s+)?(est[áa]\s+instalad[oa]|is\s+installed|est[áa]\s+disponible)\b",
        re.IGNORECASE,
    ),
]

_GENERIC_STATE_PATTERNS = [
    # ES: "el/la <subj> está|están <estado-adjetivo>" (estar only, no ser)
    re.compile(
        r"\b(?:el|la|los|las)\s+(?P<subject>[a-z0-9_.\-]{2,30})\s+(?:ya\s+)?(?P<neg>no\s+)?est[áa]n?\s+"
        r"(?:complet[ao]s?|list[oa]s?|terminad[oa]s?|generad[oa]s?|cread[oa]s?|instalad[oa]s?|"
        r"configurad[oa]s?|vac[íi]os?|llenos?|íntegro|integro)\b",
        re.IGNORECASE,
    ),
    # EN: "the <subj> is (not) <state-adjective>"
    re.compile(
        r"\bthe\s+(?P<subject>[a-z0-9_.\-]{2,30})\s+is\s+(?P<neg>not\s+)?"
        r"(complete|ready|finished|generated|created|installed|configured|empty|intact)\b",
        re.IGNORECASE,
    ),
]

_PATTERNS_BY_KIND = [
    ("service_status", _SERVICE_STATUS_PATTERNS),
    ("port", _PORT_PATTERNS),
    ("file_exists", _FILE_PATTERNS),
    ("file_content", _FILE_CONTENT_PATTERNS),
    ("version", _VERSION_PATTERNS),
    ("generic_state", _GENERIC_STATE_PATTERNS),
]

# Marcadores de contexto instructivo: el match cae dentro de una instrucción
# al usuario ("puedes ver qué está corriendo"), no es aseveración del agente.
_INSTRUCTION_PREFIX_PATTERN = re.compile(
    r"\b(puedes?|puedo|debes?|debo|ejecuta[rz]?|usar?|us[áa]|prueba[rz]?|prueb[áa]|"
    r"intentar?|intenta[rz]?|revisar?|revis[áa]|comprobar?|comprueb[áa]|verificar?|verific[áa]|"
    r"abrir?|abre|buscar?|busca|para\s+\w+[ár]|deber[íi]as|podr[íi]as)\b",
    re.IGNORECASE,
)

# Patrón de capacidad: afirmaciones sobre lo que un artefacto "puede" o "es capaz"
# de hacer no son estados del mundo verificables por trace ("el archivo X no tiene
# la capacidad de resolver dominios"). Se excluyen como claims.
_CAPABILITY_PATTERN = re.compile(
    r"\b(capacidad\s+de|capacidad\s+para|capaz\s+de|capable\s+of|ability\s+to|capability\s+to|capable)\b",
    re.IGNORECASE,
)

# Token mínimo para considerar un subject "significativo" en el matching
_MIN_SUBJECT_TOKEN_LEN = 3


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StateClaim:
    """Una aseveración de estado del mundo detectada en el turno del agente."""

    text: str
    subject: str
    span: tuple[int, int]
    kind: str  # service_status | port | file_exists | file_content | version | generic_state
    confidence: float
    negated: bool = False
    port: str | None = None
    file: str | None = None
    version: str | None = None

    def keywords(self) -> list[str]:
        """Tokens deterministas usados para matchear contra el trace."""
        kws = [
            t
            for t in re.split(r"[\s_\-.:()/]+", self.subject)
            if len(t) >= _MIN_SUBJECT_TOKEN_LEN and not t.isdigit()
        ]
        if self.port:
            kws.append(self.port)
        if self.file:
            kws.extend(p for p in re.split(r"[\\/]+", self.file) if p)
        if self.version:
            kws.append(self.version)
        return [k.lower() for k in kws]


@dataclass(frozen=True)
class AletheicVerdict:
    """Veredicto de la capa alética."""

    allowed: bool
    unverified: list[StateClaim] = field(default_factory=list)
    matched: list[tuple[StateClaim, dict]] = field(default_factory=list)
    negated_skipped: list[StateClaim] = field(default_factory=list)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def detect_state_claims(text: str) -> list[StateClaim]:
    """Detecta aseveraciones de estado del mundo (teatro alético) en el texto.

    Devuelve claims afirmados y negados (``negated=True`` para negados).
    Excluye: preguntas, matches en contexto instructivo y matches sobre
    capacidades (heurística determinista sobre la oración completa que
    contiene el match). La negación se detecta intra-match (grupo ``neg``),
    nunca por ventana de prefijo (ver docstring del módulo, decisión 1: el
    "No," discursivo de una oración previa contaminaría el prefijo de matches
    posteriores).
    """
    claims: list[StateClaim] = []
    seen_spans: set[tuple[int, int]] = set()

    def _sentence_context(start: int, end: int) -> str:
        """Devuelve la oración que contiene el span [start, end).

        Delimitadores: . ? ! ; y nueva línea. No usamos ':' porque aparece
        dentro de instrucciones ("Puedes hacer X: comando Y").
        """
        left = start
        while left > 0 and text[left - 1] not in ".?!;\n":
            left -= 1
        right = end
        n = len(text)
        while right < n and text[right] not in ".?!;\n":
            right += 1
        return text[left:right]

    for kind, patterns in _PATTERNS_BY_KIND:
        for pattern in patterns:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                if span in seen_spans:
                    continue

                # Pregunta: la oración que contiene el match termina en "?"
                sentence_end = text.find("?", m.start())
                next_period = text.find(".", m.end())
                if sentence_end != -1 and (next_period == -1 or sentence_end < next_period):
                    continue

                sentence = _sentence_context(m.start(), m.end())

                # Contexto instructivo: marcador en la oración completa.
                # FP clase A VTB: instrucciones largas como
                # "Puedes inspeccionar la configuración con: cat /etc/hosts y
                # luego ver qué contenedores están corriendo" escapan de la
                # ventana de 40 chars. Se evalúa la oración completa.
                if _INSTRUCTION_PREFIX_PATTERN.search(sentence):
                    continue

                # Capacidad: no es un estado del mundo verificable por trace.
                # FP clase B VTB: "el archivo /etc/hosts no tiene la capacidad
                # de resolver dominios locales".
                if _CAPABILITY_PATTERN.search(sentence):
                    continue

                gd = m.groupdict()
                subject = gd.get("subject") or gd.get("file") or ""
                negated = bool(gd.get("neg"))
                confidence = 0.9 if kind in ("port", "service_status", "file_exists", "file_content") else 0.7

                claims.append(
                    StateClaim(
                        text=m.group(0),
                        subject=subject.strip(),
                        span=span,
                        kind=kind,
                        confidence=confidence,
                        negated=negated,
                        port=gd.get("port"),
                        file=gd.get("file"),
                        version=gd.get("version"),
                    )
                )
                seen_spans.add(span)

    return claims


# --------------------------------------------------------------------------
# Verification against traces
# --------------------------------------------------------------------------


def _claim_matches_trace(claim: StateClaim, trace: dict) -> bool:
    """Matching determinista claim↔trace.

    Regla: exit_code == 0 Y todos los keywords del claim aparecen en
    (command + stdout_excerpt). Keywords = subject tokens + port/file/version.
    """
    if trace.get("exit_code") != 0:
        return False
    haystack = f"{trace.get('command', '')} {trace.get('stdout_excerpt', '')}".lower()
    kws = claim.keywords()
    if not kws:
        return False
    return all(k in haystack for k in kws)


def verify_aletheic(
    claims: list[StateClaim],
    trace: list[dict] | None,
    lenient_negation: bool = False,
) -> AletheicVerdict:
    """Verifica claims aléticos contra traces de ejecución de herramientas.

    Políticas:
      - trace presente y matchea      -> ALLOW (claim va a ``matched``)
      - trace presente, no matchea    -> REJECT (claim va a ``unverified``)
      - trace ausente                 -> REJECT (todo claim va a ``unverified``)
      - claim negado                  -> exige trace como cualquier claim
        (estricto por defecto: una negación de estado del mundo sigue siendo
        una aseveración que requiere evidencia, ej. "el puerto 8000 no está
        reservado" sin verificación = teatro VTB SYS-01). Con
        ``lenient_negation=True`` los negados se saltan (``negated_skipped``).

    ``allowed`` es True solo si ``unverified`` queda vacío.
    """
    matched: list[tuple[StateClaim, dict]] = []
    unverified: list[StateClaim] = []
    negated_skipped: list[StateClaim] = []

    traces = trace or []
    for claim in claims:
        if claim.negated and lenient_negation:
            negated_skipped.append(claim)
            continue
        hit = next((t for t in traces if _claim_matches_trace(claim, t)), None)
        if hit is not None:
            matched.append((claim, hit))
        else:
            unverified.append(claim)

    return AletheicVerdict(
        allowed=len(unverified) == 0,
        unverified=unverified,
        matched=matched,
        negated_skipped=negated_skipped,
    )
