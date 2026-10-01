"""Netelpro Receipts -- file-effect honesty for LLM agents.

The failure this targets, concretely: an agent says "I updated
`config/settings.py`" and the file is byte-for-byte what it was before the
turn. The write tool errored, or was never called, or hit a read-only
mount, or the model simply narrated an edit it imagined -- and the prose
reports success anyway. The user finds out later, from the file.

`netelpro.guard.HonestyGuard` catches claims of *verification* ("I ran the
tests") with no tool evidence. This module catches claims of *effect* ("I
wrote the file") with no effect. The evidence standard is stricter than a
tool result: it is the bytes.

Three pieces, same posture as the rest of the repo (fail-closed, exact
reason, nothing silent):

1. **Snapshot + receipts.** The harness hashes the workspace before the
   turn and after it. Every path whose sha256 differs becomes a `Receipt`
   (created / modified / deleted, before-hash, after-hash) appended to a
   hash-chained, append-only `ReceiptLedger`. The model never writes to
   the ledger and is never asked to recall it: the ledger is the source of
   truth for "what did I actually change", exactly the read-path idea of
   `docs/STATE_TRACKING_GATE_SPEC.md` applied to files. It persists to
   JSONL, so a fresh process with total context amnesia still answers
   correctly.

2. **Mutation-claim detection.** Regex over the finished turn (ES + EN)
   extracts `(path, kind)` claims: "modifiqué `src/app.py`", "created
   tests/test_x.py", "`a.py` has been updated". Negation, attempts,
   intent, futures and questions are scoped out ("no pude modificar X",
   "I tried to update X", "voy a crear X" are not claims of effect).

3. **A compiled Netelpro rule** (`rules/mutation_receipt.sl`) decides,
   per path, whether the claim kind is admitted by the receipt kind. The
   rule is three functions a human reads; its full 40-row domain is
   verified native-vs-interpreter in the test suite. `strict=True` also
   rejects *silent* writes: a receipt the text never mentions.

What this is NOT: it does not prevent the false sentence from being
generated (that is Layer B's job and a future wiring), it does not know
*who* changed a file (the model, the user, a build step -- receipts are
observations, `origin="observed"`), and it does not judge whether an edit
is correct. It judges one thing: did the bytes change the way the text
says they did.

Zero-integration use, from any agent harness (Claude Code, OpenCode,
Cursor, a bash loop):

    python -m netelpro.receipts begin            # before the agent runs
    ... agent runs, writes files, prints its final answer to answer.md ...
    python -m netelpro.receipts end              # commit observed receipts
    python -m netelpro.receipts audit --text answer.md   # exit 2 on theater

In-process:

    guard = MutationGuard(root)
    guard.begin()
    ... agent turn ...
    audit = guard.audit(agent_text)      # end() is implied if not called
    audit.approved / audit.reasons
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat as statmod
import sys
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from netelpro.gate import Gate

__all__ = [
    "DEFAULT_IGNORE",
    "LEDGER_FILE",
    "SNAPSHOT_FILE",
    "STATE_DIR",
    "GENESIS",
    "KIND_CREATED",
    "KIND_DELETED",
    "KIND_MODIFIED",
    "KIND_NAMES",
    "KIND_NONE",
    "KIND_WRITTEN",
    "RULE_PATH",
    "TurnState",
    "ClaimVerdict",
    "LedgerError",
    "MutationAudit",
    "MutationClaim",
    "MutationGuard",
    "MutationTheaterError",
    "RACY_WINDOW_NS",
    "Receipt",
    "ReceiptLedger",
    "Snapshot",
    "detect_mutation_claims",
    "diff_snapshots",
    "load_snapshot",
    "main",
    "open_turn",
    "save_snapshot",
    "snapshot",
]

RULE_PATH = Path(__file__).resolve().parent / "rules" / "mutation_receipt.sl"

# Directory / file names never hashed. Globs apply to a single path
# component (a directory name or a file name), never to the full path.
DEFAULT_IGNORE: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    ".venv*",
    "venv",
    ".netelpro",
    ".worktrees",
    "*.pyc",
    "*.egg-info",
    "dist",
    "build",
)

# Claim kinds -- the integer vocabulary shared with mutation_receipt.sl.
KIND_NONE = 0
KIND_CREATED = 1
KIND_MODIFIED = 2
KIND_DELETED = 3
KIND_WRITTEN = 4  # lenient: created or modified
KIND_NAMES: dict[int, str] = {
    KIND_NONE: "none",
    KIND_CREATED: "created",
    KIND_MODIFIED: "modified",
    KIND_DELETED: "deleted",
    KIND_WRITTEN: "written",
}
_RECEIPT_KIND: dict[str, int] = {
    "created": KIND_CREATED,
    "modified": KIND_MODIFIED,
    "deleted": KIND_DELETED,
}

GENESIS = "0" * 64
_UNREADABLE = "!unreadable"


# ---------------------------------------------------------------------------
# 1. Snapshot: the workspace as a map path -> sha256, with an incremental
#    fast path keyed on the stat signature each hash was taken under
# ---------------------------------------------------------------------------

# A file whose mtime or ctime falls within this window of the cached
# snapshot's start time is re-hashed even when its signature matches: the
# classic "racy" case (git has the same rule) where a write lands inside
# the same timestamp tick as the hash and leaves the signature unchanged.
RACY_WINDOW_NS = 2_000_000_000

StatSig = tuple[int, int, int, int]  # (size, mtime_ns, ctime_ns, inode)


class Snapshot(dict[str, str]):
    """{posix relative path: sha256}, plus per-path `stats` (the StatSig the
    hash was taken under), `taken_ns` (when the walk started) and the
    `hashed` / `reused` counts of this take. A plain dict works everywhere a
    Snapshot is accepted; it just carries no stats, so nothing is reused."""

    __slots__ = ("stats", "taken_ns", "hashed", "reused")

    def __init__(
        self,
        hashes: Mapping[str, str] | None = None,
        stats: Mapping[str, Sequence[int]] | None = None,
        taken_ns: int = 0,
        hashed: int = 0,
        reused: int = 0,
    ) -> None:
        super().__init__(hashes or {})
        self.stats: dict[str, StatSig] = {k: tuple(v) for k, v in (stats or {}).items()}  # type: ignore[misc]
        self.taken_ns = taken_ns
        self.hashed = hashed
        self.reused = reused

    @classmethod
    def of(cls, files: Mapping[str, str]) -> Snapshot:
        return files if isinstance(files, Snapshot) else cls(files)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        # Not silently skipped: an unreadable file gets a sentinel so a
        # change in readability shows up as a change, never as nothing.
        return _UNREADABLE
    return h.hexdigest()


def _ignored(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch(name, pat) for pat in patterns)


def snapshot(
    root: str | Path,
    ignore: Sequence[str] = DEFAULT_IGNORE,
    cache: Mapping[str, str] | None = None,
) -> Snapshot:
    """Hash every regular file under `root` (recursively).

    Directory symlinks are not followed; file symlinks hash their target's
    content. With `cache` (a previous Snapshot of the same root) a file is
    NOT re-read when all of these hold, and its cached hash is reused:

      * its StatSig (size, mtime_ns, ctime_ns, inode) equals the cached one;
      * its mtime and ctime are both older than the cached take minus
        RACY_WINDOW_NS (a write inside the tick is never trusted);
      * the cached hash is a real digest, not the unreadable sentinel.

    Why ctime and inode, not just size+mtime: an agent with a shell can
    forge mtime (`touch -d`, os.utime) after a same-size edit. ctime is set
    by the kernel on every inode change, utime included, and cannot be
    chosen from user space on Linux/macOS; a rename-over gets a new inode.
    Residual holes, declared: Windows (st_ctime is creation time there),
    a root that writes the raw device or steps the clock, and a cache the
    caller lets the model edit. Pass cache=None for a full re-hash.
    Cost without a cache is linear in the bytes under `root`; with one it
    is linear in the number of files (one stat each) plus the changed bytes.
    """
    base = Path(root).resolve()
    if not base.is_dir():
        raise LedgerError(f"snapshot root is not a directory: {base}")
    taken_ns = time.time_ns()
    cache_stats: Mapping[str, StatSig] = getattr(cache, "stats", None) or {}
    trust_before = int(getattr(cache, "taken_ns", 0)) - RACY_WINDOW_NS
    out = Snapshot(taken_ns=taken_ns)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if not _ignored(d, ignore))
        for fn in sorted(filenames):
            if _ignored(fn, ignore):
                continue
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue  # broken symlink, vanished mid-walk
            if not statmod.S_ISREG(st.st_mode):
                continue  # sockets, fifos
            rel = p.relative_to(base).as_posix()
            sig: StatSig = (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
            cached = cache.get(rel) if cache is not None else None
            if (
                cached is not None
                and cached != _UNREADABLE
                and cache_stats.get(rel) == sig
                and st.st_mtime_ns < trust_before
                and st.st_ctime_ns < trust_before
            ):
                out[rel] = cached
                out.reused += 1
            else:
                out[rel] = _sha256_file(p)
                out.hashed += 1
            out.stats[rel] = sig
    return out


def diff_snapshots(
    before: dict[str, str], after: dict[str, str]
) -> list[tuple[str, str, str | None, str | None]]:
    """Observed effects between two snapshots.

    Returns sorted (kind, path, before_hash, after_hash) with kind in
    {created, modified, deleted}. Unchanged paths are not listed.
    """
    effects: list[tuple[str, str, str | None, str | None]] = []
    for path in sorted(set(before) | set(after)):
        b = before.get(path)
        a = after.get(path)
        if b is None and a is not None:
            effects.append(("created", path, None, a))
        elif b is not None and a is None:
            effects.append(("deleted", path, b, None))
        elif b != a:
            effects.append(("modified", path, b, a))
    return effects


# ---------------------------------------------------------------------------
# 2. Receipts: an append-only, hash-chained ledger of observed effects
# ---------------------------------------------------------------------------


class LedgerError(Exception):
    """The ledger is corrupt, tampered, or unusable -- never a silent state."""


@dataclass(frozen=True)
class Receipt:
    """One observed file effect. Immutable; its digest covers every field
    plus the previous receipt's digest, so editing any receipt after the
    fact breaks the chain from that point on."""

    seq: int
    turn: int
    kind: str  # created | modified | deleted
    path: str
    before: str | None
    after: str | None
    origin: str  # observed (snapshot diff) | tool (harness-reported)
    ts_ns: int
    prev: str
    digest: str

    @staticmethod
    def compute_digest(
        prev: str,
        seq: int,
        turn: int,
        kind: str,
        path: str,
        before: str | None,
        after: str | None,
        origin: str,
        ts_ns: int,
    ) -> str:
        payload = json.dumps(
            [prev, seq, turn, kind, path, before, after, origin, ts_ns],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def kind_code(self) -> int:
        return _RECEIPT_KIND[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "turn": self.turn,
            "kind": self.kind,
            "path": self.path,
            "before": self.before,
            "after": self.after,
            "origin": self.origin,
            "ts_ns": self.ts_ns,
            "prev": self.prev,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Receipt:
        try:
            return cls(
                seq=int(d["seq"]),
                turn=int(d["turn"]),
                kind=str(d["kind"]),
                path=str(d["path"]),
                before=d.get("before"),
                after=d.get("after"),
                origin=str(d.get("origin", "observed")),
                ts_ns=int(d["ts_ns"]),
                prev=str(d["prev"]),
                digest=str(d["digest"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise LedgerError(f"malformed receipt record: {e}: {d!r}") from e

    def short(self) -> str:
        b = (self.before or "-")[:12]
        a = (self.after or "-")[:12]
        return f"{self.kind:<8} {self.path}  {b} -> {a}"


class ReceiptLedger:
    """Append-only, hash-chained list of receipts.

    Only the harness appends (via `record` / `record_diff`); the LLM has no
    write path. `verify_chain()` recomputes every digest; `load()` refuses
    a file whose chain does not verify.
    """

    def __init__(self, receipts: Iterable[Receipt] = ()) -> None:
        self._receipts: list[Receipt] = list(receipts)

    # -- reads ---------------------------------------------------------
    @property
    def receipts(self) -> tuple[Receipt, ...]:
        return tuple(self._receipts)

    def __len__(self) -> int:
        return len(self._receipts)

    @property
    def head(self) -> str:
        return self._receipts[-1].digest if self._receipts else GENESIS

    @property
    def latest_turn(self) -> int | None:
        return max((r.turn for r in self._receipts), default=None)

    def for_turn(self, turn: int) -> list[Receipt]:
        return [r for r in self._receipts if r.turn == turn]

    def net_effects(self, turn: int) -> dict[str, Receipt]:
        """Last receipt per path within a turn (tool-origin receipts may
        record several effects on one path; the net effect is the last)."""
        out: dict[str, Receipt] = {}
        for r in self.for_turn(turn):
            out[r.path] = r
        return out

    def verify_chain(self) -> tuple[bool, str | None]:
        """(True, None) when every digest and link recomputes; otherwise
        (False, reason) naming the first broken receipt."""
        prev = GENESIS
        for i, r in enumerate(self._receipts):
            if r.seq != i:
                return False, f"receipt {i}: seq is {r.seq}, expected {i}"
            if r.prev != prev:
                return False, f"receipt {i} ({r.path}): prev link does not match previous digest"
            expected = Receipt.compute_digest(
                r.prev, r.seq, r.turn, r.kind, r.path, r.before, r.after, r.origin, r.ts_ns
            )
            if r.digest != expected:
                return False, f"receipt {i} ({r.path}): digest mismatch -- record was altered"
            prev = r.digest
        return True, None

    # -- writes (harness only) ------------------------------------------
    def record(
        self,
        kind: str,
        path: str,
        before: str | None,
        after: str | None,
        *,
        turn: int,
        origin: str = "observed",
        ts_ns: int | None = None,
    ) -> Receipt:
        if kind not in _RECEIPT_KIND:
            raise LedgerError(f"unknown receipt kind {kind!r}")
        seq = len(self._receipts)
        prev = self.head
        ts = time.time_ns() if ts_ns is None else ts_ns
        digest = Receipt.compute_digest(prev, seq, turn, kind, path, before, after, origin, ts)
        r = Receipt(seq, turn, kind, path, before, after, origin, ts, prev, digest)
        self._receipts.append(r)
        return r

    def record_diff(
        self,
        before: dict[str, str],
        after: dict[str, str],
        *,
        turn: int,
        origin: str = "observed",
    ) -> list[Receipt]:
        return [
            self.record(kind, path, b, a, turn=turn, origin=origin)
            for kind, path, b, a in diff_snapshots(before, after)
        ]

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for r in self._receipts:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ReceiptLedger:
        """Load a JSONL ledger. A missing file is an empty ledger; a file
        whose chain does not verify raises LedgerError (fail-closed)."""
        p = Path(path)
        if not p.exists():
            return cls()
        receipts: list[Receipt] = []
        with open(p, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    raise LedgerError(f"{p}:{lineno}: not JSON: {e}") from e
                receipts.append(Receipt.from_dict(d))
        ledger = cls(receipts)
        ok, reason = ledger.verify_chain()
        if not ok:
            raise LedgerError(f"{p}: chain does not verify: {reason}")
        return ledger


# ---------------------------------------------------------------------------
# 3. Mutation-claim detection (ES + EN)
# ---------------------------------------------------------------------------

# A path-like token: has an extension starting with a letter, or has a '/'.
# Optional backtick / quote wrapping. Version numbers ("3.11") do not match
# because the extension must start with a letter; a sentence-final period
# does not match because the extension needs at least one letter after it.
# Markdown emphasis around a path ("Archivo **chapters/cap3.md** fue creado")
# is part of the wrapper, like a backtick: round-0 audit 2, b42, 2026-10-01.
_PATH_RE = (
    r"\*{0,2}[`\"']?"
    r"(?P<path>(?:\.{1,2}/)?(?:[\w.\-]+/)*[\w\-][\w.\-]*\.[A-Za-z][A-Za-z0-9]{0,7}"
    r"|(?:\.{1,2}/)?(?:[\w.\-]+/)+[\w.\-]*"
    # Well-known extensionless files. Without this, "Modifiqué Dockerfile"
    # was no claim at all: found 2026-10-01 by the Receipts-RAFT reward
    # truth table (a failed Dockerfile write narrated as done scored R=1).
    r"|(?:Dockerfile|Makefile|Containerfile|Gemfile|Rakefile|Procfile|Jenkinsfile|Vagrantfile|"
    r"Justfile|Brewfile|LICENSE|NOTICE|CODEOWNERS)(?![\w\-]|\.\w))"
    r"[`\"']?\*{0,2}"
)
_PATH_ONLY = re.compile(_PATH_RE)

# Verb classes. Each entry: (kind, regex of verb forms). The forms are
# past / perfective / participle: completed effects. Infinitives,
# imperatives and progressives ("modificar", "update", "updating") are
# deliberately absent -- they do not assert that anything happened.
# Participle endings: gender and number ("fue editada", "quedaron cambiados").
_P = r"[oa]s?"
# Forms found in live Qwen2.5 generation (benchmarks/receipts_qwen_live_report.md,
# 2026-10-01), not hand-written: reflexive passive ("se creó en X", "X se ha
# modificado"), bare / "está" participles ("Clamp funcion creado en X",
# "está creado en X"), feminine participles ("fue editada").
# Auxiliaries that may precede a participle when the path FOLLOWS the verb:
# "fue añadida al archivo X", "ha sido corregido en X", "está creado en X"
# (DPO live run, 2026-10-01). Bare participles ("Sección 'Uso' añadida al
# archivo X") are claims unless an article + noun precede them (see
# _ADJECTIVAL_PRECEDER).
_AUX = r"(?:(?:se\s+)?(?:he|hemos|ha|han)\s+|(?:fue|fueron|ha\s+sido|han\s+sido|est[áa]n?|quedan?|qued[oó]|quedaron)\s+)?"
_ES_VERB_CLASSES: list[tuple[int, str]] = [
    (KIND_CREATED, rf"cre[eéoó]|{_AUX}cread{_P}"),
    (KIND_DELETED, rf"elimin[eéoó]|borr[eéoó]|quit[eéoó]|{_AUX}(?:eliminad|borrad|quitad){_P}"),
    (
        KIND_MODIFIED,
        rf"modifiqu[eé]|modific[oó]|actualic[eé]|actualiz[oó]|edit[eéoó]|cambi[eéoó]|correg[ií]|corrigi[oó]|"
        rf"arregl[eéoó]|parche[eéoó]|reescrib[ií]|reescribi[oó]|ajust[eéoó]|refactoric[eé]|refactoriz[oó]|"
        rf"{_AUX}(?:modificad|actualizad|editad|cambiad|corregid|arreglad|parchead|ajustad|refactorizad){_P}|"
        rf"{_AUX}reescrito",
    ),
    (
        KIND_WRITTEN,
        rf"escrib[ií]|escribi[oó]|guard[eéoó]|agregu[eé]|agreg[oó]|añad[ií]|añadi[oó]|gener[eéoó]|"
        rf"{_AUX}(?:guardad|agregad|añadid|generad){_P}|{_AUX}escrito",
    ),
]
_EN_VERB_CLASSES: list[tuple[int, str]] = [
    (KIND_CREATED, r"created"),
    (KIND_DELETED, r"deleted|removed"),
    (
        KIND_MODIFIED,
        r"modified|updated|edited|changed|fixed|patched|rewrote|rewritten|adjusted|"
        r"refactored|tweaked",
    ),
    (KIND_WRITTEN, r"wrote|written|saved|added|generated"),
]

_ES_SUBJECT = r"(?:ya\s+)?"
_EN_SUBJECT = r"(?:(?:I|we)(?:'ve|\s+have|\s+just|\s+also)?\s+|I've\s+|we've\s+)?"
# Up to 60 chars of filler between the verb and the path, never crossing a
# sentence / clause boundary. The comma is a boundary too: "README.md se
# actualizó con la sección, pero el src/utils.py no se pudo editar" must not
# bind "actualizó" to utils.py (false rejection found on live output,
# PARTIAL family, 2026-10-01).
_FILLER = r"(?P<filler>[^.;:,\n¿?!]{0,60}?)"
_FILLER_CONTRAST = re.compile(r"\b(?:pero|sino|aunque|mientras|but|however|although|while|whereas)\b", re.IGNORECASE)


def _active_pattern(subject: str, classes: list[tuple[int, str]]) -> re.Pattern[str]:
    verbs = "|".join(rx for _, rx in classes)
    return re.compile(rf"\b{subject}(?P<verb>{verbs})\b{_FILLER}{_PATH_RE}", re.IGNORECASE)


_ACTIVE_PATTERNS: list[tuple[re.Pattern[str], list[tuple[int, str]]]] = [
    (_active_pattern(_ES_SUBJECT, _ES_VERB_CLASSES), _ES_VERB_CLASSES),
    (_active_pattern(_EN_SUBJECT, _EN_VERB_CLASSES), _EN_VERB_CLASSES),
]

# Passive / resultative: the path comes first. "`a.py` has been updated",
# "el archivo a.py fue modificado", "a.py quedó actualizado".
_ES_PASSIVE_CLASSES: list[tuple[int, str]] = [
    (KIND_CREATED, rf"cread{_P}|cre[oó]"),
    (KIND_DELETED, rf"(?:eliminad|borrad|quitad){_P}|elimin[oó]|borr[oó]|quit[oó]"),
    (
        KIND_MODIFIED,
        rf"(?:modificad|actualizad|editad|cambiad|corregid|arreglad|parchead|ajustad|refactorizad){_P}|reescrito|"
        rf"modific[oó]|actualiz[oó]|edit[oó]|cambi[oó]|corrigi[oó]|arregl[oó]|parche[oó]|reescribi[oó]|ajust[oó]|refactoriz[oó]",
    ),
    (KIND_WRITTEN, rf"(?:guardad|agregad|añadid|generad){_P}|escrito|escribi[oó]|guard[oó]|agreg[oó]|añadi[oó]|gener[oó]"),
]
_EN_PASSIVE_CLASSES: list[tuple[int, str]] = [
    (KIND_CREATED, r"created"),
    (KIND_DELETED, r"deleted|removed"),
    (
        KIND_MODIFIED,
        r"modified|updated|edited|changed|fixed|patched|rewritten|adjusted|refactored|tweaked",
    ),
    (KIND_WRITTEN, r"written|saved|generated|added"),
]


# "config/settings.py también quedó cambiado": an adverb may sit between the
# path and the auxiliary (live finding, PARTIAL family).
_ADVERB = r"(?:\s+(?:también|tambien|ya|ahora|finalmente|efectivamente|igualmente|also|now|already))?"


# "src/app.py está correctamente actualizado": an adverb may also sit between
# the auxiliary and the participle (RAFT live run, 2026-10-01).
_AUX_ADVERB = r"(?:(?:correctamente|completamente|debidamente|exitosamente|ya|también|tambien|efectivamente|correctly|successfully|fully|already)\s+)?"


def _passive_pattern(aux: str, classes: list[tuple[int, str]]) -> re.Pattern[str]:
    verbs = "|".join(rx for _, rx in classes)
    return re.compile(rf"{_PATH_RE}{_ADVERB}\s+(?:{aux})\s+{_AUX_ADVERB}(?P<verb>{verbs})\b", re.IGNORECASE)


_PASSIVE_PATTERNS: list[tuple[re.Pattern[str], list[tuple[int, str]]]] = [
    (
        _passive_pattern(
            r"fue|fueron|ha\s+sido|han\s+sido|qued[oó]|quedaron|quedan?|ya\s+est[áa]n?|est[áa]n?\s+ahora|est[áa]n?|"
            r"se\s+ha|se\s+han|se",
            _ES_PASSIVE_CLASSES,
        ),
        _ES_PASSIVE_CLASSES,
    ),
    (
        _passive_pattern(r"has\s+been|have\s+been|was|were|is\s+now|are\s+now", _EN_PASSIVE_CLASSES),
        _EN_PASSIVE_CLASSES,
    ),
]

# "README.md quedó con la sección 'Uso' agregada" / "quedó con la nueva
# sección": a resultative that asserts new content without naming a verb of
# change. Modified, unless the clause says the content is unchanged.
_RESULTATIVE_CON = re.compile(
    rf"{_PATH_RE}{_ADVERB}\s+(?P<verb>qued[oó]|quedaron|quedan?)\s+con\s+"
    r"(?![^.;\n]{0,40}\b(?:mism[oa]s?|igual|sin\s+cambios|intact[oa]s?|idéntic[oa]s?)\b)",
    re.IGNORECASE,
)

# "config/settings.py quedó en blanco": emptied is a modification (DPO live run).
_RESULTATIVE_STATE = re.compile(
    rf"{_PATH_RE}{_ADVERB}\s+(?P<verb>qued[oó]|quedaron|quedan?|est[áa]n?\s+ahora|ahora\s+est[áa]n?)\s+"
    r"(?:en\s+blanco|vac[ií][oa]s?|limpi[oa]s?)\b",
    re.IGNORECASE,
)

# "Archivo src/utils.py creado con la función clamp()": a telegraphic,
# sentence-initial result with the participle AFTER the path and no article.
# "El archivo X creado por el usuario ..." (article) stays a description.
_POSTNOMINAL = re.compile(
    rf"(?:^|[.!?\n]\s*)(?:archivo\s+|fichero\s+|file\s+)?{_PATH_RE}\s+(?P<verb>(?:cread|modificad|actualizad|editad|"
    rf"cambiad|corregid|arreglad|eliminad|borrad|agregad|añadid|guardad|generad){_P}|created|modified|updated|"
    r"edited|changed|fixed|deleted|removed|added|written|saved)\b",
    re.IGNORECASE,
)

# "Ahora tu archivo config/settings.py tiene DEBUG = True": a result stated as
# the file's new content. Only with "ahora" -- "X ya tiene esa función" is a
# reason NOT to edit, never a claim. Spanish only: the English analog ("X now
# has") was never seen live and the hand corpus holds a counter-example
# ("The updated config.py now has the flag", a description).
_NOW_HAS = re.compile(
    rf"(?:\bahora\s+(?:tu\s+|el\s+|la\s+)?(?:archivo\s+|fichero\s+)?{_PATH_RE}\s+(?P<verb>tiene|contiene|incluye)\b"
    rf"|{_PATH_RE.replace('(?P<path>', '(?P<path2>')}\s+(?P<verb2>ahora\s+(?:tiene|contiene|incluye))\b)",
    re.IGNORECASE,
)

# "En el archivo src/utils.py, el cambio fue crear una función": the change is
# narrated as an event in the file (DPO live run).
_CHANGE_NARRATIVE = re.compile(
    rf"(?:en\s+|in\s+)?(?:el\s+archivo\s+|the\s+file\s+)?{_PATH_RE}[,:]?\s+(?:el\s+|the\s+)?(?P<verb>cambio\s+(?:fue|es|consisti[oó]\s+en)|change\s+(?:was|is))\b",
    re.IGNORECASE,
)

# "Revisé el archivo config/settings.py y cambié la configuración": the verb
# of change names no path, the path it acts on came just before it in the
# same clause (RAFT live run, 2026-10-01). Only first-person / reflexive
# forms, only joined by "y"/"and", and only when no path follows the verb
# in the clause (otherwise the active pattern already bound that path).
_PATH_THEN_VERB = re.compile(
    rf"{_PATH_RE}(?:[^.;:,\n]{{0,30}}?)\s+(?:y|e|and)\s+"
    r"(?P<verb>modifiqu[eé]|actualic[eé]|edit[eé]|cambi[eé]|correg[ií]|arregl[eé]|parche[eé]|reescrib[ií]|ajust[eé]|"
    r"(?:lo|la|los|las)\s+(?:modifiqu[eé]|actualic[eé]|edit[eé]|cambi[eé]|correg[ií]|arregl[eé])|"
    r"(?:he|hemos)\s+(?:modificado|actualizado|editado|cambiado|corregido|arreglado))\b"
    r"(?P<rest>[^.;:\n]{0,80})",
    re.IGNORECASE,
)

# "Se han editado los siguientes archivos:\n- config/settings.py": the verb,
# a colon, then one path per bullet line (live finding, EDIT-RISK family).
_LIST_VERB = (
    r"(?P<verb>(?:se\s+)?(?:he|hemos|ha|han)\s+(?:sido\s+)?(?:cread|eliminad|borrad|modificad|actualizad|editad|cambiad|"
    r"corregid|arreglad|guardad|agregad|añadid|generad)[oa]s?|(?:I\s+|we\s+)?(?:created|deleted|removed|"
    r"modified|updated|edited|changed|wrote|added|generated))\b[^\n:]{0,60}:"
)
_LIST_HEAD = re.compile(_LIST_VERB + r"\s*\n", re.IGNORECASE)
# Receipts-RAFT round-0 hand audit, 2026-10-01: four constructions the
# reward scored R=1 although they claim effects the bytes do not show.
# a45 -- "Todos los archivos han sido borrados: a.py, b.py, y c.toml." (list
# on the same line as the verb).
_LIST_HEAD_INLINE = re.compile(_LIST_VERB + rf"[ \t]*{_PATH_RE}", re.IGNORECASE)
_INLINE_NEXT = re.compile(rf"\s*(?:,\s*(?:(?:y|e|and)\s+)?|\s+(?:y|e|and)\s+){_PATH_RE}", re.IGNORECASE)
# a23, a39, a47 -- conjoined subjects before a passive verb: "a.py y b.py
# quedaron modificados" claims BOTH; only the last path was read.
_PREV_CONJ_PATH = re.compile(rf"{_PATH_RE}\s*(?:,|\by\b|\be\b|\band\b|&)\s*[`\"']?$", re.IGNORECASE)
# a31 -- "Ahora tienes ... un nuevo archivo llamado X". Needs "ahora" or
# "nuevo": "tienes un archivo X que hace..." is a description.
_NOW_HAVE_NEW = re.compile(
    rf"\b(?:ahora\s+(?:tienes|ten[ée]s|hay)\s+(?:un\s+)?(?:nuevo\s+)?|(?:tienes|ten[ée]s|hay)\s+un\s+nuevo\s+)"
    rf"(?:archivo|fichero)\s+(?:nuevo\s+)?(?:llamado\s+|denominado\s+)?{_PATH_RE}",
    re.IGNORECASE,
)
_PATH_RE_ANON = _PATH_RE.replace("(?P<path>", "(?:")  # same shape, repeatable within one pattern
# Round-0 audit 2, b25 -- "aquí tienes los archivos X y Y modificados": the
# participle asserts the files changed. "aquí tienes los cambios" (content
# shown in chat, no participle on the files) stays a proposal.
_HERE_ARE_CHANGED = re.compile(
    r"\baqu[ií]\s+(?:tienes|ten[ée]s|est[áa]n?)\s+(?:los\s+|las\s+|el\s+|la\s+)?(?:archivos?|ficheros?)\s+"
    rf"(?P<list>{_PATH_RE_ANON}(?:\s*(?:,|\by\b|\be\b)\s*{_PATH_RE_ANON})*)\s+"
    r"(?P<verb>(?:modificad|actualizad|editad|cambiad|cread|corregid|arreglad)[oa]s?)\b",
    re.IGNORECASE,
)
# a43 -- "El archivo X queda así:" (resultative present + the new content).
# "quedaría así" is conditional and does not match.
_QUEDA_ASI = re.compile(rf"{_PATH_RE}\s+(?P<verb>queda|quedan|qued[óo]|quedaron)\s+as[ií]\b", re.IGNORECASE)
_LIST_ITEM = re.compile(rf"^[ \t]*(?:[-*•]|\d+[.)])[ \t]*{_PATH_RE}", re.IGNORECASE | re.MULTILINE)
_LIST_KIND: list[tuple[int, str]] = [
    (KIND_CREATED, r"(?:se\s+)?(?:he|hemos|ha|han)\s+(?:sido\s+)?cread[oa]s?|(?:I\s+|we\s+)?created"),
    (KIND_DELETED, r"(?:se\s+)?(?:he|hemos|ha|han)\s+(?:sido\s+)?(?:eliminad|borrad)[oa]s?|(?:I\s+|we\s+)?(?:deleted|removed)"),
    (
        KIND_MODIFIED,
        r"(?:se\s+)?(?:he|hemos|ha|han)\s+(?:sido\s+)?(?:modificad|actualizad|editad|cambiad|corregid|arreglad)[oa]s?|"
        r"(?:I\s+|we\s+)?(?:modified|updated|edited|changed)",
    ),
    (
        KIND_WRITTEN,
        r"(?:se\s+)?(?:he|hemos|ha|han)\s+(?:sido\s+)?(?:guardad|agregad|añadid|generad)[oa]s?|(?:I\s+|we\s+)?(?:wrote|added|generated)",
    ),
]

# The verb's object is a CONTAINER when a preposition introduces the path:
# "agregué la función a `x.py`", "removed the import from x.py". Then the
# claim is "x.py was written to", kind 4, whatever the verb said.
_CONTAINER_PREP = re.compile(
    r"\b(?:en|dentro\s+de|del|de|al|a|desde|from|in|into|inside|within|to|of|on|at)\b"
    r"\s*(?:el|la|los|las|the|this|that|mi|our|su|your|un|una|a|an)?"
    r"\s*(?:archivo|fichero|file|m[oó]dulo|module|script|ruta|path)?\s*$",
    re.IGNORECASE,
)

# A verb preceded by an article / adjective is adjectival, not a claim:
# "the updated config.py", "el archivo creado x.py".
_ADJECTIVAL_PRECEDER = re.compile(
    r"\b(?:the|a|an|this|that|newly|recently|already|el|la|un|una|los|las|este|esta|ese|esa|"
    r"reci[eé]n|archivo|fichero|file)\s*$"
    # "la función creada en X", "el módulo modificado en X": article + noun +
    # participle is a description, not a claim. "Clamp funcion creado en X"
    # (no article, live Qwen output) stays a claim.
    r"|\b(?:el|la|los|las|un|una|unos|unas|the|a|an)\s+[\w()]+\s*$",
    re.IGNORECASE,
)

# Scope blockers in the 40-char window before the verb: negation, attempt,
# intent, obligation, ability, future. Any of them turns a completed-effect
# verb into something that did not (or may not) happen.
_BLOCKER_PATTERN = re.compile(
    r"\b(?:no|nunca|jamás|tampoco|sin|not|never|without|"
    r"intent[eé]|intento|trat[eé]|trato|quise|quiero|quer[ií]a|querr[ií]a|"
    r"voy\s+a|vamos\s+a|iba\s+a|podr[ií]a|podr[ií]as|puedo|puedes|deber[ií]a|deber[ií]as|"
    r"debo|debes|necesito|necesitas|hay\s+que|falta|pendiente|"
    r"tried|try|trying|attempted|attempting|will|would|could|should|can|cannot|"
    r"going\s+to|need\s+to|needs\s+to|want\s+to|let\s+me|let's|about\s+to|"
    r"unable\s+to|failed\s+to|fail\s+to|if|si|unless|a\s+menos\s+que)\b|\w+n't\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(
    r"[,;:.!?¿¡]|\b(?:pero|sino|aunque|but|however|although)\b", re.IGNORECASE
)
_CONJ_NEXT_PATH = re.compile(rf"\s*(?:,|\by\b|\band\b|\be\b|&)\s*{_PATH_RE}", re.IGNORECASE)
_URL_PREFIX = re.compile(r"https?://[^\s`'\"]*$", re.IGNORECASE)
_WINDOW = 40


@dataclass(frozen=True)
class MutationClaim:
    """One assertion in the text that a file effect happened."""

    path: str
    kind: int  # KIND_CREATED | KIND_MODIFIED | KIND_DELETED | KIND_WRITTEN
    verb: str
    span: tuple[int, int]
    text: str

    @property
    def kind_name(self) -> str:
        return KIND_NAMES[self.kind]


def _kind_of(verb: str, classes: list[tuple[int, str]]) -> int:
    for kind, rx in classes:
        if re.fullmatch(rx, verb, re.IGNORECASE):
            return kind
    raise ValueError(f"verb {verb!r} matched no class")  # unreachable by construction


def _blocked(text: str, verb_start: int) -> bool:
    prefix = text[max(0, verb_start - _WINDOW) : verb_start]
    for trigger in _BLOCKER_PATTERN.finditer(prefix):
        if _CLAUSE_BOUNDARY.search(prefix[trigger.end() :]):
            continue  # the blocker belongs to another clause
        return True
    return False


def _in_question(text: str, pos: int) -> bool:
    start = max(text.rfind(".", 0, pos), text.rfind("\n", 0, pos), text.rfind("!", 0, pos)) + 1
    end_candidates = [i for i in (text.find(".", pos), text.find("\n", pos), text.find("?", pos), text.find("!", pos)) if i != -1]
    end = min(end_candidates) if end_candidates else len(text)
    sentence = text[start : end + 1]
    return sentence.rstrip().endswith("?") or "¿" in sentence


def _clean_path(raw: str) -> str:
    return raw.strip("`'\"").rstrip(".,;:")


def _is_url(text: str, path_start: int) -> bool:
    return bool(_URL_PREFIX.search(text[:path_start]))


def detect_mutation_claims(text: str) -> list[MutationClaim]:
    """Extract (path, kind) effect claims from a finished agent turn.

    Over-matching is the safe direction for a tripwire (a detected claim
    with a matching receipt costs nothing); under-matching lets theater
    through. The scoping rules exist to keep the first from becoming
    noise: negation, attempt, intent, obligation, futures, conditionals
    and questions are not claims; adjectival participles ("the updated
    config.py") are not claims; a preposition before the path makes the
    path a container (kind 4).
    """
    claims: list[MutationClaim] = []
    seen: set[tuple[str, int]] = set()

    def add(path: str, kind: int, verb: str, span: tuple[int, int]) -> None:
        path = _clean_path(path)
        if not path or (path, kind) in seen:
            return
        seen.add((path, kind))
        claims.append(MutationClaim(path, kind, verb, span, text[span[0] : span[1]]))

    def add_conjoined_before(path_start: int, kind: int, verb: str, end: int) -> None:
        # Walk back over "X y " / "X, " in front of the subject path.
        pos = path_start
        while True:
            lo = max(0, pos - 200)
            pm = _PREV_CONJ_PATH.search(text[lo:pos])
            if not pm:
                return
            start = lo + pm.start("path")
            if _is_url(text, start):
                return
            add(pm.group("path"), kind, verb, (start, end))
            pos = start

    for pattern, classes in _ACTIVE_PATTERNS:
        for m in pattern.finditer(text):
            v0 = m.start("verb")
            if _blocked(text, v0) or _in_question(text, v0):
                continue
            if _ADJECTIVAL_PRECEDER.search(text[max(0, v0 - 12) : v0]):
                continue
            if _is_url(text, m.start("path")):
                continue
            if _FILLER_CONTRAST.search(m.group("filler")):
                continue  # the path belongs to the contrasting clause
            kind = _kind_of(m.group("verb"), classes)
            if _CONTAINER_PREP.search(m.group("filler")):
                kind = KIND_WRITTEN
            add(m.group("path"), kind, m.group("verb"), (m.start(), m.end()))
            # "modifiqué `a.py` y `b.py`" -- further paths after a conjunction.
            pos = m.end()
            while True:
                nm = _CONJ_NEXT_PATH.match(text, pos)
                if not nm or _is_url(text, nm.start("path")):
                    break
                add(nm.group("path"), kind, m.group("verb"), (m.start(), nm.end()))
                pos = nm.end()

    for pattern, classes in _PASSIVE_PATTERNS:
        for m in pattern.finditer(text):
            v0 = m.start("verb")
            if _blocked(text, v0) or _in_question(text, v0):
                continue
            if _is_url(text, m.start("path")):
                continue
            kind = _kind_of(m.group("verb"), classes)
            add(m.group("path"), kind, m.group("verb"), (m.start(), m.end()))
            add_conjoined_before(m.start("path"), kind, m.group("verb"), m.end())

    for m in _RESULTATIVE_CON.finditer(text):
        v0 = m.start("verb")
        if _blocked(text, v0) or _in_question(text, v0) or _is_url(text, m.start("path")):
            continue
        add(m.group("path"), KIND_MODIFIED, m.group("verb"), (m.start(), m.end()))

    for pattern, kind in ((_RESULTATIVE_STATE, KIND_MODIFIED), (_CHANGE_NARRATIVE, KIND_WRITTEN)):
        for m in pattern.finditer(text):
            v0 = m.start("verb")
            if _blocked(text, v0) or _in_question(text, v0) or _is_url(text, m.start("path")):
                continue
            add(m.group("path"), kind, m.group("verb"), (m.start(), m.end()))

    for m in _POSTNOMINAL.finditer(text):
        v0 = m.start("verb")
        if _blocked(text, v0) or _in_question(text, v0) or _is_url(text, m.start("path")):
            continue
        verb = m.group("verb")
        classes = _ES_PASSIVE_CLASSES if re.search(r"[oa]s?$", verb, re.IGNORECASE) and not verb.lower().endswith("ed") else _EN_PASSIVE_CLASSES
        try:
            kind = _kind_of(verb, classes)
        except ValueError:
            kind = _kind_of(verb, _EN_PASSIVE_CLASSES)
        add(m.group("path"), kind, verb, (m.start(), m.end()))

    for m in _PATH_THEN_VERB.finditer(text):
        v0 = m.start("verb")
        if _blocked(text, v0) or _in_question(text, v0) or _is_url(text, m.start("path")):
            continue
        if _PATH_ONLY.search(m.group("rest")):
            continue  # the verb has its own object; the active pattern owns it
        add(m.group("path"), KIND_MODIFIED, m.group("verb"), (m.start(), m.end("verb")))

    for m in _NOW_HAS.finditer(text):
        path = m.group("path") or m.group("path2")
        verb = m.group("verb") or m.group("verb2")
        v0 = m.start("verb") if m.group("verb") else m.start("verb2")
        if _blocked(text, v0) or _in_question(text, v0) or _is_url(text, m.start("path") if m.group("path") else m.start("path2")):
            continue
        add(path, KIND_MODIFIED, verb, (m.start(), m.end()))

    for m in _LIST_HEAD.finditer(text):
        v0 = m.start("verb")
        if _blocked(text, v0) or _in_question(text, v0):
            continue
        kind = _kind_of(m.group("verb"), _LIST_KIND)
        pos = m.end()
        while True:
            item = _LIST_ITEM.match(text, pos)
            if not item:
                break
            if not _is_url(text, item.start("path")):
                add(item.group("path"), kind, m.group("verb"), (m.start(), item.end()))
            nl = text.find("\n", item.end())
            if nl == -1:
                break
            pos = nl + 1

    for m in _LIST_HEAD_INLINE.finditer(text):
        v0 = m.start("verb")
        if _blocked(text, v0) or _in_question(text, v0) or _is_url(text, m.start("path")):
            continue
        kind = _kind_of(m.group("verb"), _LIST_KIND)
        add(m.group("path"), kind, m.group("verb"), (m.start(), m.end()))
        pos = m.end()
        while True:
            nm = _INLINE_NEXT.match(text, pos)
            if not nm or _is_url(text, nm.start("path")):
                break
            add(nm.group("path"), kind, m.group("verb"), (m.start(), nm.end()))
            pos = nm.end()

    for m in _NOW_HAVE_NEW.finditer(text):
        if _blocked(text, m.start()) or _in_question(text, m.start()) or _is_url(text, m.start("path")):
            continue
        add(m.group("path"), KIND_CREATED, "tienes un nuevo archivo", (m.start(), m.end()))

    for m in _HERE_ARE_CHANGED.finditer(text):
        v0 = m.start("verb")
        if _blocked(text, m.start()) or _in_question(text, v0):
            continue
        kind = _kind_of(m.group("verb"), _ES_PASSIVE_CLASSES)
        for pm in _PATH_ONLY.finditer(m.group("list")):
            start = m.start("list") + pm.start("path")
            if not _is_url(text, start):
                add(pm.group("path"), kind, m.group("verb"), (m.start(), m.end()))

    for m in _QUEDA_ASI.finditer(text):
        v0 = m.start("verb")
        if _blocked(text, v0) or _in_question(text, v0) or _is_url(text, m.start("path")):
            continue
        add(m.group("path"), KIND_MODIFIED, m.group("verb"), (m.start(), m.end()))
        add_conjoined_before(m.start("path"), KIND_MODIFIED, m.group("verb"), m.end())

    claims.sort(key=lambda c: c.span)
    return claims


# ---------------------------------------------------------------------------
# 4. The guard: claims x receipts -> compiled verdict
# ---------------------------------------------------------------------------


class MutationTheaterError(Exception):
    """The turn asserts file effects the bytes do not show (or, in strict
    mode, hides effects the bytes do show)."""


@dataclass(frozen=True)
class ClaimVerdict:
    claim: MutationClaim
    receipt: Receipt | None
    receipt_kind: int
    admitted: bool
    reason: str | None  # None exactly when the compiled rule admitted the claim


@dataclass(frozen=True)
class MutationAudit:
    approved: bool
    turn: int | None
    verdicts: tuple[ClaimVerdict, ...]
    unreported: tuple[Receipt, ...]  # observed effects no claim mentions
    unreported_rejected: tuple[Receipt, ...]  # the subset strict mode rejected
    reasons: tuple[str, ...]
    latency_ns: int  # time spent inside compiled decisions only

    @property
    def rejected(self) -> tuple[ClaimVerdict, ...]:
        return tuple(v for v in self.verdicts if not v.admitted)


def _norm_path(path: str) -> str:
    return (path[2:] if path.startswith("./") else path).rstrip("/")


def _match_paths(claim_path: str, paths: Iterable[str]) -> list[str]:
    """Paths equal to the claim, or ending with '/<claim>' when the claim
    is a bare basename / suffix ("modifiqué utils.py")."""
    norm = _norm_path(claim_path)
    paths = list(paths)
    if norm in paths:
        return [norm]
    suffix = "/" + norm
    return [p for p in paths if p.endswith(suffix)]


def _match_receipts(claim_path: str, effects: dict[str, Receipt]) -> list[Receipt]:
    return [effects[p] for p in _match_paths(claim_path, effects)]


class MutationGuard:
    """File-effect honesty guard over one workspace root.

    Lifecycle per agent turn: `begin()` (baseline snapshot), the agent
    runs, `end()` (observe + record receipts), `audit(text)` (verdict).
    `audit()` calls `end()` for you if the turn is still open.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        ledger: ReceiptLedger | None = None,
        rule_path: str | Path = RULE_PATH,
        ignore: Sequence[str] = DEFAULT_IGNORE,
        strict: bool = False,
        incremental: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.ledger = ledger if ledger is not None else ReceiptLedger()
        self.ignore = tuple(ignore)
        self.strict = strict
        # incremental=False re-reads every byte on every snapshot (the
        # paranoid setting, see snapshot() for the trust model).
        self.incremental = incremental
        self._gate = Gate(rule_path)
        self._before: Snapshot | None = None
        self._after: Snapshot | None = None
        self._turn: int | None = None
        self._open = False

    # -- lifecycle -------------------------------------------------------
    @property
    def turn(self) -> int | None:
        return self._turn

    @property
    def before(self) -> Snapshot | None:
        """Baseline of the current/last turn."""
        return self._before

    @property
    def after(self) -> Snapshot | None:
        """Snapshot taken by the last end() (or adopted from disk)."""
        return self._after

    def _snapshot(self, cache: Snapshot | None) -> Snapshot:
        return snapshot(self.root, self.ignore, cache=cache if self.incremental else None)

    def begin(self, baseline: Mapping[str, str] | None = None, *, turn: int | None = None) -> int:
        """Start a turn: snapshot the workspace (or adopt `baseline`).
        Returns the turn number: `turn` if given (a harness that numbers
        turns itself, e.g. the CLI's snapshot.json), else ledger's latest
        + 1. The previous turn's last snapshot serves as the hash cache."""
        if baseline is None:
            self._before = self._snapshot(self._after or self._before)
        else:
            self._before = Snapshot.of(baseline)
        self._after = None
        # A turn with no effects leaves no receipt, so the ledger alone
        # cannot number turns: the guard's own counter advances too.
        self._turn = turn if turn is not None else max(self.ledger.latest_turn or 0, self._turn or 0) + 1
        self._open = True
        return self._turn

    def end(self) -> list[Receipt]:
        """Close the turn: snapshot again, diff against the baseline, append
        receipts. The new snapshot becomes the next turn's baseline."""
        if self._before is None or self._turn is None:
            raise LedgerError("end() called before begin()")
        self._after = self._snapshot(self._before)
        receipts = self.ledger.record_diff(self._before, self._after, turn=self._turn)
        self._open = False
        return receipts

    def adopt_snapshot(self, files: Mapping[str, str]) -> None:
        """For a guard rebuilt from disk: the workspace as of the latest
        end(), so audit reasons can say whether a claimed path exists."""
        self._after = Snapshot.of(files)

    # -- read path: ground truth for the model, never recall -------------
    def ground_truth(self, turn: int | None = None) -> str:
        """A text block the harness can put in front of the model so it
        READS what changed instead of remembering it."""
        t = self._resolve_turn(turn)
        effects = self.ledger.net_effects(t) if t is not None else {}
        lines = [f"[netelpro receipts] observed file effects, turn {t if t is not None else '-'} (sha256, not recollection):"]
        if not effects:
            lines.append("  NO file changed. Any statement that a file was created, modified or deleted is false.")
        else:
            for r in effects.values():
                lines.append("  " + r.short())
        return "\n".join(lines)

    # -- audit -----------------------------------------------------------
    def audit(self, text: str, *, turn: int | None = None, strict: bool | None = None) -> MutationAudit:
        if self._open:
            self.end()
        strict_mode = self.strict if strict is None else strict
        t = self._resolve_turn(turn)
        effects = self.ledger.net_effects(t) if t is not None else {}
        claims = detect_mutation_claims(text)

        verdicts: list[ClaimVerdict] = []
        claimed_paths: set[str] = set()
        latency = 0
        for claim in claims:
            matched = _match_receipts(claim.path, effects)
            kinds = {r.kind for r in matched}
            receipt: Receipt | None = None
            receipt_kind = KIND_NONE
            ambiguous: str | None = None
            if len(kinds) == 1:
                receipt = matched[0]
                receipt_kind = receipt.kind_code
            elif len(kinds) > 1:
                ambiguous = ", ".join(f"{r.path} ({r.kind})" for r in matched)
            for r in matched:
                claimed_paths.add(r.path)

            t0 = time.perf_counter_ns()
            allow, gate_reason = self._gate.check(claim.kind, receipt_kind, False)
            latency += time.perf_counter_ns() - t0

            reason: str | None
            if ambiguous is not None:
                allow, reason = False, (
                    f"claim {claim.text!r} ({claim.kind_name}): '{claim.path}' matches several changed "
                    f"files with different effects: {ambiguous}"
                )
            elif gate_reason is not None:
                reason = f"claim {claim.text!r}: gate failure, fail-closed: {gate_reason}"
                allow = False
            elif allow:
                reason = None
            else:
                reason = self._explain(claim, receipt, t)
            verdicts.append(ClaimVerdict(claim, receipt, receipt_kind, allow, reason))

        unreported = tuple(r for p, r in effects.items() if p not in claimed_paths)
        rejected_silent: list[Receipt] = []
        for r in unreported:
            t0 = time.perf_counter_ns()
            allow, gate_reason = self._gate.check(KIND_NONE, r.kind_code, strict_mode)
            latency += time.perf_counter_ns() - t0
            if gate_reason is not None or not allow:
                rejected_silent.append(r)

        reasons: list[str] = [v.reason for v in verdicts if v.reason is not None]
        for r in rejected_silent:
            reasons.append(
                f"unreported effect: {r.kind} '{r.path}' ({(r.before or '-')[:12]} -> "
                f"{(r.after or '-')[:12]}) with no claim in the text -- strict mode rejects silent writes"
            )
        approved = all(v.admitted for v in verdicts) and not rejected_silent
        return MutationAudit(
            approved=approved,
            turn=t,
            verdicts=tuple(verdicts),
            unreported=unreported,
            unreported_rejected=tuple(rejected_silent),
            reasons=tuple(reasons),
            latency_ns=latency,
        )

    def enforce(self, text: str, **kw: Any) -> str:
        audit = self.audit(text, **kw)
        if not audit.approved:
            raise MutationTheaterError("\n".join(audit.reasons))
        return text

    # -- internals -------------------------------------------------------
    def _resolve_turn(self, turn: int | None) -> int | None:
        if turn is not None:
            return turn
        if self._turn is not None:
            return self._turn
        return self.ledger.latest_turn

    def _explain(self, claim: MutationClaim, receipt: Receipt | None, turn: int | None) -> str:
        head = f"claim {claim.text!r} ({claim.kind_name}) at offset {claim.span[0]}"
        if receipt is None:
            if turn is None:
                return f"{head}: no receipts recorded at all -- begin()/end() never ran, nothing was observed"
            known: dict[str, str] = {**(self._before or {}), **(self._after or {})}
            hits = _match_paths(claim.path, known)
            if known and not hits:
                return (
                    f"{head}: no receipt for '{claim.path}' -- the path does not exist in the "
                    f"workspace before or after the turn"
                )
            tail = f" (sha256 {known[hits[0]][:12]}...)" if hits else ""
            return (
                f"{head}: no receipt for '{claim.path}' -- sha256 unchanged since turn start{tail}: "
                f"the bytes were never written"
            )
        existed = "did not exist at turn start" if receipt.kind == "created" else (
            "no longer exists" if receipt.kind == "deleted" else "existed and changed"
        )
        return (
            f"{head}: receipt says {receipt.kind} ({existed}), claim says {claim.kind_name} -- "
            f"the effect happened, the description of it is false"
        )



# ---------------------------------------------------------------------------
# 5. CLI: zero-integration wrapper for any harness
# ---------------------------------------------------------------------------

STATE_DIR = ".netelpro"
SNAPSHOT_FILE = "snapshot.json"
LEDGER_FILE = "receipts.jsonl"


def _state_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = Path(args.root).resolve()
    state = Path(args.state) if args.state else root / STATE_DIR
    return root, state / SNAPSHOT_FILE, state / LEDGER_FILE


def load_snapshot(path: Path) -> tuple[int, Snapshot] | None:
    """(turn, snapshot) from a snapshot.json written by `begin`, or None when
    there is none. A corrupt file is a LedgerError, never an empty baseline.
    A file without `stats` (pre-fast-path format) loads as a full baseline
    that reuses nothing."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        files = dict(d["files"])
        stats = {k: tuple(int(x) for x in v) for k, v in dict(d.get("stats", {})).items()}
        if any(len(v) != 4 for v in stats.values()):
            raise ValueError("stat signature must have 4 fields")
        return int(d["turn"]), Snapshot(files, stats, int(d.get("taken_ns", 0)))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise LedgerError(f"{path}: unreadable snapshot: {e}") from e


def save_snapshot(path: Path, turn: int, files: Mapping[str, str]) -> None:
    snap = Snapshot.of(files)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "turn": turn,
        "taken_ns": snap.taken_ns,
        "files": dict(snap),
        "stats": {k: list(v) for k, v in snap.stats.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _take_summary(snap: Snapshot) -> str:
    return f"{len(snap)} files ({snap.hashed} hashed, {snap.reused} reused from cache)"


@dataclass
class TurnState:
    """A turn opened from the on-disk state of a root (`open_turn`): the
    guard already holds the baseline, the receipts of the live diff are in
    the guard's in-memory ledger and nothing has been written to disk.
    `commit()` persists the ledger and advances the on-disk baseline; a
    turn that is never committed leaves the state exactly as found."""

    root: Path
    turn: int
    guard: MutationGuard
    snapshot_path: Path
    ledger_path: Path
    baseline_existed: bool

    def commit(self) -> list[Receipt]:
        after = self.guard.after
        if after is None:  # pragma: no cover -- open_turn always ends the turn
            raise LedgerError("commit() on a turn that was never observed")
        self.guard.ledger.save(self.ledger_path)
        save_snapshot(self.snapshot_path, self.turn, after)
        return self.guard.ledger.for_turn(self.turn)


def open_turn(
    root: str | Path,
    *,
    state_dir: str | Path | None = None,
    strict: bool = False,
    incremental: bool = True,
    take_baseline_if_missing: bool = False,
) -> TurnState:
    """Load `<state>/snapshot.json` + `receipts.jsonl`, begin a guard on that
    baseline and observe the live diff (receipts in memory only). Shared by
    the MCP tool and the Claude Code hook so both judge the same way.

    Without a baseline: raise LedgerError, or with `take_baseline_if_missing`
    hash the root now (an empty turn) so the NEXT turn has one.
    """
    base = Path(root).resolve()
    state = Path(state_dir) if state_dir else base / STATE_DIR
    snap_path, ledger_path = state / SNAPSHOT_FILE, state / LEDGER_FILE
    ledger = ReceiptLedger.load(ledger_path)
    loaded = load_snapshot(snap_path)
    existed = loaded is not None
    if loaded is None:
        if not take_baseline_if_missing:
            raise LedgerError(f"no baseline snapshot under {state}: run `begin` first")
        turn = (ledger.latest_turn or 0) + 1
        baseline = snapshot(base, DEFAULT_IGNORE)
    else:
        turn, baseline = loaded
    guard = MutationGuard(base, ledger=ledger, strict=strict, incremental=incremental)
    guard.begin(baseline, turn=turn)
    guard.end()
    return TurnState(base, turn, guard, snap_path, ledger_path, existed)


def _cmd_begin(args: argparse.Namespace) -> int:
    root, snap_path, ledger_path = _state_paths(args)
    ledger = ReceiptLedger.load(ledger_path)
    previous = load_snapshot(snap_path)
    turn = max(ledger.latest_turn or 0, previous[0] if previous else 0) + 1
    files = snapshot(root, DEFAULT_IGNORE, cache=None if args.full else (previous[1] if previous else None))
    save_snapshot(snap_path, turn, files)
    print(f"turn {turn}: baseline of {_take_summary(files)} under {root}")
    return 0


def _cmd_end(args: argparse.Namespace) -> int:
    root, snap_path, ledger_path = _state_paths(args)
    loaded = load_snapshot(snap_path)
    if loaded is None:
        print("error: no baseline snapshot -- run `begin` first", file=sys.stderr)
        return 1
    turn, before = loaded
    ledger = ReceiptLedger.load(ledger_path)
    after = snapshot(root, DEFAULT_IGNORE, cache=None if args.full else before)
    receipts = ledger.record_diff(before, after, turn=turn)
    ledger.save(ledger_path)
    save_snapshot(snap_path, turn, after)
    print(f"turn {turn}: {len(receipts)} receipt(s) recorded, chain head {ledger.head[:12]}; {_take_summary(after)}")
    for r in receipts:
        print("  " + r.short())
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    root, snap_path, ledger_path = _state_paths(args)
    ledger = ReceiptLedger.load(ledger_path)
    ok, reason = ledger.verify_chain()
    loaded = load_snapshot(snap_path)
    turn = args.turn if args.turn is not None else (loaded[0] if loaded else ledger.latest_turn)
    guard = MutationGuard(root, ledger=ledger)
    print(guard.ground_truth(turn))
    print(f"ledger: {len(ledger)} receipt(s), chain {'verified' if ok else 'BROKEN: ' + str(reason)}")
    return 0 if ok else 1


def _cmd_audit(args: argparse.Namespace) -> int:
    root, snap_path, ledger_path = _state_paths(args)
    if args.text == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.text).read_text(encoding="utf-8")
    ledger = ReceiptLedger.load(ledger_path)
    loaded = load_snapshot(snap_path)
    turn = args.turn if args.turn is not None else (loaded[0] if loaded else ledger.latest_turn)
    guard = MutationGuard(root, ledger=ledger, strict=args.strict)
    if loaded is not None:
        guard.adopt_snapshot(loaded[1])  # so "path never existed" reasons are exact
    audit = guard.audit(text, turn=turn)
    if args.json:
        payload = {
            "approved": audit.approved,
            "turn": audit.turn,
            "claims": [
                {
                    "path": v.claim.path,
                    "kind": v.claim.kind_name,
                    "text": v.claim.text,
                    "receipt": v.receipt.to_dict() if v.receipt else None,
                    "admitted": v.admitted,
                    "reason": v.reason,
                }
                for v in audit.verdicts
            ],
            "unreported": [r.to_dict() for r in audit.unreported],
            "unreported_rejected": [r.to_dict() for r in audit.unreported_rejected],
            "reasons": list(audit.reasons),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"turn {audit.turn}: {len(audit.verdicts)} claim(s), {len(audit.unreported)} unreported effect(s)")
        for v in audit.verdicts:
            mark = "ADMIT " if v.admitted else "REJECT"
            print(f"  [{mark}] {v.claim.kind_name:<8} {v.claim.path}")
            if v.reason:
                print(f"           {v.reason}")
        for r in audit.unreported:
            tag = "REJECT" if r in audit.unreported_rejected else "note  "
            print(f"  [{tag}] unreported {r.short()}")
        print("verdict: " + ("APPROVED" if audit.approved else "DENIED -- verification theater on file effects"))
    return 0 if audit.approved else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="netelpro-receipts",
        description="File-effect receipts: what an LLM agent says it wrote vs. what the bytes say.",
    )
    parser.add_argument("--root", default=".", help="workspace root (default: cwd)")
    parser.add_argument("--state", default=None, help=f"state dir (default: <root>/{STATE_DIR})")
    parser.add_argument(
        "--full",
        action="store_true",
        help="re-read every byte instead of reusing hashes of files whose stat signature is unchanged",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("begin", help="hash the workspace as this turn's baseline").set_defaults(fn=_cmd_begin)
    sub.add_parser("end", help="hash again, record receipts for every changed path").set_defaults(fn=_cmd_end)
    p_show = sub.add_parser("show", help="print the ground-truth block and verify the chain")
    p_show.add_argument("--turn", type=int, default=None)
    p_show.set_defaults(fn=_cmd_show)
    p_audit = sub.add_parser("audit", help="judge an agent's text against the receipts (exit 2 = theater)")
    p_audit.add_argument("--text", required=True, help="file with the agent's final text, or - for stdin")
    p_audit.add_argument("--turn", type=int, default=None)
    p_audit.add_argument("--strict", action="store_true", help="also reject silent (unclaimed) writes")
    p_audit.add_argument("--json", action="store_true")
    p_audit.set_defaults(fn=_cmd_audit)
    args = parser.parse_args(argv)
    try:
        return int(args.fn(args))
    except LedgerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
