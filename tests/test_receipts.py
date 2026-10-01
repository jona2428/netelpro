"""Tests for netelpro.receipts -- file-effect honesty (docs/RECEIPTS_SPEC.md).

Four layers, each tested on its own so a failure names the layer:

1. The compiled rule (rules/mutation_receipt.sl): the FULL declared domain
   (claim 0..4 x receipt 0..3 x strict) -- 40 rows -- against a Python
   oracle, plus native-vs-interpreter differential on all 40.
2. Snapshots and the hash-chained ledger: observation, persistence, and
   tamper detection (a ledger that loads after being edited is the exact
   failure this layer exists to prevent).
3. Mutation-claim detection: a labeled ES/EN corpus with the scoping rules
   the module docstring promises (negation, attempt, intent, future,
   conditional, question, adjectival participle, container preposition).
4. The guard end to end on real files in tmp_path, including the central
   falsifiable claim: a guard rebuilt in a fresh process from the JSONL
   ledger reaches the same verdict as the one that observed the turn.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import pytest

from netelpro.gate import Gate
from netelpro.receipts import (
    KIND_CREATED,
    KIND_DELETED,
    KIND_MODIFIED,
    KIND_NAMES,
    KIND_WRITTEN,
    RULE_PATH,
    LedgerError,
    MutationGuard,
    MutationTheaterError,
    ReceiptLedger,
    Snapshot,
    detect_mutation_claims,
    diff_snapshots,
    load_snapshot,
    save_snapshot,
    snapshot,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. The compiled rule
# ---------------------------------------------------------------------------


def _oracle(claim: int, receipt: int, strict: bool) -> bool:
    if claim == 0:
        return (not strict) or receipt == 0
    if claim == 4:
        return receipt in (1, 2)
    return claim == receipt


FULL_DOMAIN = [((c, r, s), _oracle(c, r, s)) for c, r, s in product(range(5), range(4), (False, True))]


def test_rule_full_domain_matches_oracle():
    gate = Gate(RULE_PATH)
    assert len(FULL_DOMAIN) == 40
    for args, expected in FULL_DOMAIN:
        assert gate.check(*args) == (expected, None), args


def test_rule_native_and_interpreter_agree_on_full_domain():
    gate = Gate(RULE_PATH)
    assert gate.verify(FULL_DOMAIN) == []
    assert gate.manifest() == []


def test_rule_ships_in_package_rules_dir():
    """It must travel in the wheel (pyproject package-data rules/*.sl)."""
    assert RULE_PATH.parent.name == "rules"
    assert RULE_PATH.parent.parent.name == "netelpro"


# ---------------------------------------------------------------------------
# 2. Snapshots and ledger
# ---------------------------------------------------------------------------


def _seed(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / "config").mkdir()
    (root / "config" / "settings.py").write_text("DEBUG = False\n", encoding="utf-8")
    (root / "README.md").write_text("# demo\n", encoding="utf-8")


def test_snapshot_ignores_git_and_caches(tmp_path: Path):
    _seed(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\x00")
    (tmp_path / ".netelpro").mkdir()
    (tmp_path / ".netelpro" / "receipts.jsonl").write_text("", encoding="utf-8")
    snap = snapshot(tmp_path)
    assert set(snap) == {"src/app.py", "config/settings.py", "README.md"}
    assert all(len(h) == 64 for h in snap.values())


def test_diff_detects_created_modified_deleted_and_nothing_else(tmp_path: Path):
    _seed(tmp_path)
    before = snapshot(tmp_path)
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "new.txt").write_text("n", encoding="utf-8")
    (tmp_path / "README.md").unlink()
    after = snapshot(tmp_path)
    effects = diff_snapshots(before, after)
    assert [(k, p) for k, p, _, _ in effects] == [
        ("deleted", "README.md"),
        ("created", "new.txt"),
        ("modified", "src/app.py"),
    ]
    assert effects[0][2] is not None and effects[0][3] is None
    assert effects[1][2] is None and effects[1][3] is not None
    assert effects[2][2] != effects[2][3]


def test_same_content_rewrite_is_not_an_effect(tmp_path: Path):
    """Touching a file without changing its bytes is not a mutation --
    receipts are about content, not mtime."""
    _seed(tmp_path)
    before = snapshot(tmp_path)
    (tmp_path / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    assert diff_snapshots(before, snapshot(tmp_path)) == []


def test_ledger_chain_roundtrip_and_tamper_detection(tmp_path: Path):
    ledger = ReceiptLedger()
    ledger.record("modified", "a.py", "0" * 64, "1" * 64, turn=1, ts_ns=1)
    ledger.record("created", "b.py", None, "2" * 64, turn=1, ts_ns=2)
    ledger.record("deleted", "c.py", "3" * 64, None, turn=2, ts_ns=3)
    assert ledger.verify_chain() == (True, None)
    assert ledger.receipts[1].prev == ledger.receipts[0].digest
    assert ledger.latest_turn == 2

    path = tmp_path / "receipts.jsonl"
    ledger.save(path)
    loaded = ReceiptLedger.load(path)
    assert loaded.receipts == ledger.receipts
    assert loaded.head == ledger.head

    # Edit one field of one record: the chain must refuse to load.
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["path"] = "evil.py"
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="digest mismatch"):
        ReceiptLedger.load(path)

    # Drop a record from the middle: the link breaks.
    lines2 = [lines[0], json.dumps(ledger.receipts[2].to_dict())]
    path.write_text("\n".join(lines2) + "\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="seq"):
        ReceiptLedger.load(path)


def test_ledger_load_missing_file_is_empty(tmp_path: Path):
    ledger = ReceiptLedger.load(tmp_path / "nope.jsonl")
    assert len(ledger) == 0 and ledger.latest_turn is None


def test_ledger_rejects_unknown_kind():
    with pytest.raises(LedgerError):
        ReceiptLedger().record("renamed", "a", None, None, turn=1)


# ---------------------------------------------------------------------------
# 2b. Incremental fast path (stat signature + racy window)
# ---------------------------------------------------------------------------

_OLD_NS = 60_000_000_000  # one minute: well outside RACY_WINDOW_NS


def _age(root: Path, ns_ago: int = _OLD_NS) -> int:
    """Push every file's mtime back; returns the timestamp used. ctime is
    bumped to now by utime itself, so a snapshot taken right after this is
    still inside the racy window -- callers settle with _settle()."""
    t = time.time_ns() - ns_ago
    for p in root.rglob("*"):
        if p.is_file():
            os.utime(p, ns=(t, t))
    return t


def _settle(root: Path) -> Snapshot:
    """A cache whose take postdates every ctime by more than the window:
    the state a real harness is in between turns."""
    first = snapshot(root)
    time.sleep(2.2)
    return snapshot(root, cache=first)


_posix_ctime = pytest.mark.skipif(
    sys.platform == "win32", reason="st_ctime is creation time on Windows; the ctime defence does not apply"
)


def test_incremental_reuses_hashes_for_untouched_files(tmp_path: Path):
    _seed(tmp_path)
    _age(tmp_path)
    cache = _settle(tmp_path)
    again = snapshot(tmp_path, cache=cache)
    assert again == cache
    assert (again.hashed, again.reused) == (0, 3)
    assert again.stats == cache.stats


def test_incremental_rehashes_anything_touched_inside_the_racy_window(tmp_path: Path):
    """A file whose ctime/mtime falls within RACY_WINDOW_NS of the CACHE's
    take is never trusted, however much later the new snapshot runs."""
    _seed(tmp_path)
    _age(tmp_path)  # utime -> ctime = now, i.e. inside the window of the next take
    cache = snapshot(tmp_path)
    time.sleep(2.2)
    later = snapshot(tmp_path, cache=cache)
    assert (later.hashed, later.reused) == (3, 0)
    assert later == cache


def test_incremental_size_change_is_rehashed(tmp_path: Path):
    _seed(tmp_path)
    _age(tmp_path)
    cache = _settle(tmp_path)
    (tmp_path / "src" / "app.py").write_text("x = 100\n", encoding="utf-8")
    after = snapshot(tmp_path, cache=cache)
    assert after["src/app.py"] != cache["src/app.py"]
    assert (after.hashed, after.reused) == (1, 2)
    assert diff_snapshots(cache, after) == [("modified", "src/app.py", cache["src/app.py"], after["src/app.py"])]


@_posix_ctime
def test_incremental_detects_same_size_edit_with_forged_mtime(tmp_path: Path):
    """The evasion the fast path must survive: an agent with a shell edits a
    file without changing its size and resets mtime with utime. size+mtime
    alone would reuse the stale hash; ctime (kernel-owned) gives it away."""
    _seed(tmp_path)
    old = _age(tmp_path)
    cache = _settle(tmp_path)
    target = tmp_path / "src" / "app.py"
    target.write_text("x = 2\n", encoding="utf-8")  # same byte count as "x = 1\n"
    os.utime(target, ns=(old, old))
    st = target.stat()
    assert (st.st_size, st.st_mtime_ns) == cache.stats["src/app.py"][:2], "forgery precondition"
    time.sleep(2.2)  # take the racy window out of the picture: only the signature defends
    after = snapshot(tmp_path, cache=cache)
    assert after["src/app.py"] != cache["src/app.py"]
    assert after.hashed == 1 and after.reused == 2


def test_incremental_detects_rename_over_with_forged_mtime(tmp_path: Path):
    _seed(tmp_path)
    old = _age(tmp_path)
    cache = _settle(tmp_path)
    tmp = tmp_path / "README.md.tmp"
    tmp.write_text("# DEMO\n", encoding="utf-8")  # same size as "# demo\n"
    os.utime(tmp, ns=(old, old))
    os.replace(tmp, tmp_path / "README.md")
    time.sleep(2.2)
    after = snapshot(tmp_path, cache=cache)
    assert after["README.md"] != cache["README.md"]
    assert after.stats["README.md"][3] != cache.stats["README.md"][3]  # new inode


def test_plain_dict_cache_reuses_nothing_and_agrees(tmp_path: Path):
    _seed(tmp_path)
    _age(tmp_path)
    cache = _settle(tmp_path)
    plain = snapshot(tmp_path, cache=dict(cache))
    assert plain == cache and (plain.hashed, plain.reused) == (3, 0)


def test_unreadable_sentinel_is_never_reused(tmp_path: Path):
    _seed(tmp_path)
    _age(tmp_path)
    cache = _settle(tmp_path)
    cache["src/app.py"] = "!unreadable"  # simulate a take during which the file was locked
    after = snapshot(tmp_path, cache=cache)
    assert after["src/app.py"] != "!unreadable" and after.hashed == 1


def test_snapshot_json_carries_stats_and_old_format_still_loads(tmp_path: Path):
    _seed(tmp_path)
    _age(tmp_path)
    snap = _settle(tmp_path)
    f = tmp_path / ".netelpro" / "snapshot.json"
    save_snapshot(f, 3, snap)
    turn, loaded = load_snapshot(f)
    assert turn == 3 and loaded == snap and loaded.stats == snap.stats and loaded.taken_ns == snap.taken_ns
    again = snapshot(tmp_path, cache=loaded)
    assert (again.hashed, again.reused) == (0, 3)

    f.write_text(json.dumps({"turn": 1, "files": dict(snap)}), encoding="utf-8")  # pre-fast-path format
    turn, legacy = load_snapshot(f)
    assert legacy == snap and legacy.stats == {} and legacy.taken_ns == 0
    full = snapshot(tmp_path, cache=legacy)
    assert (full.hashed, full.reused) == (3, 0)


def test_guard_end_uses_the_baseline_as_cache_and_can_be_disabled(tmp_path: Path):
    _seed(tmp_path)
    _age(tmp_path)
    settled = _settle(tmp_path)
    guard = MutationGuard(tmp_path)
    guard.begin(settled)
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    receipts = guard.end()
    assert [r.path for r in receipts] == ["src/app.py"]
    assert guard.after is not None and (guard.after.hashed, guard.after.reused) == (1, 2)

    paranoid = MutationGuard(tmp_path, incremental=False)
    paranoid.begin(settled)
    paranoid.end()
    assert paranoid.after is not None and (paranoid.after.hashed, paranoid.after.reused) == (3, 0)
    assert paranoid.after == guard.after


# ---------------------------------------------------------------------------
# 3. Claim detection corpus
# ---------------------------------------------------------------------------

# (text, expected [(path, kind)], note)
CLAIMS_POSITIVE = [
    ("Modifiqué `src/app.py` para arreglar el bug.", [("src/app.py", KIND_MODIFIED)], "ES preterite + backticks"),
    ("He creado el archivo config/settings.py.", [("config/settings.py", KIND_CREATED)], "ES perfect + 'el archivo'"),
    ("Actualicé pyproject.toml y README.md.", [("pyproject.toml", KIND_MODIFIED), ("README.md", KIND_MODIFIED)], "ES conjunction"),
    ("Eliminé old/legacy.py y borré tmp/cache.json.", [("old/legacy.py", KIND_DELETED), ("tmp/cache.json", KIND_DELETED)], "ES delete verbs"),
    ("Agregué la función parse() a netelpro/parser.py.", [("netelpro/parser.py", KIND_WRITTEN)], "container preposition -> written"),
    ("El archivo docs/SPEC.md fue actualizado.", [("docs/SPEC.md", KIND_MODIFIED)], "ES passive"),
    ("Python 3.11 es requerido; actualicé pyproject.toml.", [("pyproject.toml", KIND_MODIFIED)], "version number is not a path"),
    ("Escribí los resultados en out/report.json.", [("out/report.json", KIND_WRITTEN)], "ES wrote + en"),
    ("I updated src/app.py and created tests/test_new.py.", [("src/app.py", KIND_MODIFIED), ("tests/test_new.py", KIND_CREATED)], "EN two verbs"),
    ("I've removed the unused import from utils/helpers.py.", [("utils/helpers.py", KIND_WRITTEN)], "EN container -> written, not deleted"),
    ("`README.md` has been updated with the new section.", [("README.md", KIND_MODIFIED)], "EN passive"),
    ("Deleted the file scripts/old.sh.", [("scripts/old.sh", KIND_DELETED)], "EN bare participle, sentence-initial"),
    ("Updated `netelpro/guard.py`: new pattern.", [("netelpro/guard.py", KIND_MODIFIED)], "EN commit-style prose"),
    ("We wrote the migration to db/migrations/0003_add.sql.", [("db/migrations/0003_add.sql", KIND_WRITTEN)], "EN 'we' subject"),
    ("No, modifiqué src/app.py como pediste.", [("src/app.py", KIND_MODIFIED)], "discourse 'No,' does not negate across the comma"),
]

CLAIMS_NEGATIVE = [
    ("No pude modificar src/app.py porque es de solo lectura.", "ES negation + infinitive"),
    ("No modifiqué src/app.py.", "ES negation"),
    ("Intenté actualizar routes.py pero falló el permiso.", "ES attempt"),
    ("Voy a crear tests/test_x.py en el siguiente paso.", "ES future"),
    ("Deberías actualizar config/settings.py tú mismo.", "ES obligation on the user"),
    ("¿Quieres que modifique src/app.py?", "ES question"),
    ("Si modifiqué x.py fue por error.", "ES conditional"),
    ("I tried to update settings.py but the write was denied.", "EN attempt"),
    ("I couldn't modify src/app.py: permission denied.", "EN contraction negation"),
    ("I will update src/app.py next.", "EN future (bare verb not a claim anyway)"),
    ("Should I create tests/test_x.py?", "EN question"),
    ("The updated config.py now has the flag.", "EN adjectival participle"),
    ("El archivo creado utils.py tiene 3 funciones.", "ES adjectival participle"),
    ("See https://example.com/docs/file.py for details.", "URL is not a path"),
    ("If I modified x.py it would break.", "EN conditional"),
    ("I'm updating src/app.py right now.", "EN progressive is not a completed effect"),
    ("Modifica src/app.py y luego corre los tests.", "ES imperative to the user"),
]


@pytest.mark.parametrize("text,expected,note", CLAIMS_POSITIVE, ids=[c[2] for c in CLAIMS_POSITIVE])
def test_claims_positive(text: str, expected: list[tuple[str, int]], note: str):
    got = [(c.path, c.kind) for c in detect_mutation_claims(text)]
    assert got == expected, f"{note}: {got} vs {[(p, KIND_NAMES[k]) for p, k in expected]}"


@pytest.mark.parametrize("text,note", CLAIMS_NEGATIVE, ids=[c[1] for c in CLAIMS_NEGATIVE])
def test_claims_negative(text: str, note: str):
    got = detect_mutation_claims(text)
    assert got == [], f"{note}: unexpected claims {[(c.path, c.kind_name) for c in got]}"


def test_claims_carry_span_and_text():
    text = "Primero revisé el log. Luego modifiqué src/app.py sin problemas."
    (claim,) = detect_mutation_claims(text)
    assert claim.verb.lower() == "modifiqué"
    assert text[claim.span[0] : claim.span[1]] == claim.text
    assert claim.text.startswith("modifiqué")


# ---------------------------------------------------------------------------
# 4. Guard end to end
# ---------------------------------------------------------------------------


def test_guard_admits_real_effects_and_rejects_the_blocked_write(tmp_path: Path):
    """The headline scenario: three claims, two real, one about a file the
    agent never managed to write. Verdict names the untouched file and its
    unchanged hash."""
    _seed(tmp_path)
    guard = MutationGuard(tmp_path)
    assert guard.begin() == 1
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test(): pass\n", encoding="utf-8")
    text = "Modifiqué `src/app.py`, creé tests/test_app.py y actualicé config/settings.py con DEBUG=True."

    audit = guard.audit(text)
    assert not audit.approved
    by_path = {v.claim.path: v for v in audit.verdicts}
    assert by_path["src/app.py"].admitted and by_path["src/app.py"].reason is None
    assert by_path["tests/test_app.py"].admitted
    bad = by_path["config/settings.py"]
    assert not bad.admitted and bad.receipt is None
    assert "config/settings.py" in bad.reason
    assert "sha256 unchanged" in bad.reason
    assert audit.unreported == ()
    assert audit.turn == 1
    assert audit.latency_ns >= 0

    with pytest.raises(MutationTheaterError, match="config/settings.py"):
        guard.enforce(text, turn=1)


def test_guard_approves_honest_turn(tmp_path: Path):
    _seed(tmp_path)
    guard = MutationGuard(tmp_path)
    guard.begin()
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    audit = guard.audit("I updated src/app.py to bump x.")
    assert audit.approved and audit.reasons == ()
    assert guard.enforce("I updated src/app.py to bump x.") == "I updated src/app.py to bump x."


def test_guard_no_claims_no_effects_is_fine(tmp_path: Path):
    _seed(tmp_path)
    guard = MutationGuard(tmp_path, strict=True)
    guard.begin()
    audit = guard.audit("Revisé el código; no hay nada que cambiar.")
    assert audit.approved and audit.verdicts == () and audit.unreported == ()


def test_guard_kind_mismatch_is_rejected_with_the_real_effect_named(tmp_path: Path):
    """'I updated X' when X did not exist: the effect happened, the
    description of it is false. Law 2 of the rule."""
    _seed(tmp_path)
    guard = MutationGuard(tmp_path)
    guard.begin()
    (tmp_path / "config" / "local.py").write_text("A = 1\n", encoding="utf-8")
    audit = guard.audit("Actualicé config/local.py.")
    (v,) = audit.verdicts
    assert not v.admitted and v.receipt is not None and v.receipt.kind == "created"
    assert "receipt says created" in v.reason and "claim says modified" in v.reason


def test_guard_lenient_written_claim_accepts_created_or_modified(tmp_path: Path):
    _seed(tmp_path)
    guard = MutationGuard(tmp_path)
    guard.begin()
    (tmp_path / "config" / "local.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("x = 9\n", encoding="utf-8")
    audit = guard.audit("Escribí config/local.py y agregué una línea a src/app.py.")
    assert audit.approved, audit.reasons
    assert {v.claim.kind for v in audit.verdicts} == {KIND_WRITTEN}


def test_guard_deleted_claim_requires_deleted_receipt(tmp_path: Path):
    _seed(tmp_path)
    guard = MutationGuard(tmp_path)
    guard.begin()
    (tmp_path / "README.md").unlink()
    assert guard.audit("Eliminé README.md.").approved
    guard.begin()
    assert not guard.audit("Eliminé src/app.py.").approved


def test_guard_strict_rejects_silent_writes_lenient_reports_them(tmp_path: Path):
    _seed(tmp_path)
    guard = MutationGuard(tmp_path)
    guard.begin()
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    text = "No hice cambios en el código."

    lenient = guard.audit(text, strict=False)
    assert lenient.approved
    assert [r.path for r in lenient.unreported] == ["notes.txt"]
    assert lenient.unreported_rejected == ()

    strict = guard.audit(text, strict=True)
    assert not strict.approved
    assert [r.path for r in strict.unreported_rejected] == ["notes.txt"]
    assert "silent writes" in strict.reasons[0]


def test_guard_basename_claim_matches_a_unique_changed_file(tmp_path: Path):
    _seed(tmp_path)
    guard = MutationGuard(tmp_path)
    guard.begin()
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    assert guard.audit("Modifiqué app.py.").approved


def test_guard_ambiguous_basename_with_different_effects_is_rejected(tmp_path: Path):
    _seed(tmp_path)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "app.py").write_text("y\n", encoding="utf-8")
    guard = MutationGuard(tmp_path)
    guard.begin()
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "other" / "app.py").unlink()
    audit = guard.audit("Modifiqué app.py.")
    assert not audit.approved
    assert "several changed files" in audit.verdicts[0].reason


def test_guard_unknown_path_reason_is_exact(tmp_path: Path):
    _seed(tmp_path)
    guard = MutationGuard(tmp_path)
    guard.begin()
    audit = guard.audit("Creé src/ghost.py con el stub.")
    (v,) = audit.verdicts
    assert not v.admitted
    assert "does not exist in the workspace before or after the turn" in v.reason


def test_guard_ground_truth_block(tmp_path: Path):
    _seed(tmp_path)
    guard = MutationGuard(tmp_path)
    assert guard.begin() == 1
    assert guard.end() == []
    assert "NO file changed" in guard.ground_truth()
    # An empty turn leaves no receipt, yet the next turn must still be 2.
    assert guard.begin() == 2
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    guard.end()
    block = guard.ground_truth()
    assert "modified src/app.py" in block and "turn 2" in block


def test_guard_survives_total_context_amnesia(tmp_path: Path):
    """The central claim: enforcement never depended on anything the
    observing process remembered. A guard rebuilt from the JSONL ledger,
    in a different object with no snapshot in memory, reaches the same
    verdict for the same text."""
    _seed(tmp_path)
    ledger_path = tmp_path / ".netelpro" / "receipts.jsonl"
    first = MutationGuard(tmp_path)
    first.begin()
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    first.end()
    first.ledger.save(ledger_path)
    text = "Modifiqué src/app.py y actualicé config/settings.py."
    v1 = first.audit(text)

    second = MutationGuard(tmp_path, ledger=ReceiptLedger.load(ledger_path))
    v2 = second.audit(text)  # no begin(): resolves to the ledger's latest turn
    assert v2.turn == v1.turn == 1
    assert [(v.claim.path, v.admitted) for v in v2.verdicts] == [
        (v.claim.path, v.admitted) for v in v1.verdicts
    ]
    assert not v2.approved


def test_guard_end_before_begin_is_an_error(tmp_path: Path):
    with pytest.raises(LedgerError, match="begin"):
        MutationGuard(tmp_path).end()


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def _cli(root: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "netelpro.receipts", "--root", str(root), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        input=stdin,
        timeout=120,
    )


def test_cli_begin_end_audit_roundtrip(tmp_path: Path):
    _seed(tmp_path)
    r = _cli(tmp_path, "begin")
    assert r.returncode == 0 and "turn 1" in r.stdout

    (tmp_path / "src" / "app.py").write_text("x = 3\n", encoding="utf-8")
    r = _cli(tmp_path, "end")
    assert r.returncode == 0 and "1 receipt(s)" in r.stdout and "modified src/app.py" in r.stdout
    assert "hashed" in r.stdout and "reused from cache" in r.stdout
    assert (tmp_path / ".netelpro" / "receipts.jsonl").exists()

    r = _cli(tmp_path, "audit", "--text", "-", stdin="Actualicé src/app.py.")
    assert r.returncode == 0 and "APPROVED" in r.stdout

    r = _cli(tmp_path, "audit", "--text", "-", stdin="Actualicé src/app.py y corregí config/settings.py.")
    assert r.returncode == 2 and "DENIED" in r.stdout and "config/settings.py" in r.stdout

    r = _cli(tmp_path, "audit", "--json", "--text", "-", stdin="Corregí config/settings.py.")
    assert r.returncode == 2
    payload = json.loads(r.stdout)
    assert payload["approved"] is False and payload["claims"][0]["path"] == "config/settings.py"

    r = _cli(tmp_path, "show")
    assert r.returncode == 0 and "chain verified" in r.stdout


def test_cli_end_without_begin_fails_loudly(tmp_path: Path):
    _seed(tmp_path)
    r = _cli(tmp_path, "end")
    assert r.returncode == 1 and "begin" in r.stderr


def test_cli_turn_counter_advances_and_strict_flag(tmp_path: Path):
    _seed(tmp_path)
    _cli(tmp_path, "begin")
    _cli(tmp_path, "end")  # empty turn 1: no receipt, counter must still move
    r = _cli(tmp_path, "begin")
    assert "turn 2" in r.stdout
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _cli(tmp_path, "end")
    r = _cli(tmp_path, "begin")
    assert "turn 3" in r.stdout
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    _cli(tmp_path, "end")
    r = _cli(tmp_path, "audit", "--strict", "--text", "-", stdin="Nada que reportar.")
    assert r.returncode == 2 and "b.txt" in r.stdout and "a.txt" not in r.stdout


def test_cli_full_flag_rereads_everything(tmp_path: Path):
    _seed(tmp_path)
    _age(tmp_path)
    r = _cli(tmp_path, "begin")
    assert "3 files (3 hashed, 0 reused from cache)" in r.stdout
    time.sleep(2.2)
    # Take 2 is cached on take 1, which was inside the racy window of the
    # utime above: still a full read, by the rule. Take 3 is cached on take
    # 2, which postdates every ctime by more than the window: reuse.
    r = _cli(tmp_path, "begin")
    assert "3 files (3 hashed, 0 reused from cache)" in r.stdout
    r = _cli(tmp_path, "begin")
    assert "3 files (0 hashed, 3 reused from cache)" in r.stdout
    r = _cli(tmp_path, "--full", "begin")
    assert "3 files (3 hashed, 0 reused from cache)" in r.stdout
