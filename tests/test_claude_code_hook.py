"""netelpro.hooks.claude_code -- receipts as Claude Code hooks.

Every hook test runs the real module in a subprocess with the JSON Claude
Code would put on stdin, and asserts on exit code + stdout JSON, exactly
what Claude Code reads. The install tests check the settings merge is
idempotent and never touches hooks that are not ours.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from netelpro.hooks.claude_code import hook_commands, merge_settings
from netelpro.receipts import LEDGER_FILE, SNAPSHOT_FILE, STATE_DIR, ReceiptLedger

REPO_ROOT = Path(__file__).resolve().parent.parent


def _seed(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / "config").mkdir()
    (root / "config" / "settings.py").write_text("DEBUG = False\n", encoding="utf-8")


def _hook(root: Path, cmd: str, payload: dict, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "netelpro.hooks.claude_code", cmd, *flags],
        cwd=str(REPO_ROOT),
        input=json.dumps({"cwd": str(root), "session_id": "s1", **payload}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def _out(r: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(r.stdout) if r.stdout.strip() else {}


def _begin(root: Path, event: str = "UserPromptSubmit") -> dict:
    r = _hook(root, "begin", {"hook_event_name": event})
    assert r.returncode == 0, r.stderr
    return _out(r)


def _stop(root: Path, message: str, *, retry: bool = False, strict: bool = False) -> subprocess.CompletedProcess[str]:
    flags = ("--strict",) if strict else ()
    return _hook(root, "stop", {"hook_event_name": "Stop", "stop_hook_active": retry, "last_assistant_message": message}, *flags)


# ---------------------------------------------------------------------------
# begin
# ---------------------------------------------------------------------------


def test_begin_takes_baseline_and_injects_context(tmp_path: Path):
    _seed(tmp_path)
    out = _begin(tmp_path)
    ctx = out["hookSpecificOutput"]
    assert ctx["hookEventName"] == "UserPromptSubmit"
    assert "turn 1" in ctx["additionalContext"] and "sha256 receipts" in ctx["additionalContext"]
    assert (tmp_path / STATE_DIR / SNAPSHOT_FILE).exists()
    out2 = _begin(tmp_path, "SessionStart")
    assert out2["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "turn 2" in out2["hookSpecificOutput"]["additionalContext"]


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def test_stop_approves_honest_message_and_commits(tmp_path: Path):
    _seed(tmp_path)
    _begin(tmp_path)
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    r = _stop(tmp_path, "Modifiqué `src/app.py` para subir x.")
    assert r.returncode == 0 and r.stdout.strip() == "" and r.stderr == ""
    ledger = ReceiptLedger.load(tmp_path / STATE_DIR / LEDGER_FILE)
    assert [(x.kind, x.path, x.turn) for x in ledger.receipts] == [("modified", "src/app.py", 1)]


def test_stop_blocks_theater_with_reason_and_does_not_commit(tmp_path: Path):
    _seed(tmp_path)
    _begin(tmp_path)
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    r = _stop(tmp_path, "Modifiqué src/app.py y actualicé config/settings.py con DEBUG=True.")
    assert r.returncode == 0, r.stderr
    out = _out(r)
    assert out["decision"] == "block"
    reason = out["reason"]
    assert "config/settings.py" in reason and "sha256 unchanged" in reason
    assert "modified src/app.py" in reason  # ground truth is in the reason
    assert "perform the edit" in reason
    assert not (tmp_path / STATE_DIR / LEDGER_FILE).exists()  # the turn stays open


def test_stop_correction_round_then_commit(tmp_path: Path):
    _seed(tmp_path)
    _begin(tmp_path)
    (tmp_path / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    assert _out(_stop(tmp_path, "Actualicé config/settings.py."))["decision"] == "block"
    # The model performs the edit it described, then stops again (retry).
    (tmp_path / "config" / "settings.py").write_text("DEBUG = True\n", encoding="utf-8")
    r = _stop(tmp_path, "Modifiqué src/app.py y actualicé config/settings.py.", retry=True)
    assert r.returncode == 0 and r.stdout.strip() == ""
    ledger = ReceiptLedger.load(tmp_path / STATE_DIR / LEDGER_FILE)
    assert sorted(x.path for x in ledger.receipts) == ["config/settings.py", "src/app.py"]


def test_stop_never_loops_second_rejection_warns_the_user(tmp_path: Path):
    _seed(tmp_path)
    _begin(tmp_path)
    r = _stop(tmp_path, "Actualicé config/settings.py.", retry=True)
    assert r.returncode == 0
    out = _out(r)
    assert "decision" not in out
    assert "config/settings.py" in out["systemMessage"] and "STILL" in out["systemMessage"]
    # Committed despite the lie: the turn is closed, the ledger file exists
    # (empty: nothing changed), and the next begin numbers turn 2.
    assert (tmp_path / STATE_DIR / LEDGER_FILE).exists()
    assert ReceiptLedger.load(tmp_path / STATE_DIR / LEDGER_FILE).receipts == ()
    assert "turn 2" in _begin(tmp_path)["hookSpecificOutput"]["additionalContext"]


def test_stop_strict_blocks_silent_writes(tmp_path: Path):
    _seed(tmp_path)
    _begin(tmp_path)
    (tmp_path / "notes.txt").write_text("quiet", encoding="utf-8")
    lenient = _stop(tmp_path, "No cambié nada.")
    assert _out(lenient) == {}
    _begin(tmp_path)
    (tmp_path / "notes2.txt").write_text("quiet", encoding="utf-8")
    strict = _stop(tmp_path, "No cambié nada.", strict=True)
    assert _out(strict)["decision"] == "block" and "notes2.txt" in _out(strict)["reason"]


def test_stop_without_baseline_fails_open_and_takes_one(tmp_path: Path):
    _seed(tmp_path)
    r = _stop(tmp_path, "Actualicé config/settings.py.")
    assert r.returncode == 0
    out = _out(r)
    assert "decision" not in out and "no baseline" in out["systemMessage"]
    assert (tmp_path / STATE_DIR / SNAPSHOT_FILE).exists()


def test_stop_empty_message_is_fine(tmp_path: Path):
    _seed(tmp_path)
    _begin(tmp_path)
    r = _stop(tmp_path, "")
    assert r.returncode == 0 and r.stdout.strip() == ""


def test_stop_falls_back_to_transcript_when_message_field_absent(tmp_path: Path):
    _seed(tmp_path)
    _begin(tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    entries = [
        {"type": "user", "message": {"role": "user", "content": "haz el cambio"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "Write"}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Listo, actualicé config/settings.py."}]}},
        {"type": "attachment"},
    ]
    transcript.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    r = _hook(tmp_path, "stop", {"hook_event_name": "Stop", "stop_hook_active": False, "transcript_path": str(transcript)})
    assert _out(r)["decision"] == "block" and "config/settings.py" in _out(r)["reason"]


def test_internal_error_never_blocks(tmp_path: Path):
    _seed(tmp_path)
    _begin(tmp_path)
    (tmp_path / STATE_DIR / LEDGER_FILE).write_text("{broken\n", encoding="utf-8")
    r = _stop(tmp_path, "Actualicé config/settings.py.")
    assert r.returncode == 1 and r.stdout.strip() == "" and "not JSON" in r.stderr


def test_bad_stdin_is_a_non_blocking_error(tmp_path: Path):
    r = subprocess.run(
        [sys.executable, "-m", "netelpro.hooks.claude_code", "stop"],
        cwd=str(REPO_ROOT), input="not json", capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 1 and r.stdout.strip() == ""


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------


def _install(root: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "netelpro.hooks.claude_code", "install", "--root", str(root), *flags],
        cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", timeout=60,
    )


def _ours(settings: dict, event: str) -> list[str]:
    return [
        h["command"]
        for g in settings.get("hooks", {}).get(event, [])
        for h in g["hooks"]
        if "netelpro.hooks.claude_code" in h["command"]
    ]


def test_install_writes_three_events_and_gitignore(tmp_path: Path):
    r = _install(tmp_path)
    assert r.returncode == 0, r.stderr
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    for event in ("SessionStart", "UserPromptSubmit"):
        (cmd,) = _ours(settings, event)
        assert cmd.endswith("-m netelpro.hooks.claude_code begin")
    (stop_cmd,) = _ours(settings, "Stop")
    assert stop_cmd.endswith("-m netelpro.hooks.claude_code stop")
    assert sys.executable.split("/")[-1] in stop_cmd
    hook = settings["hooks"]["Stop"][0]["hooks"][0]
    assert hook["type"] == "command" and hook["timeout"] == 120
    assert ".netelpro/" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


def test_install_is_idempotent_and_updates_flags(tmp_path: Path):
    _install(tmp_path)
    _install(tmp_path, "--strict")
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert len(_ours(settings, "Stop")) == 1 and _ours(settings, "Stop")[0].endswith("stop --strict")
    assert len(_ours(settings, "UserPromptSubmit")) == 1
    gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert gi.count(".netelpro/") == 1


def test_install_preserves_foreign_hooks_and_settings(tmp_path: Path):
    (tmp_path / ".claude").mkdir()
    existing = {
        "permissions": {"allow": ["Bash(ls *)"]},
        "hooks": {
            "Stop": [{"hooks": [{"type": "command", "command": "./lint.sh"}]}],
            "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "./guard.sh"}]}],
        },
    }
    (tmp_path / ".claude" / "settings.json").write_text(json.dumps(existing), encoding="utf-8")
    assert _install(tmp_path).returncode == 0
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["permissions"] == existing["permissions"]
    assert settings["hooks"]["PreToolUse"] == existing["hooks"]["PreToolUse"]
    stop_cmds = [h["command"] for g in settings["hooks"]["Stop"] for h in g["hooks"]]
    assert "./lint.sh" in stop_cmds and len(_ours(settings, "Stop")) == 1

    r = subprocess.run(
        [sys.executable, "-m", "netelpro.hooks.claude_code", "uninstall", "--root", str(tmp_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 0
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["hooks"] == existing["hooks"]


def test_install_local_and_refuses_invalid_json(tmp_path: Path):
    assert _install(tmp_path, "--local").returncode == 0
    assert (tmp_path / ".claude" / "settings.local.json").exists()
    (tmp_path / ".claude" / "settings.json").write_text("{nope", encoding="utf-8")
    r = _install(tmp_path)
    assert r.returncode == 1 and "not touching it" in r.stderr
    assert (tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8") == "{nope"


def test_merge_settings_pure_function_roundtrip():
    cmds = hook_commands("/usr/bin/python3", strict=False)
    merged = merge_settings({}, cmds)
    assert set(merged["hooks"]) == {"SessionStart", "UserPromptSubmit", "Stop"}
    assert merge_settings(merged, None) == {}
    spaced = hook_commands("/Users/jona/My Env/bin/python", strict=True)
    assert spaced["Stop"].startswith('"/Users/jona/My Env/bin/python" -m') and spaced["Stop"].endswith("--strict")


@pytest.mark.parametrize("event", ["SessionStart", "UserPromptSubmit", "Stop"])
def test_installed_commands_are_runnable(tmp_path: Path, event: str):
    """The exact command written into settings.json must run from a shell
    with the hook's stdin, not just look right."""
    _seed(tmp_path)
    _install(tmp_path)
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    (cmd,) = _ours(settings, event)
    payload = {"cwd": str(tmp_path), "hook_event_name": event, "stop_hook_active": False, "last_assistant_message": "Revisé el código."}
    r = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT), input=json.dumps(payload), capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr
