"""netelpro_receipts over the MCP server: the read path from INSIDE the agent.

Two layers, the second over the real stdio process:

1. `dispatch()` in-process: configuration is server-side only (no root ->
   fail-closed error, no silent empty baseline), actions are read-only
   (never a ledger write, no 'begin' action that could move the baseline
   after the model wrote), text is size-capped, and the verdict matches
   `MutationGuard` for the same text.
2. Wire: the real server spawned with NETELPRO_RECEIPTS_ROOT takes its
   baseline BEFORE the first request, so a file written during the session
   shows up as an effect and a claim about an unwritten file is rejected.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from netelpro import mcp_server
from netelpro.receipts import LEDGER_FILE, SNAPSHOT_FILE, STATE_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent


def _seed(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / "config").mkdir()
    (root / "config" / "settings.py").write_text("DEBUG = False\n", encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _seed(tmp_path)
    monkeypatch.setenv(mcp_server.RECEIPTS_ROOT_ENV, str(tmp_path))
    return tmp_path


def _errors(res: dict) -> list[str]:
    return [e["message"] for e in res.get("errors", [])]


# ---------------------------------------------------------------------------
# 1. in-process dispatch
# ---------------------------------------------------------------------------


def test_unconfigured_root_fails_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(mcp_server.RECEIPTS_ROOT_ENV, raising=False)
    res = mcp_server.dispatch("netelpro_receipts", {"action": "show"})
    assert res["ok"] is False
    assert mcp_server.RECEIPTS_ROOT_ENV in _errors(res)[0]
    assert res["errors"][0]["phase"] == "receipts"


def test_root_not_a_directory_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(mcp_server.RECEIPTS_ROOT_ENV, str(tmp_path / "nope"))
    res = mcp_server.dispatch("netelpro_receipts", {"action": "show"})
    assert res["ok"] is False and "not a directory" in _errors(res)[0]


def test_tool_is_listed_with_schema():
    (tool,) = [t for t in mcp_server.TOOLS_LIST if t["name"] == "netelpro_receipts"]
    props = tool["inputSchema"]["properties"]
    assert props["action"]["enum"] == ["show", "audit"]
    assert set(props) == {"action", "text", "strict"}
    assert "read-only" in tool["description"].lower()


def test_show_reports_live_effects_and_persists_the_baseline(workspace: Path):
    first = mcp_server.dispatch("netelpro_receipts", {})
    assert first["ok"] and first["effects"] == [] and first["turn"] == 1
    assert (workspace / STATE_DIR / SNAPSHOT_FILE).exists()
    assert "NO file changed" in first["ground_truth"]

    (workspace / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    (workspace / "new.txt").write_text("n", encoding="utf-8")
    res = mcp_server.dispatch("netelpro_receipts", {"action": "show"})
    assert [(e["kind"], e["path"]) for e in res["effects"]] == [("created", "new.txt"), ("modified", "src/app.py")]
    assert "modified src/app.py" in res["ground_truth"]
    assert res["root"] == str(workspace.resolve())


def test_tool_never_writes_the_ledger_and_cannot_move_the_baseline(workspace: Path):
    mcp_server.dispatch("netelpro_receipts", {})
    (workspace / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    for _ in range(3):
        res = mcp_server.dispatch("netelpro_receipts", {"action": "show"})
        assert [e["path"] for e in res["effects"]] == ["src/app.py"]
    assert not (workspace / STATE_DIR / LEDGER_FILE).exists()
    res = mcp_server.dispatch("netelpro_receipts", {"action": "begin"})
    assert res["ok"] is False and "unknown action" in _errors(res)[0]
    res = mcp_server.dispatch("netelpro_receipts", {"action": "end"})
    assert res["ok"] is False


def test_audit_matches_the_guard_verdict(workspace: Path):
    mcp_server.dispatch("netelpro_receipts", {})
    (workspace / "src" / "app.py").write_text("x = 2\n", encoding="utf-8")
    text = "Modifiqué src/app.py y actualicé config/settings.py."
    res = mcp_server.dispatch("netelpro_receipts", {"action": "audit", "text": text})
    assert res["ok"] is True and res["approved"] is False
    assert [(c["path"], c["admitted"]) for c in res["claims"]] == [("src/app.py", True), ("config/settings.py", False)]
    assert res["claims"][0]["receipt"]["kind"] == "modified"
    assert res["claims"][1]["receipt"] is None
    assert "sha256 unchanged" in res["claims"][1]["reason"]
    assert res["reasons"] == [res["claims"][1]["reason"]]
    assert res["unreported"] == []


def test_audit_strict_rejects_silent_writes(workspace: Path):
    mcp_server.dispatch("netelpro_receipts", {})
    (workspace / "quiet.txt").write_text("q", encoding="utf-8")
    lenient = mcp_server.dispatch("netelpro_receipts", {"action": "audit", "text": "No cambié nada."})
    assert lenient["approved"] is True and [r["path"] for r in lenient["unreported"]] == ["quiet.txt"]
    strict = mcp_server.dispatch("netelpro_receipts", {"action": "audit", "text": "No cambié nada.", "strict": True})
    assert strict["approved"] is False and [r["path"] for r in strict["unreported_rejected"]] == ["quiet.txt"]


def test_audit_input_validation(workspace: Path):
    res = mcp_server.dispatch("netelpro_receipts", {"action": "audit", "text": "   "})
    assert res["ok"] is False and "non-empty" in _errors(res)[0]
    res = mcp_server.dispatch("netelpro_receipts", {"action": "audit", "text": 42})
    assert res["ok"] is False and "string" in _errors(res)[0]
    big = "a" * (mcp_server.MAX_SOURCE_BYTES + 1)
    res = mcp_server.dispatch("netelpro_receipts", {"action": "audit", "text": big})
    assert res["ok"] is False and res["errors"][0]["phase"] == "limit"


def test_harness_baseline_from_cli_begin_is_respected(workspace: Path):
    """When the harness ran `netelpro-receipts begin`, the tool diffs against
    THAT baseline and reports its turn number, never re-hashing its own."""
    subprocess.run(
        [sys.executable, "-m", "netelpro.receipts", "--root", str(workspace), "begin"],
        cwd=str(REPO_ROOT), check=True, capture_output=True, text=True,
    )
    (workspace / "src" / "app.py").write_text("x = 3\n", encoding="utf-8")
    res = mcp_server.dispatch("netelpro_receipts", {"action": "show"})
    assert res["turn"] == 1 and [e["path"] for e in res["effects"]] == ["src/app.py"]


def test_corrupt_ledger_is_a_structured_error(workspace: Path):
    state = workspace / STATE_DIR
    state.mkdir()
    (state / LEDGER_FILE).write_text("{not json\n", encoding="utf-8")
    res = mcp_server.dispatch("netelpro_receipts", {"action": "show"})
    assert res["ok"] is False and "not JSON" in _errors(res)[0]


# ---------------------------------------------------------------------------
# 2. over the wire
# ---------------------------------------------------------------------------


class _Server:
    def __init__(self, env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "netelpro.mcp_server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", cwd=str(REPO_ROOT), env=env,
        )
        self._id = 0

    def call(self, method: str, params: dict) -> dict:
        self._id += 1
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self) -> str:
        assert self.proc.stdin and self.proc.stderr
        self.proc.stdin.close()
        self.proc.wait(timeout=30)
        return self.proc.stderr.read()


def test_wire_baseline_predates_the_first_request(tmp_path: Path):
    _seed(tmp_path)
    env = {**os.environ, mcp_server.RECEIPTS_ROOT_ENV: str(tmp_path)}
    server = _Server(env)
    try:
        init = server.call("initialize", {"protocolVersion": "2024-11-05"})
        assert init["result"]["serverInfo"]["name"] == "netelpro-mcp"
        # Baseline must already exist: taken at server start, not at first call.
        assert (tmp_path / STATE_DIR / SNAPSHOT_FILE).exists()
        names = {t["name"] for t in server.call("tools/list", {})["result"]["tools"]}
        assert "netelpro_receipts" in names

        (tmp_path / "config" / "settings.py").write_text("DEBUG = True\n", encoding="utf-8")
        resp = server.call(
            "tools/call",
            {"name": "netelpro_receipts", "arguments": {"action": "audit", "text": "Actualicé config/settings.py y creé src/new.py."}},
        )
        assert resp.get("error") is None
        result = resp["result"]
        assert result["isError"] is False
        body = result["structuredContent"]
        assert body["approved"] is False
        assert [(c["path"], c["admitted"]) for c in body["claims"]] == [("config/settings.py", True), ("src/new.py", False)]
        assert "does not exist in the workspace" in body["claims"][1]["reason"]
        assert json.loads(result["content"][0]["text"]) == body
    finally:
        stderr = server.close()
    assert stderr.strip() == ""


def test_wire_unconfigured_server_still_serves_other_tools(tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if k != mcp_server.RECEIPTS_ROOT_ENV}
    server = _Server(env)
    try:
        server.call("initialize", {"protocolVersion": "2024-11-05"})
        resp = server.call("tools/call", {"name": "netelpro_receipts", "arguments": {}})
        assert resp["result"]["isError"] is True
        assert mcp_server.RECEIPTS_ROOT_ENV in resp["result"]["errors"][0]["message"]
        resp = server.call("tools/call", {"name": "netelpro_eval", "arguments": {"source": "(+ 1 2)"}})
        assert resp["result"]["structuredContent"]["result"] == 3
    finally:
        server.close()
