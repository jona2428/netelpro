"""Adversarial contract-first test corpus for the Netelpro MCP server.

This suite is written CONTRACT-FIRST against the documented MCP server contract
(netelpro/mcp_server.py, being written in parallel by another worker). It does
NOT depend on the server's implementation details -- only on the wire contract:

  - stdio JSON-RPC 2.0, line-delimited.
  - Methods: initialize, tools/list, tools/call, ping.
  - Errors: -32700 malformed JSON, -32601 unknown method, -32602 invalid params,
    -32603 internal.
  - Domain errors -> isError:true with JSON {ok:false, errors:[{phase, line, col,
    message}]} where phase in {parse, cap, hole, runtime, limit, codegen}.
  - Tools: netelpro_compile{source, backend?}, netelpro_eval{source, native?=false},
    netelpro_verify{source, cases[{args,expected}]}, netelpro_spec{category?, query?}.
  - Limits: MAX_SOURCE_BYTES=65536, PARSE_DEPTH_BUDGET=64, EVAL_TIMEOUT_S=3.0,
    MAX_CASES=100, string result cap 65536. eval/verify run in a subprocess;
    timeout -> phase 'limit', never a hang.

Valid .sl syntax is grounded by reading the real package (netelpro/evaluator.py,
netelpro/parser.py, netelpro/rule_filter.py, netelpro/caps.py, spec/arity_table.json).

Collection NEVER errors: if netelpro/mcp_server.py does not exist or does not
import, every test skips cleanly via pytest.skip.
"""
from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Server availability guard (collection must never error)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
SERVER_PATH = ROOT / "netelpro" / "mcp_server.py"

# Contract constants
MAX_SOURCE_BYTES = 65536
PARSE_DEPTH_BUDGET = 64
EVAL_TIMEOUT_S = 3.0
MAX_CASES = 100
STRING_RESULT_CAP = 65536

# Grounded valid .sl sources (verified against the real package).
VALID_ADD = "(+ 1 2)"
VALID_DEFN = "(defn f (x) (* x x)) (f 3)"
VALID_STRCAT = '(str-cat "a" "b")'
VALID_PRINT = '(grant io) (print "hi")'


def _server_available() -> bool:
    """Return True only if the server module exists AND imports cleanly."""
    if not SERVER_PATH.exists():
        return False
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("netelpro.mcp_server", SERVER_PATH)
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return True
    except Exception:
        return False


SERVER_AVAILABLE = _server_available()


# ---------------------------------------------------------------------------
# MCP stdio client with generous timeouts (cross-platform, no hangs)
# ---------------------------------------------------------------------------


class _LineReader:
    """Reads a single line from a stream with a hard timeout via a daemon thread."""

    def __init__(self, stream):
        self._stream = stream
        self._q = queue.Queue()

    def readline(self, timeout: float) -> str:
        def _read():
            try:
                line = self._stream.readline()
                self._q.put(("ok", line))
            except Exception as exc:  # pragma: no cover - defensive
                self._q.put(("err", exc))

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError(f"recv timed out after {timeout}s")
        kind, payload = self._q.get()
        if kind == "err":
            raise payload
        return payload


class MCPClient:
    """Line-delimited JSON-RPC 2.0 client over a subprocess's stdio pipes."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.reader = _LineReader(proc.stdout)
        self._lock = threading.Lock()

    def send(self, obj: dict) -> None:
        with self._lock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def recv(self, timeout: float = 10.0) -> dict:
        line = self.reader.readline(timeout)
        if not line:
            raise EOFError("server closed stdout (EOF)")
        return json.loads(line)

    def call(self, method: str, params: dict, timeout: float = 10.0) -> dict:
        """Send a request and return the parsed response dict."""
        self.send({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        return self.recv(timeout)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.proc.stdin.close()
        with contextlib.suppress(Exception):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except Exception:
            with contextlib.suppress(Exception):
                self.proc.kill()


def launch_server() -> MCPClient:
    """Spawn [sys.executable, '-m', 'netelpro.mcp_server'] with stdio pipes."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "netelpro.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(ROOT),
        env=env,
        text=True,
        bufsize=1,
    )
    return MCPClient(proc)


@pytest.fixture(scope="module")
def server():
    if not SERVER_AVAILABLE:
        pytest.skip("netelpro/mcp_server.py not available yet (being written in parallel)")
    client = launch_server()
    try:
        yield client
    finally:
        client.close()


def _domain_errors(resp: dict) -> list[dict]:
    """Extract the domain error list from a response, tolerating shape variance."""
    if not isinstance(resp, dict):
        return []
    result = resp.get("result")
    if isinstance(result, dict) and result.get("ok") is False:
        return result.get("errors", []) or []
    # Some servers may surface errors at top level
    if resp.get("error") is not None:
        return []
    return []


def _phases(resp: dict) -> list[str]:
    return [e.get("phase") for e in _domain_errors(resp) if isinstance(e, dict)]


# ---------------------------------------------------------------------------
# Protocol / handshake edges
# ---------------------------------------------------------------------------


def test_initialize_handshake(server):
    resp = server.call("initialize", {"protocolVersion": "2024-11-05"})
    assert "result" in resp or "error" in resp
    # A successful initialize must not be a domain error
    assert resp.get("error") is None


def test_ping_roundtrip(server):
    resp = server.call("ping", {})
    assert resp.get("error") is None
    assert "result" in resp


def test_unknown_method_is_error(server):
    resp = server.call("bogus_method", {})
    assert resp.get("error") is not None
    assert resp["error"].get("code") == -32601


def test_malformed_json_is_parse_error(server):
    # Send a complete line that is NOT valid JSON (must be rejected with -32700).
    server.proc.stdin.write('{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": }\n')
    server.proc.stdin.flush()
    resp = server.recv()
    assert resp.get("error") is not None
    assert resp["error"].get("code") == -32700


def test_wrong_jsonrpc_version(server):
    server.call("ping", {})
    # Send a request with a bogus jsonrpc version
    server.send({"jsonrpc": "1.0", "id": 2, "method": "ping", "params": {}})
    resp2 = server.recv()
    # Either rejected with an error or tolerated; must not crash the session
    assert isinstance(resp2, dict)


def test_id_null(server):
    server.send({"jsonrpc": "2.0", "id": None, "method": "ping", "params": {}})
    resp = server.recv()
    assert isinstance(resp, dict)


def test_tools_list_shape(server):
    resp = server.call("tools/list", {})
    assert resp.get("error") is None
    result = resp.get("result")
    assert isinstance(result, dict)
    tools = result.get("tools")
    assert isinstance(tools, list)
    assert len(tools) >= 4
    for t in tools:
        assert isinstance(t, dict)
        assert "name" in t
        assert "description" in t
        assert "inputSchema" in t
    names = {t["name"] for t in tools}
    assert {"netelpro_compile", "netelpro_eval", "netelpro_verify", "netelpro_spec"} <= names


# ---------------------------------------------------------------------------
# Domain error phases
# ---------------------------------------------------------------------------


def test_unknown_tool_name_is_error(server):
    resp = server.call("tools/call", {"name": "no_such_tool", "arguments": {}})
    assert resp.get("error") is not None


def test_empty_source(server):
    resp = server.call("tools/call", {"name": "netelpro_eval", "arguments": {"source": ""}})
    # Empty source evaluates to nil cleanly (valid fiscal program), never a hang.
    result = resp.get("result")
    assert isinstance(result, dict)
    assert result.get("ok") is True


def test_valid_add_eval(server):
    resp = server.call("tools/call", {"name": "netelpro_eval", "arguments": {"source": VALID_ADD}})
    assert resp.get("error") is None
    result = resp.get("result")
    assert isinstance(result, dict)
    assert result.get("ok") is True


def test_valid_defn_eval(server):
    resp = server.call("tools/call", {"name": "netelpro_eval", "arguments": {"source": VALID_DEFN}})
    assert resp.get("error") is None
    result = resp.get("result")
    assert isinstance(result, dict)
    assert result.get("ok") is True


def test_hole_manifest_surfaced_no_execution(server):
    # (sorry "...") is a legal hole: manifest surfaced, no execution.
    resp = server.call(
        "tools/call",
        {"name": "netelpro_eval", "arguments": {"source": '(sorry "not implemented")'}},
    )
    # Either a domain error with phase 'hole'/'runtime', or a result carrying a
    # manifest. It must NOT be a clean ok with a value.
    assert resp.get("error") is not None or _domain_errors(resp)


def test_unicode_in_string_literal(server):
    src = '(str-cat "héllo" " wörld")'
    resp = server.call("tools/call", {"name": "netelpro_eval", "arguments": {"source": src}})
    assert resp.get("error") is None
    result = resp.get("result")
    assert isinstance(result, dict)
    assert result.get("ok") is True


# ---------------------------------------------------------------------------
# Limits / resource exhaustion (must terminate, never hang)
# ---------------------------------------------------------------------------


def test_tco_infinite_loop_terminates_with_limit(server):
    """(defn spin [n] (spin (+ n 1))) (spin 0) must terminate with phase 'limit'."""
    src = "(defn spin (n) (spin (+ n 1))) (spin 0)"
    # Watchdog: hard-kill the subprocess if it hangs beyond a generous bound.
    watchdog = threading.Timer(EVAL_TIMEOUT_S + 8.0, lambda: server.proc.kill())
    watchdog.daemon = True
    watchdog.start()
    try:
        resp = server.call(
            "tools/call",
            {"name": "netelpro_eval", "arguments": {"source": src}},
            timeout=EVAL_TIMEOUT_S + 10.0,
        )
    finally:
        watchdog.cancel()
    # Must be a domain error (isError) with phase 'limit' -- never a hang.
    assert resp.get("error") is not None or _domain_errors(resp)
    assert "limit" in _phases(resp)


def test_exponential_strcat_bomb_terminates(server):
    """Exponential str-cat bomb must terminate (limit or runtime), never hang."""
    src = (
        '(defn bomb [n] (if (== n 0) "x" (str-cat (bomb (- n 1)) (bomb (- n 1))))) '
        "(bomb 20)"
    )
    watchdog = threading.Timer(EVAL_TIMEOUT_S + 8.0, lambda: server.proc.kill())
    watchdog.daemon = True
    watchdog.start()
    try:
        resp = server.call(
            "tools/call",
            {"name": "netelpro_eval", "arguments": {"source": src}},
            timeout=EVAL_TIMEOUT_S + 10.0,
        )
    finally:
        watchdog.cancel()
    assert resp.get("error") is not None or _domain_errors(resp)


def test_deep_nesting_clean_error(server):
    """'(' * 1000 must be a clean parse/limit error, no traceback."""
    src = "(" * 1000 + ")" * 1000
    resp = server.call("tools/call", {"name": "netelpro_eval", "arguments": {"source": src}})
    assert resp.get("error") is not None or _domain_errors(resp)
    # No raw traceback text should leak into the response
    blob = json.dumps(resp)
    assert "Traceback" not in blob


def test_source_over_65536_bytes_rejected(server):
    src = "(+ 1 2)\n" * (MAX_SOURCE_BYTES // 8 + 1)
    assert len(src.encode("utf-8")) > MAX_SOURCE_BYTES
    resp = server.call("tools/call", {"name": "netelpro_eval", "arguments": {"source": src}})
    assert resp.get("error") is not None or _domain_errors(resp)


def test_verify_101_cases_rejected(server):
    cases = [{"args": [1, 2], "expected": 3} for _ in range(MAX_CASES + 1)]
    resp = server.call(
        "tools/call",
        {"name": "netelpro_verify", "arguments": {"source": VALID_ADD, "cases": cases}},
    )
    assert resp.get("error") is not None or _domain_errors(resp)


def test_verify_string_args_with_quotes_newlines_escapes(server):
    """verify string args containing quotes/newlines/escapes must not crash."""
    cases = [
        {"args": ['he said "hi"\nline2\tend'], "expected": True},
        {"args": ["back\\slash"], "expected": False},
    ]
    resp = server.call(
        "tools/call",
        {"name": "netelpro_verify", "arguments": {"source": VALID_STRCAT, "cases": cases}},
    )
    # Must respond (error or result), never hang or crash the session.
    assert isinstance(resp, dict)


# ---------------------------------------------------------------------------
# Session state / sequential calls
# ---------------------------------------------------------------------------


def test_two_sequential_calls_state_clean(server):
    r1 = server.call("tools/call", {"name": "netelpro_eval", "arguments": {"source": VALID_ADD}})
    r2 = server.call("tools/call", {"name": "netelpro_eval", "arguments": {"source": VALID_DEFN}})
    assert r1.get("error") is None
    assert r2.get("error") is None
    assert r1.get("result", {}).get("ok") is True
    assert r2.get("result", {}).get("ok") is True


# ---------------------------------------------------------------------------
# netelpro_spec
# ---------------------------------------------------------------------------


def test_spec_returns_entries_for_if_defn_strcat(server):
    resp = server.call("tools/call", {"name": "netelpro_spec", "arguments": {}})
    assert resp.get("error") is None
    result = resp.get("result")
    assert isinstance(result, dict)
    # The spec must describe at least if/defn/str-cat.
    blob = json.dumps(result)
    for head in ("if", "defn", "str-cat"):
        assert head in blob
