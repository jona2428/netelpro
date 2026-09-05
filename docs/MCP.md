# Netelpro MCP Server

Status: **v0.1 wire contract** (2026-09-05). Implementation: `netelpro/mcp_server.py` — one module, standard library only. Contract proven by `netelpro/tests/test_mcp_adversarial.py` (21 cases). Language semantics, capabilities, and holes: see [SPEC.md](SPEC.md).

## 1. What it is

An MCP-style **stdio JSON-RPC 2.0 server** exposing the Netelpro compiler, evaluator, and verifier to LLM clients. Transport is **line-delimited JSON**: each line read from `stdin` is parsed as one JSON-RPC 2.0 message, and each response is written as a single newline-terminated JSON line to `stdout` and flushed (`_send_jsonrpc_response`). No `Content-Length` framing is required.

- `protocolVersion`: `2024-11-05`
- `serverInfo`: `{"name": "netelpro-mcp", "version": "0.1.0"}` — the server's own version, independent of the package version in `pyproject.toml`
- Methods: `initialize`, `notifications/initialized`, `ping`, `tools/list`, `tools/call`
- JSON-RPC error codes: `-32700` malformed JSON, `-32600` invalid request, `-32601` unknown method or tool, `-32602` invalid params, `-32603` internal error
- Zero external dependencies for the transport: the server loop uses only `json`, `os`, `subprocess`, `sys`, `tempfile`. The native backend needs `llvmlite`, but every native code path imports it lazily and reports a structured `codegen` error (`"llvmlite not available"`) if absent — the server itself never fails to start.

Domain failures are **not** JSON-RPC errors. A tool that ran and rejected the program answers with `isError: true` and a structured payload:

```json
{"ok": false, "errors": [{"phase": "cap", "line": 1, "col": 1, "message": "..."}], "hole_manifest": []}
```

where `phase ∈ {parse, cap, hole, runtime, limit, codegen}`, plus `rule_filter` for `netelpro_verify` compile-stage rejections. The same payload is mirrored in `structuredContent` and serialized into `content[0].text`. Notifications (requests without `id`) are executed but produce no response.

## 2. Launch

```bash
python -m netelpro --mcp          # CLI flag, wired in netelpro/__main__.py
python -m netelpro.mcp_server     # equivalent: module main() defaults to the stdio loop
```

`--mcp` takes precedence over file evaluation: `__main__.py` imports `_run_stdio_server` from `netelpro.mcp_server`, runs it until EOF on `stdin`, and returns `0`. The loop reconfigures both streams to UTF-8 and never writes anything to `stdout` except JSON-RPC responses.

Wire-in sequence (real captured lines; `-->` client to server, `<--` server to client):

```
--> {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}
<-- {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "netelpro-mcp", "version": "0.1.0"}}}
--> {"jsonrpc": "2.0", "method": "notifications/initialized"}
--> {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
<-- {"jsonrpc": "2.0", "id": 2, "result": {"tools": [ ...4 tools... ]}}
--> {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "netelpro_eval", "arguments": {"source": "(+ 1 2)"}}}
<-- {"jsonrpc": "2.0", "id": 3, "result": {"ok": true, "result": 3, "stdout": "", "errors": [], "hole_manifest": [], "content": [{"type": "text", "text": "{\"ok\": true, \"result\": 3, \"stdout\": \"\", \"errors\": [], \"hole_manifest\": []}"}], "structuredContent": {"ok": true, "result": 3, "stdout": "", "errors": [], "hole_manifest": []}, "isError": false}}
```

`notifications/initialized` sent as a true notification (no `id`) gets no reply; sent with an `id` it answers `{}`. `ping` answers `{}`. The examples below show only `structuredContent` — the envelope always adds `content`, `structuredContent`, and `isError`.

## 3. Tools

`tools/list` returns exactly these four entries (`TOOLS_LIST`); the schemas below are copied verbatim from the code and are the source of truth.

### 3.1 `netelpro_compile`

Statically compiles and audits Netelpro source code without execution. Runs parse, capability, and hole auditing, plus optional LLVM codegen validation.

```json
{
  "name": "netelpro_compile",
  "description": "Statically compiles and audits Netelpro source code without execution. Runs parse, capability, and hole auditing, plus optional LLVM codegen validation.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "source": {
        "type": "string",
        "description": "The Netelpro source code to compile and validate."
      },
      "backend": {
        "type": "string",
        "enum": ["interpreter", "native"],
        "default": "interpreter",
        "description": "Compilation backend target ('interpreter' or 'native')."
      }
    },
    "required": ["source"]
  }
}
```

Output: `{ok, errors, hole_manifest, grants}`. `grants` is the sorted list of capabilities granted by top-level `(grant ...)` forms; `hole_manifest` lists every `(sorry "...")` with coordinates and reason. With `"backend": "native"` the source is additionally run through `compile_program` (llvmlite required).

```
structuredContent: {"ok": true, "errors": [], "hole_manifest": [], "grants": []}
```

### 3.2 `netelpro_eval`

Executes Netelpro source code in an isolated subprocess with strict limits. Returns final evaluated value and captured stdout.

```json
{
  "name": "netelpro_eval",
  "description": "Executes Netelpro source code in an isolated subprocess with strict limits. Returns final evaluated value and captured stdout.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "source": {
        "type": "string",
        "description": "The Netelpro source code to execute."
      },
      "native": {
        "type": "boolean",
        "default": false,
        "description": "Whether to execute natively via LLVM JIT instead of the reference interpreter."
      }
    },
    "required": ["source"]
  }
}
```

Output: `{ok, result, stdout, errors, hole_manifest}`. `result` is the value of the last top-level expression, JSON-serialized (`nil` → `null`); `stdout` is everything the program printed, truncated to `RESULT_STRING_CAP`.

```
source: "(defn f (x) (* x x)) (f 3)"
structuredContent: {"ok": true, "result": 9, "stdout": "", "errors": [], "hole_manifest": []}

source: "(grant io) (print \"hi\")"
structuredContent: {"ok": true, "result": null, "stdout": "hi", "errors": [], "hole_manifest": []}
```

**Print capability requirement.** `print` is gated by the language's static capability system: the source must declare `(grant io)` at top level, otherwise the call is prosecuted as a compile error before anything runs (SPEC.md §5, §11 — capabilities as types):

```
source: "(print \"hi\")"
structuredContent: {"ok": false, "result": null, "stdout": "", "errors": [{"phase": "cap", "line": 1, "col": 1, "message": "capability 'io' required by 'print' but not granted — add (grant io) at top level"}], "hole_manifest": []}
```

With `"native": true` the program runs through the LLVM JIT; `result` is the machine-code return value and native `printf` output is captured from the worker's real fd 1.

### 3.3 `netelpro_verify`

Performs differential parity verification on a Netelpro rule filter (`defn filter-rule ...`), testing cases across native LLVM JIT and reference interpreter.

```json
{
  "name": "netelpro_verify",
  "description": "Performs differential parity verification on a Netelpro rule filter (defn filter-rule ...), testing cases across native LLVM JIT and reference interpreter.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "source": {
        "type": "string",
        "description": "Netelpro source code defining (defn filter-rule ...)."
      },
      "cases": {
        "type": "array",
        "description": "Array of test vectors with args and expected boolean result (max 100).",
        "items": {
          "type": "object",
          "properties": {
            "args": {
              "type": "array",
              "items": {"type": ["integer", "boolean", "string"]},
              "description": "Positional arguments to pass to filter-rule."
            },
            "expected": {
              "type": "boolean",
              "description": "Expected boolean decision outcome."
            }
          },
          "required": ["args", "expected"]
        }
      }
    },
    "required": ["source", "cases"]
  }
}
```

Output: `{ok, mismatches, manifest, errors}`. Each mismatch carries the args plus **both** engines' decisions — `interpreted` (reference) and `native` (JIT) — so a divergence is a fact, not a guess. `ok` is true only when `mismatches` is empty.

```
source: "(defn filter-rule (x) (> x 10))"
cases:  [{"args": [20], "expected": true}, {"args": [5], "expected": false}]
structuredContent: {"ok": true, "mismatches": [], "manifest": [], "errors": []}
```

### 3.4 `netelpro_spec`

Provides static language specification knowledge, special forms, primitives, capabilities, and sorry-hole mechanisms.

```json
{
  "name": "netelpro_spec",
  "description": "Provides static language specification knowledge, special forms, primitives, capabilities, and sorry-hole mechanisms.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "category": {
        "type": "string",
        "enum": ["all", "special_forms", "primitives", "capabilities"],
        "default": "all",
        "description": "Category of specification knowledge to inspect."
      },
      "query": {
        "type": "string",
        "description": "Optional search term to filter forms and capabilities."
      }
    }
  }
}
```

Output: `{version, forms, capabilities}` — `version` is `"0.1.0"`; `forms` merges `SPECIAL_FORMS` and `PRIMITIVES` (each entry: `arity`, `sig`, `desc`, and where applicable `scope`/`eval`/`phase`/`semantics`/`capabilities`); `capabilities` carries the capability-system summary, known capabilities (`io`), and the `sorry`-hole contract. `query` filters by name, signature, or description.

```
arguments: {"category": "special_forms", "query": "grant"}
structuredContent: {"version": "0.1.0", "forms": {"grant": {"arity": [1, null], "sig": "(grant cap...)", "scope": "top-level only", "phase": "grammar reserved, enforced in Phase 3", "desc": "Top-level capability declaration; grants effects (e.g. io) file-wide."}}, "capabilities": {}}
```

## 4. Limits and containment

Module-level constants, overridable by tests via attribute assignment:

| Constant | Value | Enforced on | Effect when exceeded |
|---|---|---|---|
| `MAX_SOURCE_BYTES` | 65536 | all tools, pre-parse | `phase: "limit"` — `source size (N bytes) exceeds MAX_SOURCE_BYTES (65536)` |
| `PARSE_DEPTH_BUDGET` | 64 | all tools, pre-parse | `phase: "limit"` with exact `line:col` — `bracket nesting depth 65 exceeds budget 64` |
| `EVAL_TIMEOUT_S` | 3.0 | `netelpro_eval`, `netelpro_verify` | child process killed; `phase: "limit"` — `evaluation timed out` |
| `MAX_CASES` | 100 | `netelpro_verify` | `phase: "limit"` — `cases count (101) exceeds MAX_CASES (100)` |
| `RESULT_STRING_CAP` | 65536 | `netelpro_eval` stdout capture | captured stdout truncated to 65536 characters |
| `native` | `false` (default) | `netelpro_eval`, `netelpro_compile` | native JIT runs only when explicitly requested |

Security rationale — why an LLM-facing eval tool is safe here:

- **eval/verify run in subprocesses.** The server never evaluates untrusted source in its own process: `tool_eval` and `tool_verify` spawn `python -m netelpro.mcp_server --exec-eval|--exec-verify --out <tempfile>` as a child, feed the payload over `stdin`, and read the structured result back from a temp JSON file. The parent stays responsive no matter what the child does.
- **Timeout kill returns `phase: "limit"`.** `communicate(timeout=EVAL_TIMEOUT_S)` → `TimeoutExpired` → `proc.kill()` → structured `limit` error. An infinite loop is a 3-second bounded cost, never a hang (proven by `test_tco_infinite_loop_terminates_with_limit`).
- **Native JIT deaths are contained in the subprocess.** Hard native failures — stack overflow, div-by-zero `exit(1)` (SPEC.md §13) — die inside the child; the parent converts the dead child into a structured `runtime` error and keeps serving. Catchable failures (`CodegenError`, `StrayError`, `RecursionError` → `"recursion depth exceeded"`) are converted to structured errors inside the worker itself.
- **Pre-parse gates run before any parse.** Byte size and bracket depth are checked on raw text, so pathological inputs never reach the recursive-descent parser.

The `--exec-eval` / `--exec-verify` worker modes are internal plumbing, not part of the client contract.

## 5. Adversarial test suite

`netelpro/tests/test_mcp_adversarial.py` — 21 cases, written contract-first against the wire contract (not the implementation), spawning the real server via `[sys.executable, "-m", "netelpro.mcp_server"]` with hard timeouts and watchdogs so no case can hang CI. Collection never errors: if the server module is missing or does not import, every test skips cleanly. Key scenarios:

- **Protocol edges:** `initialize` handshake; `ping` roundtrip; unknown method → `-32601`; malformed JSON line → `-32700`; wrong `jsonrpc` version must not crash the session; `id: null` notification; `tools/list` shape (≥ 4 tools, each with `name`/`description`/`inputSchema`); unknown tool name.
- **Semantics:** empty source evaluates to `nil` cleanly (`ok: true`); `(+ 1 2)`; `(defn f (x) (* x x)) (f 3)`; `(sorry "not implemented")` surfaces the hole manifest instead of executing to a clean value; unicode in string literals.
- **Resource exhaustion:** TCO infinite loop `(defn spin (n) (spin (+ n 1))) (spin 0)` terminates with `phase: "limit"`; exponential `str-cat` bomb terminates (limit or runtime, never a hang); 1000-deep bracket nesting → clean error with no traceback leak; source over 65536 bytes rejected; `netelpro_verify` with 101 cases rejected; verify string args containing quotes, newlines, and backslashes must not crash.
- **Session:** two sequential `tools/call`s leave no state between them; `netelpro_spec` describes at least `if`, `defn`, and `str-cat`.

## 6. Client integration

A minimal client speaking line-delimited JSON-RPC over a subprocess (verified against the real server):

```python
import json, subprocess, sys

proc = subprocess.Popen([sys.executable, "-m", "netelpro", "--mcp"],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        text=True, encoding="utf-8", bufsize=1)

def rpc(method, params, id_):
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": id_,
                                 "method": method, "params": params}) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())   # one line in, one line out

rpc("initialize", {"protocolVersion": "2024-11-05"}, 1)
proc.stdin.write('{"jsonrpc": "2.0", "method": "notifications/initialized"}\n')  # notification: no reply
print(rpc("tools/list", {}, 2)["result"]["tools"][0]["name"])                    # netelpro_compile
print(rpc("tools/call", {"name": "netelpro_eval",
      "arguments": {"source": "(+ 1 2)"}}, 3)["result"]["structuredContent"])    # {'ok': True, 'result': 3, ...}
proc.stdin.close()
```

Rules of the transport: one JSON message per line, flush after each write, notifications get no reply, and a domain rejection arrives as `isError: true` with `structuredContent.ok == false` — check `ok`, not the JSON-RPC layer, to judge the program.

## 7. In-process usage

The same tools are callable without the wire: `from netelpro.mcp_server import dispatch`, then `dispatch("netelpro_eval", {"source": "(+ 1 2)"})` returns the identical structured dict. Unknown tool names raise `ValueError`.