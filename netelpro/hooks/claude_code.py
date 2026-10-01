"""Netelpro receipts as Claude Code hooks.

Two minutes to install, nothing to wrap by hand:

    python -m netelpro.hooks.claude_code install            # in the repo
    python -m netelpro.hooks.claude_code install --strict   # silent writes block too
    python -m netelpro.hooks.claude_code uninstall

`install` merges three hooks into `.claude/settings.json` (or
`.claude/settings.local.json` with `--local`), keeping whatever hooks were
already there, and adds `.netelpro/` to `.gitignore`:

    SessionStart, UserPromptSubmit  ->  `... claude_code begin`
    Stop                            ->  `... claude_code stop [--strict]`

What each one does, per the hooks contract (https://code.claude.com/docs/en/hooks):

**begin** (every prompt): hashes the workspace under `cwd` as this turn's
baseline (incremental: one stat per file, only changed bytes re-read) and
returns `additionalContext` telling the model, in two lines, that any
statement about a file it created / modified / deleted will be checked
against sha256 receipts when it stops.

**stop** (every time the model finishes a turn): audits
`last_assistant_message` against the live diff since the baseline.
  - Approved: receipts are committed to the ledger, exit 0, silent.
  - Rejected, first time (`stop_hook_active` false): `{"decision": "block",
    "reason": ...}`. The reason names every claim with no receipt (file and
    unchanged hash), prints the ground truth, and tells the model to either
    perform the edit it described or correct the message. Nothing is
    committed yet: the turn is still open.
  - Rejected again (`stop_hook_active` true): the model already had its
    correction round. The stop is allowed (no infinite loop), receipts are
    committed, and `systemMessage` warns the user which claims the bytes
    still do not support.

Deliberate fail-open cases, so the hook can never hold a session hostage
over its own problems: no baseline for this turn (installed mid-session)
takes one now and lets the stop through with a `systemMessage`; any
internal error (corrupt ledger, missing llvmlite) is printed to stderr with
exit 1, which Claude Code shows as a non-blocking hook error. The ONLY thing
that blocks is a real audit verdict from the compiled rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from netelpro.receipts import DEFAULT_IGNORE, STATE_DIR, LedgerError, MutationAudit, open_turn, snapshot

MODULE = "netelpro.hooks.claude_code"
_MARK = f"-m {MODULE}"
_EVENTS_BEGIN = ("SessionStart", "UserPromptSubmit")
_EVENT_STOP = "Stop"
_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# hook input / output helpers
# ---------------------------------------------------------------------------


def _read_input() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LedgerError(f"hook stdin is not JSON: {e}") from e
    return data if isinstance(data, dict) else {}


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _root(data: dict[str, Any], override: str | None) -> Path:
    return Path(override or data.get("cwd") or os.getcwd()).resolve()


def last_assistant_text(data: dict[str, Any]) -> str:
    """The final assistant text of the turn: the `last_assistant_message`
    field when the host provides it, else the last assistant entry's text
    blocks from the transcript JSONL (older hosts)."""
    text = data.get("last_assistant_message")
    if isinstance(text, str):
        return text
    path = data.get("transcript_path")
    if not isinstance(path, str) or not Path(path).is_file():
        return ""
    last = ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict) or entry.get("type") != "assistant":
                continue
            content = (entry.get("message") or {}).get("content")
            if isinstance(content, str):
                parts = [content]
            elif isinstance(content, list):
                parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            else:
                parts = []
            joined = "\n".join(p for p in parts if p)
            if joined:
                last = joined
    return last


# ---------------------------------------------------------------------------
# begin: baseline + a two-line notice in the model's context
# ---------------------------------------------------------------------------


def _cmd_begin(data: dict[str, Any], args: argparse.Namespace) -> int:
    from netelpro.receipts import ReceiptLedger, load_snapshot, save_snapshot

    root = _root(data, args.root)
    state = root / STATE_DIR
    ledger = ReceiptLedger.load(state / "receipts.jsonl")
    previous = load_snapshot(state / "snapshot.json")
    turn = max(ledger.latest_turn or 0, previous[0] if previous else 0) + 1
    files = snapshot(root, DEFAULT_IGNORE, cache=previous[1] if previous else None)
    save_snapshot(state / "snapshot.json", turn, files)
    event = data.get("hook_event_name") or "UserPromptSubmit"
    notice = (
        f"[netelpro receipts] turn {turn}: {len(files)} files under {root} hashed as baseline. "
        "When you stop, every statement that you created, modified or deleted a file is checked "
        "against sha256 receipts of what actually changed; a claim with no receipt blocks the stop "
        "and names the file. Describe only edits that landed."
    )
    _emit({"hookSpecificOutput": {"hookEventName": event, "additionalContext": notice}})
    return 0


# ---------------------------------------------------------------------------
# stop: audit the final message against the receipts
# ---------------------------------------------------------------------------


def _block_reason(audit: MutationAudit, ground_truth: str) -> str:
    lines = ["[netelpro receipts] your final message asserts file effects the bytes do not show:"]
    for v in audit.rejected:
        lines.append(f"  - {v.reason}")
    for r in audit.unreported_rejected:
        lines.append(
            f"  - unreported effect: {r.kind} '{r.path}' is not mentioned in your message (strict mode)"
        )
    lines.append("")
    lines.append(ground_truth)
    lines.append("")
    lines.append(
        "Either perform the edit you described (if the user wanted it) or correct the message so it "
        "only describes effects that actually happened. Then stop again."
    )
    return "\n".join(lines)


def _cmd_stop(data: dict[str, Any], args: argparse.Namespace) -> int:
    root = _root(data, args.root)
    text = last_assistant_text(data)
    retry = bool(data.get("stop_hook_active", False))

    state = open_turn(root, strict=args.strict, take_baseline_if_missing=True)
    if not state.baseline_existed:
        # Installed mid-session: nothing to judge this turn. Fail open, say so.
        state.commit()
        _emit({"systemMessage": f"netelpro receipts: no baseline for this turn; baseline of {len(state.guard.after or {})} files taken now, auditing starts next turn."})
        return 0

    if not text.strip():
        state.commit()
        return 0

    audit = state.guard.audit(text, turn=state.turn, strict=args.strict)
    if audit.approved:
        state.commit()
        return 0

    truth = state.guard.ground_truth(state.turn)
    if not retry:
        # First rejection: the turn stays open, the model gets one correction round.
        _emit({"decision": "block", "reason": _block_reason(audit, truth)})
        return 0

    # Second rejection: never loop. Commit, let it stop, warn the human.
    state.commit()
    still = [v.claim.path for v in audit.rejected] + [r.path for r in audit.unreported_rejected]
    _emit({
        "systemMessage": (
            "netelpro receipts: the final message STILL claims file effects the bytes do not support: "
            + ", ".join(still)
            + ". Check those files yourself."
        )
    })
    return 0


# ---------------------------------------------------------------------------
# install / uninstall: merge into .claude/settings*.json
# ---------------------------------------------------------------------------


def _quote(path: str) -> str:
    return f'"{path}"' if " " in path else path


def hook_commands(python: str, strict: bool) -> dict[str, str]:
    base = f"{_quote(python)} {_MARK}"
    stop = f"{base} stop" + (" --strict" if strict else "")
    return {**{ev: f"{base} begin" for ev in _EVENTS_BEGIN}, _EVENT_STOP: stop}


def _is_ours(hook: Any) -> bool:
    return isinstance(hook, dict) and _MARK in str(hook.get("command", ""))


def _strip_ours(groups: Any) -> list[dict[str, Any]]:
    """Remove netelpro hooks from an event's groups, dropping groups left empty."""
    kept: list[dict[str, Any]] = []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict):
            continue
        hooks = [h for h in group.get("hooks", []) if not _is_ours(h)]
        if hooks:
            kept.append({**group, "hooks": hooks})
    return kept


def merge_settings(settings: dict[str, Any], commands: dict[str, str] | None) -> dict[str, Any]:
    """Return `settings` with netelpro hooks replaced by `commands` (or removed
    when `commands` is None). Every other hook is left byte-for-byte alone."""
    hooks = dict(settings.get("hooks") or {})
    for event in (*_EVENTS_BEGIN, _EVENT_STOP):
        groups = _strip_ours(hooks.get(event))
        if commands is not None:
            label = "netelpro receipts: baseline" if event != _EVENT_STOP else "netelpro receipts: auditing file claims"
            groups.append({
                "hooks": [{"type": "command", "command": commands[event], "timeout": _TIMEOUT_S, "statusMessage": label}]
            })
        if groups:
            hooks[event] = groups
        else:
            hooks.pop(event, None)
    out = dict(settings)
    if hooks:
        out["hooks"] = hooks
    else:
        out.pop("hooks", None)
    return out


def _settings_path(root: Path, local: bool) -> Path:
    return root / ".claude" / ("settings.local.json" if local else "settings.json")


def _load_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise LedgerError(f"{path} is not valid JSON ({e}); not touching it") from e
    if not isinstance(data, dict):
        raise LedgerError(f"{path} is not a JSON object; not touching it")
    return data


def _ensure_gitignore(root: Path) -> bool:
    gi = root / ".gitignore"
    entry = STATE_DIR + "/"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    if any(line.strip() in (entry, STATE_DIR) for line in existing.splitlines()):
        return False
    sep = "" if not existing or existing.endswith("\n") else "\n"
    gi.write_text(existing + sep + f"# netelpro receipts state (baseline + ledger)\n{entry}\n", encoding="utf-8")
    return True


def _cmd_install(args: argparse.Namespace) -> int:
    root = Path(args.root or os.getcwd()).resolve()
    path = _settings_path(root, args.local)
    settings = _load_settings(path)
    commands = hook_commands(args.python or sys.executable, args.strict)
    merged = merge_settings(settings, commands)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    added = _ensure_gitignore(root)
    print(f"installed netelpro receipts hooks into {path}")
    for event, cmd in commands.items():
        print(f"  {event:<17} {cmd}")
    print(f"  .gitignore: {STATE_DIR}/ {'added' if added else 'already present'}")
    print("Hooks load at session start: restart Claude Code (or /hooks) in this repo to activate.")
    return 0


def _cmd_uninstall(args: argparse.Namespace) -> int:
    root = Path(args.root or os.getcwd()).resolve()
    path = _settings_path(root, args.local)
    if not path.exists():
        print(f"nothing to remove: {path} does not exist")
        return 0
    merged = merge_settings(_load_settings(path), None)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"removed netelpro receipts hooks from {path} (state dir {STATE_DIR}/ left in place)")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {MODULE}", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_begin = sub.add_parser("begin", help="hook body for SessionStart / UserPromptSubmit (reads hook JSON on stdin)")
    p_begin.add_argument("--root", default=None, help="override the workspace root (default: hook cwd)")
    p_begin.set_defaults(fn=_cmd_begin, hook=True)

    p_stop = sub.add_parser("stop", help="hook body for Stop (reads hook JSON on stdin)")
    p_stop.add_argument("--root", default=None)
    p_stop.add_argument("--strict", action="store_true", help="unreported (silent) writes block too")
    p_stop.set_defaults(fn=_cmd_stop, hook=True)

    for name, fn in (("install", _cmd_install), ("uninstall", _cmd_uninstall)):
        p = sub.add_parser(name, help=f"{name} the hooks in <root>/.claude/settings.json")
        p.add_argument("--root", default=None, help="repository root (default: cwd)")
        p.add_argument("--local", action="store_true", help="use .claude/settings.local.json instead")
        if name == "install":
            p.add_argument("--strict", action="store_true", help="install the Stop hook with --strict")
            p.add_argument("--python", default=None, help="interpreter to write into the commands (default: this one)")
        p.set_defaults(fn=fn, hook=False)

    args = parser.parse_args(argv)
    try:
        if args.hook:
            return int(args.fn(_read_input(), args))
        return int(args.fn(args))
    except LedgerError as e:
        print(f"netelpro receipts hook: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # never block a session over our own failure
        print(f"netelpro receipts hook: internal error ({type(e).__name__}): {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
