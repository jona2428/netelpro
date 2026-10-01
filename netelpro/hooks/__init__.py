"""Agent-harness hooks that wire Netelpro's enforcement layers into real
tools without the tool having to know Netelpro exists.

- `netelpro.hooks.claude_code`: file-effect receipts as Claude Code hooks
  (baseline at every prompt, audit of the final message at every stop).
"""
