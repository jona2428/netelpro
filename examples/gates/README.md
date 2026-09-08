# Domain gate examples (compiled `.sl`, fail-closed host)

Three auditable approval rules in Netelpro `.sl`, each with its contract
and truth table documented in-file, verified by `tests/test_gate_examples.py`:

| rule | domain | contract |
|---|---|---|
| `expense_approval.sl` | spend controls | `(filter-rule amount manager emergency) -> 1\|0` |
| `content_moderation.sl` | publish gating | `(filter-rule toxicity reports age_verified) -> 1\|0` |
| `robot_interlock.sl` | safety interlock | `(filter-rule door_open speed estop) -> 1\|0` |

## The pattern for external agent harnesses

1. Write the rule as a pure `filter-rule` over Int/0-1 flags, with the
   contract and truth table as comments (see any file above).
2. Compile it once at startup: `gate = netelpro.gate.Gate("rule.sl")`
   — strict mode, raises `GateError` on unreadable file or compile error.
3. Decide fail-closed at the sensitive action:
   `allow, reason = gate.check(*args)` — never raises;
   `reason is None` exactly when the compiled rule decided, and any
   `(False, reason)` is a declared failure, never a silent one.
4. Audit: `gate.manifest()` lists declared `sorry` holes; every input
   space of a boolean/integer rule is exhaustively testable
   (see the test file for the pattern).

Run the verification yourself:

```bash
python -m pytest tests/test_gate_examples.py -q
```