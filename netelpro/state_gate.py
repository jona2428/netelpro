"""State-Tracking Gate: v0.1 pilot (rate limiting / backoff).

See docs/STATE_TRACKING_GATE_SPEC.md section 5. State that must not be
misremembered lives in a store the caller (LLM/agent) never writes to
directly and is never asked to recall from its own context -- only a
passing compiled Netelpro contract (examples/gates/rate_limit.sl) can
advance it. A caller that has "forgotten" how many times it already tried
(stale context, a fresh process, a summarized conversation) gets the same
correct answer as one with perfect recollection, because the answer never
came from the caller in the first place.

Resolves two of EPISTEMIC_GATE_SPEC.md's open holes for this pilot rule:
- Boundary semantics (hole 1): at retries == max-retries with the cooldown
  not yet elapsed, this DENIES -- a hard stop, not a warning.
- Cooldown expiry (hole 2): when the cooldown fully elapses, the counter
  RESETS (this attempt becomes retries=1 of a new window) rather than
  carrying accumulated history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from netelpro.gate import Gate

_DEFAULT_RULE = Path(__file__).resolve().parent.parent / "examples" / "gates" / "rate_limit.sl"


@dataclass
class _ResourceState:
    retries: int = 0
    first_attempt_ms: Optional[float] = None


class RetryLimiter:
    """Rate-limiting state store gated by a compiled Netelpro contract.

    `resource_id` identifies whatever is being retried (a tool call, an API
    endpoint, an agent action) -- the limiter tracks each independently.
    """

    def __init__(
        self,
        rule_path: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._gate = Gate(rule_path or _DEFAULT_RULE)
        self._clock = clock
        self._state: Dict[str, _ResourceState] = {}

    def get_state(self, resource_id: str) -> tuple[int, float]:
        """Read path: current (retries, elapsed_ms) for a resource, straight
        from the store -- never derived from anything the caller claims."""
        st = self._state.get(resource_id)
        if st is None or st.first_attempt_ms is None:
            return 0, 0.0
        elapsed_ms = (self._clock() - st.first_attempt_ms) * 1000.0
        return st.retries, elapsed_ms

    def attempt(
        self,
        resource_id: str,
        max_retries: int,
        cooldown_ms: int,
    ) -> tuple[bool, Optional[str]]:
        """Write path: propose one retry attempt for `resource_id`.

        Fail-closed: any gate failure (compile error, arity misuse, non-bool
        return) denies with Gate.check's own explicit reason. On approval,
        the store is incremented here -- the only place it ever changes --
        before returning.
        """
        st = self._state.setdefault(resource_id, _ResourceState())
        now = self._clock()
        elapsed_ms = 0.0 if st.first_attempt_ms is None else (now - st.first_attempt_ms) * 1000.0

        allowed, reason = self._gate.check(st.retries, max_retries, int(elapsed_ms), cooldown_ms)
        if not allowed:
            return False, reason or (
                f"rate limit: {st.retries}/{max_retries} retries used, "
                f"{elapsed_ms:.0f}ms/{cooldown_ms}ms cooldown elapsed"
            )

        if st.first_attempt_ms is None or elapsed_ms >= cooldown_ms:
            st.first_attempt_ms = now
            st.retries = 1
        else:
            st.retries += 1
        return True, None

    def reset(self, resource_id: str) -> None:
        """Explicit reset -- a harness-level operation, never something the
        LLM can trigger by claiming something in text."""
        self._state.pop(resource_id, None)
