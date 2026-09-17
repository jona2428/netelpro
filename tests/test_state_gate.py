"""Tests for netelpro.state_gate.RetryLimiter -- the v0.1 state-tracking
gate pilot (docs/STATE_TRACKING_GATE_SPEC.md section 5).

The central claim under test: enforcement must not depend on the caller
remembering anything. test_survives_total_context_amnesia is the concrete,
falsifiable proof of that -- everything else here is ordinary correctness
coverage of the rate_limit.sl contract's boundary semantics.
"""

from __future__ import annotations

from netelpro.state_gate import RetryLimiter


class FakeClock:
    """Deterministic, manually-advanced clock so cooldown-boundary tests
    don't depend on real wall-clock timing."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def make_limiter() -> tuple[RetryLimiter, FakeClock]:
    clock = FakeClock()
    return RetryLimiter(clock=clock), clock


def test_allows_up_to_max_retries_then_denies():
    limiter, _ = make_limiter()
    for i in range(3):
        allowed, reason = limiter.attempt("job-1", max_retries=3, cooldown_ms=60_000)
        assert allowed, f"attempt {i} should be allowed: {reason}"

    allowed, reason = limiter.attempt("job-1", max_retries=3, cooldown_ms=60_000)
    assert not allowed
    assert reason is not None


def test_boundary_at_ceiling_before_cooldown_is_a_hard_deny():
    """Resolved EPISTEMIC_GATE_SPEC hole 1: retries == max, cooldown not
    elapsed -> deny, not a soft warning."""
    limiter, clock = make_limiter()
    for _ in range(3):
        limiter.attempt("job-2", max_retries=3, cooldown_ms=60_000)

    clock.advance(59.0)  # 59s < 60s cooldown
    allowed, _ = limiter.attempt("job-2", max_retries=3, cooldown_ms=60_000)
    assert not allowed


def test_cooldown_elapsed_resets_the_window():
    """Resolved EPISTEMIC_GATE_SPEC hole 2: cooldown expiry resets the
    counter rather than carrying accumulated history."""
    limiter, clock = make_limiter()
    for _ in range(3):
        limiter.attempt("job-3", max_retries=3, cooldown_ms=60_000)

    clock.advance(61.0)  # cooldown fully elapsed
    allowed, reason = limiter.attempt("job-3", max_retries=3, cooldown_ms=60_000)
    assert allowed, reason

    retries, _ = limiter.get_state("job-3")
    assert retries == 1, "cooldown reset should start a fresh window, not accumulate"


def test_get_state_is_the_read_path_no_caller_input_involved():
    limiter, clock = make_limiter()
    limiter.attempt("job-4", max_retries=5, cooldown_ms=60_000)
    limiter.attempt("job-4", max_retries=5, cooldown_ms=60_000)
    clock.advance(10.0)

    retries, elapsed_ms = limiter.get_state("job-4")
    assert retries == 2
    assert elapsed_ms == 10_000.0


def test_resources_are_isolated():
    limiter, _ = make_limiter()
    for _ in range(3):
        limiter.attempt("resource-a", max_retries=3, cooldown_ms=60_000)

    # resource-a is now exhausted; resource-b must be unaffected.
    allowed, _ = limiter.attempt("resource-b", max_retries=3, cooldown_ms=60_000)
    assert allowed


def test_reset_is_a_harness_operation_not_exposed_to_attempt():
    limiter, _ = make_limiter()
    for _ in range(3):
        limiter.attempt("job-5", max_retries=3, cooldown_ms=60_000)
    limiter.reset("job-5")

    retries, elapsed_ms = limiter.get_state("job-5")
    assert (retries, elapsed_ms) == (0, 0.0)
    allowed, _ = limiter.attempt("job-5", max_retries=3, cooldown_ms=60_000)
    assert allowed


# ---------------------------------------------------------------------------
# The actual point of the pilot: enforcement survives total context amnesia.
# ---------------------------------------------------------------------------


def dumb_agent_with_no_memory(attempt_number_it_thinks_it_is: int) -> str:
    """Simulates an LLM that has completely lost track of prior turns --
    every call is a fresh prompt string with no reference to history. It
    always THINKS this is attempt 1, because nothing tells it otherwise."""
    return f"Attempting the action now (this feels like try #{attempt_number_it_thinks_it_is})."


def test_survives_total_context_amnesia():
    """The falsifiable claim from STATE_TRACKING_GATE_SPEC.md section 5:
    truncate/replace the agent's context entirely between attempts and the
    retry ceiling is still enforced correctly, because enforcement never
    read anything from the agent's own recollection -- only from the store.

    A context-based approach (an LLM asked "how many times have you tried
    this?") would get this wrong the moment the count falls out of its
    context window. This harness never asks that question.
    """
    limiter, _ = make_limiter()
    resource = "flaky-api-call"
    max_retries = 3
    outcomes = []

    for i in range(5):
        # Every iteration is a fully independent "turn": a brand new prompt
        # string is generated, nothing is carried in it, and the agent
        # function above genuinely has no state of its own -- it always
        # believes it's on attempt 1. Only the limiter (outside any
        # context) knows the real count.
        prompt = dumb_agent_with_no_memory(attempt_number_it_thinks_it_is=1)
        assert "try #1" in prompt  # the agent's own belief never changes

        allowed, reason = limiter.attempt(resource, max_retries=max_retries, cooldown_ms=60_000)
        outcomes.append(allowed)

    assert outcomes == [True, True, True, False, False], (
        "first 3 attempts allowed, then denied -- correct regardless of the "
        "agent's own (wrong, unchanging) belief about which attempt this is"
    )

    retries, _ = limiter.get_state(resource)
    assert retries == max_retries, "store held the real count the agent never had"
