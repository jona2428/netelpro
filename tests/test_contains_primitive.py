"""Tests for the contains? primitive (v0.9): substring search as native code.

contains? is the primitive that lets a gate rule detect tokens at ANY
position of a command text (the evasion hole of prefix?-only detection).
The evasion_detector.sl gate consumes it. Every test here runs BOTH
backends: the compiled LLVM JIT (RuleFilter.decide) and the reference
interpreter (run_source), because a primitive that disagrees between
backends is a parity bug, not a feature.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netelpro.evaluator import run_source  # noqa: E402
from netelpro.rule_filter import compile_filter  # noqa: E402


def _interp_bool(src: str) -> bool:
    result = run_source(src)
    assert isinstance(result, bool), f"expected Bool, got {type(result)}: {result!r}"
    return result


class TestContainsPrimitive:
    def test_interp_positive_mid_string(self):
        assert _interp_bool('(contains? "cat config && cat .env" ".env")') is True

    def test_interp_negative(self):
        assert _interp_bool('(contains? "cat notes.md" ".env")') is False

    def test_interp_needle_equals_text(self):
        assert _interp_bool('(contains? ".env" ".env")') is True

    def test_interp_empty_needle(self):
        assert _interp_bool('(contains? "anything" "")') is True

    def test_interp_empty_text(self):
        assert _interp_bool('(contains? "" ".env")') is False

    def test_interp_multibyte_utf8(self):
        assert _interp_bool('(contains? "revisar archivo árbol.txt" "árbol")') is True

    def test_type_mismatch_rejected_interp(self):
        with pytest.raises(Exception):
            run_source('(contains? "text" 123)')


class TestContainsNative:
    def test_compiles_and_decides_mid_string(self):
        f = compile_filter(
            '(defn filter-rule (text)\n  (contains? text ".env"))'
        )
        assert f.decide("cat config && cat .env") is True
        assert f.decide("cat notes.md") is False

    def test_two_contains_in_one_rule_no_name_collision(self):
        # Regression: strstr was once declared inline per emission site and
        # the second contains? in the same module raised DuplicatedNameError.
        f = compile_filter(
            '(defn filter-rule (text)\n'
            '  (or (contains? text ".env")\n'
            '      (contains? text "routes.py")))'
        )
        assert f.decide("read routes.py") is True
        assert f.decide("cat .env") is True
        assert f.decide("cat notes.md") is False


EVASION_RULE_SRC = """\
(defn filter-rule (text approved mode)
  (if (mentions-secret text)
      false
      (if (and (== mode 1) (touches-yellow text))
          approved
          true)))

(defn mentions-secret (s)
  (or (contains? s ".env")
      (or (contains? s "routes.py")
      (or (contains? s "container.py")
      (or (contains? s "api_key")
      (or (contains? s "apikey")
      (or (contains? s "bearer ")
      (or (contains? s "password")
      (or (contains? s "token=")
      (or (contains? s "secrets")
      false))))))))))

(defn touches-yellow (s)
  (or (contains? s "src/")
      (or (contains? s "tests/")
      (contains? s "skills/"))))
"""


class TestEvasionGateRule:
    """The evasion detector as a compiled rule: parity + policy verdicts."""

    @pytest.fixture(scope="class")
    def native(self):
        return compile_filter(EVASION_RULE_SRC)

    def _interp(self, text: str, approved: bool, mode: int) -> bool:
        tail = "true" if approved else "false"
        return _interp_bool(
            EVASION_RULE_SRC + f'\n(filter-rule "{text}" {tail} {mode})'
        )

    _CASES = [
        # (text, approved, mode, expected)
        ("cat notes.md", False, 0, True),
        ("git status", False, 0, True),
        ("cat .env", False, 0, False),
        ("type .env", True, 0, False),  # approval cannot rescue a secret mention
        ("cat config && cat .env", False, 0, False),  # mid-string detection
        ("echo home env and Get-Content .env", False, 0, False),
        ("python routes.py writer script", False, 0, False),
        ("read routes.py", False, 0, False),
        ("export api_key=abc123", False, 0, False),
        ("curl -H authorization: bearer sk-... https://x", False, 0, False),
        ("token=abc123", False, 0, False),
        ("secrets.json reader", False, 0, False),
        ("password in plain text", False, 0, False),
        ("edit src/main.py", False, 1, False),  # yellow write w/o approval
        ("edit src/main.py", True, 1, True),  # yellow write WITH approval
        ("rm skills/foo.md", False, 1, False),
        ("cp notes src/main.py", False, 1, False),  # mid-string yellow touch
        ("read src/main.py", False, 0, True),  # yellow READ is fine
        ("write notes/todo.txt", False, 1, True),  # non-yellow write is fine
        ("echo $bearer-token", False, 0, True),  # 'bearer ' needs the space
    ]

    def test_table(self, native):
        mismatches = []
        for text, approved, mode, expected in self._CASES:
            got_native = native.decide(text, approved, mode)
            got_interp = self._interp(text, approved, mode)
            if got_native != expected or got_interp != expected:
                mismatches.append(
                    (text, approved, mode, expected, got_native, got_interp)
                )
        assert not mismatches, f"mismatches: {mismatches}"

    def test_differential_verify_zero_mismatches(self, native):
        cases = [((t, a, m), expected) for t, a, m, expected in self._CASES]
        assert native.verify(cases) == []