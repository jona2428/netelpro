"""Tests for netelpro.lib -- prelude concatenation and contract generation.

Everything here runs against the REAL compiler (RuleFilter: LLVM native JIT
plus the reference interpreter), never a reimplementation. The central test
is the round-trip over the live host contracts: extract the spec from a real
``.sl``, re-render it, and prove the rendered contract decides identically to
the original on every point of its declared domain.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netelpro.ast_nodes import Sorry
from netelpro.lib import (
    ContractSpec,
    LibError,
    Slot,
    concat_with_prelude,
    contract_from_source,
    prelude_source,
    render_contract,
    render_truth_table,
)
from netelpro.parser import parse
from netelpro.rule_filter import RuleFilter, RuleFilterError

# ---------------------------------------------------------------------------
# Locating the live host contracts (outside this repo, in FinalFront/data).
# ---------------------------------------------------------------------------

_CONTRACT_DIR_CANDIDATES = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "contracts",
    Path("C:/Users/Jona/Desktop/FinalFront/data/contracts"),
    Path("/mnt/c/Users/Jona/Desktop/FinalFront/data/contracts"),
)


def _contract_dir() -> Path | None:
    for cand in _CONTRACT_DIR_CANDIDATES:
        if cand.is_dir():
            return cand
    return None


_LIVE_CONTRACTS = sorted(
    (p for p in _contract_dir().glob("*.sl") if p.is_file())
    if _contract_dir() is not None
    else []
)


# ---------------------------------------------------------------------------
# Prelude
# ---------------------------------------------------------------------------


def test_prelude_parses_and_has_no_holes():
    """The prelude must stay closed: a `sorry` here breaks every rule that
    concatenates it, including rules that never call the offending helper."""
    result = parse(prelude_source())
    assert result.ok, result.errors
    holes = [n for form in result.program.forms for n in _walk(form) if isinstance(n, Sorry)]
    assert holes == [], f"prelude must contain no sorry holes, found {len(holes)}"


def _walk(node):
    """Shallow generic walk over AST dataclass fields."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        for value in getattr(current, "__dict__", {}).values():
            if isinstance(value, list):
                stack.extend(v for v in value if hasattr(v, "__dict__"))
            elif hasattr(value, "__dict__"):
                stack.append(value)


def test_prelude_concatenated_with_defn_rule_keeps_parity():
    source = concat_with_prelude(
        """
(defn filter-rule (a b c)
  (and (all-of3 a b c) (not (any-zero3 a b c))))
"""
    )
    rule = RuleFilter(source)
    cases = [
        ((1, 1, 1), True),
        ((1, 1, 0), False),
        ((0, 0, 0), False),
        ((1, 0, 1), False),
    ]
    assert rule.verify(cases) == []


def test_prelude_collision_with_contract_is_a_hard_error():
    """A contract redefining a prelude name must fail loudly, not shadow it."""
    source = concat_with_prelude(
        """
(defn all-of3 (a b c) true)
(defn filter-rule (a b c) (all-of3 a b c))
"""
    )
    with pytest.raises(RuleFilterError) as exc:
        RuleFilter(source)
    assert "duplicate" in str(exc.value).lower()


def test_prelude_standalone_has_no_filter_rule():
    """The prelude is a library, not a rule: on its own there is no entry."""
    with pytest.raises(RuleFilterError) as exc:
        RuleFilter(prelude_source())
    assert "filter-rule" in str(exc.value)


# ---------------------------------------------------------------------------
# render_truth_table / render_contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arity", [2, 3, 4])
def test_rendered_contract_covers_product_and_compiles(arity):
    slots = tuple(Slot(f"flag-{i}", f"1 si la condicion {i} se cumple; 0 si no") for i in range(arity))
    spec = ContractSpec(title=f"Contrato de prueba de {arity} slots.", slots=slots)
    source = render_contract(spec)
    rule = RuleFilter(source)

    cases = []
    for combo in itertools.product((0, 1), repeat=arity):
        expected = 1 if all(combo) else 0
        cases.append((combo, expected))
    assert rule.verify_int(cases) == []


def test_rendered_contract_has_mandatory_backstop():
    spec = ContractSpec(title="t", slots=(Slot("a"), Slot("b")))
    source = render_contract(spec)
    assert source.rstrip().endswith("((_ _) -> 0))")


def test_render_truth_table_rejects_wrong_arity_row():
    with pytest.raises(LibError):
        render_truth_table("filter-rule", (Slot("a"), Slot("b")), (((1,), 1),))


def test_slot_name_validation():
    for bad in ("Flag", "2flag", "flag_name", "flag name", ""):
        with pytest.raises(LibError):
            Slot(bad)


def test_product_cap_is_enforced():
    slots = tuple(Slot(f"f{i}") for i in range(9))
    with pytest.raises(LibError) as exc:
        ContractSpec(title="t", slots=slots)
    assert "256" in str(exc.value)


def test_contract_needs_at_least_two_slots():
    with pytest.raises(LibError):
        ContractSpec(title="t", slots=(Slot("a"),))


def test_duplicate_slot_names_rejected():
    with pytest.raises(LibError):
        ContractSpec(title="t", slots=(Slot("a"), Slot("a")))


# ---------------------------------------------------------------------------
# contract_from_source -- the guard rail
# ---------------------------------------------------------------------------


def test_non_canonical_contract_is_refused():
    """A contract whose rows are NOT the canonical all-of-N shape must be
    refused, so this can never silently 'normalize' different semantics."""
    canonical = """
(truth-table filter-rule
  (declared : (Int 0 1))
  (path-covered : (Int 0 1))
  ((0 _) -> 0)
  ((_ 0) -> 0)
  ((1 1) -> 1)
  ((_ _) -> 0))
"""
    # This one IS canonical; prove the extractor accepts it.
    spec = contract_from_source(canonical)
    assert spec.admit == 1 and spec.reject == 0 and len(spec.slots) == 2
    assert [s.name for s in spec.slots] == ["declared", "path-covered"]

    non_canonical = """
(truth-table filter-rule
  (a : (Int 0 1))
  (b : (Int 0 1))
  ((1 1) -> 1)
  ((0 1) -> 0)
  ((_ _) -> 0))
"""
    with pytest.raises(LibError):
        contract_from_source(non_canonical)


def test_contract_from_source_preserves_non_default_verdicts():
    """A canonical-shaped contract with verdicts other than 0/1 must round-trip
    with its own verdicts, not be forced onto the defaults."""
    source = """
(truth-table filter-rule
  (a : (Int 0 1))
  (b : (Int 0 1))
  ((0 _) -> 7)
  ((_ 0) -> 7)
  ((1 1) -> 3)
  ((_ _) -> 7))
"""
    spec = contract_from_source(source)
    assert (spec.admit, spec.reject) == (3, 7)
    rendered = render_contract(spec)
    rule = RuleFilter(rendered)
    assert rule.decide_int(1, 1) == 3
    assert rule.decide_int(0, 1) == 7
    assert rule.decide_int(2, 2) == 7


def test_contract_from_source_rejects_non_truth_table():
    with pytest.raises(LibError):
        contract_from_source("(defn filter-rule (a) (== a 1))")


# ---------------------------------------------------------------------------
# Round-trip over the live host contracts (the test that matters)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _LIVE_CONTRACTS, reason="live contract directory not reachable")
def test_roundtrip_matches_every_live_contract():
    """For every live contract: extract spec -> re-render -> prove the rendered
    contract decides identically to the original over its whole declared
    domain, plus out-of-domain probes (the backstop path).

    This is the evidence that generation can replace the hand-written copies
    without changing a single verdict.
    """
    checked = 0
    for path in _LIVE_CONTRACTS:
        original_src = path.read_text(encoding="utf-8-sig")
        try:
            spec = contract_from_source(original_src)
        except LibError:
            # Not a canonical all-of-N contract; the extractor refuses by
            # design and this test does not claim anything about it.
            continue

        rendered = render_contract(spec)
        original = RuleFilter(original_src)
        generated = RuleFilter(rendered)

        arity = len(spec.slots)
        domain_cases = list(itertools.product((0, 1), repeat=arity))

        # Declared domain: full differential verification (native vs
        # interpreter vs expected) on BOTH the original and the generated
        # contract, plus native-vs-native agreement.
        for combo in domain_cases:
            native_orig = original.decide_int(*combo)
            native_gen = generated.decide_int(*combo)
            assert native_orig == native_gen, (
                f"{path.name}: verdict diverged at {combo!r}: "
                f"original={native_orig} generated={native_gen}"
            )
        assert original.verify_int([(c, original.decide_int(*c)) for c in domain_cases]) == []
        assert generated.verify_int([(c, generated.decide_int(*c)) for c in domain_cases]) == []

        # Out-of-domain probes reach the backstop row on the NATIVE path only.
        # The reference interpreter rejects them at parse time
        # ("literal N outside declared enumeration"), so differential
        # verification is unavailable there -- this asserts native-vs-native
        # agreement, which is what the FFI boundary actually exercises.
        for probe in (2, -1, 7):
            for i in range(arity):
                combo = tuple(probe if j == i else 1 for j in range(arity))
                assert original.decide_int(*combo) == generated.decide_int(*combo), (
                    f"{path.name}: backstop diverged at out-of-domain {combo!r}"
                )
        checked += 1

    assert checked > 0, "no canonical contracts were found to round-trip"


@pytest.mark.skipif(not _LIVE_CONTRACTS, reason="live contract directory not reachable")
def test_live_contracts_report_how_many_are_canonical():
    """Visibility, not enforcement: report the real split between canonical
    all-of-N contracts and the ones with bespoke policy rows."""
    canonical, bespoke = [], []
    for path in _LIVE_CONTRACTS:
        try:
            contract_from_source(path.read_text(encoding="utf-8-sig"))
            canonical.append(path.name)
        except LibError:
            bespoke.append(path.name)
    assert len(canonical) + len(bespoke) == len(_LIVE_CONTRACTS)
    assert canonical, "expected at least one canonical contract in the live set"
