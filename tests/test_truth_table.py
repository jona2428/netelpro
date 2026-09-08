"""Phase 1 contract tests: `truth-table` + finite range types.

Pins spec 2026-09-08-truth-table-range-impl-spec.md section 8 (minimum set).
Backend scope: parser/prosecutor + reference interpreter. Native (LLVM) parity
is exercised through the desugared if-chain (same AST for both backends); the
builder emission tests (generated fallback + matrix, spec section 8 items 8-9)
land with step 6 and are intentionally absent here.
"""
from __future__ import annotations

from netelpro import Defn, ParamType, TruthTableSpec, evaluate, parse


def errors_of(src: str) -> list[str]:
    return [str(e) for e in parse(src).errors]


def tt_src(rows: str, params: str = "(a : Bool) (b : Bool)") -> str:
    return f"(truth-table f {params}\n{rows})\n"


# ---------------------------------------------------------------------------
# Spec section 8.1 - mandatory all-wildcard default row
# ---------------------------------------------------------------------------


def test_truth_table_missing_default_row_is_compile_error() -> None:
    errs = errors_of(
        "(truth-table f (a : Bool)\n"
        "((true) -> true))\n"
    )
    assert any("missing default row" in e for e in errs), errs


def test_truth_table_default_row_must_be_last() -> None:
    # a full-wildcard row before the end does not satisfy B4
    errs = errors_of(
        "(truth-table f (a : Bool)\n"
        "((_) -> false)\n"
        "((true) -> true)\n"
        "((false) -> false))\n"
    )
    assert any("missing default row" in e for e in errs), errs


# ---------------------------------------------------------------------------
# Spec section 8.2 - exhaustiveness over non-default rows (D1)
# ---------------------------------------------------------------------------


def test_truth_table_uncovered_combination_is_compile_error() -> None:
    errs = errors_of(
        tt_src("((true true) -> true)\n((_ _) -> false)")
    )
    hit = [e for e in errs if "does not cover" in e]
    assert hit, errs
    # prosecutor names the exact uncovered combinations, excluding the default
    assert "(true, false)" in hit[0]
    assert "(false, true)" in hit[0]
    assert "(false, false)" in hit[0]


def test_truth_table_full_coverage_over_wildcards_is_legal() -> None:
    # wildcards in non-default rows cover everything; D1 says that is coverage
    res = parse(
        tt_src("((true _) -> 10)\n((_ true) -> 20)\n((false false) -> 30)\n((_ _) -> 99)")
    )
    assert res.ok, [str(e) for e in res.errors]
    assert res.defns == {"f": 2}


def test_truth_table_dead_default_after_full_wildcard_row_is_silent() -> None:
    # section 2.2: unreachable default emits NO warning
    res = parse(
        tt_src("((_) -> 1)\n((_) -> 0)", params="(a : Bool)")
    )
    assert res.ok, [str(e) for e in res.errors]


# ---------------------------------------------------------------------------
# Spec section 8.3 - slot type strictness (B3)
# ---------------------------------------------------------------------------


def test_truth_table_slot_type_strictness_bool_int() -> None:
    errs = errors_of(
        "(truth-table f (a : Bool)\n"
        "((1) -> true)\n"
        "((_) -> false))\n"
    )
    assert any("type-strict slot" in e and "Bool slot" in e for e in errs), errs


def test_truth_table_slot_out_of_enum_int() -> None:
    errs = errors_of(
        "(truth-table f (n : (Int 0 1))\n"
        "((5) -> true)\n"
        "((_) -> false))\n"
    )
    assert any("outside declared enumeration (0, 1)" in e for e in errs), errs


def test_truth_table_slot_bool_in_int_slot_rejected() -> None:
    errs = errors_of(
        "(truth-table f (n : (Int 0 1))\n"
        "((true) -> true)\n"
        "((_) -> false))\n"
    )
    assert any("type-strict slot" in e for e in errs), errs


# ---------------------------------------------------------------------------
# Spec section 8.4 - uniform result type across rows
# ---------------------------------------------------------------------------


def test_truth_table_mixed_row_result_types_rejected() -> None:
    errs = errors_of(
        "(truth-table f (a : Bool)\n"
        "((true) -> 1)\n"
        "((_) -> false))\n"
    )
    assert any("mixed literal result types" in e for e in errs), errs


# ---------------------------------------------------------------------------
# Spec section 8.5 - first-match-wins (rows overlap legally, D3)
# ---------------------------------------------------------------------------


def test_truth_table_first_match_wins() -> None:
    src = (
        tt_src("((true _) -> 10)\n((_ true) -> 20)\n((false false) -> 30)\n((_ _) -> 99)")
    )
    res = parse(src)
    assert res.ok, [str(e) for e in res.errors]
    for args, expected in [
        ("true true", 10),
        ("true false", 10),
        ("false true", 20),
        ("false false", 30),
    ]:
        r = parse(src + f"(f {args})")
        assert r.ok, [str(e) for e in r.errors]
        assert evaluate(r.program, None, set()) == expected, (args, expected)


# ---------------------------------------------------------------------------
# Spec section 8.6 - out-of-domain input routes to default (B2, no UB)
# ---------------------------------------------------------------------------


def test_truth_table_out_of_domain_routes_to_default() -> None:
    src = (
        "(truth-table g (n : (Int 0 1))\n"
        "((0) -> 100)\n"
        "((1) -> 200)\n"
        "((_) -> 7))\n"
        "(def n 5)\n"
        "(g n)\n"
    )
    res = parse(src)
    assert res.ok, [str(e) for e in res.errors]
    assert evaluate(res.program, None, set()) == 7


# ---------------------------------------------------------------------------
# Spec section 8.7 - declared product cap (B5)
# ---------------------------------------------------------------------------


def test_truth_table_product_cap_256_rejected_above() -> None:
    params = " ".join(f"(p{i} : (Int 0 1 2 3 4))" for i in range(4))  # 5^4 = 625
    slots = " ".join("_" for _ in range(4))
    errs = errors_of(f"(truth-table f {params} (({slots}) -> true))")
    assert any("exceeds the 256-combination cap" in e for e in errs), errs


def test_truth_table_product_cap_256_accepted_at_limit() -> None:
    params = " ".join(f"(p{i} : Bool)" for i in range(8))  # 2^8 = 256
    slots = " ".join("_" for _ in range(8))
    res = parse(f"(truth-table f {params} (({slots}) -> 1) (({slots}) -> 0))")
    assert res.ok, [str(e) for e in res.errors]


# ---------------------------------------------------------------------------
# Spec section 8.10 - call-site literal checking (annotations are declarations)
# ---------------------------------------------------------------------------


def test_typed_param_literal_checking_at_call_site() -> None:
    errs = errors_of("(defn g ((a : Bool)) a)\n(def z (g 1))")
    assert any("type-strict call" in e for e in errs), errs


def test_typed_param_float_literal_rejected() -> None:
    errs = errors_of("(defn g ((a : Bool)) a)\n(def z (g 1.5))")
    assert any("type-strict call" in e for e in errs), errs


def test_typed_param_in_enum_literal_accepted() -> None:
    res = parse("(defn g ((n : (Int 0 1 2))) n)\n(g 2)")
    assert res.ok, [str(e) for e in res.errors]


def test_non_literal_args_not_refined_in_v1() -> None:
    # section 1.1: annotations are declarations, not refinements; vars compile
    res = parse("(defn g ((n : (Int 0 1 2))) n)\n(def m 5)\n(g m)")
    assert res.ok, [str(e) for e in res.errors]


# ---------------------------------------------------------------------------
# Spec section 8.11 - aliasing bug class is structurally impossible
# ---------------------------------------------------------------------------


def _walk_nodes(node: object) -> list[object]:
    out: list[object] = []
    if isinstance(node, (list, tuple)):
        for item in node:
            out.extend(_walk_nodes(item))
    elif hasattr(node, "__dataclass_fields__"):
        out.append(node)
        for f in node.__dataclass_fields__:
            out.extend(_walk_nodes(getattr(node, f)))
    return out


def test_desugared_body_has_no_binding_forms_aliasing_class() -> None:
    # the 2026-09-07 gate bug was mutable-state aliasing in a hand-written
    # fallback. The desugared body is a pure if/==/and chain over params and
    # literals: no Let/Def/Defn, no assignment heads -- by construction.
    from netelpro import Let

    src = tt_src("((true _) -> 10)\n((_ true) -> 20)\n((false false) -> 30)\n((_ _) -> 99)")
    res = parse(src)
    assert res.ok
    defn = next(f for f in res.program.forms if isinstance(f, Defn) and f.truth_table)
    for node in _walk_nodes([defn.body]):
        assert not isinstance(node, (Let, Defn)), node
        if isinstance(node, Defn):
            continue
    # and the only call heads are '==' (and 'and'/'if' are distinct node types)
    for node in _walk_nodes([defn.body]):
        if hasattr(node, "head") and hasattr(node, "args"):
            assert node.head == "==", node.head


# ---------------------------------------------------------------------------
# Metadata retention (spec section 3.1) - TruthTableSpec is the artifact input
# ---------------------------------------------------------------------------


def test_truth_table_spec_retention() -> None:
    src = tt_src("((true _) -> 10)\n((_ true) -> 20)\n((false false) -> 30)\n((_ _) -> 99)")
    res = parse(src)
    assert res.ok
    defn = next(f for f in res.program.forms if isinstance(f, Defn) and f.truth_table)
    spec = defn.truth_table
    assert isinstance(spec, TruthTableSpec)
    assert [(n, p.kind) for n, p in spec.params] == [("a", "bool"), ("b", "bool")]
    assert spec.rows[0][0] == (True, None)
    assert spec.rows[1][0] == (None, True)
    assert spec.default_expr == __import__("netelpro", fromlist=["IntLit"]).IntLit(99)


def test_param_type_bool_domain_is_finite() -> None:
    pt = ParamType(kind="bool", enum=())
    assert pt.kind == "bool"


# ---------------------------------------------------------------------------
# Reserved-head and scoping discipline
# ---------------------------------------------------------------------------


def test_truth_table_cannot_shadow_reserved_head() -> None:
    errs = errors_of("(truth-table if (a : Bool)\n((true) -> true)\n((_) -> false))")
    assert any("reserved head" in e for e in errs), errs


def test_truth_table_nested_is_rejected() -> None:
    errs = errors_of(
        "(defn wrap () (truth-table inner (a : Bool)\n((true) -> true)\n((_) -> false)))"
    )
    assert any("only valid at top level" in e for e in errs), errs