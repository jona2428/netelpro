"""Phase 5 test suite: LLVM native codegen (codegen.py).

Testing philosophy — DIFFERENTIAL: every compiled program is compared against
the tree-walking interpreter (the same source, same semantics, two engines).
The interpreter is the reference implementation; the native backend must agree
exactly. This is the language's own thesis applied to itself: agreement is
verified mechanically, never assumed.

Rejected-form cases are prosecuted as CodegenError with exact coordinates —
the prosecutor never falls back silently.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from straylight.codegen import CodegenError, compile_and_run, compile_program
from straylight.evaluator import evaluate
from straylight.parser import parse


def _parse_ok(src: str):
    pr = parse(src)
    assert pr.ok, f"parse failed unexpectedly: {[e.message for e in pr.errors]}"
    return pr.program


def run_both(src: str):
    """Differential execution: (interpreter_value, native_result)."""
    program = _parse_ok(src)
    return evaluate(program), compile_and_run(program)


def assert_agree(src: str) -> int:
    interp, native = run_both(src)
    if isinstance(interp, bool):
        assert native == int(interp), f"mismatch for {src!r}: interp={interp} native={native}"
    elif interp is None:
        assert native == 0, f"mismatch for {src!r}: interp=nil native={native}"
    else:
        assert native == interp, f"mismatch for {src!r}: interp={interp} native={native}"
    return native


def assert_rejected(src: str, fragment: str, stage: str = "codegen") -> None:
    """Assert a form dies, at the declared stage, with the expected message fragment.

    stage='parse':   the parser fiscal rejects it BEFORE codegen sees it.
    stage='codegen': it parses fine; the codegen prosecutor rejects it.
    """
    pr = parse(src)
    if stage == "parse":
        assert not pr.ok, f"expected parse error for {src!r}"
        msgs = [e.message for e in pr.errors]
        assert any(fragment in m for m in msgs), f"parse messages {msgs} lack {fragment!r}"
        return
    assert pr.ok, f"parse failed unexpectedly: {[e.message for e in pr.errors]}"
    with pytest.raises(CodegenError) as excinfo:
        compile_and_run(pr.program)
    assert fragment in str(excinfo.value)


class TestDifferentialArithmetic:
    """Native i64 arithmetic must agree with the interpreter exactly."""

    @pytest.mark.parametrize(
        "src",
        [
            "(+ 2 3)",
            "(+ -5 12)",
            "(- 10 4)",
            "(- 4 10)",
            "(- -4 -10)",
            "(* 7 6)",
            "(* -7 6)",
            "(+ (* 2 3) (- 10 4))",
        ],
    )
    def test_arithmetic_matches_interpreter(self, src: str) -> None:
        assert_agree(src)

    @pytest.mark.parametrize(
        "src",
        [
            "(quot 7 2)",
            "(quot -7 2)",
            "(quot 7 -2)",
            "(quot -7 -2)",
            "(rem 7 2)",
            "(rem -7 2)",
            "(rem 7 -2)",
            "(rem -7 -2)",
        ],
    )
    def test_quot_rem_truncation_matches_interpreter(self, src: str) -> None:
        assert_agree(src)


class TestDifferentialLogic:
    """Comparisons, booleans, strict-typing control flow agree with the interpreter."""

    @pytest.mark.parametrize(
        "src",
        [
            "(< 1 2)",
            "(< 2 1)",
            "(<= 2 2)",
            "(> 3 2)",
            "(>= 2 3)",
            "(== 5 5)",
            "(== 5 6)",
            "(!= 5 6)",
            "(!= 5 5)",
        ],
    )
    def test_comparisons_match_interpreter(self, src: str) -> None:
        assert_agree(src)

    def test_if_takes_then(self) -> None:
        assert_agree("(if (< 1 2) 10 20)")

    def test_if_takes_else(self) -> None:
        assert_agree("(if (< 2 1) 10 20)")

    def test_and_short_circuit_true_path(self) -> None:
        assert_agree("(and (< 1 2) (< 3 4))")

    def test_and_short_circuit_false_path(self) -> None:
        assert_agree("(and (< 2 1) (< 3 4))")

    def test_or_short_circuit(self) -> None:
        assert_agree("(or (< 2 1) (< 3 4))")

    def test_not(self) -> None:
        assert_agree("(not (< 2 1))")

    def test_nested_if_in_defn(self) -> None:
        assert_agree("(defn classify (n) (if (< n 0) -1 (if (== n 0) 0 1)))\n(classify -5)\n(classify 0)")

    def test_let_chain(self) -> None:
        assert_agree("(let a 3 (let b 4 (+ (* a a) (* b b))))")

    def test_def_literals(self) -> None:
        assert_agree("(def limit 100)\n(def flag true)\n(+ limit (if flag 1 0))")

    def test_shadowing_param_by_let(self) -> None:
        assert_agree("(defn f (x) (let x (+ x 1) (* x 2)))\n(f 5)")

    def test_bool_return_zext(self) -> None:
        """Top-level Bool: interpreter yields True, native yields 1 (i1 zext i64)."""
        assert_agree("(== (+ 1 1) 2)")


class TestDifferentialFunctions:
    """defn calls, forward refs, native recursion and TCO agree with the interpreter."""

    def test_defn_call(self) -> None:
        assert_agree("(defn add (x y) (+ x y))\n(add 2 3)")

    def test_forward_reference(self) -> None:
        assert_agree("(defn a () (b))\n(defn b () 7)\n(a)")

    def test_mutual_recursion_native_calls(self) -> None:
        assert_agree(
            "(defn even? (n) (if (== n 0) true (odd? (- n 1))))\n"
            "(defn odd? (n) (if (== n 0) false (even? (- n 1))))\n"
            "(even? 10)"
        )

    def test_non_tail_recursion_fib(self) -> None:
        assert_agree("(defn fib (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))\n(fib 15)")

    def test_tco_sum_to_100000(self) -> None:
        """TCO is the point: 100k-deep tail recursion in CONSTANT stack space."""
        native = assert_agree("(defn sum-to (n acc) (if (== n 0) acc (sum-to (- n 1) (+ acc n))))\n(sum-to 100000 0)")
        assert native == 5000050000

    def test_tco_count_down_500000(self) -> None:
        """Deeper still: 500k iterations must not overflow the host stack."""
        assert_agree("(defn count (n) (if (== n 0) 0 (count (- n 1))))\n(count 500000)")

    def test_tco_through_if_both_branches(self) -> None:
        """Tail calls from BOTH if branches (then and else) hit the loop back-edge."""
        assert_agree(
            "(defn parity-acc (n acc) (if (== n 0) acc (if (< n 3) (parity-acc (- n 1) (+ acc 2)) (parity-acc (- n 2) acc))))\n"
            "(parity-acc 10000 0)"
        )


class TestProsecutorRejections:
    """Unrepresented forms die with exact coordinates. No silent fallback.

    Two prosecution stages are documented as they REALLY are:
    - The Phase 1 parser fiscal already kills at PARSE: first-class calls, arity
      mismatches (primitives AND defns), reserved-head redefinitions.
    - The Phase 5 codegen kills what parses fine: unrepresented literals/forms,
      list/str primitives (parse-valid arities), cap enforcement, type conflicts,
      undefined symbols, def-of-non-literal.
    """

    def test_float_literal_rejected_at_codegen(self) -> None:
        assert_rejected("(defn f () 3.14)", "Float literals are not supported", stage="codegen")

    def test_string_literal_rejected_at_codegen(self) -> None:
        assert_rejected('(defn f () "hola")', "String literals are not supported", stage="codegen")

    def test_nil_literal_rejected_at_codegen(self) -> None:
        assert_rejected("(defn f () nil)", "Nil literals are not supported", stage="codegen")

    def test_list_literal_rejected_at_codegen(self) -> None:
        assert_rejected("(defn f () (list 1 2))", "List literals are not supported", stage="codegen")

    def test_first_class_call_rejected_at_parse(self) -> None:
        """Parser law (verified in Phase 4): no first-class calls in v0.1."""
        assert_rejected(
            "(defn f () ((fn (x) x) 1))",
            "form head must be a symbol, not a nested form",
            stage="parse",
        )

    def test_sorry_rejected_at_codegen(self) -> None:
        assert_rejected('(defn f () (sorry "not now"))', "Cannot compile 'sorry' hole", stage="codegen")

    @pytest.mark.parametrize(
        "head",
        ["cons", "nth", "str-cat"],
    )
    def test_non_compilable_primitives_rejected_at_codegen(self, head: str) -> None:
        assert_rejected(
            f"(defn f (x) ({head} x x))",
            f"primitive '{head}' is not supported",
            stage="codegen",
        )

    def test_division_rejected_at_codegen(self) -> None:
        assert_rejected("(defn f (x) (/ x x))", "'/' is not supported in native codegen v0.1", stage="codegen")

    def test_list_call_desugars_to_listlit_and_rejected(self) -> None:
        """(list ...) is parsed as ListLit directly (the only open-arity form) —
        so the codegen rejection surfaces as the ListLit message."""
        assert_rejected("(defn f (x) (list x x))", "List literals are not supported", stage="codegen")

    @pytest.mark.parametrize(
        "head",
        ["head", "tail", "is-nil", "len", "str-len", "int->str", "str->int", "int->float"],
    )
    def test_unary_list_str_primitives_rejected_at_parse(self, head: str) -> None:
        """These die at PARSE only because my probe used wrong arity (2 instead of 1).

        With their REAL arity-1 shape they parse fine and are rejected at CODEGEN.
        Both stages are honest; this documents the parse-stage for the arity itself.
        """
        assert_rejected(f"(defn f (x) ({head} x x))", "expects 1 operand(s)", stage="parse")

    def test_unary_list_prims_real_arity_rejected_at_codegen(self) -> None:
        """The same primitives with their TRUE arity-1: parse OK, codegen rejects."""
        for head in ("head", "tail", "is-nil", "len"):
            assert_rejected(f"(defn f (xs) ({head} xs))", f"primitive '{head}' is not supported", stage="codegen")

    def test_print_without_grant_rejected_at_codegen(self) -> None:
        assert_rejected("(defn f () (print 1))", "capability 'io' required by 'print' but not granted", stage="codegen")

    def test_def_non_literal_rejected_at_codegen(self) -> None:
        assert_rejected("(def x (+ 1 2))", "'def' value must be an Int or Bool literal", stage="codegen")

    def test_undefined_symbol_rejected_at_codegen(self) -> None:
        assert_rejected("(defn f (x) y)", "undefined symbol 'y'", stage="codegen")

    def test_undefined_symbol_exact_coords(self) -> None:
        program = _parse_ok("(defn f (x) y)")
        with pytest.raises(CodegenError) as excinfo:
            compile_and_run(program)
        assert "line 1, col 13" in str(excinfo.value)

    def test_if_condition_not_bool_rejected_at_call_site(self) -> None:
        """Type inference: (if n 1 2) makes n Bool — the defn compiles alone;
        the conflict fires when an Int call-site unifies against the Bool param."""
        assert_rejected("(defn f (n) (if n 1 2))\n(f 5)", "expected Bool", stage="codegen")

    def test_arith_on_bool_rejected_at_call_site(self) -> None:
        """Type inference: (+ b 1) makes b Int — conflict fires at an incompatible call-site."""
        assert_rejected("(defn f (b) (+ b 1))\n(f true)", "type mismatch", stage="codegen")

    def test_compare_bool_rejected_at_codegen(self) -> None:
        assert_rejected("(defn f (b) (== b true))", "expected Int", stage="codegen")

    def test_if_branch_type_mismatch_rejected_at_codegen(self) -> None:
        assert_rejected("(defn f (c) (if c 1 true))", "type mismatch for 'if' branches", stage="codegen")

    def test_call_arity_mismatch_rejected_at_parse(self) -> None:
        """Parser law: defn arity is enforced at PARSE (registry-based)."""
        assert_rejected(
            "(defn f (x y) (+ x y))\n(f 1)",
            "'f' expects 2 operand(s) (declared by defn), found 1",
            stage="parse",
        )

    def test_error_in_uncalled_defn_still_prosecuted_at_codegen(self) -> None:
        """The static walk rejects errors in bodies NEVER called — no lazy escape."""
        assert_rejected("(defn hidden () 3.14)\n(defn main () 42)", "Float literals are not supported", stage="codegen")

    def test_duplicate_def_rejected_at_codegen(self) -> None:
        """(def x 1)(def x 2): the parser checks defn duplicates, codegen adds def dups."""
        assert_rejected("(def x 1)\n(def x 2)", "duplicate definition of 'x'", stage="codegen")

    def test_redefining_reserved_head_rejected_at_parse(self) -> None:
        """Parser law: reserved heads cannot be redefined with defn."""
        assert_rejected("(defn print (x) x)", "'print' is a reserved head", stage="parse")


class TestNativePrint:
    """print compiles to printf; verified end-to-end via subprocess (real fd 1)."""

    CHILD = (
        "import sys; sys.path.insert(0, r'{root}'); "
        "from straylight.parser import parse; from straylight.codegen import compile_and_run; "
        "pr = parse(sys.argv[1]); r = compile_and_run(pr.program); sys.exit(0 if r == 0 else 1)"
    )

    def _run_child(self, src: str) -> subprocess.CompletedProcess[str]:
        root = str(ROOT)
        return subprocess.run(
            [sys.executable, "-c", self.CHILD.format(root=root), src],
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_top_level_print_emits_line(self) -> None:
        child = self._run_child("(grant io)\n(print 42)")
        assert "42\n" in child.stdout, f"stdout={child.stdout!r} stderr={child.stderr!r}"

    def test_print_via_explicit_main_call(self) -> None:
        child = self._run_child("(grant io)\n(defn main () (print 7))\n(main)")
        assert "7\n" in child.stdout, f"stdout={child.stdout!r} stderr={child.stderr!r}"

    def test_print_in_loop_tco(self) -> None:
        child = self._run_child("(grant io)\n(defn loop (n) (if (== n 0) 0 (let d (print n) (loop (- n 1)))))\n(loop 3)")
        assert "1\n" in child.stdout and "2\n" in child.stdout and "3\n" in child.stdout


class TestJITArtifact:
    """The CompiledProgram artifact contract."""

    def test_run_is_repeatable(self) -> None:
        program = _parse_ok("(defn sq (x) (* x x))\n(sq 12)")
        cp = compile_program(program)
        assert cp.run() == 144
        assert cp.run() == 144  # JIT artifact is reusable — same address, same math

    def test_module_ir_is_emitted(self) -> None:
        program = _parse_ok("(+ 1 2)")
        cp = compile_program(program)
        ir_text = str(cp.module)
        assert 'define i64 @"main"()' in ir_text or "define i64 @main()" in ir_text
        assert "__fmt_int" in ir_text or "@main" in ir_text
class TestCLINativeFlag:
    """The --native CLI flag: same static passes, native execution engine."""

    CLI = (
        "import sys; sys.path.insert(0, r'{root}'); "
        "from straylight.__main__ import main; "
        "sys.exit(main(['--native', r'{sl}']) if r'{sl}' else 1)"
    )

    def _write_sl(self, tmp_path, name: str, src: str):
        sl = tmp_path / name
        sl.write_text(src, encoding="utf-8")
        return str(sl)

    def test_native_flag_runs_and_prints(self, tmp_path) -> None:
        """Native printf writes to the REAL fd 1 — verify via subprocess, not capsys."""
        sl = self._write_sl(tmp_path, "ok_native.sl", "(grant io)\n(print 42)")
        child = subprocess.run(
            [sys.executable, "-m", "straylight", "--native", sl],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(ROOT),
        )
        assert child.returncode == 0, f"stderr={child.stderr!r}"
        assert "42\n" in child.stdout, f"stdout={child.stdout!r}"

    def test_native_flag_returns_value(self, tmp_path, capsys) -> None:
        from straylight.__main__ import main

        sl = self._write_sl(tmp_path, "val.sl", "(defn fib (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))\n(fib 15)")
        rc = main(["--native", sl])
        captured = capsys.readouterr()
        assert rc == 0
        assert "=> 610" in captured.out

    def test_native_flag_respects_caps_before_codegen(self, tmp_path, capsys) -> None:
        from straylight.__main__ import main

        sl = self._write_sl(tmp_path, "nogrant.sl", "(defn f () (print 1))")
        rc = main(["--native", sl])
        captured = capsys.readouterr()
        assert rc == 1
        assert "capability 'io' required by 'print'" in captured.err

    def test_native_flag_codegen_error_is_prosecutorial(self, tmp_path, capsys) -> None:
        from straylight.__main__ import main

        sl = self._write_sl(tmp_path, "float.sl", "(defn f () 3.14)")
        rc = main(["--native", sl])
        captured = capsys.readouterr()
        assert rc == 1
        assert "compile error" in captured.err
        assert "Float literals are not supported" in captured.err