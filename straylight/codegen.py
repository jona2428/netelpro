"""Straylight native code generator and JIT engine -- Phase 4 LLVM native backend.

The prosecutor thesis:
Straylight programs are compiled directly to native machine code via LLVM without
hidden coercions, silent fallbacks, or dynamic type ambiguities. Every rejection
at codegen is an honest prosecutorial declaration enforced before JIT execution.

Scope and Guarantees in v0.1:
1. Value Model:
   - Integer values compile to native 64-bit signed integers (`i64`).
   - Boolean values compile to native 1-bit flags (`i1`).
   - Strict Boolean typing: branching (`if`) and logical (`and`, `or`, `not`)
     conditions must evaluate to `i1`. Any attempt to pass integers or unrepresented
     types to boolean constructs is prosecuted as a compile error with exact source
     provenance.
   - Strict Integer typing: arithmetic (`+`, `-`, `*`, `quot`, `rem`) and comparison
     (`<`, `<=`, `>`, `>=`, `==`, `!=`) operands must evaluate to `i64`. Booleans crossing
     arithmetic or comparison operators are rejected at compile time.

2. Explicit Rejection Policy (Unrepresented Forms in v0.1):
   - Float literals (`FloatLit`) and float division (`/`) are rejected: floating-point
     semantics are deferred to later revisions.
   - String literals (`StrLit`), Nil (`NilLit`), and List construction (`ListLit`) are
     rejected: boxed heap structures are unrepresented in v0.1.
   - Anonymous functions (`Fn`) and typed hole placeholders (`Sorry`) are rejected:
     closures and incomplete programs must not generate native artifacts.
   - Non-compilable primitives (`list`, `cons`, `head`, `tail`, `is-nil`, `len`, `nth`,
     `str-cat`, `str-len`, `int->str`, `str->int`, `int->float`) are rejected with exact coordinates.
   - Top-level `def` bindings must evaluate to literal `IntLit` or `BoolLit`.
   - Any function call with an arity mismatch or referencing an undeclared symbol is
     reported with exact line and column provenance.

3. Static Capability Auditing:
   - Effects are verified at compile time. The `print` primitive requires that the translation
     unit declares `(grant io)` at top level. Missing capability grants are prosecuted at compile
     time before code emission.

4. Tail Call Optimization (TCO):
   - Named function definitions (`defn`) eliminate stack growth for recursive calls in tail
     position to the same function. Parameter slots (`alloca i64`) are updated in-place followed
     by an unconditional branch back to the loop header, guaranteeing that recursive loops
     (such as `sum-to 100000 0`) execute in constant stack space.
   - Non-tail calls (such as branching recursion in `fib`) compile to direct native calls.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import llvmlite.binding as llb
import llvmlite.ir as ir

from straylight.ast_nodes import (
    And,
    BoolLit,
    Call,
    Def,
    Defn,
    FloatLit,
    Fn,
    Grant,
    If,
    IntLit,
    Let,
    ListLit,
    NilLit,
    Node,
    Or,
    Program,
    Sorry,
    StrLit,
    Sym,
)
from straylight.evaluator import StrayError


class CodegenError(StrayError):
    """Prosecutorial compilation diagnostic record with exact source coordinates."""

    def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
        self.message = message
        self.line = line
        self.col = col
        super().__init__(f"compile error at line {line}, col {col}: {message}")


TYPE_INT = "Int"
TYPE_BOOL = "Bool"
TYPE_NIL = "Nil"


class TypeVar:
    """Type variable for bidirectional unification."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.bound: str | TypeVar | None = None

    def find(self) -> str | TypeVar:
        curr: str | TypeVar = self
        while isinstance(curr, TypeVar) and curr.bound is not None:
            curr = curr.bound
        return curr

    def resolve(self) -> str:
        res = self.find()
        return TYPE_INT if isinstance(res, TypeVar) else res


def _unify(
    t1: str | TypeVar,
    t2: str | TypeVar,
    line: int,
    col: int,
    context: str = "",
) -> str | TypeVar:
    if isinstance(t1, TypeVar):
        t1 = t1.find()
    if isinstance(t2, TypeVar):
        t2 = t2.find()

    if isinstance(t1, TypeVar) and isinstance(t2, TypeVar):
        if t1 is not t2:
            t1.bound = t2
        return t2
    elif isinstance(t1, TypeVar):
        t1.bound = t2
        return t2
    elif isinstance(t2, TypeVar):
        t2.bound = t1
        return t1
    else:
        if t1 != t2:
            ctx_msg = f" for {context}" if context else ""
            raise CodegenError(f"type mismatch{ctx_msg}: expected {t1}, got {t2}", line, col)
        return t1


NON_COMPILABLE_PRIMITIVES: set[str] = {
    "/",
    "list",
    "cons",
    "head",
    "tail",
    "is-nil",
    "len",
    "nth",
    "str-cat",
    "str-len",
    "int->str",
    "str->int",
    "int->float",
}

COMPILABLE_PRIMITIVES: set[str] = {
    "+",
    "-",
    "*",
    "quot",
    "rem",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "not",
    "print",
}

RESERVED_HEADS: set[str] = {
    "def",
    "defn",
    "let",
    "fn",
    "if",
    "and",
    "or",
    "sorry",
    "grant",
    *NON_COMPILABLE_PRIMITIVES,
    *COMPILABLE_PRIMITIVES,
}


@dataclass
class GlobalDef:
    name: str
    type: str
    value: int | bool
    node: Def


@dataclass
class DefnInfo:
    node: Defn
    name: str
    param_names: list[str]
    param_vars: list[TypeVar]
    ret_var: TypeVar
    param_types: list[str] | None = None
    ret_type: str = ""


_LLVM_INITIALIZED = False


def _init_llvm() -> None:
    global _LLVM_INITIALIZED
    if not _LLVM_INITIALIZED:
        llb.initialize_native_target()
        llb.initialize_native_asmprinter()
        _LLVM_INITIALIZED = True


@dataclass
class FunctionContext:
    fn_name: str
    llvm_fn: ir.Function
    ret_type_str: str
    param_names: list[str]
    param_slots: dict[str, ir.AllocaInstr]
    loop_header: ir.Block


class CompiledProgram:
    """A JIT-compiled Straylight native program backed by an LLVM execution engine."""

    def __init__(
        self,
        engine: Any,
        entry_point: str = "main",
        module: ir.Module | None = None,
    ) -> None:
        self.engine = engine
        self.entry_point = entry_point
        self.module = module

    def run(self) -> int:
        """Executes top-level forms and returns the i64 result of the last expression form (nil -> 0)."""
        addr = self.engine.get_function_address(self.entry_point)
        c_fn = ctypes.CFUNCTYPE(ctypes.c_int64)(addr)
        return int(c_fn())


def _to_llvm_type(t: str) -> ir.Type:
    return ir.IntType(1) if t == TYPE_BOOL else ir.IntType(64)


def _llvm_name_for(stray_name: str) -> str:
    if stray_name in ("main", "printf", "exit", "__stray_div_zero_error"):
        return f"__sl_user_{stray_name}__"
    return stray_name


def compile_program(program: Program) -> CompiledProgram:
    """Statically validates and compiles a Straylight Program AST to native code using LLVM.

    Raises CodegenError on any unrepresented form, type conflict, undefined reference,
    or arity mismatch before invoking the native JIT engine.
    """
    if not isinstance(program, Program):
        raise CodegenError(f"expected Program AST node, got {type(program).__name__}", 0, 0)

    # -----------------------------------------------------------------------
    # Pass 1: Top-level declaration collection and arity verification
    # -----------------------------------------------------------------------
    granted_caps: set[str] = set()
    global_defs: dict[str, GlobalDef] = {}
    defns: dict[str, DefnInfo] = {}

    for form in program.forms:
        if isinstance(form, Grant):
            for cap in form.caps:
                c_name = cap.name if isinstance(cap, Sym) else str(cap)
                granted_caps.add(c_name)
        elif isinstance(form, Def):
            name = form.name.name
            if name in RESERVED_HEADS:
                raise CodegenError(f"cannot redefine reserved head '{name}'", form.line, form.col)
            if name in global_defs or name in defns:
                raise CodegenError(f"duplicate definition of '{name}'", form.line, form.col)
            if isinstance(form.value, IntLit):
                global_defs[name] = GlobalDef(name, TYPE_INT, form.value.value, form)
            elif isinstance(form.value, BoolLit):
                global_defs[name] = GlobalDef(name, TYPE_BOOL, form.value.value, form)
            else:
                raise CodegenError(
                    f"'def' value must be an Int or Bool literal in v0.1, got {type(form.value).__name__}",
                    form.line,
                    form.col,
                )
        elif isinstance(form, Defn):
            name = form.name.name
            if name in RESERVED_HEADS:
                raise CodegenError(f"cannot redefine reserved head '{name}'", form.line, form.col)
            if name in global_defs or name in defns:
                raise CodegenError(f"duplicate definition of '{name}'", form.line, form.col)
            param_names: list[str] = []
            for p in form.params:
                p_name = p.name if isinstance(p, Sym) else str(p)
                if p_name in RESERVED_HEADS:
                    p_line = getattr(p, "line", 0) or form.line
                    p_col = getattr(p, "col", 0) or form.col
                    raise CodegenError(f"parameter '{p_name}' shadows reserved head", p_line, p_col)
                if p_name in param_names:
                    p_line = getattr(p, "line", 0) or form.line
                    p_col = getattr(p, "col", 0) or form.col
                    raise CodegenError(f"duplicate parameter '{p_name}' in defn '{name}'", p_line, p_col)
                param_names.append(p_name)
            p_vars = [TypeVar(f"{name}.{p}") for p in param_names]
            r_var = TypeVar(f"{name}.ret")
            defns[name] = DefnInfo(
                node=form,
                name=name,
                param_names=param_names,
                param_vars=p_vars,
                ret_var=r_var,
            )

    # -----------------------------------------------------------------------
    # Pass 2: Static walk & type inference (rejecting invalid nodes BEFORE JIT)
    # -----------------------------------------------------------------------
    def typecheck(
        node: Node,
        lex_env: dict[str, str | TypeVar],
        current_defn: DefnInfo | None,
    ) -> str | TypeVar:
        if isinstance(node, IntLit):
            return TYPE_INT
        if isinstance(node, BoolLit):
            return TYPE_BOOL
        if isinstance(node, FloatLit):
            raise CodegenError("Float literals are not supported in native codegen v0.1", node.line, node.col)
        if isinstance(node, StrLit):
            raise CodegenError("String literals are not supported in native codegen v0.1", node.line, node.col)
        if isinstance(node, NilLit):
            raise CodegenError("Nil literals are not supported in native codegen v0.1", node.line, node.col)
        if isinstance(node, ListLit):
            raise CodegenError("List literals are not supported in native codegen v0.1", node.line, node.col)
        if isinstance(node, Fn):
            raise CodegenError("Anonymous functions (fn) are not supported in native codegen v0.1", node.line, node.col)
        if isinstance(node, Sorry):
            raise CodegenError(f"Cannot compile 'sorry' hole: {node.reason.value!r}", node.line, node.col)
        if isinstance(node, (Def, Defn, Grant)):
            raise CodegenError(f"'{type(node).__name__.lower()}' is only permitted at top level", node.line, node.col)

        if isinstance(node, Sym):
            if node.name in lex_env:
                return lex_env[node.name]
            if node.name in global_defs:
                return global_defs[node.name].type
            raise CodegenError(f"undefined symbol '{node.name}'", node.line, node.col)

        if isinstance(node, Let):
            val_t = typecheck(node.value, lex_env, current_defn)
            new_env = lex_env.copy()
            new_env[node.name.name] = val_t
            return typecheck(node.body, new_env, current_defn)

        if isinstance(node, If):
            c_t = typecheck(node.cond, lex_env, current_defn)
            _unify(TYPE_BOOL, c_t, node.cond.line, node.cond.col, "'if' condition")
            then_t = typecheck(node.then, lex_env, current_defn)
            else_t = typecheck(node.else_, lex_env, current_defn)
            return _unify(then_t, else_t, node.line, node.col, "'if' branches")

        if isinstance(node, And):
            l_t = typecheck(node.l, lex_env, current_defn)
            _unify(TYPE_BOOL, l_t, node.l.line, node.l.col, "'and' left operand")
            r_t = typecheck(node.r, lex_env, current_defn)
            _unify(TYPE_BOOL, r_t, node.r.line, node.r.col, "'and' right operand")
            return TYPE_BOOL

        if isinstance(node, Or):
            l_t = typecheck(node.l, lex_env, current_defn)
            _unify(TYPE_BOOL, l_t, node.l.line, node.l.col, "'or' left operand")
            r_t = typecheck(node.r, lex_env, current_defn)
            _unify(TYPE_BOOL, r_t, node.r.line, node.r.col, "'or' right operand")
            return TYPE_BOOL

        if isinstance(node, Call):
            head = node.head
            if head == "/":
                raise CodegenError("'/' is not supported in native codegen v0.1 (returns Float)", node.line, node.col)
            if head in NON_COMPILABLE_PRIMITIVES:
                raise CodegenError(f"primitive '{head}' is not supported in native codegen v0.1", node.line, node.col)

            if head in ("+", "-", "*", "quot", "rem"):
                if len(node.args) != 2:
                    raise CodegenError(f"'{head}' expects 2 arguments, got {len(node.args)}", node.line, node.col)
                a0_t = typecheck(node.args[0], lex_env, current_defn)
                _unify(TYPE_INT, a0_t, node.args[0].line, node.args[0].col, f"'{head}' first operand")
                a1_t = typecheck(node.args[1], lex_env, current_defn)
                _unify(TYPE_INT, a1_t, node.args[1].line, node.args[1].col, f"'{head}' second operand")
                return TYPE_INT

            if head in ("<", "<=", ">", ">=", "==", "!="):
                if len(node.args) != 2:
                    raise CodegenError(f"'{head}' expects 2 arguments, got {len(node.args)}", node.line, node.col)
                a0_t = typecheck(node.args[0], lex_env, current_defn)
                _unify(TYPE_INT, a0_t, node.args[0].line, node.args[0].col, f"'{head}' first operand")
                a1_t = typecheck(node.args[1], lex_env, current_defn)
                _unify(TYPE_INT, a1_t, node.args[1].line, node.args[1].col, f"'{head}' second operand")
                return TYPE_BOOL

            if head == "not":
                if len(node.args) != 1:
                    raise CodegenError(f"'not' expects 1 argument, got {len(node.args)}", node.line, node.col)
                a0_t = typecheck(node.args[0], lex_env, current_defn)
                _unify(TYPE_BOOL, a0_t, node.args[0].line, node.args[0].col, "'not' operand")
                return TYPE_BOOL

            if head == "print":
                if len(node.args) != 1:
                    raise CodegenError(f"'print' expects 1 argument, got {len(node.args)}", node.line, node.col)
                if "io" not in granted_caps:
                    raise CodegenError("capability 'io' required by 'print' but not granted", node.line, node.col)
                a0_t = typecheck(node.args[0], lex_env, current_defn)
                _unify(TYPE_INT, a0_t, node.args[0].line, node.args[0].col, "'print' operand")
                return TYPE_NIL

            if head in defns:
                target_defn = defns[head]
                expected_arity = len(target_defn.param_names)
                if len(node.args) != expected_arity:
                    raise CodegenError(
                        f"'{head}' expects {expected_arity} argument(s), got {len(node.args)}",
                        node.line,
                        node.col,
                    )
                for i, (arg_node, p_var) in enumerate(zip(node.args, target_defn.param_vars, strict=True)):
                    arg_t = typecheck(arg_node, lex_env, current_defn)
                    _unify(p_var, arg_t, arg_node.line, arg_node.col, f"argument {i + 1} to '{head}'")
                return target_defn.ret_var

            raise CodegenError(f"unknown function or primitive '{head}'", node.line, node.col)

        raise CodegenError(
            f"unsupported AST node '{type(node).__name__}'", getattr(node, "line", 0), getattr(node, "col", 0)
        )

    # Typecheck all defn bodies
    for d_info in defns.values():
        initial_env = {p_name: p_var for p_name, p_var in zip(d_info.param_names, d_info.param_vars, strict=True)}
        body_t = typecheck(d_info.node.body, initial_env, d_info)
        _unify(d_info.ret_var, body_t, d_info.node.line, d_info.node.col, f"return of '{d_info.name}'")

    # Typecheck top-level expressions
    for form in program.forms:
        if not isinstance(form, (Grant, Def, Defn)):
            typecheck(form, {}, None)

    # Resolve all TypeVars
    for d_info in defns.values():
        d_info.param_types = [p.resolve() for p in d_info.param_vars]
        d_info.ret_type = d_info.ret_var.resolve()

    # -----------------------------------------------------------------------
    # Pass 3: LLVM IR Generation
    # -----------------------------------------------------------------------
    _init_llvm()
    mod = ir.Module(name="straylight_jit")
    i64 = ir.IntType(64)
    i32 = ir.IntType(32)
    i1 = ir.IntType(1)

    printf_ty = ir.FunctionType(i32, [ir.PointerType()], var_arg=True)
    printf_fn = ir.Function(mod, printf_ty, name="printf")

    exit_ty = ir.FunctionType(ir.VoidType(), [i32])
    exit_fn = ir.Function(mod, exit_ty, name="exit")

    # Global runtime error helper for division by zero
    msg_bytes = bytearray(b"runtime error: division by zero\n\0")
    msg_const = ir.Constant(ir.ArrayType(ir.IntType(8), len(msg_bytes)), msg_bytes)
    msg_gv = ir.GlobalVariable(mod, msg_const.type, name="__str_div_zero")
    msg_gv.linkage = "internal"
    msg_gv.global_constant = True
    msg_gv.initializer = msg_const

    div_zero_ty = ir.FunctionType(ir.VoidType(), [])
    div_zero_fn = ir.Function(mod, div_zero_ty, name="__stray_div_zero_error")
    div_zero_bb = div_zero_fn.append_basic_block("entry")
    div_zero_b = ir.IRBuilder(div_zero_bb)
    fmt_div_zero_ptr = div_zero_b.bitcast(msg_gv, ir.PointerType())
    div_zero_b.call(printf_fn, [fmt_div_zero_ptr])
    div_zero_b.call(exit_fn, [ir.Constant(i32, 1)])
    div_zero_b.unreachable()

    # Format string for print
    fmt_int_bytes = bytearray(b"%lld\n\0")
    fmt_int_const = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt_int_bytes)), fmt_int_bytes)
    fmt_int_gv = ir.GlobalVariable(mod, fmt_int_const.type, name="__fmt_int")
    fmt_int_gv.linkage = "internal"
    fmt_int_gv.global_constant = True
    fmt_int_gv.initializer = fmt_int_const

    # Pre-declare user functions
    llvm_functions: dict[str, ir.Function] = {}
    for name, d_info in defns.items():
        assert d_info.param_types is not None
        param_tys = [_to_llvm_type(pt) for pt in d_info.param_types]
        ret_ty = _to_llvm_type(d_info.ret_type)
        fn_ty = ir.FunctionType(ret_ty, param_tys)
        llvm_fn = ir.Function(mod, fn_ty, name=_llvm_name_for(name))
        llvm_functions[name] = llvm_fn

    # Expression compiler supporting TCO
    def compile_expr(
        node: Node,
        env: dict[str, ir.Value | ir.AllocaInstr],
        is_tail: bool,
        builder: ir.IRBuilder,
        ctx: FunctionContext | None,
    ) -> ir.Value | None:
        if builder.block.is_terminated:
            return None

        if isinstance(node, IntLit):
            val = ir.Constant(i64, node.value)
            if is_tail:
                builder.ret(val)
            return val

        if isinstance(node, BoolLit):
            val = ir.Constant(i1, 1 if node.value else 0)
            if is_tail:
                builder.ret(val)
            return val

        if isinstance(node, Sym):
            if node.name in env:
                entry = env[node.name]
                val = builder.load(entry, name=node.name) if isinstance(entry, ir.AllocaInstr) else entry
            elif node.name in global_defs:
                g = global_defs[node.name]
                val = ir.Constant(_to_llvm_type(g.type), g.value)
            else:
                raise CodegenError(f"undefined symbol '{node.name}'", node.line, node.col)
            if is_tail:
                builder.ret(val)
            return val

        if isinstance(node, Let):
            val = compile_expr(node.value, env, is_tail=False, builder=builder, ctx=ctx)
            new_env = env.copy()
            assert val is not None
            new_env[node.name.name] = val
            return compile_expr(node.body, new_env, is_tail=is_tail, builder=builder, ctx=ctx)

        if isinstance(node, If):
            cond_val = compile_expr(node.cond, env, is_tail=False, builder=builder, ctx=ctx)
            assert cond_val is not None
            if is_tail:
                then_bb = builder.append_basic_block("if.then")
                else_bb = builder.append_basic_block("if.else")
                builder.cbranch(cond_val, then_bb, else_bb)

                builder.position_at_end(then_bb)
                compile_expr(node.then, env, is_tail=True, builder=builder, ctx=ctx)

                builder.position_at_end(else_bb)
                compile_expr(node.else_, env, is_tail=True, builder=builder, ctx=ctx)
                return None
            else:
                then_bb = builder.append_basic_block("if.then")
                else_bb = builder.append_basic_block("if.else")
                merge_bb = builder.append_basic_block("if.merge")
                builder.cbranch(cond_val, then_bb, else_bb)

                builder.position_at_end(then_bb)
                then_val = compile_expr(node.then, env, is_tail=False, builder=builder, ctx=ctx)
                then_end = builder.block
                then_reached = not then_end.is_terminated
                if then_reached:
                    builder.branch(merge_bb)

                builder.position_at_end(else_bb)
                else_val = compile_expr(node.else_, env, is_tail=False, builder=builder, ctx=ctx)
                else_end = builder.block
                else_reached = not else_end.is_terminated
                if else_reached:
                    builder.branch(merge_bb)

                builder.position_at_end(merge_bb)
                if then_reached and else_reached:
                    assert then_val is not None and else_val is not None
                    phi = builder.phi(then_val.type, name="if.val")
                    phi.add_incoming(then_val, then_end)
                    phi.add_incoming(else_val, else_end)
                    return phi
                elif then_reached:
                    return then_val
                elif else_reached:
                    return else_val
                else:
                    builder.unreachable()
                    return None

        if isinstance(node, And):
            l_val = compile_expr(node.l, env, is_tail=False, builder=builder, ctx=ctx)
            assert l_val is not None
            l_end = builder.block
            rhs_bb = builder.append_basic_block("and.rhs")
            merge_bb = builder.append_basic_block("and.merge")
            builder.cbranch(l_val, rhs_bb, merge_bb)

            builder.position_at_end(rhs_bb)
            r_val = compile_expr(node.r, env, is_tail=False, builder=builder, ctx=ctx)
            assert r_val is not None
            r_end = builder.block
            r_reached = not r_end.is_terminated
            if r_reached:
                builder.branch(merge_bb)

            builder.position_at_end(merge_bb)
            phi = builder.phi(i1, name="and.res")
            phi.add_incoming(ir.Constant(i1, 0), l_end)
            if r_reached:
                phi.add_incoming(r_val, r_end)
            if is_tail:
                builder.ret(phi)
            return phi

        if isinstance(node, Or):
            l_val = compile_expr(node.l, env, is_tail=False, builder=builder, ctx=ctx)
            assert l_val is not None
            l_end = builder.block
            rhs_bb = builder.append_basic_block("or.rhs")
            merge_bb = builder.append_basic_block("or.merge")
            builder.cbranch(l_val, merge_bb, rhs_bb)

            builder.position_at_end(rhs_bb)
            r_val = compile_expr(node.r, env, is_tail=False, builder=builder, ctx=ctx)
            assert r_val is not None
            r_end = builder.block
            r_reached = not r_end.is_terminated
            if r_reached:
                builder.branch(merge_bb)

            builder.position_at_end(merge_bb)
            phi = builder.phi(i1, name="or.res")
            phi.add_incoming(ir.Constant(i1, 1), l_end)
            if r_reached:
                phi.add_incoming(r_val, r_end)
            if is_tail:
                builder.ret(phi)
            return phi

        if isinstance(node, Call):
            head = node.head
            if head in ("+", "-", "*"):
                a = compile_expr(node.args[0], env, is_tail=False, builder=builder, ctx=ctx)
                b = compile_expr(node.args[1], env, is_tail=False, builder=builder, ctx=ctx)
                assert a is not None and b is not None
                if head == "+":
                    res = builder.add(a, b)
                elif head == "-":
                    res = builder.sub(a, b)
                else:
                    res = builder.mul(a, b)
                if is_tail:
                    builder.ret(res)
                return res

            if head in ("quot", "rem"):
                a = compile_expr(node.args[0], env, is_tail=False, builder=builder, ctx=ctx)
                b = compile_expr(node.args[1], env, is_tail=False, builder=builder, ctx=ctx)
                assert a is not None and b is not None
                is_zero = builder.icmp_signed("==", b, ir.Constant(i64, 0))
                with builder.if_then(is_zero):
                    builder.call(div_zero_fn, [])
                    builder.unreachable()
                res = builder.sdiv(a, b) if head == "quot" else builder.srem(a, b)
                if is_tail:
                    builder.ret(res)
                return res

            if head in ("<", "<=", ">", ">=", "==", "!="):
                a = compile_expr(node.args[0], env, is_tail=False, builder=builder, ctx=ctx)
                b = compile_expr(node.args[1], env, is_tail=False, builder=builder, ctx=ctx)
                assert a is not None and b is not None
                res = builder.icmp_signed(head, a, b)
                if is_tail:
                    builder.ret(res)
                return res

            if head == "not":
                a = compile_expr(node.args[0], env, is_tail=False, builder=builder, ctx=ctx)
                assert a is not None
                res = builder.xor(a, ir.Constant(i1, 1))
                if is_tail:
                    builder.ret(res)
                return res

            if head == "print":
                a = compile_expr(node.args[0], env, is_tail=False, builder=builder, ctx=ctx)
                assert a is not None
                fmt_ptr = builder.bitcast(fmt_int_gv, ir.PointerType())
                builder.call(printf_fn, [fmt_ptr, a])
                res = ir.Constant(i64, 0)
                if is_tail:
                    builder.ret(res)
                return res

            if head in llvm_functions:
                if is_tail and ctx is not None and head == ctx.fn_name:
                    # SELF-TAIL CALL: TCO loop back-edge
                    arg_vals = [compile_expr(arg, env, is_tail=False, builder=builder, ctx=ctx) for arg in node.args]
                    for p_name, val in zip(ctx.param_names, arg_vals, strict=True):
                        assert val is not None
                        builder.store(val, ctx.param_slots[p_name])
                    builder.branch(ctx.loop_header)
                    return None
                else:
                    target_fn = llvm_functions[head]
                    arg_vals = [compile_expr(arg, env, is_tail=False, builder=builder, ctx=ctx) for arg in node.args]
                    res = builder.call(target_fn, arg_vals)
                    if is_tail:
                        builder.ret(res)
                    return res

        raise CodegenError(
            f"unsupported AST node '{type(node).__name__}'", getattr(node, "line", 0), getattr(node, "col", 0)
        )

    # Compile user defn bodies with parameter slots and TCO loop
    for d_info in defns.values():
        llvm_fn = llvm_functions[d_info.name]
        entry_bb = llvm_fn.append_basic_block("entry")
        b = ir.IRBuilder(entry_bb)

        assert d_info.param_types is not None
        param_slots: dict[str, ir.AllocaInstr] = {}
        for p_name, p_ty_str, arg in zip(d_info.param_names, d_info.param_types, llvm_fn.args, strict=True):
            arg.name = f"arg.{p_name}"
            slot = b.alloca(_to_llvm_type(p_ty_str), name=f"slot.{p_name}")
            b.store(arg, slot)
            param_slots[p_name] = slot

        loop_header = llvm_fn.append_basic_block("loop_header")
        b.branch(loop_header)
        b.position_at_end(loop_header)

        fn_ctx = FunctionContext(
            fn_name=d_info.name,
            llvm_fn=llvm_fn,
            ret_type_str=d_info.ret_type,
            param_names=d_info.param_names,
            param_slots=param_slots,
            loop_header=loop_header,
        )
        initial_env: dict[str, ir.Value | ir.AllocaInstr] = dict(param_slots)
        compile_expr(d_info.node.body, initial_env, is_tail=True, builder=b, ctx=fn_ctx)

    # Compile top-level expressions in main()
    main_ty = ir.FunctionType(i64, [])
    main_fn = ir.Function(mod, main_ty, name="main")
    main_bb = main_fn.append_basic_block("entry")
    mb = ir.IRBuilder(main_bb)

    expr_forms = [f for f in program.forms if not isinstance(f, (Grant, Def, Defn))]
    if not expr_forms:
        mb.ret(ir.Constant(i64, 0))
    else:
        for f in expr_forms[:-1]:
            compile_expr(f, env={}, is_tail=False, builder=mb, ctx=None)

        last_f = expr_forms[-1]
        last_val = compile_expr(last_f, env={}, is_tail=False, builder=mb, ctx=None)
        if not mb.block.is_terminated:
            if last_val is None:
                mb.ret(ir.Constant(i64, 0))
            elif last_val.type == i1:
                mb.ret(mb.zext(last_val, i64))
            else:
                mb.ret(last_val)

    # -----------------------------------------------------------------------
    # Pass 4: Native JIT Compilation using TargetMachine
    # -----------------------------------------------------------------------
    target = llb.Target.from_default_triple()
    tm = target.create_target_machine(opt=2)
    backing = llb.parse_assembly("")
    engine = llb.create_mcjit_compiler(backing, tm)

    llvm_mod = llb.parse_assembly(str(mod))
    engine.add_module(llvm_mod)
    engine.finalize_object()

    return CompiledProgram(engine=engine, entry_point="main", module=mod)


def compile_and_run(program: Program) -> int:
    """Convenience helper to compile and run a Straylight program AST."""
    return compile_program(program).run()
