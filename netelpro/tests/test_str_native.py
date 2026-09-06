"""Unit and JIT tests for Netelpro string production and arena allocator (str_native.py).

Tests:
1. Pure Python interpreter counterparts:
   - str-cat combinations (normal, empty, unicode, emojis).
   - int->str boundary vectors: 0, 7, -42, 2147483647, 9223372036854775807, -9223372036854775808.
   - Type verification on invalid inputs.
2. Bridge protocol helpers:
   - read_arena_string on ctypes pointers and integer addresses.
   - NULL pointer handling raising StrNativeError with line=0, col=0.
   - consume_arena semantic marker.
3. Native arena emission and real JIT execution (guarded by pytest.importorskip("llvmlite")):
   - emit_arena_globals emission sanity.
   - emit_strlit execution and verification.
   - emit_str_cat execution of two string literals.
   - emit_int_to_str execution of -42 and edge cases.
   - Arena overflow (> 64KB) yielding NULL and raising StrNativeError via bridge.
   - Per-call reset protocol ensuring subsequent calls reuse arena from offset 0.
"""

from __future__ import annotations

import ctypes
from typing import Any

import pytest
from netelpro.str_native import (
    ARENA_SIZE,
    NUL_SENTINEL,
    StrNativeError,
    consume_arena,
    emit_arena_alloc,
    emit_arena_globals,
    emit_arena_reset,
    emit_int_to_str,
    emit_str_cat,
    emit_strlit,
    interp_int_to_str,
    interp_str_cat,
    read_arena_string,
)

# ---------------------------------------------------------------------------
# Section 1: Interpreter Counterpart Tests
# ---------------------------------------------------------------------------


def test_interp_str_cat_combos() -> None:
    """Verify interp_str_cat with standard, empty, and unicode strings."""
    assert interp_str_cat("hello ", "world") == "hello world"
    assert interp_str_cat("", "") == ""
    assert interp_str_cat("prefix", "") == "prefix"
    assert interp_str_cat("", "suffix") == "suffix"
    assert interp_str_cat("café ", "☕") == "café ☕"
    assert interp_str_cat("日本語", "テスト") == "日本語テスト"
    assert interp_str_cat("🚀", "✨") == "🚀✨"


def test_interp_str_cat_type_errors() -> None:
    """Verify interp_str_cat raises TypeError when operands are not str."""
    with pytest.raises(TypeError, match="'str-cat' operands must be Str"):
        interp_str_cat(123, "abc")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="'str-cat' operands must be Str"):
        interp_str_cat("abc", True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="'str-cat' operands must be Str"):
        interp_str_cat(None, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        (0, "0"),
        (7, "7"),
        (-42, "-42"),
        (2147483647, "2147483647"),
        (9223372036854775807, "9223372036854775807"),
        (-9223372036854775808, "-9223372036854775808"),
    ],
)
def test_interp_int_to_str_boundary_vectors(val: int, expected: str) -> None:
    """Verify interp_int_to_str across all required integer boundaries."""
    assert interp_int_to_str(val) == expected


def test_interp_int_to_str_type_errors() -> None:
    """Verify interp_int_to_str raises TypeError for non-int and bool values."""
    with pytest.raises(TypeError, match="'int->str' operand must be Int"):
        interp_int_to_str(True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="'int->str' operand must be Int"):
        interp_int_to_str(False)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="'int->str' operand must be Int"):
        interp_int_to_str(3.14)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="'int->str' operand must be Int"):
        interp_int_to_str("42")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Section 2: Bridge Protocol Tests
# ---------------------------------------------------------------------------


def test_read_arena_string_valid() -> None:
    """Verify read_arena_string extracts UTF-8 strings from pointers and addresses."""
    # From c_char_p
    p_char = ctypes.c_char_p(b"Hello Netelpro\x00")
    assert read_arena_string(p_char) == "Hello Netelpro"

    # From memory buffer and integer address
    buf = ctypes.create_string_buffer("Unicode: café 🚀".encode() + NUL_SENTINEL)
    addr = ctypes.addressof(buf)
    assert read_arena_string(addr) == "Unicode: café 🚀"

    # From c_void_p
    p_void = ctypes.c_void_p(addr)
    assert read_arena_string(p_void) == "Unicode: café 🚀"


def test_read_arena_string_null_raises_str_native_error() -> None:
    """Verify read_arena_string raises StrNativeError with line=0, col=0 on NULL."""
    for null_ptr in [None, 0, ctypes.c_char_p(None), ctypes.c_void_p(0), ctypes.c_void_p(None)]:
        with pytest.raises(StrNativeError) as excinfo:
            read_arena_string(null_ptr)
        err = excinfo.value
        assert err.line == 0
        assert err.col == 0
        assert "arena overflow" in err.message.lower()


def test_consume_arena_marker() -> None:
    """Verify consume_arena executes as a valid no-op without exception."""
    consume_arena()


# ---------------------------------------------------------------------------
# Section 3: LLVM Native Emission and JIT Execution Tests
# ---------------------------------------------------------------------------


def _init_and_compile_jit(mod: Any) -> Any:
    """Compile an LLVM module to native machine code using the established recipe."""
    llb = pytest.importorskip("llvmlite.binding")

    llb.initialize_native_target()
    llb.initialize_native_asmprinter()
    target = llb.Target.from_default_triple()
    tm = target.create_target_machine(opt=2)
    backing = llb.parse_assembly("")
    engine = llb.create_mcjit_compiler(backing, tm)
    llvm_mod = llb.parse_assembly(str(mod))
    engine.add_module(llvm_mod)
    engine.finalize_object()
    return engine


def test_arena_globals_emission() -> None:
    """Verify emit_arena_globals creates the static 64KB array and bump pointer."""
    ir = pytest.importorskip("llvmlite.ir")

    mod = ir.Module(name="test_globals_module")
    arena_gv, head_gv = emit_arena_globals(mod)

    assert arena_gv.name == "__netelpro_arena"
    assert head_gv.name == "arena_head"
    assert "__netelpro_arena" in mod.globals
    assert "arena_head" in mod.globals

    # Array type must have capacity ARENA_SIZE = 65536
    assert arena_gv.type.pointee.count == ARENA_SIZE
    assert head_gv.type.pointee.width == 64


def test_jit_strlit_emission() -> None:
    """Emit a string literal into the arena, execute via JIT, and read via bridge."""
    ir = pytest.importorskip("llvmlite.ir")

    mod = ir.Module(name="test_strlit_mod")
    emit_arena_globals(mod)

    fn = ir.Function(mod, ir.FunctionType(ir.PointerType(), []), "emit_lit_test")
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    emit_arena_reset(b)
    ptr = emit_strlit(b, "Hello from JIT!")
    b.ret(ptr)

    engine = _init_and_compile_jit(mod)
    addr = engine.get_function_address("emit_lit_test")
    c_fn = ctypes.CFUNCTYPE(ctypes.c_void_p)(addr)
    res_ptr = c_fn()

    assert res_ptr is not None
    assert read_arena_string(res_ptr) == "Hello from JIT!"


def test_jit_str_cat_emission() -> None:
    """Emit str-cat of two literals, execute via JIT, and verify resulting buffer."""
    ir = pytest.importorskip("llvmlite.ir")

    mod = ir.Module(name="test_str_cat_mod")
    emit_arena_globals(mod)

    fn = ir.Function(mod, ir.FunctionType(ir.PointerType(), []), "emit_cat_test")
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    emit_arena_reset(b)
    p1 = emit_strlit(b, "Netelpro ")
    p2 = emit_strlit(b, "Compiler v0.5")
    cat_ptr = emit_str_cat(b, p1, p2, mod)
    b.ret(cat_ptr)

    engine = _init_and_compile_jit(mod)
    addr = engine.get_function_address("emit_cat_test")
    c_fn = ctypes.CFUNCTYPE(ctypes.c_void_p)(addr)
    res_ptr = c_fn()

    assert res_ptr is not None
    assert read_arena_string(res_ptr) == "Netelpro Compiler v0.5"


def test_jit_int_to_str_minus_42() -> None:
    """Emit int->str of -42, execute via JIT, and verify resulting buffer."""
    ir = pytest.importorskip("llvmlite.ir")

    mod = ir.Module(name="test_i2s_neg42_mod")
    emit_arena_globals(mod)

    fn = ir.Function(mod, ir.FunctionType(ir.PointerType(), []), "emit_neg42_test")
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    emit_arena_reset(b)
    p = emit_int_to_str(b, -42, mod)
    b.ret(p)

    engine = _init_and_compile_jit(mod)
    addr = engine.get_function_address("emit_neg42_test")
    c_fn = ctypes.CFUNCTYPE(ctypes.c_void_p)(addr)
    res_ptr = c_fn()

    assert res_ptr is not None
    assert read_arena_string(res_ptr) == "-42"


@pytest.mark.parametrize(
    "val",
    [
        0,
        7,
        -42,
        2147483647,
        9223372036854775807,
        -9223372036854775808,
    ],
)
def test_jit_int_to_str_all_vectors(val: int) -> None:
    """Emit int->str taking an i64 parameter, execute with all boundary vectors via JIT."""
    ir = pytest.importorskip("llvmlite.ir")

    mod = ir.Module(name=f"test_i2s_param_mod_{abs(val)}")
    emit_arena_globals(mod)

    fn = ir.Function(mod, ir.FunctionType(ir.PointerType(), [ir.IntType(64)]), "int_to_str_param")
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    emit_arena_reset(b)
    p = emit_int_to_str(b, fn.args[0], mod)
    b.ret(p)

    engine = _init_and_compile_jit(mod)
    addr = engine.get_function_address("int_to_str_param")
    c_fn = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int64)(addr)
    res_ptr = c_fn(val)

    assert res_ptr is not None
    assert read_arena_string(res_ptr) == str(val)


def test_jit_arena_overflow_yields_null_and_raises_error() -> None:
    """Construct a string concat exceeding 64KB, verify NULL return and StrNativeError."""
    ir = pytest.importorskip("llvmlite.ir")

    mod = ir.Module(name="test_overflow_mod")
    emit_arena_globals(mod)

    fn = ir.Function(mod, ir.FunctionType(ir.PointerType(), []), "overflow_test")
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    emit_arena_reset(b)

    # First consume 40,000 bytes in the arena
    emit_arena_alloc(b, 40000)

    # Two strings of 15,000 bytes each: total concat needed = 15000 + 15000 + 1 = 30001
    # 40000 + 30001 = 70001 > 65536 (ARENA_SIZE), forcing overflow
    p1 = emit_strlit(b, "A" * 15000)
    p2 = emit_strlit(b, "B" * 15000)
    cat_ptr = emit_str_cat(b, p1, p2, mod)
    b.ret(cat_ptr)

    engine = _init_and_compile_jit(mod)
    addr = engine.get_function_address("overflow_test")
    c_fn = ctypes.CFUNCTYPE(ctypes.c_void_p)(addr)
    res_ptr = c_fn()

    # Native pointer must be NULL (None or 0)
    assert res_ptr is None or res_ptr == 0

    # Bridge protocol must raise StrNativeError
    with pytest.raises(StrNativeError) as excinfo:
        read_arena_string(res_ptr)
    assert excinfo.value.line == 0
    assert excinfo.value.col == 0


def test_jit_per_call_reset_protocol() -> None:
    """Verify that two sequential calls reuse the arena from offset 0 after reset."""
    ir = pytest.importorskip("llvmlite.ir")

    mod = ir.Module(name="test_reset_protocol_mod")
    emit_arena_globals(mod)

    # Function 1: resets arena and writes a 32-byte literal
    fn1 = ir.Function(mod, ir.FunctionType(ir.PointerType(), []), "call_1")
    b1 = ir.IRBuilder(fn1.append_basic_block("entry"))
    emit_arena_reset(b1)
    p1 = emit_strlit(b1, "First Call: 0123456789abcdef01234")
    b1.ret(p1)

    # Function 2: resets arena and writes a different literal
    fn2 = ir.Function(mod, ir.FunctionType(ir.PointerType(), []), "call_2")
    b2 = ir.IRBuilder(fn2.append_basic_block("entry"))
    emit_arena_reset(b2)
    p2 = emit_strlit(b2, "Second Call: Reused Offset Zero!!")
    b2.ret(p2)

    engine = _init_and_compile_jit(mod)
    addr1 = engine.get_function_address("call_1")
    addr2 = engine.get_function_address("call_2")

    c_fn1 = ctypes.CFUNCTYPE(ctypes.c_void_p)(addr1)
    c_fn2 = ctypes.CFUNCTYPE(ctypes.c_void_p)(addr2)

    res1 = c_fn1()
    consume_arena()
    res2 = c_fn2()

    assert res1 is not None and res2 is not None
    # Both allocations must start at the exact same base address (offset 0 in @__netelpro_arena)
    assert res1 == res2

    # The second call successfully produced its string at offset 0
    assert read_arena_string(res2) == "Second Call: Reused Offset Zero!!"
