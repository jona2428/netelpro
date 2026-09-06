"""Netelpro string production native engine and arena allocator -- Phase 5 backend.

This module provides the v0.5 string-production capability for Netelpro, including:
1. Static 64KB arena allocator emission for LLVM native code generation.
2. IR emitters for string literals, string concatenation, and integer-to-string conversion.
3. Pure Python interpreter counterparts ensuring 100% differential testing fidelity.
4. Host bridge protocol helpers to decode arena strings and handle overflow errors.

The Per-Call Arena Protocol:
-----------------------------
In Netelpro's native execution model, strings are represented at the native boundary
as NUL-terminated byte buffers (i8* / ptr). Dynamic string operations (literals,
`str-cat`, `int->str`) allocate temporary buffers from a thread-local or translation-unit-level
static 64KB arena (`@__netelpro_arena`).

Key mechanics:
- `@__netelpro_arena`: A static global byte array `[65536 x i8]` initialized to zero.
- `@arena_head`: An `i64` global tracking the current bump offset in bytes (0 to 65536).
- `emit_arena_reset(builder)`: Stores 0 to `@arena_head`. In the per-call protocol, this
  reset is emitted at the entry point of each compiled function or rule evaluation, ensuring
  constant memory footprint across repeated invocations.
- `emit_arena_alloc(builder, nbytes)`: Loads `@arena_head`, checks if `old_head + nbytes > 65536`.
  If within bounds, bumps `@arena_head` and returns an `i8*` pointer via GEP into
  `@__netelpro_arena` at `old_head`. On overflow (> 65536), returns an `i8*` null pointer without
  updating `@arena_head`.
- `read_arena_string(ptr)`: Bridges native string pointers back to Python `str`. If the native
  pointer is NULL (arena overflow occurred during execution), it raises `StrNativeError`
  with line=0, col=0.
- `consume_arena()`: A host-side no-op placeholder representing consumption of the current
  arena lifetime before the next reset.
- `INT64_MIN` conversion: In 64-bit two's complement arithmetic, negating `-9223372036854775808`
  produces `0x8000000000000000`. By treating this bit pattern as unsigned (`udiv`/`urem` by 10),
  it represents positive `+9223372036854775808`, enabling exact ASCII conversion without
  undefined behavior or signed integer overflow.
- String length resolution: `emit_str_cat` uses an external declaration of the standard C
  runtime function `strlen` to determine NUL-terminated string lengths.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

try:
    import llvmlite.ir as ir
except ImportError:  # pragma: no cover
    ir = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARENA_SIZE: int = 65536
"""Static capacity of the Netelpro string arena in bytes (64 KB)."""

NUL_SENTINEL: bytes = b"\x00"
"""NUL byte terminator for native C strings."""


# ---------------------------------------------------------------------------
# Bridge Protocol Exceptions & Helpers
# ---------------------------------------------------------------------------


class StrNativeError(Exception):
    """Prosecutorial diagnostic record for string native arena and conversion errors.

    Carries source coordinates (defaulting to line=0, col=0) and error message.
    """

    def __init__(
        self,
        message: str = "arena overflow: null string pointer",
        line: int = 0,
        col: int = 0,
    ) -> None:
        self.message: str = message
        self.line: int = line
        self.col: int = col
        super().__init__(f"string native error at line {line}, col {col}: {message}")

    def __str__(self) -> str:
        return f"string native error at line {self.line}, col {self.col}: {self.message}"


def read_arena_string(ptr: Any) -> str:
    """Decode a NUL-terminated UTF-8 string from a native pointer or integer address.

    Args:
        ptr: A ctypes pointer (e.g. c_char_p, c_void_p, POINTER(c_char)) or integer address.

    Returns:
        The decoded Python unicode string up to the NUL terminator.

    Raises:
        StrNativeError: If ptr is NULL (0, None, or null pointer), indicating arena overflow.
    """
    if ptr is None:
        raise StrNativeError("arena overflow: null string pointer", line=0, col=0)

    if isinstance(ptr, int):
        if ptr == 0:
            raise StrNativeError("arena overflow: null string pointer", line=0, col=0)
        c_str = ctypes.cast(ptr, ctypes.c_char_p).value
        if c_str is None:
            raise StrNativeError("arena overflow: null string pointer", line=0, col=0)
        return c_str.decode("utf-8")

    if isinstance(ptr, ctypes.c_char_p):
        if ptr.value is None:
            raise StrNativeError("arena overflow: null string pointer", line=0, col=0)
        return ptr.value.decode("utf-8")

    if hasattr(ptr, "value"):
        val = ptr.value
        if val is None or val == 0:
            raise StrNativeError("arena overflow: null string pointer", line=0, col=0)
        c_str = ctypes.cast(ptr, ctypes.c_char_p).value
        if c_str is None:
            raise StrNativeError("arena overflow: null string pointer", line=0, col=0)
        return c_str.decode("utf-8")

    try:
        addr = ctypes.cast(ptr, ctypes.c_void_p).value
        if addr is None or addr == 0:
            raise StrNativeError("arena overflow: null string pointer", line=0, col=0)
        c_str = ctypes.cast(ptr, ctypes.c_char_p).value
        if c_str is None:
            raise StrNativeError("arena overflow: null string pointer", line=0, col=0)
        return c_str.decode("utf-8")
    except Exception as e:
        if isinstance(e, StrNativeError):
            raise
        raise StrNativeError(f"cannot read arena string: {e}", line=0, col=0) from e


def consume_arena() -> None:
    """No-op placeholder for the arena reset protocol.

    In the Netelpro calling convention, the arena resets per call at the emission
    of the rule/function entry point (`emit_arena_reset`). This host function serves
    as a semantic lifecycle marker indicating that previously returned string pointers
    are invalidated before subsequent calls.
    """
    pass


# ---------------------------------------------------------------------------
# Pure Python Interpreter Counterparts
# ---------------------------------------------------------------------------


def interp_str_cat(a: str, b: str) -> str:
    """Concatenate two strings (interpreter counterpart).

    Args:
        a: First string operand.
        b: Second string operand.

    Returns:
        The concatenated string `a + b`.

    Raises:
        TypeError: If either operand is not a Python str.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError(
            f"'str-cat' operands must be Str, got {type(a).__name__} and {type(b).__name__}"
        )
    return a + b


def interp_int_to_str(n: int) -> str:
    """Convert an integer to a decimal ASCII string (interpreter counterpart).

    Args:
        n: 64-bit integer to convert.

    Returns:
        The decimal string representation ('-' for negatives, '0' for zero).

    Raises:
        TypeError: If n is not an integer or is a boolean.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"'int->str' operand must be Int, got {type(n).__name__}")
    return str(n)


# ---------------------------------------------------------------------------
# LLVM IR Emitters & Allocator
# ---------------------------------------------------------------------------


def _get_or_emit_arena(module: ir.Module) -> ir.GlobalVariable:
    """Retrieve or declare the static 64KB arena array in the module."""
    if "__netelpro_arena" in module.globals:
        return module.get_global("__netelpro_arena")
    if "arena_base" in module.globals:
        return module.get_global("arena_base")

    arena_arr_ty = ir.ArrayType(ir.IntType(8), ARENA_SIZE)
    arena_gv = ir.GlobalVariable(module, arena_arr_ty, name="__netelpro_arena")
    arena_gv.initializer = ir.Constant(arena_arr_ty, None)
    return arena_gv


def _get_or_emit_arena_head(module: ir.Module) -> ir.GlobalVariable:
    """Retrieve or declare the i64 bump-pointer offset global in the module."""
    if "arena_head" in module.globals:
        return module.get_global("arena_head")

    head_gv = ir.GlobalVariable(module, ir.IntType(64), name="arena_head")
    head_gv.initializer = ir.Constant(ir.IntType(64), 0)
    return head_gv


def _get_or_declare_memcpy(module: ir.Module) -> ir.Function:
    """Retrieve or declare the llvm.memcpy intrinsic."""
    name = "llvm.memcpy.p0.p0.i64"
    if name in module.globals:
        return module.get_global(name)

    ptr_ty = ir.PointerType()
    memcpy_ty = ir.FunctionType(
        ir.VoidType(),
        [ptr_ty, ptr_ty, ir.IntType(64), ir.IntType(1)],
    )
    return ir.Function(module, memcpy_ty, name=name)


def _get_or_declare_strlen(module: ir.Module) -> ir.Function:
    """Retrieve or declare the external strlen C runtime function."""
    name = "strlen"
    if name in module.globals:
        return module.get_global(name)

    ptr_ty = ir.PointerType()
    strlen_ty = ir.FunctionType(ir.IntType(64), [ptr_ty])
    return ir.Function(module, strlen_ty, name=name)


def emit_arena_globals(
    module: ir.Module,
    builder_factory_context: Any = None,
) -> tuple[ir.GlobalVariable, ir.GlobalVariable]:
    """Emit the static 64KB arena globals into an llvmlite ir.Module.

    Emits:
      - `@__netelpro_arena`: `[65536 x i8]` global array (zero-initialized).
      - `@arena_head`: `i64` global bump-pointer offset (initialized to 0).

    Args:
        module: llvmlite ir.Module into which globals are emitted.
        builder_factory_context: Optional contextual state passed by compiler orchestrator.

    Returns:
        Tuple of `(arena_base_gv, arena_head_gv)`.
    """
    arena_gv = _get_or_emit_arena(module)
    head_gv = _get_or_emit_arena_head(module)
    return arena_gv, head_gv


def emit_arena_reset(builder: ir.IRBuilder) -> None:
    """Reset the arena bump pointer by storing 0 to @arena_head.

    Args:
        builder: llvmlite IRBuilder positioned at the function/rule entry point.
    """
    head_gv = _get_or_emit_arena_head(builder.module)
    builder.store(ir.Constant(ir.IntType(64), 0), head_gv)


def emit_arena_alloc(
    builder: ir.IRBuilder,
    nbytes: int | ir.Value,
    align: int = 1,
) -> ir.Value:
    """Allocate `nbytes` from the static 64KB arena using aligned bump allocation.

    Loads `@arena_head`, adds `nbytes`, and verifies that `old_head + nbytes <= ARENA_SIZE`.
    If capacity is exceeded (> 65536), returns `i8*` null without advancing `@arena_head`.
    Otherwise, updates `@arena_head` (aligned if align > 1) and returns a pointer
    to the allocated memory via GEP into `@__netelpro_arena` at `old_head`.

    Args:
        builder: llvmlite IRBuilder for the current basic block.
        nbytes: Number of bytes to allocate (int or 64-bit ir.Value).
        align: Byte alignment boundary (default 1).

    Returns:
        An `i8*` (ptr) ir.Value pointing to the allocated arena buffer,
        or NULL pointer on arena overflow.
    """
    module = builder.module
    head_gv = _get_or_emit_arena_head(module)
    arena_gv = _get_or_emit_arena(module)

    i64 = ir.IntType(64)
    ptr_ty = ir.PointerType()

    if isinstance(nbytes, int):
        nbytes_val = ir.Constant(i64, nbytes)
    elif isinstance(nbytes, ir.Value):
        if getattr(nbytes.type, "width", 64) < 64:
            nbytes_val = builder.zext(nbytes, i64, name="nbytes_i64")
        else:
            nbytes_val = nbytes
    else:
        raise TypeError(f"nbytes must be int or ir.Value, got {type(nbytes).__name__}")

    old_head = builder.load(head_gv, name="arena_old_head")
    new_head = builder.add(old_head, nbytes_val, name="arena_new_head")

    if align > 1:
        bumped_head = builder.and_(
            builder.add(new_head, ir.Constant(i64, align - 1)),
            ir.Constant(i64, ~(align - 1)),
            name="arena_bumped_head",
        )
    else:
        bumped_head = new_head

    overflow = builder.icmp_unsigned(
        ">", new_head, ir.Constant(i64, ARENA_SIZE), name="arena_overflow"
    )

    with builder.if_else(overflow) as (then_overflow, otherwise_ok):
        with then_overflow:
            null_ptr = ir.Constant(ptr_ty, None)
            then_bb = builder.block
        with otherwise_ok:
            builder.store(bumped_head, head_gv)
            alloc_ptr = builder.gep(
                arena_gv,
                [ir.Constant(i64, 0), old_head],
                name="arena_alloc_ptr",
            )
            otherwise_bb = builder.block

    phi = builder.phi(ptr_ty, name="arena_result_ptr")
    phi.add_incoming(null_ptr, then_bb)
    phi.add_incoming(alloc_ptr, otherwise_bb)
    return phi


def emit_strlit(builder: ir.IRBuilder, py_str: str) -> ir.Value:
    """Allocate len+1 bytes in the arena and memcpy a NUL-terminated UTF-8 literal.

    Args:
        builder: llvmlite IRBuilder for the current basic block.
        py_str: Python string literal to embed and copy into the arena.

    Returns:
        An `i8*` (ptr) ir.Value pointing to the arena string,
        or NULL pointer if the arena capacity is exceeded.
    """
    module = builder.module
    encoded = py_str.encode("utf-8") + NUL_SENTINEL
    nbytes = len(encoded)

    str_data = bytearray(encoded)
    const_arr = ir.Constant(ir.ArrayType(ir.IntType(8), nbytes), str_data)
    gv_name = module.get_unique_name("__netelpro_strlit")
    const_gv = ir.GlobalVariable(module, const_arr.type, name=gv_name)
    const_gv.linkage = "internal"
    const_gv.global_constant = True
    const_gv.initializer = const_arr

    dst_ptr = emit_arena_alloc(builder, nbytes)

    memcpy_fn = _get_or_declare_memcpy(module)
    is_null = builder.icmp_unsigned("==", dst_ptr, ir.Constant(ir.PointerType(), None))
    with builder.if_else(is_null) as (then_null, otherwise_ok):
        with then_null:
            pass
        with otherwise_ok:
            src_ptr = builder.bitcast(const_gv, ir.PointerType())
            builder.call(
                memcpy_fn,
                [
                    dst_ptr,
                    src_ptr,
                    ir.Constant(ir.IntType(64), nbytes),
                    ir.Constant(ir.IntType(1), 0),
                ],
            )

    return dst_ptr


def emit_str_cat(
    builder: ir.IRBuilder,
    a_ptr: ir.Value | Sequence[ir.Value],
    b_ptr: ir.Value | ir.Module | None = None,
    module: ir.Module | None = None,
) -> ir.Value:
    """Concatenate two NUL-terminated strings into a newly allocated arena buffer.

    Reads lengths via external `strlen`, allocates `len(a) + len(b) + 1` bytes
    in the arena, and copies `a`, `b`, and a terminating NUL byte into the destination.
    If either input pointer is NULL or arena capacity is exceeded, returns a NULL pointer.

    Supports both signatures:
      - `emit_str_cat(builder, a_ptr, b_ptr, module)`
      - `emit_str_cat(builder, [a_ptr, b_ptr], module)`

    Args:
        builder: llvmlite IRBuilder for the current basic block.
        a_ptr: First string pointer (or sequence containing both [a, b]).
        b_ptr: Second string pointer (or module if a_ptr is a sequence).
        module: LLVM module for external declarations.

    Returns:
        An `i8*` (ptr) ir.Value pointing to the concatenated string,
        or NULL pointer on overflow or null input.
    """
    if isinstance(a_ptr, (list, tuple)):
        first_ptr = a_ptr[0]
        second_ptr = a_ptr[1]
        mod = b_ptr if isinstance(b_ptr, ir.Module) else (module or builder.module)
    else:
        first_ptr = a_ptr
        if not isinstance(b_ptr, ir.Value):
            raise TypeError("b_ptr must be an ir.Value when a_ptr is a single value")
        second_ptr = b_ptr
        mod = module if isinstance(module, ir.Module) else getattr(builder, "module", None)

    if mod is None:
        raise ValueError("module must be provided or builder must have builder.module")

    ptr_ty = ir.PointerType()
    null_const = ir.Constant(ptr_ty, None)

    # Null input guard
    a_is_null = builder.icmp_unsigned("==", first_ptr, null_const, name="str_cat_a_null")
    b_is_null = builder.icmp_unsigned("==", second_ptr, null_const, name="str_cat_b_null")
    either_null = builder.or_(a_is_null, b_is_null, name="str_cat_input_null")

    with builder.if_else(either_null) as (then_null, otherwise_ok):
        with then_null:
            null_res = ir.Constant(ptr_ty, None)
            bb_null = builder.block
        with otherwise_ok:
            strlen_fn = _get_or_declare_strlen(mod)
            memcpy_fn = _get_or_declare_memcpy(mod)
            i64 = ir.IntType(64)
            i8 = ir.IntType(8)
            i1 = ir.IntType(1)

            len_a = builder.call(strlen_fn, [first_ptr], name="str_cat_len_a")
            len_b = builder.call(strlen_fn, [second_ptr], name="str_cat_len_b")
            total_len = builder.add(
                builder.add(len_a, len_b), ir.Constant(i64, 1), name="str_cat_total_len"
            )

            dst_ptr = emit_arena_alloc(builder, total_len)
            is_alloc_null = builder.icmp_unsigned(
                "==", dst_ptr, null_const, name="str_cat_alloc_null"
            )

            with builder.if_else(is_alloc_null) as (then_alloc_null, otherwise_alloc_ok):
                with then_alloc_null:
                    pass
                with otherwise_alloc_ok:
                    builder.call(memcpy_fn, [dst_ptr, first_ptr, len_a, ir.Constant(i1, 0)])
                    dst_b = builder.gep(dst_ptr, [len_a], source_etype=i8, name="str_cat_dst_b")
                    builder.call(memcpy_fn, [dst_b, second_ptr, len_b, ir.Constant(i1, 0)])
                    dst_nul = builder.gep(
                        dst_ptr,
                        [builder.add(len_a, len_b)],
                        source_etype=i8,
                        name="str_cat_dst_nul",
                    )
                    builder.store(ir.Constant(i8, 0), dst_nul)

            bb_ok = builder.block

    phi_out = builder.phi(ptr_ty, name="str_cat_result")
    phi_out.add_incoming(null_res, bb_null)
    phi_out.add_incoming(dst_ptr, bb_ok)
    return phi_out


def emit_int_to_str(
    builder: ir.IRBuilder,
    n_i64: int | ir.Value | Sequence[int | ir.Value],
    module: ir.Module | None = None,
) -> ir.Value:
    """Convert a 64-bit integer to a decimal ASCII string in the arena.

    Allocates up to 21 bytes in the arena (1 sign + 19 digits + 1 NUL byte).
    Converts via a digit loop using unsigned division/remainder by 10 (`udiv`/`urem`).
    Handles negatives by extracting the absolute value; INT64_MIN (-9223372036854775808)
    is handled correctly via the two's complement unsigned trick (0 - INT64_MIN =
    0x8000000000000000, which in unsigned 64-bit arithmetic is +9223372036854775808).

    Supports both signatures:
      - `emit_int_to_str(builder, n_i64, module)`
      - `emit_int_to_str(builder, [n_i64], module)`

    Args:
        builder: llvmlite IRBuilder for the current basic block.
        n_i64: 64-bit integer operand (int, ir.Value, or 1-element sequence).
        module: LLVM module for external declarations.

    Returns:
        An `i8*` (ptr) ir.Value pointing to the converted ASCII string,
        or NULL pointer on arena overflow.
    """
    if isinstance(n_i64, (list, tuple)):
        raw_val = n_i64[0]
    else:
        raw_val = n_i64

    mod = module if isinstance(module, ir.Module) else getattr(builder, "module", None)
    if mod is None:
        raise ValueError("module must be provided or builder must have builder.module")

    i64 = ir.IntType(64)
    i8 = ir.IntType(8)
    i1 = ir.IntType(1)
    ptr_ty = ir.PointerType()

    if isinstance(raw_val, int):
        val_i64 = ir.Constant(i64, raw_val)
    elif isinstance(raw_val, ir.Value):
        if getattr(raw_val.type, "width", 64) < 64:
            val_i64 = builder.sext(raw_val, i64, name="i2s_sext")
        else:
            val_i64 = raw_val
    else:
        raise TypeError(f"n_i64 must be int or ir.Value, got {type(raw_val).__name__}")

    # Allocate stack scratchpad of 21 bytes
    stk = builder.alloca(ir.ArrayType(i8, 21), name="i2s_stk")
    # Pre-store NUL at index 20
    p_nul = builder.gep(stk, [ir.Constant(i64, 0), ir.Constant(i64, 20)], name="i2s_nul")
    builder.store(ir.Constant(i8, 0), p_nul)

    cursor_slot = builder.alloca(i64, name="i2s_cursor_slot")
    builder.store(ir.Constant(i64, 20), cursor_slot)

    fn = builder.function
    zero_bb = fn.append_basic_block("i2s_zero")
    nonzero_bb = fn.append_basic_block("i2s_nonzero")
    merge_bb = fn.append_basic_block("i2s_merge")

    is_zero = builder.icmp_signed("==", val_i64, ir.Constant(i64, 0), name="i2s_is_zero")
    builder.cbranch(is_zero, zero_bb, nonzero_bb)

    # Zero branch
    builder.position_at_end(zero_bb)
    builder.store(ir.Constant(i64, 19), cursor_slot)
    p_zero = builder.gep(stk, [ir.Constant(i64, 0), ir.Constant(i64, 19)], name="i2s_zero_char")
    builder.store(ir.Constant(i8, ord("0")), p_zero)
    builder.branch(merge_bb)

    # Nonzero branch
    builder.position_at_end(nonzero_bb)
    is_neg = builder.icmp_signed("<", val_i64, ir.Constant(i64, 0), name="i2s_is_neg")
    neg_val = builder.sub(ir.Constant(i64, 0), val_i64, name="i2s_neg_val")
    abs_val = builder.select(is_neg, neg_val, val_i64, name="i2s_abs_val")

    val_slot = builder.alloca(i64, name="i2s_val_slot")
    builder.store(abs_val, val_slot)

    loop_bb = fn.append_basic_block("i2s_loop")
    after_loop_bb = fn.append_basic_block("i2s_after_loop")

    builder.branch(loop_bb)

    # Digit extraction loop
    builder.position_at_end(loop_bb)
    curr_val = builder.load(val_slot, name="i2s_curr_val")
    curr_cursor = builder.load(cursor_slot, name="i2s_curr_cursor")
    new_cursor = builder.sub(curr_cursor, ir.Constant(i64, 1), name="i2s_new_cursor")
    builder.store(new_cursor, cursor_slot)

    rem = builder.urem(curr_val, ir.Constant(i64, 10), name="i2s_rem")
    digit_char = builder.trunc(
        builder.add(rem, ir.Constant(i64, ord("0"))), i8, name="i2s_digit_char"
    )
    p_digit = builder.gep(stk, [ir.Constant(i64, 0), new_cursor], name="i2s_p_digit")
    builder.store(digit_char, p_digit)

    next_val = builder.udiv(curr_val, ir.Constant(i64, 10), name="i2s_next_val")
    builder.store(next_val, val_slot)

    has_more = builder.icmp_unsigned(">", next_val, ir.Constant(i64, 0), name="i2s_has_more")
    builder.cbranch(has_more, loop_bb, after_loop_bb)

    # After loop: prepend '-' sign if negative
    builder.position_at_end(after_loop_bb)
    neg_bb = fn.append_basic_block("i2s_neg_sign")
    after_neg_bb = fn.append_basic_block("i2s_after_neg")

    builder.cbranch(is_neg, neg_bb, after_neg_bb)

    builder.position_at_end(neg_bb)
    c_neg = builder.load(cursor_slot, name="i2s_c_neg")
    c_neg_new = builder.sub(c_neg, ir.Constant(i64, 1), name="i2s_c_neg_new")
    builder.store(c_neg_new, cursor_slot)
    p_minus = builder.gep(stk, [ir.Constant(i64, 0), c_neg_new], name="i2s_p_minus")
    builder.store(ir.Constant(i8, ord("-")), p_minus)
    builder.branch(after_neg_bb)

    builder.position_at_end(after_neg_bb)
    builder.branch(merge_bb)

    # Merge: allocate in arena and copy
    builder.position_at_end(merge_bb)
    final_cursor = builder.load(cursor_slot, name="i2s_final_cursor")
    str_len = builder.sub(ir.Constant(i64, 21), final_cursor, name="i2s_str_len")

    dst_ptr = emit_arena_alloc(builder, str_len)

    null_const = ir.Constant(ptr_ty, None)
    is_null = builder.icmp_unsigned("==", dst_ptr, null_const, name="i2s_alloc_null")

    with builder.if_else(is_null) as (then_alloc_null, otherwise_alloc_ok):
        with then_alloc_null:
            pass
        with otherwise_alloc_ok:
            memcpy_fn = _get_or_declare_memcpy(mod)
            src_ptr = builder.gep(stk, [ir.Constant(i64, 0), final_cursor], name="i2s_src_ptr")
            builder.call(memcpy_fn, [dst_ptr, src_ptr, str_len, ir.Constant(i1, 0)])

    return dst_ptr


__all__ = [
    "ARENA_SIZE",
    "NUL_SENTINEL",
    "StrNativeError",
    "consume_arena",
    "emit_arena_alloc",
    "emit_arena_globals",
    "emit_arena_reset",
    "emit_int_to_str",
    "emit_str_cat",
    "emit_strlit",
    "interp_int_to_str",
    "interp_str_cat",
    "read_arena_string",
]
