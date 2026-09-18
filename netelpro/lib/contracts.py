"""Netelpro standard library: contract generation and the shared prelude.

Two pieces, both driven by measured evidence rather than speculation:

``prelude.sl``
    Bool-dialect helpers (``all-ofN`` / ``any-zeroN``) for ``defn`` bodies.
    Netelpro has no ``include`` (SPEC section 8 keeps modules out of scope),
    so the host CONCATENATES the prelude before the rule source. Verified
    against the real compiler: concatenation preserves native/interpreter
    parity, a name collision with the contract is a hard error rather than a
    silent shadow, and an impure helper the rule never calls leaves gate
    purity GREEN.

``render_contract``
    Renders a canonical truth-table contract: N declared slots, one
    zero-rejection row per slot, an all-ones admit row, and the mandatory
    all-wildcard backstop.

Why the second one exists
-------------------------
Twenty-three of the live ``.sl`` contracts in the host are the SAME shape
written by hand: "every flag is 1, or reject". Thirteen of them are the
identical ``((1 1 1) -> 1)`` + zero-rows + backstop block. A prelude helper
CANNOT factor those out: truth-table exhaustiveness is decided syntactically
over row PATTERNS, so a wildcard row that calls a helper covers nothing, and
a Bool-returning helper is a type mismatch inside an ``Int 0 1`` row slot.
The duplication lives exactly where a helper cannot enter. Generation is the
mechanism that actually removes it -- the same mechanism this project already
uses in ``builders/truth_table_builder.py`` and ``zone_rule_generator.py``:
one source, derived artifacts.

This module is the *capability*. It does not rewrite the existing contracts:
that is a deliberate migration step, and it changes what the SHA256
provenance digest in ``epistemic_edge`` certifies (contract-only today,
generator+spec afterwards).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "GENERATED_MARKER",
    "PRELUDE_PATH",
    "ContractSpec",
    "LibError",
    "Slot",
    "concat_with_prelude",
    "contract_from_source",
    "formal_block_start",
    "migrate_source",
    "prelude_source",
    "render_contract",
    "render_truth_table",
]

PRELUDE_PATH = Path(__file__).with_name("prelude.sl")

# A netelpro symbol: lowercase, digits, hyphens. Deliberately strict -- a
# slot name that would need escaping is a LibError, never a silent mangle.
_SYMBOL_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


class LibError(Exception):
    """A contract cannot be rendered from the given spec."""


@dataclass(frozen=True)
class Slot:
    """One declared truth-table parameter.

    ``name`` is the netelpro symbol (kebab-case). ``meaning`` is the human
    description rendered into the contract header -- write it the way the
    hand-written contracts do, e.g. "1 si el comando es no-vacio; 0 si no".
    """

    name: str
    meaning: str = ""

    def __post_init__(self) -> None:
        if not _SYMBOL_RE.match(self.name):
            raise LibError(
                f"slot name {self.name!r} is not a valid netelpro symbol "
                "(lowercase letters, digits and hyphens; must start with a letter)"
            )


@dataclass(frozen=True)
class ContractSpec:
    """A canonical all-of-N contract: every slot must be 1, or reject.

    ``admit`` is the verdict when every slot is 1; ``reject`` is the verdict
    for any input with at least one slot at 0, and for the backstop row.
    Defaults model the dominant host pattern (0/1 flags, admit = 1).
    """

    title: str
    slots: tuple[Slot, ...]
    rule_name: str = "filter-rule"
    admit: int = 1
    reject: int = 0
    provenance: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.slots) < 2:
            raise LibError(
                f"an all-of-N contract needs at least 2 slots, got {len(self.slots)}"
            )
        if not _SYMBOL_RE.match(self.rule_name):
            raise LibError(f"rule name {self.rule_name!r} is not a valid netelpro symbol")
        seen: set[str] = set()
        for slot in self.slots:
            if slot.name in seen:
                raise LibError(f"duplicate slot name {slot.name!r}")
            seen.add(slot.name)
        # ParamType is (Int 0 1) here, so the declared product is 2**N. The
        # host's exhaustiveness prosecutor caps the product at 256.
        if 2 ** len(self.slots) > 256:
            raise LibError(
                f"declared product {2 ** len(self.slots)} for {len(self.slots)} "
                "slots exceeds the prosecutor cap of 256"
            )


def prelude_source() -> str:
    """Read the standard prelude exactly as the host will concatenate it."""
    return PRELUDE_PATH.read_text(encoding="utf-8")


def concat_with_prelude(rule_source: str) -> str:
    """Prelude + rule, in the order the compiler needs.

    The prelude comes FIRST so that a contract defining a name the prelude
    already defines is a hard compile error (`duplicate defn`), not a silent
    override. Fail-closed by construction.
    """
    return prelude_source() + "\n" + rule_source


def render_truth_table(
    rule_name: str,
    slots: tuple[Slot, ...],
    rows: tuple[tuple[tuple[int | None, ...], int], ...],
    default: int = 0,
) -> str:
    """Render an explicit truth-table block. General tool under render_contract.

    ``rows`` is ``((slots_tuple, verdict), ...)`` where a slot is an int or
    ``None`` for the wildcard ``_``. The all-wildcard ``default`` row is
    appended last: it is mandatory in the language and is the structural
    backstop, not part of coverage.
    """
    arity = len(slots)
    lines = [f"(truth-table {rule_name}"]
    for slot in slots:
        lines.append(f"  ({slot.name} : (Int 0 1))")
    for slots_row, verdict in rows:
        if len(slots_row) != arity:
            raise LibError(
                f"row {slots_row!r} has {len(slots_row)} slots, table declares {arity}"
            )
        pattern = " ".join("_" if s is None else str(int(s)) for s in slots_row)
        lines.append(f"  (({pattern}) -> {int(verdict)})")
    lines.append(f"  (({' '.join('_' for _ in range(arity))}) -> {int(default)}))")
    return "\n".join(lines) + "\n"


def _all_of_rows(spec: ContractSpec) -> tuple[tuple[tuple[int | None, ...], int], ...]:
    """Zero-rejection row per slot, then the all-ones admit row."""
    arity = len(spec.slots)
    rows: list[tuple[tuple[int | None, ...], int]] = []
    for i in range(arity):
        row = tuple(None if j != i else 0 for j in range(arity))
        rows.append((row, spec.reject))
    rows.append((tuple(1 for _ in range(arity)), spec.admit))
    return tuple(rows)


def _canonical_patterns(arity: int) -> tuple[tuple[int | None, ...], ...]:
    """The slot patterns of a canonical contract, without any verdicts.

    Structural only, so a comparison against a real contract never depends on
    which verdicts that contract happens to use.
    """
    patterns = [tuple(None if j != i else 0 for j in range(arity)) for i in range(arity)]
    patterns.append(tuple(1 for _ in range(arity)))
    return tuple(patterns)


def _header(spec: ContractSpec) -> str:
    arity = len(spec.slots)
    product = 2 ** arity
    lines = [
        f"; {spec.title}",
        "; Fuente unica de verdad compilada con LLVM a codigo maquina nativo.",
        ";",
        "; Parametros en frontera FFI (Int 0 1):",
    ]
    width = max(len(s.name) for s in spec.slots)
    for slot in spec.slots:
        lines.append(f";   {slot.name.ljust(width)}  {slot.meaning or '(sin descripcion)'}")
    lines += [
        ";",
        "; Veredicto:",
        f";   {spec.admit} = ADMIT  (todas las condiciones conformes; procede la operacion)",
        f";   {spec.reject} = REJECT (violacion de contrato; abortar antes de tocar I/O)",
        ";",
        f"; Cobertura formal exhaustiva: {arity} slots -> {product} combinaciones del",
        "; producto declarado, bajo el tope de 256 del prosecutor. Las filas de rechazo",
        "; por slot mas la fila all-ones cubren el producto completo por si solas;",
        "; la fila all-wildcard final es backstop estructural obligatorio e inalcanzable.",
    ]
    for note in spec.notes:
        lines.append(f"; {note}")
    if spec.provenance:
        lines.append(f"; {spec.provenance}")
    return "\n".join(lines) + "\n"


def render_contract(spec: ContractSpec) -> str:
    """Render a complete, compilable ``.sl`` contract from a spec."""
    body = render_truth_table(
        spec.rule_name, spec.slots, _all_of_rows(spec), default=spec.reject
    )
    return _header(spec) + body


def contract_from_source(source: str) -> ContractSpec:
    """Extract a ContractSpec from an existing ``.sl`` truth-table contract.

    Uses the real parser and the real AST, not a regex. Only contracts whose
    row block is exactly the canonical all-of-N shape are accepted; anything
    else raises LibError, so this can never silently "normalize" a contract
    whose semantics differ.
    """
    from netelpro.ast_nodes import Defn, IntLit
    from netelpro.parser import parse

    result = parse(source)
    if not result.ok:
        raise LibError(f"source does not parse: {result.errors[0].message}")

    defn = None
    for form in result.program.forms:
        if isinstance(form, Defn) and form.name.name == "filter-rule":
            defn = form
            break
    if defn is None or defn.truth_table is None:
        raise LibError("no truth-table definition of 'filter-rule' in source")

    spec = defn.truth_table
    slots = tuple(Slot(name) for name, _ in spec.params)
    arity = len(slots)

    # The AST keeps the mandatory all-wildcard backstop OUT of `rows`: it is
    # `default_expr`. So `spec.rows` is exactly the data rows, in order.
    #
    # Every row body must be a plain Int literal. Anything else (a computed
    # expression, a helper call) is not a canonical contract and is refused
    # rather than guessed at.
    data_rows: list[tuple[tuple[int | None, ...], int]] = []
    for slots_row, expr in spec.rows:
        if not isinstance(expr, IntLit):
            raise LibError(
                f"row {slots_row!r} body is {type(expr).__name__}, not an Int "
                "literal; refusing to re-derive a contract with computed rows"
            )
        data_rows.append((tuple(slots_row), int(expr.value)))
    default_expr = spec.default_expr
    if not isinstance(default_expr, IntLit):
        raise LibError(
            f"backstop body is {type(default_expr).__name__}, not an Int literal"
        )
    default = int(default_expr.value)
    rows_tuple = tuple(data_rows)

    expected = _canonical_patterns(arity)
    actual = tuple(slots_row for slots_row, _ in rows_tuple)
    if actual != expected:
        raise LibError(
            "contract is not the canonical all-of-N shape; refusing to re-derive it "
            f"(row patterns were {actual!r})"
        )
    # Verdicts are read back from the real rows, so a contract using something
    # other than the 0/1 defaults round-trips with its own verdicts intact.
    return ContractSpec(
        title="",
        slots=slots,
        admit=rows_tuple[-1][1],
        reject=default,
    )


# ---------------------------------------------------------------------------
# Migration: normalise the FORMAL BLOCK of an existing contract in place.
# ---------------------------------------------------------------------------

# Inserted immediately above the formal block. The prefix is the detection
# key: it must stay stable so the marker can be recognised and replaced
# instead of duplicated.
GENERATED_MARKER = (
    "; BLOQUE FORMAL GENERADO por netelpro.lib.contracts -- no editar a mano; "
    "scripts/migrate_contracts.py --check vigila la deriva."
)
_MARKER_PREFIX = "; BLOQUE FORMAL GENERADO por netelpro.lib.contracts"


def formal_block_start(source: str) -> int:
    """Character offset where the formal block begins (the ``(truth-table`` line).

    Everything before it -- the prose header with the parameter meanings, the
    verdict legend and the coverage notes -- is preserved verbatim by the
    migration. Only the block below it is regenerated.
    """
    offset = 0
    for line in source.splitlines(keepends=True):
        if line.lstrip().startswith("(truth-table"):
            return offset
        offset += len(line)
    raise LibError("no '(truth-table' line found; nothing to migrate")


def _strip_generated_marker(header: str) -> str:
    """Drop a previously inserted marker so migration is idempotent."""
    kept = [line for line in header.splitlines(keepends=True) if not line.startswith(_MARKER_PREFIX)]
    return "".join(kept)


def migrate_source(source: str) -> str:
    """Regenerate the formal block of a canonical contract, header untouched.

    The contract file stays the source of truth for *meaning* (slot names live
    in the block, their descriptions in the prose header). What this
    normalises is the *shape*: parameter declarations, one zero-rejection row
    per slot, the all-ones admit row, and the mandatory all-wildcard backstop,
    in that order.

    Raises LibError if the contract is not the canonical all-of-N shape, so a
    contract with bespoke policy rows can never be silently reshaped.

    Idempotent: running it on its own output returns identical text.
    """
    spec = contract_from_source(source)
    start = formal_block_start(source)
    header = _strip_generated_marker(source[:start])
    if header and not header.endswith("\n"):
        header += "\n"
    block = render_truth_table(
        spec.rule_name, spec.slots, _all_of_rows(spec), default=spec.reject
    )
    return f"{header}{GENERATED_MARKER}\n{block}"


def main(argv: list[str] | None = None) -> int:
    """CLI: render a canonical contract, or dump the prelude.

    ``python -m netelpro.lib.contracts --prelude``
    ``python -m netelpro.lib.contracts --slots a,b,c --title "..."``
    ``python -m netelpro.lib.contracts --from path/to/contract.sl``
    """
    parser = argparse.ArgumentParser(prog="netelpro.lib.contracts")
    parser.add_argument("--prelude", action="store_true", help="print prelude.sl")
    parser.add_argument("--slots", help="comma-separated slot names")
    parser.add_argument("--title", default="Contrato formal Netelpro.")
    parser.add_argument("--admit", type=int, default=1)
    parser.add_argument("--reject", type=int, default=0)
    parser.add_argument("--from", dest="from_path", help="re-render an existing contract")
    args = parser.parse_args(argv)

    try:
        if args.prelude:
            sys.stdout.write(prelude_source())
            return 0
        if args.from_path:
            source = Path(args.from_path).read_text(encoding="utf-8")
            spec = contract_from_source(source)
            spec = ContractSpec(
                title=args.title, slots=spec.slots, admit=spec.admit, reject=spec.reject
            )
            sys.stdout.write(render_contract(spec))
            return 0
        if not args.slots:
            parser.error("one of --prelude, --slots or --from is required")
        slots = tuple(Slot(name.strip()) for name in args.slots.split(",") if name.strip())
        spec = ContractSpec(
            title=args.title, slots=slots, admit=args.admit, reject=args.reject
        )
        sys.stdout.write(render_contract(spec))
        return 0
    except LibError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
