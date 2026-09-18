"""Netelpro standard library.

``prelude.sl`` holds the shared Bool-dialect helpers; ``contracts`` renders
canonical truth-table contracts. See ``contracts`` for why the duplication in
the host contracts cannot be factored by a helper and must be generated.
"""
from __future__ import annotations

from netelpro.lib.contracts import (
    GENERATED_MARKER,
    PRELUDE_PATH,
    ContractSpec,
    LibError,
    Slot,
    concat_with_prelude,
    contract_from_source,
    formal_block_start,
    migrate_source,
    prelude_source,
    render_contract,
    render_truth_table,
)

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
