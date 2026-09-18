"""CLI entrypoint: ``python -m netelpro.lib``.

Delegates to ``netelpro.lib.contracts.main``. This module exists so the
documented invocation does not hit the runpy warning that
``python -m netelpro.lib.contracts`` produces (the package ``__init__``
imports ``contracts`` eagerly, so runpy finds it already in sys.modules).
"""
from __future__ import annotations

from netelpro.lib.contracts import main

if __name__ == "__main__":
    raise SystemExit(main())
