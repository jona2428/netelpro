"""Compatibility shim: the gate demo now ships inside the package.

Run it with either command -- both work from a checkout:

    python examples/gate_demo.py
    python -m netelpro.gate_demo

After `pip install netelpro`, use the module form; it needs no checkout.
"""

import sys

from netelpro.gate_demo import main

if __name__ == "__main__":
    sys.exit(main())