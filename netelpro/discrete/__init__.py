"""Netelpro Discrete -- Motor de Memoria Binaria Asociativa y Síntesis Lógica (Zero-Float).

Proporciona computación simbólica e hiperdimensional sin números flotantes:
- BitVector: Vectores binarios hiperdimensionales optimizados en CPU.
- BinaryMemoryBank: Memoria asociativa de microsegundos (Zero-Float RAG).
- DiscreteLogicSynthesizer: Inductor y compilador de contratos formales LLVM sin gradientes.
- DiscreteArticulator: Capa conversacional humana para evitar respuestas robóticas.
"""

from netelpro.discrete.bit_vector import BitVector
from netelpro.discrete.binary_memory import BinaryMemoryBank, MemoryMatch, MemoryRecord
from netelpro.discrete.logic_synthesizer import DiscreteLogicSynthesizer
from netelpro.discrete.articulator import DiscreteArticulator

__all__ = [
    "BitVector",
    "BinaryMemoryBank",
    "MemoryRecord",
    "MemoryMatch",
    "DiscreteLogicSynthesizer",
    "DiscreteArticulator",
]
