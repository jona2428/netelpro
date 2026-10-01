"""Vector Hiperdimensional Binario (Zero-Float BitVector).

Representa vectores densos de bits de alta dimensión (por defecto 512 o 2048 bits)
empaquetados en enteros sin signo de 64 bits (uint64).

Todas las operaciones (XOR de enlace, mayorías, distancias de Hamming) se ejecutan
a nivel de instrucción de CPU (POPCNT, bitwise AND/OR/XOR/NOT) sin utilizar tensores,
matrices continuas ni operaciones de coma flotante.
"""

from __future__ import annotations

import array
import hashlib
from typing import Sequence


class BitVector:
    """Vector binario hiperdimensional de dimensión fija empaquetado en uint64."""

    __slots__ = ("num_bits", "num_words", "words")

    def __init__(self, num_bits: int = 512, words: array.array | None = None) -> None:
        if num_bits % 64 != 0:
            raise ValueError(f"num_bits debe ser múltiplo de 64 (recibido {num_bits})")
        self.num_bits: int = num_bits
        self.num_words: int = num_bits // 64

        if words is not None:
            if len(words) != self.num_words:
                raise ValueError(
                    f"Longitud de palabras incorrecta: esperada {self.num_words}, recibida {len(words)}"
                )
            self.words: array.array = words
        else:
            self.words = array.array("Q", [0] * self.num_words)

    @classmethod
    def random(cls, num_bits: int = 512, seed: int | None = None) -> BitVector:
        """Crea un vector pseudo-aleatorio balanceado usando hash determinista."""
        import random

        rng = random.Random(seed)
        words = array.array("Q", [rng.getrandbits(64) for _ in range(num_bits // 64)])
        return cls(num_bits=num_bits, words=words)

    @classmethod
    def from_text(cls, text: str, num_bits: int = 512) -> BitVector:
        """Genera una huella binaria densa (SimHash discreto) a partir de texto.

        Cero floats: utiliza contadores enteros puros sobre n-gramas de palabras y caracteres.
        """
        num_words = num_bits // 64
        counters = [0] * num_bits

        tokens = text.lower().strip().split()
        if not tokens:
            return cls(num_bits=num_bits)

        # Generar características relevantes
        features: list[str] = list(tokens)
        # Añadir bigramas de palabras para contexto
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]} {tokens[i+1]}")
        # Añadir trigramas de caracteres en palabras significativas
        for t in tokens:
            if len(t) >= 4:
                for i in range(min(len(t) - 2, 4)):
                    features.append(t[i : i + 3])

        # Tamaño de digest necesario en bytes (mínimo 64 bytes para blake2b)
        digest_bytes = min(num_words * 8, 64)

        for feat in features:
            raw = feat.encode("utf-8")
            h = hashlib.blake2b(raw, digest_size=digest_bytes).digest()
            for w_idx in range(num_words):
                byte_start = (w_idx * 8) % digest_bytes
                word_val = int.from_bytes(h[byte_start : byte_start + 8], "little")
                base_bit = w_idx * 64
                for bit_offset in range(64):
                    if (word_val >> bit_offset) & 1:
                        counters[base_bit + bit_offset] += 1
                    else:
                        counters[base_bit + bit_offset] -= 1

        words = array.array("Q", [0] * num_words)
        for word_idx in range(num_words):
            word_val = 0
            base_bit = word_idx * 64
            for bit_offset in range(64):
                if counters[base_bit + bit_offset] >= 0:
                    word_val |= 1 << bit_offset
            words[word_idx] = word_val

        return cls(num_bits=num_bits, words=words)

    def hamming_distance(self, other: BitVector) -> int:
        """Calcula la distancia de Hamming exacta mediante POPCNT nativo."""
        if self.num_bits != other.num_bits:
            raise ValueError("Los vectores deben tener la misma dimensionalidad")

        dist = 0
        w1 = self.words
        w2 = other.words
        for i in range(self.num_words):
            dist += (w1[i] ^ w2[i]).bit_count()
        return dist

    def similarity_percent(self, other: BitVector) -> int:
        """Porcentaje de similitud discreto (0 a 100) sin números flotantes."""
        dist = self.hamming_distance(other)
        return ((self.num_bits - dist) * 100) // self.num_bits

    def __xor__(self, other: BitVector) -> BitVector:
        """Operación de enlace asociativo (Binding / Unbinding) mediante XOR bit a bit."""
        if self.num_bits != other.num_bits:
            raise ValueError("Dimensiones incompatibles")
        w_out = array.array(
            "Q", [a ^ b for a, b in zip(self.words, other.words, strict=True)]
        )
        return BitVector(num_bits=self.num_bits, words=w_out)

    def __and__(self, other: BitVector) -> BitVector:
        """Intersección booleana bit a bit."""
        if self.num_bits != other.num_bits:
            raise ValueError("Dimensiones incompatibles")
        w_out = array.array(
            "Q", [a & b for a, b in zip(self.words, other.words, strict=True)]
        )
        return BitVector(num_bits=self.num_bits, words=w_out)

    def __or__(self, other: BitVector) -> BitVector:
        """Unión booleana bit a bit."""
        if self.num_bits != other.num_bits:
            raise ValueError("Dimensiones incompatibles")
        w_out = array.array(
            "Q", [a | b for a, b in zip(self.words, other.words, strict=True)]
        )
        return BitVector(num_bits=self.num_bits, words=w_out)

    def __invert__(self) -> BitVector:
        """Negación booleana bit a bit (NOT / ~)."""
        mask_64 = 0xFFFFFFFFFFFFFFFF
        w_out = array.array("Q", [a ^ mask_64 for a in self.words])
        return BitVector(num_bits=self.num_bits, words=w_out)

    @classmethod
    def bundle(cls, vectors: Sequence[BitVector]) -> BitVector:
        """Agrega múltiples vectores por voto de mayoría discreta (Superposición / Bundle)."""
        if not vectors:
            raise ValueError("Lista de vectores vacía")
        if len(vectors) == 1:
            return vectors[0]

        num_bits = vectors[0].num_bits
        threshold = len(vectors) // 2
        counters = [0] * num_bits

        for vec in vectors:
            for word_idx in range(vec.num_words):
                w = vec.words[word_idx]
                base_bit = word_idx * 64
                for bit_offset in range(64):
                    if (w >> bit_offset) & 1:
                        counters[base_bit + bit_offset] += 1

        num_words = num_bits // 64
        words = array.array("Q", [0] * num_words)
        for word_idx in range(num_words):
            word_val = 0
            base_bit = word_idx * 64
            for bit_offset in range(64):
                if counters[base_bit + bit_offset] > threshold:
                    word_val |= 1 << bit_offset
            words[word_idx] = word_val

        return cls(num_bits=num_bits, words=words)

    def to_bytes(self) -> bytes:
        """Serializa a bytes binarios crudos."""
        return self.words.tobytes()

    @classmethod
    def from_bytes(cls, raw: bytes, num_bits: int = 512) -> BitVector:
        """Carga desde bytes binarios crudos."""
        words = array.array("Q")
        words.frombytes(raw)
        return cls(num_bits=num_bits, words=words)
