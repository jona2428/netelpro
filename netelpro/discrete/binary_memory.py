"""Banco de Memoria Asociativa Binaria de Netelpro (Zero-Float Memory Bank).

Implementa memoria asociativa y recuperación tipo RAG basada exclusivamente
en álgebra booleana, vectores hiperdimensionales discretos y distancia de Hamming.

- Cero tensores de PyTorch.
- Cero números flotantes en almacenamiento o comparación.
- Cero GPUs: recuperación asociativa en microsegundos en CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

from netelpro.discrete.bit_vector import BitVector


@dataclass
class MemoryRecord:
    """Registro individual de memoria asociativa binaria."""

    concept: str
    content: str
    vector: BitVector
    contract: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class MemoryMatch:
    """Resultado de una consulta a la memoria binaria."""

    record: MemoryRecord
    hamming_distance: int
    similarity_percent: int


class BinaryMemoryBank:
    """Almacén asociativo de hechos, contratos y conocimiento en silicio binario."""

    def __init__(self, num_bits: int = 512) -> None:
        self.num_bits: int = num_bits
        self.records: list[MemoryRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def store(
        self,
        concept: str,
        content: str,
        contract: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Almacena o actualiza un hecho en la memoria asociativa.

        Retorna el índice del registro.
        """
        vec = BitVector.from_text(concept, num_bits=self.num_bits)
        meta = metadata or {}

        # Si ya existe un concepto idéntico o con similitud > 95%, actualizar
        best_idx = -1
        max_sim = -1
        for idx, rec in enumerate(self.records):
            sim = rec.vector.similarity_percent(vec)
            if sim > max_sim:
                max_sim = sim
                best_idx = idx

        if best_idx >= 0 and max_sim >= 95:
            # Superposición / actualización de memoria existente
            existing = self.records[best_idx]
            existing.content = content
            if contract:
                existing.contract = contract
            existing.metadata.update(meta)
            existing.access_count += 1
            existing.vector = BitVector.bundle([existing.vector, vec])
            return best_idx

        # Nuevo registro
        rec = MemoryRecord(
            concept=concept,
            content=content,
            vector=vec,
            contract=contract,
            metadata=meta,
        )
        self.records.append(rec)
        return len(self.records) - 1

    def recall(
        self,
        query: str | BitVector,
        top_k: int = 3,
        min_similarity_percent: int = 40,
    ) -> list[MemoryMatch]:
        """Recupera los recuerdos más relevantes usando distancia de Hamming directa."""
        if not self.records:
            return []

        if isinstance(query, BitVector):
            q_vec = query
        else:
            q_vec = BitVector.from_text(query, num_bits=self.num_bits)

        matches: list[MemoryMatch] = []

        for rec in self.records:
            dist = rec.vector.hamming_distance(q_vec)
            sim = ((self.num_bits - dist) * 100) // self.num_bits
            if sim >= min_similarity_percent:
                rec.access_count += 1
                matches.append(
                    MemoryMatch(
                        record=rec,
                        hamming_distance=dist,
                        similarity_percent=sim,
                    )
                )

        matches.sort(key=lambda m: m.hamming_distance)
        return matches[:top_k]

    def save_to_file(self, file_path: str | Path) -> None:
        """Guarda la memoria en disco en formato binario compacto."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_bits": self.num_bits,
            "records": [
                {
                    "concept": r.concept,
                    "content": r.content,
                    "contract": r.contract,
                    "metadata": r.metadata,
                    "access_count": r.access_count,
                    "created_at": r.created_at,
                    "words": list(r.vector.words),
                }
                for r in self.records
            ],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> BinaryMemoryBank:
        """Carga el banco de memoria desde disco."""
        import array

        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        bank = cls(num_bits=data.get("num_bits", 512))
        for item in data.get("records", []):
            words = array.array("Q", item["words"])
            vec = BitVector(num_bits=bank.num_bits, words=words)
            bank.records.append(
                MemoryRecord(
                    concept=item["concept"],
                    content=item["content"],
                    vector=vec,
                    contract=item.get("contract"),
                    metadata=item.get("metadata", {}),
                    access_count=item.get("access_count", 0),
                    created_at=item.get("created_at", 0.0),
                )
            )
        return bank
