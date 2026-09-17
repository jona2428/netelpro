"""Banco de Memoria Binaria Escribible y Persistente de Netelpro (Writeable Neural Memory Bank).

Permite al Netelpro Mini/Mega LLM escribir y recuperar recuerdos continuos directamente
en un búfer binario persistente (.bin) sin requerir reentrenamiento de los pesos base.
Las ranuras de memoria se proyectan como vectores latentes en el espacio de atención.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn.functional as F


class NetelproBinaryMemoryBank:
    """Banco de ranuras de memoria neuronal escribible en silicio binario."""

    def __init__(
        self,
        num_slots: int = 128,
        dim: int = 256,
        memory_file: str | Path | None = None,
        decay_factor: float = 0.99,
    ) -> None:
        self.num_slots = num_slots
        self.dim = dim
        self.memory_file = Path(memory_file) if memory_file else None
        self.decay_factor = decay_factor

        if HAS_TORCH:
            # Tensor de ranuras en memoria (Slots x Dim)
            self.slots = torch.zeros((num_slots, dim), dtype=torch.float32)
            self.slot_masks = torch.zeros(num_slots, dtype=torch.bool)
        else:
            self.slots = None
            self.slot_masks = None

        # Metadatos textuales e historial de auditoría
        self.slot_metadata: list[dict[str, Any]] = [{} for _ in range(num_slots)]

        if self.memory_file and self.memory_file.exists():
            self.load()

    def write_memory(
        self,
        concept: str,
        vector: torch.Tensor,
        confidence: float = 1.0,
    ) -> int:
        """Escribe un vector de memoria en la ranura óptima y persiste en binario."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch es requerido para el banco de memoria.")

        # Normalizar vector de entrada
        vec = vector.detach().view(-1).float()
        if vec.numel() != self.dim:
            if vec.numel() < self.dim:
                vec = F.pad(vec, (0, self.dim - vec.numel()))
            else:
                vec = vec[: self.dim]
        vec = F.normalize(vec, p=2, dim=0)

        # Buscar si ya existe una ranura muy similar (actualización asociativa)
        best_slot = -1
        max_sim = -1.0

        for i in range(self.num_slots):
            if self.slot_masks[i]:
                sim = float(torch.dot(vec, self.slots[i]).item())
                if sim > max_sim:
                    max_sim = sim
                    best_slot = i

        # Si similitud > 0.85, actualizar ranura existente con combinación convexa
        if best_slot >= 0 and max_sim > 0.85:
            target_slot = best_slot
            self.slots[target_slot] = F.normalize(
                0.6 * self.slots[target_slot] + 0.4 * vec, p=2, dim=0
            )
        else:
            # Buscar primera ranura vacía
            empty_slots = [i for i in range(self.num_slots) if not self.slot_masks[i]]
            if empty_slots:
                target_slot = empty_slots[0]
            else:
                # Si todas están ocupadas, reemplazar la de menor acceso (LRU)
                oldest_idx = min(
                    range(self.num_slots),
                    key=lambda idx: self.slot_metadata[idx].get("last_accessed", 0),
                )
                target_slot = oldest_idx

            self.slots[target_slot] = vec
            self.slot_masks[target_slot] = True

        now = time.time()
        self.slot_metadata[target_slot] = {
            "slot": target_slot,
            "concept": concept,
            "created_at": now,
            "last_accessed": now,
            "access_count": self.slot_metadata[target_slot].get("access_count", 0) + 1,
            "confidence": confidence,
        }

        # Guardar inmediatamente en disco si hay archivo configurado
        if self.memory_file:
            self.save()

        return target_slot

    def retrieve_relevant_memories(
        self,
        query_vector: torch.Tensor,
        top_k: int = 3,
        threshold: float = 0.3,
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """Recupera las memorias binarias más relevantes por similitud de coseno."""
        if not HAS_TORCH:
            return []

        active_indices = [i for i in range(self.num_slots) if self.slot_masks[i]]
        if not active_indices:
            return []

        q = query_vector.detach().view(-1).float()
        if q.numel() != self.dim:
            if q.numel() < self.dim:
                q = F.pad(q, (0, self.dim - q.numel()))
            else:
                q = q[: self.dim]
        q = F.normalize(q, p=2, dim=0)

        results = []
        for idx in active_indices:
            sim = float(torch.dot(q, self.slots[idx]).item())
            if sim >= threshold:
                results.append((idx, sim, self.slot_metadata[idx]))
                # Actualizar estadísticas de acceso
                self.slot_metadata[idx]["last_accessed"] = time.time()
                self.slot_metadata[idx]["access_count"] = (
                    self.slot_metadata[idx].get("access_count", 0) + 1
                )

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_memory_prefix_tensor(self, top_k: int = 4) -> torch.Tensor | None:
        """Devuelve un tensor de prefijo latente con las memorias más activas para inyectar en atención."""
        if not HAS_TORCH:
            return None

        active_indices = [i for i in range(self.num_slots) if self.slot_masks[i]]
        if not active_indices:
            return None

        # Ordenar por frecuencia y recencia
        active_indices.sort(
            key=lambda i: self.slot_metadata[i].get("last_accessed", 0), reverse=True
        )
        selected = active_indices[:top_k]
        return self.slots[selected].clone()

    def list_memories(self) -> list[dict[str, Any]]:
        """Lista todos los recuerdos activos."""
        active = []
        for i in range(self.num_slots):
            if self.slot_masks[i]:
                active.append(dict(self.slot_metadata[i]))
        return active

    def clear(self) -> None:
        """Borra la memoria viva en silicio."""
        if HAS_TORCH:
            self.slots.zero_()
            self.slot_masks.zero_()
        self.slot_metadata = [{} for _ in range(self.num_slots)]
        if self.memory_file and self.memory_file.exists():
            self.save()

    def save(self, filepath: str | Path | None = None) -> Path:
        """Serializa las ranuras de memoria a binario puro y JSON de metadatos."""
        out = Path(filepath) if filepath else self.memory_file
        if out is None:
            raise ValueError("No se especificó la ruta de archivo para guardar la memoria.")
        out.parent.mkdir(parents=True, exist_ok=True)

        meta_file = out.with_suffix(".json")
        data = {
            "num_slots": self.num_slots,
            "dim": self.dim,
            "active_count": int(self.slot_masks.sum().item()) if HAS_TORCH else 0,
            "metadata": self.slot_metadata,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if HAS_TORCH:
            payload = {
                "slots": self.slots,
                "masks": self.slot_masks,
            }
            torch.save(payload, out)

        return out

    def load(self, filepath: str | Path | None = None) -> None:
        """Carga las ranuras de memoria binaria persistentes desde disco."""
        src = Path(filepath) if filepath else self.memory_file
        if src is None or not src.exists():
            return

        meta_file = src.with_suffix(".json")
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.slot_metadata = data.get("metadata", [{} for _ in range(self.num_slots)])

        if HAS_TORCH and src.exists():
            payload = torch.load(src, weights_only=True)
            self.slots = payload["slots"]
            self.slot_masks = payload["masks"]
