"""Netelpro Mini LLM: High-level Neuro-Symbolic Language Model.

Bundles the NetelproTransformer architecture, native tokenizer, streaming
generation, and formal silicon audit certificates into an easy-to-use,
production-ready pretrainable and fine-tunable mini model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator, Optional, Sequence

from netelpro.neuro.certificate import AuditCertificate
from netelpro.neuro.memory import NetelproBinaryMemoryBank
from netelpro.neuro.stream import NetelproStreamProcessor
from netelpro.neuro.tokenizer import NetelproTokenizer
from netelpro.neuro.transformer import NetelproTransformer, NetelproTransformerConfig
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn as nn
    _ModuleBase = nn.Module
else:
    class _ModuleBase:  # type: ignore
        pass


class NetelproMiniLLM(_ModuleBase):
    """The Netelpro Mini Language Model."""

    def __init__(
        self,
        config: Optional[NetelproTransformerConfig] = None,
        tokenizer: Optional[NetelproTokenizer] = None,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("NetelproMiniLLM requires PyTorch.")

        super().__init__()
        self.tokenizer = tokenizer or NetelproTokenizer()

        if config is None:
            self.config = NetelproTransformerConfig(
                vocab_size=self.tokenizer.vocab_size,
                block_size=256,
                n_layer=4,
                n_head=4,
                n_embd=128,
                dropout=0.1,
            )
        else:
            self.config = config
            self.config.vocab_size = self.tokenizer.vocab_size

        self.model = NetelproTransformer(self.config)
        self.stream_processor = NetelproStreamProcessor(
            allowed_min=0,
            allowed_max=self.config.vocab_size,
            safety_state=1,
        )
        self.memory_bank = NetelproBinaryMemoryBank(
            num_slots=256,
            dim=self.config.n_embd,
        )

    def remember(self, concept: str, confidence: float = 1.0) -> int:
        """Encodes a factual concept into a latent vector and writes it directly to the binary memory bank."""
        self.model.eval()
        tokens = self.tokenizer.encode(concept, add_special_tokens=False)
        if not tokens:
            return -1
        idx = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            tok_emb = self.model.wte(idx)
            concept_vec = tok_emb.mean(dim=1).squeeze(0)
        return self.memory_bank.write_memory(concept, concept_vec, confidence=confidence)

    def recall(
        self, query: str, top_k: int = 3, threshold: float = 0.25
    ) -> list[tuple[int, float, dict[str, Any]]]:
        """Retrieves relevant memory records for a given query from the binary memory bank."""
        self.model.eval()
        tokens = self.tokenizer.encode(query, add_special_tokens=False)
        if not tokens:
            return []
        idx = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            tok_emb = self.model.wte(idx)
            query_vec = tok_emb.mean(dim=1).squeeze(0)
        return self.memory_bank.retrieve_relevant_memories(query_vec, top_k=top_k, threshold=threshold)

    def list_memories(self) -> list[dict[str, Any]]:
        """Returns all persistent memories in the writeable binary bank."""
        return self.memory_bank.list_memories()

    def clear_memories(self) -> None:
        """Clears all persistent memories in the writeable binary bank."""
        self.memory_bank.clear()

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        control_flags: int | Sequence[int] = 1,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], AuditCertificate]:
        return self.model(idx, targets=targets, control_flags=control_flags)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        control_flags: int = 1,
    ) -> tuple[str, AuditCertificate]:
        """Generates text from a text prompt and emits an audit certificate."""
        self.model.eval()
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not tokens or tokens[0] != self.tokenizer.bos_token_id:
            tokens.insert(0, self.tokenizer.bos_token_id)

        idx = torch.tensor([tokens], dtype=torch.long)
        out_tokens = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            control_flags=control_flags,
        )
        generated_ids = out_tokens[0].tolist()[len(tokens) :]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Get formal certificate for the final state
        _, _, cert = self.model(out_tokens[:, -min(len(out_tokens[0]), self.config.block_size) :], control_flags=control_flags)
        return response_text, cert

    def stream_chat(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.4,
        top_k: int = 5,
        repetition_penalty: float = 1.25,
        control_flags: int = 1,
    ) -> Generator[tuple[str, dict[str, Any]], None, None]:
        """Streams response tokens one by one with real-time audit details."""
        self.model.eval()
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not tokens or tokens[0] != self.tokenizer.bos_token_id:
            tokens.insert(0, self.tokenizer.bos_token_id)

        curr_tokens = list(tokens)

        for step in range(max_new_tokens):
            ctx = curr_tokens[-self.config.block_size :]
            idx_tensor = torch.tensor([ctx], dtype=torch.long)

            logits, _, cert = self.model(idx_tensor, control_flags=control_flags)
            next_logits = logits[0, -1, :].clone()

            # Apply repetition penalty to recent tokens
            if repetition_penalty != 1.0:
                recent_ids = set(curr_tokens[-32:])
                for tid in recent_ids:
                    if next_logits[tid] > 0:
                        next_logits[tid] /= repetition_penalty
                    else:
                        next_logits[tid] *= repetition_penalty

            # Filter logits using silicon processor
            filtered_logits = self.stream_processor.process_hf_logits(
                idx_tensor, next_logits.unsqueeze(0)
            ).squeeze(0)

            if temperature <= 0.0 or top_k == 1:
                next_id = int(torch.argmax(filtered_logits).item())
            else:
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(filtered_logits, min(top_k, filtered_logits.size(-1)))
                    filtered_logits[filtered_logits < v[[-1]]] = float("-inf")

                probs = torch.softmax(filtered_logits / max(1e-5, temperature), dim=-1)
                if torch.isnan(probs).any() or probs.sum() <= 0:
                    next_id = self.tokenizer.eos_token_id
                else:
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

            curr_tokens.append(next_id)
            tok_text = self.tokenizer.decode([next_id], skip_special_tokens=False)

            audit_info = {
                "step": step + 1,
                "token_id": next_id,
                "total_neurons": sum(r.total_neurons for r in cert.records),
                "active_neurons": sum(r.active_neurons for r in cert.records),
                "suppressed_neurons": sum(r.suppressed_neurons for r in cert.records),
                "latency_us": cert.total_latency_us,
            }

            yield tok_text, audit_info

            if next_id == self.tokenizer.eos_token_id:
                break

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Saves model weights, configuration, and tokenizer to directory."""
        dir_path = Path(save_directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 1. Save config
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "block_size": self.config.block_size,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "n_embd": self.config.n_embd,
            "dropout": self.config.dropout,
            "scale_factor": self.config.scale_factor,
            "z_min": self.config.z_min,
            "z_max": self.config.z_max,
        }
        with open(dir_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        # 2. Save tokenizer
        self.tokenizer.save(dir_path / "vocab.json")

        # 3. Save weights
        torch.save(self.model.state_dict(), dir_path / "model.pt")

        # 4. Save writeable binary memory bank
        self.memory_bank.save(dir_path / "live_memory.bin")

    @classmethod
    def from_pretrained(cls, save_directory: str | Path) -> NetelproMiniLLM:
        """Loads a pretrained NetelproMiniLLM from directory."""
        dir_path = Path(save_directory)
        with open(dir_path / "config.json", "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        tokenizer = NetelproTokenizer.load(dir_path / "vocab.json")

        config = NetelproTransformerConfig(
            vocab_size=cfg_data["vocab_size"],
            block_size=cfg_data["block_size"],
            n_layer=cfg_data["n_layer"],
            n_head=cfg_data["n_head"],
            n_embd=cfg_data["n_embd"],
            dropout=cfg_data.get("dropout", 0.0),
            scale_factor=cfg_data.get("scale_factor", 1000.0),
            z_min=cfg_data.get("z_min", -5000),
            z_max=cfg_data.get("z_max", 5000),
        )

        minillm = cls(config=config, tokenizer=tokenizer)
        state_dict = torch.load(dir_path / "model.pt", weights_only=True)
        minillm.model.load_state_dict(state_dict)

        mem_file = dir_path / "live_memory.bin"
        if mem_file.exists():
            minillm.memory_bank.load(mem_file)

        return minillm
