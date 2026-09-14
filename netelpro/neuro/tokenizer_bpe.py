"""Netelpro Frozen BPE Tokenizer for teo_v2 Scale-Up.

Implements a frozen Byte-Level BPE tokenizer using Hugging Face tokenizers.
Preserves exact special token IDs, provides byte-level fallback with zero UNKs,
and persists as an immutable artifact with metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIAL_TOKENS: list[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|user|>",
    "<|assistant|>",
    "<|thought|>",
    "<|endthought|>",
    "<|contract|>",
    "<|endcontract|>",
    "<|audit|>",
]

SPECIAL_TOKEN_TO_ID: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}


class VocabDict(dict[str, Any]):
    """Dictionary supporting both dict item lookup and callable lookup."""

    def __call__(self, key: Any) -> Any:
        return self.get(key)


class NetelproBPETokenizer:
    """Frozen Byte-Level BPE Tokenizer for Netelpro LLM models."""

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = self._create_base_tokenizer()

        self._ensure_special_tokens()
        self._vocab_cache: VocabDict | None = None
        self._id_vocab_cache: VocabDict | None = None
        self.metadata: dict[str, Any] = {}

    def _clear_cache(self) -> None:
        self._vocab_cache = None
        self._id_vocab_cache = None

    @classmethod
    def _create_base_tokenizer(cls) -> Tokenizer:
        """Creates a base BPE tokenizer initialized with special tokens and full byte alphabet."""
        vocab: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        offset = len(vocab)
        for i, b in enumerate(pre_tokenizers.ByteLevel.alphabet()):
            vocab[b] = offset + i

        tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<|unk|>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.add_special_tokens([AddedToken(s, special=True) for s in SPECIAL_TOKENS])
        return tok

    def _ensure_special_tokens(self) -> None:
        """Verifies and ensures special tokens have exact fixed IDs 0..10."""
        for expected_id, token in enumerate(SPECIAL_TOKENS):
            actual_id = self._tokenizer.token_to_id(token)
            if actual_id is None:
                self._tokenizer.add_special_tokens([AddedToken(token, special=True)])
                actual_id = self._tokenizer.token_to_id(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"Special token {token} has ID {actual_id}, expected exact ID {expected_id}"
                )

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including added special tokens."""
        return self._tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def unk_token_id(self) -> int:
        return 3

    @property
    def user_token_id(self) -> int:
        return 4

    @property
    def assistant_token_id(self) -> int:
        return 5

    @property
    def thought_token_id(self) -> int:
        return 6

    @property
    def endthought_token_id(self) -> int:
        return 7

    @property
    def contract_token_id(self) -> int:
        return 8

    @property
    def endcontract_token_id(self) -> int:
        return 9

    @property
    def audit_token_id(self) -> int:
        return 10

    @property
    def token_to_id(self) -> VocabDict:
        """Vocabulary mapping token string to token id."""
        if self._vocab_cache is None:
            self._vocab_cache = VocabDict(self._tokenizer.get_vocab())
        return self._vocab_cache

    @property
    def id_to_token(self) -> VocabDict:
        """Vocabulary mapping token id to token string."""
        if self._id_vocab_cache is None:
            vocab = self._tokenizer.get_vocab()
            self._id_vocab_cache = VocabDict({v: k for k, v in vocab.items()})
        return self._id_vocab_cache

    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]:
        """Returns vocabulary mapping."""
        return self._tokenizer.get_vocab(with_added_tokens=with_added_tokens)

    def train(
        self,
        files: list[str | Path] | list[Any],
        vocab_size: int = 32768,
        min_frequency: int = 2,
        limit: int | None = None,
    ) -> NetelproBPETokenizer:
        """Trains BPE model on corpus files, preserving exact special token IDs.

        Args:
            files: List of file paths (JSONL or plain text).
            vocab_size: Target vocabulary size.
            min_frequency: Minimum token frequency for BPE merges.
            limit: Optional maximum number of lines read per file.

        Returns:
            self
        """
        file_list: list[str | Path] = list(files) if isinstance(files, (list, tuple)) else [files]

        def _iter_corpus() -> Iterator[str]:
            for entry in file_list:
                p = Path(entry)
                if not p.is_file():
                    continue
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    line_count = 0
                    for line in fh:
                        if limit is not None and line_count >= limit:
                            break
                        line_count += 1
                        raw = line.strip()
                        if not raw:
                            continue
                        if raw.startswith("{") and raw.endswith("}"):
                            try:
                                data = json.loads(raw)
                                if isinstance(data, dict):
                                    extracted = False
                                    for k in ("prompt", "completion", "response", "text", "content"):
                                        val = data.get(k)
                                        if val and isinstance(val, str):
                                            yield val
                                            extracted = True
                                    if "messages" in data and isinstance(data["messages"], list):
                                        for msg in data["messages"]:
                                            if isinstance(msg, dict) and "content" in msg:
                                                c = msg.get("content")
                                                if c and isinstance(c, str):
                                                    yield c
                                                    extracted = True
                                    if extracted:
                                        continue
                            except Exception:
                                pass
                        yield line

        tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tok.train_from_iterator(_iter_corpus(), trainer=trainer)
        self._tokenizer = tok
        self._clear_cache()
        self._ensure_special_tokens()
        return self

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encodes string to a list of token IDs using BPE.

        If add_special_tokens is True, prepends <|bos|> and appends <|eos|>.
        """
        if not text and not add_special_tokens:
            return []

        encoding = self._tokenizer.encode(text)
        token_ids: list[int] = list(encoding.ids)

        if add_special_tokens:
            return [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Decodes sequence of token IDs back into text."""
        if not token_ids:
            return ""
        if hasattr(token_ids, "tolist"):
            ids = token_ids.tolist()
        else:
            ids = list(token_ids)
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def format_chat(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Formats chat dialog using standard Netelpro delimiters."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}<|eos|>\n"
            elif role == "system":
                formatted += f"<|contract|>\n{content}\n<|endcontract|>\n"

        if add_generation_prompt:
            formatted += "<|assistant|>\n"
        return formatted

    def save(self, path: str | Path) -> None:
        """Saves tokenizer artifact and metadata sidecar."""
        p = Path(path)
        if p.is_dir() or (not p.suffix and not p.exists()):
            p.mkdir(parents=True, exist_ok=True)
            artifact_file = p / "tokenizer.json"
            meta_file = p / "tokenizer.meta.json"
            dot_meta_file = p / ".meta.json"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            artifact_file = p
            meta_file = p.with_name(f"{p.stem}.meta.json")
            dot_meta_file = p.parent / ".meta.json"

        self._tokenizer.save(str(artifact_file))

        meta_content = {
            "tokenizer_type": "NetelproBPETokenizer",
            "vocab_size": self.vocab_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "special_tokens": {tok: self._tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
            "special_token_ids": {tok: self._tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, ensure_ascii=False, indent=2)

        if dot_meta_file != meta_file:
            try:
                with open(dot_meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta_content, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    @classmethod
    def load(cls, path: str | Path) -> NetelproBPETokenizer:
        """Loads frozen tokenizer from saved artifact.

        Raises:
            FileNotFoundError: If the path or tokenizer artifact does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Tokenizer artifact path does not exist: {path}")

        if p.is_dir():
            cand = p / "tokenizer.json"
            if cand.is_file():
                artifact_file = cand
            else:
                json_files = [
                    f for f in p.glob("*.json")
                    if not f.name.endswith(".meta.json") and not f.name.startswith(".meta")
                ]
                if json_files:
                    artifact_file = json_files[0]
                else:
                    raise FileNotFoundError(f"No tokenizer.json found in directory: {path}")
        elif p.is_file():
            if p.name.endswith(".meta.json"):
                base_name = p.name[:-10] + ".json"
                base_file = p.with_name(base_name)
                if base_file.is_file():
                    artifact_file = base_file
                else:
                    artifact_file = p.with_name("tokenizer.json")
            else:
                artifact_file = p
        else:
            raise FileNotFoundError(f"Invalid path for tokenizer: {path}")

        if not artifact_file.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {artifact_file}")

        hf_tok = Tokenizer.from_file(str(artifact_file))
        instance = cls(tokenizer=hf_tok)

        # Load sidecar metadata if present
        meta_candidates = [
            artifact_file.with_name(f"{artifact_file.stem}.meta.json"),
            artifact_file.parent / ".meta.json",
        ]
        for mc in meta_candidates:
            if mc.is_file():
                try:
                    with open(mc, "r", encoding="utf-8") as mf:
                        instance.metadata = json.load(mf)
                    break
                except Exception:
                    pass

        return instance

    def __repr__(self) -> str:
        return f"<NetelproBPETokenizer vocab_size={self.vocab_size} pad={self.pad_token_id} bos={self.bos_token_id} eos={self.eos_token_id}>"
