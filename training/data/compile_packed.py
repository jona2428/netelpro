"""Streaming packed corpus compiler and loader for Netelpro / Teo v2.

Compiles unstructured or structured text into ONE contiguous uint16 token
stream with ZERO padding tokens, suitable for high-efficiency nanoGPT-style
packed-sequence autoregressive pretraining.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

import numpy as np

# Ensure project root is on sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class FallbackTokenizer:
    """Deterministic character-level fallback tokenizer matching Netelpro tokenizer API.

    Used when neither NetelproBPETokenizer nor NetelproTokenizer is available.
    """

    def __init__(self, custom_vocab: dict[str, int] | None = None) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        self.special_tokens = {
            "<|pad|>": self.pad_token_id,
            "<|bos|>": self.bos_token_id,
            "<|eos|>": self.eos_token_id,
            "<|unk|>": self.unk_token_id,
        }
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}

        if custom_vocab is not None:
            self.token_to_id = dict(custom_vocab)
        else:
            self.token_to_id = dict(self.special_tokens)
            base_chars = (
                [chr(i) for i in range(32, 127)]
                + list("\n\r\t ")
                + list("áéíóúÁÉÍÓÚñÑüÜ¿¡—–«»\"'“”")
            )
            for ch in base_chars:
                if ch not in self.token_to_id:
                    self.token_to_id[ch] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for ch in text:
            ids.append(self.token_to_id.get(ch, self.unk_token_id))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        chars = []
        for tid in token_ids:
            tid_int = int(tid)
            if tid_int in self.id_to_special:
                if not skip_special_tokens:
                    chars.append(self.id_to_special[tid_int])
            else:
                chars.append(self.id_to_token.get(tid_int, "<|unk|>"))
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> FallbackTokenizer:
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(custom_vocab=vocab)


def load_tokenizer(tokenizer_path: str | Path | Any | None = None) -> Any:
    """Loads tokenizer from artifact or creates default/fallback tokenizer.

    Imports NetelproBPETokenizer lazily inside this function to remain valid
    if netelpro.neuro.tokenizer_bpe is being created concurrently. Falls back to
    NetelproTokenizer or FallbackTokenizer.
    """
    if tokenizer_path is not None and hasattr(tokenizer_path, "encode") and hasattr(tokenizer_path, "decode"):
        return tokenizer_path

    # 1. Guarded lazy import of parallel NetelproBPETokenizer
    try:
        from netelpro.neuro.tokenizer_bpe import NetelproBPETokenizer
        if tokenizer_path is not None:
            return NetelproBPETokenizer.load(tokenizer_path)
        return NetelproBPETokenizer()
    except (ImportError, AttributeError):
        pass
    except Exception:
        pass

    # 2. Try native NetelproTokenizer
    try:
        from netelpro.neuro.tokenizer import NetelproTokenizer
        if tokenizer_path is not None:
            p = Path(tokenizer_path)
            if p.is_file():
                return NetelproTokenizer.load(p)
        else:
            return NetelproTokenizer()
    except (ImportError, AttributeError):
        pass
    except Exception:
        pass

    # 3. Fallback char-level tokenizer
    if tokenizer_path is not None:
        p = Path(tokenizer_path)
        if p.is_file():
            try:
                return FallbackTokenizer.load(p)
            except Exception:
                pass
    return FallbackTokenizer()


def iter_documents(input_path: str | Path, min_doc_chars: int = 80) -> Iterator[str]:
    """Streams text documents from a file or directory (.txt or .jsonl).

    For .jsonl files, parses each line for 'text' or 'prompt' fields.
    Drops documents with fewer than min_doc_chars characters.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")

    files: list[Path] = []
    if p.is_file():
        files = [p]
    elif p.is_dir():
        txt_files = sorted(p.rglob("*.txt"))
        jsonl_files = sorted(p.rglob("*.jsonl"))
        files = sorted(txt_files + jsonl_files)
        if not files:
            files = sorted(p.rglob("*.json"))

    for f in files:
        suffix = f.suffix.lower()
        if suffix == ".jsonl":
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = None
                    if isinstance(record, dict):
                        if "text" in record and isinstance(record["text"], str):
                            text = record["text"]
                        elif "prompt" in record and isinstance(record["prompt"], str):
                            text = record["prompt"]

                    if text and len(text.strip()) >= min_doc_chars:
                        yield text
        elif suffix == ".txt":
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if len(content.strip()) >= min_doc_chars:
                    yield content
            except Exception:
                continue
        elif suffix == ".json":
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    for item in data:
                        text = None
                        if isinstance(item, str):
                            text = item
                        elif isinstance(item, dict):
                            text = item.get("text") or item.get("prompt")
                        if text and len(text.strip()) >= min_doc_chars:
                            yield text
                elif isinstance(data, dict):
                    text = data.get("text") or data.get("prompt")
                    if text and len(text.strip()) >= min_doc_chars:
                        yield text
            except Exception:
                continue
        else:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if len(content.strip()) >= min_doc_chars:
                    yield content
            except Exception:
                continue


def compile_corpus_packed(
    input_path: str | Path,
    out_bin: str | Path,
    tokenizer: Any = None,
    max_tokens: int | None = None,
    min_doc_chars: int = 80,
    chunk_tokens: int = 50_000_000,
) -> dict[str, Any]:
    """Compiles a corpus into ONE contiguous uint16 stream with ZERO padding.

    Each document is formatted as: [bos] + encode(doc) + [eos].
    Tokens are written in raw uint16 little-endian chunks, flushed periodically.

    Writes:
      - <out_bin> : Raw uint16 contiguous stream
      - <out_bin>.meta.json : Sidecar metadata

    Returns:
      Dictionary containing compilation metadata.
    """
    tok = load_tokenizer(tokenizer)
    bos_id = getattr(tok, "bos_token_id", 1)
    eos_id = getattr(tok, "eos_token_id", 2)
    pad_id = getattr(tok, "pad_token_id", 0)

    vocab_size = getattr(tok, "vocab_size", None)
    if vocab_size is None and hasattr(tok, "token_to_id"):
        vocab_size = len(tok.token_to_id)
    if vocab_size is None:
        vocab_size = 65536

    out_path = Path(out_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    total_tokens = 0
    total_docs = 0
    buffer: list[int] = []

    def flush_chunk(f_out) -> None:
        nonlocal buffer
        if not buffer:
            return
        chunk_arr = np.array(buffer, dtype="<u2")
        if pad_id is not None:
            assert not np.any(chunk_arr == pad_id), (
                f"Validation failure: detected pad_token_id ({pad_id}) in packed token stream!"
            )
        chunk_bytes = chunk_arr.tobytes()
        f_out.write(chunk_bytes)
        hasher.update(chunk_bytes)
        buffer.clear()

    with open(out_path, "wb") as f_out:
        for doc in iter_documents(input_path, min_doc_chars=min_doc_chars):
            if max_tokens is not None and total_tokens >= max_tokens:
                break

            try:
                raw_tokens = tok.encode(doc, add_special_tokens=False)
            except TypeError:
                raw_tokens = tok.encode(doc)

            # Strip existing bos/eos if the tokenizer emitted them
            if len(raw_tokens) > 0 and raw_tokens[0] == bos_id:
                raw_tokens = raw_tokens[1:]
            if len(raw_tokens) > 0 and raw_tokens[-1] == eos_id:
                raw_tokens = raw_tokens[:-1]

            # Enforce zero padding in raw tokens
            if pad_id is not None:
                raw_tokens = [t for t in raw_tokens if t != pad_id]

            # Construct [bos] + encode(doc) + [eos]
            doc_seq = [bos_id] + list(raw_tokens) + [eos_id]

            if pad_id is not None:
                assert pad_id not in doc_seq, f"Pad id {pad_id} found in compiled document sequence!"

            if max_tokens is not None:
                remaining = max_tokens - total_tokens
                if len(doc_seq) > remaining:
                    doc_seq = doc_seq[:remaining]

            if not doc_seq:
                continue

            buffer.extend(doc_seq)
            total_tokens += len(doc_seq)
            total_docs += 1

            if len(buffer) >= chunk_tokens:
                flush_chunk(f_out)

            if max_tokens is not None and total_tokens >= max_tokens:
                break

        flush_chunk(f_out)

    meta_data: dict[str, Any] = {
        "total_tokens": int(total_tokens),
        "total_docs": int(total_docs),
        "vocab_size": int(vocab_size),
        "tokenizer_path": str(tokenizer) if tokenizer is not None else "default",
        "sha256": hasher.hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dtype": "uint16",
    }

    # Write primary sidecar: <out>.meta.json
    meta_path = Path(f"{out_path}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(meta_data, f_meta, indent=2)

    # If out_path has .bin suffix, also write <stem>.meta.json for convenience
    if out_path.suffix == ".bin":
        alt_meta_path = out_path.with_suffix(".meta.json")
        if alt_meta_path != meta_path:
            with open(alt_meta_path, "w", encoding="utf-8") as f_meta:
                json.dump(meta_data, f_meta, indent=2)

    return meta_data


# ---------------------------------------------------------------------------
# Loader utilities
# ---------------------------------------------------------------------------


def load_stream(bin_path: str | Path | np.ndarray) -> np.memmap:
    """Loads a binary uint16 token stream as a memory-mapped NumPy view."""
    if isinstance(bin_path, np.memmap):
        return bin_path
    if isinstance(bin_path, np.ndarray):
        return bin_path  # type: ignore[return-value]

    p = Path(bin_path)
    if not p.exists():
        raise FileNotFoundError(f"Binary stream file not found: {p}")

    if p.stat().st_size == 0:
        return np.empty(0, dtype=np.uint16)  # type: ignore[return-value]

    return np.memmap(p, dtype=np.uint16, mode="r")


def get_batch(
    stream: np.ndarray,
    block_size: int,
    batch_size: int,
    generator: np.random.Generator | None = None,
    device: str | None = None,
    return_tensor: bool = False,
) -> tuple[np.ndarray | Any, np.ndarray | Any]:
    """Samples random packed batches from token stream for training.

    Args:
        stream: uint16 contiguous token array (e.g. from load_stream).
        block_size: Length of each sample sequence.
        batch_size: Number of sequences in batch.
        generator: np.random.Generator instance for deterministic/reproducible sampling.
        device: Optional target device if returning torch tensors.
        return_tensor: If True, returns torch.Tensor instead of np.ndarray.

    Returns:
        (inputs, targets): int64 arrays of shape [batch_size, block_size]
        where targets is inputs shifted by 1 position.
    """
    if hasattr(stream, "cpu"):
        stream = stream.cpu()
    if hasattr(stream, "numpy"):
        stream = stream.numpy()

    stream_len = len(stream)
    seq_len = block_size + 1
    if stream_len < seq_len:
        raise ValueError(
            f"Stream length ({stream_len}) is too short for block_size ({block_size}) + 1."
        )

    if generator is None:
        generator = np.random.default_rng()
    elif isinstance(generator, (int, np.integer)):
        generator = np.random.default_rng(int(generator))

    max_offset = stream_len - block_size
    offsets = generator.integers(0, max_offset, size=batch_size)

    inputs = np.empty((batch_size, block_size), dtype=np.int64)
    targets = np.empty((batch_size, block_size), dtype=np.int64)

    for i, off in enumerate(offsets):
        inputs[i] = stream[off : off + block_size]
        targets[i] = stream[off + 1 : off + block_size + 1]

    if device is not None or return_tensor:
        try:
            import torch

            t_inputs = torch.from_numpy(inputs)
            t_targets = torch.from_numpy(targets)
            if device is not None:
                t_inputs = t_inputs.to(device)
                t_targets = t_targets.to(device)
            return t_inputs, t_targets
        except ImportError:
            pass

    return inputs, targets


def stream_stats(
    stream: np.ndarray,
    pad_token_id: int = 0,
) -> dict[str, Any]:
    """Computes diagnostic statistics on a token stream.

    Returns:
        dict with total_tokens, top10_unique_tokens, and pad_fraction.
    """
    if hasattr(stream, "cpu"):
        stream = stream.cpu()
    if hasattr(stream, "numpy"):
        stream = stream.numpy()

    total_tokens = int(len(stream))
    if total_tokens == 0:
        return {
            "total_tokens": 0,
            "top10_unique_tokens": [],
            "pad_fraction": 0.0,
        }

    pad_count = int(np.count_nonzero(stream == pad_token_id))
    pad_fraction = float(pad_count / total_tokens)

    counts = np.bincount(stream)
    top_indices = np.argsort(counts)[::-1]
    top10_unique_tokens = [int(idx) for idx in top_indices if counts[idx] > 0][:10]

    return {
        "total_tokens": total_tokens,
        "top10_unique_tokens": top10_unique_tokens,
        "pad_fraction": pad_fraction,
    }


def main(args_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compile text corpus into a contiguous uint16 packed stream with zero padding."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input file (.txt, .jsonl) or directory containing corpus files.",
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        default=None,
        help="Path to frozen tokenizer artifact (JSON vocab or BPE model).",
    )
    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="Output .bin file path.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional maximum token cap.",
    )
    parser.add_argument(
        "--min-doc-chars",
        type=int,
        default=80,
        help="Minimum characters required per document (default: 80).",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=50_000_000,
        help="Tokens to buffer before flushing to disk (default: 50,000,000).",
    )

    args = parser.parse_args(args_list)

    meta = compile_corpus_packed(
        input_path=args.input,
        out_bin=args.out,
        tokenizer=args.tokenizer,
        max_tokens=args.max_tokens,
        min_doc_chars=args.min_doc_chars,
        chunk_tokens=args.chunk_tokens,
    )

    print("✅ Packed corpus compilation completed:")
    print(f"   - Documents processed: {meta['total_docs']:,}")
    print(f"   - Total tokens: {meta['total_tokens']:,}")
    print(f"   - Vocab size: {meta['vocab_size']:,}")
    print(f"   - Output file: {args.out}")
    print(f"   - SHA256: {meta['sha256']}")
    print(f"   - Sidecar metadata: {args.out}.meta.json")


if __name__ == "__main__":
    main()
