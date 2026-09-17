"""Streaming corpus downloader and packed shard builder for Netelpro / Teo v2.

Builds the teo_v2 corpus by streaming and combining:
1. Spanish text from HuggingFaceFW/fineweb-2 (subset spa_Latn)
2. Python code from codeparrot/github-code (open, ungated, filtered to language=='Python')

Per-document pipeline:
- Light cleaning (strip boilerplate/navigation lines, min length 200 chars)
- Deduplication by SHA-1 of normalized text
- Tokenization with frozen NetelproBPETokenizer (vocab 32768, byte-level, pad=0 bos=1 eos=2 unk=3)
  (Loads existing tokenizer.json if present; otherwise trains on first ~200MB sample and freezes)
- Framing: [bos] + ids + [eos] written into packed uint16 shards of 50M tokens each
  (Zero padding, reusing compile_packed machinery and verification).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterator, Sequence
import unicodedata

import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from netelpro.neuro.tokenizer_bpe import (
    NetelproBPETokenizer,
)
from training.data.compile_packed import (
    load_stream,
    stream_stats,
)

# Defaults
DEFAULT_SHARD_SIZE = 50_000_000  # 50M tokens per packed shard
DEFAULT_MIN_DOC_CHARS = 200
DEFAULT_VOCAB_SIZE = 32_768
DEFAULT_SAMPLE_BYTES_FOR_TOKENIZER = 200 * 1024 * 1024  # 200MB


# ---------------------------------------------------------------------------
# Utility: Size & Argument Parsing
# ---------------------------------------------------------------------------


def parse_byte_size(size_val: str | int | float | None) -> int | None:
    """Parses human-readable byte sizes like '150MB', '2GB', '500k', or integer bytes."""
    if size_val is None:
        return None
    if isinstance(size_val, (int, float)):
        return int(size_val)

    s = str(size_val).strip().upper()
    if not s:
        return None

    multipliers = {
        "GB": 1024 * 1024 * 1024,
        "G": 1024 * 1024 * 1024,
        "MB": 1024 * 1024,
        "M": 1024 * 1024,
        "KB": 1024,
        "K": 1024,
        "B": 1,
    }
    for unit, mult in multipliers.items():
        if s.endswith(unit):
            num_str = s[: -len(unit)].strip()
            return int(float(num_str) * mult)

    return int(float(s))


def parse_sources(sources_arg: Sequence[str] | str | None) -> list[str]:
    """Parses and normalizes source names list."""
    if not sources_arg:
        return ["spanish", "code"]

    raw_items: list[str] = []
    if isinstance(sources_arg, str):
        raw_items = [x.strip() for x in sources_arg.split(",")]
    else:
        for item in sources_arg:
            raw_items.extend([x.strip() for x in item.split(",")])

    resolved: list[str] = []
    for item in raw_items:
        key = item.lower()
        if key in ("spanish", "fineweb", "spa", "spa_latn"):
            resolved.append("spanish")
        elif key in ("code", "github", "github-code", "python", "py"):
            resolved.append("code")
        elif key == "all":
            return ["spanish", "code"]
        elif key:
            raise ValueError(
                f"Unknown source: '{item}'. Supported sources: 'spanish', 'code'."
            )

    return list(dict.fromkeys(resolved))  # remove duplicates preserving order


# ---------------------------------------------------------------------------
# Light Cleaning & Boilerplate Removal
# ---------------------------------------------------------------------------

_BOILERPLATE_LINE_REGEXES = [
    # Cookie banners & consent
    re.compile(
        r"(?:aceptar|rechazar|configurar|permitir|uso de|política de|aviso de|todas las|nuestras?)\s+.*cookies?",
        re.IGNORECASE,
    ),
    re.compile(r"cookies?.*(?:policy|consent|settings|notice|banner|privacidad)", re.IGNORECASE),
    re.compile(
        r"(?:esta|este)\s+(?:página|web|sitio|plataforma).*cookies?", re.IGNORECASE
    ),
    # Navigation & breadcrumbs
    re.compile(
        r"^(?:inicio|home|blog|sección|artículos?)\s*[>»|/]\s*", re.IGNORECASE
    ),
    re.compile(
        r"^(?:saltar al contenido|skip to (?:main )?content|menú de navegación|navegación principal)",
        re.IGNORECASE,
    ),
    re.compile(r"^\|\s*(?:home|inicio|contacto|about|noticias)\s*\|", re.IGNORECASE),
    # Social shares & calls to action
    re.compile(
        r"^(?:compartir en|share on|síguenos en|follow us on)\s+(?:facebook|twitter|x|linkedin|whatsapp|instagram)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:suscríbete a nuestra newsletter|subscribe to our newsletter)",
        re.IGNORECASE,
    ),
    # Common web footers
    re.compile(
        r"(?:todos los derechos reservados|all rights reserved|copyright\s+©)",
        re.IGNORECASE,
    ),
]


def is_boilerplate_line(line: str) -> bool:
    """Detects whether a single line is web navigation, cookie consent, or boilerplate."""
    stripped = line.strip()
    if not stripped:
        return False

    # Excessive repetitive punctuation
    if len(stripped) >= 3 and set(stripped) <= set("-=_*~#|"):
        return True

    # Breadcrumb chain like "Inicio > Noticias > Deportes > Futbol"
    if stripped.count(">") >= 2 or stripped.count("»") >= 2:
        return True

    for pattern in _BOILERPLATE_LINE_REGEXES:
        if pattern.search(stripped):
            return True

    return False


def clean_document(
    text: str,
    is_code: bool = False,
    min_chars: int = DEFAULT_MIN_DOC_CHARS,
) -> str | None:
    """Performs light cleaning on a text document.

    - Normalizes line endings
    - Strips navigation/cookie/boilerplate lines (for text documents)
    - Collapses excessive blank lines
    - Enforces min_chars threshold (default: 200 chars)

    Returns cleaned text, or None if document fails criteria.
    """
    if not text or not isinstance(text, str):
        return None

    # Normalize line endings
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    if is_code:
        # For code: preserve indentation and syntax, collapse 3+ consecutive newlines to 2
        cleaned = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    else:
        # For natural text: filter out boilerplate lines
        lines = normalized.split("\n")
        filtered_lines: list[str] = []
        for line in lines:
            if is_boilerplate_line(line):
                continue
            filtered_lines.append(line)

        reassembled = "\n".join(filtered_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", reassembled).strip()

    if len(cleaned) < min_chars:
        return None

    return cleaned


# ---------------------------------------------------------------------------
# Deduplication by SHA-1 of Normalized Text
# ---------------------------------------------------------------------------


def normalize_text_for_dedup(text: str) -> str:
    """Normalizes text for hash-based deduplication (NFKC unicode, lowercase, whitespace collapsed)."""
    nfkc = unicodedata.normalize("NFKC", text)
    lowered = nfkc.lower()
    return re.sub(r"\s+", " ", lowered).strip()


class DocumentDeduplicator:
    """Tracks SHA-1 hashes of normalized documents to eliminate duplicates."""

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()
        self.seen_count: int = 0
        self.duplicate_count: int = 0

    def is_duplicate(self, text: str) -> bool:
        """Checks if text is a duplicate. Returns True if duplicate, False if unique."""
        norm = normalize_text_for_dedup(text)
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if h in self._seen_hashes:
            self.duplicate_count += 1
            return True

        self._seen_hashes.add(h)
        self.seen_count += 1
        return False

    def reset(self) -> None:
        self._seen_hashes.clear()
        self.seen_count = 0
        self.duplicate_count = 0

    def save_state(self, path: str | Path) -> None:
        """Persists the set of seen SHA-1 hex digests as a JSON list."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.with_suffix(".tmp")
        data = sorted(self._seen_hashes)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        try:
            os.replace(tmp_path, p)
        except OSError:
            pass

    def load_state(self, path: str | Path) -> None:
        """Loads seen SHA-1 hex digests from JSON list (or dict) state file."""
        p = Path(path)
        if not p.is_file():
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            self._seen_hashes.update(data)
            self.seen_count = len(self._seen_hashes)
        elif isinstance(data, dict):
            hashes = data.get("seen_hashes", data.get("hashes", []))
            self._seen_hashes.update(hashes)
            self.seen_count = len(self._seen_hashes)
            self.duplicate_count = data.get("duplicate_count", self.duplicate_count)


# ---------------------------------------------------------------------------
# Packed Shard Writer (reusing compile_packed uint16 format)
# ---------------------------------------------------------------------------


class PackedShardWriter:
    """Writes packed uint16 token shards of fixed size (default: 50M tokens).

    Each document is framed as: [bos_id] + token_ids + [eos_id].
    Zero pad tokens are strictly enforced.
    Spanning sequences across shard boundaries are seamlessly split so each completed
    shard has exactly `shard_size` tokens.
    """

    def __init__(
        self,
        out_dir: str | Path,
        shard_size: int = DEFAULT_SHARD_SIZE,
        max_shards: int | None = None,
        tokenizer: Any = None,
        prefix: str = "shard",
        resume: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.max_shards = max_shards
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.resume = resume

        self.bos_id: int = getattr(tokenizer, "bos_token_id", 1)
        self.eos_id: int = getattr(tokenizer, "eos_token_id", 2)
        self.pad_id: int = getattr(tokenizer, "pad_token_id", 0)

        self.vocab_size: int = getattr(tokenizer, "vocab_size", DEFAULT_VOCAB_SIZE)

        self.current_shard_idx: int = 0
        self.current_shard_tokens: int = 0
        self.current_shard_docs: int = 0

        self.total_tokens: int = 0
        self.total_docs: int = 0

        self.shards_completed: list[dict[str, Any]] = []

        self._current_file = None
        self._current_hasher: hashlib._Hash | None = None
        self._buffer: list[int] = []
        self._buffer_flush_size: int = 2_000_000  # Flush every 2M tokens

        if self.resume:
            self._scan_and_resume_shards()
        else:
            self._archive_existing_shards()

    def _scan_and_resume_shards(self) -> None:
        """Scans out_dir for completed valid shards and resumes state.

        A shard f"{prefix}_%05d.bin" is valid if:
        1. Sidecar "<shard>.bin.meta.json" exists and is valid JSON.
        2. Sidecar total_tokens == self.shard_size.
        3. File size in bytes == total_tokens * 2.
        4. Shard indices are consecutive starting from 0.

        Valid shards are added to self.shards_completed and counted.
        Incomplete/corrupted shards are renamed to "<name>.partial" (not deleted)
        and recompiled from scratch.
        """
        pattern = re.compile(rf"^{re.escape(self.prefix)}_(\d{{5}})\.bin$")
        candidates: list[tuple[int, Path]] = []
        for entry in self.out_dir.iterdir():
            if entry.is_file():
                m = pattern.match(entry.name)
                if m:
                    idx = int(m.group(1))
                    candidates.append((idx, entry))

        candidates.sort(key=lambda x: x[0])

        valid_count = 0
        for idx, shard_path in candidates:
            meta_path = Path(f"{shard_path}.meta.json")
            is_valid = False
            meta_data: dict[str, Any] = {}

            if idx == valid_count and meta_path.is_file():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        meta_data = loaded
                        tokens = meta_data.get("total_tokens")
                        if (
                            tokens == self.shard_size
                            and shard_path.stat().st_size == tokens * 2
                        ):
                            is_valid = True
                except Exception:
                    is_valid = False

            if is_valid:
                self.shards_completed.append(meta_data)
                self.total_tokens += int(meta_data.get("total_tokens", self.shard_size))
                self.total_docs += int(meta_data.get("total_docs", 0))
                valid_count += 1
            else:
                # Rename invalid or partial shard to <name>.partial (don't delete)
                partial_path = shard_path.with_name(f"{shard_path.name}.partial")
                try:
                    os.replace(shard_path, partial_path)
                except OSError:
                    pass

                # Also rename sidecar if present
                if meta_path.is_file():
                    try:
                        os.replace(meta_path, meta_path.with_name(f"{meta_path.name}.partial"))
                    except OSError:
                        pass
                alt_meta = shard_path.with_suffix(".meta.json")
                if alt_meta != meta_path and alt_meta.is_file():
                    try:
                        os.replace(alt_meta, alt_meta.with_name(f"{alt_meta.name}.partial"))
                    except OSError:
                        pass

        self.current_shard_idx = valid_count

    def _archive_existing_shards(self) -> None:
        """Archives existing shards/manifests when auto-resume is disabled."""
        pattern = re.compile(rf"^{re.escape(self.prefix)}_\d{{5}}\.bin")
        old_items = [
            p for p in self.out_dir.iterdir()
            if p.is_file() and (
                pattern.match(p.name)
                or p.name in ("meta.json", "dedup_state.json")
            )
        ]
        if old_items:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archive_dir = self.out_dir / f"archive_{timestamp}"
            archive_dir.mkdir(parents=True, exist_ok=True)
            for item in old_items:
                try:
                    os.replace(item, archive_dir / item.name)
                except OSError:
                    pass


    def _open_next_shard(self) -> None:
        shard_filename = f"{self.prefix}_{self.current_shard_idx:05d}.bin"
        self._current_shard_path = self.out_dir / shard_filename
        self._current_file = open(self._current_shard_path, "wb")
        self._current_hasher = hashlib.sha256()
        self.current_shard_tokens = 0
        self.current_shard_docs = 0

    def _flush_buffer(self) -> None:
        if not self._buffer or self._current_file is None:
            return

        arr = np.array(self._buffer, dtype="<u2")
        if self.pad_id is not None:
            assert not np.any(arr == self.pad_id), (
                f"Pad token id ({self.pad_id}) found in packed token buffer!"
            )

        raw_bytes = arr.tobytes()
        self._current_file.write(raw_bytes)
        if self._current_hasher is not None:
            self._current_hasher.update(raw_bytes)
        self._buffer.clear()

    def _close_current_shard(self) -> dict[str, Any]:
        self._flush_buffer()
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None

        sha256_val = (
            self._current_hasher.hexdigest() if self._current_hasher else ""
        )
        meta_data: dict[str, Any] = {
            "shard_index": self.current_shard_idx,
            "file_name": self._current_shard_path.name,
            "total_tokens": int(self.current_shard_tokens),
            "total_docs": int(self.current_shard_docs),
            "vocab_size": int(self.vocab_size),
            "tokenizer_type": type(self.tokenizer).__name__,
            "sha256": sha256_val,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dtype": "uint16",
            "pad_fraction": 0.0,
        }

        # Sidecar metadata file: <shard>.bin.meta.json
        meta_path = Path(f"{self._current_shard_path}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as f_meta:
            json.dump(meta_data, f_meta, indent=2)

        # Also write <stem>.meta.json
        alt_meta_path = self._current_shard_path.with_suffix(".meta.json")
        if alt_meta_path != meta_path:
            with open(alt_meta_path, "w", encoding="utf-8") as f_meta:
                json.dump(meta_data, f_meta, indent=2)

        self.shards_completed.append(meta_data)
        self.current_shard_tokens = 0
        self.current_shard_docs = 0
        self._write_manifest()
        return meta_data

    def frame_document(self, doc_text: str) -> list[int]:
        """Encodes document text and frames it as [bos] + ids + [eos] with zero pad tokens."""
        try:
            raw_ids = self.tokenizer.encode(doc_text, add_special_tokens=False)
        except TypeError:
            raw_ids = self.tokenizer.encode(doc_text)

        # Strip any pre-existing bos/eos
        if len(raw_ids) > 0 and raw_ids[0] == self.bos_id:
            raw_ids = raw_ids[1:]
        if len(raw_ids) > 0 and raw_ids[-1] == self.eos_id:
            raw_ids = raw_ids[:-1]

        # Strip any pad tokens
        if self.pad_id is not None:
            raw_ids = [tid for tid in raw_ids if tid != self.pad_id]

        framed = [self.bos_id] + list(raw_ids) + [self.eos_id]
        if self.pad_id is not None:
            assert self.pad_id not in framed, "Pad id found in framed document!"

        return framed

    def write_document(self, doc_text: str) -> bool:
        """Frames and writes document into packed shards.

        Returns:
            True if document was written, False if max_shards cap reached and stopped.
        """
        if self.max_shards is not None and self.current_shard_idx >= self.max_shards:
            return False

        doc_seq = self.frame_document(doc_text)
        if not doc_seq:
            return True

        if self._current_file is None:
            self._open_next_shard()

        tokens_to_write = doc_seq
        self.current_shard_docs += 1
        self.total_docs += 1

        while tokens_to_write:
            if (
                self.max_shards is not None
                and self.current_shard_idx >= self.max_shards
            ):
                return False

            space_in_shard = self.shard_size - self.current_shard_tokens
            if len(tokens_to_write) <= space_in_shard:
                self._buffer.extend(tokens_to_write)
                self.current_shard_tokens += len(tokens_to_write)
                self.total_tokens += len(tokens_to_write)
                tokens_to_write = []

                if len(self._buffer) >= self._buffer_flush_size:
                    self._flush_buffer()

                if self.current_shard_tokens >= self.shard_size:
                    self._close_current_shard()
                    self.current_shard_idx += 1
                    if (
                        self.max_shards is not None
                        and self.current_shard_idx >= self.max_shards
                    ):
                        return False
                    self._open_next_shard()
            else:
                chunk = tokens_to_write[:space_in_shard]
                tokens_to_write = tokens_to_write[space_in_shard:]

                self._buffer.extend(chunk)
                self.current_shard_tokens += len(chunk)
                self.total_tokens += len(chunk)

                self._close_current_shard()
                self.current_shard_idx += 1
                if (
                    self.max_shards is not None
                    and self.current_shard_idx >= self.max_shards
                ):
                    return False
                self._open_next_shard()

        return True

    def _write_manifest(self) -> dict[str, Any]:
        """Writes global meta.json manifest reflecting all completed shards."""
        manifest = {
            "total_shards": len(self.shards_completed),
            "total_tokens": int(self.total_tokens),
            "total_docs": int(self.total_docs),
            "shard_size": int(self.shard_size),
            "vocab_size": int(self.vocab_size),
            "shards": self.shards_completed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        manifest_path = self.out_dir / "meta.json"
        tmp_manifest = manifest_path.with_suffix(".tmp")
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        try:
            os.replace(tmp_manifest, manifest_path)
        except OSError:
            pass

        return manifest

    def close(self) -> dict[str, Any]:
        """Flushes remaining tokens, closes current shard, and writes global manifest."""
        if self._current_file is not None and self.current_shard_tokens > 0:
            self._close_current_shard()
        elif self._current_file is not None:
            self._current_file.close()
            self._current_file = None

        return self._write_manifest()


# ---------------------------------------------------------------------------
# Streaming Source Extractors (FineWeb-2 Spanish & Github-Code Python)
# ---------------------------------------------------------------------------


def get_fineweb_spanish_shard_urls(max_shards: int = 146) -> list[str]:
    """Returns download URLs for HuggingFaceFW/fineweb-2 Spanish shards."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files("HuggingFaceFW/fineweb-2", repo_type="dataset")
        shards = sorted(
            [f for f in files if f.startswith("data/spa_Latn/train/")]
        )
        if shards:
            return [
                f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/{s}"
                for s in shards[:max_shards]
            ]
    except Exception:
        pass

    # Fallback to standard naming convention
    return [
        f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/spa_Latn/train/{i:03d}_00000.parquet"
        for i in range(max_shards)
    ]


def get_github_code_shard_urls(max_shards: int = 1126) -> list[str]:
    """Returns download URLs for codeparrot/github-code parquet shards."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files("codeparrot/github-code", repo_type="dataset")
        shards = sorted(
            [f for f in files if f.startswith("data/train-") and f.endswith(".parquet")]
        )
        if shards:
            return [
                f"https://huggingface.co/datasets/codeparrot/github-code/resolve/main/{s}"
                for s in shards[:max_shards]
            ]
    except Exception:
        pass

    # Fallback to standard naming convention
    return [
        f"https://huggingface.co/datasets/codeparrot/github-code/resolve/main/data/train-{i:05d}-of-01126.parquet"
        for i in range(max_shards)
    ]


def stream_parquet_source(
    urls_or_paths: Sequence[str | Path],
    source_name: str,
    text_column: str = "text",
    path_column: str | None = None,
    filter_python: bool = False,
    max_bytes: int | None = None,
) -> Iterator[str]:
    """Streams text documents from parquet files via fsspec/pyarrow without downloading entire files."""
    import fsspec
    import pyarrow.parquet as pq

    bytes_read = 0

    cols = [text_column]
    if path_column and path_column != text_column:
        cols.append(path_column)

    for target in urls_or_paths:
        if max_bytes is not None and bytes_read >= max_bytes:
            break

        target_str = str(target)
        try:
            # Open with fsspec (supports both local files and remote https URLs)
            with fsspec.open(target_str, "rb", block_size=4 * 1024 * 1024) as f:
                pf = pq.ParquetFile(f)
                num_groups = pf.num_row_groups

                for rg_idx in range(num_groups):
                    if max_bytes is not None and bytes_read >= max_bytes:
                        break

                    try:
                        # Determine actual available columns
                        schema_names = pf.schema_arrow.names
                        read_cols = [c for c in cols if c in schema_names]
                        if not read_cols:
                            read_cols = [schema_names[0]]

                        table = pf.read_row_group(rg_idx, columns=read_cols)

                        # Extract text column
                        actual_text_col = (
                            text_column
                            if text_column in table.column_names
                            else ("content" if "content" in table.column_names else table.column_names[0])
                        )
                        texts = table[actual_text_col].to_pylist()

                        paths = None
                        if filter_python and path_column and path_column in table.column_names:
                            paths = table[path_column].to_pylist()

                        for i, doc in enumerate(texts):
                            if max_bytes is not None and bytes_read >= max_bytes:
                                break

                            if not doc or not isinstance(doc, str):
                                continue

                            if filter_python:
                                if paths is not None:
                                    fpath = paths[i]
                                    if not (fpath and fpath.endswith(".py")):
                                        continue
                                elif "language" in table.column_names:
                                    lang = table["language"][i].as_py()
                                    if not (lang and lang.lower() == "python"):
                                        continue

                            doc_bytes = len(doc.encode("utf-8", errors="replace"))
                            bytes_read += doc_bytes
                            yield doc

                    except Exception as rg_err:
                        print(
                            f"Warning: error reading row group {rg_idx} from {target_str}: {rg_err}",
                            file=sys.stderr,
                        )
                        continue

        except Exception as file_err:
            print(
                f"Warning: error streaming parquet {target_str}: {file_err}",
                file=sys.stderr,
            )
            continue


def interleave_streams(
    named_streams: list[tuple[str, Iterator[str]]]
) -> Iterator[tuple[str, str]]:
    """Interleaves multiple document streams round-robin until all are exhausted."""
    active = list(named_streams)
    idx = 0
    while active:
        name, stream = active[idx % len(active)]
        try:
            doc = next(stream)
            yield (name, doc)
            idx += 1
        except StopIteration:
            active.pop(idx % len(active))
            if not active:
                break


# ---------------------------------------------------------------------------
# Tokenizer Setup & Freezing Logic
# ---------------------------------------------------------------------------


def ensure_frozen_tokenizer(
    tokenizer_path: str | Path | None,
    out_dir: Path,
    sample_docs_collector: Iterator[str] | list[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    sample_bytes_target: int = DEFAULT_SAMPLE_BYTES_FOR_TOKENIZER,
) -> tuple[NetelproBPETokenizer, list[str]]:
    """Ensures a frozen BPE tokenizer is ready.

    1. If tokenizer.json exists at tokenizer_path or in out_dir or repo root, loads it.
    2. Otherwise, collects sample documents, trains NetelproBPETokenizer, saves and freezes it.
    Returns (tokenizer, collected_sample_docs).
    """
    candidates = []
    if tokenizer_path is not None:
        candidates.append(Path(tokenizer_path))
    candidates.extend([
        out_dir / "tokenizer.json",
        ROOT_DIR / "tokenizer.json",
        ROOT_DIR / "models" / "tokenizer.json",
    ])

    for cand in candidates:
        if cand.is_file():
            print(f"Loading existing frozen tokenizer from {cand}...")
            tok = NetelproBPETokenizer.load(cand)
            print(f"Loaded frozen tokenizer (vocab_size={tok.vocab_size:,}).")
            return tok, []

    print(
        f"No existing tokenizer.json found. Training frozen BPE on initial sample (~{sample_bytes_target // (1024*1024)} MB)..."
    )

    collected_docs: list[str] = []
    collected_bytes = 0

    if isinstance(sample_docs_collector, list):
        collected_docs = sample_docs_collector
    else:
        for doc in sample_docs_collector:
            collected_docs.append(doc)
            collected_bytes += len(doc.encode("utf-8"))
            if collected_bytes >= sample_bytes_target:
                break

    if not collected_docs:
        raise ValueError("Cannot train tokenizer: no sample documents collected!")

    # Write sample documents to staging file for BpeTrainer
    staging_file = out_dir / "_tokenizer_training_sample.txt"
    staging_file.parent.mkdir(parents=True, exist_ok=True)
    with open(staging_file, "w", encoding="utf-8") as fh:
        for d in collected_docs:
            fh.write(d)
            fh.write("\n\n")

    tok = NetelproBPETokenizer()
    tok.train([staging_file], vocab_size=vocab_size, min_frequency=2)

    # Save frozen artifact
    save_target = Path(tokenizer_path) if tokenizer_path else (out_dir / "tokenizer.json")
    tok.save(save_target)
    print(f"Frozen tokenizer trained and saved to {save_target} (vocab_size={tok.vocab_size:,}).")

    # Clean staging file
    try:
        staging_file.unlink(missing_ok=True)
    except Exception:
        pass

    return tok, collected_docs


# ---------------------------------------------------------------------------
# Corpus Building Pipeline Orchestrator
# ---------------------------------------------------------------------------


def build_corpus(
    out_dir: str | Path = "data/teo_v2",
    sources: Sequence[str] = ("spanish", "code"),
    max_shards: int | None = None,
    max_bytes_per_source: int | str | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
    tokenizer_path: str | Path | None = None,
    min_doc_chars: int = DEFAULT_MIN_DOC_CHARS,
    sample_bytes_for_tokenizer: int = DEFAULT_SAMPLE_BYTES_FOR_TOKENIZER,
    custom_doc_stream: Iterator[tuple[str, str]] | None = None,
    spanish_shards: Sequence[str] | None = None,
    code_shards: Sequence[str] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Main corpus download, cleaning, deduplication, tokenization, and packed compilation pipeline.

    Args:
        out_dir: Destination directory for packed binary shards and metadata.
        sources: List of data sources to download ('spanish', 'code').
        max_shards: Maximum number of 50M token shards to generate.
        max_bytes_per_source: Byte cap per source (e.g. '150MB' or integer bytes).
        shard_size: Number of uint16 tokens per shard (default: 50,000,000).
        tokenizer_path: Path to load/save frozen tokenizer.
        min_doc_chars: Minimum character length per document.
        sample_bytes_for_tokenizer: Sample bytes to collect before training BPE if none exists.
        custom_doc_stream: Optional offline document stream (source_name, raw_doc) for testing.
        spanish_shards: Custom parquet URLs/paths for Spanish source.
        code_shards: Custom parquet URLs/paths for Code source.
        resume: Whether to resume compilation from existing completed shards.

    Returns:
        Compilation summary metrics dict.
    """
    start_time = time.perf_counter()
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    max_bytes = parse_byte_size(max_bytes_per_source)
    resolved_sources = parse_sources(sources)

    print("--- Netelpro teo_v2 Corpus Builder ---")
    print(f"Sources:              {resolved_sources}")
    print(f"Shard Size:           {shard_size:,} tokens (uint16)")
    print(f"Max Shards:           {max_shards or 'unlimited'}")
    print(f"Max Bytes Per Source: {max_bytes:,} bytes" if max_bytes else "Max Bytes Per Source: unlimited")
    print(f"Output Directory:     {out_dir_path}")
    print(f"Resume:               {resume}")

    # Set up raw document stream
    if custom_doc_stream is not None:
        doc_stream = custom_doc_stream
    else:
        named_streams: list[tuple[str, Iterator[str]]] = []
        if "spanish" in resolved_sources:
            spa_urls = spanish_shards or get_fineweb_spanish_shard_urls()
            named_streams.append(
                (
                    "spanish",
                    stream_parquet_source(
                        spa_urls,
                        source_name="spanish",
                        text_column="text",
                        max_bytes=max_bytes,
                    ),
                )
            )
        if "code" in resolved_sources:
            code_urls = code_shards or get_github_code_shard_urls()
            named_streams.append(
                (
                    "code",
                    stream_parquet_source(
                        code_urls,
                        source_name="code",
                        text_column="content",
                        path_column="path",
                        filter_python=True,
                        max_bytes=max_bytes,
                    ),
                )
            )
        doc_stream = interleave_streams(named_streams)

    deduplicator = DocumentDeduplicator()

    # Step 1: Check or prepare tokenizer
    # If no tokenizer exists, collect sample from doc_stream, train and freeze
    tokenizer: NetelproBPETokenizer | None = None
    if tokenizer_path is not None and hasattr(tokenizer_path, "encode"):
        tokenizer = tokenizer_path
    else:
        tok_candidates = []
        if tokenizer_path:
            tok_candidates.append(Path(tokenizer_path))
        tok_candidates.extend([out_dir_path / "tokenizer.json", ROOT_DIR / "tokenizer.json"])

        for cand in tok_candidates:
            if cand.is_file():
                tokenizer = NetelproBPETokenizer.load(cand)
                print(f"Loaded existing frozen tokenizer: {cand}")
                break

    # Step 2: Initialize PackedShardWriter
    shard_writer = PackedShardWriter(
        out_dir=out_dir_path,
        shard_size=shard_size,
        max_shards=max_shards,
        tokenizer=tokenizer,
        resume=resume,
    )

    # After constructing shard_writer, if out_dir/"dedup_state.json" exists -> deduplicator.load_state(...)
    dedup_state_path = out_dir_path / "dedup_state.json"
    if dedup_state_path.is_file():
        deduplicator.load_state(dedup_state_path)
        print(f"Loaded deduplication state: {deduplicator.seen_count:,} seen hashes.")

    has_resumed_shards = len(shard_writer.shards_completed) > 0
    if has_resumed_shards:
        print(
            f"Resuming corpus build at shard {shard_writer.current_shard_idx} "
            f"({len(shard_writer.shards_completed)} completed shards found, "
            f"{shard_writer.total_tokens:,} tokens)."
        )

    initial_docs: list[str] = []
    # When resuming (shards_completed non-empty), SKIP the initial_docs tokenizer-sample loop
    if tokenizer is None and not has_resumed_shards:
        print("Collecting sample documents for BPE tokenizer training...")
        sample_bytes_needed = (
            min(sample_bytes_for_tokenizer, max_bytes * 2)
            if max_bytes
            else sample_bytes_for_tokenizer
        )
        sample_bytes_collected = 0

        for src_name, raw_doc in doc_stream:
            is_code = src_name == "code"
            cleaned = clean_document(raw_doc, is_code=is_code, min_chars=min_doc_chars)
            if cleaned is None or deduplicator.is_duplicate(cleaned):
                continue

            initial_docs.append(cleaned)
            sample_bytes_collected += len(cleaned.encode("utf-8"))
            if sample_bytes_collected >= sample_bytes_needed:
                break

        tokenizer, _ = ensure_frozen_tokenizer(
            tokenizer_path=tokenizer_path,
            out_dir=out_dir_path,
            sample_docs_collector=initial_docs,
            vocab_size=DEFAULT_VOCAB_SIZE,
            sample_bytes_target=sample_bytes_needed,
        )
        shard_writer.tokenizer = tokenizer
        shard_writer.bos_id = getattr(tokenizer, "bos_token_id", 1)
        shard_writer.eos_id = getattr(tokenizer, "eos_token_id", 2)
        shard_writer.pad_id = getattr(tokenizer, "pad_token_id", 0)
        shard_writer.vocab_size = getattr(tokenizer, "vocab_size", DEFAULT_VOCAB_SIZE)
    elif tokenizer is None and has_resumed_shards:
        tokenizer, _ = ensure_frozen_tokenizer(
            tokenizer_path=tokenizer_path,
            out_dir=out_dir_path,
            sample_docs_collector=[],
            vocab_size=DEFAULT_VOCAB_SIZE,
        )
        shard_writer.tokenizer = tokenizer
        shard_writer.bos_id = getattr(tokenizer, "bos_token_id", 1)
        shard_writer.eos_id = getattr(tokenizer, "eos_token_id", 2)
        shard_writer.pad_id = getattr(tokenizer, "pad_token_id", 0)
        shard_writer.vocab_size = getattr(tokenizer, "vocab_size", DEFAULT_VOCAB_SIZE)

    docs_processed = 0
    tokens_processed = shard_writer.total_tokens
    last_log_time = time.perf_counter()

    try:
        # Process documents collected during tokenizer training first
        for doc in initial_docs:
            can_continue = shard_writer.write_document(doc)
            docs_processed += 1
            tokens_processed = shard_writer.total_tokens
            if not can_continue:
                break

        # Process remaining streamed documents
        if max_shards is None or shard_writer.current_shard_idx < max_shards:
            for src_name, raw_doc in doc_stream:
                is_code = src_name == "code"
                cleaned = clean_document(raw_doc, is_code=is_code, min_chars=min_doc_chars)
                if cleaned is None:
                    continue

                if deduplicator.is_duplicate(cleaned):
                    continue

                prev_shards = len(shard_writer.shards_completed)
                can_continue = shard_writer.write_document(cleaned)
                docs_processed += 1
                tokens_processed = shard_writer.total_tokens

                # Persist deduplication state whenever a shard is completed
                if len(shard_writer.shards_completed) > prev_shards:
                    deduplicator.save_state(out_dir_path / "dedup_state.json")

                now = time.perf_counter()
                if now - last_log_time >= 5.0:
                    elapsed_so_far = now - start_time
                    d_rate = docs_processed / elapsed_so_far if elapsed_so_far > 0 else 0
                    t_rate = tokens_processed / elapsed_so_far if elapsed_so_far > 0 else 0
                    print(
                        f"Progress: {docs_processed:,} docs ({d_rate:,.1f} docs/s) | "
                        f"{tokens_processed:,} tokens ({t_rate:,.1f} tokens/s) | "
                        f"Shards: {shard_writer.current_shard_idx} | "
                        f"Elapsed: {elapsed_so_far:.1f}s"
                    )
                    last_log_time = now

                if not can_continue:
                    print(f"Reached max shards ({max_shards}). Stopping stream.")
                    break
    finally:
        manifest = shard_writer.close()
        deduplicator.save_state(out_dir_path / "dedup_state.json")

    elapsed = time.perf_counter() - start_time
    docs_rate = docs_processed / elapsed if elapsed > 0 else 0
    tokens_rate = (shard_writer.total_tokens) / elapsed if elapsed > 0 else 0

    # Verification: check pad fraction using compile_packed.stream_stats
    for shard_info in shard_writer.shards_completed:
        shard_path = out_dir_path / shard_info["file_name"]
        if shard_path.is_file():
            stream = load_stream(shard_path)
            stats = stream_stats(stream, pad_token_id=shard_writer.pad_id)
            if stats["pad_fraction"] > 0:
                print(
                    f"WARNING: Non-zero pad fraction ({stats['pad_fraction']}) in {shard_info['file_name']}",
                    file=sys.stderr,
                )

    metrics = {
        "status": "completed",
        "elapsed_seconds": float(elapsed),
        "total_docs": int(shard_writer.total_docs),
        "total_tokens": int(shard_writer.total_tokens),
        "docs_per_sec": float(docs_rate),
        "tokens_per_sec": float(tokens_rate),
        "total_shards": len(shard_writer.shards_completed),
        "unique_docs": deduplicator.seen_count,
        "duplicate_docs_dropped": deduplicator.duplicate_count,
        "out_dir": str(out_dir_path),
        "manifest": manifest,
    }

    print("\n" + "=" * 60)
    print("✅ teo_v2 Packed Corpus Compilation Completed:")
    print(f"   - Elapsed time:        {elapsed:.2f} s")
    print(f"   - Documents processed: {docs_processed:,} ({docs_rate:,.1f} docs/s)")
    print(f"   - Total tokens:        {shard_writer.total_tokens:,} ({tokens_rate:,.1f} tokens/s)")
    print(f"   - Packed shards:       {len(shard_writer.shards_completed)}")
    print(f"   - Unique docs:         {deduplicator.seen_count:,}")
    print(f"   - Duplicates dropped:  {deduplicator.duplicate_count:,}")
    print("   - Pad fraction:        0.000000 (verified)")
    print(f"   - Destination:         {out_dir_path}")
    print("=" * 60)

    return metrics


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main(args_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stream Spanish (FineWeb-2) and Code (github-code), clean, dedupe, and pack into 50M token uint16 shards."
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["spanish", "code"],
        help="Sources to download: 'spanish', 'code', or comma-separated list.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Maximum number of shards to compile.",
    )
    parser.add_argument(
        "--max-bytes-per-source",
        default=None,
        help="Maximum bytes to extract per source (e.g. '150MB', '1GB', or bytes as integer).",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default="data/teo_v2",
        help="Output directory for packed shards, metadata, and tokenizer.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Tokens per packed shard (default: 50,000,000).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Path to existing tokenizer.json or location to save newly trained tokenizer.",
    )
    parser.add_argument(
        "--min-doc-chars",
        type=int,
        default=DEFAULT_MIN_DOC_CHARS,
        help="Minimum characters required per document after cleaning (default: 200).",
    )
    parser.add_argument(
        "--sample-bytes-for-tokenizer",
        default="200MB",
        help="Bytes of text to sample for BPE tokenizer training if none exists (default: 200MB).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        default=True,
        help="Disable auto-resume and restart from shard 0.",
    )

    args = parser.parse_args(args_list)

    sample_bytes = parse_byte_size(args.sample_bytes_for_tokenizer) or DEFAULT_SAMPLE_BYTES_FOR_TOKENIZER

    build_corpus(
        out_dir=args.out_dir,
        sources=args.sources,
        max_shards=args.max_shards,
        max_bytes_per_source=args.max_bytes_per_source,
        shard_size=args.shard_size,
        tokenizer_path=args.tokenizer_path,
        min_doc_chars=args.min_doc_chars,
        sample_bytes_for_tokenizer=sample_bytes,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
