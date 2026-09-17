"""Tests for the balanced multi-domain dataset builder."""

from __future__ import annotations

from pathlib import Path
import sys


# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from netelpro.neuro.tokenizer_bpe import NetelproBPETokenizer
from training.data.compile_packed import load_stream, stream_stats
from training.data.build_balanced_dataset import (
    CURATED_NETELPRO_SYSTEMS,
    build_balanced_dataset,
    format_dialogue,
)


def test_format_dialogue():
    """Verifies that format_dialogue produces correctly structured Netelpro delimiters."""
    formatted = format_dialogue(
        prompt="wena teo",
        response="¡Wena, hermano!",
        thought="Saludar calurosamente.",
        system_prompt="Regla: ser honesto",
    )

    assert "<|contract|>\nRegla: ser honesto\n<|endcontract|>" in formatted
    assert "<|user|>\nwena teo" in formatted
    assert "<|thought|>\nSaludar calurosamente.\n<|endthought|>" in formatted
    assert "<|assistant|>\n¡Wena, hermano!" in formatted


def test_curated_systems_encoding():
    """Verifies that all curated Netelpro & systems items encode cleanly with the frozen tokenizer."""
    tok_path = ROOT_DIR / "data" / "teo_v2" / "tokenizer.json"
    assert tok_path.exists(), f"Frozen tokenizer not found at {tok_path}"

    tokenizer = NetelproBPETokenizer.load(tok_path)
    assert tokenizer.vocab_size == 32768

    for item in CURATED_NETELPRO_SYSTEMS:
        doc = format_dialogue(item["prompt"], item["response"], item.get("thought"))
        tokens = tokenizer.encode(doc, add_special_tokens=True)
        assert len(tokens) > 20
        assert tokens[0] == tokenizer.bos_token_id
        assert tokens[-1] == tokenizer.eos_token_id
        assert tokenizer.pad_token_id not in tokens


def test_build_balanced_dataset_small(tmp_path: Path):
    """Verifies that build_balanced_dataset creates valid continuous packed shards with zero pad."""
    tok_path = ROOT_DIR / "data" / "teo_v2" / "tokenizer.json"
    out_dir = tmp_path / "balanced_test"

    result = build_balanced_dataset(
        out_dir=out_dir,
        tokenizer_path=tok_path,
        target_shards=1,
        shard_size=10_000,  # Small shard for fast test
        persona_multiplier=5,
        resume=False,
        offline=True,
    )

    assert result["status"] == "completed"
    shards = sorted(out_dir.glob("shard_*.bin"))
    assert len(shards) >= 1

    stream = load_stream(shards[0])
    stats = stream_stats(stream)

    assert stats["pad_fraction"] == 0.0
    assert stats["total_tokens"] >= 10_000
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "tokenizer.json").exists()
