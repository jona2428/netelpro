"""Tests for build_massive_corpus.py."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from netelpro.neuro.tokenizer_bpe import NetelproBPETokenizer
from training.data.compile_packed import load_stream, stream_stats
from training.data.build_massive_corpus import (
    ADVANCED_REASONING_SAMPLES,
    build_massive_corpus,
    format_qa_turn,
)


def test_format_qa_turn():
    """Verifies that format_qa_turn correctly formats reasoning thoughts and contract."""
    turn = format_qa_turn(
        prompt="¿Qué es Netelpro?",
        response="Es la arquitectura SBT en silicio.",
        thought="1. Explicar Netelpro.\n2. Destacar compuertas STE.",
        system_prompt="Regla: ser formal",
    )
    assert "<|contract|>\nRegla: ser formal\n<|endcontract|>" in turn
    assert "<|user|>\n¿Qué es Netelpro?" in turn
    assert "<|thought|>\n1. Explicar Netelpro.\n2. Destacar compuertas STE.\n<|endthought|>" in turn
    assert "<|assistant|>\nEs la arquitectura SBT en silicio." in turn


def test_reasoning_samples_encode_cleanly():
    """Verifies that advanced reasoning samples encode cleanly with 32k frozen tokenizer."""
    tok_path = ROOT_DIR / "data" / "teo_v2" / "tokenizer.json"
    assert tok_path.exists()

    tok = NetelproBPETokenizer.load(tok_path)
    for sample in ADVANCED_REASONING_SAMPLES:
        doc = format_qa_turn(sample["prompt"], sample["response"], sample.get("thought"))
        tokens = tok.encode(doc, add_special_tokens=True)
        assert len(tokens) > 50
        assert tokens[0] == tok.bos_token_id
        assert tokens[-1] == tok.eos_token_id
        assert tok.pad_token_id not in tokens


def test_build_massive_corpus_small(tmp_path: Path):
    """Verifies build_massive_corpus builds valid continuous packed binary shards in offline mode."""
    tok_path = ROOT_DIR / "data" / "teo_v2" / "tokenizer.json"
    out_dir = tmp_path / "massive_test"

    result = build_massive_corpus(
        out_dir=out_dir,
        tokenizer_path=tok_path,
        target_shards=1,
        shard_size=10_000,
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
