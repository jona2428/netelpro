"""
Balanced High-Density Master Dataset Builder for Teo v1.
Combines:
- 68 Foundational Textbooks (Logic, Trivium, Mathematics, Science, Psychology, Law)
- 17 Multi-Language Coder Samples (Python, TS/JS, C/C++, Rust, Go, SQL, Bash/PS, HTML/CSS)
- 6 Teo Conversational & Identity Samples (Jona recognition, philosophy, silicon self-awareness)

Produces a clean, balanced binary dataset without massive duplicate bloat,
allowing fast SIMD gradient convergence and sub-1GB RAM footprint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.data.multilang_coder_corpus import MULTILANG_CODER_SAMPLES
from training.data.teo_conversational_corpus import TEO_CONVERSATIONAL_SAMPLES
from training.data.english_conversational_corpus import ENGLISH_CONVERSATIONAL_SAMPLES
from training.data.synthetic_textbooks import generate_synthetic_textbooks
from training.data.compile_binary import compile_corpus_to_bin


def build_teo_balanced_corpus() -> list[dict[str, str]]:
    # 1. Base unique textbooks (take first unique 68 from the 20x expanded list)
    all_textbooks = generate_synthetic_textbooks()
    unique_textbooks = all_textbooks[:68]

    # 2. Multi-language coding samples
    coder_prompts = []
    for item in MULTILANG_CODER_SAMPLES:
        prompt = (
            f"<|user|>\n{item['prompt']}\n"
            f"<|assistant|>\n"
            f"<|thought|>\n{item['thought']}\n<|endthought|>\n"
            f"{item['response']}<|eos|>"
        )
        coder_prompts.append({"prompt": prompt})

    # 3. Teo Spanish conversational samples
    conv_prompts = []
    for item in TEO_CONVERSATIONAL_SAMPLES:
        prompt = (
            f"<|user|>\n{item['prompt']}\n"
            f"<|assistant|>\n"
            f"<|thought|>\n{item['thought']}\n<|endthought|>\n"
            f"{item['response']}<|eos|>"
        )
        conv_prompts.append({"prompt": prompt})

    # 4. Teo English conversational samples
    eng_prompts = []
    for item in ENGLISH_CONVERSATIONAL_SAMPLES:
        prompt = (
            f"<|user|>\n{item['prompt']}\n"
            f"<|assistant|>\n"
            f"<|thought|>\n{item['thought']}\n<|endthought|>\n"
            f"{item['response']}<|eos|>"
        )
        eng_prompts.append({"prompt": prompt})

    # Build balanced mix:
    # 68 textbooks x 3 = 204
    # 17 coder x 6 = 102
    # 6 conversational Spanish x 8 = 48
    # 7 conversational English x 8 = 56
    # Total = 410 samples per epoch
    corpus: list[dict[str, str]] = []
    for _ in range(3):
        corpus.extend(unique_textbooks)
    for _ in range(6):
        corpus.extend(coder_prompts)
    for _ in range(8):
        corpus.extend(conv_prompts)
    for _ in range(8):
        corpus.extend(eng_prompts)
    return corpus


def main() -> None:
    data_dir = Path(__file__).parent
    jsonl_out = data_dir / "teo_train.jsonl"
    bin_out = data_dir / "teo_train.bin"

    print("=" * 80)
    print("🧠 COMPILANDO DATASET BALANCEADO DE TEO V1 (MULTI-PROGRAMADOR & IDENTIDAD)")
    print("=" * 80)

    corpus = build_teo_balanced_corpus()
    print(f"📦 Total de muestras equilibradas: {len(corpus):,} bloques de alta densidad.")

    with open(jsonl_out, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Archivo JSONL: {jsonl_out} ({jsonl_out.stat().st_size / 1024:.1f} KB)")
    print("⚙️ Compilando a binario puro (.bin uint16, bloque 384)...")

    n_samples, b_size, n_bytes = compile_corpus_to_bin(jsonl_out, bin_out, block_size=384)
    print("🚀 Compilación binaria lista:")
    print(f"   • Muestras: {n_samples:,}")
    print(f"   • Bloque:   {b_size} tokens")
    print(f"   • Tamaño:   {n_bytes:,} bytes ({n_bytes / 1024:.1f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
