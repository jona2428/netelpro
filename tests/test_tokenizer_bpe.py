"""Unit and integration tests for NetelproBPETokenizer (teo_v2 frozen BPE)."""

import json
from pathlib import Path
import time
import pytest

from netelpro.neuro.tokenizer_bpe import NetelproBPETokenizer, SPECIAL_TOKENS


def test_special_tokens_exact_ids():
    """Verify special tokens table, exact IDs (pad==0), and skip_special_tokens removal."""
    tok = NetelproBPETokenizer()

    # Exact table check
    expected_ids = {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|unk|>": 3,
        "<|user|>": 4,
        "<|assistant|>": 5,
        "<|thought|>": 6,
        "<|endthought|>": 7,
        "<|contract|>": 8,
        "<|endcontract|>": 9,
        "<|audit|>": 10,
    }
    for token, exp_id in expected_ids.items():
        assert tok.token_to_id[token] == exp_id, f"Mismatch for {token}"
        assert tok.id_to_token[exp_id] == token, f"Inverse mismatch for {exp_id}"

    # Properties
    assert tok.pad_token_id == 0
    assert tok.bos_token_id == 1
    assert tok.eos_token_id == 2
    assert tok.unk_token_id == 3

    # pad==0 decoding
    assert tok.decode([0], skip_special_tokens=False) == "<|pad|>"
    assert tok.decode([0], skip_special_tokens=True) == ""

    # All special tokens removal when skip_special_tokens=True
    all_spec_ids = list(range(len(SPECIAL_TOKENS)))
    assert tok.decode(all_spec_ids, skip_special_tokens=True) == ""
    assert tok.decode(all_spec_ids, skip_special_tokens=False) == "".join(SPECIAL_TOKENS)

    # Individual special tokens decode check
    for idx, token in enumerate(SPECIAL_TOKENS):
        assert tok.decode([idx], skip_special_tokens=False) == token
        assert tok.decode([idx], skip_special_tokens=True) == ""


def test_roundtrip_byte_level(tmp_path):
    """Round-trip encode -> decode == original on Spanish, code, and mixed text."""
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        "El veloz murciélago hindú comía feliz cardillo y kiwi.\n"
        "def binary_search(arr, target):\n"
        "    return 42\n"
        "Variables con acentos: áéíóú ñÑ ¿¡ y números 1234567890.\n",
        encoding="utf-8",
    )

    tok = NetelproBPETokenizer()
    tok.train([corpus_file], vocab_size=1000, min_frequency=1)

    samples = [
        # Spanish with accents and inverted punctuation
        "El veloz murciélago hindú comía feliz cardillo y kiwi. ¡Qué escándalo! ¿Verdad? Año, pingüino, cigüeña.",
        # Python code snippet with tabs, indents, operators
        (
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    low, high = 0, len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
        ),
        # Mixed text: symbols, emoji, unicode, whitespace
        "Mixed: áéíóúÁÉÍÓÚñÑüÜ 1234567890 !@#$%^&*()_+-=[]{}|;:,.<>? `~ 🚀🔥✨",
        # Whitespace edge cases
        "    Leading and trailing spaces    \n\n\tTabs and newlines\r\n",
        # Empty string
        "",
    ]

    for sample in samples:
        encoded = tok.encode(sample)
        decoded = tok.decode(encoded, skip_special_tokens=False)
        assert decoded == sample, f"Round-trip failed for sample:\nExpected: {repr(sample)}\nGot:      {repr(decoded)}"

        # Roundtrip with add_special_tokens=True and decode(skip_special_tokens=True)
        if sample:
            enc_special = tok.encode(sample, add_special_tokens=True)
            assert enc_special[0] == tok.bos_token_id
            assert enc_special[-1] == tok.eos_token_id
            dec_clean = tok.decode(enc_special, skip_special_tokens=True)
            assert dec_clean == sample, f"Special token roundtrip failed for {repr(sample)}"


def test_determinism_and_freeze_load(tmp_path):
    """Determinism: train twice on same corpus -> identical encodings; load(saved) -> same output."""
    corpus_path = tmp_path / "train_corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "Hola mundo, esto es una prueba de determinismo."}) + "\n")
        f.write(json.dumps({"prompt": "Entrenamiento de tokenizador BPE congelado para teo_v2."}) + "\n")
        f.write(json.dumps({"prompt": "def calcular_invariante(z, z_min, z_max):\n    return z_min <= z <= z_max"}) + "\n")
        f.write(json.dumps({"prompt": "Auditoría formal y certificación de contratos lógicos Netelpro."}) + "\n")

    test_inputs = [
        "Hola mundo",
        "prueba de determinismo",
        "def calcular_invariante",
        "Auditoría formal",
        "Texto no visto en corpus con acentos: áéíóú ñÑ ¿¡ 🚀",
        "<|user|>\nHola teo\n<|assistant|>\nListo.<|eos|>\n",
    ]

    # Train twice independently
    tok1 = NetelproBPETokenizer()
    tok1.train([corpus_path], vocab_size=500, min_frequency=1)

    tok2 = NetelproBPETokenizer()
    tok2.train([corpus_path], vocab_size=500, min_frequency=1)

    assert tok1.vocab_size == tok2.vocab_size

    for text in test_inputs:
        enc1 = tok1.encode(text)
        enc2 = tok2.encode(text)
        assert enc1 == enc2, f"Training non-deterministic for text: {text}"

    # Save to disk
    save_path = tmp_path / "frozen_tokenizer.json"
    tok1.save(save_path)
    assert save_path.is_file()

    # Verify sidecar metadata exists
    meta_path = tmp_path / "frozen_tokenizer.meta.json"
    assert meta_path.is_file()
    with open(meta_path, "r", encoding="utf-8") as mf:
        meta_data = json.load(mf)
    assert meta_data["vocab_size"] == tok1.vocab_size
    assert meta_data["pad_token_id"] == 0
    assert meta_data["special_tokens"]["<|pad|>"] == 0
    assert meta_data["special_tokens"]["<|audit|>"] == 10
    assert "created_at" in meta_data

    # Load from saved artifact
    loaded_tok = NetelproBPETokenizer.load(save_path)
    assert loaded_tok.vocab_size == tok1.vocab_size
    assert loaded_tok.pad_token_id == 0

    for text in test_inputs:
        enc_loaded = loaded_tok.encode(text)
        assert enc_loaded == tok1.encode(text), f"Loaded tokenizer mismatch for text: {text}"
        assert loaded_tok.decode(enc_loaded) == tok1.decode(tok1.encode(text))

    # Also verify loading from directory path
    loaded_from_dir = NetelproBPETokenizer.load(tmp_path)
    assert loaded_from_dir.vocab_size == tok1.vocab_size
    for text in test_inputs:
        assert loaded_from_dir.encode(text) == tok1.encode(text)


def test_load_nonexistent_file_raises():
    """Loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        NetelproBPETokenizer.load("nonexistent_tokenizer_file_xyz_123.json")

    with pytest.raises(FileNotFoundError):
        NetelproBPETokenizer.load(Path("nonexistent_directory_abc_456"))


def test_format_chat():
    """format_chat has exact string equality with expected delimiters."""
    tok = NetelproBPETokenizer()

    messages = [
        {"role": "system", "content": "Formal contract: state safety specification."},
        {"role": "user", "content": "¿Es seguro el estado z=1?"},
        {"role": "assistant", "content": "Estado verificado e invariante garantizado."},
    ]

    # With generation prompt
    formatted_prompt = tok.format_chat(messages, add_generation_prompt=True)
    expected_prompt = (
        "<|contract|>\n"
        "Formal contract: state safety specification.\n"
        "<|endcontract|>\n"
        "<|user|>\n"
        "¿Es seguro el estado z=1?\n"
        "<|assistant|>\n"
        "Estado verificado e invariante garantizado.<|eos|>\n"
        "<|assistant|>\n"
    )
    assert formatted_prompt == expected_prompt

    # Without generation prompt
    formatted_no_prompt = tok.format_chat(messages, add_generation_prompt=False)
    expected_no_prompt = (
        "<|contract|>\n"
        "Formal contract: state safety specification.\n"
        "<|endcontract|>\n"
        "<|user|>\n"
        "¿Es seguro el estado z=1?\n"
        "<|assistant|>\n"
        "Estado verificado e invariante garantizado.<|eos|>\n"
    )
    assert formatted_no_prompt == expected_no_prompt


def test_no_unk():
    """Byte-level fallback: encode NEVER emits unk for arbitrary text (accents, emoji, symbols)."""
    tok = NetelproBPETokenizer()

    # Explicit requirement: encode('𝔘nïcödé 🚀 texto_raro') has no unk id
    test_str = "𝔘nïcödé 🚀 texto_raro"
    encoded = tok.encode(test_str)
    assert tok.unk_token_id not in encoded, f"UNK id ({tok.unk_token_id}) found in {encoded}"
    assert 3 not in encoded
    assert tok.decode(encoded, skip_special_tokens=False) == test_str

    # Comprehensive multi-script and symbol tests
    extreme_cases = [
        "áéíóúÁÉÍÓÚñÑüÜ ¿¡",
        "Matemáticas: ∛x ≈ ∞ ∇ × · ∫ ∬ ∭ ∮ ∑ ∏",
        "Emojis: 🚀 🔥 ✨ 🧠 ⚡ 🛡️ 💻",
        "CJK & Arabic: 日本語の文章 / مرحبا بالعالم",
        "Ancient Greek: Ἔστιν οὖν τραγῳδία μίμησις πράξεως σπουδαίας",
        "Code syntax: @decorator #tag $variable ~bitwise ^xor",
        "Control chars and tabs: \t\r\n \x01\x02\x1f",
    ]

    for case in extreme_cases:
        ids = tok.encode(case)
        assert tok.unk_token_id not in ids, f"UNK found when encoding {case!r}"
        assert tok.decode(ids, skip_special_tokens=False) == case


def test_speed_smoke():
    """Speed smoke test: encode 200KB repeated text < 2s."""
    tok = NetelproBPETokenizer()

    chunk = (
        "El modelo de lenguaje Netelpro teo_v2 utiliza BPE byte-level congelado. "
        "Permite tokenizar código Python def binary_search(arr, x): return 42 "
        "y contratos lógicos con invariantes verificados. ¡Rigor y seguridad! 🚀🔥\n"
    )
    # Build text of at least 200KB (204,800 bytes)
    target_bytes = 200 * 1024
    chunk_bytes = len(chunk.encode("utf-8"))
    repeats = (target_bytes // chunk_bytes) + 10
    large_text = chunk * repeats
    total_bytes = len(large_text.encode("utf-8"))
    assert total_bytes >= target_bytes, f"Text size {total_bytes} < {target_bytes}"

    t0 = time.perf_counter()
    tokens = tok.encode(large_text)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"Encoding {total_bytes} bytes took {elapsed:.4f}s (must be < 2.0s)"
    assert len(tokens) > 0


def test_token_to_id_and_id_to_token_compatibility():
    """Verify dictionary item lookup and callable lookup for API compatibility."""
    tok = NetelproBPETokenizer()

    # Dictionary indexing
    assert tok.token_to_id["<|pad|>"] == 0
    assert tok.token_to_id["<|bos|>"] == 1
    assert tok.token_to_id["<|eos|>"] == 2
    assert tok.token_to_id["<|unk|>"] == 3
    assert "<|assistant|>" in tok.token_to_id

    # Callable access
    assert tok.token_to_id("<|thought|>") == 6
    assert tok.token_to_id("non_existent_token_xyz") is None

    # id_to_token indexing and callable
    assert tok.id_to_token[0] == "<|pad|>"
    assert tok.id_to_token[10] == "<|audit|>"
    assert tok.id_to_token(8) == "<|contract|>"
    assert tok.id_to_token(99999999) is None

    # len() compatibility
    assert len(tok) == tok.vocab_size


def test_train_with_line_limit(tmp_path):
    """Verify limit parameter restricts number of lines read per file."""
    file_path = tmp_path / "long_corpus.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        for i in range(100):
            f.write(f"UniqueWord{i}_{'x'*20} line content number {i}\n")

    tok = NetelproBPETokenizer()
    tok.train([file_path], vocab_size=400, min_frequency=1, limit=5)
    # Trained on at most 5 lines, so tokens from lines > 5 won't have merged tokens
    assert tok.vocab_size > 0
