"""Compilador de Corpus a Formato Binario Nativo Netelpro (.bin).

Convierte el corpus de texto estructurado a un búfer binario puro (uint16 contiguo).
La red neuronal lee directamente este bloque de memoria sin parsear strings,
json ni caracteres humanos, operando a la velocidad del bus de silicio.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from netelpro.neuro.tokenizer import NetelproTokenizer


def compile_corpus_to_bin(
    jsonl_path: str | Path,
    bin_path: str | Path,
    block_size: int = 384,
) -> tuple[int, int, int]:
    """Compila cada muestra a una secuencia binaria fija de enteros uint16 (2 bytes por token).
    
    Retorna: (num_samples, block_size, total_bytes)
    """
    jsonl_file = Path(jsonl_path)
    bin_file = Path(bin_path)
    bin_file.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = NetelproTokenizer()
    pad_id = tokenizer.pad_token_id

    samples_count = 0
    total_tokens = 0

    with open(jsonl_file, "r", encoding="utf-8") as f_in, open(bin_file, "wb") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            tokens = tokenizer.encode(data["prompt"], add_special_tokens=True)

            # Ajustar exactamente al block_size + 1 (para x e y)
            seq_len = block_size + 1
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            else:
                tokens = tokens + [pad_id] * (seq_len - len(tokens))

            # Escribir en binario puro: formato unsigned short ('H' = 16 bits = 2 bytes)
            packed = struct.pack(f"<{len(tokens)}H", *tokens)
            f_out.write(packed)

            samples_count += 1
            total_tokens += len(tokens)

    total_bytes = bin_file.stat().st_size
    return samples_count, block_size, total_bytes


if __name__ == "__main__":
    src = Path(__file__).parent / "mega_train.jsonl"
    dst = Path(__file__).parent / "mega_train.bin"
    n_samples, b_size, n_bytes = compile_corpus_to_bin(src, dst, block_size=384)
    print("✅ Compilación binaria completada:")
    print(f"   - Muestras procesadas: {n_samples}")
    print(f"   - Tamaño de bloque: {b_size} tokens")
    print(f"   - Archivo binario: {dst} ({n_bytes:,} bytes / {n_bytes / 1024:.2f} KB)")
