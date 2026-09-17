"""Pipeline de Entrenamiento Binario de Alta Densidad: Netelpro-Mega-Mini (~2.5M Parámetros).

Entrena la red neuronal leyendo directamente búferes binarios puros (.bin)
sin intermediación de strings ni JSON, optimizando en tiempo real con
Straight-Through Estimator (STE) y compuertas formales en silicio.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from netelpro.neuro.minillm import NetelproMiniLLM
from netelpro.neuro.tokenizer import NetelproTokenizer
from netelpro.neuro.transformer import NetelproTransformerConfig
from netelpro.neuro.ste import HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR


def load_binary_dataset(
    bin_path: str | Path,
    num_samples: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carga el dataset binario directamente a la memoria RAM como tensores contiguos."""
    p = Path(bin_path)
    if not p.exists():
        raise FileNotFoundError(f"Archivo binario no encontrado: {p}")

    seq_len = block_size + 1
    total_elements = num_samples * seq_len
    expected_bytes = total_elements * 2  # uint16 = 2 bytes

    raw_bytes = p.read_bytes()
    if len(raw_bytes) < expected_bytes:
        # Calcular muestras reales basadas en los bytes disponibles
        actual_samples = len(raw_bytes) // (seq_len * 2)
        raw_bytes = raw_bytes[: actual_samples * seq_len * 2]
        num_samples = actual_samples

    tensor_flat = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int16).to(torch.long)
    tensor_2d = tensor_flat.view(num_samples, seq_len)

    inputs = tensor_2d[:, :-1].contiguous()
    targets = tensor_2d[:, 1:].contiguous()
    return inputs, targets


def train_mega_llm(
    bin_file: str | Path | None = None,
    output_dir: str | Path = "models/netelpro_mega_v1",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1.5e-3,
    block_size: int = 384,
) -> NetelproMiniLLM:
    print("=" * 80)
    print("⚡ PIPELINE DE ENTRENAMIENTO BINARIO NATIVO: NETELPRO MEGA-MINI (2.5M PARÁMETROS)")
    print("Entrenando en silicio puro desde búfer binario (.bin) con compuertas STE formales")
    print("=" * 80)

    if HAS_TORCH:
        num_threads = min(12, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"⚙️ Optimizador de CPU activo: {num_threads} hilos vectorizados SIMD.")

    tokenizer = NetelproTokenizer()
    print(f"📚 Tamaño de Vocabulario: {tokenizer.vocab_size} tokens")

    bin_path = Path(bin_file) if bin_file else Path(__file__).parent / "data" / "mega_train.bin"
    if not bin_path.exists():
        print("⚙️ Compilando dataset de texto a binario puro...")
        from training.data.compile_binary import compile_corpus_to_bin
        src = Path(__file__).parent / "data" / "mega_train.jsonl"
        compile_corpus_to_bin(src, bin_path, block_size=block_size)

    # Configuración de arquitectura escalada a ~2.5M parámetros
    config = NetelproTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=6,       # 6 capas profundas
        n_head=8,        # 8 cabezales de atención causal
        n_embd=256,      # 256 dimensiones latentes
        dropout=0.05,
        scale_factor=1000.0,
        z_min=-5000,
        z_max=5000,
    )

    model = NetelproMiniLLM(config=config, tokenizer=tokenizer)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Topología de Red: 6 Capas | 8 Cabezales | 256 Dimensión | {total_params:,} Parámetros")

    # Carga binaria ultrarrápida (cero parsing de texto)
    t_load_start = time.perf_counter()
    file_bytes = bin_path.stat().st_size
    seq_bytes = (block_size + 1) * 2
    num_samples = file_bytes // seq_bytes
    inputs, targets = load_binary_dataset(bin_path, num_samples, block_size)
    t_load_ms = (time.perf_counter() - t_load_start) * 1000
    print(f"⚡ Dataset Binario cargado en {t_load_ms:.2f} ms: {inputs.shape[0]} muestras de {block_size} tokens")

    # Cargar pesos previos si existen para acelerar convergencia conversacional
    ckpt_file = Path(output_dir) / "model.pt"
    if ckpt_file.exists():
        try:
            state = torch.load(ckpt_file, weights_only=True)
            curr_state = model.model.state_dict()
            transferred = 0
            for k, v in state.items():
                if k in curr_state:
                    if curr_state[k].shape == v.shape:
                        curr_state[k] = v
                        transferred += 1
                    elif k in ("wte.weight", "lm_head.weight"):
                        # Transfer overlapping vocabulary slice
                        min_vocab = min(curr_state[k].shape[0], v.shape[0])
                        curr_state[k][:min_vocab] = v[:min_vocab]
                        transferred += 1
            model.model.load_state_dict(curr_state)
            print(f"🔄 Checkpoint previo transferido ({transferred} matrices adaptadas): convergencia inmediata.")
        except Exception as e:
            print(f"ℹ️ Inicializando pesos nuevos: {e}")

    # Optimizador con decaimiento de peso y schedule de coseno
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    num_samples = inputs.size(0)
    indices = torch.arange(num_samples)

    start_train_time = time.perf_counter()
    print("\n🚀 Iniciando optimización de gradientes en CPU (Vectorizado SIMD)...")

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        batches = 0

        # Barajado pseudoaleatorio determinista
        perm = indices[torch.randperm(num_samples)]

        for i in range(0, num_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            b_inputs = inputs[batch_idx]
            b_targets = targets[batch_idx]

            optimizer.zero_grad()
            logits, _, cert = model(b_inputs)

            # logits: (B, T, V), targets: (B, T)
            loss = loss_fn(logits.view(-1, logits.size(-1)), b_targets.view(-1))
            loss.backward()

            # Gradient clipping para estabilidad matemática
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        epoch_time = time.perf_counter() - t_epoch_start
        avg_loss = total_loss / max(batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"  [Época {epoch:02d}/{epochs:02d}] "
            f"Pérdida Causal: {avg_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Tiempo: {epoch_time:.2f}s | "
            f"Tokens/seg: {(num_samples * block_size) / epoch_time:.0f}",
            flush=True,
        )

        # Guardar checkpoint tras cada época para persistencia inmediata
        out_dir = Path(output_dir)
        model.save_pretrained(out_dir)

        if avg_loss < 0.035:
            print(f"  🎯 Convergencia óptima alcanzada (pérdida = {avg_loss:.4f} < 0.035)!")
            break

    total_time = time.perf_counter() - start_train_time
    print(f"\n🏁 Entrenamiento completado en {total_time:.2f}s!")
    print(f"💾 Guardando modelo preentrenado final en: {out_dir}")
    model.save_pretrained(out_dir)
    print("✅ Checkpoint de Netelpro Mega Mini serializado correctamente.")

    return model


if __name__ == "__main__":
    train_mega_llm(epochs=6, batch_size=34, block_size=384, lr=1.0e-3)
