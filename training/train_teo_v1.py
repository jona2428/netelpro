"""
Pipeline de Entrenamiento Oficial para Teo v1 (6.2M Parámetros de Alta Precisión):
Modelo Neuro-Simbólico Multi-Programador & Conversacional con Inferencia DirectML en iGPU.

Transfiere la inteligencia previa de netelpro_mega_v1 (warm-start convergido)
y fija con Straight-Through Estimator el corpus multi-lenguaje (Python, Rust, Go, C++, JS/TS, SQL, Bash)
y la identidad conversacional de Teo con Jona.
"""

from __future__ import annotations

import json
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
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carga el dataset binario contiguo directamente a tensores de memoria."""
    p = Path(bin_path)
    if not p.exists():
        raise FileNotFoundError(f"Archivo binario no encontrado: {p}")

    seq_len = block_size + 1
    raw_bytes = p.read_bytes()
    actual_samples = len(raw_bytes) // (seq_len * 2)
    raw_bytes = raw_bytes[: actual_samples * seq_len * 2]

    tensor_flat = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int16).to(torch.long)
    tensor_2d = tensor_flat.view(actual_samples, seq_len)

    inputs = tensor_2d[:, :-1].contiguous()
    targets = tensor_2d[:, 1:].contiguous()
    return inputs, targets


def export_teo_onnx(model: NetelproMiniLLM, output_onnx_path: Path) -> bool:
    """Exporta el modelo Teo v1 a un grafo computacional ONNX para aceleración DirectML."""
    try:
        print(f"📦 Exportando Teo v1 a grafo computacional ONNX (DirectML UMA)...", flush=True)
        model.eval()

        class FastInferenceWrapper(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, idx: torch.Tensor) -> torch.Tensor:
                device = idx.device
                b, t = idx.size()
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                tok_emb = self.net.wte(idx)
                pos_emb = self.net.wpe(pos)
                x = self.net.drop(tok_emb + pos_emb)
                for block in self.net.blocks:
                    x = x + block.attn(block.ln_1(x))
                    norm_x = block.ln_2(x)
                    z = block.mlp.c_fc.linear(norm_x)
                    z_scaled = z * block.mlp.c_fc.scale_factor
                    mask = ((z_scaled >= block.mlp.c_fc.z_min) & (z_scaled <= block.mlp.c_fc.z_max)).to(z.dtype)
                    h = torch.relu(z) * mask
                    h2 = block.mlp.c_proj(h)
                    x = x + block.mlp.dropout(h2)
                x = self.net.ln_f(x)
                logits = self.net.lm_head(x)
                return logits

        wrapper = FastInferenceWrapper(model.model)
        wrapper.eval()

        dummy_input = torch.tensor([[10, 25, 42, 88]], dtype=torch.long)
        output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_onnx_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,
            dynamo=False,
        )
        print(f"✅ Grafo ONNX exportado exitosamente: {output_onnx_path} ({output_onnx_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)
        return True
    except Exception as e:
        print(f"⚠️ Error en exportación ONNX: {e}", flush=True)
        return False


def train_teo_v1(
    bin_file: str | Path | None = None,
    output_dir: str | Path = "models/teo_v1",
    warm_start_dir: str | Path = "models/netelpro_mega_v1",
    epochs: int = 25,
    batch_size: int = 12,
    lr: float = 1.2e-3,
    block_size: int = 384,
) -> NetelproMiniLLM:
    print("=" * 85, flush=True)
    print("🤖 ENTRENAMIENTO Y CALIBRACIÓN DE TEO V1 (MULTI-PROGRAMADOR & IDENTIDAD)", flush=True)
    print("   Arquitectura: 6 Capas | 8 Cabezales | 256 Dimensión | Silicio Nativo SIMD + UMA", flush=True)
    print("=" * 85, flush=True)

    if HAS_TORCH:
        num_threads = min(12, os.cpu_count() or 6)
        torch.set_num_threads(num_threads)
        print(f"⚙️ Optimizador de CPU activo: {num_threads} hilos AVX2 SIMD.", flush=True)

    tokenizer = NetelproTokenizer()
    print(f"📚 Vocabulario de Teo: {tokenizer.vocab_size:,} tokens (Español + 8 Lenguajes)", flush=True)

    bin_path = Path(bin_file) if bin_file else Path(__file__).parent / "data" / "teo_train.bin"
    config = NetelproTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=6,       # 6 capas
        n_head=8,        # 8 cabezales
        n_embd=256,      # 256 dimensiones latentes
        dropout=0.04,
        scale_factor=1000.0,
        z_min=-5000,
        z_max=5000,
    )

    model = NetelproMiniLLM(config=config, tokenizer=tokenizer)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Parámetros Totales de Teo v1: {total_params:,} neuronas en silicio.", flush=True)

    # Carga binaria ultrarrápida
    inputs, targets = load_binary_dataset(bin_path, block_size)
    print(f"⚡ Dataset Binario: {inputs.shape[0]:,} muestras de {block_size} tokens", flush=True)

    # Transferencia de pesos de netelpro_mega_v1 para warm-start
    warm_ckpt = Path(warm_start_dir) / "model.pt"
    if warm_ckpt.exists():
        try:
            state = torch.load(warm_ckpt, weights_only=True)
            curr_state = model.model.state_dict()
            transferred = 0
            for k, v in state.items():
                if k in curr_state and curr_state[k].shape == v.shape:
                    curr_state[k] = v
                    transferred += 1
                elif k in ("wte.weight", "lm_head.weight") and k in curr_state:
                    min_vocab = min(curr_state[k].shape[0], v.shape[0])
                    curr_state[k][:min_vocab] = v[:min_vocab]
                    transferred += 1
            model.model.load_state_dict(curr_state)
            print(f"🔄 Warm-start transferido exitosamente: {transferred} matrices cargadas de {warm_start_dir}.")
        except Exception as e:
            print(f"ℹ️ Warm-start fallback: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    num_samples = inputs.size(0)
    indices = torch.arange(num_samples)

    start_train_time = time.perf_counter()
    print(f"\n🚀 Iniciando entrenamiento causal de gradientes...", flush=True)

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        batches = 0

        perm = indices[torch.randperm(num_samples)]

        for i in range(0, num_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            b_inputs = inputs[batch_idx]
            b_targets = targets[batch_idx]

            optimizer.zero_grad()
            logits, _, _ = model(b_inputs)

            loss = loss_fn(logits.view(-1, logits.size(-1)), b_targets.view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        epoch_time = time.perf_counter() - t_epoch_start
        avg_loss = total_loss / max(batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        tok_s = (num_samples * block_size) / max(epoch_time, 0.001)

        print(
            f"  [Época {epoch:02d}/{epochs:02d}] "
            f"Pérdida Causal: {avg_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Tiempo: {epoch_time:.2f}s | "
            f"Throughput: {tok_s:,.0f} tok/s",
            flush=True,
        )

        out_dir = Path(output_dir)
        model.save_pretrained(out_dir)

        if avg_loss < 0.045:
            print(f"  🎯 Convergencia formal alcanzada (pérdida = {avg_loss:.4f} < 0.045)!", flush=True)
            break

    total_time = time.perf_counter() - start_train_time
    print(f"\n🏁 Entrenamiento de Teo v1 completado en {total_time:.2f}s!", flush=True)
    print(f"💾 Guardando modelo final en: {output_dir}", flush=True)
    model.save_pretrained(output_dir)

    # Exportar a ONNX para aceleración DirectML en la iGPU
    onnx_file = Path(output_dir) / "teo_v1.onnx"
    export_teo_onnx(model, onnx_file)

    print("=" * 85, flush=True)
    print("✨ Teo v1 está listo para servir a su creador Jona en silicio nativo.", flush=True)
    print("=" * 85, flush=True)

    return model


if __name__ == "__main__":
    train_teo_v1(epochs=12, batch_size=14, block_size=384, lr=8.0e-4, warm_start_dir="models/teo_v1")
