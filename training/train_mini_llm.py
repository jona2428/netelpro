"""Entrenamiento y Preentrenamiento del Netelpro Mini LLM.

Carga el corpus curado de razonamiento y contratos, tokeniza los ejemplos,
optimiza los pesos mediante AdamW y el Straight-Through Estimator (STE),
y serializa el modelo preentrenado con soporte de checkpointing completo.
"""

from __future__ import annotations

import json
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
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR


def load_dataset(corpus_path: Path, tokenizer: NetelproTokenizer, max_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes JSONL dataset into input and target tensors padded to max_len."""
    inputs_list = []
    targets_list = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            full_text = item["prompt"]
            tokens = tokenizer.encode(full_text, add_special_tokens=True)

            if len(tokens) > max_len + 1:
                tokens = tokens[: max_len + 1]
            elif len(tokens) < max_len + 1:
                # Pad with pad_token_id
                tokens = tokens + [tokenizer.pad_token_id] * (max_len + 1 - len(tokens))

            inputs_list.append(tokens[:-1])
            targets_list.append(tokens[1:])

    x = torch.tensor(inputs_list, dtype=torch.long)
    y = torch.tensor(targets_list, dtype=torch.long)
    return x, y


def train_mini_llm(
    corpus_file: str | Path | None = None,
    output_dir: str | Path = "models/netelpro_mini_v1",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 2e-3,
) -> NetelproMiniLLM:
    print("=" * 75)
    print("🚀 PIPELINE DE ENTRENAMIENTO: NETELPRO MINI LLM (NEURO-SIMBÓLICO)")
    print("Entrenando arquitectura Transformer causal con compuertas en silicio LLVM + STE")
    print("=" * 75)

    corpus_path = Path(corpus_file) if corpus_file else Path(__file__).parent / "data" / "mini_llm_train.jsonl"
    if not corpus_path.exists():
        from training.data.mini_llm_corpus import save_corpus_file
        save_corpus_file(corpus_path)

    tokenizer = NetelproTokenizer()
    print(f"📚 Tamaño de Vocabulario: {tokenizer.vocab_size} tokens")

    block_size = 128
    config = NetelproTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=3,
        n_head=4,
        n_embd=64,
        dropout=0.05,
        scale_factor=1000.0,
        z_min=-5000,
        z_max=5000,
    )

    minillm = NetelproMiniLLM(config=config, tokenizer=tokenizer)
    total_params = sum(p.numel() for p in minillm.parameters())
    print(f"🧠 Parámetros Totales del Modelo: {total_params:,} pesos flotantes + Compuertas LLVM")

    # Load and encode data
    print(f"\n📥 Cargando corpus desde: {corpus_path.relative_to(Path(__file__).parent.parent)}")
    inputs, targets = load_dataset(corpus_path, tokenizer, max_len=block_size)
    print(f"📊 Total de Muestras de Entrenamiento: {len(inputs)}")

    optimizer = optim.AdamW(minillm.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n⚡ Iniciando entrenamiento en CPU...")
    n_batches = len(inputs) // batch_size
    t_start = time.perf_counter()

    minillm.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(inputs))
        shuffled_x = inputs[perm]
        shuffled_y = targets[perm]

        total_loss = 0.0
        for b in range(n_batches):
            bx = shuffled_x[b * batch_size : (b + 1) * batch_size]
            by = shuffled_y[b * batch_size : (b + 1) * batch_size]

            optimizer.zero_grad()
            logits, loss, cert = minillm(bx, targets=by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            print(f"  [Época {epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Latencia Silicio: {cert.total_latency_us:.1f} µs")

    t_total = time.perf_counter() - t_start
    print(f"\n⏱️ Entrenamiento completado en {t_total:.2f} segundos ({t_total/epochs:.2f}s por época).")

    # Save pretrained model
    out_path = Path(output_dir)
    print(f"\n💾 Guardando checkpoint preentrenado en: {out_path}...")
    minillm.save_pretrained(out_path)
    print("✅ Checkpoint guardado exitosamente (config.json, vocab.json, model.pt).")

    # Quick Verification Generation
    print("\n" + "=" * 75)
    print("🔮 PRUEBA DE GENERACIÓN POST-ENTRENAMIENTO:")
    test_prompt = "<|user|>\n¿Quién eres y cuál es tu arquitectura?\n<|assistant|>\n"
    response, cert = minillm.generate_text(test_prompt, max_new_tokens=40, temperature=0.6)
    print(f"Prompt: {test_prompt.strip()}")
    print(f"Respuesta:\n{response.strip()}")
    print(f"📜 Certificado Emitido en {cert.total_latency_us:.2f} µs | Neuronas Activas: {cert.records[0].active_neurons}/{cert.records[0].total_neurons}")
    print("=" * 75)

    return minillm


if __name__ == "__main__":
    train_mini_llm()
