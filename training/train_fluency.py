"""Entrenamiento de Fluidez en Lenguaje Natural para el Netelpro Mini LLM.

Afina los pesos del modelo sobre el corpus de oraciones completas y diálogos
gramaticalmente fluidos en español, eliminando el balbuceo de caracteres.
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
from netelpro.neuro.ste import HAS_TORCH
from training.train_mini_llm import load_dataset

if HAS_TORCH:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR


def train_fluency():
    print("=" * 75)
    print("📖 ENTRENAMIENTO DE FLUIDEZ EN ESPAÑOL: NETELPRO MINI LLM")
    print("Enseñando estructuras oracionales humanas, coherencia y articulación limpia")
    print("=" * 75)

    model_dir = Path("models/netelpro_mini_v1")
    if not model_dir.exists():
        print("Cargando arquitectura base...")
        minillm = NetelproMiniLLM()
    else:
        print(f"📦 Cargando checkpoint previo desde: {model_dir}...")
        minillm = NetelproMiniLLM.from_pretrained(model_dir)

    tokenizer = minillm.tokenizer

    fluency_path = Path("training/data/fluency_train.jsonl")
    if not fluency_path.exists():
        from training.data.fluency_corpus import save_fluency_corpus
        save_fluency_corpus(fluency_path)

    print(f"📥 Cargando corpus de fluidez...")
    inputs, targets = load_dataset(fluency_path, tokenizer, max_len=minillm.config.block_size)
    print(f"📊 Muestras a entrenar: {len(inputs)}")

    optimizer = optim.AdamW(minillm.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 6
    batch_size = 32
    n_batches = len(inputs) // batch_size
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n⚡ Iniciando entrenamiento de articulación y fluidez en CPU...")
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
        print(f"  [Época {epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Latencia Silicio: {cert.total_latency_us:.1f} µs")

    t_total = time.perf_counter() - t_start
    print(f"\n⏱️ Entrenamiento completado en {t_total:.2f} segundos.")

    print(f"\n💾 Guardando modelo con fluidez perfeccionada en: {model_dir}...")
    minillm.save_pretrained(model_dir)
    print("✅ Checkpoint actualizado exitosamente.")

    # Verification generations
    print("\n" + "=" * 75)
    print("🔮 PRUEBA DE FLUIDEZ EN LENGUAJE NATURAL:")

    test_queries = [
        "¿Quién eres?",
        "¿Qué es una falacia lógica?",
        "¿Qué es la neurona Netelpro?",
    ]

    for q in test_queries:
        prompt = f"<|user|>\n{q}\n<|assistant|>\n"
        print(f"\n[Usuario]: {q}")
        print("[Netelpro Mini LLM]: ", end="", flush=True)
        for token_str, _ in minillm.stream_chat(prompt, max_new_tokens=40, temperature=0.3, top_k=3, repetition_penalty=1.2):
            print(token_str, end="", flush=True)
        print()

    print("=" * 75)


if __name__ == "__main__":
    train_fluency()
