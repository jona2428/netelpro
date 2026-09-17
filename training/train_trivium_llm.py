"""Entrenamiento Especializado del Trívium en el Netelpro Mini LLM.

Realiza un fine-tuning del modelo preentrenado sobre el corpus unificado
(razonamiento formal + retórica + dialéctica + detección de falacias),
fortaleciendo su capacidad de análisis en ciencias sociales y pensamiento crítico.
"""

from __future__ import annotations

import json
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


def train_trivium():
    print("=" * 75)
    print("🏛️ ENTRENAMIENTO DEL TRÍVIUM: CIENCIAS SOCIALES, RETÓRICA Y DIALÉCTICA")
    print("Elevando el Netelpro Mini LLM con razonamiento crítico humanista y silicio formal")
    print("=" * 75)

    model_dir = Path("models/netelpro_mini_v1")
    if not model_dir.exists():
        print("Cargando arquitectura base...")
        minillm = NetelproMiniLLM()
    else:
        print(f"📦 Cargando checkpoint previo desde: {model_dir}...")
        minillm = NetelproMiniLLM.from_pretrained(model_dir)

    tokenizer = minillm.tokenizer

    # Prepare combined corpus
    trivium_path = Path("training/data/trivium_train.jsonl")
    base_path = Path("training/data/mini_llm_train.jsonl")

    combined_samples = []
    for p in (base_path, trivium_path):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        combined_samples.append(json.loads(line))

    combined_corpus_path = Path("training/data/combined_trivium_train.jsonl")
    with open(combined_corpus_path, "w", encoding="utf-8") as f:
        for s in combined_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"📚 Dataset combinado: {len(combined_samples)} ejemplos de entrenamiento.")

    inputs, targets = load_dataset(combined_corpus_path, tokenizer, max_len=minillm.config.block_size)

    optimizer = optim.AdamW(minillm.parameters(), lr=8e-4, weight_decay=1e-2)
    epochs = 8
    batch_size = 16
    n_batches = len(inputs) // batch_size
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n⚡ Iniciando entrenamiento de humanidades y retórica en CPU...")
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

    print(f"\n💾 Guardando modelo actualizado en: {model_dir}...")
    minillm.save_pretrained(model_dir)
    print("✅ Checkpoint actualizado exitosamente con la sabiduría del Trívium.")

    # Demonstration prompt
    print("\n" + "=" * 75)
    print("🔮 PRUEBA DE EVALUACIÓN DE FALACIAS:")
    test_p = "<|user|>\nAnaliza este argumento: 'O estás conmigo o estás contra el país'.\n<|assistant|>\n"
    resp, cert = minillm.generate_text(test_p, max_new_tokens=45, temperature=0.6)
    print(f"Prompt: {test_p.strip()}")
    print(f"Respuesta:\n{resp.strip()}")
    print(f"📜 Certificado Emitido en {cert.total_latency_us:.2f} µs")
    print("=" * 75)


if __name__ == "__main__":
    train_trivium()
