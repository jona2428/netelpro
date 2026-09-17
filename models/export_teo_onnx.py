"""
Exportador de Teo v1 a Formato ONNX Optimizado para DirectML (iGPU UMA AMD Radeon).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from netelpro.neuro.minillm import NetelproMiniLLM


class TeoFastInferenceWrapper(nn.Module):
    """Grafo de inferencia directo para acelerador DirectML en iGPU."""

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
            z_scaled = z * 1000.0
            gate_mask = ((z_scaled >= -5000.0) & (z_scaled <= 5000.0)).to(z.dtype)
            h = torch.relu(z) * gate_mask
            h2 = block.mlp.c_proj(h)
            x = x + block.mlp.dropout(h2)

        x = self.net.ln_f(x)
        logits = self.net.lm_head(x)
        return logits


def export_teo() -> None:
    model_dir = Path(__file__).parent / "teo_v1"
    onnx_out = model_dir / "teo_v1.onnx"

    print("=" * 80)
    print(f"📦 Cargando checkpoint de Teo v1 desde {model_dir}...")
    llm = NetelproMiniLLM.from_pretrained(model_dir)
    llm.eval()

    print("⚙️ Construyendo wrapper de inferencia pura en silicio...")
    wrapper = TeoFastInferenceWrapper(llm.model)
    wrapper.eval()

    dummy_input = torch.tensor([[10, 25, 42, 88]], dtype=torch.long)

    print(f"🚀 Exportando a {onnx_out}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_out),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

    print("✅ Exportación completada exitosamente:")
    print(f"   • Archivo ONNX: {onnx_out}")
    print(f"   • Tamaño: {onnx_out.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    export_teo()
