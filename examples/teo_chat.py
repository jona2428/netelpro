"""
Teo v1 — CLI Pulcro estilo Ollama.
Inferencia nativa con aceleración iGPU UMA (DirectML) y streaming en tiempo real.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Generator, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from netelpro.neuro.minillm import NetelproMiniLLM
from netelpro.neuro.tokenizer import NetelproTokenizer
from netelpro.neuro.memory import NetelproBinaryMemoryBank

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class TeoDirectMLEngine:
    """Motor de inferencia acelerado en silicio iGPU UMA vía DirectML."""

    def __init__(self, onnx_path: Path):
        self.onnx_path = str(onnx_path)
        providers = ort.get_available_providers()

        if "DmlExecutionProvider" in providers:
            self.provider = "DmlExecutionProvider"
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                self.onnx_path,
                sess_options=opts,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
            self.device_name = "AMD Radeon Vega 7 (DirectML UMA)"
        else:
            self.provider = "CPUExecutionProvider"
            self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
            self.device_name = "CPU SIMD (AVX2)"

    def stream_generate(
        self,
        tokenizer: NetelproTokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        stop_tokens: list[str] | None = None,
        repetition_penalty: float = 1.15,
        temperature: float = 0.0,
    ) -> Generator[str, None, None]:
        import numpy as np

        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        curr = list(tokens)
        stop_ids = {
            tokenizer.token_to_id[s]
            for s in (stop_tokens or ["<|eos|>", "<|user|>"])
            if s in tokenizer.token_to_id
        }
        stop_ids.add(tokenizer.eos_token_id)

        for _ in range(max_new_tokens):
            inp = np.array([curr[-384:]], dtype=np.int64)
            logits = self.session.run(["logits"], {"input_ids": inp})[0][0, -1, :].copy()

            # Penalización suave de repetición
            if repetition_penalty != 1.0 and len(curr) > len(tokens):
                recent = set(curr[-(min(32, len(curr) - len(tokens))):])
                for rid in recent:
                    if logits[rid] > 0:
                        logits[rid] /= repetition_penalty
                    else:
                        logits[rid] *= repetition_penalty

            if temperature > 0.0:
                logits = logits / max(temperature, 1e-4)
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / np.sum(exp_l)
                next_id = int(np.random.choice(len(probs), p=probs))
            else:
                next_id = int(np.argmax(logits))

            if next_id in stop_ids:
                break

            curr.append(next_id)
            tok_str = tokenizer.id_to_token.get(next_id, "")
            yield tok_str


def print_help():
    print("""
Comandos disponibles:
  /? , /help       Mostrar esta ayuda
  /think           Activar/desactivar visualización del razonamiento interno
  /clear           Limpiar el historial de conversación
  /stats           Ver telemetría del acelerador de silicio
  /remember <dato> Fijar un concepto en el banco de memoria binaria
  /memories        Listar hechos almacenados en memoria viva
  /forget          Limpiar el banco de memoria
  /bye , /exit     Salir de Teo
""")


def run_teo_terminal(model_dir: str = "models/teo_v1"):
    model_path = Path(model_dir)

    if not model_path.exists():
        print(f"Error: Modelo no encontrado en '{model_path}'.")
        return

    # Cargar tokenizer y modelo
    llm = NetelproMiniLLM.from_pretrained(model_path)
    memory_bank = llm.memory_bank

    # Comprobar acelerador DirectML
    onnx_file = model_path / "teo_v1.onnx"
    dml_engine = None
    device_name = "CPU SIMD (AVX2)"

    if onnx_file.exists() and HAS_ORT:
        try:
            dml_engine = TeoDirectMLEngine(onnx_file)
            device_name = dml_engine.device_name
        except Exception:
            pass

    # Banner pulcro estilo Ollama
    print(f"\033[1mTeo v1\033[0m ({device_name})")
    print("Send a message (/? for help)")
    print()

    conversation_history: list[dict[str, str]] = []
    show_thought = False
    last_tps = 0.0
    last_tokens_count = 0

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Teo.")
            break

        if not user_input:
            continue

        # Sanitizar emojis y caracteres no soportados para no inyectar <|unk|> en el silicio
        clean_user_input = re.sub(r'[\U00010000-\U0010ffff]', '', user_input)
        clean_user_input = re.sub(r'\s+', ' ', clean_user_input).strip()
        if not clean_user_input:
            clean_user_input = user_input

        # Comandos rápidos estilo Ollama
        cmd = user_input.lower()
        if cmd in ("/bye", "/exit", "/quit"):
            break

        if cmd in ("/?", "/help"):
            print_help()
            continue

        if cmd == "/think":
            show_thought = not show_thought
            status = "activado" if show_thought else "desactivado"
            print(f"Razonamiento interno: {status}\n")
            continue

        if cmd == "/clear":
            conversation_history.clear()
            print("Historial reiniciado.\n")
            continue

        if cmd == "/stats":
            print(f"Acelerador: {device_name} | Última velocidad: {last_tps:.1f} tok/s ({last_tokens_count} tokens)\n")
            continue

        if user_input.startswith("/remember "):
            fact = user_input[10:].strip()
            slot = llm.remember(fact)
            print(f"Memoria fijada en ranura #{slot}: '{fact}'\n")
            continue

        if cmd == "/memories":
            mems = memory_bank.list_memories()
            if not mems:
                print("Banco de memoria vacío.\n")
            else:
                for m in mems:
                    print(f"  #{m.get('slot')}: {m.get('concept')}")
                print()
            continue

        if cmd == "/forget":
            memory_bank.clear()
            print("Memorias vivas borradas.\n")
            continue

        # Construir prompt limpio (evita la retroalimentación de respuestas previas que desalinean la atención)
        mems = memory_bank.list_memories()
        if mems:
            mem_summary = " ".join(m.get("concept", "") for m in mems[-2:])
            base_prompt = (
                f"<|contract|>\nContexto: {mem_summary}\n<|endcontract|>\n"
                f"<|user|>\n{clean_user_input}\n<|assistant|>\n"
            )
        else:
            base_prompt = f"<|user|>\n{clean_user_input}\n<|assistant|>\n"

        if not base_prompt.startswith("<|bos|>"):
            base_prompt = "<|bos|>" + base_prompt

        # Streaming en tiempo real
        t0 = time.perf_counter()
        token_count = 0
        in_thought = False
        full_thought_parts = []
        full_answer_parts = []

        try:
            if dml_engine is not None:
                stream = dml_engine.stream_generate(
                    llm.tokenizer,
                    base_prompt,
                    max_new_tokens=220,
                    stop_tokens=["<|eos|>", "<|user|>"],
                )
            else:
                # Fallback generator
                def pt_stream():
                    text, _ = llm.generate_text(base_prompt, max_new_tokens=220, temperature=0.0)
                    for tok in text.split(" "):
                        yield tok + " "
                stream = pt_stream()

            for tok in stream:
                token_count += 1

                if tok == "<|thought|>":
                    in_thought = True
                    if show_thought:
                        sys.stdout.write("\033[90m💭 ")
                        sys.stdout.flush()
                    continue

                if tok == "<|endthought|>":
                    in_thought = False
                    if show_thought:
                        sys.stdout.write("\033[0m\n\n")
                        sys.stdout.flush()
                    continue

                if in_thought:
                    full_thought_parts.append(tok)
                    if show_thought:
                        sys.stdout.write(tok)
                        sys.stdout.flush()
                else:
                    if tok not in ("<|eos|>", "<|bos|>", "<|pad|>", "<|assistant|>", "<|user|>"):
                        if not full_answer_parts and tok.startswith("\n"):
                            tok = tok.lstrip("\n")
                        if tok:
                            full_answer_parts.append(tok)
                            sys.stdout.write(tok)
                            sys.stdout.flush()

            # Si el modelo no cerró <|endthought|>, no perder la respuesta generada
            if not full_answer_parts and full_thought_parts and not show_thought:
                fallback_txt = "".join(full_thought_parts).strip()
                for spec in ("<|eos|>", "<|bos|>", "<|pad|>", "<|assistant|>", "<|user|>", "<|thought|>", "<|endthought|>"):
                    fallback_txt = fallback_txt.replace(spec, "")
                if fallback_txt:
                    full_answer_parts.append(fallback_txt)
                    sys.stdout.write(fallback_txt)
                    sys.stdout.flush()

            sys.stdout.write("\n\n")
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n[Generación interrumpida]\n")

        t1 = time.perf_counter()
        dur = max(t1 - t0, 0.001)
        last_tps = token_count / dur
        last_tokens_count = token_count

        final_answer = "".join(full_answer_parts).strip()
        if final_answer:
            conversation_history.append({"role": "assistant", "content": final_answer})


if __name__ == "__main__":
    run_teo_terminal()
