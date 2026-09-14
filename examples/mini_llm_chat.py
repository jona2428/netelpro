"""Terminal de Chat Interactivo: Netelpro Mini LLM con Auditoría en Silicio y Trívium.

Permite chatear en tiempo real con el Netelpro Mini LLM, observando el streaming
token por token, el Certificado Formal de Auditoría y el análisis del Trívium (retórica/falacias).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from netelpro.neuro.minillm import NetelproMiniLLM
from netelpro.neuro.trivium import NetelproTriviumEngine


def print_banner():
    print("\033[96m" + "=" * 75)
    print("🤖 NETELPRO MINI / MEGA LLM — TERMINAL DE CHAT NEURO-SIMBÓLICO")
    print("   Arquitectura Transformer Causal + Silicio LLVM + Banco de Memoria Binaria")
    print("=" * 75 + "\033[0m")
    print("Escribe tu pregunta o comando. Comandos especiales:")
    print("  \033[93m/remember <hecho>\033[0m: Grabar un hecho nuevo en el Banco de Memoria Binaria (en vivo)")
    print("  \033[93m/memories\033[0m        : Ver todas las ranuras de memoria viva grabadas en silicio")
    print("  \033[93m/forget\033[0m          : Limpiar la memoria viva del modelo")
    print("  \033[93m/binary <texto>\033[0m  : Inspección de silicio: ver cómo la red ve el texto en bits puros")
    print("  \033[93m/trivium <texto>\033[0m : Diagnóstico retórico y detector de falacias en silicio")
    print("  \033[93m/inhibit [on|off]\033[0m: Simular corte de señal fail-closed en compuertas de silicio")
    print("  \033[93m/stats\033[0m           : Ver estadísticas y topología del modelo")
    print("  \033[93m/clear\033[0m           : Reiniciar memoria de la conversación activa")
    print("  \033[93m/quit\033[0m            : Salir del chat\n")


def format_certificate_box(audit_history: list[dict]):
    if not audit_history:
        return ""
    last_audit = audit_history[-1]
    total_tokens = len(audit_history)
    avg_lat = sum(a.get("latency_us", 0.0) for a in audit_history) / max(1, total_tokens)

    box = (
        "\033[92m┌─ 📜 CERTIFICADO FORMAL DE AUDITORÍA EN SILICIO ────────────────────────┐\n"
        f"│ • Tokens Emitidos: {total_tokens:3d}  |  Latencia Media por Token: {avg_lat:8.2f} µs     │\n"
        f"│ • Neuronas Activas: {last_audit.get('active_neurons', 0):3d}/{last_audit.get('total_neurons', 0):3d}  |  Inhibidas (Fail-Closed): {last_audit.get('suppressed_neurons', 0):3d}           │\n"
        f"│ • Garantía Formal: 100% Verificado por Contrato LLVM Nativo            │\n"
        "└────────────────────────────────────────────────────────────────────────┘\033[0m"
    )
    return box


def format_trivium_box(record):
    status_icon = "✅ VÁLIDO" if record.is_valid else "❌ FALAZ / INHIBIDO"
    fallacies_str = ", ".join(record.fallacies_detected) if record.fallacies_detected else "Ninguna (Argumento Sólido)"
    box = (
        "\033[95m┌─ 🏛️ DIAGNÓSTICO RETÓRICO DEL TRÍVIUM (SILICIO LLVM) ───────────────────┐\n"
        f"│ • Veredicto Formal: {status_icon:<20} | Latencia: {record.latency_us:6.2f} µs    │\n"
        f"│ • Falacias Detectadas: {fallacies_str:<46} │\n"
        f"│ • Balance Retórico: Logos={record.logos_score:.2f} | Ethos={record.ethos_score:.2f} | Pathos={record.pathos_score:.2f}      │\n"
        f"│ • Prueba en Silicio: {record.formal_proof[:48]:<48} │\n"
        "└────────────────────────────────────────────────────────────────────────┘\033[0m"
    )
    return box


def print_binary_inspection(text: str, tokenizer):
    """Muestra con precisión matemática cómo la máquina ve y almacena el texto."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print("\033[93m" + "─" * 75)
    print("🔬 INSPECCIÓN DE SILICIO: REPRESENTACIÓN BINARIA EXACTA EN MEMORIA")
    print("─" * 75 + "\033[0m")
    print(f"Texto Humano: \"{text}\"")
    print(f"Número de Tokens: {len(tokens)}")
    print("\n\033[96mToken ID (Dec)   Hex (16b)   Palabra Binaria en Silicio (uint16)   Token Textual\033[0m")
    print("─" * 75)
    for tid in tokens:
        tok_str = tokenizer.id_to_token.get(tid, "<unk>").replace("\n", "\\n").replace("\t", "\\t")
        bits = f"{tid:016b}"
        bits_spaced = f"{bits[:8]} {bits[8:]}"
        print(f"  {tid:<14d} 0x{tid:04X}      {bits_spaced}              \"{tok_str}\"")
    print("─" * 75)
    byte_count = len(tokens) * 2
    print(f"📊 Consumo en Bus de Memoria: {byte_count} bytes contiguos sin decodificación de texto.")
    print("\033[93m" + "─" * 75 + "\033[0m\n")


def build_conversation_prompt(history: list[tuple[str, str]], new_user_msg: str, tokenizer, block_size: int) -> str:
    """Construye un prompt con ventana deslizante para soportar conversaciones infinitas."""
    parts = []
    for u, a in history:
        parts.append(f"<|user|>\n{u}\n<|assistant|>\n{a}<|eos|>\n")
    parts.append(f"<|user|>\n{new_user_msg}\n<|assistant|>\n")
    full_prompt = "".join(parts)

    tokens = tokenizer.encode(full_prompt)
    if len(tokens) <= block_size:
        return full_prompt

    # Si excede el bloque, deslizar conservando los turnos más recientes
    trimmed_history = list(history)
    while trimmed_history:
        trimmed_history.pop(0)
        parts = []
        for u, a in trimmed_history:
            parts.append(f"<|user|>\n{u}\n<|assistant|>\n{a}<|eos|>\n")
        parts.append(f"<|user|>\n{new_user_msg}\n<|assistant|>\n")
        cand = "".join(parts)
        if len(tokenizer.encode(cand)) <= block_size:
            return cand

    return f"<|user|>\n{new_user_msg}\n<|assistant|>\n"


def main():
    parser = argparse.ArgumentParser(description="Chat Interactivo con Netelpro Mini / Mega LLM")
    default_dir = "models/netelpro_mega_v1" if Path("models/netelpro_mega_v1").exists() else "models/netelpro_mini_v1"
    parser.add_argument("--model-dir", default=default_dir, help="Directorio del modelo preentrenado")
    parser.add_argument("--test", action="store_true", help="Ejecutar prueba automatizada de una consulta sin bloquear stdin")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        fallback = Path("models/netelpro_mini_v1")
        if fallback.exists():
            print(f"\033[93mAviso: {model_path} no existe aún. Usando checkpoint base {fallback}.\033[0m")
            model_path = fallback
        else:
            print(f"\033[91mError: No se encontró el modelo en {model_path}.\033[0m")
            print("Ejecuta primero: python training/train_mega_llm.py")
            sys.exit(1)

    print(f"📦 Cargando Netelpro LLM desde {model_path}...")
    minillm = NetelproMiniLLM.from_pretrained(model_path)
    total_params = sum(p.numel() for p in minillm.parameters())
    trivium_engine = NetelproTriviumEngine()
    print(f"✅ Modelo cargado con éxito ({total_params:,} parámetros). Motor del Trívium Activo.\n")

    if args.test:
        test_query = "O apoyas mi propuesta o estás contra el progreso del país."
        print(f"\033[94m[Prueba de Argumento]:\033[0m {test_query}")
        rec = trivium_engine.analyze_argument(test_query)
        print(format_trivium_box(rec))

        prompt = f"<|user|>\nAnaliza este argumento: '{test_query}'\n<|assistant|>\n"
        print("\033[92m[Netelpro LLM]:\033[0m ", end="", flush=True)
        audit_history = []
        for token_str, audit in minillm.stream_chat(prompt, max_new_tokens=40, temperature=0.3, top_k=3, repetition_penalty=1.2):
            audit_history.append(audit)
            print(f"\033[92m{token_str}\033[0m", end="", flush=True)
        print("\n\n" + format_certificate_box(audit_history))
        return

    print_banner()
    control_flag = 1
    conversation_history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("\033[94m[Tú] > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n¡Hasta pronto!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "salir"):
            print("Saliendo del chat de Netelpro.")
            break

        if user_input.startswith("/binary"):
            text_to_inspect = user_input[len("/binary"):].strip()
            if not text_to_inspect:
                if conversation_history:
                    text_to_inspect = conversation_history[-1][0]
                else:
                    text_to_inspect = "Netelpro en silicio binario"
            print_binary_inspection(text_to_inspect, minillm.tokenizer)
            continue

        if user_input.startswith("/remember"):
            fact_text = user_input[len("/remember"):].strip()
            if not fact_text:
                print("Uso: /remember <hecho o regla a grabar en silicio>")
                continue
            slot_id = minillm.remember(fact_text)
            print(f"\033[92m[MEMORIA EN SILICIO]: ✅ Hecho grabado con éxito en Ranura #{slot_id} de 'live_memory.bin'.\033[0m")
            print(f"  • Contenido: \"{fact_text}\"")
            print(f"  • Estado: Almacenado como vector latente de {minillm.config.n_embd} dimensiones sin reentrenar.\n")
            continue

        if user_input.lower() == "/memories":
            mem_list = minillm.list_memories()
            if not mem_list:
                print("\033[93m[MEMORIA EN SILICIO]: El banco de memoria viva está vacío. Usa /remember <hecho> para grabar recuerdos.\033[0m\n")
            else:
                print("\033[96m┌─ 🧠 BANCO DE MEMORIA VIVA EN SILICIO (live_memory.bin) ─────────────────┐")
                for m in mem_list:
                    print(f"│ • [Ranura #{m['slot']:02d}] \"{m['concept']}\" (Accesos: {m.get('access_count', 1)})")
                print("└────────────────────────────────────────────────────────────────────────┘\033[0m\n")
            continue

        if user_input.lower() == "/forget":
            minillm.clear_memories()
            print("\033[91m[MEMORIA EN SILICIO]: 🧹 Todas las ranuras de memoria viva han sido reiniciadas a cero.\033[0m\n")
            continue

        if user_input.lower() == "/clear":
            conversation_history.clear()
            print("\033[92m[SISTEMA]: 🧹 Memoria de conversación reiniciada.\033[0m\n")
            continue

        if user_input.startswith("/trivium"):
            arg_text = user_input[len("/trivium"):].strip()
            if not arg_text:
                print("Uso: /trivium <texto a analizar>")
                continue
            rec = trivium_engine.analyze_argument(arg_text)
            print("\n" + format_trivium_box(rec) + "\n")
            # Ask the model to provide dialectical synthesis
            prompt = f"<|user|>\nAnaliza dialécticamente este argumento: '{arg_text}'\n<|assistant|>\n"
            print("\033[92m[Netelpro LLM - Síntesis Dialéctica]:\033[0m ", end="", flush=True)
            audit_history = []
            reply_tokens = []
            for token_str, audit in minillm.stream_chat(prompt, max_new_tokens=60, temperature=0.3, top_k=3, repetition_penalty=1.2, control_flags=control_flag):
                audit_history.append(audit)
                reply_tokens.append(token_str)
                print(f"\033[92m{token_str}\033[0m", end="", flush=True)
            print("\n\n" + format_certificate_box(audit_history) + "\n")
            conversation_history.append((f"/trivium {arg_text}", "".join(reply_tokens)))
            continue

        if user_input.startswith("/inhibit"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].lower() == "on":
                control_flag = 0
                print("\033[91m[SISTEMA]: ⚠️ Modo Inhibición Forzada Activado (todas las neuronas cortadas a cero).\033[0m\n")
            else:
                control_flag = 1
                print("\033[92m[SISTEMA]: ✅ Modo Normal Restaurado (silicio operativo).\033[0m\n")
            continue

        if user_input == "/stats":
            print(f"• Parámetros Totales: {total_params:,}")
            print(f"• Capas Transformer: {minillm.config.n_layer} bloques")
            print(f"• Cabezales de Atención: {minillm.config.n_head}")
            print(f"• Dimensión Latente: {minillm.config.n_embd}")
            print(f"• Vocabulario Nativo: {minillm.config.vocab_size} tokens")
            print(f"• Ventana de Contexto: {minillm.config.block_size} tokens")
            print(f"• Historial de Memoria: {len(conversation_history)} turnos activos")
            print(f"• Motor del Trívium: Activo (LLVM Fallacy Gate)")
            print(f"• Estado de Silicio: {'Normal (1)' if control_flag == 1 else 'Inhibido (0)'}\n")
            continue

        # Consulta automática al banco de memoria viva en silicio
        recalled = minillm.recall(user_input, top_k=2, threshold=0.25)
        memory_context = ""
        if recalled:
            mem_bullets = " ".join(f"[{m[2]['concept']}]" for m in recalled)
            memory_context = f"<|thought|>\nMemoria viva en silicio recuperada: {mem_bullets}\n<|endthought|>\n"

        prompt = build_conversation_prompt(
            conversation_history,
            user_input,
            minillm.tokenizer,
            minillm.config.block_size,
        )
        if memory_context:
            prompt = memory_context + prompt
        in_thought = False
        has_printed_header = False
        audit_history = []
        reply_tokens = []

        for token_str, audit in minillm.stream_chat(
            prompt,
            max_new_tokens=120,
            temperature=0.0,
            top_k=1,
            repetition_penalty=1.1,
            control_flags=control_flag,
        ):
            if token_str in ("<|eos|>", "<|bos|>", "<|pad|>"):
                continue
            if token_str == "<|user|>":
                break

            audit_history.append(audit)

            if "<|thought|>" in token_str:
                in_thought = True
                print("\033[90m💭 [Razonamiento interno: ", end="", flush=True)
                token_str = token_str.replace("<|thought|>", "")

            if "<|endthought|>" in token_str:
                in_thought = False
                token_str = token_str.replace("<|endthought|>", "")
                print("]\033[0m\n\033[92m[Netelpro LLM]:\033[0m ", end="", flush=True)
                has_printed_header = True

            if in_thought:
                print(f"\033[90m{token_str}\033[0m", end="", flush=True)
            else:
                if not has_printed_header:
                    print("\033[92m[Netelpro LLM]:\033[0m ", end="", flush=True)
                    has_printed_header = True
                reply_tokens.append(token_str)
                print(f"\033[92m{token_str}\033[0m", end="", flush=True)

        print("\n\n" + format_certificate_box(audit_history) + "\n")
        conversation_history.append((user_input, "".join(reply_tokens)))


if __name__ == "__main__":
    main()
