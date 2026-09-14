"""Massive Multi-Source Knowledge Corpus Builder for Teo v2 (Netelpro SBT).

Compiles a deep 250M - 500M token knowledge diet across:
1. Cosmopedia v2 (Textbooks, Academic Web-Courses & STEM).
2. Advanced Spanish Multi-Turn Dialogues (OpenAssistant & SFT).
3. Systems Architecture & Code (Rust Crates, C++20, C# .NET 8, Python).
4. Chain-of-Thought Reasoning Traces (<|thought|> ... <|endthought|>).
5. Netelpro SBT Formal Hardware Lore (STE, Silicon Band-Gates, Jona as Creator).

Outputs contiguous packed uint16 binary shards with zero padding tokens,
framed with [bos] (1) and [eos] (2), with full resume persistence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterator, Sequence

import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from netelpro.neuro.tokenizer_bpe import (
    NetelproBPETokenizer,
    SPECIAL_TOKENS,
    SPECIAL_TOKEN_TO_ID,
)
from training.data.compile_packed import load_stream, stream_stats
from training.data.download_corpus import (
    DocumentDeduplicator,
    PackedShardWriter,
    clean_document,
)
from training.data.teo_conversational_corpus import TEO_CONVERSATIONAL_SAMPLES

# ---------------------------------------------------------------------------
# Advanced STEM, Systems & Reasoning Seed Knowledge
# ---------------------------------------------------------------------------

ADVANCED_REASONING_SAMPLES: list[dict[str, str]] = [
    {
        "prompt": "¿Cómo se implementa un algoritmo de grafos como Dijkstra en Rust con una cola de prioridad (BinaryHeap)?",
        "thought": (
            "1. Descomponer el algoritmo de Dijkstra: distancias mínimas desde un nodo origen a todos los demás en un grafo con pesos no negativos.\n"
            "2. En Rust, std::collections::BinaryHeap es un max-heap por defecto. Para que funcione como min-heap, debemos envolver el estado en una estructura que invierta la comparación (Ordering::Reverse o implementar Ord).\n"
            "3. Estructurar el grafo como lista de adyacencia Vec<Vec<(usize, u32)>>.\n"
            "4. Escribir código idiomático, seguro, sin unwrap inseguros y con explicación paso a paso."
        ),
        "response": (
            "En Rust, para implementar el algoritmo de Dijkstra de forma eficiente ($O((V + E) \\log V)$), "
            "usamos `std::collections::BinaryHeap`. Dado que en Rust `BinaryHeap` es un max-heap, debemos invertir "
            "el orden para extraer siempre el nodo con menor distancia acumulada:\n\n"
            "```rust\n"
            "use std::cmp::Ordering;\n"
            "use std::collections::BinaryHeap;\n\n"
            "#[derive(Copy, Clone, Eq, PartialEq)]\n"
            "struct State {\n"
            "    cost: u32,\n"
            "    position: usize,\n"
            "}\n\n"
            "// Invertimos la comparacion para convertir MaxHeap en MinHeap\n"
            "impl Ord for State {\n"
            "    fn cmp(&self, other: &Self) -> Ordering {\n"
            "        other.cost.cmp(&self.cost)\n"
            "            .then_with(|| self.position.cmp(&other.position))\n"
            "    }\n"
            "}\n\n"
            "impl PartialOrd for State {\n"
            "    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {\n"
            "        Some(self.cmp(other))\n"
            "    }\n"
            "}\n\n"
            "struct Edge {\n"
            "    node: usize,\n"
            "    cost: u32,\n"
            "}\n\n"
            "pub fn dijkstra(adj_list: &[Vec<Edge>], start: usize) -> Vec<u32> {\n"
            "    let mut dist: Vec<u32> = vec![u32::MAX; adj_list.len()];\n"
            "    let mut heap = BinaryHeap::new();\n\n"
            "    dist[start] = 0;\n"
            "    heap.push(State { cost: 0, position: start });\n\n"
            "    while let Some(State { cost, position }) = heap.pop() {\n"
            "        if cost > dist[position] {\n"
            "            continue;\n"
            "        }\n"
            "        for edge in &adj_list[position] {\n"
            "            let next = State {\n"
            "                cost: cost + edge.cost,\n"
            "                position: edge.node,\n"
            "            };\n"
            "            if next.cost < dist[next.position] {\n"
            "                dist[next.position] = next.cost;\n"
            "                heap.push(next);\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    dist\n"
            "}\n"
            "```\n\n"
            "**Detalles clave de arquitectura:**\n"
            "- La verificación `cost > dist[position]` descarta entradas obsoletas en el heap sin necesidad de operaciones costosas de decremento de clave.\n"
            "- Memoria contigua en vector pre-asignado que minimiza cache misses."
        ),
    },
    {
        "prompt": "Explica la paradoja EPR y el teorema de Bell en física cuántica. ¿Existe el realismo local?",
        "thought": (
            "1. Definir la paradoja formulada por Einstein, Podolsky y Rosen (1935): objeción a la interpretación de Copenhague sobre el entrelazamiento cuántico ('acción fantasmal a la distancia').\n"
            "2. El supuesto de EPR: debe existir un 'realismo local' con variables ocultas locales que determinen el resultado antes de la medición.\n"
            "3. La demostración matemática de John Bell (1964): las desigualdades de Bell establecen un límite superior para las correlaciones estadísticas si el realismo local es verdadero.\n"
            "4. Evidencia experimental (Aspect, Clauser, Zeilinger - Premio Nobel 2022): la naturaleza viola las desigualdades de Bell. El realismo local es falso."
        ),
        "response": (
            "La paradoja de Einstein-Podolsky-Rosen (EPR, 1935) y el Teorema de Bell (1964) representan el debate más profundo "
            "sobre la naturaleza ontológica de la realidad en la historia de la física.\n\n"
            "### 1. La Objeción EPR y el Realismo Local\n"
            "Albert Einstein se resistía a la interpretación probabilística de la mecánica cuántica ('Dios no juega a los dados'). "
            "Junto a Podolsky y Rosen, postuló dos principios aparentemente incuestionables:\n"
            "- **Realismo:** Las partículas poseen propiedades físicas bien definidas antes de ser observadas.\n"
            "- **Localidad:** Ninguna influencia física puede viajar más rápido que la velocidad de la luz en el vacío ($c$).\n\n"
            "Si dos partículas entrelazadas se separan a millones de kilómetros y medir una determina instantáneamente el estado de la otra, "
            "EPR concluyó que la mecánica cuántica estaba 'incompleta' y que debían existir **variables ocultas locales** predeterminadas.\n\n"
            "### 2. El Teorema de Bell (1964)\n"
            "Durante 30 años esto fue un debate filosófico estéril, hasta que el físico norirlandés John Stewart Bell formuló una desigualdad matemática: "
            "si el realismo local es correcto, la correlación estadística entre mediciones polarizadas en distintos ángulos no puede superar un límite numérico estricto ($|S| \\le 2$).\n\n"
            "Sin embargo, la mecánica cuántica predecía que las partículas entrelazadas pueden correlacionarse hasta $|S| = 2\\sqrt{2} \\approx 2.828$ (el límite de Cirel'son).\n\n"
            "### 3. Conclusión Experimental y Filosofía\n"
            "Los experimentos sin resquicios (*loophole-free*) de Alain Aspect, John Clauser y Anton Zeilinger (Premio Nobel 2022) "
            "demostraron que la naturaleza **viola sistemáticamente las desigualdades de Bell**.\n\n"
            "**La conclusión científica es tajante:** El universo no es localmente realista. O bien el mundo es no-local (las entidades están conectadas instantáneamente), "
            "o bien las propiedades no existen de forma definida hasta que se produce la interacción de medida."
        ),
    },
    {
        "prompt": "¿Por qué Netelpro SBT es una arquitectura superior a los transformers comerciales tradicionales para dispositivos de borde (Edge)?",
        "thought": (
            "1. Describir la arquitectura Netelpro SBT (Silicon-Bounded Transformer).\n"
            "2. Contrastar con transformers de punto flotante libre (LLaMA, Gemma, GPT-2): desbordes numéricos, alto consumo energético de FPUs, memoria redundante.\n"
            "3. Enfatizar el rol de Jonathan (Jona) como arquitecto creador.\n"
            "4. Resaltar compuertas de banda STE [-5000, 5000], factor de escala 1000.0, cero padding y UMA DirectML."
        ),
        "response": (
            "La arquitectura **Netelpro SBT (Silicon-Bounded Transformer)**, diseñada por mi creador **Jonathan (Jona)**, "
            "resuelve los tres cuellos de botella fundamentales que impiden a los transformers tradicionales operar con eficiencia en dispositivos de borde:\n\n"
            "1. **Estabilidad Bounded en Silicio (`[-5000, 5000]`):**\n"
            "   Los modelos convencionales operan con números decimales flotantes libres (`float16`/`bfloat16`), lo que exige unidades de coma flotante (FPU) "
            "   pesadas y genera explosiones de gradientes (*loss spikes*). Netelpro SBT acota cada tensor en silicio a escala entera fija (`1000.0`), "
            "   eliminando el desborde numérico por diseño.\n\n"
            "2. **Estimador Transparente en Silicio (STE):**\n"
            "   El *Straight-Through Estimator* permite que los gradientes fluyan limpiamente durante el entrenamiento sin truncar la capacidad de aprendizaje, "
            "   pero inhibiendo a cero absoluto (*fail-closed*) cualquier activación fuera de especificación.\n\n"
            "3. **Inferencia Ultra-Liviana en Memoria Unificada (UMA):**\n"
            "   Con apenas 124 millones de parámetros densamente entrenados y empaquetados en streams binarios uint16 continuos sin tokens de relleno (`pad=0`), "
            "   Netelpro SBT consume menos de 300 MB de RAM y vuela a 40-50 tokens por segundo en procesadores AMD Ryzen y gráficas integradas Vega, "
            "   gastando menos del 10% de la energía de un modelo comercial inflado."
        ),
    },
    {
        "prompt": "¿Cómo se diseña un asignador de memoria arena (Arena Allocator) en C++ para motores de juegos o IA?",
        "thought": (
            "Explicar el patrón Arena / Linear Allocator: reservar un bloque contiguo de memoria y despachar punteros incrementando un offset. Liberación en O(1) de golpe."
        ),
        "response": (
            "Un **Arena Allocator** (o asignador lineal) es una de las técnicas de gestión de memoria más rápidas en ingeniería de sistemas. "
            "En lugar de llamar a `malloc` o `new` (que buscan en listas libres y fragmentan el heap), reserva un búfer contiguo grande por adelantado "
            "y despacha memoria simplemente incrementando un puntero de desplazamiento (*offset*):\n\n"
            "```cpp\n"
            "#include <cstddef>\n"
            "#include <cstdint>\n"
            "#include <new>\n"
            "#include <iostream>\n\n"
            "class ArenaAllocator {\n"
            "public:\n"
            "    ArenaAllocator(size_t bytes)\n"
            "        : total_size(bytes), offset(0) {\n"
            "        buffer = new uint8_t[bytes];\n"
            "    }\n\n"
            "    ~ArenaAllocator() {\n"
            "        delete[] buffer;\n"
            "    }\n\n"
            "    void* allocate(size_t bytes, size_t alignment = alignof(std::max_align_t)) {\n"
            "        uintptr_t current = reinterpret_cast<uintptr_t>(buffer + offset);\n"
            "        uintptr_t aligned = (current + (alignment - 1)) & ~(alignment - 1);\n"
            "        size_t new_offset = (aligned - reinterpret_cast<uintptr_t>(buffer)) + bytes;\n\n"
            "        if (new_offset > total_size) {\n"
            "            return nullptr; // Memoria agotada en el arena\n"
            "        }\n\n"
            "        offset = new_offset;\n"
            "        return reinterpret_cast<void*>(aligned);\n"
            "    }\n\n"
            "    // Liberacion instantanea O(1) de todas las reservas\n"
            "    void reset() {\n"
            "        offset = 0;\n"
            "    }\n\n"
            "private:\n"
            "    uint8_t* buffer;\n"
            "    size_t total_size;\n"
            "    size_t offset;\n"
            "};\n"
            "```\n\n"
            "**Ventajas:**\n"
            "- Asignación en tiempo constante $O(1)$ (solo una suma y una máscara de alineación de bits).\n"
            "- Liberación instantánea $O(1)$ para todos los objetos reseteando el `offset` a cero.\n"
            "- Máxima localidad de caché (datos contiguos en memoria física)."
        ),
    },
]

# ---------------------------------------------------------------------------
# Bilingual (ES<->EN) Translation Seed Bank
# ---------------------------------------------------------------------------
#
# The rest of the corpus skews heavily toward raw English prose (Cosmopedia)
# vs. Spanish instruction-following dialogue (Alpaca-ES). Without an explicit
# signal tying the two languages together, a small model sees "prose in EN"
# and "Q&A in ES" as nearly disjoint distributions and never learns to map
# between them, producing incoherent replies. These pairs teach the direct
# ES<->EN correspondence explicitly, in both directions.

TRANSLATION_PAIRS: list[dict[str, str]] = [
    {
        "es": "Hola, soy Teo, un modelo de lenguaje creado por Jonathan usando la arquitectura Netelpro.",
        "en": "Hello, I am Teo, a language model created by Jonathan using the Netelpro architecture.",
    },
    {
        "es": "¿Podrías explicarme cómo funciona la memoria de contexto O(1) en Netelpro SDS?",
        "en": "Could you explain to me how the O(1) context memory works in Netelpro SDS?",
    },
    {
        "es": "La compuerta de silicio limita cada activación al rango [-5000, 5000] con un factor de escala de 1000.0.",
        "en": "The silicon gate bounds every activation to the range [-5000, 5000] with a scale factor of 1000.0.",
    },
    {
        "es": "Un asignador de memoria arena reserva un bloque contiguo grande y despacha punteros incrementando un desplazamiento.",
        "en": "An arena memory allocator reserves one large contiguous block and dispatches pointers by incrementing an offset.",
    },
    {
        "es": "El algoritmo de Dijkstra encuentra las distancias mínimas desde un nodo origen usando una cola de prioridad.",
        "en": "Dijkstra's algorithm finds the minimum distances from a source node using a priority queue.",
    },
    {
        "es": "Gracias por tu ayuda, ¿podrías darme un ejemplo de código en Rust?",
        "en": "Thanks for your help, could you give me a code example in Rust?",
    },
    {
        "es": "El teorema de Bell demuestra que el realismo local no describe correctamente la naturaleza cuántica.",
        "en": "Bell's theorem proves that local realism does not correctly describe quantum nature.",
    },
    {
        "es": "No entiendo esta parte, ¿me lo podrías explicar de una forma más simple?",
        "en": "I don't understand this part, could you explain it to me in a simpler way?",
    },
    {
        "es": "Netelpro SBT elimina el desborde numérico acotando cada tensor en silicio a escala entera fija.",
        "en": "Netelpro SBT eliminates numeric overflow by bounding every tensor in silicon to a fixed integer scale.",
    },
    {
        "es": "¿Cuál es la diferencia entre un puntero y una referencia en C++?",
        "en": "What is the difference between a pointer and a reference in C++?",
    },
    {
        "es": "El estimador recto (STE) permite que los gradientes fluyan sin truncar la capacidad de aprendizaje.",
        "en": "The Straight-Through Estimator (STE) lets gradients flow without truncating learning capacity.",
    },
    {
        "es": "Buenos días, ¿en qué puedo ayudarte hoy?",
        "en": "Good morning, how can I help you today?",
    },
    {
        "es": "El modelo Netelpro SDS logra memoria de inferencia constante eliminando por completo el KV-Cache.",
        "en": "The Netelpro SDS model achieves constant inference memory by completely eliminating the KV-Cache.",
    },
    {
        "es": "Perdón, no entendí bien la pregunta, ¿la podrías repetir con otras palabras?",
        "en": "Sorry, I didn't quite understand the question, could you repeat it in other words?",
    },
    {
        "es": "Un router de mezcla dispersa de expertos activa solo unos pocos expertos por token, no la red entera.",
        "en": "A sparse mixture-of-experts router activates only a few experts per token, not the whole network.",
    },
]


def generate_translation_stream(multiplier: int = 100) -> Iterator[str]:
    """Yields explicit ES<->EN translation Q&A turns in both directions.

    Trains the direct mapping between languages that the rest of the corpus
    (English prose vs. Spanish dialogue) never demonstrates on its own.
    """
    pairs = list(TRANSLATION_PAIRS)
    for _ in range(multiplier):
        random.shuffle(pairs)
        for item in pairs:
            yield format_qa_turn(f"Traduce al inglés: {item['es']}", item["en"])
            yield format_qa_turn(f"Traduce al español: {item['en']}", item["es"])


def stream_english_instructions(
    max_samples: int = 50000,
    offline: bool = False,
) -> Iterator[str]:
    """Streams English instruction-following dialogues (symmetric to Alpaca-ES).

    Without this, the corpus only sees raw English prose (Cosmopedia) and
    Spanish Q&A (Alpaca-ES) — never English Q&A — so the model never learns
    the conversational instruction/response pattern in English at all.
    """
    if offline:
        for item in ADVANCED_REASONING_SAMPLES:
            yield format_qa_turn(item["prompt"], item["response"], item.get("thought"))
        return

    try:
        from datasets import load_dataset

        print("📡 Connecting to Hugging Face: tatsu-lab/alpaca...")
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        count = 0
        for row in ds:
            inst = row.get("instruction", "")
            inp = row.get("input", "")
            out = row.get("output", "")

            full_prompt = f"{inst}\n{inp}".strip() if inp else inst.strip()
            if not full_prompt or not out or len(out) < 25:
                continue

            yield format_qa_turn(full_prompt, out)
            count += 1
            if count >= max_samples:
                break
    except Exception as e:
        print(f"⚠️ English instructions stream fallback to local bank ({e})")
        for item in ADVANCED_REASONING_SAMPLES:
            yield format_qa_turn(item["prompt"], item["response"], item.get("thought"))


# ---------------------------------------------------------------------------
# Formatters & Generators
# ---------------------------------------------------------------------------


def format_qa_turn(
    prompt: str,
    response: str,
    thought: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Formats a structured conversational turn with standard Netelpro delimiters."""
    doc = ""
    if system_prompt:
        doc += f"<|contract|>\n{system_prompt.strip()}\n<|endcontract|>\n"
    doc += f"<|user|>\n{prompt.strip()}\n"
    if thought:
        doc += f"<|thought|>\n{thought.strip()}\n<|endthought|>\n"
    doc += f"<|assistant|>\n{response.strip()}"
    return doc


def generate_reasoning_and_persona_stream(multiplier: int = 100) -> Iterator[str]:
    """Generates rich Netelpro SBT lore, reasoning chains, and persona dialogues."""
    samples = []

    # Persona & identity samples
    for item in TEO_CONVERSATIONAL_SAMPLES:
        p, r, t = item.get("prompt"), item.get("response"), item.get("thought")
        if p and r:
            samples.append((p, r, t))

    # Advanced reasoning samples
    for item in ADVANCED_REASONING_SAMPLES:
        p, r, t = item.get("prompt"), item.get("response"), item.get("thought")
        if p and r:
            samples.append((p, r, t))

    for _ in range(multiplier):
        random.shuffle(samples)
        for p, r, t in samples:
            yield format_qa_turn(p, r, t)


def stream_cosmopedia_stem(
    max_samples: int = 50000,
    offline: bool = False,
) -> Iterator[str]:
    """Streams high-quality educational textbooks from Cosmopedia v2."""
    if offline:
        for item in ADVANCED_REASONING_SAMPLES:
            yield item["response"]
        return

    try:
        from datasets import load_dataset

        print("📡 Connecting to Hugging Face: HuggingFaceTB/cosmopedia-v2...")
        ds = load_dataset(
            "HuggingFaceTB/cosmopedia-v2",
            "cosmopedia-v2",
            split="train",
            streaming=True,
        )
        count = 0
        for row in ds:
            text = row.get("text", "")
            if not text or len(text) < 400:
                continue

            cleaned = clean_document(text, is_code=False, min_chars=250)
            if cleaned:
                yield cleaned
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Cosmopedia stream fallback to local bank ({e})")
        for item in ADVANCED_REASONING_SAMPLES:
            yield item["response"]


def stream_openassistant_conversations(
    max_samples: int = 50000,
    offline: bool = False,
) -> Iterator[str]:
    """Streams multi-turn high-density conversational data in Spanish."""
    if offline:
        for item in ADVANCED_REASONING_SAMPLES:
            yield format_qa_turn(item["prompt"], item["response"], item.get("thought"))
        return

    try:
        from datasets import load_dataset

        print("📡 Connecting to Hugging Face: bertin-project/alpaca-spanish...")
        ds = load_dataset("bertin-project/alpaca-spanish", split="train", streaming=True)
        count = 0
        for row in ds:
            inst = row.get("instruction", "")
            inp = row.get("input", "")
            out = row.get("output", "")

            full_prompt = f"{inst}\n{inp}".strip() if inp else inst.strip()
            if not full_prompt or not out or len(out) < 25:
                continue

            yield format_qa_turn(full_prompt, out)
            count += 1
            if count >= max_samples:
                break
    except Exception as e:
        print(f"⚠️ Conversational stream fallback to local bank ({e})")
        for item in ADVANCED_REASONING_SAMPLES:
            yield format_qa_turn(item["prompt"], item["response"], item.get("thought"))


def stream_systems_code_deep(
    max_samples: int = 50000,
    offline: bool = False,
) -> Iterator[str]:
    """Streams multi-language systems code (Python, Rust, C++, C#)."""
    if offline:
        for item in ADVANCED_REASONING_SAMPLES:
            if "```" in item["response"]:
                yield item["response"]
        return

    try:
        from datasets import load_dataset

        print("📡 Connecting to Hugging Face: codeparrot/github-code-clean...")
        # codeparrot/github-code (the original) relies on a custom loading
        # script (github-code.py) which recent `datasets` versions refuse to
        # run at all ("Dataset scripts are no longer supported"). The -clean
        # mirror ships plain Parquet shards, streamable without a script.
        target_langs = {"Python", "Rust", "C++", "C#", "C"}
        ds = load_dataset(
            "codeparrot/github-code-clean",
            streaming=True,
            split="train",
        )
        count = 0
        for row in ds:
            lang = row.get("language")
            if lang is not None and lang not in target_langs:
                continue

            code = row.get("code", "")
            if not code or len(code) < 200:
                continue

            cleaned = clean_document(code, is_code=True, min_chars=150)
            if cleaned:
                yield cleaned
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Systems code stream fallback to local bank ({e})")
        for item in ADVANCED_REASONING_SAMPLES:
            if "```" in item["response"]:
                yield item["response"]


def stream_fineweb_spanish_philosophy(
    max_samples: int = 50000,
    offline: bool = False,
) -> Iterator[str]:
    """Streams Spanish text from HuggingFaceFW/fineweb-2 (spa_Latn)."""
    if offline:
        for item in ADVANCED_REASONING_SAMPLES:
            yield item["response"]
        return

    try:
        from datasets import load_dataset

        print("📡 Connecting to Hugging Face: HuggingFaceFW/fineweb-2 (spa_Latn)...")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-2",
            "spa_Latn",
            split="train",
            streaming=True,
        )
        count = 0
        for row in ds:
            text = row.get("text", "")
            if not text or len(text) < 300:
                continue

            cleaned = clean_document(text, is_code=False, min_chars=200)
            if cleaned:
                yield cleaned
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ FineWeb-2 stream fallback to local bank ({e})")
        for item in ADVANCED_REASONING_SAMPLES:
            yield item["response"]


# ---------------------------------------------------------------------------
# Massive Corpus Builder
# ---------------------------------------------------------------------------


def build_massive_corpus(
    out_dir: str | Path = "data/teo_v2_massive",
    tokenizer_path: str | Path = "data/teo_v2/tokenizer.json",
    target_shards: int = 10,
    shard_size: int = 25_000_000,  # 25M tokens per shard -> 250M total
    persona_multiplier: int = 100,
    resume: bool = True,
    offline: bool = False,
) -> dict[str, Any]:
    """Compiles a massive multi-shard corpus into packed binary uint16 shards."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"📦 Loading tokenizer from {tokenizer_path}...")
    tok_p = Path(tokenizer_path)
    if not tok_p.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = NetelproBPETokenizer.load(tok_p)
    print(f"✅ Loaded tokenizer: vocab size {tokenizer.vocab_size:,}")

    # Copy tokenizer into output dir
    out_tok = out_dir_path / "tokenizer.json"
    if not out_tok.exists() or out_tok != tok_p:
        tokenizer.save(out_dir_path)

    shard_writer = PackedShardWriter(
        out_dir=out_dir_path,
        shard_size=shard_size,
        max_shards=target_shards,
        tokenizer=tokenizer,
        resume=resume,
    )
    deduplicator = DocumentDeduplicator()

    dedup_path = out_dir_path / "dedup_state.json"
    if dedup_path.is_file() and resume:
        deduplicator.load_state(dedup_path)
        print(f"Loaded deduplication state: {deduplicator.seen_count:,} seen hashes.")

    start_time = time.perf_counter()
    docs_written = shard_writer.total_docs
    initial_tokens = shard_writer.total_tokens

    print(f"🚀 Building massive corpus: Target {target_shards} shard(s) of {shard_size:,} tokens (Total: {target_shards * shard_size:,} tokens)...")

    # Generators
    reasoning_gen = generate_reasoning_and_persona_stream(multiplier=persona_multiplier)
    translation_gen = generate_translation_stream(multiplier=persona_multiplier)
    cosmo_gen = stream_cosmopedia_stem(max_samples=50000, offline=offline)
    conv_gen = stream_openassistant_conversations(max_samples=50000, offline=offline)
    english_conv_gen = stream_english_instructions(max_samples=50000, offline=offline)
    code_gen = stream_systems_code_deep(max_samples=50000, offline=offline)
    phil_gen = stream_fineweb_spanish_philosophy(max_samples=50000, offline=offline)

    # Weighted round-robin: instruction-following dialogue (ES + EN + explicit
    # translation pairs) now outweighs raw prose (Cosmopedia), so the model
    # fixes the Q&A pattern in both languages instead of only completing
    # English textbook text. See docs/ROADMAP_TEO_7B_MOE_SDS.md bitácora.
    stream_schedule = [
        ("reasoning_lore", reasoning_gen),
        ("reasoning_lore", reasoning_gen),
        ("translation_pairs", translation_gen),
        ("translation_pairs", translation_gen),
        ("conversations", conv_gen),
        ("conversations", conv_gen),
        ("conversations", conv_gen),
        ("english_instructions", english_conv_gen),
        ("english_instructions", english_conv_gen),
        ("cosmopedia_stem", cosmo_gen),
        ("cosmopedia_stem", cosmo_gen),
        ("systems_code", code_gen),
        ("systems_code", code_gen),
        ("philosophy_science", phil_gen),
        ("philosophy_science", phil_gen),
    ]

    last_log_time = time.perf_counter()
    can_continue = True

    try:
        while can_continue:
            docs_in_round = 0
            for domain, gen in stream_schedule:
                if shard_writer.max_shards is not None and shard_writer.current_shard_idx >= shard_writer.max_shards:
                    can_continue = False
                    break

                try:
                    doc = next(gen)
                except StopIteration:
                    continue

                if not doc:
                    continue

                if domain not in ("reasoning_lore", "translation_pairs") and deduplicator.is_duplicate(doc):
                    continue

                prev_shards = len(shard_writer.shards_completed)
                written = shard_writer.write_document(doc)
                if not written:
                    can_continue = False
                    break

                docs_in_round += 1
                docs_written += 1

                if len(shard_writer.shards_completed) > prev_shards:
                    deduplicator.save_state(dedup_path)

                now = time.perf_counter()
                if now - last_log_time >= 5.0:
                    elapsed = now - start_time
                    tokens_delta = shard_writer.total_tokens - initial_tokens
                    t_rate = tokens_delta / elapsed if elapsed > 0 else 0
                    print(
                        f"📊 Progress: {shard_writer.total_docs:,} docs | "
                        f"{shard_writer.total_tokens:,} tokens ({t_rate:,.1f} tok/s) | "
                        f"Shard {shard_writer.current_shard_idx + 1}/{target_shards} | "
                        f"Domain: {domain}"
                    )
                    last_log_time = now

            if docs_in_round == 0:
                print("ℹ️ All available document streams reached completion.")
                break
    finally:
        manifest = shard_writer.close()
        deduplicator.save_state(dedup_path)

    elapsed_total = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print("🎉 MASSIVE CORPUS COMPILATION COMPLETE")
    print(f"  • Total Completed Shards: {len(shard_writer.shards_completed)}")
    print(f"  • Total Tokens:           {shard_writer.total_tokens:,}")
    print(f"  • Total Documents:        {shard_writer.total_docs:,}")
    print(f"  • Elapsed Time:           {elapsed_total:.1f}s")
    print(f"  • Output Directory:       {out_dir_path.resolve()}")
    print("=" * 60)

    return {
        "status": "completed",
        "total_shards": len(shard_writer.shards_completed),
        "total_tokens": shard_writer.total_tokens,
        "total_docs": shard_writer.total_docs,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build massive multi-source corpus for Teo v2 (Netelpro SBT).")
    parser.add_argument("--out-dir", default="data/teo_v2_massive", help="Output directory for massive shards.")
    parser.add_argument("--tokenizer-path", default="data/teo_v2/tokenizer.json", help="Path to frozen tokenizer.json.")
    parser.add_argument("--target-shards", type=int, default=10, help="Number of target shards (default: 10 = 250M tokens).")
    parser.add_argument("--shard-size", type=int, default=25_000_000, help="Tokens per shard (default: 25M).")
    parser.add_argument("--persona-multiplier", type=int, default=100, help="Oversampling multiplier for reasoning & lore.")
    parser.add_argument("--no-resume", action="store_false", dest="resume", default=True, help="Do not resume.")
    parser.add_argument("--offline", action="store_true", help="Use local knowledge banks only.")

    args = parser.parse_args()

    build_massive_corpus(
        out_dir=args.out_dir,
        tokenizer_path=args.tokenizer_path,
        target_shards=args.target_shards,
        shard_size=args.shard_size,
        persona_multiplier=args.persona_multiplier,
        resume=args.resume,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
