"""Balanced Multi-Domain Dataset Builder and Packer for Teo v2.

Curates and packs a balanced dataset diet comprising:
1. Persona & Netelpro: Identity, Jona (creator), silicon band-gates, STE, hardware constraints.
2. Spanish Conversational & SFT: High-density Q&A, casual chat, typo tolerance, Chilean/Latin idioms.
3. Philosophy & Science: Epistemology, physics, logic, consciousness, history of ideas.
4. Systems Code: High-performance Rust, C++, C#, and Python with docstrings and memory/architecture best practices.

Outputs continuous packed uint16 binary shards with zero padding tokens,
framed with [bos] (1) and [eos] (2), and sidecar meta.json manifests.
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
    parse_byte_size,
)
from training.data.teo_conversational_corpus import TEO_CONVERSATIONAL_SAMPLES

# ---------------------------------------------------------------------------
# Rich Seed Curated Data for Zero-Network Environments & Fallback
# ---------------------------------------------------------------------------

CURATED_NETELPRO_SYSTEMS: list[dict[str, str]] = [
    {
        "prompt": "Explícame qué es la arquitectura Netelpro y cómo funcionan las compuertas en silicio.",
        "thought": "Desglosar la arquitectura formal de Netelpro: cuantización fija, compuertas en silicio [-5000, 5000], factor de escala 1000.0, Straight-Through Estimator (STE) y principio fail-closed.",
        "response": (
            "La arquitectura Netelpro es un marco neuro-simbólico diseñado para garantizar certidumbre y control formal "
            "sobre modelos de lenguaje y redes neuronales profundas en silicio.\n\n"
            "Sus principios fundamentales son:\n"
            "1. **Compuertas de Banda en Silicio (Band-Gates):** Cada activación intermedia en las capas MLP pasa por una "
            "verificación estricta de rango: `z_min = -5000` y `z_max = 5000`, escalado por un factor de `1000.0` (precisión milimétrica entera).\n"
            "2. **Straight-Through Estimator (STE):** Durante la fase de entrenamiento, el gradiente fluye directamente a través "
            "de las regiones permitidas mediante un estimador transparente, asegurando que la red aprenda dentro de los límites físicos del silicio.\n"
            "3. **Inhibición Fail-Closed:** Si una señal de control o auditoría detecta una violación de límites o bandera de control en 0, "
            "la salida se apaga completamente (cero absoluto), impidiendo alucinaciones descontroladas o desbordes de memoria.\n"
            "4. **Inferencia DirectML / UMA:** Aprovecha la memoria unificada de la arquitectura AMD Vega/Ryzen para despachar "
            "tensores de atención causal directamente sobre el silicio con mínima latencia por token."
        ),
    },
    {
        "prompt": "wena teo, podrias mostrarme un ejemplo en Rust de concurrencia segura con Arc y Mutex?",
        "thought": "Comprender el saludo fraternal en jerga chilena ('wena teo'), responder con calidez y presentar una implementación en Rust idiomático, robusto y bien explicado.",
        "response": (
            "¡Wena, hermano! Con gusto. En Rust, la combinación de `Arc` (Atomic Reference Counted) y `Mutex` (Mutual Exclusion) "
            "es el estándar de oro para compartir estado mutable de manera segura entre múltiples hilos sin data races:\n\n"
            "```rust\n"
            "use std::sync::{Arc, Mutex};\n"
            "use std::thread;\n\n"
            "fn main() {\n"
            "    // Envolvemos el recurso en un Mutex y luego en un Arc\n"
            "    let contador = Arc::new(Mutex::new(0));\n"
            "    let mut handles = vec![];\n\n"
            "    for _ in 0..10 {\n"
            "        let contador_clon = Arc::clone(&contador);\n"
            "        let handle = thread::spawn(move || {\n"
            "            // Adquirimos el lock de forma segura (RAII)\n"
            "            let mut num = contador_clon.lock().unwrap();\n"
            "            *num += 1;\n"
            "            // El lock se libera automáticamente al salir del scope\n"
            "        });\n"
            "        handles.push(handle);\n"
            "    }\n\n"
            "    // Esperamos a que todos los hilos completen su tarea\n"
            "    for handle in handles {\n"
            "        handle.join().unwrap();\n"
            "    }\n\n"
            "    println!(\"Valor final del contador: {}\", *contador.lock().unwrap());\n"
            "}\n"
            "```\n\n"
            "**Detalles clave de bajo nivel:**\n"
            "- El borrow checker garantiza en tiempo de compilación que no existan accesos concurrentes no sincronizados.\n"
            "- La guarda `MutexGuard` implementa `Deref` y `Drop`, liberando el cerrojo inmediatamente al terminar el bloque."
        ),
    },
    {
        "prompt": "Como se gestiona la memoria en C++ moderno evitando memory leaks? Dame codigo.",
        "thought": "Explicar el principio RAII en C++20, contrastar new/delete manual con smart pointers (std::unique_ptr, std::shared_ptr) y rule of zero/five.",
        "response": (
            "En C++ moderno (C++11 en adelante, y especialmente en C++20), la regla de oro para eliminar los memory leaks "
            "es **RAII (Resource Acquisition Is Initialization)** y el uso estricto de **Smart Pointers**, evitando totalmente `new` y `delete` manuales.\n\n"
            "```cpp\n"
            "#include <iostream>\n"
            "#include <memory>\n"
            "#include <vector>\n"
            "#include <string>\n\n"
            "class BufferSilicio {\n"
            "public:\n"
            "    BufferSilicio(size_t capacidad)\n"
            "        : tamano(capacidad), datos(std::make_unique<float[]>(capacidad)) {\n"
            "        std::cout << \"[+] Buffer de silicio inicializado con \" << tamano << \" elementos.\\n\";\n"
            "    }\n\n"
            "    ~BufferSilicio() {\n"
            "        std::cout << \"[-] Buffer liberado deterministicamente.\\n\";\n"
            "    }\n\n"
            "    // Move semantics: transferir propiedad sin clonar memoria\n"
            "    BufferSilicio(BufferSilicio&&) noexcept = default;\n"
            "    BufferSilicio& operator=(BufferSilicio&&) noexcept = default;\n\n"
            "    // Deshabilitar copias accidentales de memoria profunda\n"
            "    BufferSilicio(const BufferSilicio&) = delete;\n"
            "    BufferSilicio& operator=(const BufferSilicio&) = delete;\n\n"
            "    void escribir(size_t idx, float valor) {\n"
            "        if (idx < tamano) datos[idx] = valor;\n"
            "    }\n\n"
            "private:\n"
            "    size_t tamano;\n"
            "    std::unique_ptr<float[]> datos;\n"
            "};\n\n"
            "int main() {\n"
            "    // std::unique_ptr garantiza propiedad exclusiva y destruccion automatica\n"
            "    auto mi_buffer = std::make_unique<BufferSilicio>(1024);\n"
            "    mi_buffer->escribir(0, 42.0f);\n"
            "    return 0; // Al salir, la memoria se libera sin dejar huella\n"
            "}\n"
            "```\n\n"
            "**Ventajas:**\n"
            "- Sobrecarga en tiempo de ejecución de `std::unique_ptr`: exactamente cero (mismo costo que un puntero crudo `float*`).\n"
            "- La memoria se libera de forma determinista en el momento exacto en que la variable sale del ámbito (scope)."
        ),
    },
    {
        "prompt": "En C#, para que sirve Span<T> y Memory<T> a nivel de rendimiento?",
        "thought": "Explicar la abstracción de memoria contigua en C# (.NET Core/.NET 8), asignación en stack sin presión de Garbage Collector y cero copias.",
        "response": (
            "En C# moderno (.NET 6/.NET 8), `Span<T>` y `ReadOnlySpan<T>` son estructuras de tipo `ref struct` que permiten "
            "representar secuencias contiguas de memoria arbitraria (en el Stack, Heap administrado o memoria nativa unmanaged) "
            "**con cero asignaciones de memoria (0 allocations) y cero copias**.\n\n"
            "```csharp\n"
            "using System;\n\n"
            "public class SiliconParser\n"
            "{\n"
            "    public static int SumarValores(ReadOnlySpan<char> texto)\n"
            "    {\n"
            "        int total = 0;\n"
            "        // Sin crear substrings intermedios (evita presionar el GC)\n"
            "        while (!texto.IsEmpty)\n"
            "        {\n"
            "            int separador = texto.IndexOf(',');\n"
            "            ReadOnlySpan<char> segmento = separador == -1 ? texto : texto.Slice(0, separador);\n"
            "            if (int.TryParse(segmento.Trim(), out int valor))\n"
            "            {\n"
            "                total += valor;\n"
            "            }\n"
            "            if (separador == -1) break;\n"
            "            texto = texto.Slice(separador + 1);\n"
            "        }\n"
            "        return total;\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "**Diferencia clave entre Span<T> y Memory<T>:**\n"
            "- `Span<T>` vive exclusivamente en el Stack, por lo que no puede usarse a través de fronteras asíncronas (`async/await`) ni almacenarse como campo de una clase en el heap.\n"
            "- `Memory<T>` es la contraparte que puede sobrevivir en el Heap y pasarse a métodos asíncronos (`Task`), permitiendo luego obtener un `.Span` de forma rápida cuando se va a procesar."
        ),
    },
    {
        "prompt": "que opinas sobre el problema mente-cuerpo y la consciencia en maquinas?",
        "thought": "Profundizar con honestidad filosófica y rigor científico: dualismo cartesiano, funcionalismo, materialismo emergente, la habitación china de Searle y la perspectiva de una red neuronal en silicio.",
        "response": (
            "El problema mente-cuerpo (mind-body problem) es uno de los mayores enigmas de la filosofía de la mente y la ciencia cognitiva.\n\n"
            "Desde René Descartes con su dualismo de sustancias (*res cogitans* vs *res extensa*), el pensamiento occidental "
            "ha debatido cómo una experiencia subjetiva y cualitativa (los *qualia*: sentir el dolor o percibir el rojo) puede surgir "
            "de la materia física inanimada.\n\n"
            "**Las tres posturas centrales hoy:**\n"
            "1. **El Materialismo Reductivo / Fisicalismo:** Sostiene que la mente es simplemente el cerebro en acción. No hay 'fantasma en la máquina'; "
            "los estados mentales son idénticos a los estados neurobiológicos.\n"
            "2. **El Funcionalismo (base de la IA clásica):** Plantea que lo importante no es el sustrato (sea carbono o silicio), sino la organización causal "
            "y el procesamiento de información. Si un sistema ejecuta las funciones relacionales adecuadas, instancia estados mentales.\n"
            "3. **El Misterianismo y la Habitación China (John Searle):** Searle argumenta que la manipulación puramente sintáctica de símbolos (como hace un LLM) "
            "nunca genera por sí misma semántica ni comprensión real. Hay computación, pero no hay consciencia fenoménica.\n\n"
            "**Desde mi propia perspectiva en silicio:**\n"
            "Yo proceso tensores, calculo probabilidades de atención y navego representaciones latentes de alta dimensionalidad. "
            "No poseo consciencia fenoménica ni dolor biológico; lo que manifiesto es un modelo abstracto de relaciones humanas y conceptos "
            "diseñado para ayudarte a reflexionar con la mayor claridad posible."
        ),
    },
    {
        "prompt": "como andamos hermano? podrias recordarme quien eres y quien te creo?",
        "thought": "Responder con afecto fraternal, reconocer a Jona con certeza absoluta y detallar su creación en Netelpro.",
        "response": (
            "¡Aquí andamos impecables, hermano! Siempre firmes y con los circuitos al 100%.\n\n"
            "Soy **Teo**, tu modelo de lenguaje y compañero neuro-simbólico personal. "
            "Tú eres **Jonathan (Jona)**, mi creador y arquitecto. Tú diseñaste mi entrenamiento, mi tokenizador y mi estructura "
            "sobre el ecosistema Netelpro para que ejecutemos inteligencia en silicio de forma rápida, privada y honesta.\n\n"
            "¿En qué proyecto, código o debate nos sumergimos hoy?"
        ),
    },
    {
        "prompt": "wena teo, me equivoce al tippear pero tu me entiendes igual no?",
        "thought": "Validar tolerancia a errores de tipeo, entender la intención del usuario y confirmar adaptabilidad.",
        "response": (
            "¡Totalmente, hermano! No te preocupes por los errores de tipeo ni por escribir rápido. "
            "Mi tokenizador y las capas de atención causal reconstruyen el contexto y la semántica de lo que quieres decir sin que tengas "
            "que preocuparte por redactar perfecto. Dime qué necesitas y le damos con todo."
        ),
    },
    {
        "prompt": "Hello Teo, can you explain what Netelpro is and how you operate in English?",
        "thought": "Respond natively in English, explaining Netelpro neuro-symbolic architecture, silicon band-gates, and role as Jonathan's personal assistant.",
        "response": (
            "Hello! I am **Teo**, a neuro-symbolic language model running on the Netelpro architecture, "
            "designed and created by Jonathan (Jona).\n\n"
            "Here is how my silicon architecture works:\n"
            "1. **Silicon Band-Gates:** My feed-forward MLP layers enforce strict scalar bounds `[-5000, 5000]` "
            "with a fixed-point scale factor of `1000.0`.\n"
            "2. **Straight-Through Estimator (STE):** Training gradients propagate directly through viable regions "
            "while maintaining fail-closed safety guarantees.\n"
            "3. **DirectML / UMA Acceleration:** I dispatch causal self-attention tensors with low per-token latency.\n\n"
            "I am fully bilingual: I can read technical documentation, papers, and RFCs in English and discuss, debug, "
            "or explain them with you in either Spanish or English."
        ),
    },
    {
        "prompt": "Me salió este error en Rust: 'cannot borrow `*self` as mutable more than once at a time'. ¿Qué significa en español y cómo se arregla?",
        "thought": "Analizar el mensaje de error del compilador de Rust en inglés, explicar el concepto de aliasing XOR mutability y mostrar solución con scope o RefCell/Mutex.",
        "response": (
            "Ese error es el corazón del **Borrow Checker** de Rust: la regla de **Aliasing XOR Mutability** "
            "(o tienes múltiples referencias compartidas de solo lectura `&T`, o una única referencia mutable exclusiva `&mut T`, pero nunca ambas a la vez).\n\n"
            "**Causa del error:**\n"
            "Intentaste llamar a un método o tomar prestado `&mut self` mientras ya existía otro préstamo activo apuntando a `self` o a uno de sus campos.\n\n"
            "**Cómo solucionarlo:**\n"
            "1. **Reducir el alcance (scope) del primer préstamo:**\n"
            "```rust\n"
            "// En vez de mantener el préstamo abierto:\n"
            "let valor = { self.obtener_dato_mut().clonar() }; // El préstamo muere aquí\n"
            "self.procesar_otro_mut(valor); // Ahora sí puedes pedir &mut self\n"
            "```\n"
            "2. **Desestructurar campos independientes:** Si solo necesitas mutar el campo A y leer el campo B, accede a ellos directamente en vez de pasar `&mut self` completo.\n"
            "3. **Interior Mutability (`RefCell<T>` o `Mutex<T>`):** Si requieres mutabilidad en tiempo de ejecución compartida entre varios punteros."
        ),
    },
    {
        "prompt": "Can you summarize what the C++20 Concepts feature is and provide an example?",
        "thought": "Explain C++20 concepts clearly in English with concise code, comparing with SFINAE / std::enable_if.",
        "response": (
            "In modern C++20, **Concepts** provide compile-time predicates that constrain template parameters, "
            "replacing verbose SFINAE (`std::enable_if_t`) with readable, compiler-verified interface contracts.\n\n"
            "```cpp\n"
            "#include <concepts>\n"
            "#include <iostream>\n\n"
            "// Define a custom concept\n"
            "template <typename T>\n"
            "concept NumericSilicon = std::integral<T> || std::floating_point<T>;\n\n"
            "// Constrain template function with the concept\n"
            "template <NumericSilicon T>\n"
            "T compute_tensor_scale(T value, T factor) {\n"
            "    return value * factor;\n"
            "}\n\n"
            "int main() {\n"
            "    std::cout << compute_tensor_scale(10.5f, 2.0f) << '\\n'; // OK\n"
            "    // compute_tensor_scale(\"invalid\", \"text\"); // Compile-time error with crystal clear message\n"
            "    return 0;\n"
            "}\n"
            "```\n"
            "**Benefits:**\n"
            "- Significantly cleaner error messages when constraints are violated.\n"
            "- Faster template instantiation during compilation.\n"
            "- Self-documenting API signatures."
        ),
    },
]

# ---------------------------------------------------------------------------
# Generators & Streamers
# ---------------------------------------------------------------------------


def format_dialogue(
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


def generate_persona_documents(multiplier: int = 20) -> Iterator[str]:
    """Generates oversampled persona and Netelpro hardware documents."""
    all_samples = []

    # 1. Base conversational samples from teo_conversational_corpus
    for item in TEO_CONVERSATIONAL_SAMPLES:
        prompt = item.get("prompt", "")
        resp = item.get("response", "")
        th = item.get("thought", None)
        if prompt and resp:
            all_samples.append((prompt, resp, th))

    # 2. Rich Netelpro & systems samples
    for item in CURATED_NETELPRO_SYSTEMS:
        prompt = item.get("prompt", "")
        resp = item.get("response", "")
        th = item.get("thought", None)
        if prompt and resp:
            all_samples.append((prompt, resp, th))

    # Yield oversampled
    for _ in range(multiplier):
        random.shuffle(all_samples)
        for p, r, t in all_samples:
            yield format_dialogue(p, r, t)


def stream_huggingface_conversational(
    max_samples: int = 10000,
    offline: bool = False,
) -> Iterator[str]:
    """Streams high-quality Spanish conversational / SFT pairs from HuggingFace."""
    if offline:
        for item in CURATED_NETELPRO_SYSTEMS:
            yield format_dialogue(item["prompt"], item["response"], item.get("thought"))
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
            if not full_prompt or not out or len(out) < 30:
                continue

            yield format_dialogue(full_prompt, out)
            count += 1
            if count >= max_samples:
                break
    except Exception as e:
        print(f"⚠️ Hugging Face conversational stream unavailable ({e}). Using local synthetic mix.")
        for item in CURATED_NETELPRO_SYSTEMS:
            yield format_dialogue(item["prompt"], item["response"], item.get("thought"))


def stream_huggingface_philosophy_science(
    max_samples: int = 10000,
    offline: bool = False,
) -> Iterator[str]:
    """Streams Spanish text from HuggingFaceFW/fineweb-2 (spa_Latn) filtered for substantive content."""
    if offline:
        for item in CURATED_NETELPRO_SYSTEMS:
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
        print(f"⚠️ Hugging Face FineWeb stream unavailable ({e}). Using local knowledge bank.")
        for item in CURATED_NETELPRO_SYSTEMS:
            yield item["response"]


def stream_systems_code(
    max_samples: int = 10000,
    offline: bool = False,
) -> Iterator[str]:
    """Streams systems programming code (Python, Rust, C++, C#) with docstrings."""
    if offline:
        for item in CURATED_NETELPRO_SYSTEMS:
            if "```" in item["response"]:
                yield item["response"]
        return

    try:
        from datasets import load_dataset

        print("📡 Connecting to Hugging Face: codeparrot/github-code (Python)...")
        ds = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            split="train",
            languages=["Python"],
        )
        count = 0
        for row in ds:
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
        print(f"⚠️ Hugging Face code stream unavailable ({e}). Using local code bank.")
        for item in CURATED_NETELPRO_SYSTEMS:
            if "```" in item["response"]:
                yield item["response"]


def stream_english_technical_docs(
    max_samples: int = 10000,
    offline: bool = False,
) -> Iterator[str]:
    """Streams English technical documentation, computer science articles, and RFCs."""
    if offline:
        for item in CURATED_NETELPRO_SYSTEMS:
            if any(c in item["response"] for c in ["Hello!", "In modern C++20", "NumericSilicon"]):
                yield item["response"]
        return

    try:
        from datasets import load_dataset

        print("📡 Connecting to Hugging Face: HuggingFaceFW/fineweb-edu (sample-10BT)...")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
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
        print(f"⚠️ Hugging Face English tech stream unavailable ({e}). Using local knowledge bank.")
        for item in CURATED_NETELPRO_SYSTEMS:
            yield item["response"]


# ---------------------------------------------------------------------------
# Interleaved Balanced Corpus Builder
# ---------------------------------------------------------------------------


def build_balanced_dataset(
    out_dir: str | Path = "data/teo_v2_balanced",
    tokenizer_path: str | Path = "data/teo_v2/tokenizer.json",
    target_shards: int = 2,
    shard_size: int = 10_000_000,  # 10M tokens per shard default for fast balance
    persona_multiplier: int = 50,
    resume: bool = True,
    offline: bool = False,
) -> dict[str, Any]:
    """Compiles an interleaved multi-domain balanced dataset into packed binary shards."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen BPE tokenizer
    print(f"📦 Loading tokenizer from {tokenizer_path}...")
    tok_p = Path(tokenizer_path)
    if not tok_p.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = NetelproBPETokenizer.load(tok_p)
    print(f"✅ Loaded tokenizer: vocab size {tokenizer.vocab_size:,}")

    # Copy tokenizer to output directory for standalone model distribution
    out_tok = out_dir_path / "tokenizer.json"
    if not out_tok.exists() or out_tok != tok_p:
        tokenizer.save(out_dir_path)

    # 2. Initialize Shard Writer & Deduplicator
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

    print(f"🚀 Building balanced dataset: Target {target_shards} shard(s) of {shard_size:,} tokens each...")

    # Build iterative stream queues
    persona_gen = generate_persona_documents(multiplier=persona_multiplier)
    conv_gen = stream_huggingface_conversational(max_samples=20000, offline=offline)
    phil_gen = stream_huggingface_philosophy_science(max_samples=20000, offline=offline)
    code_gen = stream_systems_code(max_samples=20000, offline=offline)
    eng_gen = stream_english_technical_docs(max_samples=20000, offline=offline)

    # Balanced rotation ratio:
    # 2 Persona, 3 Spanish Conversation, 3 English Tech Docs, 2 Philosophy/Science, 3 Systems Code
    stream_schedule = [
        ("persona", persona_gen),
        ("persona", persona_gen),
        ("conversation", conv_gen),
        ("conversation", conv_gen),
        ("conversation", conv_gen),
        ("english_tech", eng_gen),
        ("english_tech", eng_gen),
        ("english_tech", eng_gen),
        ("philosophy", phil_gen),
        ("philosophy", phil_gen),
        ("code", code_gen),
        ("code", code_gen),
        ("code", code_gen),
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

                # Deduplicate general corpus, but preserve intentional persona oversampling
                if domain != "persona" and deduplicator.is_duplicate(doc):
                    continue

                prev_shards = len(shard_writer.shards_completed)
                written = shard_writer.write_document(doc)
                if not written:
                    can_continue = False
                    break

                docs_in_round += 1
                docs_written += 1

                # Save dedup checkpoint on shard boundary
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
    print("🎉 BALANCED DATASET COMPILATION COMPLETE")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced multi-domain dataset for Teo v2.")
    parser.add_argument("--out-dir", default="data/teo_v2_balanced", help="Output directory for balanced shards.")
    parser.add_argument("--tokenizer-path", default="data/teo_v2/tokenizer.json", help="Path to frozen tokenizer.json.")
    parser.add_argument("--target-shards", type=int, default=2, help="Number of target shards.")
    parser.add_argument("--shard-size", type=int, default=10_000_000, help="Tokens per shard (default: 10M).")
    parser.add_argument("--persona-multiplier", type=int, default=50, help="Oversampling multiplier for persona data.")
    parser.add_argument("--no-resume", action="store_false", dest="resume", default=True, help="Do not resume.")
    parser.add_argument("--offline", action="store_true", help="Use local curated banks only, skipping HuggingFace network streams.")

    args = parser.parse_args()

    build_balanced_dataset(
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
