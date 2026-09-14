"""Corpus de Entrenamiento del Trívium: Retórica, Dialéctica y Ciencias Sociales.

Genera ejemplos estructurados de razonamiento dialéctico, análisis sociológico
y desmontaje formal de falacias argumentativas para el Netelpro Mini LLM.
"""

from __future__ import annotations

import json
from pathlib import Path


def create_trivium_corpus() -> list[dict[str, str]]:
    """Generates a dataset of dialectical inquiries, social analysis, and fallacy debunking."""
    samples = [
        # --- 1. Desmontaje Formal de Falacias ---
        {
            "prompt": "<|user|>\nAnaliza este argumento: 'No debemos escuchar las propuestas económicas de Carlos porque es demasiado joven e inexperto'.\n<|assistant|>\n<|thought|>\nIdentificar falacia Ad Hominem: se ataca la edad/persona de Carlos en lugar de refutar los méritos de su propuesta económica.\n<|endthought|>\nEste argumento comete una falacia formal de tipo **Ad Hominem**. Descalifica la propuesta atacando la identidad del emisor (su juventud) y no la validez o evidencia de sus ideas económicas.<|eos|>"
        },
        {
            "prompt": "<|user|>\nAnaliza este argumento: 'O apoyas totalmente esta nueva ley fiscal o estás a favor de la ruina económica del país'.\n<|assistant|>\n<|thought|>\nIdentificar Falsa Dicotomía: reduce un debate tributario complejo a dos extremos forzados y excluyentes.\n<|endthought|>\nEste argumento incurre en una **Falsa Dicotomía**. Fuerza un dilema binario artificial ignorando la amplia gama de enmiendas, matices y modelos fiscales alternativos viables.<|eos|>"
        },
        {
            "prompt": "<|user|>\nAnaliza este argumento: 'Los que defienden la regulación ambiental quieren que volvamos a la Edad de Piedra sin electricidad'.\n<|assistant|>\n<|thought|>\nIdentificar Hombre de Paja: distorsiona y caricaturiza la postura ambientalista para hacerla parecer ridícula.\n<|endthought|>\nEl argumento incurre en la falacia del **Hombre de Paja**. Caricaturiza la regulación ambiental como un deseo de primitivismo, eludiendo el debate real sobre transición energética sostenible.<|eos|>"
        },
        # --- 2. Dialéctica y Ciencias Sociales (Tesis - Antítesis - Síntesis) ---
        {
            "prompt": "<|user|>\n¿Cómo se relacionan la libertad individual y el orden social en la sociología moderna?\n<|assistant|>\n<|thought|>\nDesarrollar estructura dialéctica: Tesis (autonomía individual), Antítesis (cohesión y normas estructurales), Síntesis (equilibrio institucional democrático).\n<|endthought|>\nDialécticamente, la tensión se resuelve comprendiendo que la libertad individual no se opone al orden social, sino que requiere de instituciones justas y normas compartidas para florecer sin degenerar en anomia.<|eos|>"
        },
        {
            "prompt": "<|user|>\nExplica el concepto de estratificación social desde una perspectiva crítica.\n<|assistant|>\n<|thought|>\nAnalizar distribución asimétrica de recursos, capital cultural, económico y simbólico.\n<|endthought|>\nLa estratificación social describe cómo las sociedades organizan a los grupos en jerarquías estructuradas, distribuyendo asimétricamente el poder, la riqueza material y las oportunidades vitales a lo largo de las generaciones.<|eos|>"
        },
        {
            "prompt": "<|user|>\n¿Qué diferencia a la retórica persuasiva constructiva de la demagogia?\n<|assistant|>\n<|thought|>\nContrastar el equilibrio Logos-Ethos-Pathos frente a la manipulación del miedo y el resentimiento.\n<|endthought|>\nLa retórica constructiva busca el consenso mediante la razón (Logos) y la ética (Ethos); la demagogia, en cambio, manipula pasiones primarias (Pathos desbordado) y polariza suprimiendo el rigor argumentativo.<|eos|>"
        },
        # --- 3. Razonamiento Ético y Sistémico ---
        {
            "prompt": "<|user|>\n¿Por qué es importante considerar los efectos de segundo orden al diseñar una política pública?\n<|assistant|>\n<|thought|>\nAplicar teoría de sistemas sociales: las intervenciones lineales generan consecuencias no deseadas.\n<|endthought|>\nPorque en sistemas sociales complejos, una intervención no solo afecta la variable inmediata, sino que altera incentivos y dinámicas relacionales, pudiendo agravar el problema que intentaba corregir.<|eos|>"
        },
    ]

    # Multiply and balance the samples
    expanded = []
    for _ in range(40):
        for s in samples:
            expanded.append(s)
    return expanded


def save_trivium_corpus(output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    corpus = create_trivium_corpus()
    with open(p, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return p


if __name__ == "__main__":
    target = Path(__file__).parent / "trivium_train.jsonl"
    save_trivium_corpus(target)
    print(f"Corpus del Trívium guardado con éxito en: {target}")
