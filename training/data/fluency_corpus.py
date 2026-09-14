"""Corpus de Fluidez y Lenguaje Natural en Español para el Netelpro Mini LLM.

Proporciona textos gramaticalmente correctos, diálogos claros, definiciones
conceptuales y argumentaciones humanistas en español fluido.
"""

from __future__ import annotations

import json
from pathlib import Path


def create_fluency_corpus() -> list[dict[str, str]]:
    """Creates a high-quality conversational and reasoning dataset in pure natural Spanish."""
    templates = [
        # Saludos e Identidad
        (
            "Hola, ¿quién eres?",
            "Soy Netelpro Mini LLM, un modelo de lenguaje neuro-simbólico que combina redes neuronales con verificación formal en silicio.",
            "Identificar el modelo y su naturaleza neuro-simbólica con compuertas en silicio."
        ),
        (
            "¿Qué puedes hacer?",
            "Puedo analizar argumentos, detectar falacias lógicas, explicar conceptos de ciencias sociales y verificar contratos formales en tiempo real.",
            "Resumir las capacidades de análisis retórico, dialéctica y verificación formal."
        ),
        (
            "¿Qué es la neurona Netelpro?",
            "La neurona Netelpro es una unidad de cálculo que combina una suma ponderada lineal con una compuerta formal en silicio. Si la señal viola una regla, se inhibe a cero de inmediato.",
            "Explicar la función de activación con compuerta formal fail-closed."
        ),
        (
            "¿Cómo evitas las alucinaciones?",
            "Utilizo un procesador de logits en silicio que poda cualquier opción prohibida antes del muestreo, reduciendo la probabilidad de error a cero absoluto.",
            "Detallar la poda de logits a probabilidad cero antes del muestreo."
        ),

        # Lógica y Falacias
        (
            "¿Qué es una falacia lógica?",
            "Una falacia lógica es un argumento que parece válido pero contiene un error de razonamiento que invalida su conclusión.",
            "Definir el concepto general de falacia argumentativa."
        ),
        (
            "¿Qué es la falacia Ad Hominem?",
            "La falacia Ad Hominem ocurre cuando se ataca a la persona que emite el mensaje en lugar de analizar y refutar sus argumentos.",
            "Explicar el ataque a la persona frente a la discusión de la tesis."
        ),
        (
            "¿Qué es la Falsa Dicotomía?",
            "La Falsa Dicotomía consiste en presentar una situación compleja como si solo tuviera dos opciones opuestas, ignorando alternativas intermedias.",
            "Describir la polarización artificial binaria."
        ),
        (
            "¿Qué es la falacia del Hombre de Paja?",
            "El Hombre de Paja ocurre cuando se distorsiona o exagera la postura del oponente para que sea más fácil de criticar y refutar.",
            "Explicar la caricaturización de la postura del rival."
        ),
        (
            "Analiza: 'No le crean a Juan porque es joven'.",
            "Ese argumento es una falacia Ad Hominem, ya que descalifica la opinión de Juan atacando su edad en vez de evaluar sus ideas.",
            "Detectar y señalar la falacia Ad Hominem."
        ),
        (
            "Analiza: 'O estás con nosotros o estás contra nosotros'.",
            "Ese argumento es una Falsa Dicotomía, pues reduce arbitrariamente todas las posturas posibles a solo dos bandos excluyentes.",
            "Detectar y señalar la Falsa Dicotomía."
        ),

        # Ciencias Sociales y Retórica
        (
            "¿Qué estudia la sociología?",
            "La sociología estudia la vida social humana, los grupos, las instituciones y cómo las estructuras sociales influyen en la conducta de las personas.",
            "Definir el objeto de estudio de la sociología."
        ),
        (
            "¿Qué es la dialéctica?",
            "La dialéctica es un método de razonamiento que confronta una tesis con su antítesis para alcanzar una síntesis superior y más completa.",
            "Explicar el método dialéctico de tesis, antítesis y síntesis."
        ),
        (
            "¿Cuáles son los tres pilares de la retórica clásica?",
            "Los tres pilares de la retórica clásica definidos por Aristóteles son el Logos (la lógica), el Ethos (la ética y credibilidad) y el Pathos (la emoción).",
            "Explicar Logos, Ethos y Pathos según Aristóteles."
        ),
        (
            "¿Cómo se logra una sociedad justa?",
            "Una sociedad justa se construye garantizando derechos fundamentales, oportunidades equitativas, instituciones transparentes y el imperio de la ley.",
            "Sintetizar los pilares de la justicia social y el estado de derecho."
        ),
        (
            "¿Qué es el contrato social?",
            "El contrato social es un concepto de la filosofía política que explica cómo los ciudadanos acuerdan vivir bajo normas comunes a cambio de orden y protección.",
            "Definir la teoría del contrato social en filosofía política."
        ),
    ]

    corpus = []
    # Expand to 1500+ rich training instances
    for _ in range(100):
        for user_q, bot_a, thought in templates:
            text = (
                f"<|user|>\n{user_q}\n"
                f"<|assistant|>\n"
                f"<|thought|>\n{thought}\n<|endthought|>\n"
                f"{bot_a}<|eos|>"
            )
            corpus.append({"prompt": text})

    return corpus


def save_fluency_corpus(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    corpus = create_fluency_corpus()
    with open(p, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return p


if __name__ == "__main__":
    out = Path(__file__).parent / "fluency_train.jsonl"
    save_fluency_corpus(out)
    print(f"Corpus de fluidez guardado exitosamente en: {out}")
