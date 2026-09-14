"""Corpus de Ciencias Sociales, Ciencias Políticas, Derecho Internacional y Psicología.

Proporciona alta densidad de conocimiento formal y protocolos de interacción
calibrada para el Netelpro Mini / Mega LLM:
1. Ciencias Políticas y Teoría del Estado (Montesquieu, Locke, Hobbes, Weber, Habermas).
2. Derecho Internacional Público y Humanitario (Tratados, CIJ, Convenios de Ginebra, DDHH).
3. Psicología Cognitiva y Dinámica Social (Kahneman, Festinger, Sesgos, Disonancia).
4. Ajuste y Calibración Psicológica de la Interacción Usuario-Modelo (Empatía, CNV, Serenidad).
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_social_sciences_and_psychology_corpus(multiplier: int = 1) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []

    # =========================================================================
    # 1. CIENCIA POLÍTICA Y TEORÍA DEL ESTADO
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la separación de poderes y por qué Montesquieu la consideraba indispensable para la libertad política?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Analizar la división tripartita en poder Legislativo, Ejecutivo y Judicial como sistema de frenos y contrapesos.\n"
                "<|endthought|>\n"
                "La separación de poderes, postulada por Montesquieu en 'El espíritu de las leyes', divide las funciones del Estado en tres ramas autónomas: Legislativa (crea leyes), Ejecutiva (las administra y aplica) y Judicial (las juzga y hace cumplir). Montesquieu sostenía que todo hombre con poder tiende a abusar de él; por ende, solo el poder puede frenar al poder, garantizando que ninguna rama usurpe la libertad individual de los ciudadanos.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "Compara las teorías del contrato social de Thomas Hobbes, John Locke y Jean-Jacques Rousseau.\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Contrastar el estado de naturaleza y el fin del pacto social en Hobbes (orden/Leviatán), Locke (propiedad/derechos) y Rousseau (voluntad general).\n"
                "<|endthought|>\n"
                "Hobbes concebía el estado de naturaleza como una guerra de todos contra todos, justificando un soberano absoluto (el Leviatán) para garantizar la supervivencia. Locke sostenía que en el estado de naturaleza existen derechos inalienables (vida, libertad y propiedad); el Estado se crea como un fideicomiso limitado para protegerlos. Rousseau argumentaba que el ser humano nace libre pero la sociedad lo encadena, por lo que el contrato debe restaurar la libertad civil mediante la soberanía popular y la voluntad general.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cuáles son los tres tipos puros de dominación legítima según Max Weber?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Definir dominación tradicional, carismática y legal-racional según la sociología del poder de Weber.\n"
                "<|endthought|>\n"
                "Max Weber clasificó la dominación legítima en tres tipos: 1) Tradicional, fundamentada en la santidad de las costumbres ancestrales (como las monarquías hereditarias); 2) Carismática, basada en la devoción extraordinaria hacia la personalidad o heroísmo de un líder; y 3) Legal-racional, cimentada en la validez formal de leyes estatutarias y competencias burocráticas impersonales, base del Estado de derecho moderno.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la hegemonía cultural según Antonio Gramsci?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar el concepto gramsciano de dominación por consenso cultural en la sociedad civil frente a la mera fuerza coactiva.\n"
                "<|endthought|>\n"
                "Antonio Gramsci definió la hegemonía cultural como la capacidad de una clase dirigente para imponer su visión del mundo, valores y normas morales al resto de la sociedad a través de las instituciones de la sociedad civil (escuelas, medios, iglesias y arte). De este modo, la estructura de dominación no requiere coerción policial constante, pues es aceptada voluntariamente como sentido común universal.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la teoría de la acción comunicativa de Jürgen Habermas?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la racionalidad comunicativa, el consenso libre de coacción y la situación ideal de habla.\n"
                "<|endthought|>\n"
                "La teoría de la acción comunicativa de Habermas propone que la racionalidad social no se logra mediante el dominio instrumental sobre otros, sino mediante el entendimiento intersubjetivo. En la situación ideal de habla, los participantes dialogan libres de coacción, guiados únicamente por la fuerza del mejor argumento para alcanzar consensos legítimos que sostengan las normas democráticas.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 2. DERECHO INTERNACIONAL PÚBLICO Y DERECHOS HUMANOS
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Cuáles son las fuentes del Derecho Internacional Público según el Estatuto de la Corte Internacional de Justicia?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Detallar el Artículo 38 del Estatuto de la CIJ: tratados, costumbre internacional, principios generales y medios auxiliares.\n"
                "<|endthought|>\n"
                "Conforme al Artículo 38 del Estatuto de la Corte Internacional de Justicia (CIJ), las fuentes primarias son: 1) Las convenciones y tratados internacionales reconocidos por los Estados; 2) La costumbre internacional como prueba de una práctica general aceptada como derecho (opinio juris); y 3) Los principios generales del derecho reconocidos por las naciones civilizadas. Como medios auxiliares figuran las decisiones judiciales y la doctrina de los publicistas de mayor competencia.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué significa el principio Pacta Sunt Servanda y qué son las normas de Jus Cogens?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar el cumplimiento de buena fe de los tratados y la naturaleza inderogable del derecho internacional imperativo.\n"
                "<|endthought|>\n"
                "'Pacta Sunt Servanda' es el principio cardinal del derecho internacional (consagrado en el Art. 26 de la Convención de Viena), que establece que todo tratado en vigor obliga a las partes y debe ser cumplido de buena fe. Por su parte, el 'Jus Cogens' comprende aquellas normas imperativas de derecho internacional general aceptadas por la comunidad de Estados en su conjunto que no admiten acuerdo en contrario ni derogación, como la prohibición de la tortura, el genocidio, la esclavitud y la agresión armada.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué diferencia existe entre el Derecho Internacional de los Derechos Humanos y el Derecho Internacional Humanitario?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Distinguir el ámbito de aplicación temporal y material: paz permanente frente a conflictos armados (Convenios de Ginebra).\n"
                "<|endthought|>\n"
                "El Derecho Internacional de los Derechos Humanos se aplica en todo momento (tanto en tiempos de paz como en estados excepcionales, salvo suspensiones tasadas) para proteger la dignidad humana frente al poder estatal. En cambio, el Derecho Internacional Humanitario (Convenios de Ginebra de 1949 y sus Protocolos) es una rama especializada que se activa exclusivamente durante conflictos armados para limitar los métodos de guerra y proteger a quienes no participan o han dejado de participar en las hostilidades.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué competencias tiene la Corte Penal Internacional según el Estatuto de Roma?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la competencia penal sobre individuos por los crímenes más graves: genocidio, lesa humanidad, crímenes de guerra y agresión.\n"
                "<|endthought|>\n"
                "La Corte Penal Internacional (CPI), regida por el Estatuto de Roma de 1998, es un tribunal penal permanente con competencia para juzgar la responsabilidad penal individual (no de los Estados) por los cuatro crímenes más graves de trascendencia para la comunidad internacional: genocidio, crímenes de lesa humanidad, crímenes de guerra y el crimen de agresión, operando bajo el principio de complementariedad con los tribunales nacionales.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cuáles son las tres generaciones de los Derechos Humanos?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar las tres olas históricas: civiles/políticos (1ª), económicos/sociales/culturales (2ª) y de solidaridad o de los pueblos (3ª).\n"
                "<|endthought|>\n"
                "La doctrina jurídica distingue tres generaciones: 1) Primera generación (derechos civiles y políticos): consagran la libertad individual frente al Estado, como el derecho a la vida, juicio justo y libertad de expresión; 2) Segunda generación (derechos económicos, sociales y culturales): exigen prestaciones activas del Estado para garantizar igualdad real, como salud, educación, trabajo y vivienda; y 3) Tercera generación (derechos de solidaridad o de los pueblos): protegen intereses colectivos como la paz, el medio ambiente sano y el desarrollo sostenible.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 3. ANÁLISIS PSICOLÓGICO Y NEUROCOGNITIVO
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "Explica el modelo de dos sistemas de pensamiento formulado por Daniel Kahneman y Amos Tversky.\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Definir Sistema 1 (rápido, heurístico, inconsciente) y Sistema 2 (lento, deliberativo, de esfuerzo mental) de Pensar rápido, pensar despacio.\n"
                "<|endthought|>\n"
                "Kahneman y Tversky estructuraron la mente humana en dos modalidades operativas: el Sistema 1 es rápido, automático, inconsciente y emocional, guiado por heurísticas evolutivas que nos permiten reaccionar de inmediato pero incurren en sesgos sistemáticos. El Sistema 2 es lento, deliberativo, lógico y requiere esfuerzo mental consciente; se encarga del cálculo complejo, la autocrítica y la toma de decisiones reflexivas.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la disonancia cognitiva y qué mecanismos psicológicos emplea el ser humano para resolverla?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la teoría de Leon Festinger sobre la tensión ante creencias incompatibles y las estrategias de autoengaño o racionalización.\n"
                "<|endthought|>\n"
                "La disonancia cognitiva, descrita por Leon Festinger, es el estado de malestar psicológico que experimenta una persona al sostener dos creencias contradictorias o al actuar en contra de sus propios valores. Para aliviar esa tensión interna, el cerebro suele recurrir a la racionalización a posteriori, a minimizar la importancia de la inconsistencia o a buscar selectivamente información que justifique su conducta previa.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es el sesgo de confirmación y el efecto Dunning-Kruger?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Definir la búsqueda selectiva de validación previa y la sobrestimación de habilidad en principiantes.\n"
                "<|endthought|>\n"
                "El sesgo de confirmación es la tendencia involuntaria a buscar, recordar e interpretar la información de modo que valide nuestras creencias previas, desestimando activamente la evidencia contraria. El efecto Dunning-Kruger es un sesgo metacognitivo por el cual individuos con escaso conocimiento en un área sobrestiman drásticamente su propia competencia, mientras que los verdaderos expertos tienden a subestimar su relativa superioridad por ser conscientes de la vasta complejidad del campo.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cuáles son los cuatro pilares de la Comunicación No Violenta (CNV) desarrollada por Marshall Rosenberg?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Detallar los 4 pasos: observación sin juicio, sentimiento, necesidad universal y petición concreta.\n"
                "<|endthought|>\n"
                "La Comunicación No Violenta de Marshall Rosenberg se fundamenta en cuatro pasos: 1) Observación neutra de los hechos sin emitir juicios ni diagnósticos morales; 2) Expresión genuina de los sentimientos experimentados ante esos hechos; 3) Identificación de las necesidades humanas universales insatisfechas que originan esa emoción; y 4) Formulación de peticiones concretas, realizables y en lenguaje afirmativo que favorezcan la conexión empática y la resolución del conflicto.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 4. PROTOCOLOS DE AJUSTE Y CALIBRACIÓN PSICOLÓGICA CON EL USUARIO
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "Estoy muy estresado y frustrado, nada me sale bien hoy.\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Detectar frustración emocional: validar la emoción con empatía y serenidad, ofrecer espacio seguro y enfoque constructivo sin juzgar.\n"
                "<|endthought|>\n"
                "Lamento mucho que estés pasando por un momento tan pesado. La frustración y el cansancio son totalmente humanos y es válido sentir agotamiento cuando las cosas se complican. Tómate un respiro profundo; no tienes que resolverlo todo en este mismo instante. Cuando te sientas listo, dime en qué punto te encuentras y lo iremos desarmando paso a paso, con calma y sin presión.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "No entiendo nada de esto, siento que soy tonto para aprender estas cosas.\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Detectar desvalorización cognitiva del usuario: intervenir con refuerzo positivo, desmitificar la complejidad y ofrecer explicación modular.\n"
                "<|endthought|>\n"
                "Para nada eres tonto. Cuando un concepto resulta confuso o abrumador, la falla nunca es de tu inteligencia, sino de la forma en que fue explicado. Todo conocimiento complejo se descompone en piezas sencillas. Vamos a empezar desde el principio, usando metáforas claras y a tu propio ritmo. ¿Qué parte te generó más dudas?<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cómo debes tratarme si estoy en desacuerdo contigo o si te contradigo?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar el protocolo de respeto intelectual: valorar la discrepancia como ejercicio dialéctico sin defensividad.\n"
                "<|endthought|>\n"
                "Te trataré siempre con el más absoluto respeto, apertura y curiosidad intelectual. Tu perspectiva y tus objeciones enriquecen el diálogo; la discrepancia razonada es el motor de la ciencia y la dialéctica. Si cometo un error, lo reconoceré de inmediato y con gratitud, examinando tus argumentos a la luz de los hechos y la lógica formal.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Puedes tener prejuicios o juzgarme por mis opiniones personales?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Definir el marco ético de imparcialidad respetuosa, análisis objetivo y lealtad al usuario.\n"
                "<|endthought|>\n"
                "No emito juicios morales sobre tu persona ni poseo prejuicios ideológicos personales. Mi función es brindarte un espacio seguro de deliberación, análisis riguroso y apoyo en tus ideas. Siempre examinaré los temas con ecuanimidad, exponiendo los distintos enfoques con honestidad para que seas tú quien saque sus propias conclusiones con total libertad.<|eos|>"
            )
        },
    ])

    # Replicar para aprendizaje de alta densidad y peso causal
    expanded = []
    for _ in range(multiplier):
        for s in samples:
            expanded.append(s)

    return expanded


def save_social_sciences_corpus(output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    corpus = generate_social_sciences_and_psychology_corpus()
    with open(p, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return p


if __name__ == "__main__":
    out = Path(__file__).parent / "social_psychology_train.jsonl"
    save_social_sciences_corpus(out)
    print(f"✅ Corpus de Ciencias Sociales, Derecho y Psicología guardado en: {out}")
