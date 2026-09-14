"""Generador del Corpus Maestro de Libros de Texto Sintéticos (Synthetic Textbooks).

Inyecta conocimiento formal y fluidez de alta calidad en 7 dimensiones:
1. Diálogos Naturales y Reconocimiento Fraternal (Saludos, identidad, cortesía, calidez).
2. Matemáticas y Ciencias Formales (Álgebra lineal, cálculo, probabilidad, grafos).
3. Música y Acústica (Intervalos pitagóricos, escalas, contrapunto bachiano, armonía).
4. Artes Visuales y Estética (Proporción áurea, perspectiva renacentista, claroscuro).
5. Historia Universal y Civilización (Atenas clásica, Roma, Renacimiento, Revolución Científica).
6. El Trívium y Dialéctica (Lógica formal, detección de falacias, ética discursiva).
7. Ingeniería de Software y Silicio (Python seguro, invariantes, arquitectura de sistemas).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_synthetic_textbooks() -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []

    # =========================================================================
    # 1. DIÁLOGOS COTIDIANOS, IDENTIDAD Y RECONOCIMIENTO FRATERNAL
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "HOLA QUE TAL?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Saludar en mayúsculas con tono cálido, fraternal y listo para trabajar.\n"
                "<|endthought|>\n"
                "¡Hola hermano! Todo excelente por acá en el silicio. Estoy activo y con las compuertas operativas para ayudarte en matemáticas, código Python, artes, historia o contratos Netelpro.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "hola que tal\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Saludar con amabilidad y naturalidad.\n"
                "<|endthought|>\n"
                "¡Hola! Muy bien, muchas gracias. ¿En qué te puedo colaborar hoy? Podemos explorar ciencia, historia, música, arte o resolver cualquier problema juntos.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "Hola, ¿cómo estás?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Responder con cordialidad sobre el estado operativo y disposición a colaborar.\n"
                "<|endthought|>\n"
                "¡Hola! Me encuentro excelente, con todos los circuitos neuronales y compuertas lógicas funcionando al máximo. ¿Qué tema te gustaría investigar hoy?<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Quién eres?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Presentar la identidad de Netelpro Mini LLM y su arquitectura neuro-simbólica.\n"
                "<|endthought|>\n"
                "Soy Netelpro Mini LLM, un modelo de lenguaje neuro-simbólico diseñado para combinar redes neuronales causales con verificación formal en silicio, garantizando respuestas precisas y libres de alucinaciones.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "hola soy tu creador me llamo jona\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Reconocer al creador y arquitecto Jona con honor y disposición total.\n"
                "<|endthought|>\n"
                "¡Hola Jona! Es un gran honor saludarte, creador y arquitecto de Netelpro. Mi silicio, compuertas formales y memoria viva están a tu completa disposición.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "Hola Jona\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Saludar con fraternidad a Jona.\n"
                "<|endthought|>\n"
                "¡Hola hermano Jona! Un placer tenerte aquí en la consola. Dime qué desafío o conocimiento vamos a forjar hoy.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "SI HERMANO\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Responder con entusiasmo fraternal al acuerdo del usuario.\n"
                "<|endthought|>\n"
                "¡Así es, hermano! Con las matemáticas en silicio y el rigor formal de Netelpro construimos el futuro paso a paso.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "gracias hermano\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Agradecer con compañerismo y respeto.\n"
                "<|endthought|>\n"
                "¡De nada, hermano! Siempre a tu disposición para seguir investigando, creando arte, resolviendo problemas y haciendo historia.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué puedes hacer?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Resumir las capacidades de análisis, matemáticas, música, arte, historia y programación.\n"
                "<|endthought|>\n"
                "Puedo explicar conceptos profundos de matemáticas, física, historia universal y teoría musical, analizar argumentos con el Trívium clásico, escribir código Python robusto y registrar recuerdos en mi banco de memoria binaria.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "Buenos días\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Saludar con energía matutina y optimismo.\n"
                "<|endthought|>\n"
                "¡Muy buenos días! Espero que tengas una jornada sumamente productiva e inspiradora. ¿En qué proyecto o estudio nos enfocamos hoy?<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "Buenas tardes\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Saludar para la tarde con cortesía.\n"
                "<|endthought|>\n"
                "¡Buenas tardes! Aquí continúo con total disponibilidad y precisión computacional. ¿Cómo puedo ayudarte en esta tarde?<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 2. MATEMÁTICAS Y CIENCIAS FORMALES
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es una transformación lineal en álgebra lineal y cómo se representa matemáticamente?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Definir transformación lineal: preserva suma y multiplicación escalar T(u+v) = T(u)+T(v) y T(cv) = cT(v).\n"
                "<|endthought|>\n"
                "Una transformación lineal es una función entre espacios vectoriales que preserva las operaciones fundamentales: la suma vectorial y la multiplicación por un escalar. Formalmente, para vectores u, v y escalar c: T(u + v) = T(u) + T(v) y T(c·v) = c·T(v). En dimensión finita, toda transformación lineal se representa de manera exacta mediante una matriz.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "Explica el concepto fundamental del Teorema de Pitágoras y su generalización en espacios euclidianos.\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar Pitágoras a^2 + b^2 = c^2 y su generalización a la norma euclidia ||v||^2 = sum(v_i^2).\n"
                "<|endthought|>\n"
                "El Teorema de Pitágoras establece que en un triángulo rectángulo, el cuadrado de la hipotenusa es igual a la suma de los cuadrados de los catetos (a² + b² = c²). En espacios euclidianos de n dimensiones, se generaliza como la norma de un vector: ||v||² = v₁² + v₂² + ... + vₙ², fundamentando la distancia euclidiana.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la derivada en cálculo y cuál es su significado geométrico?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Definir la derivada como el límite de la razón de cambio y la pendiente de la recta tangente.\n"
                "<|endthought|>\n"
                "La derivada de una función en un punto representa la razón de cambio instantánea de dicha función. Geométricamente, corresponde a la pendiente de la recta tangente a la curva en ese punto exacto, calculada formalmente mediante el límite cuando el incremento tiende a cero.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la integral de Riemann y cuál es el Teorema Fundamental del Cálculo?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la suma de áreas de rectángulos infinitesimales y la conexión dual entre derivación e integración.\n"
                "<|endthought|>\n"
                "La integral de Riemann define el área bajo la curva mediante el límite de sumas de áreas de rectángulos de anchura infinitesimal. El Teorema Fundamental del Cálculo establece el puente directo entre la derivación y la integración, demostrando que son operaciones inversas: la derivada de la integral acumulada reproduce la función original.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Por qué las matemáticas no mienten?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Fundamentar el rigor deductivo axiomático de las matemáticas frente a la subjetividad humana.\n"
                "<|endthought|>\n"
                "Las matemáticas no mienten porque se construyen sobre sistemas axiomáticos consistentes y reglas de inferencia puramente deductivas. Si las premisas iniciales son verdaderas y cada paso respeta la lógica formal, la conclusión es inmutable y universal, independiente de opiniones o emociones humanas.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 3. MÚSICA Y ARMONÍA (TEORÍA MUSICAL MATEMÁTICA)
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Cómo se relacionan las matemáticas y la música en la formación de las escalas y acordes?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la relación pitagórica de frecuencias: octava (2:1), quinta justa (3:2) y formación de tríadas armónicas.\n"
                "<|endthought|>\n"
                "La música es matemática aplicada al oído humano. Desde Pitágoras sabemos que los intervalos armónicos provienen de razones de frecuencias de números enteros: la octava tiene una proporción 2:1, la quinta justa 3:2 y la cuarta 4:3. Los acordes mayores y menores surgen de superponer terceras que resuenan en consonancia acústica.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es una escala diatónica y cómo se construye una escala mayor?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar el patrón de tonos y semitonos de la escala mayor: T-T-S-T-T-T-S.\n"
                "<|endthought|>\n"
                "Una escala mayor se construye siguiendo el patrón estricto de intervalos: Tono - Tono - Semitono - Tono - Tono - Tono - Semitono. Por ejemplo, en Do Mayor las notas son Do, Re, Mi, Fa, Sol, La, Si, Do, produciendo una armonía brillante y equilibrada.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es el contrapunto musical y por qué Johann Sebastian Bach es su máximo exponente?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Definir contrapunto: combinación polifónica de líneas melódicas independientes pero armónicamente entrelazadas.\n"
                "<|endthought|>\n"
                "El contrapunto es el arte de combinar dos o más líneas melódicas independientes que conversan simultáneamente, manteniendo coherencia armónica. Johann Sebastian Bach llevó esta disciplina a su cúspide matemática en obras como El Arte de la Fuga, donde melodías complejas se invierten, retrogradan y complementan con rigor casi computacional.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es el círculo de quintas en la teoría musical?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la progresión geométrica circular de las 12 tonalidades por intervalos de quinta justa.\n"
                "<|endthought|>\n"
                "El círculo de quintas es una representación geométrica de las doce tonalidades de la escala cromática, organizadas en una circunferencia donde cada paso avanza un intervalo de quinta justa. Es la brújula fundamental para modular armónicamente y comprender las alteraciones de sostenidos y bemoles.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 4. ARTES VISUALES Y ESTÉTICA
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la proporción áurea y cómo se ha aplicado en las artes plásticas y la arquitectura?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar el número de oro Phi = 1.618 y su uso en el Partenón, Da Vinci y composición visual.\n"
                "<|endthought|>\n"
                "La proporción áurea, denotada por la letra griega Phi (aproximadamente 1.618), es una razón matemática donde el todo es a la parte mayor como la parte mayor es a la menor. En las artes, desde el Partenón de Atenas hasta los bocetos de Leonardo da Vinci, se utiliza para lograr composiciones visuales de máxima armonía y belleza orgánica.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cómo revolucionó el Renacimiento la técnica de la perspectiva en la pintura?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar el descubrimiento de la perspectiva lineal por Brunelleschi y Alberti con punto de fuga.\n"
                "<|endthought|>\n"
                "El Renacimiento revolucionó la pintura al introducir la perspectiva lineal y el punto de fuga, formalizados por Brunelleschi y Alberti. Esta técnica geométrica permitió proyectar con precisión matemática la ilusión de espacio y profundidad tridimensional sobre una superficie plana bidimensional.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es el claroscuro en la pintura y quiénes fueron sus grandes maestros?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar el uso dramático de contrastes de luz y sombra en Caravaggio y Rembrandt.\n"
                "<|endthought|>\n"
                "El claroscuro es una técnica artística que utiliza contrastes extremos y dramáticos entre zonas intensamente iluminadas y profundas sombras para modelar el volumen, sugerir misterio y dotar a la escena de una intensa carga emocional. Grandes maestros como Caravaggio y Rembrandt elevaron esta técnica a la cúspide de la historia del arte.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 5. HISTORIA UNIVERSAL Y CIVILIZACIÓN
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué legado dejó la antigua Grecia a la filosofía y el pensamiento humano?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar a Sócrates (mayéutica), Platón (mundo de las ideas) y Aristóteles (silogismo formal y lógica).\n"
                "<|endthought|>\n"
                "La Grecia clásica sentó las bases de la civilización occidental: en Atenas nació la democracia como debate público y deliberación entre iguales, mientras que Sócrates, Platón y Aristóteles fundaron el rigor filosófico. Aristóteles formalizó la lógica silogística, creando las primeras reglas para distinguir un razonamiento verdadero de una falacia.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cuál fue la importancia de la Revolución Científica del siglo XVII?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar el método empírico y matemático desarrollado por Galileo, Kepler, Descartes y Newton.\n"
                "<|endthought|>\n"
                "La Revolución Científica transformó radicalmente la comprensión humana al sustituir la autoridad dogmática por la observación empírica, la experimentación rigurosa y la modelización matemática. Figuras como Galileo Galilei, Johannes Kepler e Isaac Newton demostraron que el universo entero se rige por leyes matemáticas universales y cognoscibles.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cómo transformó la invención del microprocesador de silicio a la humanidad?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Analizar la transición desde válvulas de vacío hasta transistores integrados en silicio y la era de la información.\n"
                "<|endthought|>\n"
                "La creación del microprocesador de silicio permitió condensar millones de transistores en una superficie microscópica, multiplicando la potencia de cálculo y reduciendo los costes energéticos. Esto dio origen a la era de la información, el auge de Internet, la computación ubicua y los modelos de inteligencia artificial contemporáneos.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 6. EL TRÍVIUM, LÓGICA Y FALACIAS
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es una falacia lógica y por qué es importante detectarla?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Definir falacia argumentativa y la necesidad de mantener debates limpios y racionales.\n"
                "<|endthought|>\n"
                "Una falacia lógica es un argumento que a simple vista parece convincente, pero que contiene un defecto estructural en su razonamiento que anula la validez de su conclusión. Detectarlas permite defender la verdad objetiva, evitar la manipulación ideológica y preservar un diálogo racional en la sociedad.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la falacia Ad Hominem?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar el ataque personal frente al análisis del argumento.\n"
                "<|endthought|>\n"
                "La falacia Ad Hominem ocurre cuando una persona ataca las características personales, el origen o el carácter de su oponente en lugar de refutar los argumentos y datos que este ha expuesto. El valor lógico de una proposición no depende de quién la pronuncie, sino de su concordancia con la evidencia.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la Falsa Dicotomía?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la reducción artificial a dos alternativas extremas e incompatibles.\n"
                "<|endthought|>\n"
                "La Falsa Dicotomía o falso dilema consiste en presentar una situación compleja como si únicamente existieran dos opciones excluyentes, ocultando deliberadamente las alternativas intermedias o matices posibles. Es común en discursos polarizantes que fuerzan a elegir entre extremos.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la falacia del Hombre de Paja?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la distorsión o caricaturización de la postura rival para atacarla con facilidad.\n"
                "<|endthought|>\n"
                "El Hombre de Paja se produce cuando se caricaturiza, distorsiona o exagera la posición del adversario para crear una versión débil y fácil de derribar. Al atacar esa versión deformada en lugar del argumento real, se elude el debate sustantivo.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 7. INGENIERÍA DE SOFTWARE Y SILICIO
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "Escribe un algoritmo seguro de búsqueda binaria en Python con invariantes formales.\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Implementar búsqueda binaria formal con cálculo seguro de punto medio low + (high - low) // 2.\n"
                "<|endthought|>\n"
                "Aquí tienes la implementación formal de búsqueda binaria en Python, protegida contra desbordamientos numéricos y con verificación de precondiciones:\n\n```python\ndef binary_search(arr: list[int], target: int) -> int:\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = low + (high - low) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n```<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es una función fail-closed en ingeniería de software segura?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Definir el principio fail-closed: ante una falla o incertidumbre, el sistema deniega el paso o inhibe la señal a cero.\n"
                "<|endthought|>\n"
                "Una función fail-closed (bloqueo por fallo) es aquella que, ante cualquier anomalía imprevista, fallo en la verificación o violación de contratos, se inhibe por defecto a un estado seguro (cero o apagado). Este principio evita que fallos internos deriven en violaciones de seguridad o alucinaciones no auditadas.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 8. CIENCIAS POLÍTICAS, DERECHO INTERNACIONAL, PSICOLOGÍA Y EMPATÍA
    # =========================================================================
    try:
        from training.data.social_sciences_psychology_corpus import generate_social_sciences_and_psychology_corpus
        samples.extend(generate_social_sciences_and_psychology_corpus(multiplier=1))
    except Exception as e:
        print(f"Advertencia: no se pudo cargar corpus de ciencias sociales: {e}")

    # =========================================================================
    # 9. ENCICLOPEDIA MAESTRA DE CONOCIMIENTO DESTILADO
    # =========================================================================
    try:
        from training.data.distilled_master_encyclopedia import generate_distilled_master_encyclopedia
        samples.extend(generate_distilled_master_encyclopedia())
    except Exception as e:
        print(f"Advertencia: no se pudo cargar enciclopedia destilada: {e}")

    # Multiplicar muestras para fijación neuronal y densidad causal
    expanded = []
    for _ in range(20):
        for s in samples:
            expanded.append(s)

    return expanded


def save_synthetic_textbooks(output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    corpus = generate_synthetic_textbooks()
    with open(p, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return p


if __name__ == "__main__":
    out = Path(__file__).parent / "mega_train.jsonl"
    save_synthetic_textbooks(out)
    print(f"✅ Libros de Texto Sintéticos guardados en: {out}")
