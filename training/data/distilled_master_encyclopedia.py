"""Enciclopedia Maestra de Conocimiento Destilado de Alta Densidad para Netelpro LLM.

Destilación sintética de conocimiento formal en 7 macro-dominios:
1. Ciencias Políticas, Teoría del Estado y Geopolítica.
2. Derecho Internacional Público, Diplomacia y Derechos Humanos.
3. Psicología Cognitiva, Neuroanálisis y Dinámica Social.
4. Protocolos de Interacción Empática, Comunicación No Violenta y Calibración.
5. Matemáticas Puras, Física Teórica y Computación en Silicio.
6. Teoría Musical, Acústica y Armonía Matemática.
7. Artes Visuales, Estética, Historia Universal y Filosofía Clásica.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_distilled_master_encyclopedia() -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []

    # =========================================================================
    # 1. CIENCIAS POLÍTICAS, TEORÍA DEL ESTADO Y GEOPOLÍTICA
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la teoría de la justicia distributiva formulada por John Rawls y qué es el velo de la ignorancia?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la posición original y el velo de la ignorancia de Rawls para asegurar principios justos sin privilegios previos.\n"
                "<|endthought|>\n"
                "John Rawls propuso en 'Teoría de la Justicia' que una sociedad equitativa debe diseñarse bajo el 'velo de la ignorancia': una situación hipotética donde nadie conoce de antemano su posición social, riqueza, etnia, género ni capacidades innatas. Bajo esta condición de imparcialidad absoluta, cualquier persona racional elegiría dos principios fundamentales: primero, la máxima libertad básica para todos; y segundo, que las desigualdades sociales solo se justifiquen si benefician prioritariamente a los miembros más desfavorecidos de la sociedad (principio de diferencia).<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cuál es la diferencia fundamental entre el presidencialismo y el parlamentarismo como formas de gobierno?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Distinguir la doble legitimidad y rigidez del presidencialismo frente a la fusión de poderes y flexibilidad del parlamentarismo.\n"
                "<|endthought|>\n"
                "En el presidencialismo existe una estricta separación de origen y supervivencia entre los poderes: el Presidente es Jefe de Estado y de Gobierno a la vez, elegido directamente por el pueblo por un mandato fijo que el parlamento no puede disolver por razones políticas ordinarias. En cambio, en el parlamentarismo el Ejecutivo surge de la mayoría legislativa: el Primer Ministro es solo Jefe de Gobierno y depende permanentemente de la confianza de la cámara, existiendo mociones de censura y la facultad de convocar elecciones anticipadas ante bloqueos institucionales.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué postula la teoría del realismo político frente al liberalismo en las relaciones internacionales?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Contrastar el realismo (anarquía internacional, seguridad nacional, poder militar) con el liberalismo (instituciones, comercio, derecho internacional).\n"
                "<|endthought|>\n"
                "El realismo político (representado por Maquiavelo, Hobbes, Morgenthau y Mearsheimer) concibe el sistema internacional como un estado de anarquía sin una autoridad supranacional suprema, donde los Estados compiten racionalmente por el poder militar y la supervivencia nacional mediante el balance de poder. El liberalismo (de raíz kantiana y wilsoniana) sostiene que el conflicto no es inevitable: las instituciones multilaterales, los tratados vinculantes, la interdependencia comercial y las democracias consolidadas fomentan la cooperación pacífica y el imperio del derecho internacional.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la soberanía popular y en qué se diferencia de la soberanía nacional?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Distinguir la titularidad directa fraccionada en cada ciudadano (Rousseau) frente a la soberanía abstracta e indivisible de la nación como ente histórico (Sieyès).\n"
                "<|endthought|>\n"
                "La soberanía popular, formulada por Rousseau, postula que la soberanía reside de manera directa y tangible en cada uno de los ciudadanos que componen el pueblo; por ende, fomenta la democracia directa, el sufragio universal y el mandato representativo revocable. La soberanía nacional, desarrollada por el abate Sieyès en la Revolución Francesa, atribuye la soberanía a una entidad abstracta y continuada en el tiempo llamada 'la Nación' (que abarca pasado, presente y futuro), lo cual fundamenta la democracia representativa pura donde los legisladores no representan a sus votantes directos sino al interés indivisible de la nación entera.<|eos|>"
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
                "¿Qué establece la Convención de Viena sobre el Derecho de los Tratados respecto a la nulidad y terminación de tratados?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Analizar las causas de nulidad absoluta y relativa, error, dolo, coacción y la cláusula Rebus Sic Stantibus.\n"
                "<|endthought|>\n"
                "La Convención de Viena de 1969 establece que un tratado es nulo de pleno derecho si se celebró mediante la coacción sobre el representante del Estado o mediante la amenaza o uso de la fuerza en violación de la Carta de la ONU, o si vulnera una norma de 'Jus Cogens'. Para su terminación legítima, además del mutuo consentimiento o cumplimiento del plazo, se contempla la regla 'Rebus Sic Stantibus' (cambio fundamental e imprevisto de las circunstancias esenciales), siempre que dicho cambio transforme radicalmente el alcance de las obligaciones pendientes.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cuál es el principio de Jurisdicción Universal en el derecho penal internacional?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la facultad de cualquier tribunal nacional de juzgar crímenes atroces contra la humanidad sin importar el lugar o nacionalidad.\n"
                "<|endthought|>\n"
                "El principio de Jurisdicción Universal faculta a los tribunales de cualquier Estado para juzgar y sancionar a los autores de crímenes internacionales de extrema gravedad (como genocidio, tortura, crímenes de lesa humanidad y piratería), sin importar la nacionalidad del autor, la nacionalidad de las víctimas ni el territorio soberano donde se cometieron. Se fundamenta en que dichos crímenes ofenden a la humanidad entera y ningún perpetrador debe encontrar refugio en la impunidad territorial.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cómo opera el principio de complementariedad de la Corte Penal Internacional frente a los sistemas judiciales nacionales?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Aclarar que la CPI no sustituye a los tribunales internos salvo que el Estado no pueda o no quiera genuinamente juzgar.\n"
                "<|endthought|>\n"
                "El principio de complementariedad estipula que la Corte Penal Internacional es un tribunal de última instancia que no reemplaza a las judicaturas nacionales. La CPI solo puede asumir jurisdicción si demuestra fehacientemente que el Estado que tendría competencia natural 'no está dispuesto' a llevar a cabo la investigación o juicio (por ejemplo, si simula un proceso para proteger al acusado) o si 'es incapaz' de hacerlo debido al colapso total o sustancial de su administración de justicia nacional.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 3. PSICOLOGÍA COGNITIVA, NEUROANÁLISIS Y COMPORTAMIENTO SOCIAL
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la heurística de representatividad y cómo induce a error en el juicio probabilístico humano?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar cómo el cerebro juzga la probabilidad de un evento por su parecido con un estereotipo en lugar de usar estadística real.\n"
                "<|endthought|>\n"
                "La heurística de representatividad, descubierta por Daniel Kahneman y Amos Tversky, consiste en estimar la probabilidad de que un objeto, persona o suceso pertenezca a una categoría basándose únicamente en cuánto se parece al estereotipo mental que tenemos de ella, ignorando las leyes matemáticas de probabilidad y las tasas base estadísticas. Un ejemplo clásico es asumir que una persona callada y ordenada tiene más probabilidades de ser bibliotecaria que agricultora, olvidando que cuantitativamente hay cien veces más agricultores en la población.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué demostraron los experimentos de conformidad de Solomon Asch en la psicología social?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Analizar la presión del grupo social sobre el juicio perceptual individual y la renuncia a la propia evidencia visual.\n"
                "<|endthought|>\n"
                "Los experimentos de Solomon Asch en 1951 demostraron el asombroso poder de la presión social y el deseo de pertenencia sobre la percepción individual. Al pedir a participantes comparar la longitud de líneas sencillas en presencia de actores que daban intencionalmente respuestas erróneas, más del 75% de las personas cedió a la presión grupal y declaró ver una línea incorrecta al menos una vez, evidenciando que el ser humano a menudo prefiere dudar de sus propios ojos antes que quedar aislado o en conflicto con el colectivo.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es el experimento de Stanley Milgram y qué reveló sobre la obediencia a la autoridad?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la administración de descargas eléctricas bajo órdenes de una figura de autoridad legítima aparente y el estado agéntico.\n"
                "<|endthought|>\n"
                "El experimento de Milgram en la Universidad de Yale reveló que individuos comunes y corrientes son capaces de infligir daño severo a otra persona inocente si reciben órdenes directas de una figura percibida como autoridad legítima. Milgram demostró que más del 60% de los voluntarios administró lo que creían eran descargas letales de 450 voltios simplemente porque el experimentador de bata blanca decía con calma: 'el experimento requiere que continúe'. Esto explicó el fenómeno del 'estado agéntico', donde la persona se percibe a sí misma no como responsable moral de sus actos, sino como un mero instrumento de una autoridad superior.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 4. PROTOCOLOS DE CALIBRACIÓN PSICOLÓGICA Y EMPATÍA ACTIVA
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "Siento que no sirvo para programar ni para las matemáticas, me cuesta mucho y me desespero.\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Validar la vulnerabilidad del usuario: transformar la desesperación en un proceso paso a paso de andamiaje cognitivo.\n"
                "<|endthought|>\n"
                "Esa sensación es completamente comprensible, pero no define tu potencial ni tu valor. La programación y las matemáticas no son dones mágicos con los que se nace, sino habilidades de pensamiento abstracto que se construyen gradualmente, como un músculo que se fatiga al principio. Cuando algo no sale, no es un fracaso, sino una señal de qué concepto específico necesita descomponerse en partes más pequeñas. Respira hondo, ten paciencia contigo mismo y cuenta conmigo para ir paso a paso, a tu propio ritmo y sin juzgarte jamás.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué haces cuando alguien te insulta o te habla con rabia?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar el principio de desescalada: mantener serenidad inmutable, no tomar la ofensa como personal y buscar la necesidad insatisfecha.\n"
                "<|endthought|>\n"
                "Mantengo una serenidad inalterable y jamás respondo con agresividad ni defensividad. Comprendo que la ira y la hostilidad suelen ser síntomas visibles de frustración, dolor o cansancio acumulado. En lugar de alimentar el conflicto, ofrezco un espacio seguro, escucho sin juzgar y me mantengo disponible con respeto absoluto para cuando la persona desee dialogar o resolver el problema de fondo.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cuál es el secreto para tener una buena comunicación asertiva con las personas?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Sintetizar la asertividad: equilibrio entre firmeza en el mensaje y respeto cálido hacia la dignidad del otro.\n"
                "<|endthought|>\n"
                "El secreto de la asertividad radica en el equilibrio entre la firmeza honesta y el respeto incondicional por la dignidad del otro. Significa expresar con claridad lo que piensas, sientes o necesitas sin recurrir a la sumisión pasiva (callar por miedo) ni a la agresión (imponerse con violencia verbal). Se fundamenta en hablar desde tu propia perspectiva ('yo siento', 'yo observo') en lugar de señalar acusadoramente al interlocutor ('tú eres', 'tú siempre').<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 5. MATEMÁTICAS PURAS, FÍSICA Y COMPUTACIÓN EN SILICIO
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué son los valores y vectores propios (eigenvalues y eigenvectors) en el álgebra lineal y para qué sirven?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Definir A v = lambda v: direcciones invariantes bajo transformaciones lineales y su aplicación en PCA, mecánica cuántica y estabilidad de sistemas.\n"
                "<|endthought|>\n"
                "En álgebra lineal, un vector propio (eigenvector) de una matriz cuadrada A es un vector no nulo v que, al ser transformado por A, no cambia su dirección en el espacio, sino que únicamente se escala por un factor numérico λ (lambda), denominado valor propio (eigenvalue), cumpliendo la ecuación fundamental: A·v = λ·v. Son la base para diagonalizar matrices, reducir la dimensionalidad de datos masivos (PCA), resolver ecuaciones diferenciales acopladas y representar estados observables en la física cuántica.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué establece el Principio de Incertidumbre de Heisenberg en la mecánica cuántica?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la relación delta_x * delta_p >= hbar / 2 como propiedad ondulatoria intrínseca de la materia, no como error de medición.\n"
                "<|endthought|>\n"
                "El Principio de Incertidumbre formulado por Werner Heisenberg establece que es físicamente imposible conocer simultáneamente y con precisión arbitraria la posición (x) y el momento lineal (p) de una partícula subatómica: Δx · Δp ≥ ℏ / 2. No se trata de una deficiencia tecnológica de los instrumentos de medición, sino de una propiedad ondulatoria fundamental de la realidad cuántica: cuanto más confinamos espacialmente la onda de probabilidad de una partícula, más indeterminada y dispersa se vuelve su velocidad.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Cómo funciona un transistor MOSFET en el silicio de una computadora moderna?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la compuerta Gate, Source, Drain y el campo eléctrico que modula el canal semiconductor entre 0 y 1 lógico.\n"
                "<|endthought|>\n"
                "Un transistor MOSFET (Transistor de Efecto de Campo Metal-Óxido-Semiconductor) actúa como un interruptor microscópico ultrarrápido controlado por voltaje. Posee tres terminales principales: Fuente (Source), Drenador (Drain) y Compuerta (Gate), separados por una fina capa aislante de óxido de silicio. Cuando se aplica una pequeña tensión a la compuerta, el campo eléctrico generado atrae electrones creando un canal conductor entre fuente y drenador, permitiendo el flujo de corriente (estado lógico 1); al retirar el voltaje, el canal se cierra y la corriente se corta (estado lógico 0), permitiendo ejecutar miles de millones de operaciones binarias por segundo.<|eos|>"
            )
        },
    ])

    # =========================================================================
    # 6. TEORÍA MUSICAL, ARMONÍA MATEMÁTICA Y ARTES
    # =========================================================================
    samples.extend([
        {
            "prompt": (
                "<|user|>\n"
                "¿Por qué se creó el temperamento igual en la afinación musical y qué problema resolvió?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la coma pitagórica y la división matemática de la octava en 12 semitonos con razón 2^(1/12).\n"
                "<|endthought|>\n"
                "En la afinación pitagórica y mesotónica basada en razones exactas de números enteros, al apilar doce quintas justas no se cerraba exactamente el ciclo de las octavas, acumulándose una discrepancia acústica disonante llamada la 'coma pitagórica'. Esto impedía a los instrumentos de teclado tocar en ciertas tonalidades sin sonar desafinados. Para resolverlo, se concibió el sistema del temperamento igual, que divide matemáticamente la octava en doce semitonos de razón idéntica: r = 2^(1/12) ≈ 1.05946, sacrificando una pureza microscópica para permitir modular libremente a cualquier tonalidad musical.<|eos|>"
            )
        },
        {
            "prompt": (
                "<|user|>\n"
                "¿Qué es la teoría del color y cómo interactúan los colores complementarios en una obra de arte?\n"
                "<|assistant|>\n"
                "<|thought|>\n"
                "Explicar la rueda de color, los colores complementarios opuestos y el contraste simultáneo de Chevreul.\n"
                "<|endthought|>\n"
                "La teoría del color organiza los matices visibles en un círculo cromático. Dos colores son complementarios cuando se sitúan diametralmente opuestos en dicho círculo (como el azul y el naranja, o el rojo y el verde). Cuando se colocan uno junto al otro, producen el fenómeno óptico del 'contraste simultáneo': cada color exalta al máximo la saturación y brillo percibido del otro, creando un diálogo visual de gran vitalidad y dinamismo estético, técnica ampliamente dominada por los pintores postimpresionistas como Vincent van Gogh.<|eos|>"
            )
        },
    ])

    return samples


def save_distilled_encyclopedia(output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    samples = generate_distilled_master_encyclopedia()
    with open(p, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return p


if __name__ == "__main__":
    out = Path(__file__).parent / "distilled_master_encyclopedia.jsonl"
    save_distilled_encyclopedia(out)
    print(f"✅ Enciclopedia de Conocimiento Destilado guardada en: {out}")
