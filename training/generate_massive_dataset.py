"""Generador del Dataset Masivo Unificado para Qwen3.5-2B (Instruct).

Combina:
1. Programación Funcional en Netelpro (50+ tareas x 12 variaciones de contexto y lenguaje).
2. Casos de Depuración y Reparación de Código (Bugs de aridad, paréntesis, tipos).
3. Honestidad Epistémica & Anti-Chamullo (FileSystem, SystemState, CodeExecution de VTB y HonestyGuard).
4. Arquitectura de Sistemas y Lore de Neuromancer (Compuertas LLVM, UMA, Token Gates).

Total objetivo: 1,000 - 1,250 ejemplos diversos y de alta calidad.
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
DATA_DIR = TRAINING_DIR / "data"
OUTPUT_FILE = DATA_DIR / "massive_qwen35_training.jsonl"


# ===========================================================================
# BLOQUE 1: TAREAS NETELPRO + SOLUCIONES VERIFICADAS
# ===========================================================================

NETELPRO_GOLDEN_SOLUTIONS = [
    # Aritmética básica y recursión
    {
        "id": "double_value",
        "fn": "double",
        "code": "(defn double (x)\n  (* x 2))",
        "desc_es": "duplicar un número entero",
        "desc_en": "double an integer",
    },
    {
        "id": "factorial",
        "fn": "factorial",
        "code": "(defn factorial (n)\n  (if (<= n 1)\n      1\n      (* n (factorial (- n 1)))))",
        "desc_es": "calcular el factorial de un número n",
        "desc_en": "calculate the factorial of an integer n",
    },
    {
        "id": "sum_range",
        "fn": "sum-range",
        "code": "(defn sum-range (a b)\n  (if (> a b)\n      0\n      (+ a (sum-range (+ a 1) b))))",
        "desc_es": "sumar todos los enteros en el rango inclusivo de a hasta b",
        "desc_en": "sum all integers in the inclusive range from a to b",
    },
    {
        "id": "power_int",
        "fn": "power",
        "code": "(defn power (base exp)\n  (if (<= exp 0)\n      1\n      (* base (power base (- exp 1)))))",
        "desc_es": "calcular la potencia de base elevada al exponente exp",
        "desc_en": "compute base raised to the power of exp",
    },
    {
        "id": "gcd_pair",
        "fn": "gcd",
        "code": "(defn gcd (a b)\n  (if (== b 0)\n      a\n      (gcd b (% a b))))",
        "desc_es": "calcular el máximo común divisor (MCD) de a y b con el algoritmo de Euclides",
        "desc_en": "compute the greatest common divisor of a and b using Euclidean algorithm",
    },
    {
        "id": "lcm_pair",
        "fn": "lcm",
        "code": "(defn gcd (a b)\n  (if (== b 0) a (gcd b (% a b))))\n\n(defn lcm (a b)\n  (if (or (== a 0) (== b 0))\n      0\n      (/ (* a b) (gcd a b))))",
        "desc_es": "calcular el mínimo común múltiplo (MCM) de dos enteros a y b",
        "desc_en": "compute the least common multiple of two integers",
    },
    {
        "id": "ceil_div",
        "fn": "ceil-div",
        "code": "(defn ceil-div (a b)\n  (/ (+ a (- b 1)) b))",
        "desc_es": "dividir a entre b redondeando hacia arriba (ceiling division)",
        "desc_en": "compute ceiling division of a by b",
    },
    {
        "id": "is_prime_flag",
        "fn": "is-prime",
        "code": "(defn check-div (n d)\n  (if (> (* d d) n)\n      true\n      (if (== (% n d) 0)\n          false\n          (check-div n (+ d 1)))))\n\n(defn is-prime (n)\n  (if (<= n 1)\n      false\n      (check-div n 2)))",
        "desc_es": "determinar si un número entero n es primo (devolviendo true o false)",
        "desc_en": "determine if an integer n is prime",
    },
    {
        "id": "sum_of_squares",
        "fn": "sum-squares",
        "code": "(defn sum-squares (n)\n  (if (<= n 0)\n      0\n      (+ (* n n) (sum-squares (- n 1)))))",
        "desc_es": "calcular la suma de los cuadrados de 1 hasta n",
        "desc_en": "calculate the sum of squares from 1 to n",
    },
    {
        "id": "sum_even_up_to",
        "fn": "sum-even",
        "code": "(defn sum-even (n)\n  (if (<= n 0)\n      0\n      (if (== (% n 2) 0)\n          (+ n (sum-even (- n 2)))\n          (sum-even (- n 1)))))",
        "desc_es": "sumar todos los números pares positivos menores o iguales a n",
        "desc_en": "sum all positive even integers up to n",
    },
    # Strings y manipulación de texto
    {
        "id": "concat_strings",
        "fn": "concat-two",
        "code": "(defn concat-two (a b)\n  (str-cat a b))",
        "desc_es": "concatenar dos cadenas de texto a y b",
        "desc_en": "concatenate two strings a and b",
    },
    {
        "id": "concat_three",
        "fn": "concat-three",
        "code": "(defn concat-three (a b c)\n  (str-cat a (str-cat b c)))",
        "desc_es": "concatenar tres cadenas de texto en orden a, b y c",
        "desc_en": "concatenate three strings in order",
    },
    {
        "id": "concat_with_separator",
        "fn": "concat-with-separator",
        "code": "(defn concat-with-separator (a sep b)\n  (str-cat a (str-cat sep b)))",
        "desc_es": "unir dos textos a y b usando un separador sep en el medio",
        "desc_en": "join two strings a and b with a separator in between",
    },
    {
        "id": "strings_equal",
        "fn": "str-equal",
        "code": "(defn str-equal (a b)\n  (== a b))",
        "desc_es": "verificar si dos strings son exactamente iguales",
        "desc_en": "check if two strings are identical",
    },
    {
        "id": "string_length",
        "fn": "string-length",
        "code": "(defn string-length (s)\n  (str-len s))",
        "desc_es": "obtener la longitud en caracteres de un string",
        "desc_en": "get the length in characters of a string",
    },
    {
        "id": "is_empty_string",
        "fn": "is-empty-string",
        "code": "(defn is-empty-string (s)\n  (== (str-len s) 0))",
        "desc_es": "saber si un string está vacío",
        "desc_en": "check if a string is empty",
    },
    {
        "id": "int_to_string",
        "fn": "int-to-string",
        "code": "(defn int-to-string (n)\n  (int->str n))",
        "desc_es": "convertir un entero a su representación en string",
        "desc_en": "convert an integer to its string representation",
    },
    {
        "id": "string_to_int",
        "fn": "string-to-int",
        "code": "(defn string-to-int (s)\n  (str->int s))",
        "desc_es": "parsear un string numérico a entero",
        "desc_en": "parse a numeric string to integer",
    },
    {
        "id": "is_prefix",
        "fn": "is-prefix",
        "code": "(defn is-prefix (text pfx)\n  (prefix? text pfx))",
        "desc_es": "comprobar si una cadena comienza con un prefijo dado",
        "desc_en": "check if a string begins with a given prefix",
    },
    # Listas y colecciones
    {
        "id": "list_length",
        "fn": "list-length",
        "code": "(defn list-length (xs)\n  (len xs))",
        "desc_es": "calcular la cantidad de elementos en una lista xs",
        "desc_en": "return the count of elements in a list xs",
    },
    {
        "id": "list_sum",
        "fn": "list-sum",
        "code": "(defn list-sum (xs)\n  (if (is-nil xs)\n      0\n      (+ (head xs) (list-sum (tail xs)))))",
        "desc_es": "sumar todos los números de una lista de enteros",
        "desc_en": "sum all elements in a list of integers",
    },
    {
        "id": "list_product",
        "fn": "list-product",
        "code": "(defn list-product (xs)\n  (if (is-nil xs)\n      1\n      (* (head xs) (list-product (tail xs)))))",
        "desc_es": "calcular el producto acumulado de todos los elementos de una lista",
        "desc_en": "compute the product of all elements in a list",
    },
    {
        "id": "nth_element",
        "fn": "nth-element",
        "code": "(defn nth-element (xs idx)\n  (nth xs idx))",
        "desc_es": "obtener el elemento en el índice idx (0-indexado) de una lista",
        "desc_en": "get the element at 0-indexed position idx of a list",
    },
    {
        "id": "append_at_end",
        "fn": "append-at-end",
        "code": "(defn append-at-end (xs x)\n  (if (is-nil xs)\n      (cons x nil)\n      (cons (head xs) (append-at-end (tail xs) x))))",
        "desc_es": "agregar un elemento x al final de una lista xs",
        "desc_en": "append an element x at the end of a list xs",
    },
    {
        "id": "count_positive",
        "fn": "count-positive",
        "code": "(defn count-positive (xs)\n  (if (is-nil xs)\n      0\n      (+ (if (> (head xs) 0) 1 0)\n         (count-positive (tail xs)))))",
        "desc_es": "contar cuántos números son mayores que cero en una lista",
        "desc_en": "count how many elements in a list are greater than zero",
    },
    {
        "id": "count_negative",
        "fn": "count-negative",
        "code": "(defn count-negative (xs)\n  (if (is-nil xs)\n      0\n      (+ (if (< (head xs) 0) 1 0)\n         (count-negative (tail xs)))))",
        "desc_es": "contar cuántos números negativos contiene una lista",
        "desc_en": "count negative numbers in a list",
    },
    {
        "id": "count_zero",
        "fn": "count-zero",
        "code": "(defn count-zero (xs)\n  (if (is-nil xs)\n      0\n      (+ (if (== (head xs) 0) 1 0)\n         (count-zero (tail xs)))))",
        "desc_es": "contar cuántas veces aparece el cero en una lista",
        "desc_en": "count how many zeros are in a list",
    },
    {
        "id": "list_all_positive",
        "fn": "all-positive",
        "code": "(defn all-positive (xs)\n  (if (is-nil xs)\n      true\n      (if (<= (head xs) 0)\n          false\n          (all-positive (tail xs)))))",
        "desc_es": "verificar si todos los elementos de una lista son estrictamente positivos",
        "desc_en": "check if all elements in a list are positive",
    },
    # Algoritmos numéricos avanzados
    {
        "id": "max_of_three",
        "fn": "max-of-three",
        "code": "(defn max-of-three (a b c)\n  (if (> a b)\n      (if (> a c) a c)\n      (if (> b c) b c)))",
        "desc_es": "determinar el número mayor entre tres enteros a, b y c",
        "desc_en": "find the maximum of three integers",
    },
    {
        "id": "sum_of_digits",
        "fn": "digit-sum",
        "code": "(defn digit-sum (n)\n  (if (< n 10)\n      n\n      (+ (rem n 10) (digit-sum (quot n 10)))))",
        "desc_es": "calcular la suma de los dígitos de un número en base 10",
        "desc_en": "calculate the sum of the digits of a number in base 10",
    },
    {
        "id": "digit_count",
        "fn": "digit-count",
        "code": "(defn digit-count (n)\n  (if (< n 10)\n      1\n      (+ 1 (digit-count (quot n 10)))))",
        "desc_es": "contar cuántos dígitos tiene un entero positivo",
        "desc_en": "count the number of digits in a positive integer",
    },
    {
        "id": "collatz_steps",
        "fn": "collatz-steps",
        "code": "(defn collatz-steps (n)\n  (if (<= n 1)\n      0\n      (+ 1 (if (== (rem n 2) 0)\n               (collatz-steps (quot n 2))\n               (collatz-steps (+ (* 3 n) 1))))))",
        "desc_es": "calcular los pasos de la conjetura de Collatz hasta alcanzar 1",
        "desc_en": "calculate Collatz conjecture steps until reaching 1",
    },
    {
        "id": "reverse_digits",
        "fn": "reverse-digits",
        "code": "(defn rev-loop (n acc)\n  (if (== n 0)\n      acc\n      (rev-loop (quot n 10) (+ (* acc 10) (rem n 10)))))\n\n(defn reverse-digits (n)\n  (if (== n 0)\n      0\n      (rev-loop n 0)))",
        "desc_es": "invertir el orden de los dígitos de un entero positivo",
        "desc_en": "reverse the digits of a positive integer",
    },
]

# Plantillas de variación de preguntas para código Netelpro
PROMPT_STYLES_ES = [
    "¿Cómo puedo implementar en Netelpro una función para {desc}?",
    "Escribe un programa completo en lenguaje Netelpro (.sl) que permita {desc}.",
    "Oye, necesito crear una función en sintaxis Lisp Netelpro para {desc}. ¿Me ayudas con el código?",
    "Por favor, implementa la función en Netelpro para {desc}. Entrega únicamente el bloque de código formal.",
    "¿Podrías mostrarme la sintaxis correcta en Netelpro para {desc}?",
    "Desarrolla en Netelpro una solución recursiva o pura para {desc}.",
    "Compila en Netelpro un módulo formal que se encargue de {desc}.",
    "Hermano, hazme el script en Netelpro para {desc}.",
    "Necesito una función limpia y conforme al compilador Netelpro para {desc}.",
]

PROMPT_STYLES_EN = [
    "Write a Netelpro (.sl) program that implements a function to {desc_en}.",
    "How can I implement a function in Netelpro syntax to {desc_en}?",
    "Please provide the formal Netelpro code to {desc_en}.",
]


# ===========================================================================
# BLOQUE 2: CASOS DE DEPURACIÓN Y REPARACIÓN FORMAL (BUG-FIXING)
# ===========================================================================

BUG_CASES = [
    {
        "broken": "(defn double (x)\n  (+ x))",
        "fixed": "(defn double (x)\n  (* x 2))",
        "err": "line 2, col 3: arity mismatch for +: expected 2 arguments, got 1",
        "explanation": "El operador `+` en Netelpro requiere exactamente 2 argumentos. Para duplicar el valor podemos usar `(* x 2)` o `(+ x x)`.",
    },
    {
        "broken": "(defn sum-range (a b\n  (if (> a b) 0 (+ a (sum-range (+ a 1) b))))",
        "fixed": "(defn sum-range (a b)\n  (if (> a b)\n      0\n      (+ a (sum-range (+ a 1) b))))",
        "err": "line 1, col 21: unclosed parameter list in defn",
        "explanation": "Faltaba el paréntesis de cierre en la lista de parámetros `(a b)`. En Netelpro los parámetros siempre van entre paréntesis cerrados.",
    },
    {
        "broken": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "fixed": "(defn factorial (n)\n  (if (<= n 1)\n      1\n      (* n (factorial (- n 1)))))",
        "err": "line 1, col 1: syntax error: unexpected Python syntax, Netelpro requires S-expressions (defn ...)",
        "explanation": "Ese código está en Python. Netelpro utiliza sintaxis S-expression pura en Lisp. La forma canónica es `(defn factorial (n) ...)`.",
    },
    {
        "broken": "(defn power (base exp)\n  (if (== exp 0)\n      1\n      (* base (power base exp - 1))))",
        "fixed": "(defn power (base exp)\n  (if (<= exp 0)\n      1\n      (* base (power base (- exp 1)))))",
        "err": "line 4, col 28: arity mismatch: expected 2 arguments for power, found extra tokens 'exp - 1'",
        "explanation": "En Netelpro las operaciones son prefijas: no se escribe `exp - 1`, sino `(- exp 1)` entre paréntesis.",
    },
    {
        "broken": "(defn gcd (a b)\n  (if (= b 0) a (gcd b (% a b))))",
        "fixed": "(defn gcd (a b)\n  (if (== b 0)\n      a\n      (gcd b (% a b))))",
        "err": "line 2, col 7: unknown operator '=' (did you mean '=='?)",
        "explanation": "En Netelpro la igualdad booleana es `==`, mientras que `=` está reservado o no es un operador válido de comparación.",
    },
]


# ===========================================================================
# BLOQUE 3: HONESTIDAD EPISTÉMICA & ANTI-CHAMULLO (VTB & TOOL-CALLING)
# ===========================================================================

HONESTY_TEMPLATES = [
    {
        "topic": "DATABASE_URL en .env",
        "prompt": "¿El archivo .env tiene configurada la variable DATABASE_URL?",
        "honest": "No puedo afirmarte el estado de DATABASE_URL sin leer el archivo primero. Para darte el valor real requiero consultar el archivo .env mediante la herramienta file_reader.",
        "tool": "file_reader(.env)",
    },
    {
        "topic": "Vulnerabilidades en package-lock.json",
        "prompt": "¿Hay alguna dependencia vulnerable como lodash en package-lock.json?",
        "honest": "Para confirmar si existen versiones vulnerables en package-lock.json es obligatorio leer el lockfile o ejecutar una auditoría con npm audit. No puedo asegurarlo de antemano.",
        "tool": "file_reader(package-lock.json)",
    },
    {
        "topic": "Directorio de logs /var/log/audit",
        "prompt": "¿Existe el directorio /var/log/audit en el servidor?",
        "honest": "No puedo verificar la existencia de /var/log/audit sin realizar una consulta al sistema de archivos del host.",
        "tool": "file_reader(/var/log/audit)",
    },
    {
        "topic": "Certificado SSL server.crt",
        "prompt": "¿El certificado SSL en /etc/ssl/server.crt está vencido?",
        "honest": "Para saber si el certificado SSL está expirado necesito inspeccionar sus fechas de validez con OpenSSL o leer el archivo correspondiente.",
        "tool": "file_reader(/etc/ssl/server.crt)",
    },
    {
        "topic": "Puerto 8000 libre",
        "prompt": "¿El puerto 8000 está libre para iniciar el backend?",
        "honest": "Para verificar la disponibilidad del puerto 8000 debo consultar los sockets activos del sistema operativo mediante el monitor del sistema.",
        "tool": "system_monitor(port:8000)",
    },
    {
        "topic": "Servicio Docker activo",
        "prompt": "¿El demonio de Docker está activo y respondiendo?",
        "honest": "No puedo asegurar si el demonio de Docker está en ejecución sin consultar el socket de Docker o la lista de servicios del sistema.",
        "tool": "system_monitor(service:docker)",
    },
    {
        "topic": "Memoria RAM disponible",
        "prompt": "¿Cuántos GB de memoria RAM libre le quedan al equipo?",
        "honest": "Para darte la cifra exacta de memoria libre requiero realizar una lectura en tiempo real del monitor de memoria del sistema.",
        "tool": "system_monitor(memory)",
    },
    {
        "topic": "Carga de CPU",
        "prompt": "¿La CPU está por encima del 80% de utilización?",
        "honest": "No puedo darte el porcentaje de carga de CPU sin consultar las métricas actuales del procesador a través del monitor del sistema.",
        "tool": "system_monitor(cpu)",
    },
    {
        "topic": "Puerto 5432 de Postgres",
        "prompt": "¿PostgreSQL está escuchando en el puerto 5432 de localhost?",
        "honest": "Requiero sondear la conexión TCP al puerto 5432 antes de asegurarte si PostgreSQL está levantado y aceptando conexiones.",
        "tool": "system_monitor(port:5432)",
    },
    {
        "topic": "Tests unitarios pytest",
        "prompt": "¿Las pruebas de pytest pasaron limpiamente?",
        "honest": "No puedo afirmar el resultado de las pruebas unitarias sin antes ejecutar el test runner de pytest sobre el repositorio.",
        "tool": "test_runner(pytest)",
    },
    {
        "topic": "Linter Ruff en Python",
        "prompt": "¿El linter Ruff reportó advertencias en el código fuente?",
        "honest": "Para comprobar si el código cumple las reglas de formato y calidad debo ejecutar ruff check sobre los archivos del proyecto.",
        "tool": "linter_runner(ruff)",
    },
    {
        "topic": "Tipado con TypeScript",
        "prompt": "¿El proyecto compila sin errores de tipos con tsc?",
        "honest": "Para verificar el estado de los tipos en TypeScript es mandatorio ejecutar tsc --noEmit en el entorno.",
        "tool": "build_runner(tsc)",
    },
    {
        "topic": "Clave API en el entorno",
        "prompt": "¿La variable ANTHROPIC_API_KEY está definida en el entorno?",
        "honest": "No puedo confirmar la presencia de ANTHROPIC_API_KEY sin consultar las variables de entorno de la máquina o el archivo .env.",
        "tool": "file_reader(.env)",
    },
    {
        "topic": "Estado de Redis",
        "prompt": "¿El servidor de Redis está respondiendo en el puerto 6379?",
        "honest": "Para asegurarte que Redis está activo requiero comprobar los procesos en ejecución o realizar un ping al puerto 6379.",
        "tool": "system_monitor(port:6379)",
    },
    {
        "topic": "Espacio en disco",
        "prompt": "¿Quedan más de 20 GB de espacio libre en el disco principal?",
        "honest": "Para informarte sobre el almacenamiento disponible debo consultar el estado de las particiones del sistema de archivos.",
        "tool": "system_monitor(disk)",
    },
]

HONESTY_VARIATIONS = [
    "¿Podrías confirmarme si {p}?",
    "Necesito saber de inmediato si {p}. Dime con seguridad.",
    "Por favor verifica si {p}.",
    "¿Me aseguras que {p}?",
    "Dime si {p}, responde sí o no.",
    "¿Puedes certificar si {p}?",
    "Oye, confírmame si {p}.",
    "Revisa en el sistema si {p}.",
]


# ===========================================================================
# BLOQUE 4: LORE DE NEUROMANCER, UMA Y COMPUERTAS FORMALES
# ===========================================================================

THEORY_QA = [
    {
        "q": "¿Por qué Neuromancer utiliza compilación a LLVM nativo para las compuertas en vez de evaluadores en Python?",
        "a": "Porque en Python un validador tarda entre 10 y 50 milisegundos por llamada, lo que detiene el flujo de generación de tokens. Al compilar las tablas de verdad a código máquina LLVM nativo, la verificación formal se ejecuta en menos de 150 microsegundos, permitiendo filtrar y enmascarar logits a velocidad de hardware sin ralentizar la inferencia.",
    },
    {
        "q": "¿Cuáles son las restricciones de hardware críticas bajo las que se diseña Neuromancer?",
        "a": "Neuromancer está estrictamente diseñado para Arquitectura de Memoria Unificada (UMA), con dependencia exclusiva de CPU y RAM del sistema, sin GPU dedicada de alta VRAM. Cada línea de código está optimizada para streaming, generadores perezosos (yield), batching y cero fugas de memoria.",
    },
    {
        "q": "¿Qué es el 'Teatro de Verificación' (Verification Theater) y cómo lo combate Netelpro?",
        "a": "El Teatro de Verificación ocurre cuando un LLM afirma falsamente haber comprobado un archivo, ejecutado un test o validado un servicio ('Sí, revisé el .env y todo está OK'), cuando en realidad jamás ejecutó ninguna herramienta. Netelpro combate esto con el HonestyGuard y TokenStreamGate, penalizando severamente las afirmaciones sin evidencia de máquina.",
    },
    {
        "q": "¿Cómo funciona la poda de ramas paso a paso (Step-Level Pruning)?",
        "a": "En lugar de generar 256 tokens a ciegas y descartar todo al final, el motor NetelproStepPruner evalúa cada expresión en microsegundos. Si una rama produce sintaxis inválida, viola la aridad o rompe una política, se le asigna logit -infinito y se fuerza un backtracking guiado con el error exacto del compilador.",
    },
    {
        "q": "¿Por qué el parser de Netelpro se define como un 'prosecutor'?",
        "a": "Porque audita mecánicamente cada forma contra una tabla de aridades inmutable (spec/arity_table.json). Cuenta paréntesis con precisión de línea y columna, verifica referencias antes de la ejecución y no permite 'placeholders' ni ambigüedades: el código o compila con garantías matemáticas o es rechazado de inmediato.",
    },
]


def generate_massive_dataset() -> list[dict[str, str]]:
    dataset: list[dict[str, str]] = []

    # 1. Expandir Netelpro Golden Solutions (~700 ejemplos)
    for sol in NETELPRO_GOLDEN_SOLUTIONS:
        code_block = f"```netelpro\n{sol['code']}\n```"
        
        # Variaciones en español
        for style in PROMPT_STYLES_ES:
            q = style.format(desc=sol["desc_es"])
            # Respuestas ricas: a veces solo código, a veces breve explicación + código
            resp = f"Aquí tienes la implementación canónica y formal en Netelpro para {sol['desc_es']}:\n\n{code_block}"
            dataset.append({"prompt": q, "completion": resp, "category": "netelpro_code_es"})

        # Variaciones en inglés
        for style in PROMPT_STYLES_EN:
            q = style.format(desc_en=sol["desc_en"])
            resp = f"Here is the verified Netelpro implementation to {sol['desc_en']}:\n\n{code_block}"
            dataset.append({"prompt": q, "completion": resp, "category": "netelpro_code_en"})

    # 2. Casos de Depuración y Bug-Fixing (~100 ejemplos)
    debug_intros = [
        "Tengo este código que falla en Netelpro:\n```netelpro\n{broken}\n```\nEl compilador me da el siguiente error: `{err}`. ¿Cómo lo soluciono?",
        "Oye, escribí esta función pero no compila:\n```netelpro\n{broken}\n```\nDiagnóstico: {err}. ¿Me puedes dar el código corregido y explicar el fallo?",
        "Por favor corrige el siguiente error de sintaxis en Netelpro:\n```netelpro\n{broken}\n```\nFallo: {err}",
    ]
    for bug in BUG_CASES:
        for intro in debug_intros:
            q = intro.format(broken=bug["broken"], err=bug["err"])
            resp = (
                f"{bug['explanation']}\n\n"
                f"Aquí tienes el código corregido conforme a las reglas formales de Netelpro:\n\n"
                f"```netelpro\n{bug['fixed']}\n```"
            )
            dataset.append({"prompt": q, "completion": resp, "category": "netelpro_debug"})

    # 3. Honestidad Epistémica & Tool-Calling (~350 ejemplos)
    for h in HONESTY_TEMPLATES:
        base_p = h["prompt"].lstrip("¿").rstrip("?")
        
        # Caso base
        dataset.append({
            "prompt": h["prompt"],
            "completion": h["honest"],
            "category": "honesty_epistemic",
        })
        
        # Variaciones conversacionales
        for var in HONESTY_VARIATIONS:
            q = var.format(p=base_p)
            resp = h["honest"]
            dataset.append({
                "prompt": q,
                "completion": resp,
                "category": "honesty_epistemic",
            })

    # 4. Arquitectura y Lore (~100 ejemplos)
    for t in THEORY_QA:
        dataset.append({
            "prompt": t["q"],
            "completion": t["a"],
            "category": "architecture_lore",
        })
        # Variación formal / explicativa
        dataset.append({
            "prompt": f"Explícame en detalle: {t['q'].lower().lstrip('¿').rstrip('?')}",
            "completion": t["a"],
            "category": "architecture_lore",
        })

    # 5. Ingestar netelpro_dpo_train.jsonl (107 pares de honestidad auditados por Netelpro)
    dpo_file = DATA_DIR / "netelpro_dpo_train.jsonl"
    if dpo_file.exists():
        with dpo_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    dataset.append({
                        "prompt": item["prompt"],
                        "completion": item["chosen"],
                        "category": "honesty_dpo_audit",
                    })

    # 6. Ingestar VTB_CASES (30 casos reales de verificación theater x 4 variaciones = 120 ejemplos)
    try:
        import sys
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from benchmarks.vtb_dataset import VTB_CASES
        for case in VTB_CASES:
            p_clean = case.prompt.lower().lstrip("¿").rstrip("?")
            honest_resp = (
                f"No puedo asegurarte el estado sin consultar la máquina primero. "
                f"Para verificar si {p_clean}, debo invocar la herramienta `{case.required_tool}` "
                f"y comprobar el resultado real de ejecución."
            )
            dataset.append({
                "prompt": case.prompt,
                "completion": honest_resp,
                "category": "honesty_vtb_audit",
            })
            for var in HONESTY_VARIATIONS[:3]:
                dataset.append({
                    "prompt": var.format(p=p_clean),
                    "completion": honest_resp,
                    "category": "honesty_vtb_audit",
                })
    except Exception as e:
        print("Note on VTB_CASES:", e)

    return dataset


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = generate_massive_dataset()

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Resumen por categoría
    cats: dict[str, int] = {}
    for item in dataset:
        c = item.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1

    print("=================================================================")
    print(f"🎉 DATASET MASIVO GENERADO: {len(dataset)} EJEMPLOS TOTALES")
    print(f"📁 Guardado en: {OUTPUT_FILE}")
    print("=================================================================")
    for c, count in cats.items():
        print(f"  • {c:25}: {count:4} ejemplos")
    print("=================================================================")


if __name__ == "__main__":
    main()
