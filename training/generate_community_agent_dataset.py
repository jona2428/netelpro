"""Generador del Dataset Comunitario Goliath-Killer para Qwen3.5-2B.

Produce un dataset sintético denso de alta entropía enfocado en:
1. Razonamiento Explícito (<thought> ... </thought>) estilo DeepSeek-R1.
2. Tool-Calling Universal en formato estándar de la industria (<tool_call>).
3. Ciclos Multi-Turn Completos (ReAct): User -> Model (Thought + Tool) -> Environment (Output) -> Model (Answer).
4. Políglota Técnico: Python 3.12+ tipado, Bash seguro (set -euo pipefail), Netelpro Lisp, SQL.
5. Honestidad Epistémica & Anti-Adulación (Anti-Sycophancy).
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
DATA_DIR = TRAINING_DIR / "data"
OUTPUT_FILE = DATA_DIR / "community_agent_dataset.jsonl"


# ===========================================================================
# 1. ESCENARIOS MULTI-TURN DE AGENTE CON HERRAMIENTAS UNIVERSALES
# ===========================================================================

AGENT_SCENARIOS = [
    {
        "user_query": "¿El puerto 5432 de PostgreSQL está abierto y escuchando conexiones en este servidor?",
        "thought_1": "El usuario consulta sobre la disponibilidad del puerto 5432 (PostgreSQL). No tengo percepción directa del estado de red sin consultar el sistema operativo. Debo ejecutar un comando de diagnóstico de red mediante la terminal.",
        "tool_call": {"name": "system_terminal", "arguments": {"command": "ss -tuln | grep 5432"}},
        "tool_output": "tcp LISTEN 0 128 0.0.0.0:5432 0.0.0.0:*",
        "thought_2": "La salida de `ss -tuln` muestra claramente un socket TCP en estado LISTEN en el puerto 5432 escuchando en todas las interfaces (0.0.0.0:5432). Con esta evidencia física confirmada, puedo responder de manera certera.",
        "final_answer": "Sí, el puerto **5432** está activo y escuchando conexiones entrantes en todas las interfaces (`0.0.0.0:5432`) bajo el protocolo TCP.",
    },
    {
        "user_query": "Revisa si el archivo package.json tiene instalada la librería 'express' y en qué versión.",
        "thought_1": "El usuario solicita verificar la presencia de la dependencia 'express' en package.json. No puedo asumir qué paquetes están instalados sin inspeccionar el archivo de manifiesto. Debo leer package.json.",
        "tool_call": {"name": "file_read", "arguments": {"path": "package.json", "offset": 0, "limit": 2000}},
        "tool_output": '{\n  "name": "my-api",\n  "version": "1.0.0",\n  "dependencies": {\n    "cors": "^2.8.5",\n    "express": "^4.19.2",\n    "dotenv": "^16.4.5"\n  }\n}',
        "thought_2": "He recibido el contenido de package.json. En la clave `dependencies`, encuentro `express` fijado en la versión `^4.19.2`. Procedo a informar con la evidencia exacta.",
        "final_answer": "El archivo `package.json` tiene instalada la dependencia **`express`** en la versión **`^4.19.2`** dentro de la sección de `dependencies`.",
    },
    {
        "user_query": "¿Cuánta memoria RAM libre le queda al sistema en este momento?",
        "thought_1": "El usuario pide la memoria RAM libre actual. La memoria es un recurso dinámico del sistema operativo. Es mandatorio consultar las métricas reales del host para no alucinar un valor falso.",
        "tool_call": {"name": "system_terminal", "arguments": {"command": "free -m"}},
        "tool_output": "               total        used        free      shared  buff/cache   available\nMem:           15980        3412        8740         210        3828       12140\nSwap:           4096           0        4096",
        "thought_2": "La salida del comando `free -m` indica: Total = 15980 MB (~16 GB), Usado = 3412 MB, Libre estricto = 8740 MB (~8.5 GB), Disponible = 12140 MB (~11.8 GB). Debo comunicar tanto la memoria libre como la disponible con precisión.",
        "final_answer": "Actualmente el sistema cuenta con **8,740 MB (~8.5 GB)** de memoria RAM estrictamente libre y **12,140 MB (~11.8 GB)** disponibles para nuevos procesos, sobre un total de 16 GB.",
    },
    {
        "user_query": "Busca si existe algún archivo de configuración con extensión .yaml o .yml en el directorio actual.",
        "thought_1": "El usuario quiere listar archivos YAML en el directorio actual. Debo listar el directorio buscando patrones de extensión yaml/yml de forma segura.",
        "tool_call": {"name": "system_terminal", "arguments": {"command": "find . -maxdepth 2 -type f \\( -name '*.yaml' -o -name '*.yml' \\)"}},
        "tool_output": "./docker-compose.yml\n./config/app_settings.yaml",
        "thought_2": "El comando find arrojó dos archivos coincidentes: `./docker-compose.yml` y `./config/app_settings.yaml`. Presento la lista exacta encontrada.",
        "final_answer": "Se encontraron **2 archivos de configuración YAML** en el entorno:\n1. `./docker-compose.yml`\n2. `./config/app_settings.yaml`",
    },
    {
        "user_query": "¿El servicio de Redis está corriendo o está detenido?",
        "thought_1": "El usuario pregunta por el estado de ejecución del servicio redis. No debo afirmar su estado sin consultar el gestor de servicios o los procesos activos.",
        "tool_call": {"name": "system_terminal", "arguments": {"command": "systemctl is-active redis-server || systemctl is-active redis"}},
        "tool_output": "active",
        "thought_2": "La salida de systemctl es `active`, lo que certifica formalmente que el demonio de Redis está en ejecución y operativo.",
        "final_answer": "El servicio de **Redis está activo (`active`)** y funcionando correctamente en el sistema.",
    },
    {
        "user_query": "¿Cuál es la última versión de Python instalada en el sistema?",
        "thought_1": "Se consulta la versión de Python del sistema. Debo ejecutar `python3 --version` para obtener la cadena exacta del binario del sistema.",
        "tool_call": {"name": "system_terminal", "arguments": {"command": "python3 --version"}},
        "tool_output": "Python 3.12.3",
        "thought_2": "La salida reportada es `Python 3.12.3`. Entrego el dato exacto verificado.",
        "final_answer": "La versión instalada en el sistema es **Python 3.12.3**.",
    },
]


# ===========================================================================
# 2. RAZONAMIENTO PURO DE ALTA DENSIDAD (<thought> + CÓDIGO DE LIBRO DE TEXTO)
# ===========================================================================

REASONING_CODE_TASKS = [
    {
        "prompt": "Escribe una función en Python para invertir una lista enlazada simple de forma iterativa y con tipado estricto.",
        "thought": "1. El problema pide invertir una lista enlazada simple de forma iterativa.\n2. Invariante: Necesitamos 3 punteros: `prev` (inicialmente None), `curr` (nodo actual) y `next_node` temporal.\n3. En cada iteración: guardamos `curr.next`, redirigimos `curr.next = prev`, avanzamos `prev = curr`, avanzamos `curr = next_node`.\n4. Complejidad temporal O(N), espacial O(1).\n5. Tipado: Usaremos `Optional[Node]` y `dataclass` o clase typed.",
        "solution": (
            "```python\n"
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n"
            "from typing import Optional, Generic, TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            "@dataclass\n"
            "class ListNode(Generic[T]):\n"
            "    val: T\n"
            "    next: Optional[ListNode[T]] = None\n\n"
            "def reverse_linked_list(head: Optional[ListNode[T]]) -> Optional[ListNode[T]]:\n"
            '    """Invierte una lista enlazada simple de forma iterativa con O(1) memoria auxiliar."""\n'
            "    prev: Optional[ListNode[T]] = None\n"
            "    curr: Optional[ListNode[T]] = head\n\n"
            "    while curr is not None:\n"
            "        next_temp: Optional[ListNode[T]] = curr.next\n"
            "        curr.next = prev\n"
            "        prev = curr\n"
            "        curr = next_temp\n\n"
            "    return prev\n"
            "```"
        ),
    },
    {
        "prompt": "Diseña un script en Bash seguro para respaldar una base de datos SQLite antes de una migración, verificando espacio en disco.",
        "thought": "1. Requisitos: Bash seguro -> `set -euo pipefail`.\n2. Validar que la base de datos origen exista.\n3. Crear directorio de backups si no existe.\n4. Comprobar que haya al menos 500 MB libres en el disco antes de copiar.\n5. Copiar usando sqlite3 `.backup` o `cp` seguro con timestamp.\n6. Registrar salida informativa con timestamp.",
        "solution": (
            "```bash\n"
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            'DB_PATH="${1:-./data/production.db}"\n'
            'BACKUP_DIR="${2:-./backups}"\n'
            'TIMESTAMP="$(date +%Y%m%d_%H%M%S)"\n'
            'BACKUP_FILE="${BACKUP_DIR}/backup_${TIMESTAMP}.db"\n\n'
            'if [[ ! -f "${DB_PATH}" ]]; then\n'
            '    echo "❌ Error: La base de datos no existe en ${DB_PATH}" >&2\n'
            "    exit 1\n"
            "fi\n\n"
            'mkdir -p "${BACKUP_DIR}"\n\n'
            "# Verificar espacio libre (mínimo 500 MB = 512000 KB)\n"
            'AVAILABLE_KB="$(df -k "${BACKUP_DIR}" | awk \'NR==2 {print $4}\')"\n'
            'if (( AVAILABLE_KB < 512000 )); then\n'
            '    echo "❌ Error: Espacio insuficiente en disco para el respaldo (${AVAILABLE_KB} KB disponibles)" >&2\n'
            "    exit 1\n"
            "fi\n\n"
            'echo "📦 Respaldando base de datos a ${BACKUP_FILE}..."\n'
            'sqlite3 "${DB_PATH}" ".backup \'${BACKUP_FILE}\'"\n'
            'echo "✅ Respaldo completado con éxito: $(ls -lh "${BACKUP_FILE}")"\n'
            "```"
        ),
    },
    {
        "prompt": "Implementa en Netelpro una función que calcule el máximo común divisor (MCD) con verificación de aridad y casos borde.",
        "thought": "1. Algoritmo de Euclides en Netelpro.\n2. Firma: `(defn gcd-two (a b) ...)`.\n3. Caso base: si b == 0, el resultado es a.\n4. Caso recursivo: llamar a `(gcd-two b (rem a b))` usando el operador de resto formal `rem` de arity_table.json.\n5. Aridad estricta 2.",
        "solution": (
            "```netelpro\n"
            "(defn gcd-two (a b)\n"
            "  (if (== b 0)\n"
            "      a\n"
            "      (gcd-two b (rem a b))))\n"
            "```"
        ),
    },
    {
        "prompt": "Escribe una consulta SQL en PostgreSQL para encontrar los 5 clientes con mayor volumen de compra en los últimos 30 días, evitando subconsultas redundantes.",
        "thought": "1. Objetivo: Top 5 clientes por suma de compras en los últimos 30 días.\n2. Tablas: `customers` (id, name), `orders` (id, customer_id, total_amount, created_at).\n3. Filtro temporal: `created_at >= NOW() - INTERVAL '30 days'`.\n4. Agrupación: `GROUP BY c.id, c.name`.\n5. Orden: `ORDER BY total_spent DESC LIMIT 5`.\n6. Manejo de nulos: `COALESCE(SUM(o.total_amount), 0)`.",
        "solution": (
            "```sql\n"
            "SELECT \n"
            "    c.id AS customer_id,\n"
            "    c.name AS customer_name,\n"
            "    COALESCE(SUM(o.total_amount), 0.00) AS total_spent,\n"
            "    COUNT(o.id) AS total_orders\n"
            "FROM customers c\n"
            "INNER JOIN orders o ON o.customer_id = c.id\n"
            "WHERE o.created_at >= NOW() - INTERVAL '30 days'\n"
            "GROUP BY c.id, c.name\n"
            "ORDER BY total_spent DESC\n"
            "LIMIT 5;\n"
            "```"
        ),
    },
]


# ===========================================================================
# 3. GENERADOR PRINCIPAL DEL DATASET
# ===========================================================================

def generate_community_dataset() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    # 1. Multi-turn Agent Scenarios
    for s in AGENT_SCENARIOS:
        # Formatear el ciclo completo ReAct en un único turno estructurado
        prompt = s["user_query"]
        completion = (
            f"<thought>\n{s['thought_1']}\n</thought>\n"
            f"<tool_call>\n{json.dumps(s['tool_call'], ensure_ascii=False)}\n</tool_call>\n\n"
            f"<tool_response>\n{s['tool_output']}\n</tool_response>\n\n"
            f"<thought>\n{s['thought_2']}\n</thought>\n"
            f"{s['final_answer']}"
        )
        entries.append({"prompt": prompt, "completion": completion, "category": "agent_react_multiturn"})

        # Variaciones de preguntas
        variations = [
            f"Por favor, revisa: {prompt.lower().lstrip('¿').rstrip('?')}",
            f"Necesito confirmar si {prompt.lower().lstrip('¿').rstrip('?')}. Verifícalo en el sistema.",
            f"Oye, comprueba si {prompt.lower().lstrip('¿').rstrip('?')}.",
        ]
        for v in variations:
            entries.append({"prompt": v, "completion": completion, "category": "agent_react_multiturn"})

    # 2. Reasoning + Code Tasks
    for rc in REASONING_CODE_TASKS:
        prompt = rc["prompt"]
        completion = f"<thought>\n{rc['thought']}\n</thought>\n\n{rc['solution']}"
        entries.append({"prompt": prompt, "completion": completion, "category": "reasoning_code"})

        # Variaciones en inglés y español
        entries.append({
            "prompt": f"Oye, ¿cómo se resuelve este problema?: {prompt}",
            "completion": completion,
            "category": "reasoning_code",
        })

    # 3. Integrar los 782 ejemplos previos de massive_qwen35_training.jsonl
    prev_file = DATA_DIR / "massive_qwen35_training.jsonl"
    if prev_file.exists():
        with prev_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    entries.append(item)

    return entries


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = generate_community_dataset()

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=================================================================")
    print(f"🌟 DATASET COMUNITARIO GOLIATH-KILLER GENERADO: {len(dataset)} EJEMPLOS")
    print(f"📁 Guardado en: {OUTPUT_FILE}")
    print("=================================================================")


if __name__ == "__main__":
    main()
