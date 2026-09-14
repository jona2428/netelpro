"""
Multilingual Programming Corpus for Teo v1.
Comprehensive, clean, idiomatic code examples across 8 core languages:
Python, JavaScript/TypeScript, C/C++, Rust, Go, SQL, Bash/PowerShell, HTML/CSS.
All samples follow the Neuro-Symbolic thought-directed chat protocol:
<|user|>\n...<|assistant|>\n<|thought|>\n...<|endthought|>\n...<|eos|>
"""

import json
from pathlib import Path

MULTILANG_CODER_SAMPLES = [
    # -------------------------------------------------------------
    # 1. PYTHON
    # -------------------------------------------------------------
    {
        "prompt": "Escribe un decorador en Python para reintentar una función con retroceso exponencial.",
        "thought": "Diseñar decorador con functools.wraps, argumentos max_retries y backoff_factor, capturar excepciones y aplicar time.sleep exponencial.",
        "response": (
            "Aquí tienes la implementación de un decorador de reintentos con retroceso exponencial (exponential backoff):\n\n"
            "```python\n"
            "import time\n"
            "import functools\n"
            "from typing import Callable, Any\n\n"
            "def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):\n"
            "    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:\n"
            "        @functools.wraps(func)\n"
            "        def wrapper(*args: Any, **kwargs: Any) -> Any:\n"
            "            delay = 1.0\n"
            "            for attempt in range(1, max_retries + 1):\n"
            "                try:\n"
            "                    return func(*args, **kwargs)\n"
            "                except Exception as err:\n"
            "                    if attempt == max_retries:\n"
            "                        raise err\n"
            "                    print(f\"Intento {attempt} falló: {err}. Reintentando en {delay}s...\")\n"
            "                    time.sleep(delay)\n"
            "                    delay *= backoff_factor\n"
            "        return wrapper\n"
            "    return decorator\n"
            "```\n\n"
            "**Aspectos clave:**\n"
            "- Preserva la metadata y docstring de la función mediante `@functools.wraps`.\n"
            "- Multiplica el tiempo de espera por `backoff_factor` tras cada fallo para descongestionar el servicio de destino."
        )
    },
    {
        "prompt": "¿Cómo crear una cola de prioridad basada en montículos (heap) en Python?",
        "thought": "Explicar heapq en Python. Implementar una clase PriorityQueue encapsulando heappush y heappop con manejo de tuplas de prioridad y elementos.",
        "response": (
            "En Python, las colas de prioridad se implementan eficientemente con el módulo estándar `heapq` ($O(\\log n)$ por inserción y extracción):\n\n"
            "```python\n"
            "import heapq\n"
            "from typing import Generic, TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            "class PriorityQueue(Generic[T]):\n"
            "    def __init__(self) -> None:\n"
            "        self._elements: list[tuple[int, int, T]] = []\n"
            "        self._index: int = 0\n\n"
            "    def push(self, item: T, priority: int) -> None:\n"
            "        # Guardamos (-priority para max-heap o priority para min-heap)\n"
            "        heapq.heappush(self._elements, (priority, self._index, item))\n"
            "        self._index += 1\n\n"
            "    def pop(self) -> T:\n"
            "        if self.is_empty():\n"
            "            raise IndexError(\"La cola de prioridad está vacía.\")\n"
            "        return heapq.heappop(self._elements)[2]\n\n"
            "    def is_empty(self) -> bool:\n"
            "        return len(self._elements) == 0\n"
            "```"
        )
    },
    {
        "prompt": "Explica la diferencia entre un generador y una lista por comprensión en Python.",
        "thought": "Comparar list comprehension (evaluación ansiosa en memoria, corchetes) con generator expressions (evaluación perezosa con yield, paréntesis, bajo consumo RAM).",
        "response": (
            "La diferencia principal radica en la gestión de memoria y el momento de evaluación:\n\n"
            "1. **Lista por comprensión (`[x for x in data]`)**:\n"
            "   - **Ansiosa (Eager)**: Evalúa todos los elementos y los almacena completos en RAM de inmediato.\n"
            "   - Ideal para secuencias cortas donde necesitas indexar o consultar la longitud repetidamente.\n\n"
            "2. **Generador (`(x for x in data)` o con `yield`)**:\n"
            "   - **Perezosa (Lazy)**: Genera un elemento solo cuando se le solicita mediante `next()` o un bucle `for`.\n"
            "   - Su huella de memoria es constante $O(1)$, haciéndolo idóneo para procesar flujos masivos de datos o gigabytes de texto en silicio sin agotar la memoria RAM."
        )
    },

    # -------------------------------------------------------------
    # 2. JAVASCRIPT / TYPESCRIPT
    # -------------------------------------------------------------
    {
        "prompt": "Escribe una función de debounce en TypeScript con tipado estricto.",
        "thought": "Modelar función debounce en TS. Controlar temporizador NodeJS.Timeout / number, preservar argumentos y contexto con generics.",
        "response": (
            "Aquí tienes una función `debounce` genérica y estrictamente tipada en TypeScript:\n\n"
            "```typescript\n"
            "export function debounce<T extends (...args: any[]) => void>(\n"
            "  func: T,\n"
            "  waitMs: number\n"
            "): (...args: Parameters<T>) => void {\n"
            "  let timeoutId: ReturnType<typeof setTimeout> | null = null;\n\n"
            "  return function (...args: Parameters<T>): void {\n"
            "    if (timeoutId !== null) {\n"
            "      clearTimeout(timeoutId);\n"
            "    }\n"
            "    timeoutId = setTimeout(() => {\n"
            "      func(...args);\n"
            "      timeoutId = null;\n"
            "    }, waitMs);\n"
            "  };\n"
            "}\n"
            "```\n\n"
            "**Ventajas:**\n"
            "- Utiliza `Parameters<T>` para inferir con precisión los tipos exactos de los argumentos.\n"
            "- Previene la saturación de llamadas en eventos rápidos como scroll o tipeo en inputs."
        )
    },
    {
        "prompt": "¿Cómo modelar estados de una API usando uniones discriminadas en TypeScript?",
        "thought": "Explicar discriminated union con propiedad de discriminante 'status' (idle, loading, success, error) y pattern matching con switch exhaustivo.",
        "response": (
            "Una **unión discriminada** permite representar estados mutuamente excluyentes con máxima seguridad en tiempo de compilación:\n\n"
            "```typescript\n"
            "type ApiResponse<T> =\n"
            "  | { status: 'idle' }\n"
            "  | { status: 'loading' }\n"
            "  | { status: 'success'; data: T }\n"
            "  | { status: 'error'; message: string; code: number };\n\n"
            "function renderResponse<T>(res: ApiResponse<T>): string {\n"
            "  switch (res.status) {\n"
            "    case 'idle':\n"
            "      return 'Esperando inicio...';\n"
            "    case 'loading':\n"
            "      return 'Cargando recursos...';\n"
            "    case 'success':\n"
            "      return `Datos recibidos: ${JSON.stringify(res.data)}`;\n"
            "    case 'error':\n"
            "      return `Error (${res.code}): ${res.message}`;\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "TypeScript garantiza que `res.data` solo exista cuando `status === 'success'`, erradicando errores en tiempo de ejecución."
        )
    },
    {
        "prompt": "Escribe un ejemplo de Promise.allSettled en JavaScript ES6 y explica cuándo usarlo.",
        "thought": "Presentar Promise.allSettled vs Promise.all. Demostrar cómo allSettled nunca rechaza de golpe sino que devuelve status 'fulfilled' o 'rejected'.",
        "response": (
            "A diferencia de `Promise.all` (que falla inmediatamente si una sola promesa es rechazada), `Promise.allSettled` espera a que todas concluyan independientemente del resultado:\n\n"
            "```javascript\n"
            "async function fetchAllServices() {\n"
            "  const urls = ['/api/user', '/api/notifications', '/api/metrics'];\n"
            "  const promises = urls.map(url => fetch(url).then(r => r.json()));\n\n"
            "  const results = await Promise.allSettled(promises);\n\n"
            "  results.forEach((result, idx) => {\n"
            "    if (result.status === 'fulfilled') {\n"
            "      console.log(`Endpoint ${urls[idx]} éxito:`, result.value);\n"
            "    } else {\n"
            "      console.error(`Endpoint ${urls[idx]} falló:`, result.reason);\n"
            "    }\n"
            "  });\n"
            "}\n"
            "```\n\n"
            "**Uso idóneo:** Cuando las tareas son independientes (como cargar paneles de un dashboard) y el fallo de una no debe cancelar las demás."
        )
    },

    # -------------------------------------------------------------
    # 3. C & C++
    # -------------------------------------------------------------
    {
        "prompt": "Escribe un programa en C con asignación dinámica segura de memoria y prevención de fugas.",
        "thought": "Demostrar malloc, comprobación de puntero NULL, liberación con free y seteo a NULL en lenguaje C.",
        "response": (
            "En C moderno, la gestión de memoria requiere verificar siempre el retorno de `malloc` y liberar explícitamente los recursos:\n\n"
            "```c\n"
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n\n"
            "int main(void) {\n"
            "    size_t count = 5;\n"
            "    int *buffer = (int *)malloc(count * sizeof(int));\n\n"
            "    if (buffer == NULL) {\n"
            "        fprintf(stderr, \"Error: Fallo crítico de asignación de memoria en silicio.\\n\");\n"
            "        return EXIT_FAILURE;\n"
            "    }\n\n"
            "    for (size_t i = 0; i < count; ++i) {\n"
            "        buffer[i] = (int)(i * 10);\n"
            "        printf(\"buffer[%zu] = %d\\n\", i, buffer[i]);\n"
            "    }\n\n"
            "    // Liberación estricta para evitar memory leaks\n"
            "    free(buffer);\n"
            "    buffer = NULL; // Prevenir punteros colgantes (dangling pointer)\n\n"
            "    return EXIT_SUCCESS;\n"
            "}\n"
            "```"
        )
    },
    {
        "prompt": "¿Qué es RAII en C++ y cómo se utiliza con std::unique_ptr?",
        "thought": "Explicar Resource Acquisition Is Initialization (RAII). Demostrar std::unique_ptr y std::make_unique para gestión automática de memoria en C++.",
        "response": (
            "**RAII (Resource Acquisition Is Initialization)** es el patrón fundamental de C++ donde la vida útil de un recurso está ligada a la vida útil del objeto que lo contiene:\n\n"
            "```cpp\n"
            "#include <iostream>\n"
            "#include <memory>\n"
            "#include <string>\n\n"
            "class DeviceConnection {\n"
            "public:\n"
            "    DeviceConnection(const std::string& name) : name_(name) {\n"
            "        std::cout << \"Conexión abierta: \" << name_ << std::endl;\n"
            "    }\n"
            "    ~DeviceConnection() {\n"
            "        std::cout << \"Conexión cerrada automáticamente: \" << name_ << std::endl;\n"
            "    }\n"
            "    void transmit(const std::string& data) {\n"
            "        std::cout << \"Transfiriendo: \" << data << std::endl;\n"
            "    }\n"
            "private:\n"
            "    std::string name_;\n"
            "};\n\n"
            "void process() {\n"
            "    // std::make_unique libera la memoria al salir de ámbito, incluso ante excepciones\n"
            "    auto conn = std::make_unique<DeviceConnection>(\"Puerto_COM1\");\n"
            "    conn->transmit(\"Datos de sensor\");\n"
            "} // Aquí el destructor se ejecuta automáticamente de forma determinista.\n"
            "```"
        )
    },

    # -------------------------------------------------------------
    # 4. RUST
    # -------------------------------------------------------------
    {
        "prompt": "Explica el sistema de Ownership y Borrowing en Rust con un ejemplo claro.",
        "thought": "Explicar reglas de ownership: cada valor tiene un único dueño; borrowing con referencias inmutables (&T) y mutable única (&mut T).",
        "response": (
            "El sistema de **Ownership** de Rust garantiza seguridad de memoria en tiempo de compilación sin recolector de basura (Garbage Collector):\n\n"
            "1. Cada valor en Rust tiene una variable dueña (*owner*).\n"
            "2. Solo puede haber un dueño a la vez.\n"
            "3. Cuando el dueño sale del ámbito (*scope*), el valor se destruye automáticamente (`drop`).\n\n"
            "```rust\n"
            "fn main() {\n"
            "    let mut mensaje = String::from(\"Silicio Netelpro\");\n\n"
            "    // Préstamo inmutable (&) - pueden existir múltiples lectores simultáneos\n"
            "    let len = calcular_longitud(&mensaje);\n"
            "    println!(\"Longitud de '{}': {}\", mensaje, len);\n\n"
            "    // Préstamo mutable (&mut) - solo uno a la vez para evitar condiciones de carrera (data races)\n"
            "    modificar_mensaje(&mut mensaje);\n"
            "    println!(\"Mensaje actualizado: {}\", mensaje);\n"
            "}\n\n"
            "fn calcular_longitud(s: &String) -> usize {\n"
            "    s.len() // Lee sin tomar posesión\n"
            "}\n\n"
            "fn modificar_mensaje(s: &mut String) {\n"
            "    s.push_str(\" [Verificado]\");\n"
            "}\n"
            "```"
        )
    },
    {
        "prompt": "Escribe una función en Rust que procese un Result<T, E> y use el operador interrogación `?`.",
        "thought": "Demostrar propagación de errores con '?', Result<T, std::io::Error> y lectura de archivo en Rust.",
        "response": (
            "En Rust, el operador `?` desempaqueta el valor si es `Ok(v)` o retorna anticipadamente el error si es `Err(e)`:\n\n"
            "```rust\n"
            "use std::fs::File;\n"
            "use std::io::{self, Read};\n\n"
            "pub fn leer_archivo_config(ruta: &str) -> Result<String, io::Error> {\n"
            "    let mut archivo = File::open(ruta)?;\n"
            "    let mut contenido = String::new();\n"
            "    archivo.read_to_string(&mut contenido)?;\n"
            "    Ok(contenido)\n"
            "}\n\n"
            "fn main() {\n"
            "    match leer_archivo_config(\"config.toml\") {\n"
            "        Ok(data) => println!(\"Configuración leída con éxito: {}\\n\", data),\n"
            "        Err(e) => eprintln!(\"Error al leer configuración: {}\\n\", e),\n"
            "    }\n"
            "}\n"
            "```"
        )
    },

    # -------------------------------------------------------------
    # 5. GO (GOLANG)
    # -------------------------------------------------------------
    {
        "prompt": "¿Cómo crear un patrón de Worker Pool concurrente con goroutines y canales en Go?",
        "thought": "Implementar worker pool canónico en Go: canales jobs y results, sync.WaitGroup, goroutines para procesar tareas concurrentemente.",
        "response": (
            "Aquí tienes el patrón canónico de **Worker Pool** en Go para procesar tareas concurrentes de forma controlada:\n\n"
            "```go\n"
            "package main\n\n"
            "import (\n"
            "    \"fmt\"\n"
            "    \"sync\"\n"
            "    \"time\"\n"
            ")\n\n"
            "func worker(id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {\n"
            "    defer wg.Done()\n"
            "    for j := range jobs {\n"
            "        fmt.Printf(\"Worker %d procesando tarea %d\\n\", id, j)\n"
            "        time.Sleep(50 * time.Millisecond)\n"
            "        results <- j * 2\n"
            "    }\n"
            "}\n\n"
            "func main() {\n"
            "    const numJobs = 10\n"
            "    const numWorkers = 3\n\n"
            "    jobs := make(chan int, numJobs)\n"
            "    results := make(chan int, numJobs)\n"
            "    var wg sync.WaitGroup\n\n"
            "    // Lanzar trabajadores\n"
            "    for w := 1; w <= numWorkers; w++ {\n"
            "        wg.Add(1)\n"
            "        go worker(w, jobs, results, &wg)\n"
            "    }\n\n"
            "    // Enviar tareas y cerrar canal de entrada\n"
            "    for j := 1; j <= numJobs; j++ {\n"
            "        jobs <- j\n"
            "    }\n"
            "    close(jobs)\n\n"
            "    // Esperar en rutina separada para cerrar resultados\n"
            "    go func() {\n"
            "        wg.Wait()\n"
            "        close(results)\n"
            "    }()\n\n"
            "    // Recolectar resultados\n"
            "    for res := range results {\n"
            "        fmt.Printf(\"Resultado obtenido: %d\\n\", res)\n"
            "    }\n"
            "}\n"
            "```"
        )
    },
    {
        "prompt": "Explica la convención de manejo de errores en Go con `if err != nil`.",
        "thought": "Explicar por qué Go rechaza las excepciones mágicas a favor de valores de retorno explícitos (result, error) y cómo verificar y propagar.",
        "response": (
            "En Go, los errores son valores ordinarios (`error` interface), no excepciones ocultas. Esto hace que el flujo de ejecución sea 100% predecible y explícito:\n\n"
            "```go\n"
            "package main\n\n"
            "import (\n"
            "    \"errors\"\n"
            "    \"fmt\"\n"
            ")\n\n"
            "func dividir(a, b float64) (float64, error) {\n"
            "    if b == 0 {\n"
            "        return 0, errors.New(\"división por cero no permitida en cálculo\")\n"
            "    }\n"
            "    return a / b, nil\n"
            "}\n\n"
            "func main() {\n"
            "    resultado, err := dividir(10, 0)\n"
            "    if err != nil {\n"
            "        fmt.Printf(\"Control formal de error: %v\\n\", err)\n"
            "        return\n"
            "    }\n"
            "    fmt.Println(\"Resultado:\", resultado)\n"
            "}\n"
            "```\n\n"
            "**Principio:** Cada fallo potencial debe ser atendido en el punto exacto donde ocurre, impidiendo estados corruptos en producción."
        )
    },

    # -------------------------------------------------------------
    # 6. SQL
    # -------------------------------------------------------------
    {
        "prompt": "Escribe una consulta SQL con funciones de ventana (Window Functions) para obtener los 3 productos más vendidos por categoría.",
        "thought": "Diseñar CTE con ROW_NUMBER() OVER(PARTITION BY categoria ORDER BY ventas DESC) y filtrar row_num <= 3.",
        "response": (
            "Las **funciones de ventana** permiten realizar cálculos sobre particiones de filas sin colapsar el conjunto de resultados:\n\n"
            "```sql\n"
            "WITH RankingProductos AS (\n"
            "    SELECT \n"
            "        categoria_id,\n"
            "        producto_id,\n"
            "        nombre,\n"
            "        total_ventas,\n"
            "        ROW_NUMBER() OVER (\n"
            "            PARTITION BY categoria_id \n"
            "            ORDER BY total_ventas DESC\n"
            "        ) AS ranking\n"
            "    FROM ventas_consolidadas\n"
            ")\n"
            "SELECT \n"
            "    categoria_id,\n"
            "    producto_id,\n"
            "    nombre,\n"
            "    total_ventas\n"
            "FROM RankingProductos\n"
            "WHERE ranking <= 3\n"
            "ORDER BY categoria_id, ranking ASC;\n"
            "```\n\n"
            "- `PARTITION BY categoria_id`: Reinicia el contador de ranking para cada categoría distinta.\n"
            "- `ORDER BY total_ventas DESC`: Asigna el primer lugar al producto de mayor recaudación."
        )
    },
    {
        "prompt": "¿Cuál es la diferencia entre INNER JOIN, LEFT JOIN y FULL OUTER JOIN en SQL?",
        "thought": "Explicar diferencias de álgebra relacional entre INNER (intersección), LEFT (todas de la izquierda + coincidencias) y FULL (unión de ambas con NULLs).",
        "response": (
            "La diferencia reside en cómo se manejan las filas sin coincidencia entre dos tablas:\n\n"
            "1. **`INNER JOIN`**:\n"
            "   - Devuelve únicamente las filas que tienen coincidencia en **ambas** tablas (intersección pura).\n"
            "2. **`LEFT JOIN` (o `LEFT OUTER JOIN`)**:\n"
            "   - Devuelve **todas** las filas de la tabla izquierda, junto con los datos coincidentes de la derecha.\n"
            "   - Si no hay correspondencia en la derecha, rellena sus columnas con `NULL`.\n"
            "3. **`FULL OUTER JOIN`**:\n"
            "   - Devuelve todas las filas cuando hay una coincidencia en cualquiera de las dos tablas.\n"
            "   - Si una fila de la izquierda no tiene par en la derecha, o viceversa, los campos faltantes se completan con `NULL`."
        )
    },

    # -------------------------------------------------------------
    # 7. BASH & POWERSHELL
    # -------------------------------------------------------------
    {
        "prompt": "Escribe un script seguro de Bash con control estricto de errores para procesar archivos de registro (logs).",
        "thought": "Crear script con set -euo pipefail, trap para cleanup, bucle seguro sobre archivos y uso de tuberías.",
        "response": (
            "Para crear scripts robustos y a prueba de fallos en entornos UNIX/Linux, siempre se debe usar `set -euo pipefail`:\n\n"
            "```bash\n"
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "IFS=$'\\n\\t'\n\n"
            "LOG_DIR=\"${1:-/var/log/app}\"\n"
            "OUTPUT_FILE=\"error_summary.txt\"\n\n"
            "if [[ ! -d \"${LOG_DIR}\" ]]; then\n"
            "    echo \"Error: El directorio '${LOG_DIR}' no existe.\" >&2\n"
            "    exit 1\n"
            "fi\n\n"
            "echo \"Analizando registros en ${LOG_DIR}...\"\n"
            "grep -r \"CRITICAL\" \"${LOG_DIR}\" | awk '{print $1, $2, $NF}' | sort | uniq -c > \"${OUTPUT_FILE}\"\n\n"
            "echo \"Análisis completado exitosamente en ${OUTPUT_FILE}.\"\n"
            "```\n\n"
            "- `-e`: Detiene el script inmediatamente si cualquier comando falla.\n"
            "- `-u`: Falla si se intenta usar una variable no declarada.\n"
            "- `-o pipefail`: Detecta errores en cualquier eslabón intermedio de una tubería (`|`)."
        )
    },
    {
        "prompt": "¿Cómo buscar y eliminar archivos temporales antiguos con PowerShell de forma segura?",
        "thought": "Escribir comando seguro en PowerShell con Get-ChildItem, Where-Object con criterio de fecha AddDays(-7), y Remove-Item con confirmación condicional.",
        "response": (
            "En Windows PowerShell, puedes filtrar y depurar archivos temporales antiguos con precisión:\n\n"
            "```powershell\n"
            "# Definir umbral de 7 días atrás\n"
            "$umbral = (Get-Date).AddDays(-7)\n"
            "$ruta = \"C:\\Temp\\Logs\"\n\n"
            "# Buscar archivos modificados antes del umbral\n"
            "Get-ChildItem -Path $ruta -File -Recurse |\n"
            "    Where-Object { $_.LastWriteTime -lt $umbral } |\n"
            "    ForEach-Object {\n"
            "        Write-Host \"Eliminando archivo obsoleto: $($_.FullName)\" -ForegroundColor Yellow\n"
            "        Remove-Item -Path $_.FullName -Force\n"
            "    }\n"
            "```\n\n"
            "**Seguridad:** Puedes agregar `-WhatIf` a `Remove-Item` para simular la eliminación antes de ejecutarla en producción."
        )
    },

    # -------------------------------------------------------------
    # 8. HTML5 & CSS3
    # -------------------------------------------------------------
    {
        "prompt": "Escribe una tarjeta de perfil moderna y responsiva con HTML5 semántico y CSS Grid/Flexbox.",
        "thought": "Crear estructura semántica (article, header, figure, main), estilos CSS limpios con CSS variables, Flexbox centrado y media query responsiva.",
        "response": (
            "Aquí tienes una tarjeta de usuario moderna, semántica y responsiva:\n\n"
            "```html\n"
            "<article class=\"profile-card\">\n"
            "  <div class=\"avatar-wrapper\">\n"
            "    <img src=\"avatar.webp\" alt=\"Fotografía de perfil\" class=\"avatar\" />\n"
            "  </div>\n"
            "  <div class=\"content\">\n"
            "    <h2 class=\"name\">Jona</h2>\n"
            "    <p class=\"role\">Arquitecto de Silicio & IA</p>\n"
            "    <div class=\"badge-container\">\n"
            "      <span class=\"badge\">Python</span>\n"
            "      <span class=\"badge\">Rust</span>\n"
            "      <span class=\"badge\">C++</span>\n"
            "    </div>\n"
            "  </div>\n"
            "</article>\n"
            "```\n\n"
            "```css\n"
            ".profile-card {\n"
            "  display: flex;\n"
            "  flex-direction: column;\n"
            "  align-items: center;\n"
            "  background: #1e1e2e;\n"
            "  color: #cdd6f4;\n"
            "  border-radius: 16px;\n"
            "  padding: 24px;\n"
            "  max-width: 320px;\n"
            "  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);\n"
            "}\n\n"
            ".avatar {\n"
            "  width: 96px;\n"
            "  height: 96px;\n"
            "  border-radius: 50%;\n"
            "  border: 3px solid #89b4fa;\n"
            "  object-fit: cover;\n"
            "}\n\n"
            ".badge-container {\n"
            "  display: flex;\n"
            "  gap: 8px;\n"
            "  margin-top: 12px;\n"
            "}\n\n"
            ".badge {\n"
            "  background: #313244;\n"
            "  padding: 4px 10px;\n"
            "  border-radius: 999px;\n"
            "  font-size: 0.85rem;\n"
            "}\n"
            "```"
        )
    }
]

def generate_multilang_jsonl(output_path: Path) -> int:
    """Writes the multilang coder samples into JSONL format."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in MULTILANG_CODER_SAMPLES:
            formatted_prompt = (
                f"<|user|>\n{item['prompt']}\n"
                f"<|assistant|>\n"
                f"<|thought|>\n{item['thought']}\n<|endthought|>\n"
                f"{item['response']}<|eos|>"
            )
            entry = {"prompt": formatted_prompt}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count

if __name__ == "__main__":
    out = Path(__file__).parent / "multilang_coder.jsonl"
    n = generate_multilang_jsonl(out)
    print(f"Generated {n} multilang coder training samples to {out}")
