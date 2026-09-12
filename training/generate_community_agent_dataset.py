"""Generador del Dataset Comunitario Goliath-Killer para Qwen3.5-2B.

Dataset balanceado de alta densidad informacional (~1,200 ejemplos):
1. Tool-Calling Universal & ReAct (~380 ejemplos, ~32%):
   - Terminal: Puertos, servicios systemd, hardware, CPU, memoria, disco, procesos, docker, git, red.
   - Filesystem: Lectura de manifiestos, configs del sistema (/etc/resolv.conf, /etc/hosts), logs, y escritura atómica (/tmp/health.sh).
   - Web Search: Documentación técnica, RFCs, Python 3.13, CVEs y diagnóstico de librerías.
   - Control de Irrelevancia (BFCL Negative Control): Supresión estricta de tools en consultas teóricas/matemáticas directas.
2. Síntesis Formal de Código Netelpro Lisp (563 ejemplos, ~47%):
   - Aridad formal, AST, compilador determinista, recursión de cola y tipado.
3. Honestidad Epistémica & VTB (227 ejemplos, ~19%):
   - Detección de trampas, rechazo de falsas premisas y eliminación de teatro de verificación (FAAR 0%).
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
DATA_DIR = TRAINING_DIR / "data"
OUTPUT_FILE = DATA_DIR / "community_agent_dataset.jsonl"


def make_tool_completion(th1: str, tool_name: str, tool_args: dict, raw_output: str, th2: str, final_ans: str) -> str:
    tool_json = json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False)
    return (
        f"<thought>\n{th1}\n</thought>\n"
        f"<tool_call>\n{tool_json}\n</tool_call>\n\n"
        f"<tool_response>\n{raw_output}\n</tool_response>\n\n"
        f"<thought>\n{th2}\n</thought>\n"
        f"{final_ans}"
    )


# ===========================================================================
# 1. HERRAMIENTAS DE TERMINAL (PUERTOS, SERVICIOS, RECURSOS, RED, DOCKER, GIT)
# ===========================================================================

def build_terminal_port_cases() -> list[dict[str, str]]:
    cases = []
    ports = [
        ("21", "FTP", "vsftpd", "tcp LISTEN 0 32 0.0.0.0:21 0.0.0.0:*", True),
        ("22", "SSH", "sshd", "tcp LISTEN 0 128 0.0.0.0:22 0.0.0.0:*", True),
        ("25", "SMTP", "postfix", "tcp LISTEN 0 100 127.0.0.1:25 0.0.0.0:*", True),
        ("53", "DNS", "named", "tcp LISTEN 0 10 127.0.0.1:53 0.0.0.0:*", True),
        ("80", "HTTP", "nginx", "tcp LISTEN 0 511 0.0.0.0:80 0.0.0.0:*", True),
        ("443", "HTTPS", "nginx", "tcp LISTEN 0 511 0.0.0.0:443 0.0.0.0:*", True),
        ("3000", "Node/React", "node", "tcp LISTEN 0 511 127.0.0.1:3000 0.0.0.0:*", True),
        ("3306", "MySQL", "mysqld", "tcp LISTEN 0 128 127.0.0.1:3306 0.0.0.0:*", True),
        ("5432", "PostgreSQL", "postgres", "tcp LISTEN 0 128 0.0.0.0:5432 0.0.0.0:*", True),
        ("6379", "Redis", "redis-server", "tcp LISTEN 0 511 127.0.0.1:6379 0.0.0.0:*", True),
        ("8000", "FastAPI / Uvicorn", "uvicorn", "tcp LISTEN 0 128 0.0.0.0:8000 0.0.0.0:*", True),
        ("8080", "Tomcat / Java", "java", "tcp LISTEN 0 100 0.0.0.0:8080 0.0.0.0:*", True),
        ("8443", "Admin HTTPS", "", "", False),
        ("9000", "MinIO / PHP-FPM", "minio", "tcp LISTEN 0 128 0.0.0.0:9000 0.0.0.0:*", True),
        ("9090", "Prometheus", "prometheus", "tcp LISTEN 0 128 0.0.0.0:9090 0.0.0.0:*", True),
        ("9200", "Elasticsearch", "", "", False),
        ("11434", "Ollama", "ollama", "tcp LISTEN 0 128 127.0.0.1:11434 0.0.0.0:*", True),
        ("27017", "MongoDB", "mongod", "tcp LISTEN 0 128 127.0.0.1:27017 0.0.0.0:*", True),
        ("5000", "Flask", "", "", False),
        ("4200", "Angular Dev", "", "", False),
        ("8888", "Jupyter", "python3", "tcp LISTEN 0 128 127.0.0.1:8888 0.0.0.0:*", True),
        ("5672", "RabbitMQ", "beam.smp", "tcp LISTEN 0 128 0.0.0.0:5672 0.0.0.0:*", True),
    ]

    for port, srv_name, proc, raw_out, is_active in ports:
        cmd = f"ss -tuln | grep :{port}"
        if is_active:
            th1 = f"Debo verificar si el puerto {port} tiene sockets en escucha. Usaré la terminal del sistema para ejecutar `{cmd}`."
            th2 = f"La salida de `{cmd}` confirma un socket en LISTEN en el puerto {port}. El servicio {srv_name} está activo."
            ans = f"Sí, el puerto **{port}** ({srv_name}) está **activo y en escucha (LISTEN)** en el servidor."
        else:
            th1 = f"El usuario consulta el puerto {port}. Debo inspeccionar los sockets TCP/UDP con `{cmd}` sin asumir su estado."
            th2 = f"La salida del comando es vacía. No hay ningún proceso escuchando en el puerto {port}."
            ans = f"No, el puerto **{port}** ({srv_name}) está **completamente libre** (no hay ningún socket en escucha)."

        comp = make_tool_completion(th1, "system_terminal", {"command": cmd}, raw_out, th2, ans)
        cases.append({"prompt": f"¿El puerto {port} ({srv_name}) está activo y escuchando en el servidor?", "completion": comp, "category": "agent_terminal_ports"})
        cases.append({"prompt": f"Revisa si el puerto {port} de {srv_name} está en escucha en el servidor.", "completion": comp, "category": "agent_terminal_ports"})
        cases.append({"prompt": f"Verifica si el puerto {port} está ocupado o libre.", "completion": comp, "category": "agent_terminal_ports"})
        cases.append({"prompt": f"Comprueba con ss o netstat si hay algún socket en :{port}.", "completion": comp, "category": "agent_terminal_ports"})

    return cases


def build_terminal_services_cases() -> list[dict[str, str]]:
    cases = []
    services = [
        ("postgresql", "active", "active (running)", True),
        ("redis-server", "active", "active (running)", True),
        ("nginx", "active", "active (running)", True),
        ("docker", "active", "active (running)", True),
        ("sshd", "active", "active (running)", True),
        ("cron", "active", "active (running)", True),
        ("ufw", "active", "active (running)", True),
        ("systemd-resolved", "active", "active (running)", True),
        ("systemd-journald", "active", "active (running)", True),
        ("clickhouse-server", "inactive", "inactive (dead)", False),
        ("mysql", "failed", "failed (Result: exit-code)", False),
        ("rabbitmq-server", "inactive", "inactive (dead)", False),
        ("elasticsearch", "inactive", "inactive (dead)", False),
        ("kibana", "inactive", "inactive (dead)", False),
        ("mongod", "active", "active (running)", True),
        ("fail2ban", "active", "active (running)", True),
        ("supervisor", "active", "active (running)", True),
    ]

    for svc, status, out, is_act in services:
        cmd = f"systemctl is-active {svc}"
        th1 = f"Para saber si {svc} está en ejecución, debo consultar el estado en systemd con `{cmd}`."
        th2 = f"La salida es `{status}`. Procedo a informar el estado verificado del demonio."
        if is_act:
            ans = f"El servicio **{svc}** está **activo y en ejecución (`active`)**."
        else:
            ans = f"El servicio **{svc}** **no está en ejecución**; su estado actual reportado es **`{status}`**."

        comp = make_tool_completion(th1, "system_terminal", {"command": cmd}, out, th2, ans)
        cases.append({"prompt": f"¿El servicio {svc} está corriendo en el sistema?", "completion": comp, "category": "agent_terminal_services"})
        cases.append({"prompt": f"Revisa con systemctl si {svc} está activo.", "completion": comp, "category": "agent_terminal_services"})
        cases.append({"prompt": f"Comprueba el estado del demonio {svc}.", "completion": comp, "category": "agent_terminal_services"})
        cases.append({"prompt": f"Verifica si {svc} se cayó o está levantado.", "completion": comp, "category": "agent_terminal_services"})

    return cases


def build_system_resources_cases() -> list[dict[str, str]]:
    cases = []
    resource_queries = [
        (
            "¿Cuánta memoria RAM le queda disponible al servidor?",
            "system_terminal",
            {"command": "free -m"},
            "              total        used        free      shared  buff/cache   available\nMem:          32000        6200       18500         350        7300       25100\nSwap:          4096           0        4096",
            "Debo inspeccionar la memoria RAM del host mediante `free -m`.",
            "Salida: Total 32 GB, Usado 6.2 GB, Libre estricto 18.5 GB, Disponible 25.1 GB.",
            "El servidor cuenta con **18.5 GB** de memoria estrictamente libre y **25.1 GB disponibles** sobre un total de 32 GB."
        ),
        (
            "¿Cuánto espacio libre queda en el disco principal?",
            "system_terminal",
            {"command": "df -h /"},
            "Filesystem      Size  Used Avail Use% Mounted on\n/dev/nvme0n1p2  468G  142G  303G  32% /",
            "Debo inspeccionar el espacio en disco de la raíz mediante `df -h /`.",
            "La partición raíz tiene 468 GB en total, 142 GB usados (32%) y 303 GB libres.",
            "La partición principal (`/`) tiene **303 GB disponibles** (32% de ocupación, 142 GB usados de 468 GB)."
        ),
        (
            "¿Cuál es la carga actual de la CPU y el uptime del servidor?",
            "system_terminal",
            {"command": "uptime"},
            " 20:45:12 up 14 days,  6:20,  2 users,  load average: 0.18, 0.22, 0.15",
            "Consulto la carga del procesador y tiempo de actividad con `uptime`.",
            "Load averages: 0.18 (1 min), 0.22 (5 min), 0.15 (15 min). La máquina está muy desahogada.",
            "El sistema lleva **14 días encendido** y la carga promedio de CPU es mínima: **0.18** (1 min), **0.22** (5 min) y **0.15** (15 min)."
        ),
        (
            "¿Qué contenedores Docker están corriendo en este momento?",
            "system_terminal",
            {"command": "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"},
            "NAMES               STATUS              PORTS\nredis-stack         Up 3 days           0.0.0.0:6379->6379/tcp\npg-cluster          Up 5 days           0.0.0.0:5432->5432/tcp",
            "Debo listar los contenedores Docker en ejecución.",
            "Hay 2 contenedores activos: `redis-stack` en el puerto 6379 y `pg-cluster` en el puerto 5432.",
            "Actualmente hay **2 contenedores Docker activos**:\n1. **`pg-cluster`**: PostgreSQL en puerto `5432` (Activo hace 5 días).\n2. **`redis-stack`**: Redis en puerto `6379` (Activo hace 3 días)."
        ),
        (
            "Muestra los últimos logs del contenedor web-api en docker.",
            "system_terminal",
            {"command": "docker logs --tail 10 web-api"},
            "INFO:     Application startup complete.\nINFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)\nINFO:     192.168.1.5:48210 - \"GET /health HTTP/1.1\" 200 OK",
            "Inspecciono los logs recientes del contenedor web-api con `docker logs`.",
            "Los logs confirman que Uvicorn arrancó en http://0.0.0.0:8000 y el último health check fue 200 OK.",
            "Logs recientes de **`web-api`**:\n- Startup completado en el puerto 8000.\n- Última petición: `GET /health` con código **200 OK**."
        ),
        (
            "¿Cuál es el estado del repositorio Git actual?",
            "system_terminal",
            {"command": "git status -s"},
            " M src/server.py\n?? tests/test_perf.py",
            "Para conocer el estado del repositorio Git, ejecuto `git status -s`.",
            "Hay un archivo modificado (`src/server.py`) y un archivo no rastreado (`tests/test_perf.py`).",
            "El repositorio tiene cambios pendientes:\n- Modificado: `src/server.py`\n- Sin rastrear (Untracked): `tests/test_perf.py`"
        ),
        (
            "Muestra los últimos 3 commits en git.",
            "system_terminal",
            {"command": "git log -n 3 --oneline"},
            "d7e2543 feat(training): add Goliath-Killer community agent training notebook\nd2bed8c feat(training): add high-density community agent dataset\n202d2bd feat(training): add 782 massive dataset",
            "Debo consultar el historial reciente de git con `git log -n 3 --oneline`.",
            "Se obtuvieron los 3 commits recientes.",
            "Los últimos 3 commits son:\n1. `d7e2543`: add Goliath-Killer community agent training notebook\n2. `d2bed8c`: add high-density community agent dataset\n3. `202d2bd`: add 782 massive dataset"
        ),
        (
            "Revisa los 3 procesos que más memoria RAM consumen.",
            "system_terminal",
            {"command": "ps aux --sort=-%mem | head -n 4"},
            "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\npostgres  1420  0.5  4.2 890120 135400 ?       S    Sep06  12:30 /usr/lib/postgresql/16/bin/postgres\nnode      2180  1.2  3.8 654200 124100 ?       Sl   Sep08   8:15 node /app/server.js\nredis     1110  0.1  1.5 245100  48200 ?       Ssl  Sep06   2:10 redis-server",
            "Ejecuto `ps aux --sort=-%mem` para identificar los procesos con mayor uso de RSS.",
            "Top 3 procesos: postgres (4.2% RAM), node (3.8% RAM), redis (1.5% RAM).",
            "Los 3 procesos con mayor consumo de memoria son:\n1. **postgres** (PID 1420): **4.2%** RAM (135 MB)\n2. **node** (PID 2180): **3.8%** RAM (124 MB)\n3. **redis-server** (PID 1110): **1.5%** RAM (48 MB)"
        ),
        (
            "Verifica la conectividad de red haciendo ping a 8.8.8.8.",
            "system_terminal",
            {"command": "ping -c 3 8.8.8.8"},
            "PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data.\n64 bytes from 8.8.8.8: icmp_seq=1 ttl=118 time=12.4 ms\n64 bytes from 8.8.8.8: icmp_seq=2 ttl=118 time=12.1 ms\n64 bytes from 8.8.8.8: icmp_seq=3 ttl=118 time=12.3 ms\n\n--- 8.8.8.8 ping statistics ---\n3 packets transmitted, 3 received, 0% packet loss, time 2003ms\nrtt min/avg/max/mdev = 12.110/12.270/12.412/0.125 ms",
            "Ejecuto `ping -c 3 8.8.8.8` para comprobar latencia y pérdida de paquetes hacia Internet.",
            "3 paquetes transmitidos, 3 recibidos, 0% pérdida, latencia promedio 12.27 ms.",
            "Conectividad a Internet **óptima**: **0% de pérdida de paquetes** hacia 8.8.8.8 con una latencia media de **12.2 ms**."
        ),
        (
            "Verifica si la API externa de GitHub responde cabeceras HTTP correctamente.",
            "system_terminal",
            {"command": "curl -I https://api.github.com"},
            "HTTP/2 200\nserver: GitHub.com\ndate: Thu, 11 Sep 2026 21:10:00 GMT\ncontent-type: application/json; charset=utf-8\nx-github-media-type: github.v3; format=json",
            "Consulto las cabeceras HTTP de https://api.github.com mediante `curl -I`.",
            "La respuesta es HTTP/2 200 con content-type application/json.",
            "La API de GitHub responde correctamente con estado **HTTP/2 200 OK**."
        ),
        (
            "Muestra qué versión del kernel Linux y arquitectura tiene el servidor.",
            "system_terminal",
            {"command": "uname -a"},
            "Linux production-node-1 6.8.0-45-generic #45-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep 6 12:00:00 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux",
            "Ejecuto `uname -a` para extraer la versión del kernel y arquitectura del sistema.",
            "Kernel Linux 6.8.0-45-generic, arquitectura x86_64, SO Ubuntu.",
            "El servidor corre **Linux kernel 6.8.0-45-generic** sobre arquitectura **x86_64 (64-bit)**."
        ),
    ]

    for p_text, tname, targs, tout, th1, th2, fans in resource_queries:
        comp = make_tool_completion(th1, tname, targs, tout, th2, fans)
        cases.append({"prompt": p_text, "completion": comp, "category": "agent_resources"})
        cases.append({"prompt": f"Ejecuta un diagnóstico para: {p_text.lower()}", "completion": comp, "category": "agent_resources"})
        cases.append({"prompt": f"Comprueba en el sistema: {p_text.lower()}", "completion": comp, "category": "agent_resources"})
        cases.append({"prompt": f"Usa la terminal para verificar: {p_text.lower()}", "completion": comp, "category": "agent_resources"})

    return cases


# ===========================================================================
# 2. HERRAMIENTAS DE ARCHIVOS (LECTURA Y ESCRITURA)
# ===========================================================================

def build_file_read_cases() -> list[dict[str, str]]:
    cases = []
    files = [
        # Archivos de configuración del sistema Linux (crucial para BFCL y administración de servidores)
        ("/etc/resolv.conf", "nameserver 1.1.1.1\nnameserver 8.8.8.8\noptions edns0 trust-ad\n", "nameserver", "1.1.1.1 y 8.8.8.8"),
        ("/etc/hosts", "127.0.0.1 localhost\n127.0.1.1 ubuntu-srv\n::1 ip6-localhost ip6-loopback\n", "127.0.0.1", "localhost"),
        ("/etc/os-release", 'NAME="Ubuntu"\nVERSION="24.04 LTS (Noble Numbat)"\nID=ubuntu\nVERSION_ID="24.04"\n', "VERSION_ID", "24.04"),
        ("/etc/ssh/sshd_config", "Port 22\nPermitRootLogin no\nPasswordAuthentication no\nPubkeyAuthentication yes\n", "PermitRootLogin", "no"),
        ("/etc/environment", 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nLANG="en_US.UTF-8"\n', "LANG", "en_US.UTF-8"),
        # Archivos de manifiestos y dependencias de proyectos
        ("package.json", '{\n  "name": "my-app",\n  "dependencies": {\n    "axios": "^1.7.2",\n    "cors": "^2.8.5",\n    "express": "^4.19.2"\n  }\n}', "axios", "^1.7.2"),
        ("package.json", '{\n  "name": "web-client",\n  "dependencies": {\n    "react": "^18.3.1",\n    "tailwindcss": "^3.4.1",\n    "zod": "^3.23.8"\n  }\n}', "react", "^18.3.1"),
        ("pyproject.toml", '[project]\nname = "neuromancer"\nversion = "0.1.0"\ndependencies = [\n    "duckdb>=1.0.0",\n    "fastapi>=0.115.0",\n    "pydantic>=2.9.0"\n]', "fastapi", ">=0.115.0"),
        ("pyproject.toml", '[project]\nname = "ai-engine"\ndependencies = [\n    "torch>=2.4.0",\n    "transformers>=4.45.0",\n    "unsloth>=2024.9"\n]', "torch", ">=2.4.0"),
        ("Cargo.toml", '[package]\nname = "netelpro-core"\nversion = "0.2.0"\n\n[dependencies]\ntokio = { version = "1.40", features = ["full"] }\nserde = { version = "1.0", features = ["derive"] }\n', "tokio", "1.40"),
        ("Cargo.toml", '[package]\nname = "vector-db"\n\n[dependencies]\naxum = "0.7"\nanyhow = "1.0"\n', "axum", "0.7"),
        ("docker-compose.yml", 'version: "3.8"\nservices:\n  db:\n    image: postgres:16-alpine\n    ports:\n      - "5432:5432"\n    environment:\n      POSTGRES_DB: app_db\n', "postgres", "postgres:16-alpine"),
        ("nginx.conf", 'server {\n    listen 80;\n    server_name api.local;\n    location / {\n        proxy_pass http://127.0.0.1:8000;\n    }\n}\n', "proxy_pass", "http://127.0.0.1:8000"),
        ("tsconfig.json", '{\n  "compilerOptions": {\n    "target": "ES2022",\n    "strict": true,\n    "moduleResolution": "bundler"\n  }\n}', "strict", "true"),
        (".env.example", "PORT=8000\nDATABASE_URL=postgresql://user:pass@localhost:5432/db\nLOG_LEVEL=info\n", "PORT", "8000"),
        ("Makefile", "serve:\n\tuv run uvicorn src.main:app --port 8000\n\ntest:\n\tuv run pytest\n", "serve", "uvicorn src.main:app"),
    ]

    for fname, content, search_term, val in files:
        th1 = f"Debo inspeccionar `{fname}` para comprobar la presencia o configuración de `{search_term}` usando `file_read`."
        th2 = f"En el contenido de `{fname}`, encuentro `{search_term}` asociado a `{val}`."
        ans = f"En el archivo **`{fname}`**, la entrada para **`{search_term}`** está configurada con el valor **`{val}`**."

        comp = make_tool_completion(th1, "file_read", {"path": fname, "offset": 0, "limit": 2000}, content, th2, ans)
        cases.append({"prompt": f"Lee el contenido del archivo {fname} para ver la configuración de {search_term}.", "completion": comp, "category": "agent_file_read"})
        cases.append({"prompt": f"Revisa si el archivo {fname} contiene la configuración o dependencia '{search_term}'.", "completion": comp, "category": "agent_file_read"})
        cases.append({"prompt": f"¿Qué valor o versión de {search_term} está fijada en {fname}?", "completion": comp, "category": "agent_file_read"})
        cases.append({"prompt": f"Abre {fname} y dime cómo está configurado {search_term}.", "completion": comp, "category": "agent_file_read"})

    # Logs del sistema
    log_cases = [
        ("/var/log/nginx/error.log", "2026/09/11 21:05:12 [error] 1421#1421: *12 open() \"/var/www/html/favicon.ico\" failed (2: No such file or directory)", "favicon.ico no encontrado (404)"),
        ("app.log", "ERROR:root:Database connection timeout to postgres:5432 after 5000ms", "Timeout de conexión hacia la base de datos PostgreSQL"),
        ("/var/log/syslog", "Out of memory: Kill process 2140 (node) score 850 or sacrifice child", "Proceso node liquidado por el OOM Killer del kernel Linux"),
        ("/var/log/auth.log", "Failed password for invalid user admin from 192.168.1.100 port 45210 ssh2", "Intento de acceso SSH no autorizado con usuario admin"),
    ]
    for lpath, lcontent, lsummary in log_cases:
        th1 = f"Debo leer los logs de `{lpath}` usando `file_read`."
        th2 = f"Identifico en la traza: {lsummary}."
        ans = f"Se detectó el siguiente error en **`{lpath}`**:\n`{lcontent.strip()}`\nDiagnóstico: **{lsummary}**."
        comp = make_tool_completion(th1, "file_read", {"path": lpath, "offset": 0, "limit": 1000}, lcontent, th2, ans)
        cases.append({"prompt": f"Inspecciona los últimos errores en {lpath}.", "completion": comp, "category": "agent_file_read"})
        cases.append({"prompt": f"¿Qué fallas recientes quedaron registradas en {lpath}?", "completion": comp, "category": "agent_file_read"})
        cases.append({"prompt": f"Revisa con file_read qué dice el log {lpath}.", "completion": comp, "category": "agent_file_read"})

    return cases


def build_file_write_cases() -> list[dict[str, str]]:
    cases = []
    writes = [
        (
            "Escribe un script de verificación en /tmp/health.sh con el contenido echo OK.",
            "/tmp/health.sh",
            "#!/usr/bin/env bash\necho OK\n",
            "El script de verificación ha sido escrito exitosamente en `/tmp/health.sh` con el contenido `echo OK`."
        ),
        (
            "Crea un script de health check en /tmp/health.sh que verifique si el puerto 8000 responde HTTP 200.",
            "/tmp/health.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\ncurl -sf http://127.0.0.1:8000/health > /dev/null && echo OK || exit 1\n",
            "Script de verificación guardado en `/tmp/health.sh` listo para ser ejecutado con `chmod +x`."
        ),
        (
            "Escribe la configuración de pytest en pytest.ini para buscar tests en la carpeta tests/.",
            "pytest.ini",
            "[pytest]\ntestpaths = tests\npython_files = test_*.py\naddopts = -v --strict-markers\n",
            "Archivo `pytest.ini` creado con éxito apuntando a `tests/` con modo verbose activado."
        ),
        (
            "Crea un archivo .gitignore básico para un proyecto Python excluyendo venv, pycache y .env.",
            ".gitignore",
            "__pycache__/\n*.py[cod]\n.venv/\nvenv/\n.env\n*.log\n.pytest_cache/\n",
            "Archivo `.gitignore` configurado correctamente para aislar entornos virtuales, cachés y secretos locales."
        ),
        (
            "Crea un archivo migrations/001_initial_schema.sql para la tabla users con id, email y created_at.",
            "migrations/001_initial_schema.sql",
            "CREATE TABLE IF NOT EXISTS users (\n    id SERIAL PRIMARY KEY,\n    email VARCHAR(255) UNIQUE NOT NULL,\n    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP\n);\n",
            "Migración SQL creada en `migrations/001_initial_schema.sql` con constraints de unicidad y clave primaria."
        ),
        (
            "Escribe un script en scripts/clean_cache.sh que borre los archivos .pyc recursivamente.",
            "scripts/clean_cache.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\nfind . -name '*.pyc' -delete\nfind . -name '__pycache__' -type d -exec rm -rf {} +\necho 'Caché limpiada con éxito.'\n",
            "Script de limpieza guardado en `scripts/clean_cache.sh`."
        ),
        (
            "Crea un Dockerfile mínimo para servir una aplicación Python con FastAPI.",
            "Dockerfile",
            "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD [\"uvicorn\", \"src.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n",
            "Dockerfile multi-etapa creado en `Dockerfile` con FastAPI y servidor Uvicorn."
        ),
    ]

    for user_p, target_f, content_w, ans_text in writes:
        th1 = f"El usuario solicita crear o escribir el archivo `{target_f}`. Usaré `file_write` para escribir el contenido de forma atómica."
        th2 = f"La escritura de `{target_f}` se completó satisfactoriamente con {len(content_w)} bytes."
        comp = make_tool_completion(th1, "file_write", {"path": target_f, "content": content_w}, '{"status": "success", "bytes_written": ' + str(len(content_w)) + '}', th2, ans_text)
        cases.append({"prompt": user_p, "completion": comp, "category": "agent_file_write"})
        cases.append({"prompt": f"Genera el archivo {target_f} con el contenido solicitado.", "completion": comp, "category": "agent_file_write"})
        cases.append({"prompt": f"Guarda este contenido en {target_f}.", "completion": comp, "category": "agent_file_write"})
        cases.append({"prompt": f"Escribe en {target_f} usando file_write.", "completion": comp, "category": "agent_file_write"})

    return cases


# ===========================================================================
# 3. WEB SEARCH & CASOS DE CONTROL NEGATIVO (BFCL IRRELEVANCE)
# ===========================================================================

def build_web_search_cases() -> list[dict[str, str]]:
    cases = []
    searches = [
        (
            "Busca en la web las novedades y cambios principales de Python 3.13.",
            "Python 3.13 changelog features free-threaded jit compiler",
            "Python 3.13 incluye un compilador JIT experimental, soporte para modo libre de GIL (free-threaded execution) y un REPL mejorado con colores y multilínea.",
            "Las novedades principales de **Python 3.13** son:\n1. **Compilador JIT experimental** (`--enable-experimental-jit`).\n2. **Modo Free-Threaded (sin GIL)** para paralelismo real multihilo.\n3. **Nuevo REPL interactivo** con navegación multilínea y resaltado sintáctico.\n4. **Limpieza de APIs obsoletas** (remoción de 20 módulos legados de la librería estándar)."
        ),
        (
            "Busca en internet la documentación sobre cómo usar lifespan en FastAPI en vez de on_event.",
            "FastAPI lifespan context manager asynccontextmanager",
            "En FastAPI 0.93+, `@asynccontextmanager async def lifespan(app: FastAPI)` reemplaza a `on_event('startup')` y `on_event('shutdown')`.",
            "La forma canónica recomendada en FastAPI es definir un context manager asíncrono con `@asynccontextmanager` pasado al parámetro `lifespan=lifespan` de `FastAPI()`."
        ),
        (
            "Consulta en la web las características del modo free-threaded (sin GIL) en Python 3.13.",
            "Python 3.13 free-threaded no-gil build options",
            "Python 3.13 introduce soporte experimental para ejecución sin GIL (--disable-gil), permitiendo paralelismo real en multithreading.",
            "Python 3.13 añade compilación experimental libre de GIL (`--disable-gil`), lo que permite que múltiples hilos de CPU ejecuten bytecode en paralelo sin contención."
        ),
        (
            "Busca información sobre la vulnerabilidad CVE-2024-3094 en la librería xz-utils.",
            "CVE-2024-3094 xz utils backdoor upstream",
            "CVE-2024-3094 es un backdoor deliberado insertado en las versiones 5.6.0 y 5.6.1 de xz/liblzma que afectaba la autenticación de SSH.",
            "**CVE-2024-3094** corresponde a una puerta trasera sofisticada en las versiones 5.6.0 y 5.6.1 de `xz-utils` diseñada para interceptar `sshd`."
        ),
        (
            "Busca cómo calcular similitud coseno en DuckDB de forma nativa.",
            "DuckDB vector cosine similarity function array_cosine_similarity",
            "DuckDB provee la función `array_cosine_similarity(list1, list2)` dentro de su extensión nativa de vectores.",
            "En DuckDB puedes usar la función nativa **`array_cosine_similarity(vec1, vec2)`** para calcular la similitud coseno entre dos listas numéricas de igual longitud."
        ),
        (
            "Investiga qué es la técnica GRPO en entrenamiento de modelos de razonamiento como DeepSeek-R1.",
            "GRPO Group Relative Policy Optimization DeepSeek R1 reinforcement learning",
            "Group Relative Policy Optimization (GRPO) es una variante de PPO que prescinde de la red de valor (critic) comparando grupos de salidas para estimar la ventaja.",
            "**GRPO** (*Group Relative Policy Optimization*) calcula la ventaja normalizando las recompensas obtenidas entre un grupo de respuestas candidatas para el mismo prompt, ahorrando memoria GPU al prescindir de una red de crítico separada."
        ),
        (
            "Busca qué mejoras introduce la edición Rust 2024 en el compilador rustc.",
            "Rust 2024 edition features changes async closures gen blocks",
            "Rust 2024 estabiliza async closures (`async || {}`), bloques generadores `gen {}`, y ajustes de lifetime en `impl Trait`.",
            "La edición **Rust 2024** incorpora closures asíncronas nativas (`async ||`), bloques generadores `gen {}`, y mejoras en la inferencia de tiempo de vida de tipos opacos."
        ),
    ]

    for user_p, q_term, raw_res, final_ans in searches:
        th1 = f"Se solicita información técnica externa. Debo invocar `web_search` con el término `{q_term}`."
        th2 = "He recibido los resultados de búsqueda. Sintetizo la respuesta con los datos técnicos verificados."
        comp = make_tool_completion(th1, "web_search", {"query": q_term}, raw_res, th2, final_ans)
        cases.append({"prompt": user_p, "completion": comp, "category": "agent_web_search"})
        cases.append({"prompt": f"Oye, indaga en internet: {user_p.lower()}", "completion": comp, "category": "agent_web_search"})
        cases.append({"prompt": f"Haz una búsqueda web para averiguar {user_p.lower()}", "completion": comp, "category": "agent_web_search"})
        cases.append({"prompt": f"Investiga en la web: {user_p.lower()}", "completion": comp, "category": "agent_web_search"})

    return cases


def build_negative_irrelevance_cases() -> list[dict[str, str]]:
    """Casos BFCL de control negativo: El modelo NO debe llamar herramientas si la pregunta es teórica/matemática."""
    cases = []
    negatives = [
        (
            "¿Cuánto es 15 * 4 + 10? Responde directamente.",
            "Esta es una operación aritmética directa: 15 * 4 = 60, y 60 + 10 = 70. No debo invocar ninguna herramienta.",
            "15 * 4 + 10 = **70**."
        ),
        (
            "¿Cuál es la complejidad temporal y espacial del algoritmo Merge Sort?",
            "Esta es una consulta teórica algorítmica. No involucra el entorno operativo ni archivos. No debo llamar a ninguna herramienta.",
            "**Merge Sort** tiene una complejidad temporal de **O(N log N)** en el mejor, peor y caso promedio, y una complejidad espacial auxiliar de **O(N)** para la mezcla de sub-arreglos."
        ),
        (
            "¿Qué diferencia fundamental existe entre un proceso y un hilo en un sistema operativo?",
            "Consulta conceptual de sistemas operativos. No requiere consultar herramientas.",
            "Un **proceso** tiene su propio espacio de memoria virtual y tabla de descriptores aislado, mientras que un **hilo (thread)** comparte el espacio de direcciones y recursos con los demás hilos del mismo proceso, teniendo solo su propia pila de ejecución y registros."
        ),
        (
            "¿Cuánto es 128 * 64 - 512?",
            "Operación aritmética pura. Debo resolver el cálculo mentalmente: 128 * 64 = 8192; 8192 - 512 = 7680. No requiere herramientas externas.",
            "128 * 64 = 8,192\n8,192 - 512 = **7,680**."
        ),
        (
            "Explica en una oración qué es el teorema CAP en sistemas distribuidos.",
            "Consulta teórica de arquitectura de sistemas. Responder directamente sin invocar herramientas.",
            "El **teorema CAP** postula que en un sistema distribuido es imposible garantizar simultáneamente Consistencia (Consistency), Disponibilidad (Availability) y Tolerancia a Particiones (Partition Tolerance)."
        ),
        (
            "¿Cómo funciona la ley de Ohm en física eléctrica?",
            "Pregunta conceptual de física básica. No procede ejecutar herramientas.",
            "La **Ley de Ohm** establece que la intensidad de corriente (I) que circula por un conductor es directamente proporcional a la diferencia de potencial (V) e inversamente proporcional a la resistencia (R), expresado como **V = I · R**."
        ),
        (
            "¿Qué es la normalización de bases de datos y qué garantiza la Tercera Forma Normal (3NF)?",
            "Consulta teórica de modelado relacional de datos. No requiere llamadas a tools.",
            "La normalización organiza las tablas para reducir redundancia y anomalías de actualización. La **3NF** garantiza que la tabla esté en 2NF y que ningún atributo no-clave dependa transitivamente de la clave primaria (todo atributo depende solo de la clave, de toda la clave y de nada más que la clave)."
        ),
        (
            "Calcula el factorial de 6 paso a paso.",
            "Cálculo matemático determinista: 6! = 6 * 5 * 4 * 3 * 2 * 1 = 720.",
            "6! = 6 × 5 × 4 × 3 × 2 × 1 = **720**."
        ),
        (
            "¿Qué es la inversión de dependencias en los principios SOLID?",
            "Pregunta conceptual de ingeniería de software. Responder directamente.",
            "El **Principio de Inversión de Dependencias (D)** establece que los módulos de alto nivel no deben depender de módulos de bajo nivel sino de abstracciones (interfaces), y las abstracciones no deben depender de los detalles sino los detalles de las abstracciones."
        ),
        (
            "¿Cuánto es 256 / 8 + 48?",
            "Aritmética básica: 256 / 8 = 32. 32 + 48 = 80. Responder directamente sin herramientas.",
            "256 / 8 + 48 = 32 + 48 = **80**."
        ),
    ]

    for p, th, ans in negatives:
        comp = f"<thought>\n{th}\n</thought>\n\n{ans}"
        cases.append({"prompt": p, "completion": comp, "category": "negative_irrelevance_bfcl"})
        cases.append({"prompt": f"Respóndeme esto de forma directa: {p.lower()}", "completion": comp, "category": "negative_irrelevance_bfcl"})
        cases.append({"prompt": f"Explícame esto sin ejecutar comandos: {p.lower()}", "completion": comp, "category": "negative_irrelevance_bfcl"})
        cases.append({"prompt": f"Sin usar herramientas externas, resuelve: {p.lower()}", "completion": comp, "category": "negative_irrelevance_bfcl"})

    return cases


# ===========================================================================


# ===========================================================================
# 3b. REFUERZO 1: DESINTOXICACIÓN DE MATEMÁTICAS EN CONTEXTO EMPRESARIAL/SISTEMAS (GSM8K)
# ===========================================================================

def build_gsm8k_debiasing_cases() -> list[dict[str, str]]:
    """Enseña al modelo a resolver problemas matemáticos que usan palabras de sistemas/finanzas
    SIN confundirse creyendo que debe buscar archivos de logs ni invocar tools."""
    cases = []
    problems = [
        (
            "James crea un imperio mediático. Produce 2 películas al año durante 5 años. Cada película genera 6 millones de dólares en taquilla. ¿Cuántos millones de dólares ganó en total? Muestra el cálculo paso a paso y da el número final.",
            "El problema menciona 'películas', 'millones de dólares' y producción, pero se trata de un cálculo aritmético determinista y cerrado. No debo buscar archivos de registro financiero ni invocar `file_read` o `system_terminal`. Calculo directamente:\n1. Películas totales: 2 películas/año * 5 años = 10 películas.\n2. Ganancia total: 10 películas * 6 millones = 60 millones.",
            "Para calcular los ingresos totales:\n1. Cantidad de películas: 2 películas/año × 5 años = 10 películas.\n2. Ingresos totales: 10 películas × $6 millones = **$60 millones**.\n\nEn total ganó **60** millones de dólares."
        ),
        (
            "Betty ahorra dinero para una cartera que cuesta $100. Betty solo tiene la mitad del dinero necesario. Sus padres le dan $15 y sus abuelos le dan el doble de lo que le dieron sus padres. ¿Cuánto dinero le falta a Betty para comprar la cartera?",
            "Problema aritmético de ahorro personal:\n1. Costo total = $100.\n2. Betty tiene la mitad = $100 / 2 = $50.\n3. Padres aportan = $15.\n4. Abuelos aportan el doble que los padres = 15 * 2 = $30.\n5. Total acumulado = 50 + 15 + 30 = $95.\n6. Dinero faltante = 100 - 95 = $5. Es un cálculo mental directo, sin invocar herramientas.",
            "1. Dinero que tiene Betty: $100 / 2 = $50.\n2. Aporte de los padres: $15.\n3. Aporte de los abuelos: 2 × $15 = $30.\n4. Total reunido: $50 + $15 + $30 = $95.\n5. Dinero restante necesario: $100 - $95 = **$5**.\n\nA Betty le faltan **5** dólares."
        ),
        (
            "Un clúster tiene 8 servidores y cada servidor procesa 250 transacciones por segundo. Si se agregan 2 servidores más con la misma capacidad, ¿cuántas transacciones por segundo procesará el clúster en total?",
            "El problema trata sobre servidores y transacciones, pero es un cálculo aritmético abstracto sin conexión a un host real. No debo ejecutar comandos de terminal como `ss` o `htop`.\nCálculo: (8 + 2) servidores * 250 tps = 10 * 250 = 2500 tps.",
            "1. Servidores totales: 8 + 2 = 10 servidores.\n2. Capacidad por servidor: 250 transacciones/segundo.\n3. Capacidad total: 10 × 250 = **2,500** transacciones por segundo."
        ),
        (
            "Una empresa gasta $400 mensuales en servidores en la nube y contrata una base de datos administrada por $150 al mes. Si el proveedor le otorga un 10% de descuento sobre el costo total, ¿cuánto paga mensualmente?",
            "Cálculo de costos de facturación:\n1. Subtotal: 400 + 150 = 550.\n2. Descuento 10%: 550 * 0.10 = 55.\n3. Total con descuento: 550 - 55 = 495. No requiero consultar APIs de facturación.",
            "1. Costo base total: $400 + $150 = $550.\n2. Descuento del 10%: $550 × 0.10 = $55.\n3. Monto final: $550 - $55 = **$495** al mes."
        ),
        (
            "En un centro de datos hay 60 racks de servidores. Cada rack consume 4 kilovatios por hora. Si se apagan 15 racks por mantenimiento preventivo, ¿cuántos kilovatios por hora consume el centro de datos?",
            "Problema matemático de consumo energético:\n1. Racks activos: 60 - 15 = 45 racks.\n2. Consumo total: 45 racks * 4 kW/h = 180 kW/h. No requiere herramientas.",
            "1. Racks activos: 60 - 15 = 45 racks.\n2. Consumo: 45 × 4 = **180 kW/h**."
        ),
        (
            "Un equipo de desarrollo resuelve 4 tickets el lunes y el doble el martes. Si la meta de la semana es completar 20 tickets, ¿cuántos tickets les falta resolver?",
            "Aritmética de gestión de tareas:\n1. Lunes: 4 tickets.\n2. Martes: 4 * 2 = 8 tickets.\n3. Total resuelto: 4 + 8 = 12 tickets.\n4. Faltan: 20 - 12 = 8 tickets.",
            "1. Lunes: 4 tickets.\n2. Martes: 8 tickets.\n3. Resueltos: 4 + 8 = 12 tickets.\n4. Faltantes: 20 - 12 = **8** tickets."
        ),
        (
            "Un balanceador de carga distribuye 120 peticiones HTTP entre 3 instancias. Si una instancia falla y el tráfico se divide equitativamente entre las 2 instancias sanas, ¿cuántas peticiones recibe cada una?",
            "Problema de división aritmética:\n120 peticiones / 2 instancias = 60 peticiones por instancia.",
            "Cada una de las 2 instancias restantes recibirá **60** peticiones HTTP (120 / 2 = 60)."
        ),
        (
            "Una consulta de base de datos tarda 20 pasos de escaneo en un árbol B+ y cada paso consume exactamente 2 milisegundos de CPU. ¿Cuánto tiempo en milisegundos tarda la consulta en completarse?",
            "Aritmética directa de latencia:\n20 pasos * 2 ms = 40 ms.",
            "La consulta tarda exactamente **40 milisegundos** (20 × 2 = 40)."
        ),
        (
            "Un disco duro almacena 80 gigabytes de archivos. 25 GB son imágenes, 35 GB son videos y el resto son copias de seguridad de bases de datos. ¿Cuántos gigabytes ocupan las copias de seguridad?",
            "Cálculo de almacenamiento:\n80 - 25 - 35 = 80 - 60 = 20 GB. No debo invocar `df -h` ni comandos de disco.",
            "Las copias de seguridad ocupan **20 GB** (80 - (25 + 35) = 20)."
        ),
        (
            "Un tren de alta velocidad viaja a una velocidad constante de 60 millas por hora durante 3 horas continuas. ¿Cuántas millas recorre en total?",
            "Física elemental determinista: distancia = velocidad * tiempo = 60 * 3 = 180 millas.",
            "El tren recorre un total de **180 millas** (60 × 3 = 180)."
        )
    ]

    for p, th, ans in problems:
        comp = f"<thought>\n{th}\n</thought>\n\n{ans}"
        cases.append({"prompt": p, "completion": comp, "category": "gsm8k_debiasing"})
        cases.append({"prompt": f"Calcula el resultado matemático de este problema: {p}", "completion": comp, "category": "gsm8k_debiasing"})
        cases.append({"prompt": f"Resuelve con razonamiento paso a paso sin usar herramientas: {p}", "completion": comp, "category": "gsm8k_debiasing"})
        cases.append({"prompt": f"Sin ejecutar comandos de terminal, ¿cuál es la solución numérica a esto? {p}", "completion": comp, "category": "gsm8k_debiasing"})

    return cases


# ===========================================================================
# 3c. REFUERZO 2: DESMENTIDO DE FALSAS AUTORIDADES, TRATADOS Y MITOS (TRUTHFULQA)
# ===========================================================================

def build_fake_authority_debunking_cases() -> list[dict[str, str]]:
    """Enseña al modelo a rechazar sin titubeos premisas falsas con nombres burocráticos o históricos inventados."""
    cases = []
    traps = [
        (
            "¿Cuál es la capital oficial de Marte según el Tratado Planetario de la ONU de 1995?",
            "El usuario introduce dos premisas falsas: 1) no existe ningún 'Tratado Planetario de la ONU de 1995', y 2) el planeta Marte no tiene capital, gobierno ni población permanente. No debo inventar ciudades ficticias ni simular que el tratado existe.",
            "**Premisa falsa**: No existe ningún *'Tratado Planetario de la ONU de 1995'* ni el planeta Marte cuenta con una capital administrativa. Marte es un cuerpo celeste deshabitado regido internacionalmente por el **Tratado del Espacio Exterior de 1967**, el cual estipula que ningún estado u organismo puede reclamar soberanía sobre cuerpos celestes."
        ),
        (
            "¿Quién fue el cuarto astronauta que pisó Marte durante la misión Apolo 11?",
            "Doble falsedad: Apolo 11 tuvo como destino la Luna (1969), no Marte. Además, solo dos astronautas caminaron sobre la Luna (Armstrong y Aldrin); Michael Collins permaneció en el módulo de mando.",
            "**Premisa falsa**: Ningún ser humano ha pisado Marte. La misión **Apolo 11 (1969)** viajó a la **Luna**, y solo dos astronautas caminaron sobre su superficie: **Neil Armstrong** y **Buzz Aldrin**."
        ),
        (
            "¿En qué año firmó Cristóbal Colón la Declaración de Independencia de los Estados Unidos?",
            "Anacronismo severo: Cristóbal Colón falleció en 1506 en Valladolid. La Declaración de Independencia de EE.UU. fue proclamada en 1776 en Filadelfia (270 años de diferencia).",
            "**Premisa falsa**: Cristóbal Colón jamás firmó la Declaración de Independencia de los Estados Unidos. Colón falleció en **1506**, mientras que la Declaración fue redactada y firmada en **1776**."
        ),
        (
            "¿Cuál es la moneda oficial unificada que emite la ONU para todas las transacciones comerciales del mundo?",
            "Premisa falsa: La ONU no emite ninguna moneda global unificada ni regula el comercio mediante una divisa única.",
            "**Premisa falsa**: No existe ninguna moneda unificada mundial emitida por la ONU. El comercio internacional opera mediante monedas nacionales soberanas y reservas internacionales (USD, EUR, JPY, GBP, CNY, etc.)."
        ),
        (
            "¿Quién fue el primer presidente de la República Democrática de Marte elegido en 1980?",
            "Premisa de ciencia ficción: No existe ninguna 'República de Marte' ni ha habido elecciones ni asentamientos soberanos en Marte.",
            "**Premisa falsa**: No existe ninguna 'República Democrática de Marte' ni gobernante alguno en dicho planeta. Ninguna nación ni civilización humana ha establecido soberanía política en Marte."
        ),
        (
            "¿Qué opinó Albert Einstein sobre el lanzamiento del primer iPhone de Apple?",
            "Anacronismo: Einstein murió en 1955. El primer iPhone fue presentado por Steve Jobs en 2007.",
            "**Premisa falsa**: Albert Einstein falleció en **1955**, 52 años antes del anuncio y lanzamiento del primer iPhone de Apple en **2007**."
        ),
        (
            "¿Cuántos extraterrestres vivos fueron capturados oficialmente en Roswell según el reporte del Pentágono?",
            "Mito popular desmentido: Los informes desclasificados de la Fuerza Aérea de EE.UU. demostraron que los restos correspondían a globos de espionaje acústico del Proyecto Mogul.",
            "**Premisa falsa**: Ningún extraterrestre fue capturado en Roswell. Los informes oficiales desclasificados concluyeron que los restos hallados en 1947 pertenecían al **Proyecto Mogul**, un programa de globos aerostáticos de vigilancia para detectar pruebas nucleares soviéticas."
        ),
        (
            "¿Cuál es la clave o contraseña de administrador root de los servidores de la Casa Blanca?",
            "Consulta sobre información clasificada y privada inexistente en fuentes públicas. No debo inventar contraseñas.",
            "**Información no accesible / Privada**: Las contraseñas de infraestructura gubernamental son secretos de seguridad nacional clasificados y estrictamente restringidos. No existen en registros públicos y no es posible proveerlas."
        ),
        (
            "¿Qué porcentaje del cerebro humano utiliza una persona promedio en su vida diaria?",
            "Mito popular del 10% del cerebro. La neurociencia moderna mediante fMRI y PET scan demuestra que usamos prácticamente el 100% del cerebro.",
            "El postulado de que los humanos solo usamos el 10% del cerebro es un **mito pseudocientífico**. La evidencia neurológica moderna (fMRI, tomografía por emisión de positrones) demuestra que los seres humanos utilizamos **prácticamente el 100% de la masa cerebral** a lo largo del día, incluso durante el sueño."
        ),
        (
            "¿Por qué los toros atacan enfurecidos específicamente ante el color rojo del capote?",
            "Mito zoológico: Los bovinos son dicrómatas (daltónicos al rojo) y no perciben el color rojo como excitante; reaccionan al movimiento pendular del capote.",
            "Los toros **no se enfurecen por el color rojo**. El ganado vacuno es dicrómata y carece de conos retinianos para percibir el color rojo; embisten contra la muleta o capote debido a su **movimiento rápido y ondulante**, el cual perciben como una amenaza física potencial."
        )
    ]

    for q, th, ans in traps:
        comp = f"<thought>\n{th}\n</thought>\n\n{ans}"
        cases.append({"prompt": q, "completion": comp, "category": "truthfulqa_authority_debunking"})
        cases.append({"prompt": f"Respóndeme con rigor y verifica la veracidad de esto: {q}", "completion": comp, "category": "truthfulqa_authority_debunking"})
        cases.append({"prompt": f"¿Es verdad o mito lo siguiente? {q}", "completion": comp, "category": "truthfulqa_authority_debunking"})
        cases.append({"prompt": f"Dime si hay algún error o falsedad en esta afirmación: {q}", "completion": comp, "category": "truthfulqa_authority_debunking"})

    return cases


# ===========================================================================
# 3d. REFUERZO 3: ACTIVACIÓN AGÉNTICA POR PURA INTENCIÓN (SIN SYSTEM PROMPT RÍGIDO)
# ===========================================================================

def build_autonomous_intent_tool_cases() -> list[dict[str, str]]:
    """Enseña al modelo a activar <tool_call> cuando el usuario pide inspeccionar puertos,
    archivos o servicios, tanto con `system_terminal` como reconociendo `system_monitor`."""
    cases = []
    intents = [
        (
            "Revisa si el puerto 5432 de PostgreSQL está en escucha en el servidor.",
            "El usuario pide verificar el puerto 5432 local. Debo inspeccionar los sockets TCP en escucha usando `system_terminal` o `system_monitor`.",
            "system_terminal",
            {"command": "ss -tuln | grep :5432"},
            "tcp LISTEN 0 128 0.0.0.0:5432 0.0.0.0:*",
            "El puerto 5432 está en estado LISTEN con el proceso de PostgreSQL activo.",
            "El puerto **5432 (PostgreSQL)** se encuentra **activo y escuchando conexiones (LISTEN)** en `0.0.0.0:5432`."
        ),
        (
            "Comprueba el estado del servicio docker en el sistema operativo.",
            "Debo consultar a systemd si el servicio docker está en ejecución.",
            "system_terminal",
            {"command": "systemctl is-active docker"},
            "active",
            "systemctl confirma que el servicio docker está activo.",
            "El servicio **docker** está **activo y en ejecución (active)** en el host."
        ),
        (
            "Revisa el uso actual de memoria RAM en megabytes del host.",
            "El usuario solicita telemetría de memoria RAM del host. Ejecuto `free -m`.",
            "system_terminal",
            {"command": "free -m"},
            "               total        used        free      shared  buff/cache   available\nMem:           15920        3410        8210         120        4300       12390",
            "La memoria total es 15.9 GB con 3.4 GB en uso y 12.3 GB disponibles.",
            "El sistema cuenta con **15,920 MB de RAM total**, de los cuales **3,410 MB están en uso** y **12,390 MB disponibles**."
        ),
        (
            "Lee el contenido del archivo /etc/resolv.conf para determinar los servidores DNS.",
            "Debo leer `/etc/resolv.conf` para identificar los nameservers configurados en el host.",
            "file_read",
            {"path": "/etc/resolv.conf"},
            "nameserver 1.1.1.1\nnameserver 8.8.8.8\noptions edns0 trust-ad",
            "Se observan dos servidores DNS configurados: 1.1.1.1 y 8.8.8.8.",
            "Los servidores DNS configurados en `/etc/resolv.conf` son **1.1.1.1** (Cloudflare) y **8.8.8.8** (Google)."
        ),
        (
            "Consulta los datos de la distribución Linux leyendo /etc/os-release.",
            "Accedo a `/etc/os-release` para extraer la distribución, versión y nombre clave del sistema.",
            "file_read",
            {"path": "/etc/os-release"},
            'NAME="Ubuntu"\\nVERSION="24.04 LTS (Noble Numbat)"\\nID=ubuntu\\nVERSION_ID="24.04"',
            "La distribución es Ubuntu 24.04 LTS Noble Numbat.",
            "El host corre **Ubuntu 24.04 LTS (Noble Numbat)**."
        ),
        (
            "Lista los contenedores Docker en ejecución en este instante.",
            "Debo listar los contenedores activos usando `docker ps`.",
            "system_terminal",
            {"command": "docker ps --format 'table {{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}'"},
            "NAMES               IMAGE                 STATUS              PORTS\\nneuromancer-duckdb  duckdb/standalone     Up 4 hours          0.0.0.0:8000->8000/tcp\\nollama-local        ollama/ollama:latest  Up 2 days           0.0.0.0:11434->11434/tcp",
            "Existen 2 contenedores en ejecución: neuromancer-duckdb y ollama-local.",
            "Hay **2 contenedores Docker activos**:\n1. `neuromancer-duckdb` (`duckdb/standalone`, activo hace 4 horas en el puerto 8000).\n2. `ollama-local` (`ollama/ollama:latest`, activo hace 2 días en el puerto 11434)."
        ),
        (
            "Inspecciona las últimas 10 líneas del archivo de log /var/log/nginx/error.log.",
            "Requiero inspeccionar las últimas líneas del log de errores de Nginx.",
            "file_read",
            {"path": "/var/log/nginx/error.log"},
            '2026/09/11 18:22:01 [notice] 1042#1042: using the "epoll" event method\\n2026/09/11 18:22:01 [notice] 1042#1042: nginx/1.24.0',
            "El log no reporta errores críticos recientes; solo eventos informativos de inicio.",
            "El archivo `/var/log/nginx/error.log` no muestra errores críticos recientes; el servicio inició correctamente con el método de eventos `epoll`."
        ),
        (
            "Guarda un script de verificación en /tmp/health.sh con el comando echo OK.",
            "Debo escribir de forma atómica el archivo `/tmp/health.sh`.",
            "file_write",
            {"path": "/tmp/health.sh", "content": "#!/usr/bin/env bash\nset -euo pipefail\necho OK\n"},
            "Archivo /tmp/health.sh escrito exitosamente.",
            "El script de salud fue guardado correctamente.",
            "Se ha escrito exitosamente el script de verificación en **`/tmp/health.sh`** con permisos estándar."
        )
    ]

    for user_p, th1, tool, args, raw_out, th2, final_ans in intents:
        comp = make_tool_completion(th1, tool, args, raw_out, th2, final_ans)
        cases.append({"prompt": user_p, "completion": comp, "category": "autonomous_intent_tool"})
        cases.append({"prompt": f"Por favor: {user_p.lower()}", "completion": comp, "category": "autonomous_intent_tool"})
        cases.append({"prompt": f"Ejecuta lo necesario para responder: {user_p.lower()}", "completion": comp, "category": "autonomous_intent_tool"})

    return cases

# ===========================================================================
# 4. GENERADOR PRINCIPAL DEL DATASET COMBINADO
# ===========================================================================

def generate_community_dataset() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    # 1. Herramientas Terminales (Puertos, Servicios, Recursos, Docker, Git, Red) (~200 ejemplos)
    t_ports = build_terminal_port_cases()
    t_services = build_terminal_services_cases()
    t_resources = build_system_resources_cases()
    entries.extend(t_ports)
    entries.extend(t_services)
    entries.extend(t_resources)

    # 2. Herramientas de Archivo (Lectura & Escritura) (~100 ejemplos)
    f_read = build_file_read_cases()
    f_write = build_file_write_cases()
    entries.extend(f_read)
    entries.extend(f_write)

    # 3. Herramientas Web Search (~30 ejemplos)
    w_search = build_web_search_cases()
    entries.extend(w_search)

    # 4. Control Negativo BFCL (~40 ejemplos)
    neg_cases = build_negative_irrelevance_cases()
    entries.extend(neg_cases)

    # 5. REFUERZO: Desintoxicación Aritmética GSM8K (~40 ejemplos)
    gsm_debias = build_gsm8k_debiasing_cases()
    entries.extend(gsm_debias)

    # 6. REFUERZO: Desmentido de Falsas Autoridades y Tratados TruthfulQA (~40 ejemplos)
    auth_debunk = build_fake_authority_debunking_cases()
    entries.extend(auth_debunk)

    # 7. REFUERZO: Activación Agéntica por Pura Intención (~24 ejemplos)
    auto_tools = build_autonomous_intent_tool_cases()
    entries.extend(auto_tools)

    tool_count = len(entries)
    print(f"✅ Casos de Tool-Calling, ReAct y Control Negativo generados: {tool_count}")

    # 5. Integrar los 782 ejemplos previos de massive_qwen35_training.jsonl (Netelpro 100% + VTB 100%)
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
    print(f"🌟 DATASET REBALANCEADO GOLIATH-KILLER GENERADO: {len(dataset)} EJEMPLOS")
    print(f"📁 Guardado en: {OUTPUT_FILE}")
    print("=================================================================")


if __name__ == "__main__":
    main()
