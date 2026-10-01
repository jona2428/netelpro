"""Puente al repo netelpro (tareas, verificador, prompt) + verificador de Python."""
import subprocess, sys, json, tempfile, textwrap
from config import NETELPRO_REPO

sys.path.insert(0, str(NETELPRO_REPO))
from rlvr.tasks import TASK_MODULE_NAMES, OOD_TASK_IDS, load_task, split_train_ood  # noqa: E402
from rlvr.prompting import build_prompt as _build_prompt_sl                       # noqa: E402
from rlvr.verify import verify_program                                            # noqa: E402
from rlvr.gguf_eval import extract_src                                            # noqa: E402

TRAIN_IDS, OOD_IDS = split_train_ood(list(TASK_MODULE_NAMES))

SL_SUFFIX = ("\nResponde SOLO con un bloque ```netelpro``` que contenga el programa. "
             "Sin explicación ni texto antes o después.\n")


def build_prompt_sl(task) -> str:
    """Prompt fijo de netelpro/rlvr + instrucción de solo-código (igual para todos los brazos).
    El smoke test 1 mostró que el E2B base contesta con prosa y se trunca antes de escribir el bloque."""
    return _build_prompt_sl(task) + SL_SUFFIX


def build_prompt_py(task) -> str:
    """Mismo enunciado, pero pide Python plano (medida de transferencia a un lenguaje conocido)."""
    name = task.FN_NAME.replace("-", "_")
    return (f"Escribe una función de Python llamada `{name}` para esta tarea:\n{task.DESCRIPTION_ES}\n"
            f"Responde solo con un bloque ```python``` con la función, sin explicación.\n")


def verify_python(src: str, task, n_cases: int = 20, timeout: int = 10) -> bool:
    """Ejecuta la función del modelo en subproceso y la compara con task.reference en n_cases entradas."""
    name = task.FN_NAME.replace("-", "_")
    cases = [[list(a) if isinstance(a, tuple) else a for a in args] for args in task.gen_inputs(n_cases, 12345)]
    expected = [task.reference(*args) for args in task.gen_inputs(n_cases, 12345)]
    harness = textwrap.dedent(f"""
        import json
        {textwrap.indent(src, '        ').strip()}
        cases = json.loads({json.dumps(json.dumps(cases))})
        expected = json.loads({json.dumps(json.dumps(expected, default=list))})
        out = [{name}(*c) for c in cases]
        out = [list(o) if isinstance(o, tuple) else o for o in out]
        print(json.dumps(out == expected))
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(harness)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0 and r.stdout.strip().endswith("true")
    except subprocess.TimeoutExpired:
        return False


def extract_py(out: str):
    import re
    m = re.search(r"```python\s*(.*?)```", out, re.DOTALL) or re.search(r"```\s*(.*?)```", out, re.DOTALL)
    return m.group(1).strip() if m else None
