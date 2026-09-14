"""Netelpro Native Tokenizer with Expanded Spanish, Code, and Logic Vocabulary.

Provides deterministic tokenization for natural language, structured code,
logical reasoning traces, and formal Netelpro contract delimiters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|user|>",
    "<|assistant|>",
    "<|thought|>",
    "<|endthought|>",
    "<|contract|>",
    "<|endcontract|>",
    "<|audit|>",
]


class NetelproTokenizer:
    """Vocabulary and tokenizer for the Netelpro Mini / Mega LLM."""

    def __init__(self, custom_vocab: dict[str, int] | None = None) -> None:
        if custom_vocab is not None:
            self.token_to_id = dict(custom_vocab)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        else:
            self.token_to_id = {}
            self.id_to_token = {}
            self._build_default_vocab()

        self.pad_token_id = self.token_to_id["<|pad|>"]
        self.bos_token_id = self.token_to_id["<|bos|>"]
        self.eos_token_id = self.token_to_id["<|eos|>"]
        self.unk_token_id = self.token_to_id["<|unk|>"]
        self._sorted_tokens = sorted(self.token_to_id.keys(), key=lambda s: len(s), reverse=True)

    def _build_default_vocab(self) -> None:
        idx = 0
        # 1. Special tokens
        for tok in SPECIAL_TOKENS:
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok
            idx += 1

        # 2. Common whitespaces and symbols
        basic_symbols = [
            " ", "\n", "\t", "  ", "    ", "        ",
            ".", ",", ";", ":", "!", "?", "¿", "¡",
            "(", ")", "[", "]", "{", "}", "<", ">",
            "=", "+", "-", "*", "/", "%", "^", "&", "|", "~",
            "_", "@", "#", "$", "\\", "'", '"', "`", "==", "!=", "<=", ">=",
            "->", "=>", "+=", "-=", "/*", "*/", "//",
        ]
        for sym in basic_symbols:
            if sym not in self.token_to_id:
                self.token_to_id[sym] = idx
                self.id_to_token[idx] = sym
                idx += 1

        # 3. Digits 0-9
        for d in "0123456789":
            if d not in self.token_to_id:
                self.token_to_id[d] = idx
                self.id_to_token[idx] = d
                idx += 1

        # 4. English & Spanish alphabets
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáéíóúÁÉÍÓÚñÑüÜ"
        for ch in letters:
            if ch not in self.token_to_id:
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch
                idx += 1

        # 5. Rich Spanish, English, Code, and Logic Word Vocabulary
        words = [
            # Español general y conversacional
            "el", "la", "de", "que", "y", "en", "un", "ser", "se", "no",
            "haber", "por", "con", "su", "para", "como", "estar", "tener", "le",
            "lo", "todo", "pero", "más", "hacer", "o", "poder", "este", "ya",
            "otro", "ese", "si", "me", "primer", "dar", "tiempo", "muy", "mismo",
            "yo", "también", "hasta", "año", "dos", "querer", "entre", "así",
            "primero", "desde", "grande", "eso", "ni", "nos", "llegar", "pasar",
            "Hola", "hola", "cómo", "estás", "bien", "gracias", "puedo", "ayudarte",
            "hermano", "Hermano", "tal", "Tal", "amigo", "Amigo", "jona", "Jona",
            "Jonathan", "creador", "Creador", "saludos", "Saludos", "buenas", "Buenas",
            "noches", "días", "día", "excelente", "genial", "listo", "lista", "claro",
            "hoy", "Soy", "soy", "un", "una", "unos", "unas", "modelo", "lenguaje",
            "humano", "personas", "mundo", "vida", "social", "sociedad", "cultura",
            "verdad", "razón", "razonamiento", "pensamiento", "mente", "idea",
            "ideas", "análisis", "analizar", "concepto", "conceptos", "ejemplo",
            "ejemplos", "pregunta", "respuesta", "explicar", "explicación", "claro",
            "clara", "claridad", "rigor", "correcto", "correcta", "válido", "válida",
            "falacia", "falacias", "lógica", "lógico", "retórica", "retórico",
            "dialéctica", "filosofía", "ética", "moral", "justicia", "ley", "leyes",
            "derecho", "derechos", "libertad", "orden", "instituciones", "estado",
            "política", "democracia", "consenso", "diálogo", "argumento", "argumentos",
            "tesis", "antítesis", "síntesis", "conclusión", "evidencia", "datos",
            "premisas", "premisa", "Ad", "Hominem", "Falsa", "Dicotomía", "Hombre",
            "Paja", "atacar", "persona", "opciones", "extremos", "distorsionar",
            "incurre", "comete", "señalar", "desmontar", "evaluar", "garantizar",
            "seguro", "segura", "seguridad", "alucinación", "alucinaciones", "evitar",
            "poda", "podar", "cero", "absoluto", "inmediato", "inmediata",
            "sistema", "sistemas", "complejo", "complejos", "estructura", "estructuras",
            "red", "redes", "neurona", "neuronas", "neuronal", "neuronales",
            "silicio", "hardware", "cómputo", "computación", "memoria", "tiempo",
            "microsegundos", "nanosegundos", "milésimas", "segundos", "rápido",
            "fluido", "fluidez", "auditoría", "certificado", "formal", "formales",
            "regla", "reglas", "contrato", "contratos", "invariante", "invariantes",
            "inhibición", "inhibir", "corte", "fail-closed", "admitido", "bloqueado",

            # Programación en Python
            "def", "return", "class", "import", "from", "as", "if", "elif", "else",
            "for", "while", "in", "break", "continue", "pass", "try", "except",
            "finally", "raise", "assert", "with", "yield", "lambda", "global",
            "True", "False", "None", "self", "print", "len", "range", "list",
            "dict", "set", "tuple", "str", "int", "float", "bool", "type",
            "isinstance", "enumerate", "zip", "sum", "min", "max", "sorted",
            "append", "extend", "insert", "remove", "pop", "clear", "keys",
            "values", "items", "get", "update", "split", "join", "strip",
            "replace", "find", "startswith", "endswith", "lower", "upper",
            "binary_search", "arr", "target", "low", "high", "mid", "index",
            "pytest", "test", "unittest", "function", "variable", "parameter",
            "edge_cases", "boundary", "off_by_one", "strict", "safe_execute",

            # Artes, Música, Historia y Ciencias
            "música", "musical", "armonía", "acorde", "acordes", "escala", "escalas",
            "nota", "notas", "ritmo", "melodía", "frecuencia", "frecuencias", "tono",
            "arte", "artes", "estética", "belleza", "pintura", "color", "colores",
            "composición", "áurea", "proporción", "perspectiva", "escultura",
            "historia", "histórico", "histórica", "Revolución", "Grecia", "Roma",
            "Renacimiento", "Ilustración", "humana", "civilización", "culturas",
            "matemática", "matemáticas", "álgebra", "cálculo", "derivada", "integral",
            "matriz", "matrices", "vector", "vectores", "teorema", "teoremas",
            "espacio", "espacios", "dimensión", "dimensiones", "grafo", "grafos",

            # Sintaxis Netelpro (.sl y Lisp)
            "netelpro", "Netelpro", "neuro", "gate", "RuleFilter", "defn",
            "filter-rule", "truth-table", "in-range", "and", "or", "not",
            "Int", "Bool", "step", "verify", "proof", "theorem", "state",
            "action_id", "allowed_min", "allowed_max", "safety_state",
            "potential", "z_min", "z_max", "control_flag", "action",
            "AuditCertificate", "LayerAuditRecord", "NetelproLayer",
            "NetelproTransformer", "NetelproMiniLLM", "NetelproTokenizer",
            "NetelproTriviumEngine", "TriviumAuditRecord", "NetelproActivationSTE",
            "NetelproBinaryMemoryBank",

            # Palabras comunes en inglés técnico
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
        ]

        for w in words:
            w_spaced = " " + w
            if w_spaced not in self.token_to_id:
                self.token_to_id[w_spaced] = idx
                self.id_to_token[idx] = w_spaced
                idx += 1
            if w not in self.token_to_id:
                self.token_to_id[w] = idx
                self.id_to_token[idx] = w
                idx += 1

        # Cargar vocabulario completo desde el dataset sintético si existe
        import re
        corpus_candidates = [
            Path(__file__).parent.parent.parent / "training" / "data" / "teo_train.jsonl",
            Path(__file__).parent.parent.parent / "training" / "data" / "mega_train.jsonl",
            Path(__file__).parent.parent.parent / "training" / "data" / "mini_llm_train.jsonl",
        ]
        for c_file in corpus_candidates:
            if c_file.exists():
                try:
                    with open(c_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line)
                                text = data.get("prompt", "")
                                for word in re.findall(r'[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9_]+', text):
                                    w_sp = " " + word
                                    if w_sp not in self.token_to_id:
                                        self.token_to_id[w_sp] = idx
                                        self.id_to_token[idx] = w_sp
                                        idx += 1
                                    if word not in self.token_to_id:
                                        self.token_to_id[word] = idx
                                        self.id_to_token[idx] = word
                                        idx += 1
                            except Exception:
                                pass
                except Exception:
                    pass

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encodes string to a list of token IDs using greedy longest-match with case-insensitivity."""
        tokens: list[int] = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        i = 0
        n = len(text)
        sorted_tokens = self._sorted_tokens

        while i < n:
            match_found = False
            remaining = text[i:]

            # 1. Exact match (longest match first)
            for cand in sorted_tokens:
                if remaining.startswith(cand):
                    tokens.append(self.token_to_id[cand])
                    i += len(cand)
                    match_found = True
                    break

            # 2. Case-insensitive fallback for multi-character words
            if not match_found:
                rem_lower = remaining.lower()
                for cand in sorted_tokens:
                    c_len = len(cand)
                    if c_len > 1 and rem_lower.startswith(cand.lower()):
                        tokens.append(self.token_to_id[cand])
                        i += c_len
                        match_found = True
                        break

            # 3. Single character fallback
            if not match_found:
                ch = text[i]
                tid = self.token_to_id.get(ch, self.token_to_id.get(ch.lower(), self.unk_token_id))
                tokens.append(tid)
                i += 1

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Decodes list of token IDs back into text."""
        parts = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "<|unk|>")
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            parts.append(tok)
        return "".join(parts)

    def format_chat(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Formats chat dialog using standard Netelpro delimiters."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}<|eos|>\n"
            elif role == "system":
                formatted += f"<|contract|>\n{content}\n<|endcontract|>\n"

        if add_generation_prompt:
            formatted += "<|assistant|>\n"
        return formatted

    def save(self, path: str | Path) -> None:
        """Saves vocabulary mapping to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> NetelproTokenizer:
        """Loads tokenizer from a JSON vocabulary file."""
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(custom_vocab=vocab)
