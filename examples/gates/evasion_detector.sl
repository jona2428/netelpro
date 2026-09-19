; Evasion detector as compiled native code.
; The Python _audit_sniff is regex-over-text and evadible (chr() obfuscation,
; dynamic path building, nested subprocess). This rule decides the same
; question in machine code: pure by design, compiled to LLVM, no fallback
; that can silently degrade to "allow".
;
; Inputs (the Python caller lowercases the text first):
;   text     Str  - the full command/code text being dispatched
;   approved Bool - explicit human approval recorded this turn
;   mode     Int  - 0=read, 1=write
;
; Law:
;   1. Credential/secret mentions deny ALWAYS (any mode, any approval):
;      .env / routes.py / container.py / api_key / apikey / "bearer " /
;      password / token= / secrets. contains? finds them at ANY position
;      in the text (the exact hole of prefix?-only detection).
;   2. Yellow roots with write (mode 1) need explicit approval:
;      src/ / tests/ / skills/. contains? over the TEXT (not anchored
;      prefix) because a command like "edit src/main.py" touches a yellow
;      root mid-string. Over-matching is the safe direction for a tripwire.
;   3. Everything else (read-only work) allows.

(defn mentions-secret (s)
  (or (contains? s ".env")
      (or (contains? s "routes.py")
      (or (contains? s "container.py")
      (or (contains? s "api_key")
      (or (contains? s "apikey")
      (or (contains? s "bearer ")
      (or (contains? s "password")
      (or (contains? s "token=")
      (contains? s "secrets"))))))))))

(defn touches-yellow (s)
  (or (contains? s "src/")
      (or (contains? s "tests/")
      (contains? s "skills/"))))

(defn filter-rule (text approved mode)
  (if (mentions-secret text)
      false
      (if (and (== mode 1) (touches-yellow text))
          approved
          true)))