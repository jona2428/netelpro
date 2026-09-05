; Netelpro v0.3 example: the Neuromancer zone policy as a pure gate rule.
; Strings cross the native boundary as read-only NUL-terminated i8*
; (ctypes c_char_p, UTF-8): the rule DECIDES over paths, it never produces text.
;
; Zone policy:
;   red    (.env, routes.py, container.py) -> reject always
;   yellow (src/, tests/, skills/)         -> approve only with explicit approval
;   green  (everything else)               -> allow
;
; Params: path (Str), approved (Bool), mode (Int: 0=read, 1=write).
; Boundary law (v0.3): params are Int (i64), Bool (i1) or Str (i8*); strings
; cross read-only via c_char_p; ==/!= are type-aware (icmp vs strcmp);
; prefix? is strncmp(text, prefix, strlen(prefix)) == 0.

(defn filter-rule (path approved mode)
  (if (or (== path ".env") (or (== path "routes.py") (== path "container.py")))
      false
      (if (or (prefix? path "src/") (or (prefix? path "tests/") (prefix? path "skills/")))
          (and approved (== mode 1))
          true)))