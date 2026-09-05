;; Phase 4 negative example: this must DIE at parse time — the head 'missing-fn'
;; resolves to nothing (not a primitive, not a declared defn, not a parameter).
;; The compiler-as-prosecutor refuses silent holes.
(defn main () (missing-fn 1))