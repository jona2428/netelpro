(defn add (x y) (+ x y))

;; Phase 4: a declared hole compiles clean — it is listed in the manifest,
;; never hidden. Executing it raises: hole 'not-yet' not implemented.
(defn not-yet (x) (sorry "TODO: divide implementation pending"))

(defn main () (add 2 3))

(main)