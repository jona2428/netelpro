; intentionally broken: violates the arity contract in several ways
(def x 10)
(+ 1 2 3)
(if (< x 1) 2)
(defn add (a b) (+ a b))
(add 1)
(unknown-op 1 2)
(defn dup (p p) p)
(sorry 42)