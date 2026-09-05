; Straylight v0.1 -- structurally valid demo (audited by tools/check_arity.py)

(def limit 10)

(defn clamp (x)
  (if (< x 0)
      0
      (if (> x limit) limit x)))

(defn sum-to (n acc)
  (if (== n 0)
      acc
      (sum-to (- n 1) (+ acc n))))

(defn fib (n)
  (if (< n 2)
      n
      (+ (fib (- n 1)) (fib (- n 2)))))

(def total (sum-to limit 0))
(def clamped (clamp -3))
(def fifth (fib 5))
(def label (str-cat "fib(5)=" (int->str fifth)))
(def xs (cons 1 (cons 2 (cons 3 nil))))
(def packed (list 1 2 3))
(def nothing (list))
(def flag (and (< clamped 5) (not false)))
(def ratio (/ total 4))
(def scaled (* ratio 2.5))
(let tmp (+ total 1) (- tmp 1))
(sorry "phase 4 will make this hole impossible to miss")
(grant io)