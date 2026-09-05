; Netelpro v0.1 -- Fibonacci demonstration

(defn fib (n)
  (if (< n 2)
      n
      (+ (fib (- n 1)) (fib (- n 2)))))

(fib 15)
