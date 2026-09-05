; Netelpro v0.1 -- Tail Call Optimization (TCO) demonstration

(defn sum-to (n acc)
  (if (== n 0)
      acc
      (sum-to (- n 1) (+ acc n))))

(sum-to 100000 0)
