; Gate: expense approval (finance / spend controls).
; Contract: (filter-rule amount manager emergency) -> 1 allow, 0 deny.
;   amount    : Int (i64), transaction value in base units
;   manager   : 0/1 flag -- manager pre-approval
;   emergency : 0/1 flag -- declared operational emergency
; Policy:
;   ALLOW when amount <= 500 AND manager approved,
;   OR amount <= 50 AND emergency declared (small stop-bleed spend).
; Boundary law (v0.1): all params are Int (i64). Booleans cross the
; machine boundary as 0/1 flags -- no implicit coercion, ever.
;
; Truth table (representative rows; full space tested in tests/test_gate_examples.py):
;   amount manager emergency | allow
;   500    1       0         | 1
;   501    1       0         | 0
;   50     0       1         | 1
;   51     0       1         | 0
;   500    0       0         | 0

(defn filter-rule (amount manager emergency)
  (or (and (<= amount 500) (== manager 1))
      (and (<= amount 50) (== emergency 1))))