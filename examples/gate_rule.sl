; First real use case: Neuromancer gate rule compiled to native code.
; The rule: a specialist delegation is allowed when
;   - the delegation priority is high (>= 3), AND
;   - the loop-detector confidence is not saturated (< 90), OR
;   - this is an explicit user-approved escalation (1/0 flag).
; Boundary law (v0.1): all params are Int (i64). Booleans cross the
; machine boundary as 0/1 flags -- no implicit coercion, ever.

(defn filter-rule (priority confidence approved)
  (or (and (>= priority 3) (< confidence 90))
      (== approved 1)))