; Gate: rate limiting / backoff for a retryable action.
; Contract: (filter-rule retries max-retries elapsed-ms cooldown-ms) -> 1 allow, 0 deny.
;   retries      : Int (i64), attempts already made for this resource (from the
;                  state store -- never from the caller's own recollection)
;   max-retries  : Int (i64), declared ceiling for this resource
;   elapsed-ms   : Int (i64), time since the first attempt (from the store)
;   cooldown-ms  : Int (i64), declared cooldown window
; Policy: allow a retry if the ceiling has not been reached yet, OR the
; cooldown window has fully elapsed (reset case).
; Boundary semantics (resolved from EPISTEMIC_GATE_SPEC.md hole #1): at
; retries == max-retries AND elapsed-ms < cooldown-ms, this DENIES -- the
; ceiling is a hard stop until the cooldown clears it, not a soft warning.
;
; Truth table (representative rows; full space tested in tests/test_state_gate.py):
;   retries max-retries elapsed-ms cooldown-ms | allow
;   0       3           0          60000       | 1
;   2       3           0          60000       | 1
;   3       3           0          60000       | 0
;   3       3           60000      60000       | 1
;   3       3           59999      60000       | 0

(defn filter-rule (retries max-retries elapsed-ms cooldown-ms)
  (or (< retries max-retries) (>= elapsed-ms cooldown-ms)))
