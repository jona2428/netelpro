; Gate: content moderation (publish approval for a user-generated item).
; Contract: (filter-rule toxicity reports age_verified) -> 1 publish, 0 hold.
;   toxicity     : Int (i64) 0..100, classifier score of the item
;   reports      : Int (i64), user reports on the item
;   age_verified : 0/1 flag -- reviewer has verified the author's age gating
; Policy:
;   ALLOW (publish) when toxicity < 70 AND (no reports OR age verified).
;   Any single failure denies: high toxicity, or reported content without
;   age verification. Fail-closed lives in the host (netelpro.gate):
;   a rule that cannot decide never publishes.
;
; Truth table (representative rows; full space tested in tests/test_gate_examples.py):
;   toxicity reports age_verified | allow
;   69       0       0            | 1
;   70       0       1            | 0
;   10       3       0            | 0
;   10       3       1            | 1
;   69       0       1            | 1

(defn filter-rule (toxicity reports age_verified)
  (and (< toxicity 70)
       (or (== reports 0) (== age_verified 1))))