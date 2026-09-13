; Netelpro Neuro Gate: Action Boundary Rule for Logit / Decision Pruning
; (filter-rule action_id allowed_min allowed_max safety_state) -> 1 if action allowed, 0 if masked (-inf)
;   action_id     : Int (i64), token ID or discrete action index
;   allowed_min   : Int (i64), minimum permitted ID in current state
;   allowed_max   : Int (i64), maximum permitted ID in current state
;   safety_state  : Int (i64), 1 = normal operating mode, 0 = emergency freeze

(defn filter-rule (action_id allowed_min allowed_max safety_state)
  (and (== safety_state 1)
       (and (>= action_id allowed_min)
            (<= action_id allowed_max))))
