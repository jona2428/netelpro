; Netelpro Neuro Gate: Activation Guard Rule
; (filter-rule z z_min z_max control_flag) -> 1 if activation allowed, 0 if suppressed (fail-closed).
;   z            : Int (i64), membrane potential scaled to integer
;   z_min        : Int (i64), lower stability bound
;   z_max        : Int (i64), upper stability bound
;   control_flag : Int (i64), 1 if control circuit permits firing, 0 to inhibit

(defn filter-rule (z z_min z_max control_flag)
  (and (== control_flag 1)
       (and (>= z z_min)
            (<= z z_max))))
