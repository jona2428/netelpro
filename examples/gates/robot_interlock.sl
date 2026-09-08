; Gate: robotic cell interlock (permit actuator motion).
; Contract: (filter-rule door_open speed estop) -> 1 motion permitted, 0 halt.
;   door_open : 0/1 flag -- safety door of the cell
;   speed     : Int (i64), commanded speed in mm/s
;   estop     : 0/1 flag -- emergency stop latched
; Policy (safety is the DENY side; the rule returns PERMIT):
;   PERMIT only when estop is clear AND (door closed OR speed == 0).
;   An estop latched => never permit, whatever else is true.
;   Door open with commanded motion => halt.
;
; Truth table (representative rows; full space tested in tests/test_gate_examples.py):
;   door_open speed estop | permit
;   0         100   0     | 1
;   1         100   0     | 0
;   1         0     0     | 1
;   0         100   1     | 0
;   1         100   1     | 0

(defn filter-rule (door_open speed estop)
  (and (== estop 0)
       (or (== door_open 0) (== speed 0))))