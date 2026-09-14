; Netelpro Trivium Gate: Fallacy Detector & Rhetorical Soundness Rule
; (filter-rule ad_hominem false_dichotomy straw_man premise_grounded) -> 1 if valid argument, 0 if fallacious
;
; Arguments (Int / i64):
;   ad_hominem       : 1 if attacks speaker's character/identity, 0 if addresses the thesis
;   false_dichotomy  : 1 if forces artificial binary dilemma, 0 if recognizes spectrum
;   straw_man        : 1 if caricaturizes/distorts opponent's stance, 0 if faithful representation
;   premise_grounded : 1 if premises are grounded in evidence/sound axioms, 0 if arbitrary

(defn filter-rule (ad_hominem false_dichotomy straw_man premise_grounded)
  (and (== ad_hominem 0)
       (and (== false_dichotomy 0)
            (and (== straw_man 0)
                 (== premise_grounded 1)))))
