; Netelpro standard prelude -- Boolean dialect helpers.
;
; HOW TO USE (there is no `include` in Netelpro; SPEC section 8 keeps modules
; out of scope). The host CONCATENATES this text before the rule source:
;
;     source = prelude_source() + "\n" + rule_source
;     filter = RuleFilter(source)
;
; Verified properties (native JIT vs reference interpreter, differential):
;   - concatenation preserves parity;
;   - a name collision with the contract is a HARD error, never silent
;     (`duplicate defn 'x' (already defined at top level)`);
;   - an impure helper the rule never calls leaves gate purity GREEN;
;   - a `sorry` anywhere in here breaks compilation of EVERY rule that
;     concatenates it. This file must stay closed. No holes, ever.
;
; DIALECT LAW -- read before adding anything:
;   These helpers return Bool. They compose inside `defn` bodies.
;   They CANNOT be called from `truth-table` rows: a row slot is `Int 0 1`
;   and the row body must return Int, so a Bool helper is a type mismatch.
;   Exhaustiveness is decided syntactically over row PATTERNS, not by
;   evaluating helpers -- a wildcard row calling a helper covers nothing.
;   Truth-table contracts are rendered by netelpro.lib.contracts instead.
;   Do not add Int-dialect twins here: they buy nothing.

(defn all-of2 (a b)
  (and (== a 1) (== b 1)))

(defn all-of3 (a b c)
  (and (== a 1) (and (== b 1) (== c 1))))

(defn all-of4 (a b c d)
  (and (== a 1) (and (== b 1) (and (== c 1) (== d 1)))))

(defn any-zero2 (a b)
  (or (== a 0) (== b 0)))

(defn any-zero3 (a b c)
  (or (== a 0) (or (== b 0) (== c 0))))

(defn any-zero4 (a b c d)
  (or (== a 0) (or (== b 0) (or (== c 0) (== d 0)))))
