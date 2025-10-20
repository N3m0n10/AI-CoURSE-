(define (problem p2) (:domain n_waiters)
(:objects 
    b - barist
    d1 d2 d3 d4 - drink
    w1 w2 - waiter
    t1 t2 t3 t4 - table
    balcao - balcony
)

(:init
    ;drinks
    (is-hot d1) (is-hot d2)
    (is-hot d3) (is-hot d4)
    (drink-todo d1) (drink-todo d2) 
    (drink-todo d3) (drink-todo d4)
    ;barist
    (barist-available b)
    ;waiter
    (waiter-at-b w1 balcao) (waiter-free w1) (not-holding-drink w1) (not-holding-tray w1)
    (waiter-at-t w2 t2) (waiter-free w2) (not-holding-drink w2) (not-holding-tray w2)
    (= (drinks-in-tray w1)0) (= (drinks-in-tray w2)0)
    ;table
    (needs-drink t4 d1) (needs-drink t4 d2) (needs-drink t1 d3) (needs-drink t1 d4)
    (needs-cleaning t3) 
    ;
    (not-choosing-table t1) (not-choosing-table t2)
    (not-choosing-table t3) (not-choosing-table t4)
    (=(table-size t1)1.0) (=(table-size t2)1.0) (=(table-size t3)2.0) (=(table-size t4)1.0)
    (=(distance-b-t balcao t1)2.0) (=(distance-t-b t1 balcao)2.0)
    (=(distance-b-t balcao t2)2.0) (=(distance-t-b t2 balcao)2.0)
    (=(distance-b-t balcao t3)3.0) (=(distance-t-b t3 balcao)3.0)
    (=(distance-b-t balcao t4)3.0) (=(distance-t-b t4 balcao)3.0)
    (table-free t1) (table-free t2) (table-free t3) (table-free t4)
)

(:goal (and
    (drink-served d1)
    (drink-served d2)
    (drink-served d3)
    (drink-served d4)
    (is-clean t3)
))
)