(define (problem p1) (:domain d_optic)
(:objects 
    b - barist
    d1 d2 - drink
    w - waiter
    t1 t2 t3 t4 - table
    balcao - balcony
)

(:init
    (is-cold d1) (is-cold d2)
    (drink-todo d1) (drink-todo d2)
    (barist-available b)
    ;
    (waiter-at-b balcao) (waiter-free w) (not-holding-drink w) (not-holding-tray w)
    (=(drinks-in-tray)0)
    ;
    (needs-drink t2 d1) (needs-drink t2 d2)
    (needs-cleaning t3) (needs-cleaning t4)
    ;
    (=(table-size t1)1) (=(table-size t2)1) (=(table-size t3)2) (=(table-size t4)1)
    (=(distance-b-t balcao t1)2.0) (=(distance-t-b t1 balcao)2.0)
    (=(distance-b-t balcao t2)2.0) (=(distance-t-b t2 balcao)2.0)
    (=(distance-b-t balcao t3)3.0) (=(distance-t-b t3 balcao)3.0)
    (=(distance-b-t balcao t4)3.0) (=(distance-t-b t4 balcao)3.0)
)

(:goal (and
    (drink-served d1)
    (drink-served d2)
    (is-clean t3)
    (is-clean t4)
))
)