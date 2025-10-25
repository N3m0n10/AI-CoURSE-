(define (problem p3) (:domain teste_2)
(:objects 
    b - barist
    d1 d2 d3 d4 - drink
    w - waiter
    t1 t2 t3 t4 - table
    balcao - balcony
)

(:init
    ;drinks
    (is-hot d1) (is-hot d2)
    (is-hot d3) (is-hot d4)
    (drink-todo d1) (drink-todo d2) 
    (drink-todo d3) (drink-todo d4)
    (not-drink-rejected d1) (not-drink-rejected d2) (not-drink-rejected d3) (not-drink-rejected d4)
    ;barist
    (barist-available b)
    ;waiter
    (waiter-at-b balcao) (waiter-free w) (not-holding-drink w) (not-holding-tray w)
    (=(drinks-in-tray)0)
    ;tables
    (needs-drink t4 d1) (needs-drink t4 d2) (needs-drink t1 d3) (needs-drink t1 d4)
    (needs-cleaning t3) 
    (=(table-size t1)1.0) (=(table-size t2)1.0) (=(table-size t3)2.0) (=(table-size t4)1.0)
    (=(distance-b-t balcao t1)2.0) (=(distance-t-b t1 balcao)2.0)
    (=(distance-b-t balcao t2)2.0) (=(distance-t-b t2 balcao)2.0)
    (=(distance-b-t balcao t3)3.0) (=(distance-t-b t3 balcao)3.0)
    (=(distance-b-t balcao t4)3.0) (=(distance-t-b t4 balcao)3.0)
)

(:goal (and
    (drink-served d1)
    (drink-served d2)
    (drink-served d3)
    (drink-served d4)
    (is-clean t3)
    (cool-activated d1)
    (cool-activated d2)
    (cool-activated d3)
    (cool-activated d4)
    )
)
)