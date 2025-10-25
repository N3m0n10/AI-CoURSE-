(define (problem p4) (:domain teste_2)
(:objects 
    b - barist
    d1 d2 d3 d4 d5 d6 d7 d8 - drink
    w - waiter
    t1 t2 t3 t4 - table
    balcao - balcony
)

(:init
    ;drinks
    (is-cold d1) (is-cold d2) (is-cold d3) (is-cold d4)
    (is-hot d5) (is-hot d6) (is-hot d7) (is-hot d8)
    (not-drink-rejected d1) (not-drink-rejected d2) (not-drink-rejected d3) (not-drink-rejected d4)
    (not-drink-rejected d5) (not-drink-rejected d6) (not-drink-rejected d7) (not-drink-rejected d8)
    (drink-todo d1) (drink-todo d2) (drink-todo d3) (drink-todo d4) 
    (drink-todo d5) (drink-todo d6) (drink-todo d7) (drink-todo d8) 
    ;barist
    (barist-available b)
    ;waiter
    (waiter-at-b balcao) (waiter-free w) (not-holding-drink w) (not-holding-tray w)
    (=(drinks-in-tray)0)
    ;tables
    (needs-drink t4 d1) (needs-drink t4 d2) (needs-drink t1 d3) (needs-drink t1 d4)
    (needs-drink t3 d5) (needs-drink t3 d6) (needs-drink t3 d7) (needs-drink t3 d8)
    (needs-cleaning t2) 
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
    (drink-served d5)
    (drink-served d6)
    (drink-served d7)
    (drink-served d8)
    (is-clean t2)
    (cool-activated d5)
    (cool-activated d6)
    (cool-activated d7)
    (cool-activated d8)
    )
)
)