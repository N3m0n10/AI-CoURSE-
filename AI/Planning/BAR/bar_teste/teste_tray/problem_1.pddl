(define (problem p1) (:domain BAR)
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
    (waiter-at balcao) 
    (needs-drink t2 d1) (needs-drink t2 d2)
    (needs-cleaning t3) (needs-cleaning t4)
    (=(drinks-in-tray)0)
    (=(table-size t1)1) (=(table-size t2)1) (=(table-size t3)2) (=(table-size t4)1)
    (=(distance balcao t1)2) (=(distance t1 balcao)2)
    (=(distance balcao t2)2) (=(distance t2 balcao)2)
    (=(distance balcao t3)3) (=(distance t3 balcao)3)
    (=(distance balcao t4)3) (=(distance t4 balcao)3)
)

(:goal (and
    (drink-served d1)
    (drink-served d2)
    (not(needs-cleaning t3))
    (not(needs-cleaning t4))
))

)