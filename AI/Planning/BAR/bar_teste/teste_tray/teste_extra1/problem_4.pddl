(define (problem p4) (:domain extra_3)
(:objects 
    b - barist
    d1 d2 d3 d4 d5 d6 d7 d8 - drink
    w - waiter
    t1 t2 t3 t4 - table
    balcao - balcony
)

(:init
    (is-cold d1) (is-cold d2) (is-cold d3) (is-cold d4)
    (is-hot d5) (is-hot d6) (is-hot d7) (is-hot d8)
    (drink-todo d1) (drink-todo d2) (drink-todo d3) (drink-todo d4) 
    (drink-todo d5) (drink-todo d6) (drink-todo d7) (drink-todo d8) 
    (barist-available b)
    (waiter-at balcao) 
    (needs-drink t4 d1) (needs-drink t4 d2) (needs-drink t1 d3) (needs-drink t1 d4)
    (needs-drink t3 d5) (needs-drink t3 d6) (needs-drink t3 d7) (needs-drink t3 d8)
    (needs-cleaning t2)
    (=(client-on-table t4)2) (=(client-on-table t1)2) (=(client-on-table t3)4)
    (=(client-served t4)0) (=(client-served t1)0) (=(client-served t3)0)
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
    (drink-served d3)
    (drink-served d4)
    (drink-served d5)
    (drink-served d6)
    (drink-served d7)
    (drink-served d8)
    (table-clean t1) (table-clean t2) (table-clean t3) (table-clean t4) 
))
)