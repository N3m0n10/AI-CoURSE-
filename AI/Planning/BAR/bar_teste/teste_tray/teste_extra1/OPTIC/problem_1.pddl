(define (problem p1) (:domain ex_3_optic)
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
    (not-drink-in-tray d1) (not-drink-in-tray d2)
    (barist-available b)
    (waiter-at balcao) (waiter-free w) (not-holding-tray w) (not-holding-drink w) 
    (needs-drink t2 d1) (needs-drink t2 d2)
    (needs-cleaning t3) (needs-cleaning t4)
    (=(client-on-table t2)2)
    (=(client-served t2)0)
    (=(client-on-table t1)0)
    (=(client-served t1)0)
    (=(client-on-table t3)0)
    (=(client-served t3)0)
    (=(client-on-table t4)0)
    (=(client-served t4)0)
    (=(drinks-in-tray)0)
    (=(table-size t1)1.0) (=(table-size t2)1.0) (=(table-size t3)2.0) (=(table-size t4)1.0)
    (=(distance balcao t1)2) (=(distance t1 balcao)2)
    (=(distance balcao t2)2) (=(distance t2 balcao)2)
    (=(distance balcao t3)3) (=(distance t3 balcao)3)
    (=(distance balcao t4)3) (=(distance t4 balcao)3)
    (dirty-trigger t2) (dirty-trigger t3) (dirty-trigger t4)
)

(:goal (and
    (drink-served d1)
    (drink-served d2)
    (table-clean t3)
    (table-clean t4)
    (table-clean t2)
))
)