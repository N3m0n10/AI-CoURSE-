(define (domain ex_3_optic)

(:requirements :strips :fluents :typing :equality :conditional-effects :duration-inequalities :action-costs :durative-actions :quantified-preconditions )
(:types drink barist waiter
    location - object
    table balcony - location
)

(:predicates 
    (barist-available ?b - barist) 

    ; drinks types
    (is-cold ?d - drink)
    (is-hot ?d - drink)
    ; drinks status
    (drink-todo ?d - drink)
    (drink-prepared ?d - drink)
    (drink-in-hand ?d - drink) 
    (drink-in-tray ?d - drink)
    (not-drink-in-tray ?d - drink)
    (drink-served ?d - drink)

    ;waiter status
    (holding-drink ?w - waiter)
    (not-holding-drink ?w - waiter)
    (holding-tray ?w - waiter)
    (not-holding-tray ?w - waiter)
    (waiter-at ?l - location)
    (waiter-free ?w - waiter)
    
    ;table status
    (needs-drink ?t - table ?d - drink)
    (needs-cleaning ?t - table)
    (table-clean ?t - table)
    (dirty-trigger ?t - table)
)

(:functions
    (distance ?from - location ?to - location)
    (table-size ?t - table)
    (drinks-in-tray)
    (client-served ?t - table)
    (client-on-table ?t - table)
)

;;;;;;;;acoes;;;;;;;;
(:durative-action table-served-validator
    :parameters (?t - table)
    :duration (= ?duration 0.01)
    :condition (
        at start (and 
        (=(client-served ?t)(client-on-table ?t))
        (dirty-trigger ?t)
        )
    )
    :effect (and 
        (at end (and 
        (needs-cleaning ?t)
        (not(dirty-trigger ?t))
        ))
    )
)


;;;;;;;;;;barista;;;;;;;;;;;;
(:durative-action prepare_cold
    :parameters (?b - barist ?d - drink)
    :duration (= ?duration 3.0)
    :condition (and 
        (at start (and 
        (is-cold ?d)
        (barist-available ?b)
        (drink-todo ?d)
        ))
    )
    :effect (and 
        (at start (and 
        (not(barist-available ?b))
        ))
        (at end (and 
        (barist-available ?b)
        (not(drink-todo ?d))
        (drink-prepared ?d)
        ))
    )
)

(:durative-action prepare_hot
    :parameters (?b - barist ?d - drink)
    :duration (= ?duration 5.0)
    :condition (and 
        (at start (and 
        (is-hot ?d)
        (barist-available ?b)
        (drink-todo ?d)
        ))
    )
    :effect (and 
        (at start (and 
        (not(barist-available ?b))
        ))
        (at end (and 
        (barist-available ?b)
        (not(drink-todo ?d))
        (drink-prepared ?d)
        ))
    )
)

;;;;;;;;waiter;;;;;;;;;;;

;;;;;;;;;;;;;move;;;;;;;;;;;;;
(:durative-action waiter_goto_no_tray_table_table  
    :parameters (?w - waiter ?from - table ?to - table)
    :duration (= ?duration 0.5) 
    :condition (and 
        (at start (and 
        (waiter-at ?from)
        (waiter-free ?w)
        ))
    )
    :effect (and 
        (at start (and 
        (not(waiter-free ?w))
        ))
        (at end (and 
        (waiter-free ?w)
        (waiter-at ?to)
        (not (waiter-at ?from))
        ))
    )
)

(:durative-action waiter_goto_no_tray_balcony_table
    :parameters (?w - waiter ?from - balcony ?to - table)
    :duration (= ?duration (* 0.5 (distance ?from ?to))) 
    :condition (and 
        (at start (and 
        (waiter-at ?from)
        (waiter-free ?w)
        (not-holding-tray ?w)
        ))
    )
    :effect (and 
        (at start (and 
        (not (waiter-free ?w))
        ))
        (at end (and 
        (waiter-free ?w)
        (waiter-at ?to)
        (not (waiter-at ?from))
        ))
    )
)

(:durative-action waiter_goto_no_tray_table_balcony
    :parameters (?w - waiter ?from - table ?to - balcony)
    :duration (= ?duration (* 0.5 (distance ?from ?to)))
    :condition (and 
        (at start (and 
        (waiter-at ?from)
        (waiter-free ?w)
        (not-holding-tray ?w)
        ))
    )
    :effect (and 
        (at start (and 
        (not (waiter-free ?w))
        ))
        (at end (and 
        (waiter-free ?w)
        (waiter-at ?to)
        (not (waiter-at ?from))
        ))
    )
)

(:durative-action waiter_goto_tray_table_table  
    :parameters (?w - waiter ?from - table ?to - table)
    :duration (= ?duration 1.0) 
    :condition (and 
        (at start (and 
        (waiter-at ?from)
        (waiter-free ?w)
        (holding-tray ?w)
        ))
    )
    :effect (and 
        (at start (and 
        (not (waiter-free ?w))
        ))
        (at end (and 
        (waiter-free ?w)
        (waiter-at ?to)
        (not (waiter-at ?from))
        ))
    )
)

(:durative-action waiter_goto_tray_balcony_table
    :parameters (?w - waiter ?from - balcony ?to - table)
    :duration (= ?duration (distance ?from ?to)) 
    :condition (and 
        (at start (and 
        (waiter-at ?from)
        (waiter-free ?w)
        (holding-tray ?w)
        ))
    )
    :effect (and 
        (at start (and 
        (not (waiter-free ?w))
        ))
        (at end (and 
        (waiter-free ?w)
        (waiter-at ?to)
        (not (waiter-at ?from))
        ))
    )
)

(:durative-action waiter_goto_tray_table_balcony
    :parameters (?w - waiter ?from - table ?to - balcony)
    :duration (= ?duration (distance ?from ?to))
    :condition (and 
        (at start (and 
        (waiter-at ?from)
        (waiter-free ?w)
        (holding-tray ?w)
        ))
    )
    :effect (and 
        (at start (and 
        (not (waiter-free ?w))
        ))
        (at end (and 
        (waiter-free ?w)
        (waiter-at ?to)
        (not (waiter-at ?from))
        ))
    )
)

;;;;;;;;no tray;;;;;;;;;;
(:durative-action hold_drink  
    :parameters (?w - waiter ?d - drink ?l - balcony) 
    :duration (= ?duration 0.1)
    :condition (and 
        (at start (and 
        (not-drink-in-tray ?d)
        (waiter-at ?l)
        (waiter-free ?w)
        (drink-prepared ?d)
        (not-holding-drink ?w)
        (not-holding-tray ?w)
        ;;;at balcony
        ))
    )
    :effect (and 
        (at start (not (waiter-free ?w)))
        (at end (and 
        (not (drink-prepared ?d))
        (drink-in-hand ?d)
        (holding-drink ?w)
        (not(not-holding-drink ?w))
        (waiter-free ?w)
        ))
    )
)

(:durative-action serve_drink  
    :parameters (?w - waiter ?d - drink ?t - table)
    :duration (= ?duration 4.0)  ;;;ponto principal
    :condition (and 
        (at start (and 
        (waiter-at ?t)
        (needs-drink ?t ?d)
        (waiter-free ?w)
        (drink-in-hand ?d)
        (holding-drink ?w)  
        ))
    )
    :effect (and 
        (at end (and 
        (not (drink-in-hand ?d))
        (not (holding-drink ?w))
        (not-holding-drink ?w)
        (drink-served ?d)
        (not (needs-drink ?t ?d))
        (increase (client-served ?t) 1)
        ))
    )
)

(:durative-action clean_table
    :parameters (?w - waiter ?t - table)
    :duration (= ?duration (* 2(table-size ?t)))
    :condition (and 
        (at start (and 
        (waiter-at ?t)
        (needs-cleaning ?t)
        (waiter-free ?w)
        (not-holding-drink ?w)
        (not-holding-tray ?w)
        ))
    )
    :effect (and 
        (at start 
        (not(waiter-free ?w))
        )
        (at end (and 
        (waiter-free ?w)
        (not(needs-cleaning ?t))
        (table-clean ?t)
        ))
    )
)

;;;;;;;;;;;with tray;;;;;;;;;;;
(:durative-action add_to_tray  
    :parameters (?w - waiter ?d - drink ?l - balcony) 
    :duration (= ?duration 0.1) 
    :condition (and 
        (at start (and 
        (<(drinks-in-tray)3)
        (not-drink-in-tray ?d)
        (waiter-at ?l)
        (waiter-free ?w)
        (drink-prepared ?d)
        (not-holding-drink ?w)
        (not-holding-tray ?w)
        ))
    )
    :effect (and 
        (at start 
        (not(waiter-free ?w))
        )
        (at end (and 
        (not (drink-prepared ?d))
        (drink-in-tray ?d)
        (not(not-drink-in-tray ?d))
        (increase(drinks-in-tray)1)
        (waiter-free ?w)
        ))
    )
)


(:durative-action hold_tray  
    :parameters (?w - waiter ?l - balcony) 
    :duration (= ?duration 0.1)
    :condition (and 
        (at start (and 
        (waiter-at ?l)
        (waiter-free ?w)
        (not-holding-drink ?w)
        (not-holding-tray ?w)
        ))
    )
    :effect (and 
        (at start 
        (not(waiter-free ?w))
        )
        (at end (and 
        (not(not-holding-tray ?w))
        (waiter-free ?w)
        ))
    )
)

(:durative-action leave_tray  
    :parameters (?w - waiter ?l - balcony) 
    :duration (= ?duration 0.1)
    :condition (and 
        (at start (and 
        (waiter-at ?l)
        (waiter-free ?w)
        (not-holding-drink ?w)
        (holding-tray ?w)
        ))
    )
    :effect (and 
        (at start 
        (not(waiter-free ?w))
        )
        (at end (and 
        (not-holding-tray ?w)
        (waiter-free ?w)
        ))
    )
)

(:durative-action serve-tray  
    :parameters (?w - waiter ?t - table ?d - drink) 
    :duration (= ?duration 4.0)
    :condition (and 
        (at start (and 
        (waiter-at ?t)
        (waiter-free ?w)
        (not-holding-drink ?w)
        (holding-tray ?w)
        (drink-in-tray ?d)
        (needs-drink ?t ?d)
        ))
    )
    :effect (and 
        (at end (and 
        (not(needs-drink ?t ?d))
        (not(drink-in-tray ?d))
        (not-drink-in-tray ?d)
        (decrease (drinks-in-tray) 1)
        (increase (client-served ?t) 1)
        ))
    )
)

(:durative-action serve-tray_2   
    :parameters (?w - waiter ?t - table ?d1 - drink ?d2 - drink) 
    :duration (= ?duration 4.0)
    :condition (and 
        (at start (and 
        (waiter-at ?t)
        (waiter-free ?w)
        (not-holding-drink ?w)
        (holding-tray ?w)
        (drink-in-tray ?d1)
        (drink-in-tray ?d2)
        (needs-drink ?t ?d1)
        (needs-drink ?t ?d2)
        ))
    )
    :effect (and 
        (at end (and 
        (not(needs-drink ?t ?d1))
        (not(needs-drink ?t ?d2))
        (not(drink-in-tray ?d1))
        (not(drink-in-tray ?d2))
        (not-drink-in-tray ?d1)
        (not-drink-in-tray ?d2)
        (decrease (drinks-in-tray) 2)
        (increase (client-served ?t) 2)
        ))
    )
)

(:durative-action serve-tray_3 
    :parameters (?w - waiter ?t - table ?d1 - drink ?d2 - drink ?d3 - drink)
    :duration (= ?duration 4.0)
    :condition (and 
        (at start (and 
        (waiter-at ?t)
        (waiter-free ?w)
        (not-holding-drink ?w)
        (holding-tray ?w)
        (drink-in-tray ?d1)
        (drink-in-tray ?d2)
        (drink-in-tray ?d3)
        (needs-drink ?t ?d1)
        (needs-drink ?t ?d2)
        (needs-drink ?t ?d3)
        ))
    )
    :effect (and 
        (at end (and 
        (not(needs-drink ?t ?d1))
        (not(needs-drink ?t ?d2))
        (not(needs-drink ?t ?d3))
        (not(drink-in-tray ?d1))
        (not(drink-in-tray ?d2))
        (not(drink-in-tray ?d3))
        (not-drink-in-tray ?d1)
        (not-drink-in-tray ?d2)
        (not-drink-in-tray ?d3)
        (decrease (drinks-in-tray) 3)
        (increase (client-served ?t) 3)
        ))
    )
)
)