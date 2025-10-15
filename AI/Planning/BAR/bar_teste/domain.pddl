;Header and description

(define (domain teste)

;remove requirements that are not needed
(:requirements :strips :fluents :typing :equality :conditional-effects :negative-preconditions :duration-inequalities :action-costs :durative-actions)

(:types drink barist waiter
    location - object
    table balcony - location
)

; un-comment following line if constants are needed
;(:constants )

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
    (drink-served ?d - drink)

    ;waiter status
    (holding-drink ?w - waiter)
    (holding-tray ?w - waiter)
    (waiter-at ?l - location)
    (waiter-busy ?w - waiter)

    ;table status
    (needs-drink ?t - table ?d - drink)
    (needs-cleaning ?t - table)
)

(:functions
    ;(drinks-in-hand)
    (distance ?from - location ?to - location)
    (table-size ?t - table)
)

;;;;;;;;acoes;;;;;;;;

;;;;;;;;;;barista;;;;;;;;;;;;
(:durative-action prepare_cold
    :parameters (?b - barist ?d - drink)
    :duration (= ?duration 3)
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
    :duration (= ?duration 5)
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
    :duration (= ?duration 0.5) ;(* 0.5 (distance ?from ?to))
    :condition (and 
        (at start (and 
        (waiter-at ?from)
        (not (waiter-at ?to))
        (not (waiter-busy ?w))
        ))
    )
    :effect (and 
        (at start (and 
        (waiter-busy ?w)
        ))
        (at end (and 
        (not (waiter-busy ?w))
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
        (not (waiter-at ?to))
        (not (waiter-busy ?w))
        ))
    )
    :effect (and 
        (at start (and 
        (waiter-busy ?w)
        ))
        (at end (and 
        (not (waiter-busy ?w))
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
        (not (waiter-at ?to))
        (not (waiter-busy ?w))
        ))
    )
    :effect (and 
        (at start (and 
        (waiter-busy ?w)
        ))
        (at end (and 
        (not (waiter-busy ?w))
        (waiter-at ?to)
        (not (waiter-at ?from))
        ))
    )
)

;;;;;;;;no tray;;;;;;;;;;
(:durative-action hold_drink  ;; <-- fix  
    :parameters (?w - waiter ?d - drink ?l - balcony) ;location
    :duration (= ?duration 0.1)
    :condition (and 
        (at start (and 
        ;(=(drinks-in-hand)0)
        (not (drink-in-tray ?d))
        (waiter-at ?l)
        (not (waiter-busy ?w))
        (drink-prepared ?d)
        (not(holding-drink ?w))
        (not(holding-tray ?w))
        ;;;at balcony
        ))
    )
    :effect (and 
        (at start 
        (waiter-busy ?w))
        (at end (and 
        (not (drink-prepared ?d))
        (drink-in-hand ?d)
        (holding-drink ?w)
        (not(waiter-busy ?w))
        ))
    )
)

(:durative-action serve_drink
    :parameters (?w - waiter ?d - drink ?t - table)
    :duration (= ?duration 0.1)
    :condition (and 
        (at start (and 
        (waiter-at ?t)
        (needs-drink ?t ?d)
        (not (waiter-busy ?w))
        (drink-in-hand ?d)
        (holding-drink ?w)  ;drink-in-hand implica em holding-drink mas vou deixar por enquanto
        ))
    )
    :effect (and 
        (at start 
        (waiter-busy ?w))
        (at end (and 
        (not (drink-in-hand ?d))
        (not (holding-drink ?w))
        (drink-served ?d)
        (not (needs-drink ?t ?d))
        (not(waiter-busy ?w))
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
        (not (waiter-busy ?w))
        (not(holding-drink ?w))
        ))
    )
    :effect (and 
        (at start 
        (waiter-busy ?w))
        (at end (and 
        (not (waiter-busy ?w))
        (not(needs-cleaning ?t))
        ))
    )
)

;;;;;;;;;;;with tray;;;;;;;;;;;

)