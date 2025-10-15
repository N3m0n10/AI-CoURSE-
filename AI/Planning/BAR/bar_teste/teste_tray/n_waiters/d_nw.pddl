;Header and description

(define (domain n_waiters)

;remove requirements that are not needed
(:requirements :strips :typing :equality :negative-preconditions :duration-inequalities :universal-preconditions :action-costs :durative-actions :quantified-preconditions)

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
    (drink-in-hand ?w - waiter ?d - drink) 
    (drink-in-tray ?w - waiter ?d - drink)
    (drink-served ?d - drink)

    ;waiter status
    (holding-drink ?w - waiter)
    (holding-tray ?w - waiter)
    (waiter-at ?w - waiter ?l - location)
    (waiter-busy ?w - waiter)
    (waiter-table ?w - waiter ?t - table)

    ;table status
    (needs-drink ?t - table ?d - drink)
    (needs-cleaning ?t - table)
    (choosing-table ?t - table)

    ;tray

)

(:functions
    ;(drinks-in-hand)
    (distance ?from - location ?to - location)
    (table-size ?t - table)
    (drinks-in-tray ?w - waiter)
)

;;;;;;;;acoes;;;;;;;;
(:durative-action waiter-selector
    :parameters (?w - waiter ?t - table)
    :duration (= ?duration 0.01)
    :condition (and 
        (at start (and 
        (not (exists (?w2 - waiter)
            (waiter-table ?w2 ?t))) 
        (not (choosing-table ?t)) ;lock
        ))
    )
    :effect (and 
        (at start (choosing-table ?t))
        (at end (and 
        (waiter-table ?w ?t)
        (not (choosing-table ?t))
        ))
    )
)


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
    :duration (= ?duration 0.5)
    :condition (and 
        (at start (and 
        (waiter-at ?w ?from)
        (not (waiter-at ?w ?to))
        (not (waiter-busy ?w))
        (not (holding-tray ?w))
        ))
    )
    :effect (and 
        (at start (and 
        (waiter-busy ?w)
        ))
        (at end (and 
        (not (waiter-busy ?w))
        (waiter-at ?w ?to)
        (not (waiter-at ?w ?from))
        ))
    )
)

(:durative-action waiter_goto_no_tray_balcony_table
    :parameters (?w - waiter ?from - balcony ?to - table)
    :duration (= ?duration (* 0.5 (distance ?from ?to))) 
    :condition (and 
        (at start (and 
        (waiter-at ?w ?from)
        (not (waiter-at ?w ?to))
        (not (waiter-busy ?w))
        (not (holding-tray ?w))
        ))
    )
    :effect (and 
        (at start (and 
        (waiter-busy ?w)
        ))
        (at end (and 
        (not (waiter-busy ?w))
        (waiter-at ?w ?to)
        (not (waiter-at ?w ?from))
        ))
    )
)

(:durative-action waiter_goto_no_tray_table_balcony
    :parameters (?w - waiter ?from - table ?to - balcony)
    :duration (= ?duration (* 0.5 (distance ?from ?to)))
    :condition (and 
        (at start (and 
        (waiter-at ?w ?from)
        (not (waiter-at ?w ?to))
        (not (waiter-busy ?w))
        (not (holding-tray ?w))
        (not (exists (?w2 - waiter)
            (and (not (= ?w ?w2))
                (waiter-at ?w2 ?to))));
        ))
    )
    :effect (and 
        (at start (and 
        (waiter-busy ?w)
        ))
        (at end (and 
        (not (waiter-busy ?w))
        (waiter-at ?w ?to)
        (not (waiter-at ?w ?from))
        ))
    )
)

(:durative-action waiter_goto_tray_table_table  
    :parameters (?w - waiter ?from - table ?to - table)
    :duration (= ?duration 1) 
    :condition (and 
        (at start (and 
        (waiter-at ?w ?from)
        (not (waiter-at ?w ?to))
        (not (waiter-busy ?w))
        (holding-tray ?w)
        ))
    )
    :effect (and 
        (at start (and 
        (waiter-busy ?w)
        ))
        (at end (and 
        (not (waiter-busy ?w))
        (waiter-at ?w ?to)
        (not (waiter-at ?w ?from))
        ))
    )
)

(:durative-action waiter_goto_tray_balcony_table
    :parameters (?w - waiter ?from - balcony ?to - table)
    :duration (= ?duration (distance ?from ?to)) 
    :condition (and 
        (at start (and 
        (waiter-at ?w ?from)
        (not (waiter-at ?w ?to))
        (not (waiter-busy ?w))
        (holding-tray ?w)
        ))
    )
    :effect (and 
        (at start (and 
        (waiter-busy ?w)
        ))
        (at end (and 
        (not (waiter-busy ?w))
        (waiter-at ?w ?to)
        (not (waiter-at ?w ?from))
        ))
    )
)

(:durative-action waiter_goto_tray_table_balcony
    :parameters (?w - waiter ?from - table ?to - balcony)
    :duration (= ?duration (distance ?from ?to))
    :condition (and 
        (at start (and 
        (waiter-at ?w ?from)
        (not (waiter-at ?w ?to))
        (not (waiter-busy ?w))
        (holding-tray ?w)
        (not (exists (?w2 - waiter)
            (and (not (= ?w ?w2))
                (waiter-at ?w2 ?to))));
        ))
    )
    :effect (and 
        (at start (and 
        (waiter-busy ?w)
        ))
        (at end (and 
        (not (waiter-busy ?w))
        (waiter-at ?w ?to)
        (not (waiter-at ?w ?from))
        ))
    )
)

;;;;;;;;no tray;;;;;;;;;;
(:durative-action hold_drink  ;; <-- fix  
    :parameters (?w - waiter ?d - drink ?l - balcony) ;location
    :duration (= ?duration 0.1)
    :condition (and 
        (at start (and 
        (not (drink-in-tray ?w ?d))
        (waiter-at ?w ?l)
        (not (waiter-busy ?w))
        (drink-prepared ?d)
        (not(holding-drink ?w))
        (not(holding-tray ?w))
        ;;;at balcony
        ))
    )
    :effect (and 
        (at start (waiter-busy ?w))
        (at end (and 
        (not (drink-prepared ?d))
        (drink-in-hand ?w ?d)
        (holding-drink ?w)
        (not (waiter-busy ?w))
        ))
    )
)

(:durative-action serve_drink
    :parameters (?w - waiter ?d - drink ?t - table)
    :duration (= ?duration 0.1)
    :condition (and 
        (at start (and 
        (waiter-at ?w ?t)
        (needs-drink ?t ?d)
        (not (waiter-busy ?w))
        (drink-in-hand ?w ?d)
        (holding-drink ?w)  
        (waiter-table ?w ?t) ;
        ))
    )
    :effect (and 
        (at start(waiter-busy ?w))
        (at end (and 
        (not (drink-in-hand ?w ?d))
        (not (holding-drink ?w))
        (drink-served ?d)
        (not (needs-drink ?t ?d))
        (not (waiter-busy ?w))
        ))
    )
)

(:durative-action clean_table
    :parameters (?w - waiter ?t - table)
    :duration (= ?duration (* 2(table-size ?t)))
    :condition (and 
        (at start (and 
        (waiter-at ?w ?t)
        (needs-cleaning ?t)
        (not(waiter-busy ?w))
        (not(holding-drink ?w))
        (not(holding-tray ?w))
        (waiter-table ?w ?t) ;
        ))
    )
    :effect (and 
        (at start 
        (waiter-busy ?w))
        (at end (and 
        (not(waiter-busy ?w))
        (not(needs-cleaning ?t))
        ))
    )
)

;;;;;;;;;;;with tray;;;;;;;;;;;
(:durative-action add_to_tray 
    :parameters (?w - waiter ?d - drink ?l - balcony) ;location
    :duration (= ?duration 0.1) 
    :condition (and 
        (at start (and 
        (<(drinks-in-tray ?w)3)
        (not (drink-in-tray ?w ?d))
        (waiter-at ?w ?l)
        (not (waiter-busy ?w))
        (drink-prepared ?d)
        (not(holding-drink ?w))
        (not(holding-tray ?w))
        ))
    )
    :effect (and 
        (at start 
        (waiter-busy ?w))
        (at end (and 
        (not (drink-prepared ?d))
        (drink-in-tray ?w ?d)
        (increase(drinks-in-tray ?w)1)
        (not(waiter-busy ?w))
        ))
    )
)


(:durative-action hold_tray  ;; <-- fix  
    :parameters (?w - waiter ?l - balcony) ;location
    :duration (= ?duration 0.1)
    :condition (and 
        (at start (and 
        (waiter-at ?w ?l)
        (not(waiter-busy ?w))
        (not(holding-drink ?w))
        (not(holding-tray ?w))
        ))
    )
    :effect (and 
        (at start 
        (waiter-busy ?w))
        (at end (and 
        (holding-tray ?w)
        (not(waiter-busy ?w))
        ))
    )
)

(:durative-action leave_tray  
    :parameters (?w - waiter ?l - balcony) ;location
    :duration (= ?duration 0.1)
    :condition (and 
        (at start (and 
        (waiter-at ?w ?l)
        (not(waiter-busy ?w))
        (not(holding-drink ?w))
        (holding-tray ?w)
        ))
    )
    :effect (and 
        (at start 
        (waiter-busy ?w))
        (at end (and 
        (not(holding-tray ?w))
        (not(waiter-busy ?w))
        ))
    )
)

(:durative-action serve-tray  
    :parameters (?w - waiter ?t - table ?d - drink) ;location
    :duration (= ?duration 4)
    :condition (and 
        (at start (and 
        (waiter-at ?w ?t)
        (not(waiter-busy ?w))
        (not(holding-drink ?w))
        (holding-tray ?w)
        (drink-in-tray ?w ?d)
        (needs-drink ?t ?d)
        (waiter-table ?w ?t) ;
        ))
    )
    :effect (and 
        (at start 
        (waiter-busy ?w))
        (at end (and 
        (not(waiter-busy ?w))
        (not(needs-drink ?t ?d))
        (not(drink-in-tray ?w ?d))
        (decrease (drinks-in-tray ?w) 1)
        ))
    )
)

(:durative-action serve-tray_2   
    :parameters (?w - waiter ?t - table ?d1 - drink ?d2 - drink) ;location
    :duration (= ?duration 4)
    :condition (and 
        (at start (and 
        (waiter-at ?w ?t)
        (not(waiter-busy ?w))
        (not(holding-drink ?w))
        (holding-tray ?w)
        (drink-in-tray ?w ?d1)
        (drink-in-tray ?w ?d2)
        (needs-drink ?t ?d1)
        (needs-drink ?t ?d2)
        (waiter-table ?w ?t) ;
        ))
    )
    :effect (and 
        (at start 
        (waiter-busy ?w))
        (at end (and 
        (not(waiter-busy ?w))
        (not(needs-drink ?t ?d1))
        (not(needs-drink ?t ?d2))
        (not(drink-in-tray ?w ?d1))
        (not(drink-in-tray ?w ?d2))
        (decrease (drinks-in-tray ?w) 2)
        ))
    )
)

(:durative-action serve-tray_3 
    :parameters (?w - waiter ?t - table ?d1 - drink ?d2 - drink ?d3 - drink) ;location
    :duration (= ?duration 4)
    :condition (and 
        (at start (and 
        (waiter-at ?w ?t)
        (not(waiter-busy ?w))
        (not(holding-drink ?w))
        (holding-tray ?w)
        (drink-in-tray ?w ?d1)
        (drink-in-tray ?w ?d2)
        (drink-in-tray ?w ?d3)
        (needs-drink ?t ?d1)
        (needs-drink ?t ?d2)
        (needs-drink ?t ?d3)
        (waiter-table ?w ?t) ;
        ))
    )
    :effect (and 
        (at start 
        (waiter-busy ?w))
        (at end (and 
        (not(waiter-busy ?w))
        (not(needs-drink ?t ?d1))
        (not(needs-drink ?t ?d2))
        (not(needs-drink ?t ?d3))
        (not(drink-in-tray ?w ?d1))
        (not(drink-in-tray ?w ?d2))
        (not(drink-in-tray ?w ?d3))
        (decrease (drinks-in-tray ?w) 3)
        ))
    )
)

)