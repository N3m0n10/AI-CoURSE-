(define (domain bar)
(:requirements :strips :typing :action-costs :fluents :equality :numeric-fluents)
(:types barist waiter tray drink 
location - object
table balcony - location) ;barist will be doing nothing now (adding more also do nothing)

(:predicates
    (is-hot ?d - drink)    ; indicates if drink is hot
    (is-cold ?d - drink)   ; indicates if drink is cold
    (waiter-position ?m - waiter ?l - location)  ; indicates the position of the waiter
    (drink-prepared ?d - drink) ; indicates if a drink is prepared
    (drink-todo ?d - drink)    ; initial state
    ;(drink-in-preparation ?d - drink) ; being prepared  ---for the parallel approach REMOVE LATER AAAAAAAAAA
    (drink-served ?d - drink)       ; final state
    (table-is-clean ?t - table)  ; indicates if a table is clean
    (hot_on_tray ?d - drink ?tr - tray) ; indicates if a hot drink is on the tray
    (cold_on_tray ?d - drink ?tr - tray) ; indicates if a cold drink is on tray
)

(:functions  
    (drink-state ?d - drink)  ; 0=available, 1=preparing, 2=prepared, 3=served
    (table-hots ?t - table)   ; n hot drinks to be served
    (table-colds ?t - table)  ; n cold drinks to be served
    (total-cost)
    (count-colds-on-tray)
    (count-hots-on-tray)
)


(:action prepare_cold
    :parameters (?d - drink)
    :precondition (and
        (drink-todo ?d)
        (is-cold ?d)
    )
    :effect(and
        (increase (total-cost) 3)
        (not (drink-todo ?d))
        (drink-prepared ?d)
    )
)

(:action add_to_tray_cold
    :parameters (?d - drink ?tr - tray ?l - balcony ?m - waiter)
    :precondition (and
        (drink-prepared ?d)
        (is-cold ?d)
        (waiter-position ?m ?l)  
        (<= (+ (count-colds-on-tray) (count-hots-on-tray)) 2) ; Tray can hold max 2 drinks
    )
    :effect(and
        (increase (total-cost) 1)
        (cold_on_tray ?d ?tr)   
        (not (drink-prepared ?d))
        (increase (count-colds-on-tray) 1)
)

(:action add_to_tray_hot
    :parameters (?d - drink ?tr - tray ?l - location ?m - waiter)
    :precondition (and
        (drink-prepared ?d)
        (is-cold ?d)
        (=(waiter-position ?m ?l) balcony)  ; Changed to check if waiter is at balcony
        (<= (sum (hot_on_tray ?d ?tr) (cold_on_tray ?d ?tr)) 2) ; Tray can hold max 3 drinks
    )
    :effect(and
        (increase (total-cost) 1))
        (increase (hot_on_tray ?d ?tr) 1)   
)

(:action waiter_goto
    
)

(:action serve_drink
    
)

(:action clean_table
    
)

)
)