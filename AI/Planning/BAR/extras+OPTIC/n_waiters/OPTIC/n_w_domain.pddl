(define (domain n_waiters)
(:requirements :strips :typing :equality :durative-actions :fluents)
(:types drink barist waiter table balcony)

(:predicates 
    (barist-available ?b - barist) 

    ; drinks types
    (is-cold ?d - drink)
    (is-hot ?d - drink)
    ; drinks status
    (drink-todo ?d - drink)
    (drink-prepared ?d - drink)
    (drink-in-hand ?w - waiter ?d - drink) 
    (drink-in-tray ?d - drink)
    (drink-served ?d - drink)

    ; waiter status flags (positive predicates only)
    (holding-drink ?w - waiter)
    (holding-tray ?w - waiter)
    (not-holding-tray ?w - waiter)
    (not-holding-drink ?w - waiter)
    (waiter-at-t ?w - waiter ?l - table)
    (waiter-at-b ?w - waiter ?l - balcony)
    (waiter-free ?w - waiter)
    (waiter-table ?w - waiter ?t - table)

    ; table status flags
    (needs-drink ?t - table ?d - drink)
    (needs-cleaning ?t - table)
    (not-choosing-table ?t - table)  
    (is-clean ?t - table)
    (table-free ?t - table)
)

(:functions
    (distance-t-b ?from - table ?to - balcony)
    (distance-b-t ?from - balcony ?to - table)
    (table-size ?t - table)
    (drinks-in-tray ?w - waiter)
)

;;;;;;;;;;;;;;;; actions ;;;;;;;;;;;;;;;

(:durative-action waiter-selector
    :parameters (?w - waiter ?t - table)
    :duration (= ?duration 0.01)
    :condition (and
        (at start (table-free ?t))
        (at start (not-choosing-table ?t))
    )
    :effect (and
        (at start (not (not-choosing-table ?t)))
        (at end (and
            (waiter-table ?w ?t)
            (not (table-free ?t))
            (not-choosing-table ?t)  ; libera o lock choosing
        ))
    )
)

;;;;;;;;;; barista ;;;;;;;;;;
(:durative-action prepare_cold
    :parameters (?b - barist ?d - drink)
    :duration (= ?duration 3)
    :condition (and
        (at start (is-cold ?d))
        (at start (barist-available ?b))
        (at start (drink-todo ?d))
    )
    :effect (and
        (at start (not (barist-available ?b)))
        (at end (and
            (barist-available ?b)
            (not (drink-todo ?d))
            (drink-prepared ?d)
        ))
    )
)

(:durative-action prepare_hot
    :parameters (?b - barist ?d - drink)
    :duration (= ?duration 5)
    :condition (and
        (at start (is-hot ?d))
        (at start (barist-available ?b))
        (at start (drink-todo ?d))
    )
    :effect (and
        (at start (not (barist-available ?b)))
        (at end (and
            (barist-available ?b)
            (not (drink-todo ?d))
            (drink-prepared ?d)
        ))
    )
)

;;;;;;;; waiter movements ;;;;;;;;

(:durative-action waiter_goto_no_tray_table_table
    :parameters (?w - waiter ?from - table ?to - table)
    :duration (= ?duration 0.5)
    :condition (and
        (at start (waiter-at-t ?w ?from))
        (at start (waiter-free ?w))
        (at start (not-holding-tray ?w))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (waiter-free ?w)
            (waiter-at-t ?w ?to)
            (not (waiter-at-t ?w ?from))
        ))
    )
)

(:durative-action waiter_goto_no_tray_balcony_table
    :parameters (?w - waiter ?from - balcony ?to - table)
    :duration (= ?duration (* 0.5 (distance-b-t ?from ?to)))
    :condition (and
        (at start (waiter-at-b ?w ?from))
        (at start (waiter-free ?w))
        (at start (not-holding-tray ?w))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (waiter-free ?w)
            (waiter-at-t ?w ?to)
            (not (waiter-at-b ?w ?from))
        ))
    )
)

(:durative-action waiter_goto_no_tray_table_balcony
    :parameters (?w - waiter ?from - table ?to - balcony)
    :duration (= ?duration (* 0.5 (distance-t-b ?from ?to)))
    :condition (and
        (at start (waiter-at-t ?w ?from))
        (at start (waiter-free ?w))
        (at start (not-holding-tray ?w))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (waiter-free ?w)
            (waiter-at-b ?w ?to)
            (not (waiter-at-t ?w ?from))
        ))
    )
)

(:durative-action waiter_goto_tray_table_table
    :parameters (?w - waiter ?from - table ?to - table)
    :duration (= ?duration 1)
    :condition (and
        (at start (waiter-at-t ?w ?from))
        (at start (waiter-free ?w))
        (at start (holding-tray ?w))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (waiter-free ?w)
            (waiter-at-t ?w ?to)
            (not (waiter-at-t ?w ?from))
        ))
    )
)

(:durative-action waiter_goto_tray_balcony_table
    :parameters (?w - waiter ?from - balcony ?to - table)
    :duration (= ?duration (distance-b-t ?from ?to))
    :condition (and
        (at start (waiter-at-b ?w ?from))
        (at start (waiter-free ?w))
        (at start (holding-tray ?w))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (waiter-free ?w)
            (waiter-at-t ?w ?to)
            (not (waiter-at-b ?w ?from))
        ))
    )
)

(:durative-action waiter_goto_tray_table_balcony
    :parameters (?w - waiter ?from - table ?to - balcony)
    :duration (= ?duration (distance-t-b ?from ?to))
    :condition (and
        (at start (waiter-at-t ?w ?from))
        (at start (waiter-free ?w))
        (at start (holding-tray ?w))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (waiter-free ?w)
            (waiter-at-b ?w ?to)
            (not (waiter-at-t ?w ?from))
        ))
    )
)

;;;;;;;; no tray actions ;;;;;;;;

(:durative-action hold_drink
    :parameters (?w - waiter ?d - drink ?l - balcony)
    :duration (= ?duration 0.1)
    :condition (and
        (at start (waiter-at-b ?w ?l))
        (at start (waiter-free ?w))
        (at start (drink-prepared ?d))
        (at start (not-holding-drink ?w))
        (at start (not-holding-tray ?w))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (not (drink-prepared ?d))
            (drink-in-hand ?w ?d)
            (holding-drink ?w)
            (not (not-holding-drink ?w))
            (waiter-free ?w)
        ))
    )
)

(:durative-action serve_drink
    :parameters (?w - waiter ?d - drink ?t - table)
    :duration (= ?duration 0.1)
    :condition (and
        (at start (waiter-at-t ?w ?t))
        (at start (needs-drink ?t ?d))
        (at start (waiter-free ?w))
        (at start (drink-in-hand ?w ?d))
        (at start (holding-drink ?w))
        (at start (waiter-table ?w ?t))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (not (drink-in-hand ?w ?d))
            (not (holding-drink ?w))
            (not-holding-drink ?w)
            (drink-served ?d)
            (not (needs-drink ?t ?d))
            (waiter-free ?w)
        ))
    )
)

(:durative-action clean_table
    :parameters (?w - waiter ?t - table)
    :duration (= ?duration (* 2 (table-size ?t)))
    :condition (and
        (at start (waiter-at-t ?w ?t))
        (at start (needs-cleaning ?t))
        (at start (waiter-free ?w))
        (at start (not-holding-drink ?w))
        (at start (not-holding-tray ?w))
        (at start (waiter-table ?w ?t))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (not (needs-cleaning ?t))
            (is-clean ?t)
            (waiter-free ?w)
        ))
    )
)

;;;;;;;; with tray ;;;;;;;;;;;

(:durative-action add_to_tray
    :parameters (?w - waiter ?d - drink ?l - balcony)
    :duration (= ?duration 0.1)
    :condition (and
        (at start (< (drinks-in-tray ?w) 3))
        (at start (waiter-at-b ?w ?l))
        (at start (waiter-free ?w))
        (at start (drink-prepared ?d))
        (at start (not-holding-drink ?w))
        (at start (not-holding-tray ?w))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (not (drink-prepared ?d))
            (drink-in-tray ?d)
            (increase (drinks-in-tray ?w) 1)
            (waiter-free ?w)
        ))
    )
)

(:durative-action hold_tray
    :parameters (?w - waiter ?l - balcony)
    :duration (= ?duration 0.1)
    :condition (and
        (at start (waiter-at-b ?w ?l))
        (at start (waiter-free ?w))
        (at start (not-holding-drink ?w))
        (at start (not-holding-tray ?w))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (holding-tray ?w)
            (waiter-free ?w)
        ))
    )
)

(:durative-action leave_tray
    :parameters (?w - waiter ?l - balcony)
    :duration (= ?duration 0.1)
    :condition (and
        (at start (waiter-at-b ?w ?l))
        (at start (waiter-free ?w))
        (at start (holding-tray ?w))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (not (holding-tray ?w))
            (waiter-free ?w)
        ))
    )
)

(:durative-action serve-tray
    :parameters (?w - waiter ?t - table ?d - drink)
    :duration (= ?duration 0.1)
    :condition (and
        (at start (waiter-at-t ?w ?t))
        (at start (waiter-free ?w))
        (at start (not-holding-drink ?w))
        (at start (holding-tray ?w))
        (at start (drink-in-tray ?d))
        (at start (needs-drink ?t ?d))
        (at start (waiter-table ?w ?t))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (not (needs-drink ?t ?d))
            (not (drink-in-tray ?d))
            (decrease (drinks-in-tray ?w) 1)
            (waiter-free ?w)
        ))
    )
)

(:durative-action serve-tray_2
    :parameters (?w - waiter ?t - table ?d1 - drink ?d2 - drink)
    :duration (= ?duration 0.1)
    :condition (and
        (at start (waiter-at-t ?w ?t))
        (at start (waiter-free ?w))
        (at start (not-holding-drink ?w))
        (at start (holding-tray ?w))
        (at start (drink-in-tray ?d1))
        (at start (drink-in-tray ?d2))
        (at start (needs-drink ?t ?d1))
        (at start (needs-drink ?t ?d2))
        (at start (waiter-table ?w ?t))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (not (needs-drink ?t ?d1))
            (not (needs-drink ?t ?d2))
            (not (drink-in-tray ?d1))
            (not (drink-in-tray ?d2))
            (decrease (drinks-in-tray ?w) 2)
            (waiter-free ?w)
        ))
    )
)

(:durative-action serve-tray_3
    :parameters (?w - waiter ?t - table ?d1 - drink ?d2 - drink ?d3 - drink)
    :duration (= ?duration 0.1)
    :condition (and
        (at start (waiter-at-t ?w ?t))
        (at start (waiter-free ?w))
        (at start (not-holding-drink ?w))
        (at start (holding-tray ?w))
        (at start (drink-in-tray ?d1))
        (at start (drink-in-tray ?d2))
        (at start (drink-in-tray ?d3))
        (at start (needs-drink ?t ?d1))
        (at start (needs-drink ?t ?d2))
        (at start (needs-drink ?t ?d3))
        (at start (waiter-table ?w ?t))
    )
    :effect (and
        (at start (not (waiter-free ?w)))
        (at end (and
            (not (needs-drink ?t ?d1))
            (not (needs-drink ?t ?d2))
            (not (needs-drink ?t ?d3))
            (not (drink-in-tray ?d1))
            (not (drink-in-tray ?d2))
            (not (drink-in-tray ?d3))
            (decrease (drinks-in-tray ?w) 3)
            (waiter-free ?w)
        ))
    )
)
)
