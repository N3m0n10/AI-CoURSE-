(define(:domain example)

(:requirements :strips :typing)

(:types object)

(:predicates
 (p ?x - object)
 (q ?y - object)
 (r ?x - object)
)

(:action ex1
 :parameters (?x - object ?y - object)
 :precondition (and (p ?x) (q ?y))
 :effect (and (r ?x) (not (p ?x)))
)
)