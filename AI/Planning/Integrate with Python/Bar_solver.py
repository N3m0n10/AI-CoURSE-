from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.model import Problem

# Create a PDDLReader instance
reader = PDDLReader()

# Parse a complete PDDL problem from domain and problem files
problem = reader.parse_problem(
    'Planning/BAR/bar_teste/teste_tray/hot_drinks/d_hot.pddl', 
    'Planning/BAR/bar_teste/teste_tray/hot_drinks/p_hot.pddl'
    #'Planning/BAR/bar_teste/teste_tray/domain.pddl',
    #'Planning/BAR/bar_teste/teste_tray/problem_4.pddl'
    #'Planning/BAR/bar_teste/teste_tray/versão base compatível com OPTIC/domain_optic.pddl',
    #'Planning/BAR/bar_teste/teste_tray/versão base compatível com OPTIC/problem_optic.pddl'
)

# call planner
with OneshotPlanner(problem_kind=problem.kind) as planner:       #test planners requirements
#with OneshotPlanner(name='lpg') as planner:
    result = planner.solve(problem)
    plan = result.plan
    #tamer
    
    if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
        print("✅ Plano encontrado!")
        # Imprime o plano de forma legível
        if hasattr(plan, 'timed_actions'):
            # For temporal plans
            for start_time, action, duration in plan.timed_actions:
                print(f"{start_time:6.2f}: {action} [{duration}]")
        else:
            # For sequential plans
            for i, action in enumerate(plan.actions):
                print(f"{i}: {action}")
    else:
        print(f"❌ Nenhum plano encontrado. Status: {result.status}")