from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus
import logging
import time

# Enable logging
logging.basicConfig(level=logging.INFO)

reader = PDDLReader()
problem = reader.parse_problem('Planning/BAR/bar_teste/teste_tray/hot_drinks/d_hot.pddl', 'Planning/BAR/bar_teste/teste_tray/hot_drinks/p_hot.pddl')

print("🔍 Starting planning process...")
start_time = time.time()

# Try multiple planners in order of speed
planners_to_try = [
    ('lpg', {}),
    ('tamer', {'heuristic': 'hff', 'weight': 1.5}),
    ('enshp')
]

for planner_name, params in planners_to_try:
    print(f"\n🚀 Trying {planner_name}...")
    try:
        with OneshotPlanner(name=planner_name, params=params) as planner:
            result = planner.solve(problem)
            plan = result.plan
            
        if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
            print("✅ Plano encontrado!")
            plan = result.plan
            
            if hasattr(plan, 'timed_actions'):
                print(f"Plano temporal com {len(plan.timed_actions)} ações:")
                print("-" * 60)
                
                for start, action, duration in plan.timed_actions:
                    # Convert Fraction to float for clean display
                    start_float = float(start)
                    dur_float = float(duration)
                    print(f"{start_float:6.1f}: {action} [{dur_float:.1f}]")
                    
            else:
                print(f"Plano sequencial com {len(plan.actions)} ações:")
                print("-" * 50)
                for i, action in enumerate(plan.actions):
                    print(f"{i+1:2d}. {action}")

        else:
            print(f"❌ {planner_name} failed: {result.status}")
            
    except Exception as e:
        print(f"⚠️ {planner_name} error: {e}")
        continue