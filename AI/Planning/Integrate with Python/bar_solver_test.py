from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus
import logging
import time

# Enable logging
logging.basicConfig(level=logging.INFO)

reader = PDDLReader()
problem = reader.parse_problem('Planning/BAR/bar_teste/teste_tray/versão base compatível com OPTIC/domain_optic.pddl',
                               'Planning/BAR/bar_teste/teste_tray/versão base compatível com OPTIC/problem_optic_p1.pddl')

print("🔍 Starting planning process...")
start_time = time.time()

# Try multiple planners in order of speed
# ...existing code...
# Use um planner temporal (optic) antes dos clássicos; ajusta conforme disponível na sua instalação
planners_to_try = [
    ('optic', {}),            # temporal planner — retorna timed_actions se o problema for durativo
    ('lpg', {}),
    ('tamer', {'heuristic': 'hff', 'weight': 1.5}),   # rápido, solução aceitável
    ('tamer', {'heuristic': 'hff', 'weight': 1.2}),  # balanceado
    ('tamer', {'heuristic': 'hff', 'weight': 1.0})  # mais próximo do ótimo
]

def analyze_timed_plan(timed_actions):
    # timed_actions: iterable de (start, action, duration)
    acts = []
    for start, action, duration in timed_actions:
        s = float(start)
        d = float(duration)
        e = s + d
        acts.append((s, e, action))
    acts.sort(key=lambda x: x[0])
    makespan = max((e for _, e, _ in acts), default=0.0)
    # detectar se existe sobreposição entre quaisquer duas ações
    has_parallel = any(acts[i][1] > acts[i+1][0] for i in range(len(acts)-1))
    return acts, makespan, has_parallel

for planner_name, params in planners_to_try:
    print(f"\n🚀 Trying {planner_name}...")
    try:
        with OneshotPlanner(name=planner_name, params=params) as planner:
            result = planner.solve(problem)
        if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
            print("✅ Plano encontrado!")
            plan = result.plan
            if hasattr(plan, 'timed_actions'):
                acts, makespan, has_parallel = analyze_timed_plan(plan.timed_actions)
                print(f"Plano temporal com {len(acts)} ações (makespan={makespan:.2f}):")
                print("-" * 60)
                for s, e, action in acts:
                    print(f"{s:6.2f} - {e:6.2f}: {action}")
                print(f"\n⏱️ Tempo de planejamento (wall-clock): {time.time() - start_time:.2f} s")
                print("🔁 O plano contém paralelismo." if has_parallel else "➡️ Plano efetivamente sequencial (sem sobreposição).")
            else:
                print(f"Plano sequencial com {len(plan.actions)} ações:")
                print("-" * 50)
                for i, action in enumerate(plan.actions):
                    print(f"{i+1:2d}. {action}")
                print("➡️ Plano sequencial retornado — use um planner temporal (ex: 'optic') e verifique se o domínio/problema têm durativas.")
        else:
            print(f"❌ {planner_name} failed: {result.status}")
    except Exception as e:
        print(f"⚠️ {planner_name} error: {e}")
        continue
# ...existing code...