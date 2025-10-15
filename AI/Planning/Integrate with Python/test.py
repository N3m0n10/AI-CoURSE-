from unified_planning.shortcuts import *

# List all available planners
env = get_environment()
print("Available planners:")
for planner_name in env.factory.engines:
    print(f"  - {planner_name}")