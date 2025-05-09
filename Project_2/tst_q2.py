import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy

# Grid world constants
GRID_ROWS = 4
GRID_COLS = 12
START_STATE = (3, 0)  # S1 at bottom left
GOAL_STATE = (3, 11)  # G at bottom right
CLIFF_ROW = 3  # The cliff is on the bottom row

# Learning parameters
ALPHA = 0.3
EPSILON = 0.000001
GAMMA = 0.9

generic_actions = ["left","right", "up", "down"] 

def create_grid(rows=GRID_ROWS, cols=GRID_COLS) -> Dict:
    """Creates grid with S1 at bottom left"""
    S = {}
    state_id = 1  # Starting from S1
    for i in range(rows):
        for j in range(cols):
            S[state_id] = (i, j)
            state_id += 1
    return S

def is_cliff_state(state: Tuple) -> bool:
    """Check if state is in the cliff region"""
    row, col = state
    return row == CLIFF_ROW and 0 < col < GRID_COLS - 1

def get_reward(state: Tuple) -> float:
    """Define rewards based on state"""
    if is_cliff_state(state):
        return -100
    elif state == GOAL_STATE:
        return 1
    else:
        return -1

def initialize_values(actions: List[str], pos: Tuple) -> List[float]:
    """Initialize Q-values for each state-action pair"""
    state_weights = []
    for _ in actions:
        state_weights.append(0) ## Sem conhecimento do entorno
    return state_weights

def get_next_state(state: Tuple, action: str) -> Tuple:
    """Get next state based on action"""
    row, col = state
    if action == "left":
        return (row, max(0, col - 1))
    elif action == "right":
        return (row, min(GRID_COLS - 1, col + 1))
    elif action == "up":
        return (max(0, row - 1), col)
    elif action == "down":
        return (min(GRID_ROWS - 1, row + 1), col)
    return state

def possible_acts(state: Tuple) -> List[str]:
    """Get valid actions for current state"""
    row, col = state
    actions = []
    if col > 0:
        actions.append("left")
    if col < GRID_COLS - 1:
        actions.append("right")
    if row > 0:
        actions.append("up")
    if row < GRID_ROWS - 1:
        actions.append("down")
    return actions

def create_q_table(states,grid_dim):
    q_table = {}
    for state_id, pos in states.items():
        q_table[state_id] = initialize_values(generic_actions, pos)
    return q_table

def q_learning(episodes=1000):
    # Initialize environment
    S = create_grid()
    q_table = create_q_table(S, (GRID_ROWS, GRID_COLS))
    converging = 0
    
    for episode in range(episodes):
        state = START_STATE
        total_reward = 0
        old_qtable = deepcopy(q_table)
        
        while True:
            # Epsilon-greedy action selection
            if np.random.random() < EPSILON:
                action = np.random.choice(possible_acts(state))
            else:
                state_id = list(S.keys())[list(S.values()).index(state)]
                action_values = q_table[state_id]
                action = generic_actions[np.argmax(action_values)]
            
            # Take action
            next_state = get_next_state(state, action)
            reward = get_reward(next_state)
            total_reward += reward
            
            # Reset to start if fell into cliff
            if is_cliff_state(next_state):
                next_state = START_STATE
            
            # Q-learning update
            current_state_id = list(S.keys())[list(S.values()).index(state)]
            next_state_id = list(S.keys())[list(S.values()).index(next_state)]
            
            current_q = q_table[current_state_id][generic_actions.index(action)]
            next_max_q = max(q_table[next_state_id])
            
            # TD Update
            td_target = reward + GAMMA * next_max_q
            td_error = td_target - current_q
            q_table[current_state_id][generic_actions.index(action)] += ALPHA * td_error
            
            state = next_state
            
            if state == GOAL_STATE:
                break
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
        
        if old_qtable == q_table:  ## early stop  
            if converging == 19: ## 20 iguais para convergencia
                break
            converging += 1
            

    print(f"Episode {episode}, Total Reward: {total_reward}")
    return q_table

if __name__ == "__main__":
    trained_q_table = q_learning()