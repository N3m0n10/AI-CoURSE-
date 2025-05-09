import numpy as np


"""
NOTE: CLIFF PROBLEM
assert random init values
check actions
for each action cheack q values for each
sum probs
do the TD atualize for values
"""

generic_actions = ["left","right", "up", "down"]  


"Rewards are:"
"-1 for normal steps"
"-100 for clif"
"+1 for terminal"

## Cliff restarts player pos !!
alpha = 0.3  ## (0,1]
e = 0.001    ## e > 0, e near 0

#see how many states there are, them
def initialize_values(actions,max_grid:tuple,pos):
    state_weights = [0,0,0,0] # for actions in grid world 
    for act in actions:
        nextP = [0,0]
        match act: 
            case "left":
                nextP = pos + (-1,0)
            
        if nextP[0] < 0 or nextP[0] > max_grid[0] \
        or nextP[1] < 0 or nextP[1] > max_grid[1]:
            state_weights.append(float("-inf"))

        else:
            state_weights.append(np.random())  # random init values
    return state_weights

def create_grid(grids:tuple,grid_dim = 20) -> list: # only retangular grids
    # each pos is a State
    S = {}
    k = 0
    for i in range(grids[0]):
        for j in range(grids[1]):
            k += 1
            S[k] = (i,j)  # estado k tem tais posições
    return S
    ## constructions of grid

#TODO: create clif !!!! TODO

#NOTE: try to find a way to stipulate possible moies per state only one time

def create_q_table(states,grid_dim):
    q_table = {}
    for state_id, pos in states.items():
        q_table[state_id] = initialize_values(generic_actions, grid_dim, pos)
    return q_table


def possible_acts(state):
    " se a soma do movimento ao estado "
    "resultar em um index inexistente na"
    "matrix to tabuleiro -> ação inválida"
    pass

#q-learning

#do enviroment inits
while True: ### comparar Q values da iter anterior com atual, se dif menor que e -> convergência!
    actual_state = S["0"]
    while True: # while not terminal 
        # action == random 
        # Q_obs(estado,ação) = rw do prox_estado + gamma * max (transitions_values(state))
        # calcula a diferença temporal = Q atual - Q_obs 
        # atualiza o valor do estado ação com a DT