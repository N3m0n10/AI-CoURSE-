import numpy as np
import matplotlib.pyplot as plt

# Entrega #4 - Dynamic Programming - TEMPLATE
#
# value iteration, Gambler's Problem
# slide 35

state_set = list(range(1,100))  # 1, 2, ..., 99

V = np.zeros( (len(state_set)+1, 1) )

ph = 0.4    # probability of heads

# returns next_state and its probability
def next_states(s, a):
    return [
        (s+a, ph),  # head
        (s-a, 1-ph)  # tails
    ]

def expected_value(s, a):
    gamma = 0.8 # <YOUR CODE HERE>
    evalue = 0
    for snext, prob in next_states(s, a):
        # reward is 1 if action leads to goal of 100 = s+a,
        # which is the sum of capital (s) and stake (a)
        if snext <= 0:
            continue # evalue += 0
        if snext >= 100:
            evalue += prob # prob * 1 
        else: 
            evalue += prob * gamma*V[snext]
        # <YOUR CODE HERE>
        # terminate if snext = 0 or 100 (dummy states for termination)
        # <YOUR CODE HERE>
    return evalue

def policy(s):
    action_set = range(0, min(s, 100-s)+1)
    # One step lookahead to find the best action for this state
    values_actions = [np.float64(expected_value(s,a)) for a in action_set]# <YOUR CODE HERE>
    return np.argmax(values_actions)  # 0,1,.... min(s,100-s)

for i in range(48): ## sweeps
    Delta = 10
    k = 0
    theta = 1e-6
    while Delta > theta:
        Delta = 0
        for s in state_set:
            v = V[s]
            action_set = range(0, min(s, 100-s)+1)   # o quanto pode ser apostado # <YOUR CODE HERE>
            V[s] = max(expected_value(s, a) for a in action_set) # maior valor esperado
            Delta = max(Delta, abs(V[s] - v))
        k += 1

    #print(V)
    if i <= 4 or i%16 == 0:
        plt.plot(V, label= f"{i}")  ## V --> probabilidade de vitória
        plt.legend()

    final_policy = [policy(s) for s in state_set]


plt.figure()
plt.bar(state_set, final_policy, align='center', alpha=0.5)
plt.plot(state_set, final_policy,'.')
plt.show()