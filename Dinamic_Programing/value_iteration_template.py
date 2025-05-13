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
    gamma = 1 # <YOUR CODE HERE>
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

Delta = 10
k = 0
theta = 1e-16  # <YOUR CODE HERE>
while Delta > theta:
    Delta = 0
    for s in state_set:
        v = V[s]
        action_set = range(0, min(s, 100-s)+1)   # o quanto pode ser apostado # <YOUR CODE HERE>
        V[s] = max(expected_value(s, a) for a in action_set) # maior valor esperado
        Delta = max(Delta, abs(V[s] - v))
    k += 1

#print(V)
plt.plot(V)  ## V --> probabilidade de vitória
plt.title(f"Values   theta: {theta}, gamma: 0.6")

def policy(s):
    action_set = range(0, min(s, 100-s)+1)
    # One step lookahead to find the best action for this state
    values_actions = [float(expected_value(s,a)) for a in action_set]# <YOUR CODE HERE>
    return np.argmax(values_actions)  # 0,1,.... min(s,100-s)

final_policy = [policy(s) for s in state_set]


plt.figure()
plt.bar(state_set, final_policy, align='center', alpha=0.5)
plt.plot(state_set, final_policy,'.')
plt.title(f"Policy  theta: {theta}, gamma: 1")
plt.show()