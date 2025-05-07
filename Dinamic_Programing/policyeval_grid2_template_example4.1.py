import numpy as np
import matplotlib.pyplot as plt

# Entrega #4 - Dynamic Programming - TEMPLATE
#
# grid world, episodic task (terminal state)
# bellman equation, slide 29
# policy evaluation / prediction, for estimating V, page 27

actions = ['l', 'r', 'u', 'd']

# transition function: 
# maps current state s, and action a to next state ns and reward
# returns a tuple <next state>, <reward>
def act(s, a):
    ns = list(s)
    reward = -1  

    # special case
    if s == [0,0] or s == [3,3]:
        return ns, 0  
    if a == 'r':
        ns[1] += 1
    elif a == 'l':
        ns[1] -= 1
    elif a == 'u':
        ns[0] -= 1
    elif a == 'd':
        ns[0] += 1

    # test for action taking outside the grid
    for i in [0,1]: # for both dimensions
        if ns[i] < 0:
            ns[i] = 0
        if ns[i] >= 4:
            ns[i] = 3
    return ns, reward

pi_as = 0.25  # pi(a/s), equiprobable random policy
gamma = 0.99 # discount factor  

Delta = 10
k = 0
# mx, my = 4,4
value = np.zeros((4,4))

while Delta > 0.01:
    Delta = 0
    # for all states in state space (grid (i,j))
    for i in range(4):
        for j in range(4):
            v = 0 
            s = [i,j]
            for a in actions:
                snext, rw = act(s, a)
                v += pi_as * (rw + gamma*value[snext[0],snext[1]])
            Delta = max(Delta, abs(v - value[i,j]))
            value[i, j] = v
    k += 1

print(value)
print(f"iter num: {k}")

plt.imshow(value, cmap='Greys')

for i in range(4):
    for j in range(4):
        if (i,j) == (0,0) or (i,j) == (3,3):
            print('a')
            plt.text(j, i, f'{value[i,j]:.0f}', 
                    ha='center', va='center', color='white')
        else:
            plt.text(j, i, f'{value[i,j]:.1f}', 
                    ha='center', va='center', color='black')

plt.xticks([])  # Remove eixo 
plt.yticks([])  

plt.title('Value Function Grid')
plt.show()
