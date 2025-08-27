###############
##### Plotando fronteira de decisão não-linear
###############

from numpy import array, linspace, zeros , exp, insert, dot
import matplotlib.pyplot as plt

theta = [array([[ 7.2704242 , -4.27655366, -8.19256878],
       [ 3.40443947,  6.42453847,  6.50018174]]), array([[ 2.51093983,  0.28010074, -9.20520683],
       [-0.66456553, 11.02729546, -3.57725468]]), array([[-6.20261854, -7.98032006,  7.19606822]])]

x1s = linspace(-1,1.5,50)
x2s = linspace(-1,1.5,50)
z= zeros((len(x1s),len(x2s)))

#y = h(x) = 1/(1+exp(- z))
#z = theta.T * x

def sigmoid(z, derivative=False):
    sig = 1 / (1 + exp(-z))
    if derivative:
        return z * (1 - z)
    return sig

def net_z_output(X, W, l, activation_func=sigmoid):
    Z = []
    A = [X.reshape(-1)]  # Ensure it's a flat vector
    AWB = [insert(X, 0, 1)]  # Add bias term to input

    for i in range(l - 1):
        z = dot(W[i], AWB[i])
        Z.append(z)
        a = activation_func(z)
        A.append(a)
        AWB.append(insert(a, 0, 1))  # Add bias for next layer

    return  Z[-1]


for i in range(len(x1s)):
    for j in range(len(x2s)):
        x = array([x1s[i], x2s[j]]).reshape(2,-1)
        z[i,j] = net_z_output(x,theta,len(theta)+1)  # (x,theta,n_layers)
plt.contour(x1s,x2s,z.T,0)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()