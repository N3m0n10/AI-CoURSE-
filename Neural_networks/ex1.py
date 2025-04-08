import numpy as np
import matplotlib.pyplot as plt

# 2 inputs , numerical
# 1 hidden layer
# 2 outputs, 2 hidden neurons (4 neurons)

# least squares 4all
# sigmoid classifier 4all
# ACTIVATION == y

T = np.array([0.8,1.0]) 
X = np.array([0.0,1.0])
learning_rate = 1

W1 = np.array([[0.1,0.2],[-0.1,0.1]])
W2 = np.array([[1.,-1.],[1.,1.]])

h = np.array([.0,.0])  #output

def squared_error(y, t):
    return ((y - t)**2)

def sigmoid ( z ) :   
    return 1 / (1 + np.e**-z)

def forward_prop(X, W1, W2):
    global h
    # Calculate the hidden layer activations
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)

    # Calculate the output layer activations
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    h = A2

    return A1, A2 , 

def J(A, T, W1, W2, lbd=0):
    m = T.shape[0]
    cost = sum(squared_error(A, T) / (2 * m))
    if lbd > 0:  # regularization
        reg_term = (lbd / (2 * m)) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        cost += reg_term
    return cost

def w_derivative(w, a1, node_error, m, lbd, prev_delta=0):
    delta = prev_delta + a1 * node_error
    gradient = (1/m * delta) + (lbd * w)
    return gradient, delta
    

def layer_error(a, t):
    delta2 = a[1] - t
    delta1 = np.dot(delta2, W2.T) * (a[0] * (1 - a[0]))
    return delta1, delta2

def w_derivative2(w, a, T, m, lbd):
    dlt = a[0][1] * layer_error(a, T)[0][1]
    D = 1 / m * dlt + (lbd * w)
    return D


# 1)
## first run
y = forward_prop(X, W1, W2)
j = J(y[1], T, W1, W2)
gradient, new_delta = w_derivative(W2[1][1], y[0][1], y[1][1] -  T[1], m=T.shape[0], lbd=0)
W2[1][1] -= learning_rate * gradient
print(f"output {h}, Cost: {j}, W2[1][1]: {W2[1][1]} not used in the update")  
## second run
y = forward_prop(X, W1, W2)
j = J(y[1], T, W1, W2)
print(f"output {h}, Cost: {j}, W2[1][1]: {W2[1][1]}")
W2[1][1] -= learning_rate*w_derivative(W2[1][1],y[0][1], y[1][1] - T[1], m =  T.shape[0] \
                                       , lbd =0.1,prev_delta= new_delta)[0]  
print("Wkl - regularized: ", W2[1][1])

# the weight [Wkl] is slight decreased due to regularization

# 2)
# C - The error deacreased, wich is expected, so the network is probably working right.
#     The variation of the output being small might be due to a too strong regularization 
#     or a too small learning rate. 

# 3)
W1 = np.array([[0.1,0.2],[-0.1,0.1]])
W2 = np.array([[1.,-1.],[1.,1.]])

y = forward_prop(X, W1, W2)
j = J(y[1], T, W1, W2)
gradient = w_derivative2(W1[1][1], y, T, m=T.shape[0], lbd=0)
W2[0][1] -= learning_rate * gradient
print(f"output {h}, Cost: {j}, W2[0][1]: {W2[0][1]} not used in the update") 

y = forward_prop(X, W1, W2)
j = J(y[1], T, W1, W2)
gradient = w_derivative2(W1[1][1], y, T, m=T.shape[0], lbd=0)
print(f"output {h}, Cost: {j}, W2[0][1]: {W2[0][1]}") 
