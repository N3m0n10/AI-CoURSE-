import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 2 inputs , numerical
# 1 hidden layer
# 2 outputs, 2 hidden neurons (4 neurons)

# least squares 4all
# sigmoid classifier 4all

# ACTIVATION = y
lbd = 0.1
#delta = [np.zeros(2,2), np.zeros(2,2)]
delta = 0
t = np.array([0.8,1.]) #outpot
#X = 

w1 = np.array([[0.1,-0.1],[0.2,0.1]])
w2 = np.array([[1.,1.],[-1.,1.]])
initil_theta = np.hstack(w1, w2)
h = np.array([.0,.0])

def squared_error(y, t):
    return ((y - t)**2)/2

def sigmoid ( z ) :   #.expit
    return 1 / (1 + np.e**-z)

def foward_prop(X, W1, W2):
    global h
    # X is the input data
    # W1 is the weights for the first layer
    # W2 is the weights for the second layer

    # Calculate the hidden layer activations
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)

    # Calculate the output layer activations
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    h = A2

    return A1, A2

def backward_prop(X, Y, A1, A2, W2):
    # Calculate the gradients for W2
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2)

    # Calculate the gradients for W1
    dZ1 = np.dot(dZ2, W2.T) * (A1 * (1 - A1))
    dW1 = np.dot(X.T, dZ1)

    return dW1, dW2

def J(A, T):
    m = T.shape[0]
    return squared_error(A,T) / (2 * m)

def E_der_to_w(w,a1,layer_error, m , lbd):
    global delta
    global D
    delta = delta + np.dot(a1[2][2], layer_error[2,2])
    D = 1/m * delta + lbd * w
    return D

def J(A, Y, W1, W2, lbd=0.0):
   
    m = Y.shape[0]  # Number of training examples
    
    # Compute the least squares cost
    squared_errors = np.sum((A - Y) ** 2)  # Sum of squared errors
    cost = (1 / (2 * m)) * squared_errors  # Mean squared error
    
    # Add regularization term (optional)
    if lbd > 0:
        reg_term = (lbd / (2 * m)) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        cost += reg_term
    
    return cost

def gradient_descent(X, Y, W1, W2, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        # Forward propagation
        A1, A2 = foward_prop(X, W1, W2)

        # Compute cost
        cost = J(A2, Y, W1, W2)
        
        # Backward propagation
        dW1, dW2 = backward_prop(X, Y, A1, A2, W2)

        # Update weights
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")

    return W1, W2

rsult = minimize(J, initil_theta, args=(X, t), method='TNC', jac=gradient_descent)