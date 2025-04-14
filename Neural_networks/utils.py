import numpy as np

# TODO: MAKE THIS A CLASS 
# TODO: TEST

class NeuralNetwork:
    def __init__(self,name, args:list):
        self.name = f"name"

    ## TODO

def squared_error(y, t):
    return ((y - t)**2)

#cross entropy
def bin_logistic_error():
    pass

def categorical_logistic_error():
    pass

def sigmoid (z, derivative=False):
    if derivative == True:
        return sigmoid(z) * (1 - sigmoid(z))  ##have an if for each sig #-> REDO?
    return 1 / (1 + np.e**-z)

def forward_prop(X, W : list, l : int):
    Z = []
    A = [X] # activations

    # Calculate the output layer activations
    for i in range(l - 1):
        Z.append(np.dot(A[i], W[i]))
        A.append(sigmoid(Z[i]))

    return Z , A

def J(A, T, W1 = None, W2 = None, lbd=0, error_func=squared_error): # T = Y
    m = T.shape[0]
    cost = sum(error_func(A, T) / (2 * m))
    if lbd > 0:  # regularization
        reg_term = (lbd / (2 * m)) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        cost += reg_term
    return cost

def j(theta): ## error func for gradient check
    activation = forward_prop(X, theta, l)[1][-1]  # those eill be class atrbt so no need to add to args
    cost = J(activation, T) #no regularization
    return cost

def gradient_descent(X, Y, W1, W2, learning_rate=0.01, epochs=1000, lbd=0):
    m = Y.shape[0]
    n_layers = len(W) + 1
    W = [W1, W2]
    for epoch in range(epochs):
        # Forward propagation
        Z , A = forward_prop(X, W, n_layers)

        # Compute cost
        cost = J(A, Y, W1, W2)
        
        # Backward propagation
        dW = backward_prop(X, Y, A, W, n_layers)

        # Error Derivatives and weights
        D = []
        for i in range(n_layers - 1):
            D.append(dW[i] / m + lbd * W[i])
            W[i] -= learning_rate * D[i]

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")

    return W

def backward_prop(X, Y, A, W, l, activation_derivative=sigmoid):
    A.reverse()
    dz, dw = [], []
    dz.append(A[0] - Y)
    dw.append(np.dot(A[0].T, dz[0]))

    if l > 0:
        for i in range(1, l - 1):  
            dz.append(np.dot(dz[-1], W[-i].T) * activation_derivative())  ##this is sig devivation ##TODO: call the activator derivation funcion
            if i == l - 2:
                dw.append(np.dot(X.T, dz[-1]))
            else:
                dw.append(np.dot(A[i +1].T, dz[i]))
            
    return dw.reverse()

#provisory
def gradient_validation(theta):
    grad_aprox = np.zeros(len(theta))
    for i in range(len(theta)):
        theta_plus = np.array(theta)
        theta_minus = np.array(theta)
        theta_plus[i] += 1e-4  #make epsilon a class atribute
        theta_minus[i] -= 1e-4
        grad_aprox[i] = j(theta_plus) - j(theta_minus) / (2 * 1e-4)



