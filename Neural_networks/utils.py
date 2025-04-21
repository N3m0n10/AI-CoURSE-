import numpy as np

# TODO: MAKE THIS A CLASS 
# TODO: TEST

class NeuralNetwork:
    ## args: input, output, hidden, activation_func, learning_rate, epochs, 
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

def forward_prop(X, W : list, l : int,single_output = False):
    Z = []
    A = [X] # activations

    # Calculate the output layer activations
    for i in range(l - 1):
        Z.append(np.dot(A[i], W[i].T))  ## A must have to be separeted by neuron
        A.append(sigmoid(Z[i]))

    return A

def J(A, T, w = None, lbd=0, error_func=squared_error): # T = dados de saída
    m = len(T)
    cost = np.sum(error_func(A, T) / (2 * m))
    if lbd > 0:  # regularization
        sum += np.sum(w[i] for i in range(len(w)) ** 2) 
        reg_term = (lbd / (2 * m)) * sum
        cost += reg_term
    return cost

def j(theta,X,T,l): ## error func for gradient check
    activation = forward_prop(X, theta, l)[-1]  # those eill be class atrbt so no need to add to args
    cost = J(activation, T) #no regularization
    return cost

def gradient_descent(X, Y, w:list, learning_rate=0.01, epochs = 500 , lbd=0,single_output = False):
    M = len(X)
    last_dw = [np.zeros_like(w_i) for w_i in w]
    print(last_dw)
    n_layers = len(w) + 1
    initial_weights = w.copy()
    W = w.copy()
    for epoch in range(epochs):
        H = [] 
        for inp in range(M):
            # Forward propagation
            A = forward_prop(X[inp], W, n_layers)
            
            # Backward propagation  ## dW is delta
            dW = backward_prop(X[inp], Y[inp], A, W, n_layers, last_delta=last_dw,iter = inp)
            last_dw = dW # acumulate all the dW
            print("atualized dw", last_dw)
            H.append(A[-1])  # Store the output of the last layer for each input

            # Error Derivatives and weights
        D = []
        for i in range(n_layers - 2):
            D.append(dW[i] / M + lbd * W[i])
            W[i] -= learning_rate * D[i]

        #gradient validation
        #if epoch == 0:  # add a if for each iter --> check other implementation
        #    gdr_stack = []
        #    for i in range(len(initial_weights)):
        #        for j in initial_weights[i]:
        #            gdr_stack.append(np.hstack(j))
        #    print("gdr_stack", gdr_stack)
        #    grad_chk = gradient_validation(gdr_stack)
        #    if D - 0.0001 <= grad_chk <= D + 0.0001: #1e-4
        #        print("Gradient check failed")
        #        return None
                
        cost = J(H, Y)

        if epoch % 100 == 0:
            print(f"tht = {W}, cost = {cost}")


    return W , H

def gradient_for_op_minimize():
    # Remember that there can't be args, so match the variables propely
    # There's no "epoch" here
    for inp in range(M):
        A = forward_prop(X, W, n_layers)
        dW = backward_prop(X, Y, A, W, n_layers, last_delta=last_dw)
        last_dw = dW
    D = []
    for i in range(n_layers - 1):
        D.append(dW[i] / M + lbd * W[i])
        W[i] -= learning_rate * D[i]

    return W


def backward_prop(X, Y, A, W, l, activation_derivative=sigmoid, last_delta = [], iter=None): ## FIX ITER I
    if last_delta is None:
        last_delta = [np.zeros_like(w) for w in W]  # Initialize last_delta if not provided

    A_reversed = A[::-1]  # Reverse a copy of A to avoid modifying the original
    ldt_reverse = last_delta[::-1]  # Reverse last_delta to match the order of W
    dz, dw = [], []

    # Output layer delta
    dz.append(A_reversed[0] - Y)
    dz_reshaped = dz[0].reshape(-1, 1)  
    A_reversed_reshaped = A_reversed[1].reshape(1, -1)
    print(ldt_reverse[0])
    print(A_reversed[1].T)
    print(dz[0])
    dw.append(ldt_reverse[0] + np.dot(A_reversed_reshaped.T, dz_reshaped).flatten())

    # Hidden layers
    for i in range(1, l - 2):
        dz.append(np.dot(W[-i].T,dz[-1]) * activation_derivative(A_reversed[i], derivative=True))
        if i == l - 2:  # First hidden layer (connected to input)
            dw.append(last_delta[0][i] + np.dot(X[iter].T, dz[-1]))
        else:  # Other hidden layers
            print("A_reversed[i + 1].T", A_reversed[i + 1].T)
            print("dz[-1]", dz[-1])
            print("lst_dlt", last_delta[i])
            dw.append(last_delta[i] + np.dot(A_reversed[i + 1].T, dz[-1]))

    dw.reverse()  # Reverse dw to match the order of W
    return dw

#provisory
def gradient_validation(theta): # use np.hstack()
    grad_aprox = np.zeros(len(theta))
    for i in range(len(theta)):
        theta_plus = np.array(theta)
        theta_minus = np.array(theta)
        theta_plus[i] += 1e-4  #make epsilon a class atribute
        theta_minus[i] -= 1e-4
        grad_aprox[i] = j(theta_plus) - j(theta_minus) / (2 * 1e-4)
    return grad_aprox

def initialize_weights():
    pass

