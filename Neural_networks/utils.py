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

def j(theta): ## error func for gradient check
    activation = forward_prop(X, theta, l)[-1]  # those eill be class atrbt so no need to add to args
    cost = J(activation, T) #no regularization
    return cost

def gradient_descent(X, Y, w:list, learning_rate=0.01, epochs = 500 , lbd=0,single_output = False):
    M = len(X)
    last_dw = [0,0,0,0,0,0,0,0,0,0,0,0,0,0] ## change
    print(last_dw)
    n_layers = len(w) + 1
    initial_weights = w.copy()
    W = w.copy() 
    for epoch in range(epochs):
        for inp in range(M):
            # Forward propagation
            A = forward_prop(X[inp], W, n_layers)
            
            # Backward propagation  ## dW is delta
            dW = backward_prop(X[inp], Y[inp], A, W, n_layers, last_delta=last_dw,iter = inp)
            last_dw = dW # acumulate all the dW
            print("atualized dw", last_dw)

        # Error Derivatives and weights
        D = []
        for i in range(n_layers - 1):
            D.append(dW[i] / M + lbd * W[i])
            W[i] -= learning_rate * D[i]

        if epoch == 0:  # add a if for each iter --> check other implementation
            grad_chk = gradient_validation(np.hstack(initial_weights))
            if D - 0.0001 <= grad_chk <= D + 0.0001: #1e-4
                print("Gradient check failed")
                return None
            
        # Compute cost
            cost = J(A, Y)

        if epoch % 100 == 0:
            print(f"tht = {W}, cost = {cost}")
    return W

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


def backward_prop(X, Y, A, W, l,iter, activation_derivative=sigmoid, last_delta = []):
    A.reverse()
    dz, dw = [], []
    dz.append(A[0] - Y)
    dw.append(last_delta[iter] + np.dot(A[0].T, dz[0]))

    if l > 0:
        for i in range(1, l - 1):  
            dz.append(np.dot(dz[-1], W[-i].T) * activation_derivative(A[i],derivative = True))  ##this is sig devivation ##TODO: call the activator derivation funcion
            if i == l - 2:
                dw.append(last_delta[i] + np.dot(X.T, dz[-1]))
            else:
                dw.append(last_delta[i] + np.dot(A[i +1].T, dz[i]))
            
    dw.reverse()
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

def initialize_weights():
    pass

