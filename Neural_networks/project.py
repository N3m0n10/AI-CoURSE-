## save thetas and result to a txt file
from utils import J,j, gradient_descent, forward_prop, backward_prop, sigmoid
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

#result = op.minimize(fun,J,jac=gradient, x0=theta, method='TNC', args=(X, Y, lbd))

## logistic regression
## cross entropy error
## 2 hidden (for now)
## hypostesis wil change the layer number

# tesing and validation
# cross validation

init_epsilon = 0.01

input = open("classification2.txt", "r")
training_data = input.readlines()
training_data = [line.strip().split(",") for line in training_data]
X = np.array([[float(data[0]), float(data[1])] for data in training_data])
Y = np.array([float(data[2]) for data in training_data])
input.close()
M = len(X)
## TODO: separate training and test data

#provisory
theta0 = np.random.rand(4) * 2 * init_epsilon - init_epsilon
theta1 = np.random.rand(4) * 2 * init_epsilon - init_epsilon
theta2 = np.random.rand(2) * 2 * init_epsilon - init_epsilon
W = [theta0, theta1, theta2]
n_layers = len(W) + 1


## test 1 ->  
result = gradient_descent(X, Y, [theta0, theta1, theta2], learning_rate=0.01, epochs=500, lbd=0)
# result = op.minimize(fun=J, jac=ut.gradient_for_op_minimize, x0=theta, method='TNC', args=(X, Y, lbd))
error = J(result, Y, result, lbd=0)  # no regularization
print(result)
print(error)