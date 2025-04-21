## save thetas and result to a txt file
from utils import J,j, gradient_descent, forward_prop, backward_prop, sigmoid, gradient_validation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import os

## TODO: BIAS

#result = op.minimize(fun,J,jac=gradient, x0=theta, method='TNC', args=(X, Y, lbd))

## logistic regression
## cross entropy error
## 2 hidden (for now)
## hypostesis wil change the layer number

# tesing and validation
# cross validation

init_epsilon = 0.01
learning_rate = 0.01

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "classification2.txt")

try:
    with open(data_path, "r") as input_file:
        training_data = input_file.readlines()
except FileNotFoundError:
    print(f"Error: Could not find {data_path}")
    print(f"Current working directory: {os.getcwd()}")
training_data = [line.strip().split(",") for line in training_data]
X = np.array([[float(data[0]), float(data[1])] for data in training_data])
Y = np.array([float(data[2]) for data in training_data])
input_file.close()
M = len(X)
## TODO: separate training and test data

#provisory
theta0 = np.random.rand(2,2) * 2 * init_epsilon - init_epsilon
theta1 = np.random.rand(2,2) * 2 * init_epsilon - init_epsilon
theta2 = np.random.rand(1,2) * 2 * init_epsilon - init_epsilon

W = [theta0, theta1, theta2]
n_layers = len(W) + 1

a = forward_prop(X[0],W,n_layers)
print(a[-1])

# gradient validation 
theta = np.concatenate([w.flatten() for w in W])
print(theta)
grad_extimate = gradient_validation(theta)
# cmompare 
# start descent

## test 1 ->  
result, H = gradient_descent(X, Y, W, learning_rate=0.01, epochs=100, lbd=0)
print(result)
prediction = []
#for i in range(len(X)):
#    prediction.append(forward_prop(X[i], result, len(result) + 1)[-1]) 
# result = op.minimize(fun=J, jac=ut.gradient_for_op_minimize, x0=theta, method='TNC', args=(X, Y, lbd))
error = J(H, Y, lbd=0)  # no regularization
print(result)
print(error)