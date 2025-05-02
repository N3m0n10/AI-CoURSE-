## save thetas and result to a txt file
from utils import J,j, gradient_descent, forward_prop, backward_prop, sigmoid,\
    gradient_validation, bin_logistic_error, gradient_for_op_minimize, gradient_descent_for_grad_check
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import os

## logistic regression
## cross entropy error
## 2 hidden (for now)
## hypostesis will change the layer number

# testing and validation
# cross validation

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "classification2.txt")

try:
    with open(data_path, "r") as input_file:
        training_data = input_file.readlines()
except FileNotFoundError:
    print(f"Error: Could not find {data_path}")
    print(f"Current working directory: {os.getcwd()}")
training_data = [line.strip().split(",") for line in training_data]
input_file.close()
####################################
########## RANDOMIZE INPUT #########
####################################
np.random.shuffle(training_data)
####################################
####################################
####################################
input_data = np.array([[float(data[0]), float(data[1])] for data in training_data])
X_mean = input_data.mean(axis=0)
X_std = input_data.std(axis=0)
data = (input_data - X_mean) / (X_std + 1e-8)
#X = data
X = input_data
Y = np.array([float(data[2]) for data in training_data], dtype=np.float32).reshape(-1, 1)

####################################
###### data sets conditioning ######
####################################
train_set_X , train_set_Y = X[:80] , Y[:80]  ## 80 for train
valid_set_X , valid_set_Y = X[80:103] , Y[80:103]  ## 23 for validation
check_set_X , check_set_Y = X[103:] , Y[103:]  ## 15 for test

M_train = len(train_set_X)
M_valid = len(valid_set_X)
M_check = len(check_set_X)

####################################
####################################
####################################
 
def xavier_init(n_out, n_in):
    return np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)

shapes = [[(10, 3), (7, 11), (1, 8)],
          #[(2, 3), (2, 3), (1, 3)],
          [(15, 3), (15, 16), (1, 16)],
          [(10, 3), (10, 11), (10, 11), (1, 11)]
          #[(15, 3), (25, 16), (15, 26), (1, 16)],
          #[(30, 3), (35, 31), (30, 36), (1, 31)]
          ]

#constants
init_epsilon = 1e-2
epochs = 10000

#Variable hyper paramiters
regularizer = [0,0.01,0.001]
learning_rate = [0.1,0.01,0.001]
results = []
keys = []

for lbd in regularizer:
    for alpha in learning_rate:
        for shape in shapes:

            W = []

            for k in range(len(shape)):
                W.append(xavier_init(shape[k][0],shape[k][1]))

            n_layers = len(W) + 1

            ###############################
            ##### gradient validation #####
            ###############################
            grad_checked = False
            while not grad_checked:
                theta = np.concatenate([w.flatten() for w in W])
                initial_theta = np.hstack(theta)  
                analytical_grad = gradient_descent_for_grad_check(train_set_X, train_set_Y, W, lbd=lbd)
                numerical_grad = gradient_validation(theta, train_set_X, train_set_Y,shape)
                analytical_grad = np.concatenate([g.flatten() for g in analytical_grad])
                
                if sum(np.abs(analytical_grad - numerical_grad))/len(theta) < 1e-2:
                    print("Gradient check passed")
                    grad_checked = True
                else:
                    print("gradient_check_failed")
            ###############################
            ##### gradient validation #####
            ###############################
            ###############################
            ########## TRAINING ###########
            ###############################
            acc = 0
            acc_list , rslt_list = [] , []
            for tests in range(3):  ## in case of non convergence, but 1 training is expected for majors shapes
                print(f"training: {tests}")
                
                result = gradient_descent(train_set_X, train_set_Y, W, learning_rate=alpha, epochs=epochs, lbd=lbd)
                reshaped_W = result[0]

                for i in range(M_valid):  
                    if valid_set_Y[i] == np.round(forward_prop(valid_set_X[i], reshaped_W, n_layers)[0][-1]): acc+=1
                acc = acc / M_valid

                if acc >= 0.75:
                    break

                acc_list.append(acc)
                rslt_list.append(result)

            if tests >= 4:  ## acc wasn't fullfiled
                result,acc = rslt_list[np.argmax(acc_list)], max(acc_list)  ## chooses best

            results.append([f"lbd:{lbd},alpha:{alpha},\n shape:{shape}",[result,acc]])  

            ###############################
            ########## TRAINING ###########
            ###############################

################validation################
acc_check = []
acc = 0
for rs in results:
    acc_check.append(rs[1][1])

choosen = results[np.argmax(acc_check)]
del acc_check 
################validation################
############################
###### final accuracy ######
############################
for i in range(M_check):  
    if check_set_Y[i] == np.round(forward_prop(check_set_X[i], choosen[1][0][0], n_layers)[0][-1]): acc+=1

final_acc = acc / M_check
weights = choosen[1][0][0]
history_loss = choosen[1][0][1]

with open("Neural_networks/training_results_a).txt", "a+") as f:
    f.seek(0)  # Go to start of file
    old = f.read()  # Read existing content
    f.seek(0)  # Go back to start
    f.truncate()  # Clear the file
    text = "#####################################\n"+ \
    f"{choosen[0]} \n" +\
    f"accuary: {final_acc} \n"+\
    f"loss: {history_loss}\n"+\
    f"Final weights: {weights}\n"
    f.write(old + text)

## TODO: HISTORY LOSS PLOT
plt.plot(np.linspace(0, np.pi, 500),history_loss,label = "Loss")
plt.title(f"Loss of {choosen[0]}")
plt.legend()
plt.show()