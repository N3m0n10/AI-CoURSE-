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
## hypostesis will change the layer number

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "classification2.txt")

try:
    with open(data_path, "r") as input_file:
        training_data = input_file.readlines()
except FileNotFoundError:
    print(f"Error: Could not find {data_path}")
    print(f"Current working directory: {os.getcwd()}")
training_data = [line.strip().split(",") for line in training_data]
####################################
########## RANDOMIZE INPUT #########
####################################

input_data = np.array([[float(data[0]), float(data[1])] for data in training_data])
X = input_data
Y = np.array([int(data[2]) for data in training_data]).reshape(-1, 1)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y= Y[indices]
#indices = None

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

shapes = [[(50,3),(1,51)]
          #[(10, 3), (7, 11), (1, 8)],
          #[(2, 3), (2, 3), (1, 3)],
          #[(15, 3), (15, 16), (1, 16)],
          #[(10, 3), (10, 11), (10, 11), (1, 11)],
          #[(15, 3), (25, 16), (15, 26), (1, 16)],
          #[(30, 3), (35, 31), (30, 36), (1, 31)]
          ]

#constants
init_epsilon = 1e-2
epochs = 10000

#Variable hyper paramiters
#regularizer = [0,0.001]
#learning_rate = [0.01,0.001]
regularizer = [0]
learning_rate = [0.5]
results = []
keys = []

for lbd in regularizer:
    for alpha in learning_rate:
        for shape in shapes:
            print(lbd,alpha,shape)

            W = []

            for k in range(len(shape)):
                W.append(xavier_init(shape[k][0],shape[k][1]))

            n_layers = len(W) + 1

            ###############################
            ##### gradient validation #####
            ###############################
            
            
            theta = np.concatenate([w.flatten() for w in W])
            initial_theta = np.hstack(theta)  
            analytical_grad = gradient_descent_for_grad_check(train_set_X, train_set_Y, W, lbd=lbd)
            numerical_grad = gradient_validation(theta, train_set_X, train_set_Y,shape)
            analytical_grad = np.concatenate([g.flatten() for g in analytical_grad])
            
            if sum(np.abs(analytical_grad - numerical_grad))/len(theta) < 1e-2:
                print("Gradient check passed")
                
            else:
                print("gradient_check_failed")
                continue
        
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

                if acc >= 0.6:
                    break

                acc_list.append(acc)
                rslt_list.append(result)

            if tests >= 2:  ## acc wasn't fullfiled
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
print(choosen)
print(choosen[1][0][0])
del acc_check 
################validation################
############################
###### final accuracy ######
############################

weights = choosen[1][0][0]
history_loss = choosen[1][0][1]
for result in results:
    with open("Neural_networks/Project_1/training_results_c).txt", "a+") as f:
        f.seek(0)  # Go to start of file
        old = f.read()  # Read existing content
        f.seek(0)  # Go back to start
        f.truncate()  # Clear the file
        text = ("#####################################\n"
        f"indeces:{indices}\n"
        f"{result[0]} \n" 
        #f"accuary: {final_acc} \n"+\
        f"loss: {result[1][0][1][-1]}\n"
        f"Final weights: {result[1][0][0]}\n"
        "#######################################"
        )
        f.write(old + text)

## TODO: HISTORY LOSS PLOT
plt.plot(np.linspace(0, epochs,epochs),history_loss,label = "Loss")
plt.title(f"Loss of {choosen[0]}")
plt.legend()
plt.show()