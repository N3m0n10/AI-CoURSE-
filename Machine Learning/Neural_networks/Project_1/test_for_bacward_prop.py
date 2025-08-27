from utils import backward_prop_batch , gradient_validation, xavier_init #, forward_prop_batch 
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "classification2.txt")

try:
    with open(data_path, "r") as input_file:
        training_data = input_file.readlines()
except FileNotFoundError:
    print(f"Error: Could not find {data_path}")
    print(f"Current working directory: {os.getcwd()}")

training_data = [line.strip().split(",") for line in training_data]
input_data = np.array([[float(data[0]), float(data[1])] for data in training_data])
X = input_data
Y = np.array([int(data[2]) for data in training_data]).reshape(-1, 1)
####################################
###### data sets conditioning ######
####################################
batch_size = 20
Xlist , Ylist = []
iter_num = range(np.round(len(X)/batch_size))
for j in iter_num:
    pass  #TODO

##############################
###### hyperparamiters #######
##############################
init_epsilon = 1e-2
epochs = 5000
lbd = 0.00001

##############################
###### Initialize Theta ######
##############################
shapes = [(10,3),(10,11),(1,11)]
theta = [xavier_init(shape[0],shape[1]) for shape in shapes]

for iter in range(epochs):
    pass


