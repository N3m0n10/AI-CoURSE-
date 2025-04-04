import numpy as np
import matplotlib.pyplot as plt

# 2 inputs , numerical
# 1 hidden layer
# 2 outputs, 2 hidden neurons (4 neurons)

# least squares 4all
# sigmoid classifier 4all

# ACTIVATION = y

def sigmoid ( z ) :   #.expit
    return 1 / (1 + np.e**-z)