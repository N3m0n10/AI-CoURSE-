import numpy as np
import matplotlib.pyplot as plt
import os

start_time = os.times()[4]
def f_true ( x ) :
    return 2 + 0.8 * x

# Add after f_true definition
def h(x, theta):
    return theta[0] + theta[1] * x

# dataset {( x , y ) }
xs = np . linspace ( -3 , 3 , 100)
ys = np . array ( [ f_true ( x ) + np . random . randn () *0.5 for x in xs ])
X = np.vstack([np.ones_like(xs), xs]).T  # Add bias term
tht = np.linalg.inv(X.T @ X) @ X.T @ ys

def print_modelo ( theta , xs , ys ) :
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.scatter(xs, ys, color='blue', label='Data points', s=1.5)
    y_pred = [h(x, theta) for x in xs]
    y_true = [f_true(x) for x in xs]
    ax.plot(xs, y_pred, 'r-', label='Model')
    ax.plot(xs, y_true, 'g--', label='True function')
    ax.legend()
    ax.set_title(f"Linear Regression [θ₀:{theta[0]:.2f}, θ₁:{theta[1]:.2f}]")
    
    plt.tight_layout()
    plt.show()

finish_time = (os.times()[4] - start_time)
print(f"{finish_time:.2f} seconds")
print_modelo(tht,xs,ys)          