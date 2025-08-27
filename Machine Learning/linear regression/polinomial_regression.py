print("NEMO RAMOS LOPES NETO")
import numpy as np
import matplotlib.pyplot as plt
import os


def f_true ( x) :
    return 2 + 0.8 * x + 0.7 * x**2

points = 100
order = 2
# dataset {( x , y ) }
xs = np.linspace(-3, 3, points)
ys = np . array ( [ f_true (x) + np . random . randn () *0.5 for x in xs] ) 


tht = np.array([np.random.random() for o in range(order + 1)]) # initial theta
a = 0.1 # learning rate
epochs = 50 # iterations 
m = len(ys) # len(data)
factor = a/m



'''Hyphotesis'''
def h (x , theta ): #sum all tethas*repective_X
    h_sum = 0
    for a in range(order + 1):
        h_sum += theta[a]*x**a
    return h_sum

'''Cost function'''
def J ( sum) : 
    return sum/(2*m) 

def gradient (i , theta , xs , ys , n) : 
    grad_sum = 0
    for _ in range(points): #Sum the predction for each row of x and y for the desirable theta
        grad_sum += (h(xs[_],theta) - ys[_])*xs[_]**i
    return grad_sum

def print_modelo ( theta , xs , ys ,cost) :
    fig = plt.figure(figsize=(15, 5))
    
    # Loss function plot
    ax1 = fig.add_subplot(121)
    ax1.plot(cost, 'b-')
    ax1.set_title("Loss function per iteration")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Error")
    
    # 3D surface plot
    ax2 = fig.add_subplot(122)
    
    ax2.scatter(xs, ys, color='blue', label='Data points',s = 1.5)  # Plot data points
    y_pred = [h(x, theta) for x in xs]  # Generate predictions
    y_true = [f_true(x) for x in xs]    # Generate true function
    ax2.plot(xs, y_pred, 'r-', label='Model')  # Plot model line
    ax2.plot(xs, y_true, 'g--', label='True function')  # Plot true function
    ax2.legend()
    ax2.set_title(f"Polinomial Regression a={a}")
    
    plt.tight_layout()
    plt.show()

start_time = os.times()[4]
loss_list = []
for _ in range(epochs):
    loss = 0
    ## calculate new gradient and loss
    for tet in range(order + 1):  # for each theta
        grad = gradient(tet,tht,xs,ys,order)
        ## calculate new theta
        tht[tet] -= factor*grad
    for p in range(points):
        loss += (h(xs[p],tht) - ys[p])**2
    loss = J(loss)
    if _ % 1000 == 0:
        print(tht)
    
    loss_list.append(loss)
    ## append values for plot - > loss func for epoch 
print(tht)
print(loss)

finish_time = (os.times()[4] - start_time)
print(f"{finish_time:.2f} seconds")

print_modelo(tht,xs,ys,loss_list)   