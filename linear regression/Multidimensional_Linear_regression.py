# NEMO RAMOS LOPES NETO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f_true ( x0 , x1 , x2 ) :
    return 2 + 0.8 * x1 + 0.7 * x2

points = 100
variables = 2
# dataset {( x , y ) }
xs = np.array([[1 for i in range(points)],*[np . linspace ( -3 , 3 , 100) for _ in range(variables)]]).T
# xs = [[x0==1,x1,x2], ... ]
ys = np . array ( [ f_true ( x0 ,x1 , x2) + np . random . randn () *0.5 for x0,x1,x2 in xs] ) 


tht = np.array([np.random.random() for o in range(variables + 1)]) # initial theta
a = 0.001 # learning rate
epochs = 3000 # iterations 
m = len(ys) # len(data)
factor = a/m



'''Hyphotesis'''
def h (x , theta ,i, n = variables): #sum all tethas*repective_X
    h_sum = 0
    for a in range(n +1):
        h_sum += theta[a]*x[i][a]
    return h_sum

'''Cost function'''
def J ( sum) : 
    return sum/(2*m) 

def gradient (i , theta , xs , ys , n) : 
    grad_sum = 0
    for _ in range(points): #Sum the predction for each row of x and y for the desirable theta
        grad_sum += (h(xs,theta,_) - ys[_])*xs[_][i]
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
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Create meshgrid for surface plot
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Predicted values using learned parameters
    Z_pred = theta[0] + theta[1]*X1 + theta[2]*X2
    
    # True function values
    Z_true = 2 + 0.8*X1 + 0.7*X2
    
    # Plot surfaces
    ax2.plot_surface(X1, X2, Z_pred, alpha=0.5, cmap='viridis', label='Predicted')
    ax2.plot_surface(X1, X2, Z_true, alpha=0.3, cmap='plasma', label='True')
    
    # Plot training points
    ax2.scatter(xs[:,1], xs[:,2], ys, c='red', s=10, label='Training points')
    
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('Y')
    ax2.set_title('True vs Predicted Surface')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

loss_list = []
for _ in range(epochs):
    grad_sum = np.zeros(variables + 1)
    loss = 0
    ## calculate new gradient and loss
    for tet in range(variables + 1):  # for each theta
        grad = gradient(tet,tht,xs,ys,variables)
        ## calculate new theta
        tht[tet] -= factor*grad
    for w in range(points):
        loss += (h(xs,tht,w) - ys[w])**2
    loss = J(loss)
    if _ % 1000 == 0:
        print(tht)
    
    loss_list.append(loss)
    ## append values for plot - > loss func for epoch 
print(tht)

print_modelo(tht,xs,ys,loss_list)   