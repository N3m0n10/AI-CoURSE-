import numpy as np
import matplotlib.pyplot as plt
from matplotlib . pyplot import subplot , plot , show , clf , vlines
import os

#checar probabilidade
#trocar theta
#loop

# conjunto de dados {( x , y ) }
mean0 , std0 = -0.4 , 0.5
mean1 , std1 = 0.9 , 0.3
m = 200

x1s = np . random . randn ( m //2) * std1 + mean1
x0s = np . random . randn ( m //2) * std0 + mean0

#would rather adjust with the listed arguments, outp: array([[1, 4],...


xs = np . hstack (( x1s , x0s ) )
ys = np . hstack (( np . ones ( m //2) , np . zeros ( m //2) ) )

plot( xs [: m //2] , ys [: m //2] , '.')
plot( xs [ m //2:] , ys [ m //2:] , '.')
show()

def sigmoid ( z ) :   #.expit
    return 1 / (1 + np.e**-z)


def h (x , theta ) :
    return sigmoid ( np.array([theta @ x[: m //2], theta @ x[ m //2:]] ))
    #return sigmoid ( theta.T @ x)

def cost (h , y ) :
    return  y*np.log(h) + (1 - y)*np.log(1-h)

def J ( theta , xs , ys ) :
    pass
print(x1s)
print("xos",x0s)
print(xs)
print(xs[0])
def gradient (i , theta , xs , ys ) :
    dif = h(xs,theta) - ys[i]
    return np.array([dif*xs[: m //2 + i], dif*xs[ m //2:]])

def plot_fronteira ( theta ) :
# use vlines () para plotar uma reta vertical
    pass

def print_modelo ( theta , xs , ys ) :
    pass

def accuracy ( ys , predictions ) :
    num = sum ( ys == predictions )
    return num / len ( ys )

alpha = 0.1 
epochs = 5000
theta = np.array([10 for i in range(100)]) 

for k in range ( epochs ) : # 10000
    sum_g = 0
    for train_points in range(m):
        sum_g += gradient(m,theta,xs,ys)
    theta -= alpha*sum_g


# show classication performance
#print ( " Acuracia : " , accuracy ( ys , predictions ) )