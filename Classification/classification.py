import numpy as np
import matplotlib.pyplot as plt
import os


# conjunto de dados {( x , y ) }

mean0 , std0 = -0.4 , 0.5
mean1 , std1 = 0.9 , 0.3
m = 200

x1s = np . random . randn ( m //2) * std1 + mean1
x0s = np . random . randn ( m //2) * std0 + mean0
xs = np . hstack (( x1s , x0s ) )

ys = np . hstack (( np . ones ( m //2) , np . zeros ( m //2) ) )

plt.plot( xs [: m //2] , ys [: m //2] , '.')
plt.plot( xs [ m //2:] , ys [ m //2:] , '.')
plt.show()

def sigmoid ( z ) :
    pass


sigmoid ( theta [0] + theta [1] * x )

def h (x , theta ) :
    pass




def cost (h , y ) :
    pass

def J ( theta , xs , ys ) :
    pass

def gradient (i , theta , xs , ys ) :
    pass

def plot_fronteira ( theta ) :
# use vlines () para plotar uma reta vertical
    pass

def print_modelo ( theta , xs , ys ) :
    pass

def accuracy ( ys , predictions ) :
    num = sum ( ys == predictions )
    return num / len ( ys )

alpha = None # completar
epochs = 600
theta = None # completar

for k in range ( epochs ) : # 10000

# apply gradient decent

# show classication performance
print ( ’ Acuracia : ’ , accuracy ( ys , predictions ) )