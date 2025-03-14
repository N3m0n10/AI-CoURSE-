# Author: Eric A. Antonelo, 2020

import math
import numpy as np

L = 4 * 8 # size of chromossome in bits

import struct

def floatToBits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>L', s)[0]

def bitsToFloat(b):
    s = struct.pack('>L', b)
    return struct.unpack('>f', s)[0]

def get_bits(x):
    x = floatToBits(x)
    N = 4 * 8
    bits = ''
    for bit in range(N):
        b = x & (2**bit)
        #print(b)
        bits += '1' if b > 0 else '0'
    return bits

def get_float(bits):
    x = 0
    assert(len(bits) == L)
    for i, bit in enumerate(bits):
        bit = int(bit)  # 0 or 1
        x += bit * (2**i)
    return bitsToFloat(x)

from random import randint
#####################################

''' arguments: two string of 0s and 1s
    returns: two string of 0s and 1s
'''
def crossover(crom1, crom2):  ### <--- COMPLETAR
    locus = None    # sorteie o locus para o crossover
    child1 = None   # gere o primeiro filho
    child2 = None   # gere o segundo filho
    return child1, child2


''' arguments: one string of 0s and 1s
    returns: one string of 0s and 1s
'''
def mutation(chromossome):    ### <--- COMPLETAR
    locus = None    # sorteie o locus para a mutacao
    # altere o bit no locus de chromossome
    c = None        # nova string com bit alterado
    return c


def g(x):
    return x + abs(math.sin(32 * x))

# chromossome: string  
def fitness(chromossome, return_x=False):
    x = get_float(chromossome)
    if x >= 0 and x <= math.pi:
        if return_x:
            return g(x), x
        return g(x)
    if return_x:
        return 0,0
    return 0

''' retorna um individuo (sequencia de bits) da primeira geracao
'''
def new_chromossome():        ### <--- COMPLETAR
    x = None     # crie um numero aleatorio representando um individuo da populacao inicial
    c = get_bits(x)
    return c

''' Metodo da roleta (selecao proporcional a aptidao).
    - argument fitness_pop: lista de aptidoes nao normalizadas para individuos da populacao. 
    O indice dessa lista indica o individuo com a respectiva aptidao.
    - returns: indice da lista sorteado (individuo sorteado)
''' 
def select(fitness_pop):       ### <--- COMPLETAR
    # normaliza aptidoes
    f = np.array(fitness_pop)
    f = f / np.sum(f)
    # escrever codigo de sorteio
    # ...
    return k # indice da lista sorteado


population = []
N = 100     # numero de individuos da populacao

for _ in range(N):
    population.append(new_chromossome())

iterations = 600
pm = 0.001
pc = 0.7

score = []
log = []

# ger 1 -> ger 2 -> ger 3 -> ....

for gen in range(iterations):
    print(' generation ', gen)
    fitness_pop = []
    new_pop = []

    for i in range(0,N-1):
        fitness_pop.append(fitness(population[i])) 

    score.append(np.mean(fitness_pop))

    while len(new_pop) < N:  # criar nova geração
        draw1 = select(fitness_pop)
        draw2 = draw1
        while draw1 == draw2:
            draw2 = select(fitness_pop)

        # parents, crossover
        
        par1, par2 = population[draw1], population[draw2]
        if np.random.rand() < pc:        
            child1, child2 = crossover(par1, par2) 
        else:
            child1, child2 = par1, par2

        # mutation 
        # parte da mutacao do child1 e child2
        ### <--- COMPLETAR
        
        new_pop.append(child1)
        new_pop.append(child2)

    log.append(population)
    population = new_pop
    

# 4.07: max fitness

## 
from matplotlib.pyplot import plot, show, clf
plot(score)
show()


## -
xs = np.linspace(0,math.pi,num=200)
ys = [g(x) for x in xs.tolist()]

for i, gen in enumerate(log): # for each genereation
    clf()
    plot(xs, ys)
    xcand, ycand = [], []
    for chromossome in gen:
        y, x = fitness(chromossome, return_x=True)
        xcand.append(x)
        ycand.append(y)
    plot(xcand, ycand, '.')        
    show()
    if i > 10:
        break

clf()
plot(xs, ys)
xcand, ycand = [], []    
for chromossome in log[-1]:
    y, x = fitness(chromossome, return_x=True)
    xcand.append(x)
    ycand.append(y)
plot(xcand, ycand, '.')
show()
