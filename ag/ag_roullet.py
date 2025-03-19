##NEMO RAMOS LOPES NETO
import struct
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from datetime import datetime

start_tme = os.times()
population_size = 150
mutation_rate = 0.01
pc = 0.9 ##crossover rate
L = 32
iters = 1200
version = "0.1 ROULLET WHEEL PURE"
stats = f"population: {population_size}, mutation rate: {mutation_rate}, crossover rate: {pc}"

##create population
def create_population(size):
    return np.random.uniform(0, np.pi, size).tolist()

def floatToBits ( f ) :
    s = struct.pack('>f', f)
    return struct.unpack('>L', s)[0]

def bitsToFloat ( b ) :
    s = struct.pack('>L', b)
    return struct.unpack('>f', s)[0]

# Exemplo : 1.23 ->'00010111100'
def get_bits ( x ) :
    x = floatToBits ( x )
    N = 32
    bits =''
    for bit in range ( N ) :
        b = x & (2** bit )
        bits +='1' if b > 0 else'0'
    return bits

# Exemplo :'00010111100' -> 1.23
def get_float ( bits ) :
    x = 0
    assert ( len ( bits ) == L )
    for i , bit in enumerate ( bits ) :
        bit = int ( bit ) # 0 or 1
        x += bit * (2** i )
    return bitsToFloat ( x )

## x must be >= 0 and < pi
def fitness_function(_chromosome):  #must be a float (pre conditionated)
    if 0 <= _chromosome < np.pi:
        return _chromosome + abs(np.sin(32 * _chromosome)) ##fitness function
    else:
        return 0

##get the fitness of the population
def get_fitness(population):  
    fitness_list = []
    for i in range(len(population)):
        fitness_list.append(fitness_function(population[i]))
    return fitness_list

##do the selection and crossover
def selection(population):   
    fitness_values = get_fitness(population)
    total_fitness = sum(fitness_values)
    global average_fitness
    average_fitness = total_fitness/len(population)
    probabilities = [f/total_fitness for f in fitness_values]  # roullet wheel
    couple = np.random.choice(len(population), size=2, p=probabilities)# return index
    return population[couple[0]], population[couple[1]]

def crossover(crom1, crom2):  
    global pc
    global mutation_rate
    if np.random.rand() < pc:
        locus = np.random.randint(0, L)
        child1 = crom1[:locus] + crom2[locus:]
        child2 = crom2[:locus] + crom1[locus:]
    else:
        child1, child2 = crom1, crom2  # keep the fathers
    child1 = mutation(child1)
    child2 = mutation(child2)
    return child1, child2

def mutation(crom1):    
    for i in range(L):
        if np.random.rand() < mutation_rate:
            crom1 = crom1[:i] + str(1 - int(crom1[i])) + crom1[i+1:]
    return crom1

average_fitness = 0
population = create_population(population_size)
plot_points = [] ##plt
plot_points.append(population.copy())
appt_list = []
for q in range(iters):
    try:
        population = copy.deepcopy(new_population)
    except:
        pass
    all_fitness = get_fitness(population)
    appt_list.append(average_fitness)
    new_population = []
    for i in range(population_size//2):
        crom1, crom2 = selection(population)
        crom1_bits, crom2_bits = get_bits(crom1), get_bits(crom2)
        child1, child2 = crossover(crom1_bits, crom2_bits)
        new_population.append(get_float(child1))
        new_population.append(get_float(child2))
    if q == iters//2 - 1:
        plot_points.append(new_population.copy())  ##plt
finish_time = os.times()[4]-start_tme[4]
print("time in seconds:{:.2f}".format(finish_time))  ##time for finish ag
plot_points.append(new_population.copy())
fitness_of_points = []

# LOG
try:
    converged_at  #not used in this version
except:
    converged_at = "Nan"
with open("ag/final_log.txt",'a+') as log:
    log.seek(0)  # Go to start of file
    old = log.read()  # Read existing content
    log.seek(0)  # Go back to start
    log.truncate()  # Clear the file
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    # Format the new entry with proper indentation
    new_entry = (
        "====================================\n"
        f"TIMESTAMP: {timestamp}\n"
        f"VERSION: {version}\n"
        f"Converged at generation: {converged_at}\n"
        f"Finish time: {finish_time:.2f} s\n"
        f"Stats: {stats}\n"
        "====================================\n"
    )
    
    # Write new entry followed by old content
    log.write(new_entry + old)


# Plotting section
x = np.linspace(0, np.pi, 500)
y = [fitness_function(xi) for xi in x]

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# First subplot (top-left) - Average Fitness Evolution
ax1.plot(appt_list,label = f"final Average fitness: {average_fitness:.3f}")
ax1.set_title(f'Average Fitness Evolution  - {version}')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Fitness')
ax1.legend()
ax1.grid(True)

labels = ["Initial Population", "Middle Population", "Final Population"]
colors = ["bo", "yo", "ro"]  # blue, yellow, red

# Other subplots - Population Distribution at different stages
for idx, (ax, label, color) in enumerate(zip([ax2, ax3, ax4], labels, colors)):
    # Plot fitness function
    ax.plot(x, y, 'g-', label="Fitness Function")
    # Plot population points
    ax.plot(plot_points[idx], 
            [fitness_function(xi) for xi in plot_points[idx]],
            color, 
            label=label)
    
    ax.set_title(f'Population Distribution - {label}')
    ax.set_xlabel('Input value')
    ax.set_ylabel('Fitness')
    ax.set_xlim(0, np.pi)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()