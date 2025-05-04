import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, vlines

# Dados
mean0, std0 = -0.4, 0.5
mean1, std1 = 0.9, 0.3
m = 200

x1s = np.random.randn(m // 2) * std1 + mean1
x0s = np.random.randn(m // 2) * std0 + mean0

xs = np.hstack((x1s, x0s))
ys = np.hstack((np.ones(m // 2), np.zeros(m // 2)))

plot(xs[:m // 2], ys[:m // 2], '.')
plot(xs[m // 2:], ys[m // 2:], '.')
show()

# Funções auxiliares
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(x, _theta):
    return sigmoid(_theta[0] + _theta[1] * x)

def cost(h, y):
    return -y * np.log(h) - (1 - y) * np.log(1 - h)

def J(theta, xs, ys):
    m = len(xs)
    total_cost = 0
    for i in range(m):
        total_cost += cost(h(xs[i], theta), ys[i])
    return total_cost / m

def gradient(i, theta, _xs, _ys):
    dif = h(_xs[i], theta) - _ys[i]
    return np.array([dif, dif * _xs[i]])

def accuracy(ys, predictions):
    num = sum(ys == predictions)
    return num / len(ys)

def plot_fronteira(theta):
    # Plota a fronteira de decisão
    x_boundary = -theta[0] / theta[1]
    vlines(x_boundary, ymin=0, ymax=1, colors='red', linestyles='dashed', label='Fronteira de decisão')
    plt.legend()

# Hiperparâmetros
alpha = 0.1  # Taxa de aprendizado
epochs = 2000
theta = np.array([1, 5.0])  # Inicialização dos parâmetros

# Armazenar métricas
costs = []
accuracies = []

# Treinamento
for k in range(epochs):
    sum_g = 0
    for train_points in range(m):
        sum_g += gradient(train_points, theta, xs, ys)
    theta -= alpha * sum_g / m  
    costs.append(J(theta, xs, ys))  
    predictions = (h(xs, theta) >= 0.5).astype(int)
    accuracies.append(accuracy(ys, predictions)) 

print("theta:", theta)

# Predições finais
predictions = (h(xs, theta) >= 0.5).astype(int)
print("Acurácia final:", accuracy(ys, predictions))

# Plots
plt.figure(figsize=(12, 4))

# Plot da função de custo
plt.subplot(1, 3, 1)
plt.plot(range(epochs), costs, label='Custo')
plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.title('Função de Custo')
plt.legend()

# Plot da acurácia
plt.subplot(1, 3, 2)
plt.plot(range(epochs), accuracies, label='Acurácia', color='green')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.title('Acurácia ao longo do treinamento')
plt.legend()

# Plot dos dados e da fronteira de decisão
plt.subplot(1, 3, 3)
plot(xs[:m // 2], ys[:m // 2], '.', label='Classe 1')
plot(xs[m // 2:], ys[m // 2:], '.', label='Classe 0')
plot_fronteira(theta)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fronteira de Decisão')
plt.legend()

plt.tight_layout()
plt.show()

# Testar diferentes taxas de aprendizado
for alpha_test in [0.01, 0.1, 1.0]:
    theta = np.array([1, 5.0])  
    costs = []
    for k in range(epochs):
        sum_g = 0
        for train_points in range(m):
            sum_g += gradient(train_points, theta, xs, ys)
        theta -= alpha_test * sum_g / m
        costs.append(J(theta, xs, ys))
    print(f"Taxa de aprendizado: {alpha_test}, theta final: {theta}")
    plt.plot(range(epochs), costs, label=f'alpha={alpha_test}')

plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.title('Impacto da Taxa de Aprendizado')
plt.legend()
plt.show()

## b) A acurácia pode oscilar durante o treinamento, o gradient * learning rate pode ultrapassar
## o ponto ótimo. Para 100% seria preciso um modelo de maior grau.
## d) Os resultados com maior learning_rate convergiram mais rápido