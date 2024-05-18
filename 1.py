import numpy as np
import matplotlib.pyplot as plt

# Função dois pontos aleatórios e a reta target f
def generate_target_function():
    points = np.random.uniform(-1, 1, (2, 2))
    x1, y1 = points[0]
    x2, y2 = points[1]
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

# Função calcular o valor de f(x)
def f(x, slope, intercept):
    return np.sign(x[1] - (slope * x[0] + intercept))

# Função N pontos de treinamento e rótulos
def generate_training_data(N, slope, intercept):
    X = np.random.uniform(-1, 1, (N, 2))
    y = np.array([f(x, slope, intercept) for x in X])
    return X, y

# Algoritmo de Aprendizagem Perceptron (PLA)
def pla(X, y):
    w = np.zeros(3)
    iterations = 0
    X_bias = np.c_[np.ones(X.shape[0]), X]
    while True:
        predictions = np.sign(np.dot(X_bias, w))
        misclassified = np.where(predictions != y)[0]
        if len(misclassified) == 0:
            break
        i = np.random.choice(misclassified)
        w += y[i] * X_bias[i]
        iterations += 1
    return w, iterations

# Função calculo divergência entre f e g
def calculate_disagreement(w, slope, intercept):
    test_points = np.random.uniform(-1, 1, (10000, 2))
    f_labels = np.array([f(x, slope, intercept) for x in test_points])
    g_labels = np.sign(np.dot(np.c_[np.ones(test_points.shape[0]), test_points], w))
    disagreement = np.mean(f_labels != g_labels)
    return disagreement

# Parâmetros
N = 10
num_experiments = 1000
iterations_list = []
disagreement_list = []

# Executar
for _ in range(num_experiments):
    slope, intercept = generate_target_function()
    X, y = generate_training_data(N, slope, intercept)
    w, iterations = pla(X, y)
    disagreement = calculate_disagreement(w, slope, intercept)
    iterations_list.append(iterations)
    disagreement_list.append(disagreement)

# Resultados
average_iterations = np.mean(iterations_list)
average_disagreement = np.mean(disagreement_list)

print(f"Média de iterações para convergir: {average_iterations}")
print(f"Média de divergência P[f(x) ≠ g(x)]: {average_disagreement}")

# Plotar um exemplo
slope, intercept = generate_target_function()
X, y = generate_training_data(N, slope, intercept)
w, iterations = pla(X, y)

# Plot pontos e das retas
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
x_vals = np.linspace(-1, 1, 100)
target_line = slope * x_vals + intercept
pla_line = -(w[1] / w[2]) * x_vals - (w[0] / w[2])
plt.plot(x_vals, target_line, 'g-', label='Target function f')
plt.plot(x_vals, pla_line, 'b-', label='Hypothesis g')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.legend()
plt.show()
