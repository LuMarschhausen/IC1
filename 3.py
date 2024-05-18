import numpy as np

# Função para gerar dois pontos aleatórios e a reta target f
def generate_target_function():
    points = np.random.uniform(-1, 1, (2, 2))
    x1, y1 = points[0]
    x2, y2 = points[1]
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

# Função para calcular o valor de f(x)
def f(x, slope, intercept):
    return np.sign(x[1] - (slope * x[0] + intercept))

# Função para gerar N pontos de treinamento e seus rótulos
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

# Parâmetros do experimento
N = 100
num_experiments = 1000
iterations_list = []

# Executar o experimento
for _ in range(num_experiments):
    slope, intercept = generate_target_function()
    X, y = generate_training_data(N, slope, intercept)
    _, iterations = pla(X, y)
    iterations_list.append(iterations)

# Resultados finais
average_iterations = np.mean(iterations_list)
print(f"Média de iterações para convergir: {average_iterations}")
