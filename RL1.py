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

# Regressão Linear para Classificação
def linear_regression(X, y):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    w = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
    return w

# Função para calcular E_in
def calculate_Ein(w, X, y):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    predictions = np.sign(X_bias @ w)
    Ein = np.mean(predictions != y)
    return Ein

# Parâmetros do experimento
N = 100
num_experiments = 1000
Ein_list = []

# Executar o experimento
for _ in range(num_experiments):
    slope, intercept = generate_target_function()
    X, y = generate_training_data(N, slope, intercept)
    w = linear_regression(X, y)
    Ein = calculate_Ein(w, X, y)
    Ein_list.append(Ein)

# Resultados finais
average_Ein = np.mean(Ein_list)
print(f"Média de E_in: {average_Ein}")
