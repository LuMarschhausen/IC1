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
def generate_training_data(N, slope, intercept, noise=False):
    X = np.random.uniform(-1, 1, (N, 2))
    y = np.array([f(x, slope, intercept) for x in X])
    if noise:
        # Inverter aleatoriamente 10% dos rótulos
        num_noise = int(0.1 * N)
        noise_indices = np.random.choice(N, num_noise, replace=False)
        y[noise_indices] = -y[noise_indices]
    return X, y

# Versão pocket do Algoritmo de Aprendizagem Perceptron (PLA)
def pocket_pla(X, y, max_iterations=1000):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X_bias.shape[1])
    best_w = w.copy()
    best_error = np.sum(np.sign(X_bias @ w) != y)
    
    for _ in range(max_iterations):
        predictions = np.sign(X_bias @ w)
        misclassified = np.where(predictions != y)[0]
        if len(misclassified) == 0:
            break
        i = np.random.choice(misclassified)
        w += y[i] * X_bias[i]
        current_error = np.sum(np.sign(X_bias @ w) != y)
        if current_error < best_error:
            best_error = current_error
            best_w = w.copy()
    
    return best_w

# Função para calcular o erro
def calculate_error(w, X, y):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    predictions = np.sign(X_bias @ w)
    error = np.mean(predictions != y)
    return error

# Parâmetros do experimento
N2 = 100
num_experiments = 1000
E_in_list = []
E_out_list = []

# Executar o experimento
for _ in range(num_experiments):
    slope, intercept = generate_target_function()
    X_train, y_train = generate_training_data(N2, slope, intercept, noise=True)
    w_pocket = pocket_pla(X_train, y_train)
    E_in = calculate_error(w_pocket, X_train, y_train)
    
    X_test, y_test = generate_training_data(N2, slope, intercept, noise=False)
    E_out = calculate_error(w_pocket, X_test, y_test)
    
    E_in_list.append(E_in)
    E_out_list.append(E_out)

# Resultados finais
average_E_in = np.mean(E_in_list)
average_E_out = np.mean(E_out_list)
print(f"Média de E_in: {average_E_in}")
print(f"Média de E_out: {average_E_out}")

# Gráficos scatterplot (um exemplo)
import matplotlib.pyplot as plt

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.7)
plt.title("Pontos de Treinamento com Ruído")
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.7)
plt.title("Pontos de Teste sem Ruído")
plt.show()
