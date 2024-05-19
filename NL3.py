import numpy as np

# Função target f(x1, x2)
def target_function(x):
    return np.sign(x[0]**2 + x[1]**2 - 0.6)

# Função para gerar N pontos de treinamento e seus rótulos com ruído
def generate_training_data(N):
    X = np.random.uniform(-1, 1, (N, 2))
    y = np.array([target_function(x) for x in X])
    num_noise = int(0.1 * N)
    noise_indices = np.random.choice(N, num_noise, replace=False)
    y[noise_indices] = -y[noise_indices]
    return X, y

# Função para transformar os dados
def transform(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return np.c_[np.ones(X.shape[0]), x1, x2, x1*x2, x1**2, x2**2]

# Regressão Linear
def linear_regression(Z, y):
    w = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
    return w

# Função para calcular o erro
def calculate_error(w, Z, y):
    predictions = np.sign(Z @ w)
    error = np.mean(predictions != y)
    return error

# Parâmetros do experimento
N = 1000
num_experiments = 1000
E_out_list = []

# Vetor de pesos obtido anteriormente (substitua pelos valores reais após a execução do código anterior)
w = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])

# Executar o experimento para calcular E_out
for _ in range(num_experiments):
    X_test, y_test = generate_training_data(N)
    Z_test = transform(X_test)
    E_out = calculate_error(w, Z_test, y_test)
    E_out_list.append(E_out)

# Resultados finais
average_E_out = np.mean(E_out_list)
print(f"Média de E_out: {average_E_out}")
