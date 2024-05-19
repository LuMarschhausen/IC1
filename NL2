import numpy as np

# Função target f(x1, x2)
def target_function(x):
    return np.sign(x[0]**2 + x[1]**2 - 0.6)

# Função para gerar N pontos de treinamento e seus rótulos
def generate_training_data(N):
    X = np.random.uniform(-1, 1, (N, 2))
    y = np.array([target_function(x) for x in X])
    # Adicionar ruído a 10% dos pontos
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
E_in_list = []

# Executar o experimento
for _ in range(num_experiments):
    X_train, y_train = generate_training_data(N)
    Z_train = transform(X_train)
    w = linear_regression(Z_train, y_train)
    E_in = calculate_error(w, Z_train, y_train)
    E_in_list.append(E_in)

# Resultados finais
average_E_in = np.mean(E_in_list)
print(f"Média de E_in: {average_E_in}")
print(f"Vetor de pesos w: {w}")
