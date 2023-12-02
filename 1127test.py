import numpy as np

# 生成数据集
np.random.seed(0)
n_samples = 100
n_features = 10
X = np.random.randn(n_samples, n_features)
coef = np.random.randn(n_features)
y = np.dot(X, coef) + np.random.randn(n_samples) * 0.1

# 数据集拆分
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# 定义近端算子和目标函数的梯度
def prox_l1(beta, lambda_, alpha):
    return np.sign(beta) * np.maximum(np.abs(beta) - lambda_ * alpha, 0) / (1 + lambda_ * (1 - alpha))

def prox_l2(beta, lambda_, alpha):
    return beta / (1 + lambda_ * (1 - alpha))

def prox_elastic_net(beta, lambda_, alpha, learning_rate):
    return (1 / (1 + lambda_ * (1 - alpha))) * prox_l1(beta - learning_rate * (1 + lambda_ * (1 - alpha)) * gradient(X_train, y_train, beta), lambda_, alpha)

def objective_function(X, y, beta, lambda_, alpha):
    n = X.shape[0]
    residuals = y - np.dot(X, beta)
    mse_loss = np.sum(residuals ** 2) / n
    l1_penalty = np.sum(np.abs(beta))
    l2_penalty = np.sum(beta ** 2)
    return mse_loss + lambda_ * (alpha * l1_penalty + (1 - alpha) * l2_penalty)

def gradient(X, y, beta):
    n = X.shape[0]
    residuals = y - np.dot(X, beta)
    grad = -2 * np.dot(X.T, residuals) / n
    return grad

# 定义近端梯度算法
def proximal_gradient(X, y, lambda_, alpha, learning_rate, max_iter):
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    
    for _ in range(max_iter):
        grad = gradient(X, y, beta)
        beta = prox_elastic_net(beta - learning_rate * grad, lambda_, alpha, learning_rate)
    
    return beta

# 训练模型
lambda_ = 0.1
alpha = 0.5
learning_rate = 0.01
max_iter = 100

n_samples, n_features = X_train.shape
beta = np.zeros(n_features)

for _ in range(max_iter):
    residuals = y_train - np.dot(X_train, beta)
    grad = -2 * np.dot(X_train.T, residuals) / n_samples
    beta = prox_elastic_net(beta - learning_rate * grad, lambda_, alpha, learning_rate)

# 模型评估
y_pred = np.dot(X_test, beta)
mse = np.mean((y_test - y_pred) ** 2)
print("Mean Squared Error:", mse)