import numpy as np

def huber_loss(y_true, y_pred, delta):
    residual = y_true - y_pred
    absolute_residual = np.abs(residual)
    loss = np.where(absolute_residual <= delta, residual**2, 2*delta * (absolute_residual - 0.5 * delta))
    return loss

def gradient_descent(X, y, learning_rate, num_iterations, delta):
    num_samples, num_features = X.shape
    theta = np.zeros(num_features)  # 初始化模型参数

    for _ in range(num_iterations):
        # 计算预测值
        y_pred = np.dot(X, theta)

        # 计算梯度
        residual = y - y_pred
        absolute_residual = np.abs(residual)
        gradient = np.dot(X.T, -residual * np.where(absolute_residual <= delta, 1, delta/absolute_residual)) / num_samples

        # 更新模型参数
        theta -= learning_rate * gradient

    return theta

# 准备数据
X = np.array([[1], [2], [3], [4], [5]])  # 自变量
y = np.array([30, 40, 50, 60, 70])  # 因变量

# 添加偏置项
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

# 设置超参数
learning_rate = 0.01
num_iterations = 1000
delta = 1.0

# 使用梯度下降优化模型参数
theta = gradient_descent(X, y, learning_rate, num_iterations, delta)

# 在训练集上进行预测
y_pred = np.dot(X, theta)

# 打印模型参数和预测结果
print('Model parameters:', theta)
print('Predictions:', y_pred)