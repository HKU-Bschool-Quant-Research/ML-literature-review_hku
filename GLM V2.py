import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures

# 1. 定义基函数
K = 5  # 选择5项样条级数作为基函数
poly = PolynomialFeatures(degree=2, include_bias=False)

# 2. 构建设计矩阵
X_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # 输入数据
X_design = poly.fit_transform(X_train)  # 计算样条级数展开的设计矩阵
y_train = np.array([2, 4, 6, 8, 10])  # 目标变量

# 3. 定义连接函数
def connect_function(X, theta):
    return np.dot(X, theta)

# 4. 定义目标函数
def objective_function(theta, X, y, alpha):
    y_pred = connect_function(X, theta)
    residuals = y - y_pred
    loss = np.sum(np.where(np.abs(residuals) < 1, 0.5 * residuals**2, np.abs(residuals) - 0.5))
    penalty = alpha * np.sum(np.abs(theta))
    return loss + penalty

# 5. 估计参数
alpha = 0.1  # 惩罚项参数
l1_ratio = 0.5  # Elastic Net中L1范数的比例

def proximal_operator(theta, step_size, alpha):
    return np.sign(theta) * np.maximum(np.abs(theta) - step_size * alpha, 0)

def accelerated_proximal_gradient_descent(X, y, alpha, l1_ratio, step_size, max_iter):
    theta = np.zeros(X.shape[1])
    theta_old = theta
    t = 1
    for _ in range(max_iter):
        gradient = compute_gradient(X, y, theta)  # 计算梯度
        theta_new = proximal_operator(theta - step_size * gradient, step_size, alpha)  # 近端算子
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2  # 更新步长
        theta = theta_new + (t - 1) / t_new * (theta_new - theta_old)  # 更新参数
        theta_old = theta_new
        t = t_new
    return theta

# 使用加速近端梯度下降法求解目标函数的最优解
step_size = 0.01  # 步长
max_iter = 1000  # 最大迭代次数
theta_estimated = accelerated_proximal_gradient_descent(X_design, y_train, alpha, l1_ratio, step_size, max_iter)

# 6. 进行预测
X_test = np.array([6, 7, 8]).reshape(-1, 1)  # 新的输入数据
X_design_test = poly.transform(X_test)  # 计算测试数据的样条级数展开的设计矩阵
y_pred = connect_function(X_design_test, theta_estimated)  # 使用估计的参数进行预测

print("预测结果:", y_pred)