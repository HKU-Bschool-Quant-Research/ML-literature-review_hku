import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor


# 生成样本数据
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=0)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 定义连接函数
def connection_function(X, theta):
    return np.dot(X, theta)

# 将样条函数定义为基函数
def spline_basis(X, knots):
    basis_functions = []
    for knot in knots:
        basis_functions.append((X - knot) ** 2)
    basis_functions.insert(0, np.ones_like(X))  # 添加常数项1
    basis_functions.insert(1, X)  # 添加一次项z
    return np.concatenate(basis_functions, axis=1)

# 使用样条基函数转换训练集和测试集
knots = [0, 1, 2, 3, 4]
X_train_basis = spline_basis(X_train,knots)
X_test_basis = spline_basis(X_test,knots)

# 定义带有Group Lasso惩罚的目标函数
def objective_function(theta, X, y, alpha, groups):
    residuals = y - connection_function(X, theta)
    loss = np.sum(residuals ** 2) / (2 * len(y))
    penalty = alpha * len(groups) * np.sum(np.sqrt(np.sum(theta**2, axis=1)))
    return loss + penalty

# 定义组信息
groups = [list(range(X_train_basis.shape[1]))] 

# 使用带有Elastic Net惩罚的SGDRegressor拟合广义线性模型
alpha = 0.1  # Group Lasso的正则化参数
l1_ratio = 0.5  # Elastic Net的L1和L2的混合比例
model = SGDRegressor(penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, tol=1e-4)
model.fit(X_train_basis, y_train)

# 获取估计的回归系数
theta_estimate = model.coef_

# 使用估计的回归系数进行预测
y_pred = connection_function(X_test_basis, theta_estimate)
print(y, y_pred)