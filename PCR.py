import numpy as np

# 准备数据集
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
y = np.array([1, 2, 3])

# 数据预处理
X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 主成分分析
covariance_matrix = np.cov(X_scaled.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# 选择保留的主成分数量
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1

# 选择对应的特征向量
selected_eigenvectors = eigenvectors[:, :n_components]

# 投影到所选的主成分上，得到新的特征矩阵
X_pca = X_scaled.dot(selected_eigenvectors)

# 回归分析
X_pca_with_intercept = np.concatenate((np.ones((X_pca.shape[0], 1)), X_pca), axis=1)  # 添加截距项
coefficients = np.linalg.inv(X_pca_with_intercept.T.dot(X_pca_with_intercept)).dot(X_pca_with_intercept.T).dot(y)

# 预测新样本
new_X = np.array([[13, 14, 15, 16]])
new_X_scaled = (new_X - np.mean(X, axis=0)) / np.std(X, axis=0)
new_X_pca = new_X_scaled.dot(selected_eigenvectors)
new_X_pca_with_intercept = np.concatenate((np.ones((new_X_pca.shape[0], 1)), new_X_pca), axis=1)
y_pred = new_X_pca_with_intercept.dot(coefficients)

print("预测结果：", y_pred)