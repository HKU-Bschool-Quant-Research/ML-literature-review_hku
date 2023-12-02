import numpy as np

class PLS:
    def __init__(self, n_components):
        self.n_components = n_components
        self.W = None
        self.P = None
        self.beta = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
    
    def fit(self, X, y):
        # 数据预处理
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X_scaled = (X - self.X_mean) / self.X_std
        
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        y_scaled = (y - self.y_mean) / self.y_std
        
        # 初始化
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, self.n_components))
        self.P = np.zeros((n_features, self.n_components))
        self.beta = np.zeros((self.n_components,))

        for i in range(self.n_components):
            # 计算权重向量
            w = X_scaled.T.dot(y_scaled)
            norm_w = np.linalg.norm(w)
            if norm_w != 0:
                w /= norm_w
            
            # 计算潜在变量
            t = X_scaled.dot(w)
            
            # 计算载荷向量
            p = X_scaled.T.dot(t)
            norm_p = np.linalg.norm(p)
            if norm_p != 0:
                p /= norm_p
            
            # 更新X和y
            X_scaled -= np.outer(t, p)
            y_scaled -= np.dot(t, self.beta[i])
            
            # 保存结果
            self.W[:, i] = w
            self.P[:, i] = p
            self.beta[i] = np.dot(t.T, y_scaled) / np.dot(t.T, t) if np.dot(t.T, t) != 0 else 0
    
    def predict(self, X):
        X_scaled = (X - self.X_mean) / self.X_std
        y_pred = np.zeros(X.shape[0])
        
        for i in range(self.n_components):
            t = X_scaled.dot(self.W[:, i])
            y_pred += self.beta[i] * t
        
        y_pred = (y_pred * self.y_std) + self.y_mean
        
        return y_pred

# 示例用法
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
y = np.array([1, 2, 3])
pls = PLS(n_components=2)
pls.fit(X, y)
y_pred = pls.predict(X)
print("预测结果：", y_pred)

# 示例用法
X_train = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
y_train = np.array([1, 2, 3])
X_test = np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]])
pls = PLS(n_components=2)
pls.fit(X_train, y_train)
y_pred = pls.predict(X_test)
print("预测结果：", y_pred)