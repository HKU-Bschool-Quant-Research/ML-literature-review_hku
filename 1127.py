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
            else:
                w = np.zeros_like(w)
            
            # 计算潜在变量
            t = X_scaled.dot(w)
            
            # 计算载荷向量
            p = X_scaled.T.dot(t)
            norm_p = np.linalg.norm(p)
            if norm_p != 0:
                p /= norm_p
            else:
                p = np.zeros_like(p)
            
            # 更新X和y
            X_scaled -= np.outer(t, p)
            y_scaled -= t.dot(self)