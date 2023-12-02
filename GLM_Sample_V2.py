from patsy import dmatrix, build_design_matrices
from sklearn.linear_model import SGDRegressor

# 生成一个三节点的二次样条（25, 40, 60）
transformed_x = dmatrix("bs(train, knots=(25,40,60), degree=2, include_intercept=False)", {"train": train_x}, return_type='dataframe')

# 将二次样条转换为线性函数
linear_x = build_design_matrices([transformed_x.design_info], {"train": train_x})[0]

# 根据线性函数的设计矩阵linear_x和训练集的自变量train_x计算新的训练集的因变量train_y
new_train_y = np.dot(train_x, linear_x.T)

# 使用SGDRegressor方法计算估计参数
model = SGDRegressor(loss='huber', penalty='group', alpha=0.1, l1_ratio=0.5)
model.fit(train_x, new_train_y)

# 打印估计参数
print(model.coef_)

# 构建回归后的线性模型
def linear_model(train_x, coef):
    return np.dot(train_x, coef)

# 使用回归后的线性模型进行预测
predicted_y = linear_model(test_x, model.coef_)

# 打印预测结果
print(predicted_y)

# 将线性模型转换回二次样条形式
def transformed_model(transformed_x, linear_x, coef):
    inverse_x = transformed_x.inverse_transform(linear_x)
    return np.dot(inverse_x, coef.T)
