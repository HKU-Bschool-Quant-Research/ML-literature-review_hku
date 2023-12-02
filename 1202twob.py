


# 添加二阶样条函数特征
knots = np.linspace(0, 1, 5)
bspline = sm.splines.bspline(X_train.flatten(), knots=knots, degree=2)
X_train_bspline = bspline.design_matrix
X_test_bspline = bspline(X_test.flatten())

# 构建广义线性模型
model = sm.GLM(y_train, X_train_bspline, family=sm.families.Binomial())
result = model.fit()

# 预测
y_pred = result.predict(X_test_bspline)