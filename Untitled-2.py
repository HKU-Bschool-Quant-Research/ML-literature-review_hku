
def spline_basis(X, degree=3):
n_samples, n_features = X.shape
basis_functions = []
for i in range(degree):
basis_functions.append(X ** (i + 1))
return np.concatenate(basis_functions, axis=1)