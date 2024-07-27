import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b, *argv):
    m, n = X.shape
    total_cost = 0
    for i in range(m):
        y_hat = sigmoid(np.dot(X[i], w) + b)
        loss = -1 * (y[i] * np.log(y_hat) + (1 - y[i]) * np.log(1 - y_hat))
        total_cost += loss
    total_cost /= m
    return total_cost

def compute_gradient(X, y, w, b, *argv):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i], w) + b)
        dj_db += f_wb - y[i]
        for j in range(n):
            dj_dw[j] += (f_wb - y[i]) * X[i, j]
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    cost_without_reg = compute_cost(X, y, w, b)
    reg_cost = 0.
    for j in range(n):
        reg_cost += w[j] ** 2
    reg_cost = lambda_ * reg_cost / (2 * m)
    total_cost = cost_without_reg + reg_cost
    return total_cost

def compute_gradient_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    for j in range(n):
        dj_dw[j] += lambda_ * w[j] / m
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
    return w_in, b_in

def predict(X, w, b):
    return sigmoid(np.dot(X, w) + b)

# Import Diabetes Prediction training set
df = pd.read_csv('datasets/Diabetes_Prediction.csv')
df = df[['Glucose', 'BMI', 'Outcome']].iloc[:300, :]
X = df[['Glucose', 'BMI']].to_numpy()
y = df['Outcome'].to_numpy()
features = ['Glucose', 'BMI']
target = ['Outcome']
X_norm = StandardScaler().fit_transform(X)

plt.scatter(X_norm[:, 0], X_norm[:, 1], c = y)
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('Training Set')
plt.tight_layout()
plt.show()

X_1 = PolynomialFeatures(degree=1, include_bias=False).fit_transform(X_norm)
X_2 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_norm)
X_4 = PolynomialFeatures(degree=4, include_bias=False).fit_transform(X_norm)
X_6 = PolynomialFeatures(degree=6, include_bias=False).fit_transform(X_norm)
training_sets = [X_1, X_2, X_4, X_6]
degrees = [1, 2, 4, 6]

## Without Regularization

lambda_ = 0
w_fin = []
b_fin = []

for i in range(len(training_sets)):
    initial_w = np.zeros_like(training_sets[i][0, :])
    initial_b = 0.
    w, b = gradient_descent(training_sets[i], y, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, lambda_)
    w_fin.append(w)
    b_fin.append(b)

fig, axes = plt.subplots(1, 4, figsize = (12, 3), constrained_layout = True)

for i in range(4):
    ax = axes[i]
    ax.scatter(X_norm[:, 0], X_norm[:, 1], c=y)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title('degree of {}'.format(degrees[i]))

    x_min, x_max = X_norm[:, 0].min() - 1, X_norm[:, 0].max() + 1
    y_min, y_max = X_norm[:, 1].min() - 1, X_norm[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    poly = PolynomialFeatures(degree=degrees[i], include_bias=False)
    grid_points_poly = poly.fit_transform(grid_points)

    Z = predict(grid_points_poly, w_fin[i], b_fin[i])
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, levels=[0.5], colors=['blue'])

fig.suptitle('Without Regularization lambda = {}'.format(lambda_))
plt.show()

## With Regularization

lambda_ = 1
w_fin = []
b_fin = []

for i in range(len(training_sets)):
    initial_w = np.zeros_like(training_sets[i][0, :])
    initial_b = 0.
    w, b = gradient_descent(training_sets[i], y, initial_w, initial_b, compute_cost_reg, compute_gradient_reg, alpha, iterations, lambda_)
    w_fin.append(w)
    b_fin.append(b)

fig, axes = plt.subplots(1, 4, figsize = (12, 3), constrained_layout = True)
for i in range(4):
    ax = axes[i]
    ax.scatter(X_norm[:, 0], X_norm[:, 1], c=y)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title('degree of {}'.format(degrees[i]))

    x_min, x_max = X_norm[:, 0].min() - 1, X_norm[:, 0].max() + 1
    y_min, y_max = X_norm[:, 1].min() - 1, X_norm[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    poly = PolynomialFeatures(degree=degrees[i], include_bias=False)
    grid_points_poly = poly.fit_transform(grid_points)

    Z = predict(grid_points_poly, w_fin[i], b_fin[i])
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, levels=[0.5], colors=['blue'])

fig.suptitle('With Regularization lambda = {}'.format(lambda_))
plt.show()

## With Strong Regularization

lambda_ = 100
w_fin = []
b_fin = []

for i in range(len(training_sets)):
    initial_w = np.zeros_like(training_sets[i][0, :])
    initial_b = 0.
    w, b = gradient_descent(training_sets[i], y, initial_w, initial_b, compute_cost_reg, compute_gradient_reg, alpha, iterations, lambda_)
    w_fin.append(w)
    b_fin.append(b)

fig, axes = plt.subplots(1, 4, figsize = (12, 3), constrained_layout = True)
degrees = [1, 2, 4, 6]
for i in range(4):
    ax = axes[i]
    ax.scatter(X_norm[:, 0], X_norm[:, 1], c=y)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title('degree of {}'.format(degrees[i]))

    x_min, x_max = X_norm[:, 0].min() - 1, X_norm[:, 0].max() + 1
    y_min, y_max = X_norm[:, 1].min() - 1, X_norm[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    poly = PolynomialFeatures(degree=degrees[i], include_bias=False)
    grid_points_poly = poly.fit_transform(grid_points)

    Z = predict(grid_points_poly, w_fin[i], b_fin[i])
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, levels=[0.5], colors=['blue'])

fig.suptitle('With Strong Regularization lambda = {}'.format(lambda_))
plt.show()

# # Step 3: Logistic Regression with Logistic Loss Function and Gradient Descent
# def plot_decision_boundary(X, y, model, title, poly=None):
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                          np.arange(y_min, y_max, 0.1))
#     grid = np.c_[xx.ravel(), yy.ravel()]
#     if poly:
#         grid = poly.transform(grid)
#     Z = model.predict(grid)
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
#     plt.title(title)
#     plt.show()
#
# # Logistic regression models
# models = {}
# datasets = {
#     'Linear': (X_linear, None),
#     'Quadratic': (X_quadratic, poly_quadratic),
#     '3rd Power Polynomial': (X_third, poly_third),
#     '4th Power Polynomial' : (X_fourth, poly_fourth)
# }
# for name, (data, poly) in datasets.items():
#     model = LogisticRegression(solver='lbfgs')
#     model.fit(data, y)
#     models[name] = model
#     plot_decision_boundary(X, y, model, f'{name} Features Decision Boundary', poly)
#
# # Step 4: Logistic Regression with Regularization
# # Regularization term is included in LogisticRegression by default
# for name, (data, poly) in datasets.items():
#     model = LogisticRegression(solver='lbfgs', C=1.0)  # C is the inverse of regularization strength
#     model.fit(data, y)
#     models[name] = model
#     plot_decision_boundary(X, y, model, f'{name} Features Decision Boundary with Regularization', poly)
