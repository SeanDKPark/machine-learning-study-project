import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# df = pd.read_csv('datasets/50_Startups.csv')
# df = df.drop(columns = ['State'])
df =pd.read_csv('datasets/Student_Performance.csv')
df = df.iloc[:1000, :].drop(columns = ['Extracurricular Activities'])
cols = list(df.columns)
df_x = df.iloc[:, :-1]
df_y = df.iloc[:, -1]

X_train = df_x.to_numpy()
y_train = df_y.to_numpy()
X_features = cols[:-1]
y_target = cols[-1]
m, n = X_train.shape

fig, ax = plt.subplots(1, n, figsize = (12, 3), sharey = True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, s = 10)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel(y_target)
plt.tight_layout()
plt.show()

def compute_cost(X, y , w, b):
    '''
    :param X: (m, n-1)
    :param y: (m, )
    :param w: (n, )
    :param b: (scalar)
    :return: (scalar) cost at given param and train set
    '''
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        y_hat_i = np.dot(X[i], w) + b
        cost += (y_hat_i - y[i]) ** 2
    cost /= (2*m)
    return cost

def compute_gradient(X, y, w, b):
    '''
    :param X: (m, n)
    :param y: (m, 1)
    :param w: (n, )
    :param b: (scalar)
    :return:
        dj_dw: (n, )
        dj_dw: (1, )
    '''
    m, n = X.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.0
    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    cost_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost_history.append(cost_function(X,y,w,b))
    return w, b, cost_history

w_in = np.zeros((n, ))
b_in = 150000

iter = 100
alpha = 3.0e-5

## Run Algorithm
w_final, b_final, cost_history = gradient_descent(X_train, y_train, w_in, b_in, compute_cost, compute_gradient, alpha, iter)
print('Main Case')
print(f"Final parameters b, w: {b_final:0.2f},{w_final} ")
print(f"Final Cost: {cost_history[-1]:0.8f}")

plt.plot(cost_history)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Learning Curve')
plt.tight_layout()
plt.show()

# Slower version using for loop instead of np.dot
def compute_cost_serial(X, y, w, b):
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        p = 0
        for j in range(n):
            p += X[j]*w[j]
        y_hat_i = p + b
        cost += (y_hat_i - y[i]) ** 2
    cost /= (2 * m)
    return cost

def compute_gradient_serial(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.0
    for i in range(m):
        p = 0
        for j in range(n):
            p += X[i, j]*w[j]
        error = (p + b) - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# Parallel Computing
start_time = time.time()
w_final, b_final, cost_history = gradient_descent(X_train, y_train, w_in, b_in, compute_cost, compute_gradient, alpha, iter)
end_time = time.time()
elapsed_time_func1 = end_time - start_time
print(f"Time elapsed for parallel computing: {elapsed_time_func1:.6f} seconds")

# Serial Computing
start_time = time.time()
w_final, b_final, cost_history = gradient_descent(X_train, y_train, w_in, b_in, compute_cost_serial, compute_gradient_serial, alpha, iter)
end_time = time.time()
elapsed_time_func1 = end_time - start_time
print(f"Time elapsed for serial computing: {elapsed_time_func1:.6f} seconds")


# Feature Scaling

## Process
mu = np.mean(X_train,axis=0)
sigma = np.std(X_train,axis=0)
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,1], X_train[:,2])
ax[0].set_xlabel(X_features[1]); ax[0].set_ylabel(X_features[2]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,1])
ax[1].set_xlabel(X_features[1]); ax[0].set_ylabel(X_features[2]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,1])
ax[2].set_xlabel(X_features[1]); ax[0].set_ylabel(X_features[2]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()

## Compare Distribution
def zscore_fs(X):
    mu = np.mean(X, axis = 0)
    sigma = np.mean(X, axis = 0)
    X_norm = (X-mu) / sigma
    return X_norm, mu, sigma

# normalize the original features
X_norm, X_mu, X_sigma = zscore_fs(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

def plot_histogram_with_normal(ax, data):
    """
    Plot a histogram of the data and a normal distribution curve on the given Axes object.

    Parameters:
    ax (matplotlib.axes.Axes): The Axes object to plot on.
    data (array-like): The 1D array of data to plot.
    """
    # Calculate the mean and standard deviation of the data
    mu, sigma = np.mean(data), np.std(data)

    # Plot the histogram of the data
    ax.hist(data, bins=30, density=True, alpha=0.6, label='Histogram')

    # Plot the normal distribution curve
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    ax.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
    return

fig,ax=plt.subplots(1, n, figsize=(12, 3))
for i in range(len(ax)):
    plot_histogram_with_normal(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.tight_layout()
plt.show()

fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    plot_histogram_with_normal(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features after normalization")
plt.tight_layout()
plt.show()

## Run
w_zs, b_zs, cost_history_zs = gradient_descent(X_train, y_train, w_in, b_in, compute_cost, compute_gradient, alpha, iter)
print('Z-Score Feature Scaled Case')
print(f"Final parameters b, w: {b_zs:0.2f},{w_zs} ")
print(f"Final Cost: {cost_history_zs[-1]:0.8f}")

plt.plot(cost_history_zs)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Learning Curve')
plt.tight_layout()
plt.show()
