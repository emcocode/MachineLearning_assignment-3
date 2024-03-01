import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Read CSV
invalues = pd.read_csv('A3_datasets/dist.csv', delimiter=";", header=None)
X = invalues.iloc[:, :2].values
y = invalues.iloc[:, 2].values

# Decision boundary method
def decisionBoundary(x):
    if (x > 3.94):
        return (0.5*(18 - 2*x - np.sqrt(-724 + 256*x - 16*x**2)))
    else:
        return (0.071*(174 - 22*x - np.sqrt(23123 - 6144*x + 288*x**2)))

# Grid search function
def gridSearch(grid_parameters):
    SVM_model = SVC()
    grid_search = GridSearchCV(estimator=SVM_model, param_grid=grid_parameters)
    grid_search.fit(X, y)
    return grid_search

# Print score
def printScore(grid_search):
    best_model = grid_search.best_estimator_
    val_score = best_model.score(X_val, y_val)
    print("Best hyperparameters:", grid_search.best_params_)
    print("\tTraining score:", grid_search.best_score_)
    print("\tValidation score:", val_score)

# 1. Tune hyperparameters
dist_val = pd.read_csv('A3_datasets/dist_val.csv', delimiter=";", header=None)
X_val = dist_val.iloc[:, :2].values
y_val = dist_val.iloc[:, 2].values

grid_parameters = [
    {"kernel": ["linear"], "C": [0.01, 0.1, 25, 100]},
    {"kernel": ["rbf"], "C": [0.01, 0.1, 25, 100], "gamma": [0.001, 0.01, 0.1, 1]},
    {"kernel": ["poly"], "C": [0.01, 0.1, 25, 100], "gamma": [0.001, 0.01, 0.1], "degree": [2, 3, 4, 5]}
    ]

grid_search = gridSearch(grid_parameters)
print("Best overall:")
printScore(grid_search)
print("------------------------------------------------------------------------------")


# 2. Plot
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
x_Range = np.linspace(-2, 6, 1000)

for i in range(3):
    plt.subplot(1, 3, (i+1))
    plt.plot(x_Range, [decisionBoundary(i) for i in x_Range], color='k', label='True decision bound')
    plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='o', s=15, alpha=0.8, label='0')
    plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', s=15, alpha=0.8, label='1')
    plt.xlim(-2, 6)
    plt.ylim(-4, 5)

    grid_search = gridSearch(grid_parameters[i])
    Z = grid_search.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap='inferno')
    plt.title(f"kernel: {grid_search.best_params_['kernel']}")
    plt.legend(loc="upper right")
    print(f"{grid_search.best_params_['kernel']}:")
    printScore(grid_search)

plt.show()