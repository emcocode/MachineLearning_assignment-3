import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 1. Read CSV
invalues = np.loadtxt('A3_datasets/bm.csv', delimiter=",")
X, y = invalues[:, :2], invalues[:, 2]
n_s = 5000
np.random.seed(7)
r = np.random.permutation(len(y))
X, y = X[r, :], y[r] # Shufflar
X_s, y_s = X[:n_s, :], y[: n_s] # Väljer de n_s första, dvs skapar datasettet på 5k punkter


# 2. SVM with Gaussian kernel
clf = SVC(kernel='rbf', gamma=0.5, C=20) # Förbättra
clf.fit(X_s, y_s)

y_Prediction = clf.predict(X_s)
# plt.scatter(X_s[:, 0], X_s[:, 1], y_Prediction)

train_Error = (1 - clf.fit(X_s, y_s).score(X_s, y_s))
print("Training error:", train_Error)


# 3. Plots
x_min, x_max = X_s[:, 0].min() - 0.2, X_s[:, 0].max() + 0.2
y_min, y_max = X_s[:, 1].min() - 0.2, X_s[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# SV and decision boundary
plt.subplot(1, 2, 1)
supportVector = clf.support_
plt.scatter(X_s[supportVector, 0], X_s[supportVector, 1], c='b', marker='o', s=8, alpha=0.4)
plt.contour(xx, yy, Z, cmap='inferno')

# Decision boundary and data
plt.subplot(1, 2, 2)
plt.scatter(X_s[:, 0], X_s[:, 1], c=-y_s, marker='o', s=8, alpha=0.8)
plt.contour(xx, yy, Z, cmap='inferno')

plt.show()