from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Read CSV and randomize data
invalues = np.loadtxt('A3_datasets/bm.csv', delimiter=",")
X, y = invalues[:, :2], invalues[:, 2]
r = np.random.permutation(len(y))
X, y = X[r, :], y[r]
trainX, trainY = X[:9000, :], y[:9000]
testX, testY = X[9000:, :], y[9000:]

# Randomize training data
rng = np.random.default_rng()
n = 5000
r = np.zeros([n, 100], dtype = int)
XX = np.zeros([n, 2, 100])
for i in range (100):
    r[:, i] = rng.choice(n, size = n, replace = True)
    XX[:, :, i] = trainX[r[:, i], :]

# Train 100 trees with data
trees = [None] * 100
for i in range(100):
    trees[i] = DecisionTreeClassifier()
    trees[i].fit(XX[:, :, i], trainY[r[:, i]])

# Predict the test values
sum_accuracy = 0
results, combined_results = [], []
for tree in trees:
    prediction = tree.predict(testX)
    results.append(prediction)
    correct = sum(prediction == testY)
    accuracy = correct / len(testY)
    sum_accuracy += accuracy

# Majority vote
combined_results = []
for i in range(len(results[0])):
    votes = [result[i] for result in results]
    combined_results.append(mode(votes))

# Print a and b
accuracy = sum(combined_results == testY) / len(testY)
print(f"a) Estimated generalization error: {(1 - accuracy)*100} %")
print(f"b) Average estimated generalization error: {(1 - (sum_accuracy/100))*100} %")

# Create meshgrid
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Compute 1D array using wisdom of the crowd
def crowds_choice(row):
    return np.bincount(row.astype(int)).argmax(axis=0)

# Plot the decision boundaries
ensemble = []
for i, tree in enumerate(trees):
    plt.subplot(10, 10, i+1)
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ensemble.append(Z)
    if (i < 99): # For regular
        plt.contour(xx, yy, Z, cmap='RdYlBu', alpha=0.8)
        plt.xticks(())
        plt.yticks(())
    else: # For ensemble
        predictions = np.apply_along_axis(crowds_choice, axis=0, arr=ensemble)
        plt.contour(xx, yy, predictions, cmap='RdYlGn', alpha=0.8)
        plt.xticks(())
        plt.yticks(())

plt.show()
