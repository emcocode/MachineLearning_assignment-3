import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Read CSV
invalues_train = pd.read_csv('C:/Users/colle/Desktop/A3_datasets/mnist_train.csv') # TA BORT HARDCODING
invalues_val = pd.read_csv('C:/Users/colle/Desktop/A3_datasets/mnist_test.csv') # TA BORT HARDCODING

# Part one
# Use whole MNIST train and test set
trainX_all = invalues_train.iloc[:, 1:].values
trainY_all = invalues_train.iloc[:, 0].values
valX = invalues_val.iloc[:, 1:].values
valY = invalues_val.iloc[:, 0].values

# Smaller set, for tuning hyperparameters
trainX_hyperparameters = invalues_train.iloc[10000:13000, 1:].values
trainY_hyperparameters = invalues_train.iloc[10000:13000, 0].values

# Subset of MNIST train set, for faster use
sub_train_X = invalues_train.iloc[20000:30000, 1:].values
sub_train_Y = invalues_train.iloc[20000:30000, 0].values
sub_test_X = invalues_val.iloc[3000:5000, 1:].values
sub_test_Y = invalues_val.iloc[3000:5000, 0].values

# Tuned it down to this set, may depend on data set size though. For most (smaller sets), roughly C 3-4 and gamma 0.018-0.024 seems optimal.
# When tuning on a 10k size dataset, the best hyperparameters are: C = 3, gamma = 0.023
params = {'kernel': ['rbf'], 'C': [2, 3, 4], 'gamma': [0.021, 0.022, 0.023, 0.024]}

# Normalize
norm = MinMaxScaler()
sub_train_X = norm.fit_transform(sub_train_X)
sub_test_X = norm.transform(sub_test_X)
trainX_hyperparameters = norm.fit_transform(trainX_hyperparameters)

# Hyperparameter extraction
grid_search = GridSearchCV(estimator=SVC(), param_grid=params)
grid_search.fit(trainX_hyperparameters, trainY_hyperparameters)
print(f"Optimal hyperparameters: {grid_search.best_params_}")

# Training the model
clf = SVC(kernel='rbf', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
bestModel = clf.fit(sub_train_X, sub_train_Y)
print("Test accuracy: ", (bestModel.score(sub_test_X, sub_test_Y)))

# When tuning the hyperparameters on a 10k size dataset, the best hyperparameters are: C = 3, gamma = 0.023.
# The grid was tuned in to search 'C': [2, 3, 4], 'gamma': [0.021, 0.022, 0.023, 0.024]. This was done by using larger and 
# more spread numbers such as 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]. When each result came in, I could iteratively 
# narrow it down to smaller intervals where the diversity of the dataset could determine specifically which of the parameters were the best.


# Part two
# One-vs-one
clf = SVC(kernel='rbf', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
onevsone = OneVsOneClassifier(clf)
ovo_model = onevsone.fit(sub_train_X, sub_train_Y)
print("Accuracy for the one-vs-one:", ovo_model.score(sub_test_X, sub_test_Y))

prediction = ovo_model.predict(sub_test_X)
cm_ovo = confusion_matrix(sub_test_Y, prediction)
print("Confusion Matrix for ovo:\n", cm_ovo)

# One-vs-all
classifiers = []
for i in range(10):
    train_y_i = (sub_train_Y == i)
    clf = SVC(kernel='rbf', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
    clf.fit(sub_train_X, train_y_i)
    classifiers.append(clf)

# Make predictions
predicted_y = []
for clf in classifiers:
    y_pred_i = clf.predict(sub_test_X)
    predicted_y.append(y_pred_i)

# Combining predictions and class choosing
predicted_y = np.array(predicted_y).T
prediction_final = np.argmax(predicted_y, axis=1)

print(f"Accuracy for the one-vs-all: {accuracy_score(sub_test_Y, prediction_final)}")
cm_ova = confusion_matrix(sub_test_Y, prediction_final)
print("Confusion Matrix for ova:\n", cm_ova)


# The one-vs-one appears to be more accurate than one-vs-all. In the confusion matrix, we can see that in the ova
# data is often mis-predicted as 0:s. In ovo, the errors are more evenly distributed.