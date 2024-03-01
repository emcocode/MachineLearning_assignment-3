import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

# Read data, but now directly using tensorflow
(trainX, trainY), (testX, testY) = keras.datasets.fashion_mnist.load_data()

# Define a list of classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Part one - plot 16 random samples
plt.figure()
for i in range(16):
    rand = np.random.randint(0, len(trainX))
    plt.subplot(4, 4, i+1)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(trainX[rand], cmap=plt.cm.binary)
    plt.xlabel(classes[trainY[rand]])


# Part two
# Normalization
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Converting the variables
num_classes = 10
trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)

# Defining the Multilayer Perceptron model
def MLP_model(learning_rate=1, num_hidden_layers=1, hidden_layer_size=1,
                 activation_function='relu', regularization=1):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
    for _ in range(num_hidden_layers):
        model.add(keras.layers.Dense(hidden_layer_size, activation=activation_function,
                               kernel_regularizer=keras.regularizers.l2(regularization)))
    
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Parameters for the grid search
# Use first grid the only run the 'optimal' that I found. Use the second for searching the grid.
param_grid = {
    'learning_rate': [0.001],
    'num_hidden_layers': [4],
    'hidden_layer_size': [64],
    'activation_function': ['tanh'],
    'regularization': [0.000001]
}
# Use grid below for grid search
# param_grid = {
#     'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#     'num_hidden_layers': [2, 3, 4, 5],
#     'hidden_layer_size': [32, 64, 128],
#     'activation_function': ['relu', 'tanh', 'sigmoid'],
#     'regularization': [0.000001, 0.00001, 0.0001, 0.001]
# }

# Create the Multilayer Perceptron model
model = KerasClassifier(model=MLP_model, activation_function='tanh', hidden_layer_size=64, learning_rate=0.001, num_hidden_layers=4, regularization=0.000001, epochs=15, batch_size=64, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Perform the grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(trainX, trainY, callbacks=[early_stopping], validation_data=(testX, testY))

# Print the best parameters and best accuracy
print("Best parameters: ", grid_result.best_params_)
print(f"Best accuracy: {grid_result.best_score_*100} %")


# Part three
# Predict and get values for confusion matrix
predictions = grid_result.predict(testX)
predictions = np.argmax(predictions, axis=1)
testY = np.argmax(testY, axis=1)
cm = confusion_matrix(testY, predictions)

# Plot the confusion matrix with numbers
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
plt.title('Confusion matrix')
plt.colorbar()
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()
