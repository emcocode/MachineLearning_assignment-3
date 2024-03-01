# Exercise 1.
See pdf for results.
The best hyperparameters are the best of the ones mentioned in the grid_parameters, see below. It is not optimized for each kernel, only a general
grid, so there may be other parameters outside this grid that are more optimal values for each individual kernel.

grid_parameters = [
    {"kernel": ["linear"], "C": [0.01, 0.1, 25, 100]},
    {"kernel": ["rbf"], "C": [0.01, 0.1, 25, 100], "gamma": [0.001, 0.01, 0.1, 1]},
    {"kernel": ["poly"], "C": [0.01, 0.1, 25, 100], "gamma": [0.001, 0.01, 0.1], "degree": [2, 3, 4, 5]}
    ]

# Exercise 3.
Part one.

The grid was tuned in to search 'C': [2, 3, 4], 'gamma': [0.021, 0.022, 0.023, 0.024]. This was done by using larger and 
more spread numbers such as 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]. When each result came in, I could iteratively 
narrow it down to smaller intervals where the diversity of the dataset could determine specifically which of the parameters were the best.

The result varies depending on data set size, but for most (smaller sets), roughly C 3-4 and gamma 0.018-0.024 seems optimal.
My final result, when tuning the hyperparameters on a 10k size dataset, is that the best hyperparameters are: C = 3, gamma = 0.023.
Using these parameters, while training using the entire train set and testing using the entire test set, we could achieve a test accuracy of over 98.5%!

---------------------------------------------------------------
Part two.

Using the best parameters, and data set sizes as in part one, the result was:
One-vs-one accuracy: 98.5%
One-vs-all accuracy: 97.6%

Confusion matrices can be viewed in the pdf.

Both ovo and ova provice very accurate predictions, however One-vs-one appears to be slightly more accurate. 
In the confusion matrices, we can see that in the ova data is often mis-predicted as 0:s. 
In ovo however, the errors are more evenly distributed.

# Exercise 4.
Results from a and b are printed when the program is being started.
There is also a plot with the 99 first models aswell as the ensemble model.

The ensemble model is of course the best model, as it uses the wisdom of the crowd, where all models have "voted" on each "pixel" in the meshgrid.
It is expected that the estimated generalization error using the test set of the ensemble should be lower than the average generalization error for
the individual decision trees.

# Exercise 5.
Part one.
Here we plot 16 random items along with their labels.

Part two.
Here we do the grid search for best parameters.
After the grid search, the best parameters I found was: a learning rate of 0.001, 4 hidden layers with 64 neurons, the hyperbolic tangent activation function 
and a regularization of 0.000001. See parameter grid below.

param_grid = {
    'learning_rate': [0.001],
    'num_hidden_layers': [4],
    'hidden_layer_size': [64],
    'activation_function': ['tanh'],
    'regularization': [0.000001]
}

These parameters produce an accuracy of 88.1-88.7%.
Similar result can be achieved using other parameters aswell.
To run the grid search, comment out this one and remove the comments on the full grid parameters just below in the code.

Part three.
Here we plot the confusion matrix of the results, see pdf.
We can see in the matrix that Trousers, Sandals, Sneakers, Bags and Ankle boots all are pretty easy to classify with well above 90% accuracy.
The most difficult classes to classify are T-shirt/top, Coat and Short, all of which are between 70-85% accuracy (usually). 
We can see that Shirts often are mis-classified as T-shirt/top, and somewhat vice versa. Sometimes it is also mitaken for a Pullover.
Pullover is also often mis-classified as Coat and vice versa.