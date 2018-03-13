#!/usr/bin/python3

### OBJECTIVES
# Learn fundamental TensorFlow concepts
# Use the LinearRegressor class to predict median housing prices
# Evaluate the accuracy of a model's prediction using Root Mean Square Error (RMSE)
# Improve the accuract of the model by tuning it's hyperparameters (ie: Learning Rate)

# load necessary libraries
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# settings/options
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# load the dataset
california_housing_dataframe = pd.read_csv('data/data.csv', sep=',')

# randomize the data in order to prevent any effects that might harm the
# performance of Stochastic Gradient Descent which relies on just ONE example
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

# scaling median_house_value to be in units of 1000 so it can be learned
# a little more easily with learning rates in a range that is usually used
# ex: $100,000 house is now $100
california_housing_dataframe['median_house_value'] /= 1000.0

# examine the data
#print(california_housing_dataframe.describe())

### BUILDING THE FIRST MODEL
# try to predict median_house_value. This is our label (aka 'target').
# we'll use total_rooms as our input feature
# note: our data is at city block level, so total_rooms represents the total number of rooms in that block

# STEP 1: define features and configure feature columns
# Categorical data: textual data like "home style"
# Numerical data: numbers like "total rooms"

# define the numeric input feature: total_rooms
my_feature = california_housing_dataframe[['total_rooms']]

# configure a numeric feature column for total_rooms
feature_columns = [tf.feature_column.numeric_column('total_rooms')]

# STEP 2: define the target
targets = california_housing_dataframe['median_house_value']

# STEP 3: Configure the LinearRegressor
# LinearRegressor is a class in TensorFlow used for predicting targets
# We'll train the model using GradientDescentOptimizer which implements Stochastic Gradient Descent (SGD)

# The learning_rate argument controls the size of the gradient step
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)

# clip_gradients_by_norm ensures the magnitude of the gradients do not become too large during training
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# define the LinearRegressor
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

# STEP 4: Define the Input Function
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
    
    Args:
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle: True or False. Whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated. None =
        indefinitely.
        Returns: tuple of (feature, labels) for next data batch
        """
    # convert pandas data into a dict of numpy arrays
    features = {key:np.array(value) for key, value in dict(features).items()}

    # construct a dataset and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return (features, labels)

### STEP 5: Train the Model
_ = linear_regressor.train(input_fn = lambda:my_input_fn(my_feature, targets), steps=100)

### STEP 6: Evaluate the Model

# create an input function for predictions.
# since we're only making one prediction, we don't need to shuffle
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# call predict() on the linear_regressor to make predictions
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# format predictions as a NumPy array, so we can calculate error metrics
predictions = np.array([item['predictions'][0] for item in predictions])

# print Mean Squared Error and Root Mean Error
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)

# Let's compare the RMSE to the difference of the min and max of our targets
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value
print('Min. Median House Value: {:.3f}'.format(min_house_value))
print('Max. Median House Value: {:.3f}'.format(max_house_value))
print('Difference: {:.3f}'.format(min_max_difference)) # ~ 485
print('Root Mean Squared Error (on training data):{:.3f}'.format(root_mean_squared_error)) # ~ 250

# note: the RMSE is 250, the actual difference is 485.
# this means our error spans nearly have the range of the target values (not very good)
# can we do better?

calibration_data = pd.DataFrame()
calibration_data['predictions'] = pd.Series(predictions)
calibration_data['targets'] = pd.Series(targets)
print(calibration_data.describe())

# let's create a readable scatterplot

# sampling 300 examples
sample = california_housing_dataframe.sample(n=300)

# get the min and max total_rooms from the sample
x_0 = sample['total_rooms'].min()
x_1 = sample['total_rooms'].max()

# receive the final weight and bias generated during training
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# get the predicted median_house_value for the min and max total_rooms values
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# plot the regression line
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# label the graph axis
plt.xlabel('median_house_value')
plt.ylabel('total_rooms')

# plot a scatter plot from our data sample
plt.scatter(sample['total_rooms'], sample['median_house_value'])

# display the graph
plt.show()
