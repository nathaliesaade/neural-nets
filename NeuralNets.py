
##Using Neural Nets for MNIST Digit Recognition

## I am classifying handwritten digits using raw pixels as features
## To do so, I will use a neural network with one hidden layer
## 3 layers: input layer (784+1 units), hidden layer (size 200 + 1 units) 
#and output layer (size 10 classes)
##input and hidden layer have a unit which is always equal to 1 to 
#represent bias
#for output class, each label is represented with a vector of 1 at 
#digit and 0 elsewhere

##parameters of the model are:
### W1; (n_in+1) by n_hid matrix where (i,j) entry represents the weight
# connecting the ith unit in the input layer to the jth unit in the 
#hidden layer

## W2; (n_hid+1) by n-out matrix where (i,j) entry represents the weight 
#connecting the i-th unit in the hidden layer to the j-th unit in the 
#output layer
#Note: there's an additional row for weights connecting the bias term
#to each unit in the output layer

###training with both mean-squared error and cross-entropy as loss functions


##All hidden units use tanh activation function as a choice of non-linear 
#function and the output units should use sigmoid
#use stochastic gradient descent to update weights


import numpy as np 
import scipy.io
import random
from helpers import *
from NN import *

train = scipy.io.loadmat('digit-dataset/train.mat')
test = scipy.io.loadmat('digit-dataset/test.mat')

trainDataImages, trainDataLabel = train['train_images'], train['train_labels']
testDataImages = test['test_images'] 

#training data
featureMatrixOriginal = get_feature_matrix(trainDataImages)
rowIndex = np.arange(featureMatrixOriginal.shape[0])
labelOriginal = np.reshape(trainDataLabel,(trainDataLabel.shape[0],))

#testing data
testMatrixOriginal = get_feature_matrix(testDataImages)
rowIndex_test = np.arange(testMatrixOriginal.shape[0])

ind_rand = np.random.choice(featureMatrixOriginal.shape[0], size = featureMatrixOriginal.shape[0], replace = False)
original_data = featureMatrixOriginal[ind_rand,:]
original_labels = labelOriginal[ind_rand]
original_labels_vec = np.array([vectorize(e) for e in labelOriginal])[ind_rand]

####defining training and validation sets (taking an example here of 2000 and 1000 data points)
training_data, training_labels, training_labels_vec = create_dataset(original_data, original_labels, 0, 2000)
validation_data, validation_labels, validation_labels_vec = create_dataset(original_data, original_labels, 2000, 3000)

# ##Example: Neural Network with MSE
# neural_net = NN("MSE")
# neural_train = neural_net.train(training_data, training_labels_vec, 15000)



