
import numpy as np
def get_feature_matrix(images):
    """create the feature matrix form the images"""
    featureMatrix = np.zeros((images.shape[2],images.shape[0]*images.shape[1]))
    for i in range(0,images.shape[2]):
        featureMatrix[i,:] = np.concatenate(images[:,:,i], axis=0)
        
    return featureMatrix

def create_dataset(original_data, original_labels, start_index, end_index):
    """
    Creates dataset using original data and original labels choosing the index range from the start index
    to the end index and returns the dataset of images, labels and vectorized labels
    """
    data = original_data[start_index:end_index, :]
    labels = original_labels[start_index: end_index]
    labels_vec = np.array([vectorize(e) for e in labels])[start_index: end_index]
    return data, labels, labels_vec


def vectorize(x):
    """Vectorizes the labels from a single digit to a vector of size 10
    taking a value of 1 at the corresponding digit and 0 elsewhere"""
    e = np.zeros((10,1))
    e[x] = 1
    e = e.ravel().astype(int)
    return e
 

