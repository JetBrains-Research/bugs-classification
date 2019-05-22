# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter

class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        dists = self.compute_distances_no_loops(X)
            
        return self.predict_labels(dists)

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        test_x_new = np.tile(X[:, np.newaxis, :], (1, num_train, 1))
        train_x_new = np.tile(self.train_X[np.newaxis, :, :], (num_test, 1, 1))   
        dists = np.sum(np.abs(test_x_new - train_x_new), axis = 2)
        return dists
    
    def predict_labels(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, self.train_y.dtype)
        sorted_ids = np.argsort(dists)
        for i in range(num_test):
            nearest_neighbors = sorted_ids[i][0 : self.k]
            nearest_labels = self.train_y[nearest_neighbors]
            pred[i] = Counter(nearest_labels).most_common(1)[0][0]
        return pred

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        return self.predict_labels(dists)

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        return self.predict_labels(dists)
