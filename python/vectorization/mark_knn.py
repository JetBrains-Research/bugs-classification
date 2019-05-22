# -*- coding: utf-8 -*-

import os
import numpy as np

from knn import KNN
from iterators import BatchMarkedVecIterator
from metrics import multiclass_accuracy
  

plot_dir = 'classifier_plots'
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)
    
all_folder = 'datasets/data/tokens/marked/all'
train_folder = 'datasets/data/tokens/marked/train'
valid_folder = 'datasets/data/tokens/marked/valid'

save_dir = 'models'
classifier_save_path = os.path.join(save_dir, 'classifier.pt')
       
        
def train(iterator):
    
    num_folds = 5
    train_folds_X = []
    train_folds_y = []
    
    all_data_X, all_data_y = iterator.get_all_samples()
    n_samples = all_data_X.shape[0]
    fold_size = n_samples / num_folds
    
    parts = [[i * fold_size, (i + 1) * fold_size] for i in range(num_folds)]
    parts[num_folds - 1][1] = n_samples
    parts = np.array(parts, dtype = np.int32)
    
    for i in range(num_folds):
        train_folds_X.append(all_data_X[parts[i][0] : parts[i][1]])
        train_folds_y.append(all_data_y[parts[i][0] : parts[i][1]])
    
    k_choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    k_to_accuracy = {}
    
    for k in k_choices:
        acc_values = []
        for val_id in range(num_folds):
            knn_classifier = KNN(k = k)
            train_x_arrays = [train_folds_X[i] for i in range(num_folds) if i != val_id]
            train_y_vectors = [train_folds_y[i] for i in range(num_folds) if i != val_id]
            cur_train_x = np.concatenate(train_x_arrays, axis = 0)
            cur_train_y = np.concatenate(train_y_vectors, axis = 0)
            knn_classifier.fit(cur_train_x, cur_train_y)
            
            prediction = knn_classifier.predict(train_folds_X[val_id])
            accuracy = multiclass_accuracy(prediction, train_folds_y[val_id])
            acc_values.append(accuracy)
            
            print('Train fold: %i/%i' % (min(val_id + 1, num_folds), num_folds), end = '\r')
        k_to_accuracy[k] = np.mean(acc_values)
    
    for k in sorted(k_to_accuracy):
        print('k = %d, accuracy = %f' % (k, k_to_accuracy[k]))
    

if __name__ == '__main__':
    iterator = BatchMarkedVecIterator(
        dir_path = all_folder,
        batch_size = 1
    )
    
    train(iterator)
    