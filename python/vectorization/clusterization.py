# -*- coding: utf-8 -*-

import torch
import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from iterators import BatchVecIterator
from helper import id2token_path, model_save_path, test_folder

class Clusterizer(object): 
    def __init__(self, dir_path, n_clusters):
        self.n_clusters = n_clusters
        self.clustering = AgglomerativeClustering(
            n_clusters = n_clusters, 
            affinity='cosine', 
            linkage='single'
        )  
        self.iterator = BatchVecIterator(dir_path)
               
    def create_clusters(self, save_path):   
        data = self.iterator.get_all_samples() # shape = (seq_len, n_samples, vec_dim)
        seq_len, n_samples, vec_dim = data.size()
        data = np.transpose(data.detach().numpy(), (1, 0, 2)).reshape((n_samples, seq_len * vec_dim))
        
        self.clustering.fit_predict(data) 
        labels = np.array(self.clustering.labels_)
        mean_vecs = {}
        for cur_cluster in range(self.n_clusters):
            mask = np.array([True if label == cur_cluster else False for label in labels])
            mean_vec = np.mean(data[mask], axis = 0)
            mean_vecs[cur_cluster] = mean_vec
            
        with open(save_path, 'wb') as clusters_file:
            pickle.dump(mean_vecs, clusters_file)
        return self.clustering.labels_, mean_vecs