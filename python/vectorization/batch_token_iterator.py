# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np

from helper import EDIT_NAME, PREV_NAME, UPD_NAME

class BatchTokenIterator(object):
    def __init__(self, dir_path, batch_size = 1):
        self.batch_size = batch_size
        self.dir_path = dir_path
        self.n_batched = -1
        
    def __iter__(self):        
        with open(os.path.join(self.dir_path, EDIT_NAME), 'rb') as edit_file:
            with open(os.path.join(self.dir_path, PREV_NAME), 'rb') as prev_file:
                with open(os.path.join(self.dir_path, UPD_NAME), 'rb') as upd_file:
                    edit_tokens = pickle.load(edit_file)
                    prev_tokens = pickle.load(prev_file)
                    upd_tokens = pickle.load(upd_file)
                    
                    n_samples = len(edit_tokens)
                    self.n_batches = n_samples // self.batch_size
                    edit_dim = len(edit_tokens[0][0])
        
                    indices = np.arange(n_samples)
                    np.random.shuffle(indices)
                    
                    for start in range(0, n_samples, self.batch_size):
                        end = min(start + self.batch_size, n_samples)
                        
                        batch_indices = indices[start:end]
                        
                        max_len = max(len(edit_tokens[ind]) for ind in batch_indices)
                        edit_batch = np.zeros((max_len, len(batch_indices), edit_dim), dtype = np.int64)
                        prev_batch = np.zeros((max_len, len(batch_indices)), dtype = np.int64)
                        upd_batch = np.zeros((max_len, len(batch_indices)), dtype = np.int64)
                        
                        for batch_ind, sample_ind in enumerate(batch_indices):
                            edit_batch[:len(edit_tokens[sample_ind]), batch_ind, :] = edit_tokens[sample_ind]
                            prev_batch[:len(prev_tokens[sample_ind]), batch_ind] = prev_tokens[sample_ind]
                            upd_batch[:len(upd_tokens[sample_ind]), batch_ind] = upd_tokens[sample_ind]
                         
                        yield torch.from_numpy(edit_batch) \
                            , torch.from_numpy(prev_batch) \
                            , torch.from_numpy(upd_batch)