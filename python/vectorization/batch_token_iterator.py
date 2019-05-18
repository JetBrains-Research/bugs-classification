# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np

from helper import edit_name, prev_name, upd_name

class BatchTokenIterator(object):
    def __init__(self, dir_path, batch_size = 1):
        self.batch_size = batch_size
        self.dir_path = dir_path
        self.n_batched = -1
        
        self.edit_tokens = None
        self.prev_tokens = None
        self.upd_tokens = None
        
    def get_sample_by_id(self, ind):
        if self.edit_tokens is None:
            with open(os.path.join(self.dir_path, edit_name), 'rb') as edit_file:
                self.edit_tokens = pickle.load(edit_file)
                
        if self.prev_tokens is None:
            with open(os.path.join(self.dir_path, prev_name), 'rb') as prev_file:
                self.prev_tokens = pickle.load(prev_file)
                
        if self.upd_tokens is None:
            with open(os.path.join(self.dir_path, upd_name), 'rb') as upd_file:
                self.upd_tokens = pickle.load(upd_file)
            
        seq_len = len(self.edit_tokens[ind])
        edit = np.array(self.edit_tokens[ind], dtype = np.int64).reshape((seq_len, 1, -1))
        prev = np.array(self.prev_tokens[ind], dtype = np.int64).reshape((seq_len, 1))
        upd = np.array(self.upd_tokens[ind], dtype = np.int64).reshape((seq_len, 1))
                
        return torch.from_numpy(edit), \
            torch.from_numpy(prev), \
            torch.from_numpy(upd)
    
    def get_n_samples(self):
        if self.edit_tokens is None:
            with open(os.path.join(self.dir_path, edit_name), 'rb') as edit_file:
                self.edit_tokens = pickle.load(edit_file)
        return len(self.edit_tokens)
            
        
    def __iter__(self):        
        with open(os.path.join(self.dir_path, edit_name), 'rb') as edit_file:
            with open(os.path.join(self.dir_path, prev_name), 'rb') as prev_file:
                with open(os.path.join(self.dir_path, upd_name), 'rb') as upd_file:
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