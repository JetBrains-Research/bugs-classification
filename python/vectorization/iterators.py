# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np

from helper import edit_name, prev_name, upd_name, mark_name, model_save_path, id2token_path, max_seq_len

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
                
        return edit, prev, upd
    
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
                         
                        yield edit_batch, prev_batch, upd_batch
                        
    def get_all_samples(self):
        n_samples = self.get_n_samples()
        orig_batch_size = self.batch_size
        self.batch_size = n_samples
        edits, prevs, upds = next(BatchTokenIterator.__iter__(self))
        self.batch_size = orig_batch_size
        return edits, prevs, upds
        
    
class BatchVecIterator(BatchTokenIterator):   
    def __init__(self, dir_path, batch_size = 1):
        BatchTokenIterator.__init__(self, dir_path, batch_size)
        
        with open(model_save_path, 'rb') as model_file:
            self.seq2seq_model = pickle.load(model_file)
            self.seq2seq_model.eval()
            
        with open(id2token_path, 'rb') as id2token_file:
            self.id2token = pickle.load(id2token_file)
            
    def get_vectors(self, edit_ids, prev_ids, upd_ids):
        # ids.shape = (seq_len, batch_size)
        seq_len, batch_size = upd_ids.shape
        vec_batch, _ = self.seq2seq_model.encoder(torch.from_numpy(edit_ids), 
                                               torch.from_numpy(prev_ids), 
                                               torch.from_numpy(upd_ids))
        return vec_batch.permute(1, 0, 2).contiguous().view(batch_size, -1)
       
    def get_sample_by_id(self, ind):
        edit_ids, prev_ids, upd_ids = BatchTokenIterator.get_sample_by_id(self, ind)
        return self.get_vectors(edit_ids, prev_ids, upd_ids)          
        
    def get_all_samples(self):
        edit_ids, prev_ids, upd_ids = BatchTokenIterator.get_all_samples(self)
        return self.get_vectors(edit_ids, prev_ids, upd_ids)
 
    
    
class BatchMarkedVecIterator(BatchVecIterator):   
    def __init__(self, dir_path, batch_size = 1):
        BatchVecIterator.__init__(self, dir_path, batch_size)
        
        with open(os.path.join(self.dir_path, mark_name), 'rb') as mark_file:
            marks_list = pickle.load(mark_file)
            self.marks = []
            for mark_list in marks_list:
                self.marks.append(mark_list)
        
       
    def get_sample_by_id(self, ind):
        vectors = BatchVecIterator.get_sample_by_id(ind)
        return vectors, self.marks[ind]        
        
    def get_all_samples(self):
        all_vectors = BatchVecIterator.get_all_samples(self)
        return all_vectors, self.marks
    
    def __iter__(self):
        with open(os.path.join(self.dir_path, edit_name), 'rb') as edit_file:
            with open(os.path.join(self.dir_path, prev_name), 'rb') as prev_file:
                with open(os.path.join(self.dir_path, upd_name), 'rb') as upd_file:
                    with open(os.path.join(self.dir_path, mark_name), 'rb') as mark_file:
                        edit_tokens = pickle.load(edit_file)
                        prev_tokens = pickle.load(prev_file)
                        upd_tokens = pickle.load(upd_file)
                        marks = pickle.load(mark_file)
                        
                        n_samples = len(edit_tokens)
                        self.n_batches = n_samples // self.batch_size
                        edit_dim = len(edit_tokens[0][0])
            
                        indices = np.arange(n_samples)
                        np.random.shuffle(indices)
                        
                        for start in range(0, n_samples, self.batch_size):
                            end = min(start + self.batch_size, n_samples)
                            
                            batch_indices = indices[start:end]
                            
                            edit_batch = np.zeros((max_seq_len, len(batch_indices), edit_dim), dtype = np.int64)
                            prev_batch = np.zeros((max_seq_len, len(batch_indices)), dtype = np.int64)
                            upd_batch = np.zeros((max_seq_len, len(batch_indices)), dtype = np.int64)
                            labels_batch = np.zeros((len(batch_indices)), dtype = np.int64)
                            
                            for batch_ind, sample_ind in enumerate(batch_indices):
                                edit_batch[:len(edit_tokens[sample_ind]), batch_ind, :] = edit_tokens[sample_ind][0 : max_seq_len]
                                prev_batch[:len(prev_tokens[sample_ind]), batch_ind] = prev_tokens[sample_ind][0 : max_seq_len]
                                upd_batch[:len(upd_tokens[sample_ind]), batch_ind] = upd_tokens[sample_ind][0 : max_seq_len]
                                labels_batch[batch_ind] = marks[sample_ind][0]
                             
                            yield self.get_vectors(edit_batch, prev_batch, upd_batch), labels_batch