# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np

from helper import EDIT_NAME, PREV_NAME, UPD_NAME, edit_eos, eos_vec

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
                    
                    self.n_batches = len(edit_tokens) // self.batch_size
                    cur_id = 0
                    for _ in range(self.n_batches):
                        edit_batch = []
                        prev_batch = []
                        upd_batch = []
                        max_len = -1
                        for _ in range(self.batch_size):
                            max_len = max(max_len, len(edit_tokens[cur_id]))
                            edit_batch.append(edit_tokens[cur_id])
                            prev_batch.append(prev_tokens[cur_id])
                            upd_batch.append(upd_tokens[cur_id])
                            if cur_id == len(edit_tokens) - 1:
                                cur_id = -1
                            cur_id += 1
                        
                        for i in range(self.batch_size):
                            edit_batch[i] = np.concatenate(
                                [np.array(edit_batch[i], dtype = np.float32)
                                    , np.repeat([edit_eos], max_len - len(edit_batch[i]), axis = 0)]
                                , axis = 0
                            )
                            assert len(edit_batch[i]) == max_len

                            prev_batch[i] = np.concatenate(
                                [np.array(prev_batch[i], dtype = np.float32)
                                    , np.repeat([eos_vec], max_len - len(prev_batch[i]), axis = 0)]
                                , axis = 0
                            )
                            assert len(prev_batch[i]) == max_len
                            
                            upd_batch[i] = np.concatenate(
                                [np.array(upd_batch[i], dtype = np.float32)
                                    , np.repeat([eos_vec], max_len - len(upd_batch[i]), axis = 0)]
                                , axis = 0
                            )
                            assert len(upd_batch[i]) == max_len
                        
                        edit = np.transpose(np.array(edit_batch, dtype = np.float32), axes = (1, 0, 2))
                        prev = np.transpose(np.array(prev_batch, dtype = np.float32), axes = (1, 0, 2))
                        upd = np.transpose(np.array(upd_batch, dtype = np.float32), axes = (1, 0, 2))
                        yield torch.from_numpy(edit) \
                            , torch.from_numpy(prev) \
                            , torch.from_numpy(upd)