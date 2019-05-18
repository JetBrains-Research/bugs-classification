# -*- coding: utf-8 -*-

import os
import json
import torch
import codecs
import pickle 

from helper import id2token_path, token_dir, model_save_path,\
    train_folder, valid_folder, test_folder
from batch_token_iterator import BatchTokenIterator

def write_all_tokens():
    with open(id2token_path, 'rb') as id2token_file:
        id2token = pickle.load(id2token_file)  
        with open(os.path.join(token_dir, 'all_tokens.txt'), 'w') as f:
            for ind, token in id2token.items():
                f.write(str(ind) + ' ' + str(token) + '\n')
                
def get_samples(n_samples, data_path = 'datasets/data/github_commits.dataset.jsonl'):
    with codecs.open(data_path, mode = 'r', encoding = "utf-8-sig") as data_file:
        data_rows = data_file.read().split('\n')
        json_lines = []
        n_get_samples = 0
        for row_id, row in enumerate(data_rows):
            if n_get_samples >= n_samples:
                break
            try:
                json_row = json.loads(row)
                json_lines.append(json_row)
                n_get_samples += 1
            except json.JSONDecodeError:
                pass
    prev_data_tokens = []
    upd_data_tokens = []
    for json_row in json_lines:
        prev_data_tokens.append(json_row['PrevCodeChunkTokens'])
        upd_data_tokens.append(json_row['UpdatedCodeChunkTokens'])
    return prev_data_tokens, upd_data_tokens


class Predictor(object):
    def __init__(self, dir_path):
        self.iterator = BatchTokenIterator(
            dir_path = dir_path
        ) 
        
        with open(model_save_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
            
        with open(id2token_path, 'rb') as id2token_file:
            self.id2token = pickle.load(id2token_file)
        
    def get_n_samples(self):
        return self.iterator.get_n_samples()
    
    def predict_from_data(self, ind):
        # prev.shape = updated.shape = (seq_len, 1)
        # edit.shape = (seq_olen, 1, edit_dim)
        edit, prev, updated = self.iterator.get_sample_by_id(ind)
        device = self.model.device
        
        edit_gpu = edit.to(device)
        prev_gpu = prev.to(device)
        updated_gpu = updated.to(device).contiguous()
    
        probs = self.model(edit_gpu, prev_gpu, updated_gpu)
        _, indices = torch.max(probs, 2)
        
        prev_tokens = [self.id2token[seq_id] for seq_id in prev.numpy()[:, 0]]
        updated_tokens = [self.id2token[seq_id] for seq_id in updated.numpy()[:, 0]]
        predicted_tokens = [self.id2token[seq_id] for seq_id in indices.numpy()[:, 0]]
        
        prev_code = ' '.join(prev_tokens)
        updated_code = ' '.join(updated_tokens)
        predicted_code = ' '.join(predicted_tokens)
        
        return prev_code, updated_code, predicted_code