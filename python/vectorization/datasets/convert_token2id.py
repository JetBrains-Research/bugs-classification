# -*- coding: utf-8 -*-

import os
import pickle

AFTER_EOS = '<AFTER EOS>'

class Token2IdConverter(object):
    def __init__(self, save_dir, *token_paths):
        all_tokens = set()
        for path in token_paths:
            with open(path, 'rb') as token_file:
                samples = pickle.load(token_file)
                tokens = {token for sample in samples for token in sample}
                all_tokens.update(tokens)
                
        self.token2id = {}
        self.id2token = {}
        
        for ind, token in enumerate(all_tokens):
            self.token2id[token] = ind + 1
            self.id2token[ind + 1] = token
            
        self.token2id[AFTER_EOS] = 0
        self.id2token[0] = AFTER_EOS
                
        token2id_path = os.path.join(save_dir, 'token2id.pickle')
        with open(token2id_path, 'wb') as token2id_file:
            pickle.dump(self.token2id, token2id_file)
            
        id2token_path = os.path.join(save_dir, 'id2token.pickle')
        with open(id2token_path, 'wb') as id2token_file:
            pickle.dump(self.id2token, id2token_file)
        
    def convert(self, token_path, id_path):      
        with open(token_path, 'rb') as token_file:
            with open(id_path, 'wb') as id_file:
                tokens = pickle.load(token_file)
                ids = [[self.token2id.get(token, 0) for token in sample] for sample in tokens]
                pickle.dump(ids, id_file)