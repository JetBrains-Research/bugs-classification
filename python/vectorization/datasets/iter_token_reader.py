# -*- coding: utf-8 -*-

import pickle

EOS_TOKEN = '<EOS>'

class IterTokenReader(object):
    def __init__(self, prev_data_path, upd_data_path):
        self.paths = [prev_data_path, upd_data_path]
                
    def __iter__(self):
        with open(self.paths[0], 'rb') as prev_file:
            with open(self.paths[1], 'rb') as upd_file:
                tokens = [pickle.load(prev_file), pickle.load(upd_file)]
                
                n_samples = len(tokens[0])
                for i in range(n_samples):
                    for j in range(len(self.paths)):
                        cur_tokens = tokens[j][i]
                        cur_tokens.append(EOS_TOKEN)
                        yield cur_tokens