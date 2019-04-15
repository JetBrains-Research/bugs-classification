# -*- coding: utf-8 -*-

import pickle

def write_in_file(samples, out_path):
    with open(out_path, 'wb') as out_file:
        pickle.dump(samples, out_file)

      
def divide_set(in_path, out_paths, parts):
    with open(in_path, 'rb') as in_file:
        train_path, valid_path, test_path = out_paths
        samples = pickle.load(in_file)
        n_samples = len(samples)
        train_board = int(parts[0] * n_samples)
        valid_board = int(train_board + parts[1] * n_samples)
        
        write_in_file(samples[0 : train_board], train_path)
        write_in_file(samples[train_board : valid_board], valid_path)
        write_in_file(samples[valid_board : n_samples], test_path)