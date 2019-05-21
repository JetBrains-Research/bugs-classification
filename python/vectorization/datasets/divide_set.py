# -*- coding: utf-8 -*-

import pickle

def write_in_file(samples, out_path):
    with open(out_path, 'wb') as out_file:
        pickle.dump(samples, out_file)

      
def divide_set(in_path, out_paths, set_ids):
    train_ids, valid_ids, test_ids = set_ids
    
    with open(in_path, 'rb') as in_file:
        train_path, valid_path, test_path = out_paths
        samples = pickle.load(in_file)
        
        write_in_file([samples[train_id] for train_id in train_ids], train_path)
        write_in_file([samples[valid_id] for valid_id in valid_ids], valid_path)
        write_in_file([samples[test_id] for test_id in test_ids], test_path)