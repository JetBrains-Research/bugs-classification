# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

from token_dataset import TokenDataset
from train_token2vec import TokenVecTrain
from convert_token2id import Token2IdConverter, convert_from_exists
from divide_set import divide_set

tokens_dir = 'data/tokens'

all_dir = os.path.join(tokens_dir, 'all')
if not os.path.exists(all_dir):
    os.makedirs(all_dir)
train_dir = os.path.join(tokens_dir, 'train')
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
valid_dir = os.path.join(tokens_dir, 'valid')
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)
test_dir = os.path.join(tokens_dir, 'test')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

edit_name = 'edits.pickle'
prev_name = 'prevs.pickle'
upd_name = 'updates.pickle'
mark_name = 'marks.pickle'

src_data_path = 'data/deserialization_marked_commits.dataset.jsonl'

prev_tokens_path = os.path.join(tokens_dir, 'prev_tokens.pickle')
upd_tokens_path = os.path.join(tokens_dir, 'upd_tokens.pickle')

mark_list_path = os.path.join('data/marks.txt')

w2v_path = os.path.join(tokens_dir, 'word2vec')
w2v_vecs_path = os.path.join(tokens_dir, 'word2vec_keyed_vectors')

dataset_parts = [0.8, 0.15]


'''
load data from jsonl format and divide them into previous and updated tokens, 
and edit vectors that is standart sequence alignment representation
'''
def create_token_datasets():
    print('Create token datasets')
    dataset = TokenDataset(
        data_path = src_data_path,
        edits_path = os.path.join(all_dir, edit_name),
        prevs_path = prev_tokens_path,
        upds_path = upd_tokens_path,
        mark_out_path = os.path.join(all_dir, mark_name), 
        mark_in_path = mark_list_path
    )
    dataset.prepare_token_dataset()
 
'''
load tokens from prev_tokens_path and upd_tokens_path and train gensim.Word2Vec on it
'''
def train_word2vec():
    print('Create and train token2vec')
    vocab = TokenVecTrain(
        dim = 50
        , min_count = 1
        , model_save_path = w2v_path
        , vectors_save_path = w2v_vecs_path
        , prev_data_path = prev_tokens_path
        , upd_data_path = upd_tokens_path
        , iter = 10
    )
    vocab.train(n_epochs = 100)
 
'''
convert tokens from prev_tokens_path and upd_tokens_path to vectors according word2vec model
'''
def convert_tokens_to_ids():
    print('Convert tokens to vectors')
    converter = Token2IdConverter(tokens_dir, prev_tokens_path, upd_tokens_path)    
    converter.convert(prev_tokens_path, os.path.join(all_dir, prev_name))
    converter.convert(upd_tokens_path, os.path.join(all_dir, upd_name))
    
def convert_tokens_to_ids_from_exist():
    print('Convert tokens to vectors from exist')
    token2id_path = os.path.join(tokens_dir, 'token2id.pickle')
    with open(token2id_path, 'rb') as token2id_file:
        token2id = pickle.load(token2id_file)
        
    convert_from_exists(token2id, prev_tokens_path, os.path.join(all_dir, prev_name))
    convert_from_exists(token2id, upd_tokens_path, os.path.join(all_dir, upd_name))

'''
divide dataset into 3 parts: train, valid and test
'''
def divide_dataset():
    print('Divide dataset (train, valid, test)')
    
    def run_division(name, train_ids, valid_ids, test_ids): 
        divide_set( 
            in_path = os.path.join(all_dir, name), 
            out_paths = [os.path.join(train_dir, name), os.path.join(valid_dir, name), os.path.join(test_dir, name)],
            set_ids = [train_ids, valid_ids, test_ids]
        ) 

    with open(os.path.join(all_dir, edit_name), 'rb') as edit_file: 
        n_samples = len(pickle.load(edit_file)) 
    indices = np.arange(n_samples) 
    np.random.shuffle(indices) 

    train_board = int(dataset_parts[0] * n_samples) 
    valid_board = int(train_board + dataset_parts[1] * n_samples) 
    
    train_ids = indices[0 : train_board] 
    valid_ids = indices[train_board : valid_board] 
    test_ids = indices[valid_board : n_samples] 
    
    run_division(edit_name, train_ids, valid_ids, test_ids) 
    run_division(prev_name, train_ids, valid_ids, test_ids) 
    run_division(upd_name, train_ids, valid_ids, test_ids)
    if mark_list_path is not None:
        run_division(mark_name, train_ids, valid_ids, test_ids)

if __name__ == "__main__":      
    create_token_datasets()
    train_word2vec()
    convert_tokens_to_ids()
    divide_dataset()