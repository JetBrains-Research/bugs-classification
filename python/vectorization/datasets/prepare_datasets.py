# -*- coding: utf-8 -*-

import os

from token_dataset import TokenDataset
from train_token2vec import TokenVecTrain
from convert_token2vec import Token2VecConverter
from divide_set import divide_set

all_dir = 'data/tokens/all'
if not os.path.exists(all_dir):
    os.makedirs(all_dir)
train_dir = 'data/tokens/train'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
valid_dir = 'data/tokens/valid'
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)
test_dir = 'data/tokens/test'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

edit_name = 'edits.pickle'
prev_name = 'prevs.pickle'
upd_name = 'updates.pickle'

src_data_path = 'data/github_commits.dataset.jsonl'

prev_tokens_path = 'data/tokens/prev_tokens.pickle'
upd_tokens_path = 'data/tokens/upd_tokens.pickle'

w2v_path = 'data/tokens/word2vec'
w2v_vecs_path = 'data/tokens/word2vec_keyed_vectors'

dataset_parts = [0.8, 0.15]

'''
load data from jsonl format and divide them into previous and updated tokens, 
and edit vectors that is standart sequence alignment representation
'''
def create_token_datasets():
    print('Create token datasets')
    dataset = TokenDataset(
        data_path = src_data_path
        , edits_path = os.path.join(all_dir, edit_name)
        , prevs_path = prev_tokens_path
        , upds_path = upd_tokens_path
    )
    dataset.prepare_token_dataset()
 
'''
load tokens from prev_tokens_path and upd_tokens_path and train gensim.Word2Vec on it
'''
def train_word2vec():
    print('Create and train token2vec')
    vocab = TokenVecTrain(
        dim = 300
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
def convert_tokens_to_vectors():
    print('Convert tokens to vectors')
    converter = Token2VecConverter(w2v_vecs_path)    
    converter.convert(prev_tokens_path, os.path.join(all_dir, prev_name))
    converter.convert(upd_tokens_path, os.path.join(all_dir, upd_name))

'''
divide dataset into 3 parts: train, valid and test
'''
def divide_dataset():
    print('Divide dataset (train, valid, test)')
    def run_division(name):
        divide_set(
            in_path = os.path.join(all_dir, name)
            , out_paths = [os.path.join(train_dir, name), os.path.join(valid_dir, name), os.path.join(test_dir, name)]
            , parts = dataset_parts
        )
        
    run_division(edit_name)
    run_division(prev_name)
    run_division(upd_name)

if __name__ == "__main__":      
    create_token_datasets()
    train_word2vec()
    convert_tokens_to_vectors()
    divide_dataset()