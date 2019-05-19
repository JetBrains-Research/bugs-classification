# -*- coding: utf-8 -*-

import os
import gensim
import numpy as np

eos_token = '<EOS>'
edit_eos = np.array([0, 0, 0, 0], dtype = np.float32)

token_dir = 'datasets/data/tokens'
id2token_path = os.path.join(token_dir, 'id2token.pickle')
token2id_path = os.path.join(token_dir, 'token2id.pickle')
w2v_path = os.path.join(token_dir, 'word2vec_keyed_vectors')
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)
eos_vec = np.array(w2v_model[eos_token], dtype = np.float32)        

edit_name = 'edits.pickle'
prev_name = 'prevs.pickle'
upd_name = 'updates.pickle'

save_dir = 'models'
model_save_path = os.path.join(save_dir, 'token_seq_model.pt')

all_folder = 'datasets/data/tokens/all'
train_folder = 'datasets/data/tokens/train'
valid_folder = 'datasets/data/tokens/valid'
test_folder = 'datasets/data/tokens/test'