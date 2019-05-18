# -*- coding: utf-8 -*-

import os
import gensim
import numpy as np

EOS_TOKEN = '<EOS>'
edit_eos = np.array([0, 0, 0, 0], dtype = np.float32)

token_dir = 'datasets/data/tokens'
id2token_path = os.path.join(token_dir, 'id2token.pickle')
token2id_path = os.path.join(token_dir, 'token2id.pickle')
w2v_path = os.path.join(token_dir, 'word2vec_keyed_vectors')
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)
eos_vec = np.array(w2v_model[EOS_TOKEN], dtype = np.float32)        

EDIT_NAME = 'edits.pickle'
PREV_NAME = 'prevs.pickle'
UPD_NAME = 'updates.pickle'