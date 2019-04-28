# -*- coding: utf-8 -*-

import gensim
import numpy as np

EOS_TOKEN = '<EOS>'
edit_eos = np.array([0, 0, 0, 0], dtype = np.float32)

w2v_path = 'datasets/data/tokens/word2vec_keyed_vectors'        
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)
eos_vec = np.array(w2v_model[EOS_TOKEN], dtype = np.float32)        

EDIT_NAME = 'edits.pickle'
PREV_NAME = 'prevs.pickle'
UPD_NAME = 'updates.pickle'