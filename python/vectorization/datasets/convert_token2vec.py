# -*- coding: utf-8 -*-

import pickle
import gensim

class Token2VecConverter(object):
    def __init__(self, token2vec_path):
        self.token2vec_path = token2vec_path
        
    def convert(self, token_path, vec_path):
        with open(token_path, 'rb') as token_file:
            with open(vec_path, 'wb') as vec_file:
                tokens = pickle.load(token_file)
                model = gensim.models.KeyedVectors.load_word2vec_format(self.token2vec_path)
                vecs = []
                for cur_sample in tokens:
                    sample_vec = [model[token] for token in cur_sample]
                    vecs.append(sample_vec)
                pickle.dump(vecs, vec_file)