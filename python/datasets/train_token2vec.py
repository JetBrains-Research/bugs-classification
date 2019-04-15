# -*- coding: utf-8 -*-

import gensim

from iter_token_reader import IterTokenReader

class TokenVecTrain(object):
    
    def __init__(self
                 , dim
                 , min_count
                 , model_save_path
                 , vectors_save_path
                 , prev_data_path
                 , upd_data_path
                 , iter = 5): 
        
        self.model_save_path = model_save_path
        self.vectors_save_path = vectors_save_path

        self.reader = IterTokenReader(
            prev_data_path = prev_data_path
            , upd_data_path = upd_data_path
        )
        self.model = gensim.models.Word2Vec(
                sentences = self.reader
                , size = dim
                , min_count = min_count
                , iter = iter
        )
        self.model.save(self.model_save_path)
        self.model.wv.save_word2vec_format(self.vectors_save_path)
        print('\nModel initialized \n')
  
    def train(self, n_epochs = None):      
        self.model.build_vocab(self.reader, update=True)
        self.model.save(self.model_save_path)
        self.model.wv.save_word2vec_format(self.vectors_save_path)
        print('\nVocab builded \n')
        
        if n_epochs is None:
            n_epochs = self.model.iter
            
        for i in range(n_epochs):
            self.model.train(sentences = self.reader, 
                             total_examples = self.model.corpus_count,
                             epochs = 1)
            self.model.save(self.model_save_path)
            self.model.wv.save_word2vec_format(self.vectors_save_path)