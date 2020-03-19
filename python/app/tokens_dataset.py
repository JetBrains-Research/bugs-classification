import torch
import numpy as np
from .dataset import DataSet


class TokensDataSet(DataSet):

    def __init__(self):
        self.tokens_vocab = {}
        self.token2idx = {}
        self.clusters_vocab = {}
        self.cluster2idx = {}

    def parse_tokens(self):
        self.tokens_vocab = set(self.df[self.df.columns[1:-1]].values.flatten())
        self.tokens_vocab.update(set(self.holdout_df[self.holdout_df.columns[1:-1]].values.flatten()))
        self.token2idx = {token: idx for idx, token in enumerate(self.tokens_vocab)}

        self.df = self.df.apply(lambda col: 
                                np.array([self.token2idx[token] for token in col.values]) 
                                if col.name in self.df.columns[1:-1] else col)
        self.holdout_df = self.holdout_df.apply(lambda col: 
                                                np.array([self.token2idx[token] for token in col.values]) 
                                                if col.name in self.holdout_df.columns[1:-1] else col)
        
    def make_tensors(self):
        self.X_train = torch.LongTensor(self.df[self.df.columns[1:-1]].values)
        self.X_train_lengths = torch.LongTensor(self.df['real_len'].values)
        self.y_train = torch.LongTensor(self.df['cluster'].values)
        
        self.X_dev = torch.LongTensor(self.dev_df[self.dev_df.columns[1:-1]].values)
        self.X_dev_lengths = torch.LongTensor(self.dev_df['real_len'].values)
        
        self.X_test = torch.LongTensor(self.test_df[self.test_df.columns[1:-1]].values)
        self.X_test_lengths = torch.LongTensor(self.test_df['real_len'].values)

        self._X_holdout = torch.LongTensor(self.holdout_df[self.holdout_df.columns[1:-1]].values)
        self._X_holdout_lengths = torch.LongTensor(self.holdout_df['real_len'].values)

    def send_to(self, device):
        self.X_train = self.X_train.to(device)
        self.X_train_lengths = self.X_train_lengths.to(device)
        self.y_train = self.y_train.to(device)

        self.X_dev = self.X_dev.to(device)
        self.X_dev_lengths = self.X_dev_lengths.to(device)

        self.X_test = self.X_test.to(device)
        self.X_test_lengths = self.X_test_lengths.to(device)
        
        self._X_holdout = self._X_holdout.to(device)
        self._X_holdout_lengths = self._X_holdout_lengths.to(device)
