import torch
from .dataset import DataSet


class BOWDataSet(DataSet):

    def __init__(self, path_to_train, path_to_test, k_nearest_for_train=3):
        super().__init__(path_to_train, path_to_test, k_nearest_for_train)

    def make_tensors(self):
        self.X_train = torch.LongTensor(self.df[self.df.columns[0:-1]].values)
        self.y_train = torch.LongTensor(self.df['cluster'].values)
        
        self.X_dev = torch.LongTensor(self.dev_df[self.dev_df.columns[0:-1]].values)
        self.X_test = torch.LongTensor(self.test_df[self.test_df.columns[0:-1]].values)

        self._X_holdout = torch.LongTensor(self.holdout_df[self.holdout_df.columns[0:-1]].values)

    def send_to(self, device):
        self.X_train = self.X_train.to(device)
        self.y_train = self.y_train.to(device)

        self.X_dev = self.X_dev.to(device)
        self.X_test = self.X_test.to(device)

        self._X_holdout = self._X_holdout.to(device)
