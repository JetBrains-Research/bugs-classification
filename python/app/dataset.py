import pandas as pd


class DataSet:

    def __init__(self):
        self.clusters_vocab = {}
        self.cluster2idx = {}

    def load(self, path_to_train, path_to_test, k_nearest_for_train=3):

        self.df = pd.read_csv(path_to_train, index_col='id', engine='python')
        self.holdout_df = pd.read_csv(path_to_test, index_col='id', engine='python')

        # Drop out unnecessary neighbors
        self.df = self.df[self.df.index % 10 < k_nearest_for_train]

        # Drop out records with unknown cluster
        self.df.drop(self.df[self.df.cluster == 'unknown'].index, inplace=True)
        self.holdout_df.drop(self.holdout_df[self.holdout_df.cluster == 'unknown'].index, inplace=True)

        # Update labels (clusters) set
        self.clusters_vocab = set(self.df.cluster.values)        
        self.cluster2idx = {cluster: idx for idx, cluster in enumerate(sorted(self.clusters_vocab))}

        # Encode clusters
        self.df['cluster'] = self.df['cluster'].apply(lambda cluster: self.cluster2idx[cluster])   
        
    def dev_test_split(self, ratio=0.5):
        split_index = int(self.holdout_df.shape[0] * ratio)
        self.dev_df = self.holdout_df.iloc[:split_index]
        self.test_df = self.holdout_df.iloc[split_index:]
