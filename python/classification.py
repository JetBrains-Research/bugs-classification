import getopt
import random
import sys
from time import sleep

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List

from app.tokens_dataset import TokensDataSet
from app.model import AttentionBiGRUClassifier
from utils import transform_tokens_to_csv


def parse_cmd_arguments(argv: List[str]) -> Tuple[Path, Path]:
    try:
        opts, _ = getopt.getopt(argv, "t:v", ["train=", "validate="])
    except getopt.GetoptError:
        print('You must provide two arguments:')
        print('python classification.py --train=path/to/train.txt --validate=path/to/test.txt')
        sys.exit(2)
    path_to_train, path_to_test = None, None
    for opt, arg in opts:
        if opt in ('-t', '--train'):
            path_to_train = Path(arg)
        elif opt in ('-v', '--validate'):
            path_to_test = Path(arg)
    if path_to_test is None or path_to_train is None:
        print('You must provide two arguments:')
        print('python classification.py --train path_to_train.txt --validate path_to_test.txt')
        sys.exit(2)
    return path_to_train, path_to_test


def load_dataset(path_to_train: Path, path_to_test: Path, device: torch.device) -> TokensDataSet:
    data = TokensDataSet()
    data.load(path_to_train, path_to_test, 3)
    data.parse_tokens()
    data.dev_test_split(ratio=0.5)
    data.make_tensors()
    data.send_to(device)
    return data


def seed_deterministic_random() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True


def train(data: TokensDataSet, model: torch.nn.Module, device: torch.device,
          loss: torch.nn.modules.loss, optimizer: torch.optim, batch_size: int, epochs: int) -> None:
    model = model.to(device)
    for epoch in range(epochs):
        order = np.random.permutation(data.X_train.shape[0])
        total_loss = 0
        for start_index in range(0, data.X_train.shape[0], batch_size):
            if start_index + batch_size > data.X_train.shape[0]:
                break
            optimizer.zero_grad()
            model.train()

            batch_idxs = order[start_index:(start_index + batch_size)]
            X_batch = data.X_train[batch_idxs].to(device)
            X_batch_lengths = data.X_train_lengths[batch_idxs].to(device)
            y_batch = data.y_train[batch_idxs].to(device)

            preds = model.forward(X_batch, X_batch_lengths)
            loss_val = loss(preds, y_batch)
            loss_val.backward()
            total_loss += loss_val.item()
            optimizer.step()
        print(f'epoch: {epoch} | loss: {total_loss}')


if __name__ == '__main__':
    print('Start classification')
    seed_deterministic_random()
    path_to_train, path_to_test = parse_cmd_arguments(sys.argv[1:])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Load data')
    path_to_train_csv = Path(f'{str(path_to_train)[:-4]}.csv')
    path_to_test_csv = Path(f'{str(path_to_test)[:-4]}.csv')
    transform_tokens_to_csv(path_to_tokens=path_to_train,
                            path_to_csv=path_to_train_csv)
    transform_tokens_to_csv(path_to_tokens=path_to_test,
                            path_to_csv=path_to_test_csv)
    data = load_dataset(path_to_train_csv, path_to_test_csv, device)
    model = AttentionBiGRUClassifier(tokens_per_change=11,
                                     embedding_dim=40,
                                     hidden_dim=200,
                                     atomic_dim=60,
                                     vocab_size=len(data.tokens_vocab),
                                     labels_set_size=len(data.clusters_vocab),
                                     padding_idx=data.token2idx['<PAD>'])
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print('Training model')
    train(data, model, device, loss, optimizer, batch_size=50, epochs=1)
    model.eval()
    model.to(torch.device('cpu'))
    data.send_to(torch.device('cpu'))
    validate_df = torch.cat((data.X_dev, data.X_test), dim=0)
    validate_df_length = torch.cat((data.X_dev_lengths, data.X_test_lengths), dim=0)
    y_true = data.holdout_df['cluster'].values
    pr_auc_score = model.test(X=validate_df,
                              X_real_lengths=validate_df_length,
                              y_true=y_true,
                              labels=np.array(sorted(data.clusters_vocab)))
    print(f'PR-AUC: {pr_auc_score}')
