# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq

from iterators import BatchTokenIterator
from metrics import top1
from helper import w2v_model, token2id_path, save_dir, model_save_path, \
    train_folder, valid_folder, test_folder, device, train, draw_plots

HIDDEN_SIZE = 8
N_LAYERS = 2
ENC_DROPOUT = 0.05
DEC_DROPOUT = 0.05
EDIT_DIM = 4

N_EPOCHS = 10

BATCH_SIZE = 100

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

plot_dir = 'plots'
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

def create_datasets():
    train_iterator = BatchTokenIterator(
        dir_path = train_folder
        , batch_size = BATCH_SIZE
    )
    
    valid_iterator = BatchTokenIterator(
        dir_path = valid_folder
        , batch_size = BATCH_SIZE
    )
    
    test_iterator = BatchTokenIterator(
        dir_path = test_folder
        , batch_size = BATCH_SIZE
    ) 
    
    return train_iterator, valid_iterator, test_iterator       
        
def create_model(embeddings):
    vocab_size, token_emb_dim = embeddings.shape
    
    encoder = Encoder(
        device = device
        , token_emb_dim = token_emb_dim
        , input_dim = EDIT_DIM + 2 * token_emb_dim
        , hidden_size = HIDDEN_SIZE
        , n_layers = N_LAYERS
        , dropout = ENC_DROPOUT
        , embeddings = embeddings
    )

    decoder = Decoder(
        device = device
        , enc_out_dim = 2 * HIDDEN_SIZE
        , token_emb_dim = token_emb_dim
        , hidden_size = 2 * HIDDEN_SIZE
        , vocab_size = vocab_size
        , n_layers = N_LAYERS
        , dropout = DEC_DROPOUT
        , embeddings = embeddings
    )

    return Seq2Seq(encoder, decoder, device).to(device)

  
if __name__ == '__main__':
    train_iterator, valid_iterator, test_iterator = create_datasets()
    
    with open(token2id_path, 'rb') as token2id_file:
        token2id = pickle.load(token2id_file)  
        embeddings = np.zeros((len(token2id), w2v_model.vectors.shape[1]), dtype = np.float32)
        for token, ind in token2id.items():
            if token in w2v_model.vocab:
                embeddings[ind] = w2v_model.get_vector(token)
                
    model = create_model(embeddings)
    optimizer = optim.Adagrad(model.parameters(), lr = 0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.4
                                     , patience = 3, verbose = True, min_lr = 1e-6)
    
    loss = nn.CrossEntropyLoss(ignore_index = 0)

    train_loss_history, train_acc_history, valid_acc_history = train(
        model = model,
        device = device,
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
        train_iterator = train_iterator,
        valid_iterator = valid_iterator,
        loss = loss,
        accuracy = top1,
        n_epochs = N_EPOCHS
    )
    
    draw_plots(plot_dir, train_loss_history, train_acc_history, valid_acc_history)