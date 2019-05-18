# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq

from batch_token_iterator import BatchTokenIterator
from metrics import top1, bleu
from helper import w2v_model, token2id_path, save_dir, model_save_path, \
    train_folder, valid_folder, test_folder

HIDDEN_SIZE = 64
N_LAYERS = 2
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
EDIT_DIM = 4

N_EPOCHS = 20
CLIP = 1

BATCH_SIZE = 70

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

plot_dir = 'plots'
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train(model, device, optimizer, lr_scheduler, 
          train_iterator, valid_iterator, 
          loss, accuracy, n_epochs):
    
    train_loss_history = []
    train_acc_history = []
    valid_acc_history = []
        
    best_valid = np.Inf
        
    for epoch in range(n_epochs):
        model.train()
    
        epoch_loss = 0.0 
        correct_samples = 0
        total_samples = 0
        for i_step, (edit, prev, updated) in enumerate(train_iterator):
    
            edit_gpu = edit.to(device)
            prev_gpu = prev.to(device)
            updated_gpu = updated.to(device).contiguous()
    
            probs = model(edit_gpu, prev_gpu, updated_gpu)
    
            # updated = (seq_len, batch size)
            # outputs = (seq_len, batch_size, vocab_size)       
            loss_value = loss(probs.permute(1, 2, 0), updated_gpu.t())      
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(probs, 2)
            _, cur_correct_count, cur_sum_count = accuracy(device, indices, updated_gpu)
            correct_samples += cur_correct_count
            total_samples += cur_sum_count
    
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    
            epoch_loss += loss_value.item()
    
            print('Train batch: %i/%i' % (min(i_step + 1, train_iterator.n_batches), train_iterator.n_batches), end = '\r')
    
        ave_loss = float(epoch_loss) / train_iterator.n_batches
        train_loss_history.append(ave_loss)
        
        train_acc = float(correct_samples) / total_samples
        train_acc_history.append(train_acc)
        
        valid_acc = compute_accuracy(
            model = model, 
            valid_iterator = valid_iterator, 
            accuracy = accuracy
        )
            
        valid_acc_history.append(valid_acc)
            
        lr_scheduler.step(valid_acc)
            
        if best_valid > valid_acc:
            best_valid = valid_acc
            with open(model_save_path, 'wb') as model_file:
                pickle.dump(model, model_file)
            
        print("Epoch: %i, Train loss: %f, Train acc: %f, Valid accuracy: %f" % (epoch, ave_loss, train_acc, valid_acc))
            
    return train_loss_history, train_acc_history, valid_acc_history
    
def compute_accuracy(model, valid_iterator, accuracy):   
    model.eval()    
        
    with torch.no_grad():
        correct_samples = 0
        total_samples = 0
            
        for i_step, (edit, prev, updated) in enumerate(valid_iterator):
            
            edit_gpu = edit.to(device)
            prev_gpu = prev.to(device)
            updated_gpu = updated.to(device).contiguous()
    
            probs = model(edit_gpu, prev_gpu, updated_gpu)
            _, indices = torch.max(probs, 2)
            
            _, cur_correct_count, cur_sum_count = accuracy(device, indices, updated_gpu)
            correct_samples += cur_correct_count
            total_samples += cur_sum_count
            
        return float(correct_samples) / total_samples

def draw_plots(train_loss_history, train_acc_history, valid_acc_history):
    plt.figure(figsize=(15, 7))
    plt.subplot(111)
    plt.title("Loss")
    plt.plot(train_loss_history)
    plt.savefig(os.path.join(plot_dir,'loss.png'))
    
    plt.figure(figsize=(15, 7))
    plt.subplot(111)
    plt.title("Train/validation accuracy")
    plt.plot(train_acc_history, label = 'train')
    plt.plot(valid_acc_history, label = 'valid')
    plt.legend()
    plt.savefig(os.path.join(plot_dir,'accuracy.png'))

  
if __name__ == '__main__':
    train_iterator, valid_iterator, test_iterator = create_datasets()
    
    with open(token2id_path, 'rb') as token2id_file:
        token2id = pickle.load(token2id_file)  
        embeddings = np.zeros((len(token2id), w2v_model.vectors.shape[1]), dtype = np.float32)
        for token, ind in token2id.items():
            if token in w2v_model.vocab:
                embeddings[ind] = w2v_model.get_vector(token)
                
    model = create_model(embeddings)
    optimizer = optim.Adagrad(model.parameters(), lr = 0.09)
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
    
    draw_plots(train_loss_history, train_acc_history, valid_acc_history)