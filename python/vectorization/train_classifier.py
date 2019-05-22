# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from iterators import BatchMarkedVecIterator
from helper import device, draw_plots, max_seq_len
from metrics import calc_acc

BATCH_SIZE = 30
N_EPOCHS = 50    

plot_dir = 'classifier_plots'
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)
    
train_folder = 'datasets/data/tokens/marked/train'
valid_folder = 'datasets/data/tokens/marked/valid'

save_dir = 'models'
classifier_save_path = os.path.join(save_dir, 'classifier.pt')

def create_datasets():
    train_iterator = BatchMarkedVecIterator(
        dir_path = train_folder,
        batch_size = BATCH_SIZE
    )
    
    valid_iterator = BatchMarkedVecIterator(
        dir_path = valid_folder,
        batch_size = BATCH_SIZE
    )
    
    return train_iterator, valid_iterator          
        
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
        for i_step, (vecs, marks) in enumerate(train_iterator):
    
            # vecs.shape = (seq_len, batch_size, 2 * hidden_size)
            vecs_gpu = vecs.to(device)
            marks_gpu = torch.from_numpy(marks).to(device).contiguous()
    
            probs = model(vecs_gpu)
    
            # marks = (batch size)
            # probs = (batch_size, vocab_size)  
            
            loss_value = loss(probs, marks_gpu)      
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(probs, 1)
            _, cur_correct_count, cur_sum_count = accuracy(device, indices, marks_gpu)
            correct_samples += cur_correct_count
            total_samples += cur_sum_count
    
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
            with open(classifier_save_path, 'wb') as model_file:
                pickle.dump(model, model_file)
            
        print("Epoch: %i, Train loss: %f, Train acc: %f, Valid accuracy: %f" % (epoch + 1, ave_loss, train_acc, valid_acc))
            
    return train_loss_history, train_acc_history, valid_acc_history

def compute_accuracy(model, valid_iterator, accuracy):   
    model.eval()    
        
    with torch.no_grad():
        correct_samples = 0
        total_samples = 0
            
        for i_step, (vecs, marks) in enumerate(valid_iterator):
            
            vecs_gpu = vecs.to(device)
            marks_gpu = torch.from_numpy(marks).to(device).contiguous()
    
            probs = model(vecs_gpu)
            _, indices = torch.max(probs, 1)
            
            _, cur_correct_count, cur_sum_count = accuracy(device, indices, marks_gpu)
            correct_samples += cur_correct_count
            total_samples += cur_sum_count
            
        return float(correct_samples) / total_samples
    

if __name__ == '__main__':
    train_iterator, valid_iterator = create_datasets()
    
    classifier = nn.Linear(max_seq_len * 2 * train_iterator.seq2seq_model.encoder.hidden_size, 62)
    
    optimizer = optim.Adagrad(classifier.parameters(), lr = 0.05, weight_decay = 1e-6)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.4
                                     , patience = 3, verbose = True, min_lr = 1e-6)
    
    loss = nn.CrossEntropyLoss(ignore_index = 0)
    
    train_loss_history, train_acc_history, valid_acc_history = train(
        model = classifier,
        device = device,
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
        train_iterator = train_iterator,
        valid_iterator = valid_iterator,
        loss = loss,
        accuracy = calc_acc,
        n_epochs = N_EPOCHS
    )
    
    draw_plots(plot_dir, train_loss_history, train_acc_history, valid_acc_history)
    