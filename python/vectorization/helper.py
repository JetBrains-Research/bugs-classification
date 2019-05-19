# -*- coding: utf-8 -*-

import os
import gensim
import numpy as np

eos_token = '<EOS>'
edit_eos = np.array([0, 0, 0, 0], dtype = np.float32)

token_dir = 'datasets/data/tokens'
id2token_path = os.path.join(token_dir, 'id2token.pickle')
token2id_path = os.path.join(token_dir, 'token2id.pickle')
w2v_path = os.path.join(token_dir, 'word2vec_keyed_vectors')
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)
eos_vec = np.array(w2v_model[eos_token], dtype = np.float32)        

edit_name = 'edits.pickle'
prev_name = 'prevs.pickle'
upd_name = 'updates.pickle'

save_dir = 'models'
model_save_path = os.path.join(save_dir, 'token_seq_model.pt')

all_folder = 'datasets/data/tokens/all'
train_folder = 'datasets/data/tokens/train'
valid_folder = 'datasets/data/tokens/valid'
test_folder = 'datasets/data/tokens/test'


import torch
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
            edit_gpu = torch.from_numpy(edit).to(device)
            prev_gpu = torch.from_numpy(prev).to(device)
            updated_gpu = torch.from_numpy(updated).to(device).contiguous()
    
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
            
        print("Epoch: %i, Train loss: %f, Train acc: %f, Valid accuracy: %f" % (epoch + 1, ave_loss, train_acc, valid_acc))
            
    return train_loss_history, train_acc_history, valid_acc_history

def compute_accuracy(model, valid_iterator, accuracy):   
    model.eval()    
        
    with torch.no_grad():
        correct_samples = 0
        total_samples = 0
            
        for i_step, (edit, prev, updated) in enumerate(valid_iterator):
            
            edit_gpu = torch.from_numpy(edit).to(device)
            prev_gpu = torch.from_numpy(prev).to(device)
            updated_gpu = torch.from_numpy(updated).to(device).contiguous()
    
            probs = model(edit_gpu, prev_gpu, updated_gpu)
            _, indices = torch.max(probs, 2)
            
            _, cur_correct_count, cur_sum_count = accuracy(device, indices, updated_gpu)
            correct_samples += cur_correct_count
            total_samples += cur_sum_count
            
        return float(correct_samples) / total_samples
    
def draw_plots(plot_dir, train_loss_history, train_acc_history, valid_acc_history):
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