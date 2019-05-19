# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

from helper import eos_vec, w2v_model, eos_token

def get_words_by_vectors(vectors, n = 1):
    seq_len = vectors.shape[0]
    batch_size = vectors.shape[1]
    word_matrix = []
            
    for step in range(seq_len):
        word_matrix.append([])
        for batch_id in range(batch_size):
            closest_words = np.array(w2v_model.most_similar(
                    positive = [vectors[step, batch_id]],
                    topn = n
            ))[:, 0]
            word_matrix[step].append(closest_words)
    return np.asarray(word_matrix)

def topn_vectors(decoder_output, upd_vec, device, n = 1):
    seq_len, batch_size, token_vocab_size = upd_vec.size()  

    upd_vec = upd_vec.numpy()
    decoder_output = decoder_output.numpy()    
        
    # truth.shape = (seq_len, batch_size, 1)
    truth = get_words_by_vectors(upd_vec)
    
    # predicts.shape = (seq_len, batch_size, n)
    predicts = get_words_by_vectors(decoder_output, n)
    
    correct_samples = 0
    total_samples = 0
    
    for i in range(batch_size):
        for j in range(seq_len):
            if truth[j, i, 0] == eos_token:
                break 
            
            total_samples += 1
            if truth[j, i, 0] in predicts[j, i]:
                correct_samples += 1
                
    return float(correct_samples) / total_samples, correct_samples, total_samples

def top1(device, predicts, labels):
    seq_len, batch_size = labels.size()  
    
    mask = torch.where(labels > 0, torch.ones(labels.size(), dtype = torch.uint8).to(device), 
                       torch.zeros(labels.size(), dtype = torch.uint8).to(device))
    predicts_masked = torch.masked_select(predicts, mask)
    labels_masked = torch.masked_select(labels, mask)
    correct_samples, total_samples = torch.sum(predicts_masked == labels_masked).item(), labels_masked.flatten().size(0) 
                
    return float(correct_samples) / total_samples, correct_samples, total_samples

n_grams_to_weight = {
    2 : (0.5, 0.5),
    3 : (0.34, 0.33, 0.33),
    4 : (0.25, 0.25, 0.25, 0.25)
}

def bleu(decoder_output, upd_vec, device, n_gram = 3):
    seq_len, batch_size, token_vocab_size = upd_vec.size()  
    
    eos_tensor= torch.from_numpy(eos_vec).to(device).squeeze(0)
    
    eos_id = torch.tensor(seq_len, device = device)
    eos_ids = eos_id.repeat(batch_size)
    for i in range(batch_size):
        for j in range(seq_len):
            if (torch.allclose(upd_vec[j][i], eos_tensor)):
                eos_ids[i] = j
                break  
            
    upd_vec = upd_vec.numpy()
    decoder_output = decoder_output.numpy()    
        
    # truth.shape = (seq_len, batch_size, 1)
    truth = get_words_by_vectors(upd_vec)
    
    # predicts.shape = (seq_len, batch_size, 1)
    predicts = get_words_by_vectors(decoder_output, 1)
    
    weights = n_grams_to_weight[n_gram]
    
    bleu = 0.0
    for i in range(batch_size):
        bleu += sentence_bleu(
            references = [truth[0 : eos_ids[i], i, 0]]
            , hypothesis = predicts[0 : eos_ids[i], i, 0]
            , weights = weights
        )
        
    return float(bleu) / batch_size, bleu, batch_size
        
if __name__ == '__main__':      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out = torch.from_numpy(np.ones(shape = (3, 2, 300), dtype = np.float32)).to(device)
    upd = torch.from_numpy(np.ones(shape = (3, 2, 300), dtype = np.float32)).to(device)
    upd[1, 0] = torch.from_numpy(eos_vec).to(device)
    upd[2, 0] = torch.zeros(300)
    loss = loss_with_eos(out, upd, device)
    accuracy = top1(out, upd)
    print('accuracy top n', accuracy)
    accuracy = bleu(out, upd, 2)
    print('accuracy bleu 2', accuracy)