# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

from helper import eos_vec, w2v_model, EOS_TOKEN

def loss_with_eos(decoder_output, upd_vec, device):
    # shape = (seq_len, batch size, token_vocab_size)
    seq_len, batch_size, token_vocab_size = upd_vec.size()
    
    eos_tensor= torch.from_numpy(eos_vec).to(device).squeeze(0)
    
    eos_id = torch.tensor(seq_len, device = device)
    eos_ids = eos_id.repeat(batch_size)
    for i in range(batch_size):
        for j in range(seq_len):
            if (torch.allclose(upd_vec[j][i], eos_tensor)):
                eos_ids[i] = j
                break   
    
    cosine_sums = torch.empty(batch_size, device = device)
    for i in range(batch_size):
        # cur_cos.size() = (seq_len, token_vocab_size)
        cur_cos = F.cosine_similarity(decoder_output[0 : eos_ids[i], i], upd_vec[0 : eos_ids[i], i])
        cosine_sums[i] = torch.mean(cur_cos)
        
    return 1.0 - torch.mean(cosine_sums)

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

def topn(decoder_output, upd_vec, n = 1):
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
            if truth[j, i, 0] == EOS_TOKEN:
                break 
            
            total_samples += 1
            if truth[j, i, 0] in predicts[j, i]:
                correct_samples += 1
                
    return float(correct_samples) / total_samples, correct_samples, total_samples

n_grams_to_weight = {
    2 : (0.5, 0.5),
    3 : (0.34, 0.33, 0.33),
    4 : (0.25, 0.25, 0.25, 0.25)
}

def bleu(decoder_output, upd_vec, n_gram = 3):
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