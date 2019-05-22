# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

from helper import eos_vec, w2v_model, eos_token

def top1(device, predicts, labels):
    seq_len, batch_size = labels.size()  
    
    mask = torch.where(labels > 0, torch.ones(labels.size(), dtype = torch.uint8).to(device), 
                       torch.zeros(labels.size(), dtype = torch.uint8).to(device))
    predicts_masked = torch.masked_select(predicts, mask)
    labels_masked = torch.masked_select(labels, mask)
    correct_samples, total_samples = torch.sum(predicts_masked == labels_masked).item(), labels_masked.flatten().size(0) 
                
    return float(correct_samples) / total_samples, correct_samples, total_samples

def calc_acc(device, predicts, labels):
    mask = torch.where(labels > 0, torch.ones(labels.size(), dtype = torch.uint8).to(device), 
                       torch.zeros(labels.size(), dtype = torch.uint8).to(device))
    predicts_masked = torch.masked_select(predicts, mask)
    labels_masked = torch.masked_select(labels, mask)
    correct_samples, total_samples = torch.sum(predicts_masked == labels_masked).item(), labels_masked.size(0) 
                
    return float(correct_samples) / total_samples, correct_samples, total_samples