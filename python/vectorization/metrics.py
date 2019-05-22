# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

from helper import eos_vec, w2v_model, eos_token


def check_data(prediction, ground_truth):
    if np.size(ground_truth) == 0:
        raise ValueError('No data in ground_truth')
        
    if np.size(ground_truth) != np.size(prediction):
        raise ValueError('Lengths of ground_truth and precision are not equal')
        
def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    check_data(prediction, ground_truth)
    acc_predictions = sum(1 for pred, truth in zip(prediction, ground_truth) if pred == truth)
    return acc_predictions / np.size(ground_truth)


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