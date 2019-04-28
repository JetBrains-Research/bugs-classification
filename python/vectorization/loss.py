# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F

from torch.autograd.function import Function

from helper import eos_vec

class LossWithEos(Function):

    @staticmethod
    def forward(ctx, in_feat):
        ctx.save_for_backward(in_feat)
        return in_feat.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        in_feat, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[in_feat < 0] = 0
        return grad_input

def loss_with_eos(decoder_output, upd_vec):
    # shape = (seq_len, batch size, token_vocab_size)
    seq_len, batch_size, token_vocab_size = upd_vec.size()
    
    eos_tensor= torch.from_numpy(eos_vec).squeeze(0)
    
    eos_id = torch.tensor(seq_len)
    eos_ids = eos_id.repeat(batch_size)
    for i in range(batch_size):
        for j in range(seq_len):
            if (torch.allclose(upd_vec[j][i], eos_tensor)):
                eos_ids[i] = j
                break   
    
    cosine_sums = torch.empty(batch_size)
    for i in range(batch_size):
        # cur_cos.size() = (seq_len, token_vocab_size)
        cur_cos = F.cosine_similarity(decoder_output[0 : eos_ids[i], i], upd_vec[0 : eos_ids[i], i])
        cosine_sums[i] = torch.mean(cur_cos)
        
    return 1.0 - torch.mean(cosine_sums)