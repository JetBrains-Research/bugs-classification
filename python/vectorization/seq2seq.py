# -*- coding: utf-8 -*-

import torch
from torch import nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
     
    def prepare_decoder_input(self, prev_vec, encoder_out):
        # prev_vec.shape = (batch size, vec_dim = 300)
        # encoder_out.shape = (seq_len, batch_size, 2 * hidden_size)
        return torch.cat((encoder_out.squeeze(0).float(), prev_vec), 1)     
        
    def forward(self, edits, prevs, upds):
        
        # prevs.shape = (seq_len, batch size)
        # edits.shape = (seq_len, batch size, edit_dim = 4)
        
        batch_size = prevs.size(1)
        max_len = prevs.size(0)
    
        hidden = None
        cell = None
        encoder_outputs = torch.zeros(max_len, batch_size, 2 * self.encoder.hidden_size, device = self.device)
        for enc_step in range(max_len):
            encoder_output, (hidden, cell) = self.encoder(edits[enc_step], prevs[enc_step], upds[enc_step], hidden, cell)
            encoder_outputs[enc_step] = encoder_output
            
        decoder_outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim, device = self.device)
        for dec_step in range(max_len):
            decoder_output, (hidden, cell) = self.decoder(
                enc_out = encoder_outputs[dec_step], 
                prevs = prevs[dec_step], 
                hidden = hidden,
                cell = cell
            )  
            decoder_outputs[dec_step] = decoder_output
        
        return decoder_outputs