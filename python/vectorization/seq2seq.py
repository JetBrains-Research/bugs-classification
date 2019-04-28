# -*- coding: utf-8 -*-

from torch import nn
import torch

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
     
    def prepare_decoder_input(self, prev_vec, encoder_out):
        # prev_vec.shape = (batch size, vec_dim = 300)
        # encoder_out.shape = (seq_len, batch_size, 2 * hidden_size)
        return torch.cat((encoder_out.squeeze(0).float(), prev_vec), 1)     
        
    def forward(self, prev_vecs, edit_vecs):
        
        # prev_vecs.shape = (seq_len, batch size, vec_dim = 300)
        # edit_vecs.shape = (seq_len, batch size, edit_dim = 4)
        
        batch_size = prev_vecs.shape[1]
        max_len = prev_vecs.shape[0]
        token_vocab_size = prev_vecs.shape[2]
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, token_vocab_size).to(self.device)
        
        #hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_out, encoder_hidden, encoder_cell = self.encoder(edit_vecs[0])
        decoder_out, decoder_hidden, decoder_cell = self.decoder(
            input = self.prepare_decoder_input(prev_vecs[0], encoder_out)
            , hidden = encoder_hidden
            , cell = encoder_cell
        )
        
        for sample_id in range(1, max_len):  
            encoder_out, encoder_hidden, encoder_cell = self.encoder(
                edit_vecs[sample_id], encoder_hidden, encoder_cell
            )
            
            decoder_out, decoder_hidden, decoder_cell = self.decoder(
                input = self.prepare_decoder_input(prev_vecs[sample_id], encoder_out)
                , hidden = decoder_hidden
                , cell = decoder_cell
            )
            outputs[sample_id] = decoder_out
        
        return outputs