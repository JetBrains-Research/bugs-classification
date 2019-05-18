# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

class Decoder(nn.Module):
    # input_dim = encoder_out_dim + token_emb_dim = 2 * hidden_size + token_emb_dim
    def __init__(self, device, enc_out_dim, token_emb_dim, hidden_size, vocab_size, n_layers = 2
                 , dropout = 0.2, embeddings = None, max_len = 100):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_dim = vocab_size
        self.max_len = max_len
        
        if embeddings is None:
            self.embedding = nn.Embedding(vocab_size, token_emb_dim).to(device)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings)).to(device)
            
        input_dim = enc_out_dim + token_emb_dim        
        self.lstm = nn.LSTM(
            input_size = input_dim, 
            hidden_size = hidden_size, 
            num_layers = n_layers, 
            dropout = dropout
        ).to(device)
        
        self.out = nn.Linear(hidden_size, vocab_size).to(device)       
        
    def forward(self, enc_out, prevs, hidden, cell):
        # enc_out.shape = (batch_size, 2 * encoder.hidden_size)
        # prevs.shape = (batch_size)
        prevs = self.embedding(prevs)
        # prevs.shape = (batch_size, embed_dim)
        input = torch.cat((enc_out, prevs), -1).unsqueeze(0)
        
        # input.shape = (batch_size, 2 * encoder.hidden_size + embed_dim)
        # hidden.shape = (2 * n_layers, batch_size, encoder.hidden_size)       
        output, (hidden, cell) = self.lstm(input, 
                (hidden.view(-1, prevs.size(0), self.hidden_size), 
                 cell.view(-1, prevs.size(0), self.hidden_size)))
        
        # output.shape = (1, batch_size, hidden_size)
        # hidden.shape = cell.shape = (n_layers, batch_size, hidden_size)
        prediction = F.log_softmax(self.out(output.squeeze(0)), dim = 1)
        
        #prediction.shape = (batch_size, vocab_size)
        return prediction.float(), (hidden, cell)