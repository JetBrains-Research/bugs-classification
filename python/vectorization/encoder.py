# -*- coding: utf-8 -*-

from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, device, token_emb_dim, input_dim, hidden_size = 128, n_layers = 2, dropout = 0.2
                 , embeddings = None, vocab_size = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p = dropout)
        
        if embeddings is None:
            self.embedding = nn.Embedding(vocab_size, token_emb_dim).to(device)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings)).to(device)
            
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_size,
            num_layers = n_layers, 
            dropout = dropout,
            bidirectional = True
        ).to(device)
        
        self.device = device
    
    def forward(self, edits, prevs, upds, hidden = None, cell = None):
        prevs = self.embedding(prevs)
        upds = self.embedding(upds)
        
        # edits.shape = (seq_len, batch_size, input_dim = 4)
        input = torch.cat((edits.type(torch.FloatTensor), prevs, upds), -1)
        input = input.unsqueeze(0)
        
        # input.shape = (seq_len = 1, batch_size, input_dim = 4 + 2 * token_emb_dim)
        lstm_out, (hidden, cell) = self.lstm(input) if  \
            hidden is None or cell is None else     \
            self.lstm(input, (hidden, cell))
        
        # lstm_out.shape = (seq_len = 1, batch_size, 2 * hidden_size)
        # hidden_state.shape = (lstm_layers * 2, batch_size, hidden_size)
        # cell_state.shape = (lstm_layers * 2, batch_size, hidden_size)
        return lstm_out, (hidden, cell)