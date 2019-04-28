# -*- coding: utf-8 -*-

from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, device, input_dim, hidden_size = 128, n_layers = 2, dropout = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p = dropout)
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_size,
            num_layers = n_layers, 
            dropout = dropout,
            bidirectional = True
        )
        self.device = device
    
    def forward(self, edit_vecs, hidden_state = None, cell_state = None):
        # edit_vecs.shape = (batch_size, input_dim = 4)
        x = edit_vecs.reshape((1, edit_vecs.shape[0], edit_vecs.shape[1]))
        
        # x.shape = (seq_len = 1, batch_size, input_dim = 4)
        lstm_out, (hidden_state, cell_state) = self.lstm(x) if  \
            hidden_state is None or cell_state is None else     \
            self.lstm(x, (hidden_state, cell_state))
        
        # lstm_out.shape = (seq_len = 1, batch_size, 2 * hidden_size)
        # hidden_state.shape = (lstm_layers * 2, batch_size, hidden_size)
        # cell_state.shape = (lstm_layers * 2, batch_size, hidden_size)
        return lstm_out, hidden_state, cell_state
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = self.device)