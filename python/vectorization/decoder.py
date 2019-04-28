# -*- coding: utf-8 -*-

from torch import nn

class Decoder(nn.Module):
    # input_dim = encoder_out_dim + vec_dim = 2 * hidden_size + vec_dim (300)
    # output_dim = vec_dim = 300
    def __init__(self, input_dim, hidden_size, output_dim, n_layers = 2, dropout = 0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size = input_dim, 
            hidden_size = hidden_size, 
            num_layers = n_layers, 
            dropout = dropout,
            bidirectional = True
        )
        
        self.out = nn.Linear(2 * hidden_size, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size, input_dim]
        # hidden_state.shape = (n_layers * 2, batch_size, hidden_size)
        # cell_state.shape = (n_layers * 2, batch_size, hidden_size)
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size, input_dim]                 
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        
        #output = [1, batch_size, hidden_size * 2]
        #hidden = [n_layers * 2, batch_size, hidden_size]
        #cell = [n_layers * 2, batch_size, hidden_size]

        prediction = self.out(output.squeeze(0))
        
        #prediction = [batch_size, output_dim = 300]
        return prediction.float(), hidden, cell