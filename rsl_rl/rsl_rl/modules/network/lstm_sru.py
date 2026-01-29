# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025
#
# Description: SRU_LSTM implementation - LSTM with Additive Transformation Gates
# that modulate the cell state update based on input features.

import torch
import torch.nn as nn
from typing import Optional, Tuple

class LSTMSRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMSRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Combined linear layer for all gates
        self.linear_all = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        
        # Initialize forgetting gate bias to 1 (+ delta to break symmetry)
        self.linear_all.bias.data[hidden_size:2*hidden_size] = 1.0 + torch.randn(hidden_size)

        # initialize all weights to orthogonal
        nn.init.orthogonal_(self.linear_all.weight)
        
        # Transformation Gate
        self.transform_gate = nn.Linear(input_size, hidden_size, bias=bias)
        
        # Initialize transform gate weights to orthogonal
        nn.init.orthogonal_(self.transform_gate.weight)

    def forward(self, x, h, c):
        # concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Compute all gates in a single linear transformation, and transform gate
        gates = self.linear_all(combined)
        tx = self.transform_gate(x)

        # Split gates into input, forget, cell, and output
        i, f, o, g = torch.split(gates, self.hidden_size, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g_t = torch.tanh(tx * g)

        c_next = f * c + i * g_t
        
        # Compute the new hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class LSTM_SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(LSTM_SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = nn.ModuleList([LSTMSRUCell(input_size if i == 0 else hidden_size, hidden_size) 
                                    for i in range(num_layers)])

    def forward(self, x, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert batch_first to time_first

        seq_len, batch_size, _ = x.shape

        if state is None:
            state = self.init_state(batch_size, x.device)

        outputs = []
        h, c = state
        for t in range(seq_len):
            x_t = x[t]
            new_h, new_c = [], []
            for layer_idx, cell in enumerate(self.cells):
                h_t, c_t = cell(x_t, h[layer_idx], c[layer_idx])
                new_h.append(h_t)
                new_c.append(c_t)
                x_t = h_t  # Output of current layer is input to the next
            h = torch.stack(new_h)
            c = torch.stack(new_c)
            outputs.append(h[-1])

        outputs = torch.stack(outputs)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # Convert time_first back to batch_first

        return outputs, (h, c)
    
    def init_state(self, batch_size : int, device : torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)

if __name__ == '__main__':
    batch_size = 3
    input_size = 15
    hidden_size = 128
    seq_len = 10
    batch_first = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(seq_len, batch_size, input_size).to(device)
    if batch_first:
        x = x.permute(1, 0, 2)

    print(f'Input shape: {x.shape}')
    print(f'Batch first: {batch_first}')

    model = LSTM_SRU(input_size, hidden_size, num_layers=2, batch_first=batch_first).to(device)
    out, state = model(x, None)

    print(f'Output shape: {out.shape}')
    print(f'State shape: {state[0].shape}, {state[1].shape}')
