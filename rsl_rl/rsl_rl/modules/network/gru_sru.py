# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025
#
# Description: SRU_GRU implementation - GRU with Additive Transformation Gates
# adding input-dependent modulation to the candidate hidden state.

import torch
import torch.nn as nn
from typing import Optional, Tuple

class GRUSRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUSRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Combined linear layer for all gates
        self.linear_all = nn.Linear(input_size + hidden_size, 2 * hidden_size, bias=bias)
        
        # New hidden state linear layer
        self.linear_n = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)

        # Initialize update gate bias to 1
        self.linear_all.bias.data[:hidden_size] = 1.0 + torch.randn(hidden_size)

        # Orthogonal initialization for weights
        nn.init.orthogonal_(self.linear_all.weight)
        nn.init.orthogonal_(self.linear_n.weight)
        
        # Transformation Gate
        self.transform_gate = nn.Linear(input_size, hidden_size, bias=bias)
        
        # Initialize transform gate weights to orthogonal
        nn.init.orthogonal_(self.transform_gate.weight)

    def forward(self, x, h):
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # Compute all gates in a single linear transformation, and transform gate
        gates = self.linear_all(combined)
        tx = self.transform_gate(x)
        
        # Split gates into update, reset, and new hidden state
        z, r = torch.split(gates, self.hidden_size, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        
        # Calculate the new state
        combined_new = torch.cat([x, r * h], dim=1)
        n = torch.tanh(tx * self.linear_n(combined_new))

        h_next = (1 - z) * n + z * h
        return h_next

class GRU_SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(GRU_SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = nn.ModuleList([GRUSRUCell(input_size if i == 0 else hidden_size, hidden_size) 
                                    for i in range(num_layers)])

    def forward(self, x, state : Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert batch_first to time_first

        seq_len, batch_size, _ = x.shape

        if state is None:
            state = self.init_state(batch_size, x.device)

        outputs = []
        h = state[0]
        for t in range(seq_len):
            x_t = x[t]
            new_h = []
            for layer_idx, cell in enumerate(self.cells):
                h_t = cell(x_t, h[layer_idx])
                new_h.append(h_t)
                x_t = h_t  # Output of current layer is input to the next
            h = torch.stack(new_h)
            outputs.append(h[-1])

        outputs = torch.stack(outputs)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # Convert time_first back to batch_first

        return outputs, (h, torch.zeros_like(h))
    
    def init_state(self, batch_size : int, device : torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

if __name__ == '__main__':
    batch_size = 3
    input_size = 15
    hidden_size = 128
    seq_len = 10
    batch_first = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(seq_len, batch_size, input_size).to(device)
    if batch_first:
        x = x.permute(1, 0, 2)

    print(f'Input shape: {x.shape}')
    print(f'Batch first: {batch_first}')

    model = GRU_SRU(input_size, hidden_size, num_layers=2, batch_first=batch_first).to(device)
    out, state = model(x, None)

    print(f'Output shape: {out.shape}')
    print(f'State shape: {state[0].shape}')