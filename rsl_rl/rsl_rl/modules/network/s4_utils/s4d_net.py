# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025
#
# Description: S4 model wrapper for multi-layer S4D (Diagonal State Space) networks
# with encoder and state management.

import torch
import torch.nn as nn
from network.s4_utils.s4d import S4DOnlyRNN

class S4Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, d_state=16, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.encoder = nn.Linear(input_size, hidden_size)
        
        # Use list comprehension for concise layer creation
        self.layers = nn.ModuleList([
            S4DOnlyRNN(d_model=hidden_size, d_state=d_state)
            for _ in range(num_layers)
        ])

    def forward(self, x, state=None):
        # 1. Standardize Input to (B, L, H)
        if not self.batch_first: x = x.transpose(0, 1)
        if x.ndim == 2: x = x.unsqueeze(0)
        
        # 2. Encode and Initialize State
        x = self.encoder(x)
        if state is None:
            state = self.default_state(x.size(0), x.device)

        # 3. Pass through layers
        next_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer(x, state=state[i])
            next_states.append(s)

        # 4. Restore output shape
        if not self.batch_first: x = x.transpose(0, 1)
        
        return x, next_states

    def step(self, x, state=None):
        # 1. Initialize State if missing (e.g. first step)
        if state is None: 
            state = self.default_state(x.size(0), x.device)
            
        # 2. Encode
        x = self.encoder(x)
        
        # 3. Step through layers
        next_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer.step(x, state[i])
            next_states.append(s)
            
        return x, next_states

    def setup_step(self):
        """Pre-calculates dA/dB for all layers. Call before inference loop."""
        for layer in self.layers: 
            layer.setup_step()

    def default_state(self, batch_size, device=None):
        """Get default state (list of layer states)"""
        # Auto-detect device if not provided
        if device is None: device = next(self.parameters()).device
        return [layer.default_state(batch_size, device=device) for layer in self.layers]