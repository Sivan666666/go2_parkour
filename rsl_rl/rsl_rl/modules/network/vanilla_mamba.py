# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025
#
# Description: MambaNet implementation - State-space model based on Mamba
# architecture with selective state transitions and efficient sequence modeling.

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, d_state, d_conv, expand):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.mixer = Mamba(d_model=input_dim, 
                           d_state=d_state, 
                           d_conv=d_conv, 
                           expand=expand)
        
        self.norm = RMSNorm(input_dim)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(x)

        return output

class MambaNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, d_state=16, d_conv=4, expand=2, batch_first=True):
        super(MambaNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.input_seq = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_dim, d_state, d_conv, expand) for _ in range(num_layers)
        ])

    def forward(self, x):
        # input x: (B, L, C)
        x = self.input_seq(x)
        x = F.silu(x)
        for layer in self.layers:
            x = layer(x)
        return x, None

if __name__ == '__main__':
    batch_size = 2
    input_dim = 15
    hidden_dim = 128
    d_state = 16
    d_conv = 4
    expand = 2
    L = 8
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = torch.randn(batch_size, L, input_dim).to(device) * 10
    net = MambaNet(input_dim, hidden_dim, d_state, d_conv, expand).to(device)
    output_L = net(x)
    print(output_L[0].shape)
    print(net)
        