# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025
#
# Description: S4D (Diagonal State Space) implementation - Structured state-space
# model for efficient sequence modeling with linear-time complexity.

import math
import torch
import torch.nn as nn

class S4DOnlyRNN(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_ssm = d_model 

        # 1. dt: Sample in log space
        log_dt = torch.rand(self.n_ssm) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # 2. A: S4D-Lin Initialization
        # Real part fixed to -0.5, Imag part 0 to (N/2)*pi
        real_part = -0.5 * torch.ones(self.n_ssm, self.d_state // 2)
        imag_part = torch.arange(self.d_state // 2).float().repeat(self.n_ssm, 1) * math.pi
        
        # We parameterize log(-real) to enforce negativity
        self.A_real_log = nn.Parameter(torch.log(-real_part)) 
        self.A_imag = nn.Parameter(imag_part)

        # 3. B & C: Complex Normal Init
        self.B = nn.Parameter(torch.randn(self.n_ssm, self.d_state // 2, dtype=torch.cfloat))
        self.C = nn.Parameter(torch.randn(self.n_ssm, self.d_state // 2, dtype=torch.cfloat))

        # 4. D: Skip connection
        self.D = nn.Parameter(torch.randn(self.d_model))
        
        self.activation = nn.GELU()

        # Buffers for ZOH discretization
        self.register_buffer("dA", torch.zeros(self.n_ssm, self.d_state // 2, dtype=torch.cfloat))
        self.register_buffer("dB", torch.zeros(self.n_ssm, self.d_state // 2, dtype=torch.cfloat))
        
        # Flag to track if setup has run
        self.is_setup = False

    def setup_step(self):
        """Discretizes parameters A and B using Zero-Order Hold (ZOH)."""
        dt = torch.exp(self.log_dt) # (H)
        
        # Reconstruct A = -exp(real_log) + i * imag
        A = -torch.exp(self.A_real_log) + 1j * self.A_imag # (H, N/2)
        
        # dtA = A * dt
        dtA = A * dt.unsqueeze(-1) # Broadcast dt to (H, N/2)
        
        # 1. Discretized A: exp(dt * A)
        self.dA = torch.exp(dtA)
        
        # 2. Discretized B: B * (exp(dt * A) - 1) / A
        # Using expm1 for numerical stability with small dt
        self.dB = self.B * (torch.expm1(dtA) / A)
        
        self.is_setup = True

    def default_state(self, batch_size, device=None):
        if device is None: device = self.log_dt.device
        return torch.zeros(
            batch_size, self.d_model, self.d_state // 2, 
            dtype=torch.cfloat, device=device
        )

    def step(self, x, state):
        """
        Single RNN step.
        x: (B, H)
        state: (B, H, N/2)
        """
        if not self.is_setup: 
            self.setup_step()

        # 1. State Update
        # u must broadcast to (B, H, 1) to multiply with dB (H, N/2)
        u = x.unsqueeze(-1) 
        next_state = self.dA * state + self.dB * u

        # 2. Output Calculation: C * x
        # Sum over state dimension (dim=-1) -> (B, H)
        y = torch.sum(self.C * next_state, dim=-1)
        
        # 3. Real part + Skip Connection
        # 2 * Real part accounts for the conjugate symmetric half of the state
        y = 2 * y.real + self.D * x
        
        return self.activation(y), next_state

    def forward(self, x, state=None):
        """
        Recurrent Forward Pass (Slow for training, Good for inference)
        x: (B, L, H)
        """
        batch_size, seq_len, _ = x.shape
        
        # Ensure parameters are discretized
        self.setup_step()
        
        if state is None:
            state = self.default_state(batch_size)
            
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            y_t, state = self.step(x_t, state)
            outputs.append(y_t)
            
        outputs = torch.stack(outputs, dim=1) # (B, L, H)
        
        return outputs, state