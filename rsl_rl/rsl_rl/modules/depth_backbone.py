import torch
import torch.nn as nn
import sys
import torchvision
from transformers import ViTModel, ViTConfig

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        
        if env_cfg == None:
            # 加深网络深度：3层 MLP
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 256),
                                    activation,
                                    nn.Linear(256, 128),
                                    activation,
                                    nn.Linear(128, 64)
                                )
        else:
            # 加深网络深度：3层 MLP
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 256),
                                        activation,
                                        nn.Linear(256, 128),
                                        activation,
                                        nn.Linear(128, 64)
                                    )
        
        # 加深 GRU：2层，增加隐藏层维度到 768
        self.rnn = nn.GRU(
            input_size=64, 
            hidden_size=768, 
            num_layers=2,      # 🔥 从 1 层增加到 2 层
            batch_first=True,
            dropout=0.1        # 添加 dropout 防止过拟合
        )
        
        # 加深输出 MLP：3层
        self.output_mlp = nn.Sequential(
                                nn.Linear(768, 256),
                                activation,
                                nn.Linear(256, 128),
                                activation,
                                nn.Linear(128, 32+2),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)  # [batch, 32]
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))  # [batch, 64]
        
        # RNN 处理
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)  # [batch, 1, 768]
        depth_latent = self.output_mlp(depth_latent.squeeze(1))  # [batch, 34]
        
        return depth_latent

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach().clone()


import torch
import torch.nn as nn
import math

class DepthTransformerBackbone(nn.Module):
    """轻量级单通道深度图 Transformer（显存友好）"""
    
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()
        
        self.scandots_output_dim = scandots_output_dim
        activation = nn.ELU()
        
        # ==========================================
        # 🔥 轻量级超参数（大幅减少参数量）
        # ==========================================
        self.patch_size = 16         # 🔥 从 8 增加到 16 (减少 patch 数量)
        self.embed_dim = 128         # 🔥 从 256 减少到 128 (减少隐藏维度)
        self.num_heads = 4           # 🔥 从 8 减少到 4 (减少注意力头)
        self.num_layers = 3          # 🔥 从 6 减少到 3 (减少层数)
        self.mlp_ratio = 2           # 🔥 从 4 减少到 2 (FFN 更小)
        
        # 计算 patch 数量
        self.num_patches_h = 58 // self.patch_size  # 3
        self.num_patches_w = 87 // self.patch_size  # 5
        self.num_patches = self.num_patches_h * self.num_patches_w  # 15 (原来 70)
        
        # ==========================================
        # 1. Patch Embedding（单通道卷积）
        # ==========================================
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )
        
        # ==========================================
        # 2. CLS Token 和 Position Embedding
        # ==========================================
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        
        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # ==========================================
        # 3. Transformer Encoder（更轻量）
        # ==========================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * self.mlp_ratio,  # 🔥 256 (原来 1024)
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # 🔥 Pre-LN 更稳定，收敛更快
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # ==========================================
        # 4. 输出 MLP（更简单）
        # ==========================================
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 128),  # 🔥 从 256 减少到 128
            activation,
            nn.Dropout(0.1),
            nn.Linear(128, scandots_output_dim)
        )
        
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"✅ Lightweight Depth Transformer initialized:")
        print(f"   - Input size: [58, 87]")
        print(f"   - Patch size: {self.patch_size}x{self.patch_size}")
        print(f"   - Num patches: {self.num_patches} (减少 {100 - self.num_patches/70*100:.1f}%)")
        print(f"   - Embed dim: {self.embed_dim} (减少 50%)")
        print(f"   - Num layers: {self.num_layers} (减少 50%)")
        print(f"   - Num heads: {self.num_heads} (减少 50%)")
        print(f"   - FFN hidden: {self.embed_dim * self.mlp_ratio} (减少 75%)")
        print(f"   - Output dim: {scandots_output_dim}")
        print(f"   - Total parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
        print(f"   - 估计显存占用: ~{total_params * 4 / 1024**2 * 2:.1f} MB (FP32)")

    def forward(self, images: torch.Tensor):
        """
        输入: images [batch, 58, 87] 深度图
        输出: latent [batch, scandots_output_dim] (32 维)
        """
        batch_size = images.shape[0]
        
        # ==========================================
        # 1. Patch Embedding
        # ==========================================
        x = images.unsqueeze(1)  # [batch, 1, 58, 87]
        x = self.patch_embed(x)  # [batch, 128, 3, 5]
        x = x.flatten(2).transpose(1, 2)  # [batch, 15, 128]
        
        # ==========================================
        # 2. 添加 CLS Token
        # ==========================================
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, 128]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 16, 128]
        
        # ==========================================
        # 3. 添加 Position Embedding
        # ==========================================
        x = x + self.pos_embed  # [batch, 16, 128]
        
        # ==========================================
        # 4. Transformer Encoder
        # ==========================================
        x = self.transformer(x)  # [batch, 16, 128]
        
        # ==========================================
        # 5. 取 CLS Token 作为全局特征
        # ==========================================
        cls_output = x[:, 0]  # [batch, 128]
        
        # ==========================================
        # 6. MLP 映射到目标维度
        # ==========================================
        latent = self.output_mlp(cls_output)  # [batch, 32]
        latent = self.output_activation(latent)
        
        return latent


# 使用别名保持兼容性
DepthOnlyFCBackbone58x87 = DepthTransformerBackbone