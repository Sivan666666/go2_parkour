import torch
import torch.nn as nn
import sys
import torchvision
from transformers import ViTModel, ViTConfig

# ==========================================
# 🔥 新版本 1: ResNet Backbone
# ==========================================
class DepthResNetBackbone(nn.Module):
    """
    使用 ResNet18 替代 Transformer 进行深度图特征提取
    更轻量、更快、更稳定
    """
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()
        
        self.scandots_output_dim = scandots_output_dim
        activation = nn.ELU()
        
        # ==========================================
        # 1. 使用预训练 ResNet18（去掉 FC 层）
        # ==========================================
        resnet = torchvision.models.resnet18(pretrained=False)
        
        # 修改第一层卷积（单通道输入）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # 复用 ResNet 的其他层
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # ==========================================
        # 2. 自适应平均池化（输出固定大小）
        # ==========================================
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ==========================================
        # 3. 输出 MLP
        # ==========================================
        self.output_mlp = nn.Sequential(
            nn.Linear(512, 256),
            activation,
            nn.Dropout(0.1),
            nn.Linear(256, 128),
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
        
        print(f"✅ ResNet18 Depth Backbone initialized:")
        print(f"   - Input size: [58, 87]")
        print(f"   - Architecture: ResNet18 (modified for single channel)")
        print(f"   - Output dim: {scandots_output_dim}")
        print(f"   - Total parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
        print(f"   - 估计显存占用: ~{total_params * 4 / 1024**2 * 2:.1f} MB (FP32)")

    def forward(self, images: torch.Tensor):
        """
        输入: images [batch, 58, 87] 深度图
        输出: latent [batch, scandots_output_dim] (32 维)
        """
        # ==========================================
        # 1. ResNet 前向传播
        # ==========================================
        x = images.unsqueeze(1)  # [batch, 1, 58, 87]
        
        # Conv1 + BN + ReLU + MaxPool
        x = self.conv1(x)        # [batch, 64, 29, 44]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)      # [batch, 64, 15, 22]
        
        # ResNet Blocks
        x = self.layer1(x)       # [batch, 64, 15, 22]
        x = self.layer2(x)       # [batch, 128, 8, 11]
        x = self.layer3(x)       # [batch, 256, 4, 6]
        x = self.layer4(x)       # [batch, 512, 2, 3]
        
        # ==========================================
        # 2. 全局平均池化
        # ==========================================
        x = self.avgpool(x)      # [batch, 512, 1, 1]
        x = torch.flatten(x, 1)  # [batch, 512]
        
        # ==========================================
        # 3. MLP 映射到目标维度
        # ==========================================
        latent = self.output_mlp(x)  # [batch, 32]
        latent = self.output_activation(latent)
        
        return latent

class RecurrentDepthBackbone_Attention(nn.Module):
    """
    使用 3 层 Self-Attention 替代 MLP 进行特征融合
    Proprioception 先经过 MLP 编码，再与 Depth 一起送入多层 Attention
    """
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        
        # ==========================================
        # 1. Proprioception MLP 编码器
        # ==========================================
        if env_cfg == None:
            proprio_dim = 53
        else:
            proprio_dim = env_cfg.env.n_proprio
        
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            activation,
            nn.Linear(128, 64),
            activation,
            nn.Linear(64, 32)  # 输出与 depth_latent 相同维度
        )
        
        # ==========================================
        # 2. 🔥 3 层 Self-Attention（Transformer Encoder）
        # ==========================================
        self.embed_dim = 32  # depth 和 proprio 编码后的维度
        self.num_heads = 4
        self.num_attn_layers = 3  # 🔥 从 1 层增加到 3 层
        
        # 创建 Transformer Encoder（3 层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=128,  # FFN 隐藏层维度
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN，更稳定
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_attn_layers  # 🔥 3 层
        )
        
        # ==========================================
        # 3. 融合后的 FFN
        # ==========================================
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            activation,
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # ==========================================
        # 4. GRU 层
        # ==========================================
        self.rnn = nn.GRU(
            input_size=64, 
            hidden_size=768, 
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # ==========================================
        # 5. 输出 MLP
        # ==========================================
        self.output_mlp = nn.Sequential(
            nn.Linear(768, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 32+2),
            last_activation
        )
        self.hidden_states = None
        
        # ==========================================
        # 🔥 计算各部分参数量
        # ==========================================
        proprio_params = sum(p.numel() for p in self.proprio_encoder.parameters())
        attn_params = sum(p.numel() for p in self.transformer_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_mlp.parameters())
        gru_params = sum(p.numel() for p in self.rnn.parameters())
        output_params = sum(p.numel() for p in self.output_mlp.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"✅ Recurrent Depth Backbone with 3-Layer Attention initialized:")
        print(f"   ==========================================")
        print(f"   模块分解:")
        print(f"   - Proprio Encoder:     {proprio_params:,} ({proprio_params/1e3:.1f}K)")
        print(f"   - 🔥 Transformer (3层): {attn_params:,} ({attn_params/1e3:.1f}K)")
        print(f"     * Attention:         {self.num_heads} heads × {self.num_attn_layers} layers")
        print(f"     * Embed dim:         {self.embed_dim}")
        print(f"     * FFN hidden:        128")
        print(f"   - Fusion MLP:          {fusion_params:,} ({fusion_params/1e3:.1f}K)")
        print(f"   - GRU (2层):           {gru_params:,} ({gru_params/1e3:.1f}K)")
        print(f"   - Output MLP:          {output_params:,} ({output_params/1e3:.1f}K)")
        print(f"   ==========================================")
        print(f"   总参数量:              {total_params:,} (~{total_params/1e6:.2f}M)")
        print(f"   估计显存占用:          ~{total_params * 4 / 1024**2 * 2:.1f} MB (FP32)")

    def forward(self, depth_image, proprioception):
        batch_size = depth_image.shape[0]
        
        # ==========================================
        # 1. 编码 Depth 和 Proprioception
        # ==========================================
        depth_latent = self.base_backbone(depth_image)  # [batch, 32]
        proprio_latent = self.proprio_encoder(proprioception)  # [batch, 32]
        
        # ==========================================
        # 2. 拼接成序列（作为 Transformer 的输入）
        # ==========================================
        # 将 depth 和 proprio 作为两个 token
        tokens = torch.stack([depth_latent, proprio_latent], dim=1)  # [batch, 2, 32]
        
        # ==========================================
        # 3. 🔥 3 层 Transformer Encoder
        # ==========================================
        # 每层包含: Multi-Head Attention + FFN + Residual + LayerNorm
        attn_out = self.transformer_encoder(tokens)  # [batch, 2, 32]
        
        # ==========================================
        # 4. 融合特征
        # ==========================================
        # 取平均池化（将两个 token 融合）
        fused_latent = attn_out.mean(dim=1)  # [batch, 32]
        
        # 通过 FFN
        ffn_out = self.fusion_mlp(fused_latent)  # [batch, 64]
        
        # ==========================================
        # 5. GRU 处理
        # ==========================================
        depth_latent, self.hidden_states = self.rnn(
            ffn_out[:, None, :], 
            self.hidden_states
        )  # [batch, 1, 768]
        
        # ==========================================
        # 6. 输出 MLP
        # ==========================================
        depth_latent = self.output_mlp(depth_latent.squeeze(1))  # [batch, 34]
        
        return depth_latent

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach().clone()

class RecurrentDepthBackbone_GRU(nn.Module):
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

# ==========================================
# 🔥 原始版本：轻量级 CNN
# ==========================================
class RecurrentDepthBackbone_Original(nn.Module):
    """
    原始版本：使用浅层网络和单层 GRU
    如需使用此版本，请修改配置文件
    """
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach().clone()


class DepthOnlyFCBackbone58x87_Original(nn.Module):
    """
    原始版本：使用 CNN (Conv2d + MaxPool)
    如需使用此版本，请修改配置文件
    """
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent

# 使用别名保持兼容性
RecurrentDepthBackbone = RecurrentDepthBackbone_Attention 
DepthOnlyFCBackbone58x87 = DepthResNetBackbone