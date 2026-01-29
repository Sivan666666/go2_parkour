import torch
import torch.nn as nn
import sys
import torchvision
from transformers import ViTModel, ViTConfig
from .network import LSTM_SRU_Gate
# ==========================================
# 🔥 LocoTransformer 架构 (论文实现)
# ==========================================

class DepthBackboneLocoTransformer(nn.Module):
    """
    LocoTransformer 的视觉编码器部分
    
    论文架构:
    Depth Image [1, 58, 87]
         ↓
    ConvNet (保留空间信息)
         ↓
    4×4 = 16 个 Visual Tokens [16, 128]
         ↓
    Linear Projection
         ↓
    Visual Features [16, 256]
    """
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()
        
        self.scandots_output_dim = scandots_output_dim
        self.spatial_patches = 4  # N=4, 即 4×4 patches
        self.embed_dim = 256      # 论文使用 256
        
        activation = nn.ELU()
        
        # ==========================================
        # 1. Visual ConvNet (保留空间结构)
        # ==========================================
        # 论文: "depth images with a ConvNet"
        # 输入: [batch, 1, 58, 87]
        # 输出: [batch, 128, 4, 4]
        
        self.visual_conv = nn.Sequential(
            # [1, 58, 87] -> [64, 29, 44]
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [64, 15, 22]
            
            # [64, 15, 22] -> [128, 8, 11]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # [128, 8, 11] -> [128, 4, 6]
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 自适应池化到 4×4
            nn.AdaptiveAvgPool2d((self.spatial_patches, self.spatial_patches))  # [128, 4, 4]
        )
        
        # ==========================================
        # 2. 输出投影层
        # ==========================================
        # 将 4×4×128 = 2048 维特征投影到目标维度
        self.output_proj = nn.Sequential(
            nn.Linear(128 * self.spatial_patches * self.spatial_patches, 512),
            activation,
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            activation,
            nn.Dropout(0.1),
            nn.Linear(256, scandots_output_dim)
        )
        
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation
        
        # 计算参数量
        conv_params = sum(p.numel() for p in self.visual_conv.parameters())
        proj_params = sum(p.numel() for p in self.output_proj.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"✅ LocoTransformer Visual Backbone initialized:")
        print(f"   ==========================================")
        print(f"   - Input size:          [58, 87]")
        print(f"   - Spatial patches:     {self.spatial_patches}×{self.spatial_patches} = {self.spatial_patches**2}")
        print(f"   - ConvNet output:      [128, {self.spatial_patches}, {self.spatial_patches}]")
        print(f"   - Output dim:          {scandots_output_dim}")
        print(f"   ==========================================")
        print(f"   参数统计:")
        print(f"   - ConvNet:             {conv_params:,} ({conv_params/1e3:.1f}K)")
        print(f"   - Output Projection:   {proj_params:,} ({proj_params/1e3:.1f}K)")
        print(f"   - Total:               {total_params:,} (~{total_params/1e6:.2f}M)")
        print(f"   - 估计显存:            ~{total_params * 4 / 1024**2 * 2:.1f} MB (FP32)")
        print(f"   ==========================================")

    def forward(self, images: torch.Tensor):
        """
        输入: images [batch, 58, 87] 深度图
        输出: latent [batch, scandots_output_dim] (32 维)
        """
        batch_size = images.shape[0]
        
        # ==========================================
        # 1. ConvNet 提取空间特征
        # ==========================================
        x = images.unsqueeze(1)  # [batch, 1, 58, 87]
        visual_features = self.visual_conv(x)  # [batch, 128, 4, 4]
        
        # ==========================================
        # 2. 展平并投影到输出维度
        # ==========================================
        visual_features = visual_features.flatten(1)  # [batch, 128*4*4=2048]
        latent = self.output_proj(visual_features)    # [batch, scandots_output_dim]
        latent = self.output_activation(latent)
        
        return latent


class RecurrentDepthBackbone_LocoTransformer(nn.Module):
    """
    🔥 LocoTransformer 完整架构 (参考论文 Figure 2)
    
    架构流程:
    1. Proprioception -> Linear -> [1 Token, 256]
    2. Depth Image -> ConvNet -> [16 Tokens, 256] (4×4 spatial)
    3. [Proprio Token, Visual Tokens] -> Shared Transformer (2 layers)
    4. Output Tokens -> Projection Head -> GRU -> Output
    """
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        
        # ==========================================
        # 配置参数
        # ==========================================
        if env_cfg == None:
            self.proprio_dim = 53
        else:
            self.proprio_dim = env_cfg.env.n_proprio
        
        self.embed_dim = 256  # 论文使用 256
        self.num_heads = 8
        self.num_layers = 2   # 论文使用 2 层
        self.spatial_patches = 4  # 从 base_backbone 获取的 spatial patches
        
        # ==========================================
        # 1. 🔥 获取 Visual Features 的维度
        # ==========================================
        # base_backbone 输出是 [batch, scandots_output_dim]
        # 我们需要将其映射到 [batch, num_patches, embed_dim]
        
        # 假设 base_backbone 已经输出了展平的特征
        # 我们需要重新 reshape 成 spatial tokens
        
        # 🔥 方案: 直接使用 base_backbone 的 ConvNet 部分
        if hasattr(base_backbone, 'visual_conv'):
            # 使用 LocoTransformer 的 ConvNet
            self.visual_conv = base_backbone.visual_conv
            self.visual_feature_dim = 128  # ConvNet 输出通道数
        else:
            # 兼容其他 backbone
            raise ValueError("base_backbone 必须是 DepthBackboneLocoTransformer")
        
        # ==========================================
        # 2. Proprioception Encoder (Linear -> Single Token)
        # ==========================================
        self.proprio_linear = nn.Sequential(
            nn.Linear(self.proprio_dim, 256),
            activation,
            nn.Linear(256, self.embed_dim)  # 输出 256 维 token
        )
        
        # ==========================================
        # 3. Visual Token 投影层 (128 -> 256)
        # ==========================================
        self.visual_proj = nn.Linear(self.visual_feature_dim, self.embed_dim)
        
        # ==========================================
        # 4. 🔥 Shared Transformer Encoder (2 layers)
        # ==========================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,        # 256
            nhead=self.num_heads,          # 8
            dim_feedforward=self.embed_dim * 4,  # 1024
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        
        self.shared_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers  # 2 layers
        )
        
        # ==========================================
        # 5. Projection Head
        # ==========================================
        # 总 token 数: 1 (proprio) + 16 (visual 4×4)
        self.num_tokens = 1 + self.spatial_patches ** 2  # 17
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim * self.num_tokens, 512),  # 17*256 -> 512
            activation,
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            activation,
            nn.Dropout(0.1),
            nn.Linear(256, 64)
        )
        
        # ==========================================
        # 6. GRU 层
        # ==========================================
        self.rnn = nn.GRU(
            input_size=64, 
            hidden_size=768, 
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # ==========================================
        # 7. 输出 MLP
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
        # 🔥 参数统计
        # ==========================================
        proprio_params = sum(p.numel() for p in self.proprio_linear.parameters())
        visual_proj_params = sum(p.numel() for p in self.visual_proj.parameters())
        transformer_params = sum(p.numel() for p in self.shared_transformer.parameters())
        proj_params = sum(p.numel() for p in self.projection_head.parameters())
        gru_params = sum(p.numel() for p in self.rnn.parameters())
        output_params = sum(p.numel() for p in self.output_mlp.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"✅ LocoTransformer Recurrent Backbone initialized:")
        print(f"   ==========================================")
        print(f"   模块分解:")
        print(f"   - Proprio Linear:       {proprio_params:,} ({proprio_params/1e3:.1f}K)")
        print(f"   - Visual Projection:    {visual_proj_params:,} ({visual_proj_params/1e3:.1f}K)")
        print(f"   - 🔥 Shared Transformer: {transformer_params:,} ({transformer_params/1e3:.1f}K)")
        print(f"     * Layers:             {self.num_layers}")
        print(f"     * Heads:              {self.num_heads}")
        print(f"     * Embed dim:          {self.embed_dim}")
        print(f"     * Total tokens:       {self.num_tokens} (1 proprio + {self.spatial_patches**2} visual)")
        print(f"   - Projection Head:      {proj_params:,} ({proj_params/1e3:.1f}K)")
        print(f"   - GRU (2层):            {gru_params:,} ({gru_params/1e3:.1f}K)")
        print(f"   - Output MLP:           {output_params:,} ({output_params/1e3:.1f}K)")
        print(f"   ==========================================")
        print(f"   总参数量:               {total_params:,} (~{total_params/1e6:.2f}M)")
        print(f"   估计显存:               ~{total_params * 4 / 1024**2 * 2:.1f} MB (FP32)")

    def forward(self, depth_image, proprioception):
        batch_size = depth_image.shape[0]
        
        # ==========================================
        # 1. Proprioception -> Single Token
        # ==========================================
        proprio_token = self.proprio_linear(proprioception)  # [batch, 256]
        proprio_token = proprio_token.unsqueeze(1)  # [batch, 1, 256]
        
        # ==========================================
        # 2. Depth Image -> ConvNet -> Spatial Tokens
        # ==========================================
        depth_image_expanded = depth_image.unsqueeze(1)  # [batch, 1, 58, 87]
        visual_features = self.visual_conv(depth_image_expanded)  # [batch, 128, 4, 4]
        
        # 展平空间维度 -> tokens
        visual_features = visual_features.flatten(2)  # [batch, 128, 16]
        visual_features = visual_features.transpose(1, 2)  # [batch, 16, 128]
        
        # 投影到 embed_dim
        visual_tokens = self.visual_proj(visual_features)  # [batch, 16, 256]
        
        # ==========================================
        # 3. 拼接所有 tokens
        # ==========================================
        all_tokens = torch.cat([proprio_token, visual_tokens], dim=1)  # [batch, 17, 256]
        
        # ==========================================
        # 4. 🔥 Shared Transformer Encoder
        # ==========================================
        transformer_out = self.shared_transformer(all_tokens)  # [batch, 17, 256]
        
        # ==========================================
        # 5. Projection Head
        # ==========================================
        fused_features = transformer_out.flatten(1)  # [batch, 17*256]
        projected = self.projection_head(fused_features)  # [batch, 64]
        
        # ==========================================
        # 6. GRU 处理
        # ==========================================
        rnn_out, self.hidden_states = self.rnn(
            projected[:, None, :], 
            self.hidden_states
        )  # [batch, 1, 768]
        
        # ==========================================
        # 7. 输出 MLP
        # ==========================================
        output = self.output_mlp(rnn_out.squeeze(1))  # [batch, 34]
        
        return output

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach().clone()


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

class DepthBackbone_SpatialPatches(nn.Module):
    """
    修改后的深度图骨干网络：输出空间 Patch 特征 [Batch, 16, 64]
    而非单个全局向量，以支持 Cross-Attention 检索空间信息。
    """
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()
        self.num_frames = num_frames
        activation = nn.ELU()
        
        # 提取空间特征的卷积层
        self.feature_extractor = nn.Sequential(
            # [1, 58, 87] -> [32, 27, 42]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5, stride=2),
            activation,
            # [32, 27, 42] -> [64, 13, 20]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            activation,
            # 关键：自适应池化到 4x4 的网格
            nn.AdaptiveAvgPool2d((4, 4)) # 输出 shape: [batch, 64, 4, 4]
        )
        
        # 投影层，将通道数映射到 Attention 维度 (如 64)
        self.token_dim = 64
        self.projection = nn.Linear(64, self.token_dim)
        
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        # images: [batch, 58, 87]
        x = images.unsqueeze(1) # [batch, 1, 58, 87]
        x = self.feature_extractor(x) # [batch, 64, 4, 4]
        
        # 展平空间维度：[batch, 64, 16] -> 转置：[batch, 16, 64]
        x = x.flatten(2).transpose(1, 2)
        
        # 投影并激活
        x = self.output_activation(self.projection(x))
        return x # 返回 16 个 Patch Token
    
# 假设 LSTM_SRU_Gate 已从 lstm_sru_gate.py 导入
# from lstm_sru_gate import LSTM_SRU_Gate

class RecurrentDepthBackbone_SRU(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        
        # 获取维度
        self.proprio_dim = 53 if env_cfg is None else env_cfg.env.n_proprio
        self.embed_dim = 64  # 与上面 backbone 的 token_dim 对应
        self.rnn_hidden = 512
        
        # 1. Proprioception 编码器 (将本体感知编码为 Query)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, 128),
            activation,
            nn.Linear(128, self.embed_dim)
        )
        
        # 2. 两层 Cross-Attention
        # Query: Proprio (1 token), Key/Value: Depth Patches (16 tokens)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)
            for _ in range(2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])
        
        # 3. LSTM-SRU-Gate (时序记忆模块)
        self.rnn = LSTM_SRU_Gate(
            input_size=self.embed_dim, 
            hidden_size=self.rnn_hidden, 
            num_layers=2, 
            batch_first=True
        )
        
        # 4. 输出 MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(self.rnn_hidden, 128),
            activation,
            nn.Linear(128, 34), # 对应原本的 32+2 维输出
            last_activation
        )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        # A. 获取深度图 Patch: [batch, 16, 64]
        depth_patches = self.base_backbone(depth_image)
        
        # B. 获取本体感知 Query: [batch, 1, 64]
        proprio_query = self.proprio_encoder(proprioception).unsqueeze(1)
        
        # C. 两次 Cross-Attention 处理
        x = proprio_query
        for i in range(2):
            # Query 来自 proprio, Key/Value 来自 depth patches
            attn_out, _ = self.attn_layers[i](query=x, key=depth_patches, value=depth_patches)
            x = self.norms[i](x + attn_out) # 残差连接 + 层归一化
            
        # D. LSTM-SRU-Gate (记忆模块)
        # x shape: [batch, 1, 64]
        rnn_out, self.hidden_states = self.rnn(x, self.hidden_states)
        
        # E. 映射到最终输出: [batch, 34]
        output = self.output_mlp(rnn_out.squeeze(1))
        return output

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            # SRU 的 hidden_states 是 (h, c) 元组
            h, c = self.hidden_states
            self.hidden_states = (h.detach().clone(), c.detach().clone())

# # 使用别名保持兼容性
# RecurrentDepthBackbone = RecurrentDepthBackbone_LocoTransformer
# DepthOnlyFCBackbone58x87 = DepthBackboneLocoTransformer

# RecurrentDepthBackbone = RecurrentDepthBackbone_GRU
# DepthOnlyFCBackbone58x87 = DepthTransformerBackbone

# RecurrentDepthBackbone = RecurrentDepthBackbone_Attention
# DepthOnlyFCBackbone58x87 = DepthResNetBackbone

RecurrentDepthBackbone = RecurrentDepthBackbone_Original
DepthOnlyFCBackbone58x87 = DepthOnlyFCBackbone58x87_Original