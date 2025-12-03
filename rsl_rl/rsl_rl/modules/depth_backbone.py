import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision
from transformers import ViTModel, ViTConfig
import copy
from depth_anything_3.api import DepthAnything3

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

# # 使用别名保持兼容性
# RecurrentDepthBackbone = RecurrentDepthBackbone_LocoTransformer
# DepthOnlyFCBackbone58x87 = DepthBackboneLocoTransformer

# RecurrentDepthBackbone = RecurrentDepthBackbone_GRU
# DepthOnlyFCBackbone58x87 = DepthTransformerBackbone

# RecurrentDepthBackbone = RecurrentDepthBackbone_Attention
# DepthOnlyFCBackbone58x87 = DepthResNetBackbone

class RecurrentDepthBackbone_XYH(nn.Module):
    """
    针对空心楼梯优化的 RNN 骨干：
    1. 使用 LSTM 代替 GRU (更好的记忆保持/物体恒存性)。
    2. 加入 LayerNorm (稳定训练)。
    3. 加深 Output MLP (更好的解码能力)。
    """
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        
        # 1. 输入融合层 (保持不变)
        prop_dim = 53 if env_cfg is None else env_cfg.env.n_proprio
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + prop_dim, 128),
            activation,
            nn.Linear(128, 32)
        )
        
        # 2. 核心修改：使用 LSTM 并增加 Hidden Size (可选，保持512通常够用)
        # LSTM 相比 GRU 更擅长“在输入缺失时保持记忆”
        self.rnn_hidden_dim = 512
        self.rnn = nn.LSTM(input_size=32, hidden_size=self.rnn_hidden_dim, batch_first=True)
        
        # 3. 核心修改：加入 LayerNorm
        # 这有助于防止记忆在时间步长中衰减或爆炸
        self.layer_norm = nn.LayerNorm(self.rnn_hidden_dim)

        # 4. 核心修改：加深 Output MLP
        # 从 memory 解码出 state 需要非线性变换
        self.output_mlp = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, 256), # 先压缩一下
            activation,
            nn.Linear(256, 32 + 2),
            last_activation
        )
        
        # LSTM 的 hidden state 是一个 tuple (h, c)
        self.hidden_states = None

    def forward(self, depth_image, proprioception, rgb_image):
        # [Batch, 32]
        depth_image = self.base_backbone(depth_image, rgb_image)
        
        # [Batch, 32]
        combined_input = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        
        # LSTM Forward
        # input: [Batch, Seq=1, Feature=32]
        # output: [Batch, Seq=1, Hidden=512]
        rnn_out, self.hidden_states = self.rnn(combined_input[:, None, :], self.hidden_states)
        
        # 取出序列维度
        rnn_out = rnn_out.squeeze(1)
        
        # Apply Layer Norm
        rnn_out = self.layer_norm(rnn_out)
        
        # Decode
        depth_latent = self.output_mlp(rnn_out)
        
        return depth_latent

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            # LSTM 的 hidden_states 是 (h, c) 的元组，需要分别 detach
            h, c = self.hidden_states
            self.hidden_states = (h.detach().clone(), c.detach().clone())
            
    def reset_hidden_states(self, batch_size, device):
        # 某些算法可能需要显式重置
        self.hidden_states = None


class DepthOnlyFCBackbone58x87_XYH(nn.Module):
    """
    针对空心楼梯优化的深度图骨干网络。
    特点：
    1. 移除 MaxPool，使用 Strided Conv 保留细微边缘特征。
    2. 网络加深，增强特征提取。
    """
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [Layer 1] 初始特征提取
            # 使用 padding=2 保持分辨率，先不急着缩小，看清细线条
            # 输入: [1, H, W] -> 输出: [32, H, W]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5, padding=2),
            activation,
            
            # [Layer 2] 第一次智能降采样
            # 改用 Stride=2 的卷积代替 MaxPool
            # 相比 MaxPool，它能保留更多纹理信息
            # 输出: [48, H/2, W/2]
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=2, padding=1),
            activation,

            # [Layer 3] 中间层强化
            # 保持分辨率，加深对几何形状的理解
            # 输出: [48, H/2, W/2]
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            activation,

            # [Layer 4] 第二次智能降采样
            # 再次缩小，提取高层抽象特征
            # 输出: [64, H/4, W/4]
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=1),
            activation,
            
            # 拉平
            nn.Flatten(),
            # 全连接层映射到目标维度
            # [32, 25, 39]
            nn.Linear(21120, 128),
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


class DepthAnythingTensorWrapper(nn.Module):
    def __init__(self, encoder="depth-anything/DA3METRIC-LARGE", device="cuda"):
        super().__init__()
        print(f"Loading Depth Anything V3 ({encoder})... This may take a while.")
        
        # 1. 加载模型
        # 注意：这里我们直接加载底层模型，不使用 .inference() 这种带 numpy 转换的高级 API
        try:
            self.model = DepthAnything3.from_pretrained(encoder).to(device)
        except Exception as e:
            print(f"Error loading DA3: {e}")
            raise e

        # 2. 冻结参数 (极度重要！防止显存爆炸和破坏预训练权重)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # DA3 标准输入参数
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        # DA3 推荐的推理分辨率，太小效果会很差

    def forward(self, rgb_images):
        """
        输入: rgb_images [Batch, 3, 58, 87], 范围应该是 0-1 (Float)
        输出: depth_map [Batch, 1, 58, 87], 范围 0-1 (Float, Normalized)
        """
        # 1. 严格的上下文保护，确保不计算梯度
        with torch.no_grad():


            # 3. 标准化 (Normalize)
            # (Image - Mean) / Std
            images_norm = (rgb_images - self.mean) / self.std

            # 4. 推理 (Inference)
            # DA3 的 forward 通常直接返回深度图，或者是一个 list/dict
            # 针对 Metric 模型，通常返回的是真实深度（米）
            da_output = self.model(images_norm)
            
            # 处理可能的输出格式 (有些版本返回 list)
            if isinstance(da_output, (list, tuple)):
                raw_depth = da_output[0]
            elif isinstance(da_output, dict):
                raw_depth = da_output['depth']
            else:
                raw_depth = da_output

            # 5. 下采样回 RL 分辨率 (Downsample)
            # 注意：raw_depth 可能是 [B, H, W]，需要 unsqueeze 增加通道维
            if raw_depth.dim() == 3:
                raw_depth = raw_depth.unsqueeze(1)
            depth_small = F.interpolate(raw_depth, size=(58, 87), mode='bilinear', align_corners=False)

            # 6. 归一化 (Instance Normalization)
            # 将每个样本的深度归一化到 0-1 之间，作为“相对深度”特征
            # 这对融合非常关键，因为 Metric Depth 的绝对数值可能波动，相对形状更重要
            batch_min = depth_small.flatten(2).min(dim=2, keepdim=True)[0].unsqueeze(3) # [B, 1, 1, 1]
            batch_max = depth_small.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(3) # [B, 1, 1, 1]
            
            # 防止除以 0
            depth_normalized = (depth_small - batch_min) / (batch_max - batch_min + 1e-6)

            return depth_normalized.detach() # 再次确保切断梯度


class DepthAnythingTensorWrapper(nn.Module):
    def __init__(self, encoder="depth-anything/DA3METRIC-LARGE", device="cuda"):
        super().__init__()
        print(f"Loading Depth Anything V3 ({encoder})... This may take a while.")
        
        # 1. 加载模型
        # 注意：这里我们直接加载底层模型，不使用 .inference() 这种带 numpy 转换的高级 API
        try:
            self.model = DepthAnything3.from_pretrained(encoder).to(device)
        except Exception as e:
            print(f"Error loading DA3: {e}")
            raise e

        # # =========================================================================
        # # [暴力修复补丁] 递归搜索整个模型树，强制修复 interaction_indexes
        # # =========================================================================
        # def recursive_patch(module, path="model"):
        #     patched = False
        #     # 1. 检查当前模块是否有 interaction_indexes 属性
        #     if hasattr(module, 'interaction_indexes'):
        #         val = getattr(module, 'interaction_indexes')
        #         if val is None:
        #             print(f"🔴 [DEBUG] Found buggy attribute at: {path}.interaction_indexes = None")
        #             # 针对 ViT-Large 的修复参数
        #             fixed_indexes = [4, 11, 17, 23] 
        #             setattr(module, 'interaction_indexes', fixed_indexes)
        #             print(f"🟢 [DEBUG] Patched successfully! Set to: {fixed_indexes}")
        #             patched = True
            
        #     # 2. 递归遍历所有子模块
        #     for name, child in module.named_children():
        #         if recursive_patch(child, path=f"{path}.{name}"):
        #             patched = True
            
        #     return patched

        # print("Searching for broken interaction_indexes...")
        # # 对整个模型进行地毯式搜索
        # if not recursive_patch(self.model):
        #     print("⚠️ [WARNING] Could not find any 'interaction_indexes' that is None.")
        #     print("If the code crashes, check if the model structure has changed.")
        # else:
        #     print("✅ [INFO] DA3 Model patched successfully.")
        # # =========================================================================

        # 2. 冻结参数 (极度重要！防止显存爆炸和破坏预训练权重)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 注册 buffer，确保 mean/std 自动跟随模型 device
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.infer_size = (70, 98)  

    def forward(self, rgb_images):
        with torch.no_grad():
            # 1. Resize (保持 4D: [B, 3, 518, 518])
            images_resized = F.interpolate(rgb_images, size=self.infer_size, mode='bilinear', align_corners=False)
            # 2. Normalize (保持 4D)
            images_norm = (images_resized - self.mean) / self.std

            # 3. --- [核心修复] 增加维度 ---
            # DINOv2 Backbone 期待 5D 输入: [Batch, Shots, Channels, Height, Width]
            # 我们把 Shots 设为 1
            images_input = images_norm.unsqueeze(1) # [B, 1, 3, 518, 518]

            # 4. Forward
            da_output = self.model(images_input)
            
            # 5. 解析输出
            if isinstance(da_output, (list, tuple)):
                raw_depth = da_output[0]
            elif isinstance(da_output, dict):
                raw_depth = da_output['depth']
            else:
                raw_depth = da_output
            
            # --- [核心修复] 处理 5D 输出 ---
            # 如果输出也是 5D [B, 1, H, W] 或 [B, 1, 1, H, W]，就把那个 dummy 维度去掉
            if raw_depth.dim() == 5: 
                 # 假设输出是 [B, 1, 1, H, W] -> squeeze(1) -> [B, 1, H, W]
                 raw_depth = raw_depth.squeeze(1)

            # 确保是 4D [B, 1, H, W]
            if raw_depth.dim() == 3:
                raw_depth = raw_depth.unsqueeze(1)
                
            depth_small = F.interpolate(raw_depth, size=(58, 87), mode='bilinear', align_corners=False)

            # 6. 归一化 (Instance Normalization)
            # 将每个样本的深度归一化到 0-1 之间，作为“相对深度”特征
            # 这对融合非常关键，因为 Metric Depth 的绝对数值可能波动，相对形状更重要
            batch_min = depth_small.flatten(2).min(dim=2, keepdim=True)[0].unsqueeze(3) # [B, 1, 1, 1]
            batch_max = depth_small.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(3) # [B, 1, 1, 1]
            
            # 防止除以 0
            depth_normalized = (depth_small - batch_min) / (batch_max - batch_min + 1e-6)

            return depth_normalized.detach() # 再次确保切断梯度

class RecurrentDepthBackbone_XYH_DA3(nn.Module):
    """
    [双流融合版] 针对空心楼梯优化的 RNN 骨干 (集成 DA3 版)
    """
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        
        # --- 1. 初始化 DA3 处理器 ---
        # 建议：如果显存不够，改用 "depth-anything/DA3-SMALL"
        self.da3_processor = DepthAnythingTensorWrapper(
            encoder="depth-anything/DA3METRIC-LARGE", 
            device=base_backbone.image_compression[0].weight.device # 自动获取设备
        )
        
        # --- 2. 双流骨干网络 ---
        self.base_backbone = base_backbone # 处理 Sensor Depth
        self.rgb_backbone = copy.deepcopy(base_backbone) # 处理 DA3 生成的 Depth
        
        # --- 3. 融合层 ---
        # 两个 32 维 latent + 本体 53 维
        prop_dim = 53 if env_cfg is None else env_cfg.env.n_proprio
        fusion_input_dim = 32 + 32 + prop_dim
        
        self.combination_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            activation,
            nn.Linear(128, 32)
        )
        
        # --- 4. LSTM & Output ---
        self.rnn_hidden_dim = 512
        self.rnn = nn.LSTM(input_size=32, hidden_size=self.rnn_hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.rnn_hidden_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, 256),
            activation,
            nn.Linear(256, 32 + 2),
            last_activation
        )
        
        self.hidden_states = None

    def forward(self, sensor_depth, proprioception, rgb_image):
        """
        sensor_depth: [B, 58, 87] 或 [B, 1, 58, 87] - 真实的物理深度
        proprioception: [B, 53]
        rgb_image: [B, 3, 58, 87] - 原始RGB图像 (范围0-1)
        """
        
        # --- Step 1: DA3 预处理 (Tensor运算, 无numpy, 无梯度) ---
        # [B, 3, 58, 87] -> DA3 -> Normalize -> [B, 1, 58, 87]
        # 注意：这里假设 rgb_image 是 (Batch, 3, H, W)
        # 如果输入是 (Batch, H, W, 3)，请先 permute(0, 3, 1, 2)
        da3_depth = self.da3_processor(rgb_image)
        
        # --- Step 2: 双流特征提取 ---
        # 流1: 物理传感器深度 (Absolute, Sparse)
        sensor_latent = self.base_backbone(sensor_depth) # -> [B, 32]
        
        # 流2: DA3 预测深度 (Relative, Dense)
        # 我们用专门的 rgb_backbone 来学习如何理解 DA3 的输出
        rgb_latent = self.rgb_backbone(da3_depth)      # -> [B, 32]
        
        # --- Step 3: 融合 ---
        combined_input = torch.cat((sensor_latent, rgb_latent, proprioception), dim=-1)
        fused_features = self.combination_mlp(combined_input)
        
        # --- Step 4: 时序记忆 ---
        rnn_out, self.hidden_states = self.rnn(fused_features[:, None, :], self.hidden_states)
        rnn_out = self.layer_norm(rnn_out.squeeze(1))
        
        # --- Step 5: 输出 ---
        depth_latent = self.output_mlp(rnn_out)
        
        return depth_latent

    # ... detach_hidden_states 和 reset 保持不变 ...
    def detach_hidden_states(self):
        if self.hidden_states is not None:
            h, c = self.hidden_states
            self.hidden_states = (h.detach().clone(), c.detach().clone())
    
    def reset_hidden_states(self, batch_size, device):
        self.hidden_states = None

class RecurrentDepthBackbone_XYH_RGB(nn.Module):
    """
    [双流融合版] 针对空心楼梯优化的 RNN 骨干 (修复 RGB 维度版)
    """
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        
        # 1. 保存 Sensor Depth 骨干 (不动它)
        self.base_backbone = base_backbone
        
        # 2. 创建 RGB Depth 骨干
        # 我们复制一份结构，但必须修改第一层以接受 RGB 输入
        self.rgb_backbone = copy.deepcopy(base_backbone)
        
        # 获取 buffer_len (假设 stack 在 dim 1)
        # 如果 env_cfg 不可用，这里可能需要手动指定，比如 2
        self.buffer_len = env_cfg.depth.buffer_len if env_cfg else 2
        
        # --- 修改 RGB 骨干的第一层卷积 ---
        # 假设 image_compression 是一个 Sequential，第一层是 Conv2d
        # RGB 输入通常是 3 通道。总输入通道 = 3 * buffer_len
        rgb_input_channels = 3 * self.buffer_len
        
        first_conv_layer = self.rgb_backbone.image_compression[0]
        # 检查是否确实是卷积层
        if isinstance(first_conv_layer, nn.Conv2d):
            # 创建一个新的卷积层，参数除了 in_channels 外保持一致
            new_conv = nn.Conv2d(
                in_channels=rgb_input_channels,
                out_channels=first_conv_layer.out_channels,
                kernel_size=first_conv_layer.kernel_size,
                stride=first_conv_layer.stride,
                padding=first_conv_layer.padding,
                bias=(first_conv_layer.bias is not None)
            )
            # 替换旧层
            self.rgb_backbone.image_compression[0] = new_conv
        else:
            print("Warning: Could not auto-replace first layer of RGB backbone. Check structure.")

        # 3. 融合层 (保持不变)
        # 假设 backbone 输出 32 维
        backbone_output_dim = 32 
        prop_dim = 53 if env_cfg is None else env_cfg.env.n_proprio
        
        fusion_input_dim = backbone_output_dim + backbone_output_dim + prop_dim
        
        self.combination_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            activation,
            nn.Linear(128, 32)
        )
        
        # 4. LSTM & Output (保持不变)
        self.rnn_hidden_dim = 512
        self.rnn = nn.LSTM(input_size=32, hidden_size=self.rnn_hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.rnn_hidden_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, 256),
            activation,
            nn.Linear(256, 32 + 2),
            last_activation
        )
        
        self.hidden_states = None

    def process_rgb(self, rgb_image):
        """
        专门处理 RGB 图像的辅助函数
        输入 rgb_image: [Batch, Stack, Height, Width, 4/3]
        """
        # 1. 维度调整与 Flatten
        # 取前3个通道(RGB)，忽略 Alpha
        if rgb_image.shape[-1] == 4:
            rgb_image = rgb_image[..., :3]
            
        # [Batch, Stack, H, W, 3] -> [Batch, Stack, 3, H, W]
        x = rgb_image.permute(0, 1, 4, 2, 3)
        
        # 合并 Stack 和 Channel 维度 -> [Batch, Stack*3, H, W]
        # 例如 Stack=2, 那么通道数就是 6
        B, S, C, H, W = x.shape
        x = x.reshape(B, S*C, H, W)
        
        # 归一化 (如果是 0-255 的 uint8，建议转 float 并归一化)
        # 假设外部已经转为 float，如果没有，这里最好除以 255.0
        # x = x.float() / 255.0 
        
        # 2. 通过 RGB Backbone (注意：不能直接调 forward，因为 forward 里可能有 unsqueeze)
        # 直接调用内部的 Sequential 模块
        x = self.rgb_backbone.image_compression(x)
        
        # Flatten [B, C_out, H_out, W_out] -> [B, Features]
        x = x.flatten(start_dim=1)
        
        # 如果 backbone 后面还有 MLP 层，也要过一遍 (取决于 base_backbone 的结构)
        # 假设 DepthOnlyFCBackbone 只有 image_compression 输出特征
        # 如果有 output_mlp 之类的，这里需要补上，例如:
        # x = self.rgb_backbone.output_mlp(x)
        
        return x

    def forward(self, depth_image, proprioception, rgb_image):
        # 1. 物理深度特征 (调用原版 forward)
        # depth_image: [Batch, Stack, H, W] -> 内部自处理
        sensor_latent = self.base_backbone(depth_image)
        
        # 2. RGB 深度特征 (调用我们自定义的处理函数)
        rgb_latent = self.process_rgb(rgb_image)
        
        # 3. 特征拼接
        combined_input = torch.cat((sensor_latent, rgb_latent, proprioception), dim=-1)
        
        # 4. 融合与 RNN
        fused_features = self.combination_mlp(combined_input)
        
        rnn_out, self.hidden_states = self.rnn(fused_features[:, None, :], self.hidden_states)
        rnn_out = rnn_out.squeeze(1)
        
        rnn_out = self.layer_norm(rnn_out)
        depth_latent = self.output_mlp(rnn_out)
        
        return depth_latent

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            h, c = self.hidden_states
            self.hidden_states = (h.detach().clone(), c.detach().clone())
            
    def reset_hidden_states(self, batch_size, device):
        self.hidden_states = None

RecurrentDepthBackbone = RecurrentDepthBackbone_XYH_DA3
DepthOnlyFCBackbone58x87 = DepthOnlyFCBackbone58x87_Original