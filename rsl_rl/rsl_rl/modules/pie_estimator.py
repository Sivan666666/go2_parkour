from turtle import forward
import numpy as np
from rsl_rl.modules.actor_critic import get_activation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
from torch.nn.utils.parametrizations import spectral_norm



class PIE_estimator(nn.Module):
    def __init__(self,
                 H1=10,  # 本体信息的时间步长
                 H2=5,   # 深度图的时间步长
                 proprio_dim=48,  # 本体信息维度
                 depth_height=64,  # 深度图高度
                 depth_width=64,   # 深度图宽度
                 depth_channels=1,  # 深度图通道数
                 mlp_hidden_dims=[256, 128],  # MLP编码器隐藏层维度
                 cnn_channels=[32, 64, 128],  # CNN编码器通道数
                 transformer_nhead=8,  # Transformer注意力头数
                 transformer_layers=4,  # Transformer层数
                 transformer_dim=256,  # Transformer隐藏维度
                 gru_hidden_dim=256,  # GRU隐藏维度
                 gru_layers=2,  # GRU层数
                 velocity_dim=3,  # 速度向量维度
                 foot_height_dim=4,  # 脚高维度
                 scandot_latent_dim=64,  # scandot潜在向量维度
                 global_latent_dim=128,  # 全局潜在向量维度
                 activation="elu",
                 **kwargs):
        super(PIE_estimator, self).__init__()
        
        self.H1 = H1
        self.H2 = H2
        self.proprio_dim = proprio_dim
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.depth_channels = depth_channels
        self.velocity_dim = velocity_dim
        self.foot_height_dim = foot_height_dim
        self.scandot_latent_dim = scandot_latent_dim
        self.global_latent_dim = global_latent_dim
        self.transformer_dim = transformer_dim
        self.gru_hidden_dim = gru_hidden_dim
        
        activation_fn = get_activation(activation)
        
        # 1. MLP编码器用于本体信息
        mlp_layers = []
        mlp_layers.append(nn.Linear(proprio_dim, mlp_hidden_dims[0]))
        mlp_layers.append(activation_fn)
        for i in range(len(mlp_hidden_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_hidden_dims[i], mlp_hidden_dims[i + 1]))
            mlp_layers.append(activation_fn)
        mlp_layers.append(nn.Linear(mlp_hidden_dims[-1], transformer_dim))
        self.proprio_encoder = nn.Sequential(*mlp_layers)
        
        # 2. CNN编码器用于深度图
        cnn_layers = []
        in_channels = depth_channels
        for out_channels in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            cnn_layers.append(nn.BatchNorm2d(out_channels))
            cnn_layers.append(activation_fn)
            in_channels = out_channels
        self.depth_encoder = nn.Sequential(*cnn_layers)
        
        # 计算CNN输出的展平维度
        self.cnn_output_size = self._get_cnn_output_size()
        self.depth_projector = nn.Linear(self.cnn_output_size, transformer_dim)
        
        # 3. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim * 4,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 4. GRU
        self.gru = nn.GRU(
            input_size=transformer_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )
        
        # 5. 输出头
        self.velocity_head = nn.Linear(gru_hidden_dim, velocity_dim)
        self.foot_height_head = nn.Linear(gru_hidden_dim, foot_height_dim)
        self.scandot_head = nn.Linear(gru_hidden_dim, scandot_latent_dim)
        self.global_latent_head = nn.Linear(gru_hidden_dim, global_latent_dim)
        
        # 6. Scandot解码器
        scandot_decoder_layers = []
        scandot_decoder_layers.append(nn.Linear(scandot_latent_dim, 128))
        scandot_decoder_layers.append(activation_fn)
        scandot_decoder_layers.append(nn.Linear(128, 256))
        scandot_decoder_layers.append(activation_fn)
        scandot_decoder_layers.append(nn.Linear(256, scandot_latent_dim))
        self.scandot_decoder = nn.Sequential(*scandot_decoder_layers)
        
        # 7. 状态预测解码器
        state_decoder_input_dim = global_latent_dim + velocity_dim + foot_height_dim + scandot_latent_dim
        state_decoder_layers = []
        state_decoder_layers.append(nn.Linear(state_decoder_input_dim, 256))
        state_decoder_layers.append(activation_fn)
        state_decoder_layers.append(nn.Linear(256, 512))
        state_decoder_layers.append(activation_fn)
        state_decoder_layers.append(nn.Linear(512, proprio_dim))
        self.state_decoder = nn.Sequential(*state_decoder_layers)
        
    def _get_cnn_output_size(self):
        """计算CNN输出的展平大小"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.depth_channels, self.depth_height, self.depth_width)
            output = self.depth_encoder(dummy_input)
            return output.view(1, -1).shape[1]
    
    def forward(self, proprio_history, depth_history):
        """
        前向传播
        Args:
            proprio_history: (batch_size, H1, proprio_dim) - 本体信息历史
            depth_history: (batch_size, H2, depth_channels, depth_height, depth_width) - 深度图历史
        Returns:
            velocity_pred: (batch_size, velocity_dim) - 预测的速度
            foot_height_pred: (batch_size, foot_height_dim) - 预测的脚高
            scandot_latent: (batch_size, scandot_latent_dim) - scandot潜在向量
            global_latent: (batch_size, global_latent_dim) - 全局潜在向量
        """
        batch_size = proprio_history.shape[0]
        
        # 1. 编码本体信息
        proprio_encoded = self.proprio_encoder(proprio_history)  # (batch, H1, transformer_dim)
        
        # 2. 编码深度图
        depth_flat = depth_history.view(-1, self.depth_channels, self.depth_height, self.depth_width)
        depth_features = self.depth_encoder(depth_flat)  # (batch*H2, C, H, W)
        depth_features = depth_features.view(batch_size, self.H2, -1)  # (batch, H2, cnn_output_size)
        depth_encoded = self.depth_projector(depth_features)  # (batch, H2, transformer_dim)
        
        # 3. 合并序列并通过Transformer
        combined_sequence = torch.cat([proprio_encoded, depth_encoded], dim=1)  # (batch, H1+H2, transformer_dim)
        transformer_output = self.transformer_encoder(combined_sequence)  # (batch, H1+H2, transformer_dim)
        
        # 4. 通过GRU
        gru_output, _ = self.gru(transformer_output)  # (batch, H1+H2, gru_hidden_dim)
        
        # 使用最后一个时间步的输出
        final_output = gru_output[:, -1, :]  # (batch, gru_hidden_dim)
        
        # 5. 生成输出
        velocity_pred = self.velocity_head(final_output)
        foot_height_pred = self.foot_height_head(final_output)
        scandot_latent = self.scandot_head(final_output)
        global_latent = self.global_latent_head(final_output)
        
        return velocity_pred, foot_height_pred, scandot_latent, global_latent
    
    def compute_loss(self, proprio_history, depth_history, velocity_gt, foot_height_gt, scandot_gt, next_proprio_gt):
        """
        计算训练损失
        Args:
            proprio_history: (batch_size, H1, proprio_dim)
            depth_history: (batch_size, H2, depth_channels, depth_height, depth_width)
            velocity_gt: (batch_size, velocity_dim) - 真实速度
            foot_height_gt: (batch_size, foot_height_dim) - 真实脚高
            scandot_gt: (batch_size, scandot_latent_dim) - 真实scandot
            next_proprio_gt: (batch_size, proprio_dim) - 下一时刻的真实本体信息
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 前向传播
        velocity_pred, foot_height_pred, scandot_latent, global_latent = self.forward(proprio_history, depth_history)
        
        # 1. 速度损失
        velocity_loss = F.mse_loss(velocity_pred, velocity_gt)
        
        # 2. 脚高损失
        foot_height_loss = F.mse_loss(foot_height_pred, foot_height_gt)
        
        # 3. Scandot重建损失
        scandot_reconstructed = self.scandot_decoder(scandot_latent)
        scandot_loss = F.mse_loss(scandot_reconstructed, scandot_gt)
        
        # 4. 状态预测损失
        decoder_input = torch.cat([global_latent, velocity_pred, foot_height_pred, scandot_latent], dim=-1)
        next_proprio_pred = self.state_decoder(decoder_input)
        state_prediction_loss = F.mse_loss(next_proprio_pred, next_proprio_gt)
        
        # 总损失
        total_loss = velocity_loss + foot_height_loss + scandot_loss + state_prediction_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'velocity_loss': velocity_loss.item(),
            'foot_height_loss': foot_height_loss.item(),
            'scandot_loss': scandot_loss.item(),
            'state_prediction_loss': state_prediction_loss.item()
        }
        
        return total_loss, loss_dict
    
    def inference(self, proprio_history, depth_history):
        """
        推理接口
        Args:
            proprio_history: (batch_size, H1, proprio_dim)
            depth_history: (batch_size, H2, depth_channels, depth_height, depth_width)
        Returns:
            velocity_pred: 预测的速度
            foot_height_pred: 预测的脚高
            scandot_latent: scandot潜在向量
            global_latent: 全局潜在向量
        """
        with torch.no_grad():
            return self.forward(proprio_history, depth_history)
    
    def train_step(self, optimizer, proprio_history, depth_history, velocity_gt, foot_height_gt, scandot_gt, next_proprio_gt):
        """
        单步训练
        Args:
            optimizer: 优化器
            其他参数同compute_loss
        Returns:
            loss_dict: 损失字典
        """
        optimizer.zero_grad()
        total_loss, loss_dict = self.compute_loss(
            proprio_history, depth_history, velocity_gt, 
            foot_height_gt, scandot_gt, next_proprio_gt
        )
        total_loss.backward()
        optimizer.step()
        
        return loss_dict
    

def test_pie_estimator():
    """
    测试PIE_estimator网络的正确性
    """
    print("=" * 50)
    print("开始测试 PIE_estimator 网络")
    print("=" * 50)
    
    # 1. 设置参数
    batch_size = 4
    H1 = 10
    H2 = 5
    proprio_dim = 48
    depth_height = 64
    depth_width = 64
    depth_channels = 1
    velocity_dim = 3
    foot_height_dim = 4
    scandot_latent_dim = 64
    global_latent_dim = 128
    
    # 2. 创建网络
    print("\n创建 PIE_estimator 网络...")
    estimator = PIE_estimator(
        H1=H1,
        H2=H2,
        proprio_dim=proprio_dim,
        depth_height=depth_height,
        depth_width=depth_width,
        depth_channels=depth_channels,
        mlp_hidden_dims=[256, 128],
        cnn_channels=[32, 64, 128],
        transformer_nhead=8,
        transformer_layers=2,
        transformer_dim=256,
        gru_hidden_dim=256,
        gru_layers=2,
        velocity_dim=velocity_dim,
        foot_height_dim=foot_height_dim,
        scandot_latent_dim=scandot_latent_dim,
        global_latent_dim=global_latent_dim,
        activation="elu"
    )
    
    # 统计参数数量
    total_params = sum(p.numel() for p in estimator.parameters())
    trainable_params = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 3. 创建模拟数据
    print("\n创建模拟数据...")
    proprio_history = torch.randn(batch_size, H1, proprio_dim)
    depth_history = torch.randn(batch_size, H2, depth_channels, depth_height, depth_width)
    velocity_gt = torch.randn(batch_size, velocity_dim)
    foot_height_gt = torch.randn(batch_size, foot_height_dim)
    scandot_gt = torch.randn(batch_size, scandot_latent_dim)
    next_proprio_gt = torch.randn(batch_size, proprio_dim)
    
    print(f"本体信息历史形状: {proprio_history.shape}")
    print(f"深度图历史形状: {depth_history.shape}")
    print(f"真实速度形状: {velocity_gt.shape}")
    print(f"真实脚高形状: {foot_height_gt.shape}")
    print(f"真实scandot形状: {scandot_gt.shape}")
    print(f"下一时刻本体信息形状: {next_proprio_gt.shape}")
    
    # 4. 测试前向传播
    print("\n" + "=" * 50)
    print("测试前向传播...")
    print("=" * 50)
    try:
        velocity_pred, foot_height_pred, scandot_latent, global_latent = estimator.forward(
            proprio_history, depth_history
        )
        print("✓ 前向传播成功!")
        print(f"  - 预测速度形状: {velocity_pred.shape} (期望: {(batch_size, velocity_dim)})")
        print(f"  - 预测脚高形状: {foot_height_pred.shape} (期望: {(batch_size, foot_height_dim)})")
        print(f"  - Scandot潜在向量形状: {scandot_latent.shape} (期望: {(batch_size, scandot_latent_dim)})")
        print(f"  - 全局潜在向量形状: {global_latent.shape} (期望: {(batch_size, global_latent_dim)})")
        
        # 验证输出形状
        assert velocity_pred.shape == (batch_size, velocity_dim), "速度预测形状错误"
        assert foot_height_pred.shape == (batch_size, foot_height_dim), "脚高预测形状错误"
        assert scandot_latent.shape == (batch_size, scandot_latent_dim), "Scandot潜在向量形状错误"
        assert global_latent.shape == (batch_size, global_latent_dim), "全局潜在向量形状错误"
        print("✓ 所有输出形状验证通过!")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    # 5. 测试损失计算
    print("\n" + "=" * 50)
    print("测试损失计算...")
    print("=" * 50)
    try:
        total_loss, loss_dict = estimator.compute_loss(
            proprio_history, depth_history, velocity_gt, 
            foot_height_gt, scandot_gt, next_proprio_gt
        )
        print("✓ 损失计算成功!")
        print(f"  - 总损失: {loss_dict['total_loss']:.6f}")
        print(f"  - 速度损失: {loss_dict['velocity_loss']:.6f}")
        print(f"  - 脚高损失: {loss_dict['foot_height_loss']:.6f}")
        print(f"  - Scandot损失: {loss_dict['scandot_loss']:.6f}")
        print(f"  - 状态预测损失: {loss_dict['state_prediction_loss']:.6f}")
        
        # 验证损失是否为有效数值
        assert not torch.isnan(total_loss), "总损失为NaN"
        assert not torch.isinf(total_loss), "总损失为Inf"
        print("✓ 损失值验证通过!")
        
    except Exception as e:
        print(f"✗ 损失计算失败: {e}")
        return False
    
    # 6. 测试推理接口
    print("\n" + "=" * 50)
    print("测试推理接口...")
    print("=" * 50)
    try:
        velocity_pred_inf, foot_height_pred_inf, scandot_latent_inf, global_latent_inf = estimator.inference(
            proprio_history, depth_history
        )
        print("✓ 推理接口测试成功!")
        print(f"  - 预测速度形状: {velocity_pred_inf.shape}")
        print(f"  - 预测脚高形状: {foot_height_pred_inf.shape}")
        print(f"  - Scandot潜在向量形状: {scandot_latent_inf.shape}")
        print(f"  - 全局潜在向量形状: {global_latent_inf.shape}")
        
        # 验证推理模式不产生梯度
        assert not velocity_pred_inf.requires_grad, "推理模式不应产生梯度"
        print("✓ 推理模式验证通过!")
        
    except Exception as e:
        print(f"✗ 推理接口失败: {e}")
        return False
    
    # 7. 测试训练步骤
    print("\n" + "=" * 50)
    print("测试训练步骤...")
    print("=" * 50)
    try:
        optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-4)
        
        # 记录初始参数
        initial_param = next(estimator.parameters()).clone()
        
        # 执行一步训练
        loss_dict = estimator.train_step(
            optimizer, proprio_history, depth_history, 
            velocity_gt, foot_height_gt, scandot_gt, next_proprio_gt
        )
        
        print("✓ 训练步骤执行成功!")
        print(f"  - 训练后损失: {loss_dict['total_loss']:.6f}")
        
        # 验证参数已更新
        updated_param = next(estimator.parameters())
        assert not torch.equal(initial_param, updated_param), "参数未更新"
        print("✓ 参数更新验证通过!")
        
    except Exception as e:
        print(f"✗ 训练步骤失败: {e}")
        return False
    
    # 8. 测试多批次训练
    print("\n" + "=" * 50)
    print("测试多批次训练...")
    print("=" * 50)
    try:
        num_iterations = 5
        print(f"执行 {num_iterations} 次训练迭代...")
        
        for i in range(num_iterations):
            # 生成新的随机数据
            proprio_history = torch.randn(batch_size, H1, proprio_dim)
            depth_history = torch.randn(batch_size, H2, depth_channels, depth_height, depth_width)
            velocity_gt = torch.randn(batch_size, velocity_dim)
            foot_height_gt = torch.randn(batch_size, foot_height_dim)
            scandot_gt = torch.randn(batch_size, scandot_latent_dim)
            next_proprio_gt = torch.randn(batch_size, proprio_dim)
            
            loss_dict = estimator.train_step(
                optimizer, proprio_history, depth_history,
                velocity_gt, foot_height_gt, scandot_gt, next_proprio_gt
            )
            
            print(f"  迭代 {i+1}/{num_iterations} - 总损失: {loss_dict['total_loss']:.6f}")
        
        print("✓ 多批次训练测试通过!")
        
    except Exception as e:
        print(f"✗ 多批次训练失败: {e}")
        return False
    
    # 9. 测试不同批次大小
    print("\n" + "=" * 50)
    print("测试不同批次大小...")
    print("=" * 50)
    try:
        for bs in [1, 2, 8, 16]:
            proprio_history_test = torch.randn(bs, H1, proprio_dim)
            depth_history_test = torch.randn(bs, H2, depth_channels, depth_height, depth_width)
            
            outputs = estimator.inference(proprio_history_test, depth_history_test)
            print(f"  批次大小 {bs:2d} - 输出形状: {outputs[0].shape}")
            
            assert outputs[0].shape[0] == bs, f"批次大小 {bs} 验证失败"
        
        print("✓ 不同批次大小测试通过!")
        
    except Exception as e:
        print(f"✗ 不同批次大小测试失败: {e}")
        return False
    
    # 10. 总结
    print("\n" + "=" * 50)
    print("✓✓✓ 所有测试通过! ✓✓✓")
    print("=" * 50)
    print("\n网络结构概览:")
    print(estimator)
    
    return True


if __name__ == "__main__":
    # 设置随机种子以保证可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    success = test_pie_estimator()
    
    if success:
        print("\n🎉 PIE_estimator 网络测试完成,所有功能正常!")
    else:
        print("\n❌ PIE_estimator 网络测试失败,请检查错误信息!")