import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(activation_name):
    """辅助函数：获取激活函数"""
    name = activation_name.lower()
    if name == "elu":
        return nn.ELU()
    elif name == "selu":
        return nn.SELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "lrelu":
        return nn.LeakyReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        return nn.ReLU()

def build_mlp(input_dim, hidden_dims, output_dim, activation_fn):
    """辅助函数：构建多层感知机 (MLP)"""
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation_fn)
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

class PIE_estimator(nn.Module):
    def __init__(self, 
                 # --- 核心输入维度 ---
                 prop_dim,              # 单帧本体观测维度 (o_t)
                 prop_history_len,      # H1: 本体历史长度
                 depth_shape,           # (Height, Width): 深度图尺寸
                 depth_history_len,     # H2: 深度图叠加帧数
                 num_scandots,          # 地形高度图(scandots)维度 (m_t)
                 
                 # --- 1. 本体编码器参数 (Optimized based on Appendix) ---
                 # Table V: Prop Enc. MLP {[256, 128], ELU}
                 prop_encoder_hidden_dims=[256, 128],  
                 
                 # --- 2. 深度图编码器参数 (Optimized based on Appendix) ---
                 # Table V: Depth Enc. CNN 
                 # Channels: [32, 64, 64] (Implies 4th layer keeps 64 or maps to embedding)
                 # Kernel: [7, 7, 3, 3], Stride: [4, 4, 2, 2], Padding: [3, 3, 1, 1]
                 cnn_channels=[32, 64, 64, 64], # Explicitly defining 4 layers
                 cnn_kernel_sizes=[7, 7, 3, 3],    
                 cnn_strides=[4, 4, 2, 2],         
                 cnn_paddings=[3, 3, 1, 1],        
                 
                 # --- 3. Transformer & GRU 参数 (Optimized based on Appendix) ---
                 # Table V: Tf. Enc. SA {[256, 256], 64, 1}
                 # Implies d_model=64, nhead=1, dim_feedforward=256
                 d_model=64,            
                 nhead=1,               
                 num_encoder_layers=2,  # 层数保持默认或根据需要调整，论文未明确指出堆叠层数
                 dim_feedforward=256,   # 新增参数：Transformer内部MLP维度
                 
                 # --- 4. 解码器参数 (Optimized based on Appendix) ---
                 # Table V: Est. MLP {[64, 128], ELU}
                 map_decoder_hidden_dims=[64, 128], 
                 dim_z_map=64,          # z_t^m 维度，通常与 d_model 一致
                 
                 state_decoder_hidden_dims=[64, 128],
                 dim_z_latent=64,       # z_t 维度，与 d_model 一致
                 
                 # --- 显式输出头维度 ---
                 output_dim_vel=3,      
                 output_dim_clearance=4,
                 
                 activation="elu",
                 device="cpu"):
        super(PIE_estimator, self).__init__()
        
        self.device = device
        self.activation_name = activation  # 保存激活函数名称
        self.activation = get_activation(activation)
        self.d_model = d_model
        
        # 保存所有配置参数 (用于 save/load)
        self.prop_dim = prop_dim
        self.prop_history_len = prop_history_len
        self.depth_shape = depth_shape
        self.depth_history_len = depth_history_len
        self.num_scandots = num_scandots
        self.prop_encoder_hidden_dims = prop_encoder_hidden_dims
        self.cnn_channels = cnn_channels
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.cnn_strides = cnn_strides
        self.cnn_paddings = cnn_paddings
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.map_decoder_hidden_dims = map_decoder_hidden_dims
        self.dim_z_map = dim_z_map
        self.state_decoder_hidden_dims = state_decoder_hidden_dims
        self.dim_z_latent = dim_z_latent
        self.output_dim_vel = output_dim_vel
        self.output_dim_clearance = output_dim_clearance

        # ======================================================================
        # 1. Encoders (编码器)
        # ======================================================================
        
        # [A] 本体编码器 (MLP)
        # 结构：Flatten(H1) -> MLP([256, 128]) -> d_model(64)
        self.prop_encoder = build_mlp(input_dim=prop_dim * prop_history_len, 
                                      hidden_dims=prop_encoder_hidden_dims, 
                                      output_dim=d_model, 
                                      activation_fn=self.activation)

        # [B] 深度图编码器 (CNN)
        # 结构：4层卷积，参数严格参考 Table V
        cnn_layers = []
        in_channels = 1 
        
        assert len(cnn_channels) == len(cnn_kernel_sizes) == len(cnn_strides) == len(cnn_paddings), \
            "CNN 参数列表长度不一致"
            
        for out_channels, k, s, p in zip(cnn_channels, cnn_kernel_sizes, cnn_strides, cnn_paddings):
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p))
            cnn_layers.append(self.activation)
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # CNN 投影层
        self.cnn_projection = nn.Linear(cnn_channels[-1], d_model)

        # ======================================================================
        # 2. Fusion & Memory (融合与记忆)
        # ======================================================================
        
        # Transformer
        # 参数更新：d_model=64, nhead=1, dim_feedforward=256
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # GRU
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model,num_layers=2,  # 设置为2层或更多
                          batch_first=True)

        # ======================================================================
        # 3. Estimation Heads (估计头)
        # ======================================================================
        
        self.head_vel = nn.Linear(d_model, output_dim_vel)             
        self.head_clearance = nn.Linear(d_model, output_dim_clearance) 
        self.head_map_latent = nn.Linear(d_model, dim_z_map)           
        
        # VAE Heads
        self.head_vae_mu = nn.Linear(d_model, dim_z_latent)
        self.head_vae_logvar = nn.Linear(d_model, dim_z_latent)

        # ======================================================================
        # 4. Decoders (解码器)
        # ======================================================================
        
        # [Decoder 1] Map Decoder
        # 结构：MLP([64, 128]) -> num_scandots
        self.map_decoder = build_mlp(input_dim=dim_z_map,
                                     hidden_dims=map_decoder_hidden_dims,
                                     output_dim=num_scandots,
                                     activation_fn=self.activation)
        
        # [Decoder 2] Next State Decoder
        decoder_input_dim = output_dim_vel + output_dim_clearance + dim_z_map + dim_z_latent
        self.next_state_decoder = build_mlp(input_dim=decoder_input_dim,
                                            hidden_dims=state_decoder_hidden_dims,
                                            output_dim=prop_dim, 
                                            activation_fn=self.activation)

        self.to(device)

    def reparameterize(self, mu, logvar):
        """VAE 重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_backbone(self, prop_history, depth_history):
        """
        前向传播主干:
        1. 独立卷积处理每张深度图 (Batch Folding)
        2. 处理本体信息 (Flatten H1 -> MLP)
        3. Transformer 融合
        4. GRU 记忆
        """
        batch_size = prop_history.shape[0]
        H2 = self.depth_history_len

        # --- 1. Visual Pathway (Time-Distributed CNN) ---
        # Input: (Batch, H2, H, W) -> Fold -> (Batch*H2, 1, H, W)
        cnn_input = depth_history.view(batch_size * H2, 1, depth_history.shape[2], depth_history.shape[3])
        
        # CNN Forward
        cnn_feat = self.cnn(cnn_input) 
        
        # Flatten Spatial: (Batch*H2, C, H', W') -> (Batch*H2, N_pixels, C)
        visual_tokens = cnn_feat.flatten(2).transpose(1, 2)
        # print("Visual tokens shape after CNN and projection:", visual_tokens.shape)
        # Project to d_model
        visual_tokens = self.cnn_projection(visual_tokens)
        # print("Visual tokens shape after CNN and projection:", visual_tokens.shape)
        # Unfold Time: (Batch, H2 * N_pixels, d_model)
        visual_tokens = visual_tokens.view(batch_size, -1, self.d_model)

        # --- 2. Proprioception Pathway ---
        # Flatten H1 steps: (Batch, H1 * prop_dim)
        prop_flat = prop_history.flatten(1) 
        
        # MLP Encoding: (Batch, d_model)
        prop_feat = self.prop_encoder(prop_flat)
        
        # Add sequence dim: (Batch, 1, d_model)
        prop_tokens = prop_feat.unsqueeze(1)
        
        # print(prop_tokens.shape, visual_tokens.shape)
        # --- 3. Fusion (Transformer) ---
        combined_tokens = torch.cat([prop_tokens, visual_tokens], dim=1)
        transformer_out = self.transformer(combined_tokens) 
        
        # --- 4. Memory (GRU) ---
        _, gru_hidden = self.gru(transformer_out) 
        gru_out = gru_hidden[-1] 
        # print("Prop tokens shape:", prop_tokens.shape)
        # print("Visual tokens shape:", visual_tokens.shape)
        # print("Transformer output shape:", transformer_out.shape)
        # print("GRU output shape:", gru_out.shape)
        
        return gru_out

    def forward(self, prop_history, depth_history):
        """训练模式前向传播"""
        gru_out = self.forward_backbone(prop_history, depth_history)

        # Heads
        est_vel = self.head_vel(gru_out)
        est_clearance = self.head_clearance(gru_out)
        z_map = self.head_map_latent(gru_out)
        
        # VAE
        mu = self.head_vae_mu(gru_out)
        logvar = self.head_vae_logvar(gru_out)
        z_latent = self.reparameterize(mu, logvar) 

        # Decoders
        rec_map = self.map_decoder(z_map)
        
        decoder_input = torch.cat([est_vel, est_clearance, z_map, z_latent], dim=1)
        rec_next_state = self.next_state_decoder(decoder_input)

        return {
            "est_vel": est_vel,
            "est_clearance": est_clearance,
            "z_map": z_map,
            "z_latent": z_latent,
            "mu": mu,
            "logvar": logvar,
            "rec_map": rec_map,
            "rec_next_state": rec_next_state
        }

    def compute_loss(self, model_outputs, ground_truth):
        """计算 Loss"""
        est_vel = model_outputs['est_vel']
        est_clearance = model_outputs['est_clearance']
        rec_map = model_outputs['rec_map']
        rec_next_state = model_outputs['rec_next_state']
        mu = model_outputs['mu']
        logvar = model_outputs['logvar']

        gt_vel = ground_truth['velocity']
        gt_clearance = ground_truth['foot_clearance']
        gt_map = ground_truth['scandots']
        gt_next_state = ground_truth['next_state']

        # Loss weights usually need tuning, here implies 1.0
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kl = loss_kl / est_vel.shape[0]

        loss_vel = F.mse_loss(est_vel, gt_vel)
        loss_clearance = F.mse_loss(est_clearance, gt_clearance)
        loss_map = F.mse_loss(rec_map, gt_map)
        loss_next_state = F.mse_loss(rec_next_state, gt_next_state)

        total_loss = loss_kl + loss_next_state + loss_map + loss_vel + loss_clearance

        return total_loss, {
            "loss_kl": loss_kl.item(),
            "loss_vel": loss_vel.item(),
            "loss_clearance": loss_clearance.item(),
            "loss_map": loss_map.item(),
            "loss_next_state": loss_next_state.item()
        }

    def get_parameter_groups(self):
        """参数分组"""
        cnn_params = list(self.cnn.parameters()) + list(self.cnn_projection.parameters())
        gru_params = list(self.gru.parameters())
        map_decoder_params = list(self.map_decoder.parameters())
        state_decoder_params = list(self.next_state_decoder.parameters())
        
        mlp_backbone_params = (
            list(self.prop_encoder.parameters()) + 
            list(self.transformer.parameters()) + 
            list(self.head_vel.parameters()) + 
            list(self.head_clearance.parameters()) + 
            list(self.head_map_latent.parameters()) + 
            list(self.head_vae_mu.parameters()) + 
            list(self.head_vae_logvar.parameters())
        )

        return [
            {'params': cnn_params, 'name': 'cnn'},
            {'params': gru_params, 'name': 'gru'},
            {'params': map_decoder_params, 'name': 'map_decoder'},
            {'params': state_decoder_params, 'name': 'state_decoder'},
            {'params': mlp_backbone_params, 'name': 'mlp_backbone'}
        ]

    def train_one_step(self, optimizer, prop_history, depth_history, ground_truth):
        """单步训练"""
        self.train()
        optimizer.zero_grad()
        
        prop_history = prop_history.to(self.device)
        depth_history = depth_history.to(self.device)
        for k, v in ground_truth.items():
            ground_truth[k] = v.to(self.device)

        outputs = self.forward(prop_history, depth_history)
        loss, loss_details = self.compute_loss(outputs, ground_truth)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss_details

    def inference(self, prop_history, depth_history):
        """推理接口"""
        self.eval()
        with torch.inference_mode():
            prop_history = prop_history.to(self.device)
            depth_history = depth_history.to(self.device)
            
            gru_out = self.forward_backbone(prop_history, depth_history)

            est_vel = self.head_vel(gru_out)
            est_clearance = self.head_clearance(gru_out)
            z_map = self.head_map_latent(gru_out)
            z_latent = self.head_vae_mu(gru_out) 

            return est_vel, est_clearance, z_map, z_latent
        
    def save(self, filepath):
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径 (例如 'model.pt' 或 'checkpoints/pie_estimator.pt')
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'prop_dim': self.prop_dim,
                'prop_history_len': self.prop_history_len,
                'depth_shape': self.depth_shape,
                'depth_history_len': self.depth_history_len,
                'num_scandots': self.num_scandots,
                'prop_encoder_hidden_dims': self.prop_encoder_hidden_dims,
                'cnn_channels': self.cnn_channels,
                'cnn_kernel_sizes': self.cnn_kernel_sizes,
                'cnn_strides': self.cnn_strides,
                'cnn_paddings': self.cnn_paddings,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_encoder_layers': self.num_encoder_layers,
                'dim_feedforward': self.dim_feedforward,
                'map_decoder_hidden_dims': self.map_decoder_hidden_dims,
                'dim_z_map': self.dim_z_map,
                'state_decoder_hidden_dims': self.state_decoder_hidden_dims,
                'dim_z_latent': self.dim_z_latent,
                'output_dim_vel': self.output_dim_vel,
                'output_dim_clearance': self.output_dim_clearance,
                'activation': self.activation_name,
            }
        }
        torch.save(checkpoint, filepath)
        print(f"模型已保存到: {filepath}")

    @classmethod
    def load(cls, filepath, device="cpu", **kwargs):
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
            device: 加载到的设备 ('cpu' 或 'cuda')
            **kwargs: 模型初始化参数 (如果checkpoint中没有保存配置则必须提供)
        
        Returns:
            加载好的 PIE_estimator 实例
        
        Usage:
            # 方式1: 从checkpoint加载配置 (需要保存时包含完整config)
            estimator = PIE_estimator.load('model.pt', device='cuda')
            
            # 方式2: 手动提供配置参数
            estimator = PIE_estimator.load('model.pt', device='cuda',
                                          prop_dim=45, prop_history_len=10, ...)
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # 优先使用checkpoint中的配置,否则使用kwargs
        if 'model_config' in checkpoint and not kwargs:
            model_config = checkpoint['model_config']
            print("从checkpoint加载模型配置")
        else:
            if not kwargs:
                raise ValueError("checkpoint中无配置信息,请通过kwargs提供模型参数")
            model_config = kwargs
            print("使用提供的kwargs作为模型配置")
        
        # 创建模型实例
        model = cls(device=device, **model_config)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"模型已从 {filepath} 加载到 {device}")
        return model

# ...existing code...

def evaluate_reconstruction_errors(estimator, data_prop, data_depth, ground_truth, phase="Before Training"):
    """
    评估并打印重建误差
    
    Args:
        estimator: PIE_estimator 模型
        data_prop: 本体输入数据
        data_depth: 深度图输入数据
        ground_truth: 真实值字典
        phase: 评估阶段名称 ("Before Training" 或 "After Training")
    
    Returns:
        error_dict: 包含各项误差的字典
    """
    print("\n" + "="*60)
    print(f"{phase}")
    print("="*60)
    
    with torch.inference_mode():
        estimator.eval()
        outputs = estimator.forward(data_prop, data_depth)
        
        # 提取输出
        est_vel = outputs['est_vel']
        est_clear = outputs['est_clearance']
        rec_map = outputs['rec_map']
        rec_next_state = outputs['rec_next_state']
        
        # 计算误差
        vel_error = F.mse_loss(est_vel, ground_truth['velocity']).item()
        clear_error = F.mse_loss(est_clear, ground_truth['foot_clearance']).item()
        map_error = F.mse_loss(rec_map, ground_truth['scandots']).item()
        next_state_error = F.mse_loss(rec_next_state, ground_truth['next_state']).item()
        
        # 打印估计头
        print(f"[Estimation Heads]")
        print(f"Velocity 估计前3个样本:\n{est_vel[:3].cpu().numpy()}")
        print(f"真实 Velocity 前3个样本:\n{ground_truth['velocity'][:3].cpu().numpy()}")
        print(f"Velocity MSE: {vel_error:.6f}\n")
        
        print(f"Clearance 估计前3个样本:\n{est_clear[:3].cpu().numpy()}")
        print(f"真实 Clearance 前3个样本:\n{ground_truth['foot_clearance'][:3].cpu().numpy()}")
        print(f"Clearance MSE: {clear_error:.6f}\n")
        
        # 打印解码器
        print(f"[Decoders - Reconstruction]")
        print(f"Map 重建前5个scandots:\n{rec_map[0, :5].cpu().numpy()}")
        print(f"真实 Map 前5个scandots:\n{ground_truth['scandots'][0, :5].cpu().numpy()}")
        print(f"Map Reconstruction MSE: {map_error:.6f}\n")
        
        print(f"Next State 重建前5维:\n{rec_next_state[0, :5].cpu().numpy()}")
        print(f"真实 Next State 前5维:\n{ground_truth['next_state'][0, :5].cpu().numpy()}")
        print(f"Next State Reconstruction MSE: {next_state_error:.6f}")
        print("="*60 + "\n")
        
    return {
        'vel': vel_error,
        'clear': clear_error,
        'map': map_error,
        'next_state': next_state_error
    }


def print_improvement_stats(init_errors, final_errors):
    """
    打印训练前后的误差改善统计
    
    Args:
        init_errors: 训练前的误差字典
        final_errors: 训练后的误差字典
    """
    print("\n" + "-"*60)
    print("误差改善统计 (Error Improvement)")
    print("-"*60)
    
    vel_imp = (1 - final_errors['vel'] / init_errors['vel']) * 100 if init_errors['vel'] > 0 else 0
    clear_imp = (1 - final_errors['clear'] / init_errors['clear']) * 100 if init_errors['clear'] > 0 else 0
    map_imp = (1 - final_errors['map'] / init_errors['map']) * 100 if init_errors['map'] > 0 else 0
    next_state_imp = (1 - final_errors['next_state'] / init_errors['next_state']) * 100 if init_errors['next_state'] > 0 else 0
    
    print(f"[Estimation Heads]")
    print(f"Velocity MSE:        {init_errors['vel']:.6f} → {final_errors['vel']:.6f} (改善 {vel_imp:.2f}%)")
    print(f"Clearance MSE:       {init_errors['clear']:.6f} → {final_errors['clear']:.6f} (改善 {clear_imp:.2f}%)\n")
    
    print(f"[Decoders - Reconstruction]")
    print(f"Map Reconstruction:  {init_errors['map']:.6f} → {final_errors['map']:.6f} (改善 {map_imp:.2f}%)")
    print(f"Next State Recon:    {init_errors['next_state']:.6f} → {final_errors['next_state']:.6f} (改善 {next_state_imp:.2f}%)")
    print("="*60 + "\n")


def print_network_summary(estimator):
    """
    打印网络结构和参数统计
    
    Args:
        estimator: PIE_estimator 模型
    """
    print("\n" + "="*80)
    print("网络结构与参数维度详情 (Network Architecture & Dimensions)")
    print("="*80)
    
    print("--- 网络层级结构 (Layer Hierarchy) ---")
    print(estimator)
    print("\n")
    
    # # 统计各模块参数量
    # print("--- 各模块参数统计 ---")
    
    # def count_params(module):
    #     return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # print(f"Prop Encoder:        {count_params(estimator.prop_encoder):>10,} params")
    # print(f"CNN Encoder:         {count_params(estimator.cnn):>10,} params")
    # print(f"CNN Projection:      {count_params(estimator.cnn_projection):>10,} params")
    # print(f"Transformer:         {count_params(estimator.transformer):>10,} params")
    # print(f"GRU:                 {count_params(estimator.gru):>10,} params")
    # print(f"Velocity Head:       {count_params(estimator.head_vel):>10,} params")
    # print(f"Clearance Head:      {count_params(estimator.head_clearance):>10,} params")
    # print(f"Map Latent Head:     {count_params(estimator.head_map_latent):>10,} params")
    # print(f"VAE Mu Head:         {count_params(estimator.head_vae_mu):>10,} params")
    # print(f"VAE Logvar Head:     {count_params(estimator.head_vae_logvar):>10,} params")
    # print(f"Map Decoder:         {count_params(estimator.map_decoder):>10,} params")
    # print(f"Next State Decoder:  {count_params(estimator.next_state_decoder):>10,} params")
    # print("-" * 80)
    
    # total_params = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
    # print(f"Total Trainable Parameters: {total_params:,}")
    # print("="*80)


# ==============================================================================
# 完整测试脚本 (Test Script)
# ==============================================================================
if __name__ == "__main__":
    import numpy as np
    
    print("="*80)
    print("PIE_estimator (Optimized Architecture) 完整流程测试")
    print("="*80)

    # 1. 配置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 定义超参数 (模拟 Walking 论文环境配置)
    B = 4                       # Batch Size
    H1 = 10                     # 本体历史长度
    H2 = 2                      # 深度图叠加帧数
    prop_dim = 45               # 本体维度
    depth_H, depth_W = 58, 87   # 深度图尺寸 (常见配置)
    num_scandots = 200          # 地形高度图点数
    
    # 3. 初始化 Estimator 模型 (使用优化后的参数)
    print("\n[1/4] 初始化模型 (Optimized Config)...")
    estimator = PIE_estimator(
        prop_dim=prop_dim,
        prop_history_len=H1,
        depth_shape=(depth_H, depth_W),
        depth_history_len=H2,
        num_scandots=num_scandots,
        
        # 优化后的结构参数 (参考 Table V)
        prop_encoder_hidden_dims=[256, 128],
        cnn_channels=[32, 64, 64, 64],
        cnn_kernel_sizes=[7, 7, 3, 3],
        cnn_strides=[4, 4, 2, 2],
        cnn_paddings=[3, 3, 1, 1],
        
        d_model=128,
        nhead=1,
        dim_feedforward=256,
        
        map_decoder_hidden_dims=[64, 128],
        state_decoder_hidden_dims=[64, 128],
        dim_z_map=64,
        dim_z_latent=64,
        
        device=device
    )
    print("模型初始化成功。")
    # 12. 打印网络结构统计
    print_network_summary(estimator)

    # 4. 配置优化器
    print("\n[2/4] 配置优化器...")
    param_groups = estimator.get_parameter_groups()
    optimizer = torch.optim.Adam(param_groups, lr=1e-3)
    
    print(f"优化器参数组: {len(param_groups)} 组")
    for group in param_groups:
        print(f"  - {group['name']}: {len(group['params'])} tensors")

    # 5. 生成模拟数据
    print("\n[3/4] 生成模拟数据并运行训练测试...")
    dummy_prop = torch.randn(B, H1, prop_dim, device=device)
    dummy_depth = torch.randn(B, H2, depth_H, depth_W, device=device)

    dummy_gt = {
        'velocity': torch.randn(B, 3, device=device),          
        'foot_clearance': torch.randn(B, 4, device=device),    
        'scandots': torch.randn(B, num_scandots, device=device), 
        'next_state': torch.randn(B, prop_dim, device=device)    
    }

    # 6. 训练前评估
    init_errors = evaluate_reconstruction_errors(
        estimator, dummy_prop, dummy_depth, dummy_gt, 
        phase="训练前初始估计值 (Before Training)"
    )

    # 7. 运行训练循环
    estimator.train()
    for step in range(50):
        loss_logs = estimator.train_one_step(optimizer, dummy_prop, dummy_depth, dummy_gt)
        print(f"  Step {step+1}: Total Loss = {sum(loss_logs.values()):.4f}")

    print("训练测试通过：Forward/Backward/Update 正常。")

    # 8. 训练后评估
    print("\n[4/4] 运行推理 (Inference) 测试...")
    final_errors = evaluate_reconstruction_errors(
        estimator, dummy_prop, dummy_depth, dummy_gt,
        phase="训练后最终估计值 (After Training)"
    )

    # 9. 打印改善统计
    print_improvement_stats(init_errors, final_errors)

    # 10. 验证推理接口
    est_vel, est_clear, z_map, z_latent = estimator.inference(dummy_prop, dummy_depth)
    print("推理输出形状检查:")
    print(f"  est_vel:     {list(est_vel.shape)}")
    print(f"  est_clear:   {list(est_clear.shape)}")
    print(f"  z_map:       {list(z_map.shape)}")
    print(f"  z_latent:    {list(z_latent.shape)}")

    # 11. 形状断言
    assert est_vel.shape == (B, 3)
    assert est_clear.shape == (B, 4)
    assert z_map.shape == (B, 64)
    assert z_latent.shape == (B, 64)

    print("\n" + "="*80)
    print("所有测试通过！优化后的网络结构运行正常。")
    print("="*80)

    
    # 13. 测试保存和加载功能
    print("\n" + "="*80)
    print("测试模型保存和加载")
    print("="*80)
    
    # 保存模型
    save_path = "pie_estimator_test.pt"
    estimator.save(save_path)
    
    # 加载模型 (方式1: 从checkpoint加载配置)
    loaded_estimator = PIE_estimator.load(save_path, device=device)
    
    # 验证加载的模型
    with torch.inference_mode():
        original_output = estimator.inference(dummy_prop, dummy_depth)
        loaded_output = loaded_estimator.inference(dummy_prop, dummy_depth)
        
        # 检查输出是否一致
        outputs_match = all([
            torch.allclose(original_output[i], loaded_output[i], atol=1e-6)
            for i in range(4)
        ])
        
        if outputs_match:
            print("✓ 模型加载成功,输出与原模型一致")
        else:
            print("✗ 警告: 加载模型的输出与原模型不一致")
    
    print("="*80)