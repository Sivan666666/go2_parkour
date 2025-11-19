import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体和数学字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 10

def plot_camera_angle_comparison(save_path='camera_angle_comparison.pdf', show_plot=True):
    """
    绘制不同相机角度对强化学习策略影响的性能对比图
    
    对比：
    - Self-made 10° (向下10度)
    - Self-made 20° (向下20度)
    - Self-made 30° (向下30度)
    - Self-made 40° (向下40度)
    
    迭代次数：5000 和 11000
    
    指标：
    1. Mean Reward (↑ 越高越好)
    2. Mean Episode Length (↓ 越低越好)
    3. Mean Number of Waypoints (↑ 越高越好)
    4. Mean Edge Violation (↓ 越低越好)
    """
    
    # ==========================================
    # 数据准备
    # ==========================================
    
    # Self-made 10° (007-g2-self)
    self_10 = {
        5000: {
            'reward': (16.48, 6.32),
            'length': (643.56, 229.78),
            'waypoints': (0.86, 0.25),
            'edge_violation': (0.01, 0.13)
        },
        11000: {
            'reward': (17.12, 6.18),
            'length': (667.66, 225.72),
            'waypoints': (0.86, 0.25),
            'edge_violation': (0.02, 0.13)
        }
    }
    
    # Self-made 20° (008-g2-20degree)
    self_20 = {
        5000: {
            'reward': (17.65, 6.44),
            'length': (681.68, 212.71),
            'waypoints': (0.86, 0.25),
            'edge_violation': (0.03, 0.18)
        },
        11000: {
            'reward': (18.16, 5.71),
            'length': (664.99, 189.25),
            'waypoints': (0.89, 0.24),
            'edge_violation': (0.01, 0.10)
        }
    }
    
    # Self-made 30° (010-g2-30degree)
    self_30 = {
        5000: {
            'reward': (17.75, 6.00),
            'length': (661.41, 207.20),
            'waypoints': (0.90, 0.24),
            'edge_violation': (0.03, 0.17)
        },
        11000: {
            'reward': (18.48, 5.90),
            'length': (654.15, 186.08),
            'waypoints': (0.91, 0.23),
            'edge_violation': (0.01, 0.08)
        }
    }
    
    # Self-made 40° (009-g2-40degree)
    self_40 = {
        5000: {
            'reward': (18.20, 6.28),
            'length': (676.66, 214.58),
            'waypoints': (0.88, 0.24),
            'edge_violation': (0.02, 0.15)
        },
        11000: {
            'reward': (17.87, 5.17),
            'length': (638.57, 185.67),
            'waypoints': (0.92, 0.20),
            'edge_violation': (0.01, 0.09)
        }
    }
    
    # ==========================================
    # 创建图形
    # ==========================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Camera Angle Comparison: Impact of Depth Image Angle on RL Policy', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # 颜色和标记方案
    colors = {
        'self_10': '#2E86AB',      # 蓝色
        'self_20': '#A23B72',      # 紫红色
        'self_30': '#F18F01',      # 橙色
        'self_40': '#C73E1D',      # 红色
    }
    
    markers = {
        'self_10': 'o',    # 圆圈
        'self_20': 's',    # 方块
        'self_30': '^',    # 上三角
        'self_40': 'D',    # 菱形
    }
    
    # X 轴位置（错开显示）
    iterations = [5000, 11000]
    offset = 80  # 错开的距离
    x_10 = np.array(iterations) - 1.5 * offset  # 10° 最左
    x_20 = np.array(iterations) - 0.5 * offset  # 20° 中左
    x_30 = np.array(iterations) + 0.5 * offset  # 30° 中右
    x_40 = np.array(iterations) + 1.5 * offset  # 40° 最右
    x_ticks = np.array(iterations)               # 刻度位置保持不变
    
    # ==========================================
    # 子图 1: Mean Reward
    # ==========================================
    ax1 = axes[0, 0]
    
    # 提取数据
    rewards_10 = [self_10[iter]['reward'][0] for iter in iterations]
    rewards_10_err = [self_10[iter]['reward'][1] for iter in iterations]
    
    rewards_20 = [self_20[iter]['reward'][0] for iter in iterations]
    rewards_20_err = [self_20[iter]['reward'][1] for iter in iterations]
    
    rewards_30 = [self_30[iter]['reward'][0] for iter in iterations]
    rewards_30_err = [self_30[iter]['reward'][1] for iter in iterations]
    
    rewards_40 = [self_40[iter]['reward'][0] for iter in iterations]
    rewards_40_err = [self_40[iter]['reward'][1] for iter in iterations]
    
    # 绘制折线图
    ax1.errorbar(x_10, rewards_10, yerr=rewards_10_err,
                 label='Self-made 10°', 
                 color=colors['self_10'],
                 marker=markers['self_10'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax1.errorbar(x_20, rewards_20, yerr=rewards_20_err,
                 label='Self-made 20°', 
                 color=colors['self_20'],
                 marker=markers['self_20'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax1.errorbar(x_30, rewards_30, yerr=rewards_30_err,
                 label='Self-made 30°', 
                 color=colors['self_30'],
                 marker=markers['self_30'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax1.errorbar(x_40, rewards_40, yerr=rewards_40_err,
                 label='Self-made 40°', 
                 color=colors['self_40'],
                 marker=markers['self_40'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax1.set_ylabel('Mean Reward', fontweight='bold')
    ax1.set_title('(a) Mean Reward $\\uparrow$', fontweight='bold', loc='left')
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(['5k', '11k'])
    ax1.set_xlabel('Training Iterations', fontweight='bold')
    ax1.legend(loc='upper center', fontsize=9, ncol=4, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(10, 28)
    
    # ==========================================
    # 子图 2: Mean Episode Length
    # ==========================================
    ax2 = axes[0, 1]
    
    # 提取数据
    lengths_10 = [self_10[iter]['length'][0] for iter in iterations]
    lengths_10_err = [self_10[iter]['length'][1] for iter in iterations]
    
    lengths_20 = [self_20[iter]['length'][0] for iter in iterations]
    lengths_20_err = [self_20[iter]['length'][1] for iter in iterations]
    
    lengths_30 = [self_30[iter]['length'][0] for iter in iterations]
    lengths_30_err = [self_30[iter]['length'][1] for iter in iterations]
    
    lengths_40 = [self_40[iter]['length'][0] for iter in iterations]
    lengths_40_err = [self_40[iter]['length'][1] for iter in iterations]
    
    # 绘制折线图
    ax2.errorbar(x_10, lengths_10, yerr=lengths_10_err,
                 label='Self-made 10°', 
                 color=colors['self_10'],
                 marker=markers['self_10'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax2.errorbar(x_20, lengths_20, yerr=lengths_20_err,
                 label='Self-made 20°', 
                 color=colors['self_20'],
                 marker=markers['self_20'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax2.errorbar(x_30, lengths_30, yerr=lengths_30_err,
                 label='Self-made 30°', 
                 color=colors['self_30'],
                 marker=markers['self_30'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax2.errorbar(x_40, lengths_40, yerr=lengths_40_err,
                 label='Self-made 40°', 
                 color=colors['self_40'],
                 marker=markers['self_40'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax2.set_ylabel('Mean Episode Length (steps)', fontweight='bold')
    ax2.set_title('(b) Mean Episode Length $\\downarrow$', fontweight='bold', loc='left')
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(['5k', '11k'])
    ax2.set_xlabel('Training Iterations', fontweight='bold')
    ax2.legend(loc='upper center', fontsize=9, ncol=4, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(400, 900)
    
    # ==========================================
    # 子图 3: Mean Number of Waypoints
    # ==========================================
    ax3 = axes[1, 0]
    
    # 提取数据
    waypoints_10 = [self_10[iter]['waypoints'][0] for iter in iterations]
    waypoints_10_err = [self_10[iter]['waypoints'][1] for iter in iterations]
    
    waypoints_20 = [self_20[iter]['waypoints'][0] for iter in iterations]
    waypoints_20_err = [self_20[iter]['waypoints'][1] for iter in iterations]
    
    waypoints_30 = [self_30[iter]['waypoints'][0] for iter in iterations]
    waypoints_30_err = [self_30[iter]['waypoints'][1] for iter in iterations]
    
    waypoints_40 = [self_40[iter]['waypoints'][0] for iter in iterations]
    waypoints_40_err = [self_40[iter]['waypoints'][1] for iter in iterations]
    
    # 绘制折线图
    ax3.errorbar(x_10, waypoints_10, yerr=waypoints_10_err,
                 label='Self-made 10°', 
                 color=colors['self_10'],
                 marker=markers['self_10'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax3.errorbar(x_20, waypoints_20, yerr=waypoints_20_err,
                 label='Self-made 20°', 
                 color=colors['self_20'],
                 marker=markers['self_20'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax3.errorbar(x_30, waypoints_30, yerr=waypoints_30_err,
                 label='Self-made 30°', 
                 color=colors['self_30'],
                 marker=markers['self_30'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax3.errorbar(x_40, waypoints_40, yerr=waypoints_40_err,
                 label='Self-made 40°', 
                 color=colors['self_40'],
                 marker=markers['self_40'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax3.set_ylabel('Mean Waypoints Ratio', fontweight='bold')
    ax3.set_title('(c) Mean Number of Waypoints $\\uparrow$', fontweight='bold', loc='left')
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(['5k', '11k'])
    ax3.set_xlabel('Training Iterations', fontweight='bold')
    ax3.legend(loc='upper center', fontsize=9, ncol=4, framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0.75, 1.05)
    
    # 添加参考线（100%）
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.text(11500, 1.0, '100%', ha='left', va='center', fontsize=8, color='gray')
    
    # ==========================================
    # 子图 4: Mean Edge Violation
    # ==========================================
    ax4 = axes[1, 1]
    
    # 提取数据
    violations_10 = [self_10[iter]['edge_violation'][0] for iter in iterations]
    violations_10_err = [self_10[iter]['edge_violation'][1] for iter in iterations]
    
    violations_20 = [self_20[iter]['edge_violation'][0] for iter in iterations]
    violations_20_err = [self_20[iter]['edge_violation'][1] for iter in iterations]
    
    violations_30 = [self_30[iter]['edge_violation'][0] for iter in iterations]
    violations_30_err = [self_30[iter]['edge_violation'][1] for iter in iterations]
    
    violations_40 = [self_40[iter]['edge_violation'][0] for iter in iterations]
    violations_40_err = [self_40[iter]['edge_violation'][1] for iter in iterations]
    
    # 绘制折线图
    ax4.errorbar(x_10, violations_10, yerr=violations_10_err,
                 label='Self-made 10°', 
                 color=colors['self_10'],
                 marker=markers['self_10'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax4.errorbar(x_20, violations_20, yerr=violations_20_err,
                 label='Self-made 20°', 
                 color=colors['self_20'],
                 marker=markers['self_20'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax4.errorbar(x_30, violations_30, yerr=violations_30_err,
                 label='Self-made 30°', 
                 color=colors['self_30'],
                 marker=markers['self_30'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax4.errorbar(x_40, violations_40, yerr=violations_40_err,
                 label='Self-made 40°', 
                 color=colors['self_40'],
                 marker=markers['self_40'], 
                 markersize=7,
                 linewidth=2.5,
                 capsize=4, 
                 capthick=1.5,
                 elinewidth=1.5)
    
    ax4.set_ylabel('Mean Edge Violation (feet/step)', fontweight='bold')
    ax4.set_title('(d) Mean Edge Violation $\\downarrow$', fontweight='bold', loc='left')
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels(['5k', '11k'])
    ax4.set_xlabel('Training Iterations', fontweight='bold')
    ax4.legend(loc='upper center', fontsize=9, ncol=4, framealpha=0.9)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim(0, 0.22)
    
    # ==========================================
    # 调整布局
    # ==========================================
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # ==========================================
    # 保存和显示
    # ==========================================
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, axes

def plot_self_holder_comparison(save_path='self_holder_comparison.pdf', show_plot=True):
    """
    绘制自制支架（30度向下）在不同配置下的性能对比图
    
    对比：
    - Self 30° (Aligned) - 对齐配置
    - Self 30° (Flat) - 平放配置
    
    迭代次数：5000, 11000, 21000, 31000
    
    指标：
    1. Mean Reward (↑ 越高越好)
    2. Mean Episode Length (↓ 越低越好)
    3. Mean Number of Waypoints (↑ 越高越好)
    4. Mean Edge Violation (↓ 越低越好)
    """
    
    # ==========================================
    # 数据准备
    # ==========================================
    
    # Self 30° Aligned（对齐配置）
    self_aligned = {
        5000: {
            'reward': (16.48, 6.32),
            'length': (643.56, 229.78),
            'waypoints': (0.86, 0.25),
            'edge_violation': (0.01, 0.13)
        },
        11000: {
            'reward': (17.12, 6.18),
            'length': (667.66, 225.72),
            'waypoints': (0.86, 0.25),
            'edge_violation': (0.02, 0.13)
        },
        21000: {
            'reward': (18.41, 5.99),
            'length': (680.30, 205.83),
            'waypoints': (0.90, 0.22),
            'edge_violation': (0.00, 0.05)
        },
        31000: {
            'reward': (17.69, 6.32),
            'length': (657.83, 214.30),
            'waypoints': (0.87, 0.24),
            'edge_violation': (0.02, 0.17)
        }
    }
    
    # Self 30° Flat（平放配置）
    self_flat = {
        5000: {
            'reward': (9.73, 5.84),
            'length': (560.66, 313.06),
            'waypoints': (0.50, 0.32),
            'edge_violation': (0.07, 0.29)
        },
        11000: {
            'reward': (10.94, 6.02),
            'length': (576.73, 312.46),
            'waypoints': (0.57, 0.33),
            'edge_violation': (0.04, 0.20)
        },
        21000: {
            'reward': (11.21, 5.99),
            'length': (567.54, 294.16),
            'waypoints': (0.56, 0.33),
            'edge_violation': (0.04, 0.23)
        },
        31000: {
            'reward': (11.58, 6.36),
            'length': (564.86, 295.89),
            'waypoints': (0.59, 0.32),
            'edge_violation': (0.06, 0.25)
        }
    }
    
    # ==========================================
    # 创建图形
    # ==========================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Self-Made holder (30° Down): Aligned 30° vs Flat 10° Configuration', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # 颜色和标记方案
    colors = {
        'aligned': '#2E86AB',      # 蓝色 - 对齐
        'flat': '#A23B72',         # 紫红色 - 平放
    }
    
    markers = {
        'aligned': 'o',   # 圆圈
        'flat': '^',      # 三角形
    }
    
    # X 轴位置（错开显示）
    iterations = [5000, 11000, 21000, 31000]
    offset = 200  # 错开的距离
    x_aligned = np.array(iterations) - offset  # Aligned 稍微向左
    x_flat = np.array(iterations) + offset     # Flat 稍微向右
    x_ticks = np.array(iterations)             # 刻度位置保持不变
    
    # ==========================================
    # 子图 1: Mean Reward
    # ==========================================
    ax1 = axes[0, 0]
    
    # 提取数据
    aligned_rewards = [self_aligned[iter]['reward'][0] for iter in iterations]
    aligned_rewards_err = [self_aligned[iter]['reward'][1] for iter in iterations]
    
    flat_rewards = [self_flat[iter]['reward'][0] for iter in iterations]
    flat_rewards_err = [self_flat[iter]['reward'][1] for iter in iterations]
    
    # 绘制折线图
    ax1.errorbar(x_aligned, aligned_rewards, yerr=aligned_rewards_err,
                 label='Self 30° (Aligned)', 
                 color=colors['aligned'],
                 marker=markers['aligned'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax1.errorbar(x_flat, flat_rewards, yerr=flat_rewards_err,
                 label='Self 30° (Flat)', 
                 color=colors['flat'],
                 marker=markers['flat'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax1.set_ylabel('Mean Reward', fontweight='bold')
    ax1.set_title('(a) Mean Reward $\\uparrow$', fontweight='bold', loc='left')
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(['5k', '11k', '21k', '31k'])
    ax1.set_xlabel('Training Iterations', fontweight='bold')
    ax1.legend(loc='upper center', fontsize=10, ncol=2, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 28)
    
    # ==========================================
    # 子图 2: Mean Episode Length
    # ==========================================
    ax2 = axes[0, 1]
    
    # 提取数据
    aligned_lengths = [self_aligned[iter]['length'][0] for iter in iterations]
    aligned_lengths_err = [self_aligned[iter]['length'][1] for iter in iterations]
    
    flat_lengths = [self_flat[iter]['length'][0] for iter in iterations]
    flat_lengths_err = [self_flat[iter]['length'][1] for iter in iterations]
    
    # 绘制折线图
    ax2.errorbar(x_aligned, aligned_lengths, yerr=aligned_lengths_err,
                 label='Self 30° (Aligned)', 
                 color=colors['aligned'],
                 marker=markers['aligned'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax2.errorbar(x_flat, flat_lengths, yerr=flat_lengths_err,
                 label='Self 30° (Flat)', 
                 color=colors['flat'],
                 marker=markers['flat'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax2.set_ylabel('Mean Episode Length (steps)', fontweight='bold')
    ax2.set_title('(b) Mean Episode Length $\\downarrow$', fontweight='bold', loc='left')
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(['5k', '11k', '21k', '31k'])
    ax2.set_xlabel('Training Iterations', fontweight='bold')
    ax2.legend(loc='upper center', fontsize=10, ncol=2, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(300, 950)
    
    # ==========================================
    # 子图 3: Mean Number of Waypoints
    # ==========================================
    ax3 = axes[1, 0]
    
    # 提取数据
    aligned_waypoints = [self_aligned[iter]['waypoints'][0] for iter in iterations]
    aligned_waypoints_err = [self_aligned[iter]['waypoints'][1] for iter in iterations]
    
    flat_waypoints = [self_flat[iter]['waypoints'][0] for iter in iterations]
    flat_waypoints_err = [self_flat[iter]['waypoints'][1] for iter in iterations]
    
    # 绘制折线图
    ax3.errorbar(x_aligned, aligned_waypoints, yerr=aligned_waypoints_err,
                 label='Self 30° (Aligned)', 
                 color=colors['aligned'],
                 marker=markers['aligned'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax3.errorbar(x_flat, flat_waypoints, yerr=flat_waypoints_err,
                 label='Self 30° (Flat)', 
                 color=colors['flat'],
                 marker=markers['flat'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax3.set_ylabel('Mean Waypoints Ratio', fontweight='bold')
    ax3.set_title('(c) Mean Number of Waypoints $\\uparrow$', fontweight='bold', loc='left')
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(['5k', '11k', '21k', '31k'])
    ax3.set_xlabel('Training Iterations', fontweight='bold')
    ax3.legend(loc='upper center', fontsize=10, ncol=2, framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0.2, 1.15)
    
    # 添加参考线（100%）
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.text(32000, 1.0, '100%', ha='left', va='center', fontsize=8, color='gray')
    
    # ==========================================
    # 子图 4: Mean Edge Violation
    # ==========================================
    ax4 = axes[1, 1]
    
    # 提取数据
    aligned_violations = [self_aligned[iter]['edge_violation'][0] for iter in iterations]
    aligned_violations_err = [self_aligned[iter]['edge_violation'][1] for iter in iterations]
    
    flat_violations = [self_flat[iter]['edge_violation'][0] for iter in iterations]
    flat_violations_err = [self_flat[iter]['edge_violation'][1] for iter in iterations]
    
    # 绘制折线图
    ax4.errorbar(x_aligned, aligned_violations, yerr=aligned_violations_err,
                 label='Self 30° (Aligned)', 
                 color=colors['aligned'],
                 marker=markers['aligned'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax4.errorbar(x_flat, flat_violations, yerr=flat_violations_err,
                 label='Self 30° (Flat)', 
                 color=colors['flat'],
                 marker=markers['flat'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax4.set_ylabel('Mean Edge Violation (feet/step)', fontweight='bold')
    ax4.set_title('(d) Mean Edge Violation $\\downarrow$', fontweight='bold', loc='left')
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels(['5k', '11k', '21k', '31k'])
    ax4.set_xlabel('Training Iterations', fontweight='bold')
    ax4.legend(loc='upper center', fontsize=10, ncol=2, framealpha=0.9)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim(0, 0.40)
    
    # ==========================================
    # 调整布局
    # ==========================================
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # ==========================================
    # 保存和显示
    # ==========================================
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, axes

def plot_camera_position_comparison(save_path='camera_position_comparison.pdf', show_plot=True):
    """
    绘制相同角度（30°）但不同相机位置对强化学习策略影响的性能对比图
    
    对比：
    - Self-made 30° at [0.3, 0, 0.188] (010-g2-30degree)
    - Helpful 30° at different position (005-g2-Helpful)
    
    迭代次数：5000 和 11000
    
    指标：
    1. Mean Reward (↑ 越高越好)
    2. Mean Episode Length (↓ 越低越好)
    3. Mean Number of Waypoints (↑ 越高越好)
    4. Mean Edge Violation (↓ 越低越好)
    """
    
    # ==========================================
    # 数据准备
    # ==========================================
    
    # Self-made 30° at position [0.3, 0, 0.188] (010-g2-30degree)
    self_30_pos1 = {
        5000: {
            'reward': (17.75, 6.00),
            'length': (661.41, 207.20),
            'waypoints': (0.90, 0.24),
            'edge_violation': (0.03, 0.17)
        },
        11000: {
            'reward': (18.48, 5.90),
            'length': (654.15, 186.08),
            'waypoints': (0.91, 0.23),
            'edge_violation': (0.01, 0.08)
        }
    }
    
    # Helpful 30° at different position (005-g2-Helpful)
    helpful_30 = {
        5000: {
            'reward': (18.49, 5.87),
            'length': (661.49, 197.83),
            'waypoints': (0.91, 0.22),
            'edge_violation': (0.01, 0.13)
        },
        11000: {
            'reward': (18.21, 5.58),
            'length': (655.70, 195.04),
            'waypoints': (0.91, 0.21),
            'edge_violation': (0.01, 0.12)
        }
    }
    
    # ==========================================
    # 创建图形
    # ==========================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Camera Position Comparison: Same Angle (30°) with Different Positions', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # 颜色和标记方案
    colors = {
        'self_pos1': '#F18F01',      # 橙色 - Self-made position
        'helpful': '#2E86AB',        # 蓝色 - Helpful position
    }
    
    markers = {
        'self_pos1': '^',    # 上三角
        'helpful': 'o',      # 圆圈
    }
    
    # X 轴位置（错开显示）
    iterations = [5000, 11000]
    offset = 100  # 错开的距离
    x_self = np.array(iterations) - offset    # Self-made 稍微向左
    x_helpful = np.array(iterations) + offset  # Helpful 稍微向右
    x_ticks = np.array(iterations)             # 刻度位置保持不变
    
    # ==========================================
    # 子图 1: Mean Reward
    # ==========================================
    ax1 = axes[0, 0]
    
    # 提取数据
    self_rewards = [self_30_pos1[iter]['reward'][0] for iter in iterations]
    self_rewards_err = [self_30_pos1[iter]['reward'][1] for iter in iterations]
    
    helpful_rewards = [helpful_30[iter]['reward'][0] for iter in iterations]
    helpful_rewards_err = [helpful_30[iter]['reward'][1] for iter in iterations]
    
    # 绘制折线图
    ax1.errorbar(x_self, self_rewards, yerr=self_rewards_err,
                 label='Self-made 30° [0.3, 0, 0.188]', 
                 color=colors['self_pos1'],
                 marker=markers['self_pos1'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax1.errorbar(x_helpful, helpful_rewards, yerr=helpful_rewards_err,
                 label='Helpful 30° (different pos)', 
                 color=colors['helpful'],
                 marker=markers['helpful'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax1.set_ylabel('Mean Reward', fontweight='bold')
    ax1.set_title('(a) Mean Reward $\\uparrow$', fontweight='bold', loc='left')
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(['5k', '11k'])
    ax1.set_xlabel('Training Iterations', fontweight='bold')
    ax1.legend(loc='upper center', fontsize=9.5, ncol=2, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(12, 26)
    
    # ==========================================
    # 子图 2: Mean Episode Length
    # ==========================================
    ax2 = axes[0, 1]
    
    # 提取数据
    self_lengths = [self_30_pos1[iter]['length'][0] for iter in iterations]
    self_lengths_err = [self_30_pos1[iter]['length'][1] for iter in iterations]
    
    helpful_lengths = [helpful_30[iter]['length'][0] for iter in iterations]
    helpful_lengths_err = [helpful_30[iter]['length'][1] for iter in iterations]
    
    # 绘制折线图
    ax2.errorbar(x_self, self_lengths, yerr=self_lengths_err,
                 label='Self-made 30° [0.3, 0, 0.188]', 
                 color=colors['self_pos1'],
                 marker=markers['self_pos1'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax2.errorbar(x_helpful, helpful_lengths, yerr=helpful_lengths_err,
                 label='Helpful 30° (different pos)', 
                 color=colors['helpful'],
                 marker=markers['helpful'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax2.set_ylabel('Mean Episode Length (steps)', fontweight='bold')
    ax2.set_title('(b) Mean Episode Length $\\downarrow$', fontweight='bold', loc='left')
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(['5k', '11k'])
    ax2.set_xlabel('Training Iterations', fontweight='bold')
    ax2.legend(loc='upper center', fontsize=9.5, ncol=2, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(450, 900)
    
    # ==========================================
    # 子图 3: Mean Number of Waypoints
    # ==========================================
    ax3 = axes[1, 0]
    
    # 提取数据
    self_waypoints = [self_30_pos1[iter]['waypoints'][0] for iter in iterations]
    self_waypoints_err = [self_30_pos1[iter]['waypoints'][1] for iter in iterations]
    
    helpful_waypoints = [helpful_30[iter]['waypoints'][0] for iter in iterations]
    helpful_waypoints_err = [helpful_30[iter]['waypoints'][1] for iter in iterations]
    
    # 绘制折线图
    ax3.errorbar(x_self, self_waypoints, yerr=self_waypoints_err,
                 label='Self-made 30° [0.3, 0, 0.188]', 
                 color=colors['self_pos1'],
                 marker=markers['self_pos1'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax3.errorbar(x_helpful, helpful_waypoints, yerr=helpful_waypoints_err,
                 label='Helpful 30° (different pos)', 
                 color=colors['helpful'],
                 marker=markers['helpful'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax3.set_ylabel('Mean Waypoints Ratio', fontweight='bold')
    ax3.set_title('(c) Mean Number of Waypoints $\\uparrow$', fontweight='bold', loc='left')
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(['5k', '11k'])
    ax3.set_xlabel('Training Iterations', fontweight='bold')
    ax3.legend(loc='upper center', fontsize=9.5, ncol=2, framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0.65, 1.15)
    
    # 添加参考线（100%）
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.text(11500, 1.0, '100%', ha='left', va='center', fontsize=8, color='gray')
    
    # ==========================================
    # 子图 4: Mean Edge Violation
    # ==========================================
    ax4 = axes[1, 1]
    
    # 提取数据
    self_violations = [self_30_pos1[iter]['edge_violation'][0] for iter in iterations]
    self_violations_err = [self_30_pos1[iter]['edge_violation'][1] for iter in iterations]
    
    helpful_violations = [helpful_30[iter]['edge_violation'][0] for iter in iterations]
    helpful_violations_err = [helpful_30[iter]['edge_violation'][1] for iter in iterations]
    
    # 绘制折线图
    ax4.errorbar(x_self, self_violations, yerr=self_violations_err,
                 label='Self-made 30° [0.3, 0, 0.188]', 
                 color=colors['self_pos1'],
                 marker=markers['self_pos1'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax4.errorbar(x_helpful, helpful_violations, yerr=helpful_violations_err,
                 label='Helpful 30° (different pos)', 
                 color=colors['helpful'],
                 marker=markers['helpful'], 
                 markersize=8,
                 linewidth=2.5,
                 capsize=5, 
                 capthick=2,
                 elinewidth=2)
    
    ax4.set_ylabel('Mean Edge Violation (feet/step)', fontweight='bold')
    ax4.set_title('(d) Mean Edge Violation $\\downarrow$', fontweight='bold', loc='left')
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels(['5k', '11k'])
    ax4.set_xlabel('Training Iterations', fontweight='bold')
    ax4.legend(loc='upper center', fontsize=9.5, ncol=2, framealpha=0.9)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim(0, 0.22)
    
    # ==========================================
    # 调整布局
    # ==========================================
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # ==========================================
    # 保存和显示
    # ==========================================
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, axes

# ==========================================
# 使用示例
# ==========================================
if __name__ == '__main__':
    # 生成图形 1: 10° vs 20° vs 30° vs 40° Camera Angle Comparison
    fig1, axes1 = plot_camera_angle_comparison(
        save_path='camera_angle_comparison.pdf',
        show_plot=True
    )
    fig1.savefig('camera_angle_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Camera angle comparison (10°-40°) saved!")

    # 生成图形 2: Self 30° (Aligned vs Flat)
    fig2, axes2 = plot_self_holder_comparison(
        save_path='self_holder_comparison.pdf',
        show_plot=True
    )
    fig2.savefig('self_holder_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Self holder comparison saved!")
    
    # 生成图形 3: Camera Position Comparison (Same 30° angle, different positions)
    fig3, axes3 = plot_camera_position_comparison(
        save_path='camera_position_comparison.pdf',
        show_plot=True
    )
    fig3.savefig('camera_position_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Camera position comparison saved!")