import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_comparison_steps(csv_ours, csv_no_foothold, step_width=0.2, y_range=0.8):
    """
    对比两组落足点分布：Ours vs Without Foothold
    """
    # 1. 创建画布
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_aspect('equal')
    plt.axis('off')

    # 2. 绘制黑色外边框
    rect = patches.Rectangle((0, 0), y_range, step_width, 
                             linewidth=2.5, edgecolor='black', facecolor='none', zorder=1)
    ax.add_patch(rect)

    # 3. 加载并绘制第一组数据 (Ours)
    try:
        df1 = pd.read_csv(csv_ours)
        ax.scatter(df1['dy'], df1['dx'], 
                   c='#1f77b4', s=100, alpha=0.4, 
                   edgecolors='none', label='Ours', zorder=3)
    except Exception as e:
        print(f"读取 {csv_ours} 失败: {e}")

    # 4. 加载并绘制第二组数据 (Without Foothold)
    try:
        df2 = pd.read_csv(csv_no_foothold)
        ax.scatter(df2['dy'], df2['dx'], 
                   c='#ff7f0e', s=100, alpha=0.4, 
                   edgecolors='none', label='w/o $r_{\mathrm{foothold}}$', zorder=2)
    except Exception as e:
        print(f"读取 {csv_no_foothold} 失败: {e}")

    # 5. 添加图例
    # 将图例放在矩形框外面
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
               frameon=False, ncol=2, fontsize=12)

    # 6. 保存为矢量图
    plt.savefig('footprint_comparison.pdf', bbox_inches='tight', transparent=True)
    plt.show()

# 调用示例
# 请确保文件名正确
plot_comparison_steps(
    'hollow_step_footprints.csv', 
    'hollow_step_footprints_without_foothold.csv', 
    step_width=0.2, 
    y_range=0.8
)