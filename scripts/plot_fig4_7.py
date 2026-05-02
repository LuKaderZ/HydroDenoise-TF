"""图4-7: 多层掩码融合权重分布"""
import matplotlib.pyplot as plt
from plot_utils import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

groups = {'平均SNR': 'train_avg.log', '低SNR': 'train_low.log', '高SNR': 'train_high.log'}
bar_colors = [COLORS['ConvTasNet'], COLORS['CRN'], COLORS['DCAMF']]  # blue, orange, maroon

fig, ax = plt.subplots(figsize=(8, 5.5))
y = np.arange(1, 11)
h = 0.25

for i, (name, fname) in enumerate(groups.items()):
    log_path = FUSION_LOG_DIR / fname
    if not log_path.exists():
        print(f'{name}: SKIP ({log_path} not found)')
        continue
    w = extract_fusion_weights(log_path)
    if len(w) != 10:
        print(f'{name}: expected 10 weights, got {len(w)}')
        continue

    bars = ax.barh(y + i*h, w, h, label=name, color=bar_colors[i],
                    edgecolor='black', linewidth=0.5)
    for j, val in enumerate(w):
        if val > 0.01:
            ax.text(val + 0.005, y[j] + i*h, f'{val:.2f}',
                    va='center', ha='left', fontsize=7)

ax.set_yticks(y + h)
ax.set_yticklabels([str(i) for i in range(1, 11)])
ax.set_xlabel('融合权重 (softmax)')
ax.set_ylabel('DCAM模块层数')
ax.set_title('不同信噪比下多层掩码融合权重分布', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)
all_w = np.concatenate([extract_fusion_weights(FUSION_LOG_DIR / f)
                         for f in groups.values() if (FUSION_LOG_DIR / f).exists()])
ax.set_xlim(0, all_w.max() * 1.15)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig4-7_fusion_weights_comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig4-7_fusion_weights_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print('图4-7 已保存')
