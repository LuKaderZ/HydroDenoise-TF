"""图4-3: 各模型在ShipsEar测试集一上的逐样本SI-SNRi与SDRi箱线图
补全量对比的分布信息（中位数、四分位距、离群值）
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from plot_utils import (setup_style, FIG_DIR, COLORS, DATA_T1_CLEAN, DATA_T1_NOISY,
                        CRN_DIR, CT_DIR, DP_DIR, DC_DIR, load_wav, load_est,
                        compute_sisnr, compute_sdr)

setup_style()
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'legend.fontsize': 7.5})

MODELS = [
    ('CRN',         CRN_DIR, COLORS['CRN'],         1),
    ('Conv-TasNet', CT_DIR,  COLORS['ConvTasNet'],  2),
    ('DPRNN',       DP_DIR,  COLORS['DPRNN'],       2),
    ('DCAMF-Net',   DC_DIR,  COLORS['DCAMF'],       3),
]

# ---- 逐样本计算 SI-SNRi 和 SDRi ----
all_sisnri = {}
all_sdri = {}
for name, est_dir, color, est_type in MODELS:
    sisnri_list, sdri_list = [], []
    clean_files = sorted(DATA_T1_CLEAN.glob('*.wav'))
    for idx, cf in enumerate(clean_files, start=1):
        noisy_file = DATA_T1_NOISY / cf.name
        if not noisy_file.exists():
            continue
        try:
            est = load_est(est_dir, idx, est_type)
        except FileNotFoundError:
            continue
        clean = load_wav(cf)
        noisy = load_wav(noisy_file)
        L = min(len(clean), len(noisy), len(est))
        clean, noisy, est = clean[:L], noisy[:L], est[:L]
        sisnri_list.append(compute_sisnr(est, clean) - compute_sisnr(noisy, clean))
        sdri_list.append(compute_sdr(est, clean) - compute_sdr(noisy, clean))
    all_sisnri[name] = np.array(sisnri_list)
    all_sdri[name] = np.array(sdri_list)
    print(f'{name}: {len(sisnri_list)} samples, '
          f'SI-SNRi median={np.median(sisnri_list):.2f}, mean={np.mean(sisnri_list):.2f}, '
          f'SDRi median={np.median(sdri_list):.2f}, mean={np.mean(sdri_list):.2f}')

# ---- 出图 ----
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))

colors_list = [COLORS['CRN'], COLORS['ConvTasNet'], COLORS['DPRNN'], COLORS['DCAMF']]
names_list = ['CRN', 'Conv-TasNet', 'DPRNN', 'DCAMF-Net']

for ax_idx, (metric_name, data_dict, ylabel) in enumerate([
    ('SI-SNRi', all_sisnri, 'SI-SNRi (dB)'),
    ('SDRi', all_sdri, 'SDRi (dB)'),
]):
    ax = axes[ax_idx]
    box_data = [data_dict[n] for n in names_list]

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops={'color': 'black', 'lw': 1.2},
                    flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'gray',
                                'alpha': 0.5},
                    whiskerprops={'lw': 0.8},
                    capprops={'lw': 0.8})

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(names_list, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)

    # 中位数标注
    for i, (name, vals) in enumerate(zip(names_list, box_data)):
        med = np.median(vals)
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        ax.text(i + 1, q3 + 0.5, f'{med:.1f}', ha='center', fontsize=6.5, color='#333',
                fontweight='bold')

    ax.set_title(f'({"ab"[ax_idx]}) {metric_name} 分布', fontsize=9, fontweight='bold', pad=4)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig4-3_sisnr_boxplot.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig4-3_sisnr_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# ---- 终端统计 ----
print('\n========== 图4-3 SI-SNRi/SDRi 分布统计 ==========')
for name in names_list:
    s = all_sisnri[name]; d = all_sdri[name]
    print(f'\n{name}:')
    print(f'  SI-SNRi: median={np.median(s):.2f}, Q1={np.percentile(s,25):.2f}, '
          f'Q3={np.percentile(s,75):.2f}, IQR={np.percentile(s,75)-np.percentile(s,25):.2f}, '
          f'min={s.min():.2f}, max={s.max():.2f}')
    print(f'  SDRi:    median={np.median(d):.2f}, Q1={np.percentile(d,25):.2f}, '
          f'Q3={np.percentile(d,75):.2f}, IQR={np.percentile(d,75)-np.percentile(d,25):.2f}, '
          f'min={d.min():.2f}, max={d.max():.2f}')
    # 离群值统计
    iqr_s = np.percentile(s, 75) - np.percentile(s, 25)
    lower_s = np.percentile(s, 25) - 1.5 * iqr_s
    upper_s = np.percentile(s, 75) + 1.5 * iqr_s
    n_low = np.sum(s < lower_s); n_high = np.sum(s > upper_s)
    print(f'  SI-SNRi 离群值: {n_low}低 + {n_high}高 = {n_low+n_high}个')
print('==========================================')
