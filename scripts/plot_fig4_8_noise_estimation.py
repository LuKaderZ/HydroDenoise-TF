"""图4.8 噪声估计验证 — DCAMF-Net 残差重建框架的频域与时域评估（2×2布局）"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr
from plot_utils import (setup_style, FIG_DIR, FS, DATA_T1_CLEAN, DATA_T1_NOISY,
                        DC_DIR, COLORS, load_wav, compute_sisnr)

setup_style()
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7})

clean_files = sorted(DATA_T1_CLEAN.glob('*.wav'))

psd_real_list = []
psd_est_list = []
psd_r_vals = []
sisnr_n_vals = []
f_freq = None

for k, f in enumerate(clean_files):
    dcamf_file = DC_DIR / f'{k+1:06d}.wav'
    if not dcamf_file.exists():
        continue

    clean = load_wav(DATA_T1_CLEAN / f.name)
    noisy = load_wav(DATA_T1_NOISY / f.name)
    dcamf = load_wav(dcamf_file)
    L = min(len(clean), len(noisy), len(dcamf))
    c = clean[:L]; n = noisy[:L]; e = dcamf[:L]

    noise_real = n - c
    noise_est  = n - e

    ff, p_r = welch(noise_real, FS, nperseg=2048)
    _, p_e = welch(noise_est, FS, nperseg=2048)

    if f_freq is None:
        f_freq = ff

    band = (f_freq >= 0) & (f_freq <= 4000)
    p_r_db = 10 * np.log10(p_r[band] + 1e-10)
    p_e_db = 10 * np.log10(p_e[band] + 1e-10)

    psd_real_list.append(p_r_db)
    psd_est_list.append(p_e_db)

    r_psd, _ = pearsonr(p_r_db, p_e_db)
    psd_r_vals.append(r_psd)
    sisnr_n_vals.append(compute_sisnr(noise_est, noise_real))

mask = (f_freq >= 0) & (f_freq <= 4000)
fk = f_freq[mask] / 1000
n_samples = len(psd_real_list)
psd_r_vals = np.array(psd_r_vals)
sisnr_n_vals = np.array(sisnr_n_vals)

# ============================================================
# 2×2 布局
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
color = COLORS['DCAMF']

# ---- 左上：真实噪声 PSD ----
ax = axes[0][0]
real_arr = np.array(psd_real_list)
r_med = np.median(real_arr, axis=0)
r_25, r_75 = np.percentile(real_arr, [25, 75], axis=0)
r_05, r_95 = np.percentile(real_arr, [5, 95], axis=0)
ax.fill_between(fk, r_05, r_95, color='black', alpha=0.08, lw=0)
ax.fill_between(fk, r_25, r_75, color='black', alpha=0.18, lw=0)
ax.plot(fk, r_med, 'k-', lw=1.2, label='中位数')
ax.set_title(f'真实噪声 n(t) PSD（{n_samples} 样本）')
ax.set_xlabel('频率 (kHz)'); ax.set_ylabel('PSD (dB/Hz)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3); ax.set_xlim(0, 4)
ylim_real = ax.get_ylim()

# ---- 右上：DCAMF 估计噪声 PSD ----
ax = axes[0][1]
est_arr = np.array(psd_est_list)
e_med = np.median(est_arr, axis=0)
e_25, e_75 = np.percentile(est_arr, [25, 75], axis=0)
e_05, e_95 = np.percentile(est_arr, [5, 95], axis=0)
ax.fill_between(fk, e_05, e_95, color=color, alpha=0.10, lw=0)
ax.fill_between(fk, e_25, e_75, color=color, alpha=0.22, lw=0)
ax.plot(fk, e_med, '--', color=color, lw=1.2, label='中位数')
ax.set_title(f'DCAMF-Net 估计噪声 nest(t) PSD（{n_samples} 样本）')
ax.set_xlabel('频率 (kHz)'); ax.set_ylabel('PSD (dB/Hz)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3); ax.set_xlim(0, 4)

# 统一纵轴范围
y_min = min(ylim_real[0], ax.get_ylim()[0])
y_max = max(ylim_real[1], ax.get_ylim()[1])
axes[0][0].set_ylim(y_min, y_max)
axes[0][1].set_ylim(y_min, y_max)

# ---- 左下：频率偏差曲线 ----
ax = axes[1][0]
bias_arr = est_arr - real_arr
bias_med = np.median(bias_arr, axis=0)
bias_25, bias_75 = np.percentile(bias_arr, [25, 75], axis=0)
ax.fill_between(fk, bias_25, bias_75, color=color, alpha=0.20, lw=0)
ax.plot(fk, bias_med, color=color, lw=1.2)
ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.4)
ax.set_title('噪声 PSD 估计偏差（nest − n）')
ax.set_xlabel('频率 (kHz)'); ax.set_ylabel('偏差 (dB)')
ax.grid(True, alpha=0.3); ax.set_xlim(0, 4)

# ---- 右下：PSD r vs SI-SNR(nest,n) 散点图 ----
ax = axes[1][1]
h = ax.hist2d(psd_r_vals, sisnr_n_vals, bins=60, cmap='YlOrRd', cmin=1)
plt.colorbar(h[3], ax=ax, label='样本数')
ax.set_xlabel('PSD 相关系数 r')
ax.set_ylabel('SI-SNR(nest, n) (dB)')
ax.set_title('频域匹配 vs 时域重建精度')
ax.grid(True, alpha=0.3)

# 标注中位数
ax.axvline(np.median(psd_r_vals), color='black', lw=0.6, ls='--', alpha=0.5)
ax.axhline(np.median(sisnr_n_vals), color='black', lw=0.6, ls='--', alpha=0.5)
ax.text(0.98, 0.95,
        f'r 中位数 = {np.median(psd_r_vals):.3f}\nSI-SNR 中位数 = {np.median(sisnr_n_vals):.1f} dB',
        transform=ax.transAxes, ha='right', va='top', fontsize=7,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

fig.subplots_adjust(hspace=0.40, wspace=0.28, top=0.96, bottom=0.06)

fig.savefig(FIG_DIR / 'fig4-8_noise_estimation.pdf', dpi=300)
fig.savefig(FIG_DIR / 'fig4-8_noise_estimation.png', dpi=300)
plt.show()

# ---- 终端统计 ----
print(f'=== DCAMF-Net 噪声估计验证（{n_samples} 样本）===')
print(f'PSD r 中位数 (频域):        {np.median(psd_r_vals):.4f}')
print(f'PSD r Q1-Q3:               {np.percentile(psd_r_vals, 25):.4f} - {np.percentile(psd_r_vals, 75):.4f}')
print(f'SI-SNR(nest,n) 中位数 (时域):  {np.median(sisnr_n_vals):.2f} dB')
print(f'SI-SNR(nest,n) Q1-Q3:         {np.percentile(sisnr_n_vals, 25):.2f} - {np.percentile(sisnr_n_vals, 75):.2f} dB')
print(f'平均频率偏差 (0-4kHz):       {np.median(bias_arr.mean(axis=1)):+.2f} dB')
print(f'\nSaved: {FIG_DIR / "fig4-8_noise_estimation.pdf"}')
