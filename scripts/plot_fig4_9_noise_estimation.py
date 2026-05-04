"""图4.9 噪声估计验证 — DCAMF-Net 估计噪声 vs 真实噪声 PSD"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr
from plot_utils import (setup_style, FIG_DIR, FS, DATA_T1_CLEAN, DATA_T1_NOISY,
                        DC_DIR, COLORS, load_wav, compute_sisnr)

setup_style()
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8})

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
color = COLORS['DCAMF']

# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 4.5))

real_arr = np.array(psd_real_list)
est_arr  = np.array(psd_est_list)

# 真实噪声 — 黑色中位数曲线
ax.plot(fk, np.median(real_arr, axis=0), 'k-', lw=1.2,
        label='真实噪声 n(t)')

# 估计噪声 — 红色虚线
ax.plot(fk, np.median(est_arr, axis=0), '--', color=color, lw=1.2,
        label='DCAMF 估计噪声 nest(t)')

ax.set_xlabel('频率 (kHz)')
ax.set_ylabel('PSD (dB/Hz)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 4)

# 右下角数据
median_r = np.median(psd_r_vals)
median_sisnr = np.median(sisnr_n_vals)
ax.text(0.98, 0.05,
        f'PSD r = {median_r:.3f}\n'
        f'SI-SNR(nest, n) = {median_sisnr:.2f} dB',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig4-9_noise_estimation.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig4-9_noise_estimation.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 终端统计（用于正文引用） ----
print(f'=== DCAMF-Net 噪声估计验证（{n_samples} 样本）===')
print(f'PSD r 中位数:             {np.median(psd_r_vals):.4f}')
print(f'PSD r Q1-Q3:              {np.percentile(psd_r_vals, 25):.4f} - {np.percentile(psd_r_vals, 75):.4f}')
print(f'SI-SNR(nest,n) 中位数:    {np.median(sisnr_n_vals):.2f} dB')
print(f'SI-SNR(nest,n) Q1-Q3:     {np.percentile(sisnr_n_vals, 25):.2f} - {np.percentile(sisnr_n_vals, 75):.2f} dB')
print(f'\nSaved: {FIG_DIR / "fig4-9_noise_estimation.pdf"}')
