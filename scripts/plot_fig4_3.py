"""图4-3: 各模型降噪后总体频谱对比 (2x2, 统一纵轴)"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import *

setup_style()

# ---- Detect line spectra ----
print('Detecting line spectra...')
line_freqs = find_line_spectra(n_peaks=5)
print(f'Line frequencies: {np.round(line_freqs).astype(int).tolist()} Hz')

# ---- Select best sample ----
print('Selecting best sample...')
best_idx = select_best_linespectra(line_freqs)
print(f'Selected sample: {best_idx}')

clean_files = sorted(DATA_T1_CLEAN.glob('*.wav'))
fname = clean_files[best_idx - 1].name
clean = load_wav(DATA_T1_CLEAN / fname)
noisy = load_wav(DATA_T1_NOISY / fname)
crn = load_est(CRN_DIR, best_idx, 1)
ct  = load_est(CT_DIR, best_idx, 2)
dp  = load_est(DP_DIR, best_idx, 2)
dc  = load_est(DC_DIR, best_idx, 3)

L = min(len(clean), len(noisy), len(crn), len(ct), len(dp), len(dc))
clean, noisy = clean[:L], noisy[:L]
crn, ct, dp, dc = crn[:L], ct[:L], dp[:L], dc[:L]

# ---- Compute PSDs ----
f, p_clean = psd_db(clean)
_, p_noisy  = psd_db(noisy)
_, p_crn    = psd_db(crn)
_, p_ct     = psd_db(ct)
_, p_dp     = psd_db(dp)
_, p_dc     = psd_db(dc)

mask = (f >= 0) & (f <= 4000)
f_khz = f[mask] / 1000

all_psd = np.concatenate([p_clean[mask], p_noisy[mask], p_crn[mask],
                           p_ct[mask], p_dp[mask], p_dc[mask]])
y_min = np.floor(all_psd.min() / 10) * 10
y_max = np.ceil(all_psd.max() / 10) * 10

# ---- Plot ----
models = [
    ('CRN',         p_crn, COLORS['CRN']),
    ('Conv-TasNet', p_ct,  COLORS['ConvTasNet']),
    ('DPRNN',       p_dp,  COLORS['DPRNN']),
    ('DCAMF-Net',   p_dc,  COLORS['DCAMF']),
]

fig, axes = plt.subplots(2, 2, figsize=(6.5, 6))
for ax, (name, pxx, color) in zip(axes.flat, models):
    ax.plot(f_khz, p_noisy[mask], color=COLORS['noisy'], linewidth=0.5, alpha=0.6, label='Noisy')
    ax.plot(f_khz, p_clean[mask], 'k-', linewidth=1.5, label='Clean')
    ax.plot(f_khz, pxx[mask], '--', color=color, linewidth=1.2, label=name)
    for fq in line_freqs:
        ax.axvline(fq/1000, color='gray', linestyle=':', linewidth=0.4)
    ax.set_xlabel('频率 (kHz)'); ax.set_ylabel('PSD (dB/Hz)')
    ax.set_title(name, fontweight='bold'); ax.set_xlim(0, 4); ax.set_ylim(y_min, y_max)
    ax.legend(['带噪信号', '干净信号', '降噪后'], fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig4-3_overall_spectrum_comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig4-3_overall_spectrum_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print('图4-3 已保存')
