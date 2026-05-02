"""图4-5: 各模型关键线谱功率恢复偏差条形图"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import *

setup_style()

# ---- Detect line spectra + select best sample ----
line_freqs = find_line_spectra(n_peaks=5)
print(f'Line frequencies: {np.round(line_freqs).astype(int).tolist()} Hz')
best_idx = select_best_linespectra(line_freqs)
print(f'Selected sample: {best_idx}')

# ---- Load audio ----
clean_files = sorted(DATA_T1_CLEAN.glob('*.wav'))
fname = clean_files[best_idx - 1].name
clean = load_wav(DATA_T1_CLEAN / fname)
noisy = load_wav(DATA_T1_NOISY / fname)
crn = load_est(CRN_DIR, best_idx, 1)
ct  = load_est(CT_DIR, best_idx, 2)
dp  = load_est(DP_DIR, best_idx, 2)
dc  = load_est(DC_DIR, best_idx, 3)
L = min(len(clean), len(noisy), len(crn), len(ct), len(dp), len(dc))
clean, crn, ct, dp, dc = clean[:L], crn[:L], ct[:L], dp[:L], dc[:L]

# ---- Compute power at each line frequency ----
models = {'CRN': crn, 'Conv-TasNet': ct, 'DPRNN': dp, 'DCAMF-Net': dc}
f = np.fft.rfftfreq(4096, 1/FS)
powers = np.zeros((len(line_freqs), 5))  # clean, crn, ct, dp, dc

for i, fq in enumerate(line_freqs):
    i0 = np.argmin(np.abs(f - fq))
    w = slice(max(0, i0-4), min(len(f), i0+5))
    _, p_clean = welch(clean, FS, nperseg=2048, noverlap=1024, nfft=4096)
    powers[i, 0] = np.mean(10*np.log10(p_clean[w] + 1e-10))
    for j, (name, sig) in enumerate(models.items()):
        _, pxx = welch(sig, FS, nperseg=2048, noverlap=1024, nfft=4096)
        powers[i, j+1] = np.mean(10*np.log10(pxx[w] + 1e-10))

dev = powers[:, 1:] - powers[:, 0:1]  # deviation from clean

# ---- Plot ----
bar_colors = [COLORS['CRN'], COLORS['ConvTasNet'], COLORS['DPRNN'], COLORS['DCAMF']]
model_names = ['CRN', 'Conv-TasNet', 'DPRNN', 'DCAMF-Net']
freq_labels = [f'{int(round(fq))} Hz' for fq in line_freqs]

fig, ax = plt.subplots(figsize=(6, 3.5))
x = np.arange(len(line_freqs)); w = 0.2

for i in range(4):
    bars = ax.bar(x + (i - 1.5)*w, dev[:, i], w, label=model_names[i],
                  color=bar_colors[i], edgecolor='black', linewidth=0.5)
    for j, val in enumerate(dev[:, i]):
        if abs(val) > 0.5:
            ax.text(x[j] + (i - 1.5)*w, val, f'{val:.1f}',
                    ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=7, color='black')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(freq_labels)
ax.set_xlabel('关键线谱频率'); ax.set_ylabel('相对于干净信号的功率偏差 (dB)')
ax.set_title('各模型线谱功率恢复对比 (0 dB 为完美恢复)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

dev_vals = dev.ravel()
y_hi = max(dev_vals.max() * 1.15, 2)
y_lo = min(dev_vals.min() * 1.15, -2) if dev_vals.min() < 0 else -2
ax.set_ylim(y_lo, y_hi)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig4-5_Line_Spectra_Bar.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig4-5_Line_Spectra_Bar.png', dpi=300, bbox_inches='tight')
plt.show()
print('图4-5 已保存')
