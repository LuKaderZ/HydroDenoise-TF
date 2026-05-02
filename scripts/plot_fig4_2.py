"""图4-2: 各模型降噪前后时域波形对比 (50ms瞬态窗)"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from plot_utils import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---- Select best sample ----
print('Selecting best sample...')
best_idx = select_best_transient()
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

# ---- Extract 50ms transient window ----
win_len = int(0.05 * FS)
energies = np.array([np.sum(noisy[s:s+win_len]**2) for s in range(0, L-win_len, win_len)])
start = np.argmax(energies) * win_len
end = start + win_len

signals = {
    '干净信号':      clean[start:end],
    '带噪信号':      noisy[start:end],
    'CRN':           crn[start:end],
    'Conv-TasNet':   ct[start:end],
    'DPRNN':         dp[start:end],
    'DCAMF-Net':     dc[start:end],
}
# Gaussian smooth
for k in signals: signals[k] = gaussian_filter1d(signals[k], sigma=1.0)

t_ms = np.arange(win_len) / FS * 1000

# ---- Plot ----
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
model_colors = {
    '干净信号': COLORS['clean'], '带噪信号': COLORS['noisy'],
    'CRN': COLORS['CRN'], 'Conv-TasNet': COLORS['ConvTasNet'],
    'DPRNN': COLORS['DPRNN'], 'DCAMF-Net': COLORS['DCAMF'],
}
# Unified y-axis for non-noisy signals
other_signals = np.concatenate([signals[k] for k in ['干净信号','CRN','Conv-TasNet','DPRNN','DCAMF-Net']])
y_other = (other_signals.min() - 0.05*np.ptp(other_signals), other_signals.max() + 0.05*np.ptp(other_signals))
y_noisy = (signals['带噪信号'].min() - 0.05*np.ptp(signals['带噪信号']),
           signals['带噪信号'].max() + 0.05*np.ptp(signals['带噪信号']))

for ax, (label, sig) in zip(axes.flat, signals.items()):
    ax.plot(t_ms, sig, color=model_colors[label], linewidth=0.8)
    if label != '干净信号':
        ax.plot(t_ms, signals['干净信号'], 'k-', linewidth=0.4, alpha=0.5)
    ax.set_xlabel('时间 (ms)'); ax.set_ylabel('幅度')
    ax.set_title(label, fontweight='bold')
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_ylim(y_noisy if label == 'Noisy' else y_other)
    ax.grid(True)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig4-2_time_waveform_comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig4-2_time_waveform_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print('图4-2 已保存')
