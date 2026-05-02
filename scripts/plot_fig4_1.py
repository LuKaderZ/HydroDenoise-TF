"""图4-1: 船舶信号、背景噪声及混合信号的PSD概览"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import *
from scipy.signal import welch

setup_style()

# ---- Compute average PSDs ----
def avg_psd(folder):
    files = sorted(Path(folder).glob('*.wav'))
    pxx_sum = None
    for f in files:
        sig = load_wav(f)
        _, pxx = welch(sig, FS, nperseg=1024, noverlap=512, nfft=1024)
        if pxx_sum is None: pxx_sum = pxx
        else: pxx_sum += pxx
    return pxx_sum / len(files)

print('Computing average PSDs...')
pxx_passenger = avg_psd(RAW_PASSENGER)
pxx_roro      = avg_psd(RAW_RORO)
pxx_noise     = avg_psd(RAW_WIND)
f = np.fft.rfftfreq(1024, 1/FS)

# ---- Mixed signals ----
pf = sorted(RAW_PASSENGER.glob('*.wav'))[0]
rf = sorted(RAW_RORO.glob('*.wav'))[0]
nf = sorted(RAW_WIND.glob('*.wav'))[0]
psig = load_wav(pf); rsig = load_wav(rf); nsig = load_wav(nf)

def mix(clean, noise, snr_db):
    P_s = np.mean(clean**2); P_n = np.mean(noise**2)
    alpha = np.sqrt(P_s / (P_n * 10**(snr_db/10)))
    return clean + alpha * noise

def prepare_noise(noise, target_len):
    if len(noise) < target_len:
        rep = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, rep)
    start = np.random.randint(0, len(noise) - target_len + 1)
    return noise[start:start + target_len]

noise_p = prepare_noise(nsig, len(psig))
noise_r = prepare_noise(nsig, len(rsig))
mix_p = mix(psig, noise_p, -15); mix_r = mix(rsig, noise_r, -15)
_, pxx_mp = welch(mix_p, FS, nperseg=1024, noverlap=512, nfft=1024)
_, pxx_mr = welch(mix_r, FS, nperseg=1024, noverlap=512, nfft=1024)

# ---- Plot ----
psd = lambda p: 10 * np.log10(p + 1e-10)
mask = (f >= 0) & (f <= 4000); f_plot = f[mask]
all_dB = np.concatenate([psd(pxx_passenger)[mask], psd(pxx_roro)[mask],
                          psd(pxx_noise)[mask], psd(pxx_mp)[mask], psd(pxx_mr)[mask]])
y_margin = 0.05 * (all_dB.max() - all_dB.min())

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
panels = [
    (axes[0,0], psd(pxx_passenger), '(a) 客船 PSD', COLORS['ConvTasNet']),
    (axes[0,1], psd(pxx_roro), '(b) 滚装船 PSD', '#77AC30'),
    (axes[1,0], psd(pxx_noise), '(c) 背景噪声 PSD', COLORS['noisy']),
]
for ax, data, title, color in panels:
    ax.plot(f_plot/1000, data[mask], color=color, linewidth=1.0)
    ax.set_xlabel('频率 (Hz)'); ax.set_ylabel('PSD (dB/Hz)')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(0, 4); ax.grid(True)
    ax.set_ylim(all_dB.min() - y_margin, all_dB.max() + y_margin)

# Mixed PSD (bottom-right)
ax = axes[1,1]
ax.plot(f_plot/1000, psd(pxx_mp)[mask], color=COLORS['ConvTasNet'], linewidth=1.0, label='客船混合')
ax.plot(f_plot/1000, psd(pxx_mr)[mask], color='#77AC30', linewidth=1.0, label='滚装船混合')
ax.set_xlabel('频率 (Hz)'); ax.set_ylabel('PSD (dB/Hz)')
ax.set_title('(d) 混合信号 PSD (-15 dB)', fontweight='bold')
ax.set_xlim(0, 4); ax.legend(fontsize=8); ax.grid(True)
ax.set_ylim(all_dB.min() - y_margin, all_dB.max() + y_margin)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig4-1_ShipsEar_PSD.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig4-1_ShipsEar_PSD.png', dpi=300, bbox_inches='tight')
plt.show()
print('图4-1 已保存')
