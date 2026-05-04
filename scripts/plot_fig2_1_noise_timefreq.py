"""第二章 2.1 — 不同噪声类型的时域/频域/时频图（3×3）
风噪声 / 水流噪声 / 水库噪声，每行: 时域波形 | PSD | 语谱图
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from plot_utils import setup_style, FIG_DIR, FS, RAW_WIND, load_wav
from pathlib import Path

setup_style()
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6})

RAW_ROOT = RAW_WIND.parent

sources = [
    ('风噪声', RAW_WIND, '#EDB120'),
    ('水流噪声', RAW_ROOT / 'flow', '#4DB0B0'),
    ('水库噪声（未知噪声）', RAW_ROOT / 'reservoir', '#7E2F8E'),
]

fig, axes = plt.subplots(3, 3, figsize=(6.5, 4.6))

for row, (label, folder, color) in enumerate(sources):
    files = sorted(folder.glob('*.wav'))
    if not files:
        for col in range(3):
            axes[row][col].text(0.5, 0.5, '无数据', transform=axes[row][col].transAxes,
                                ha='center', va='center', fontsize=10, color='#999')
            axes[row][col].set_title(label)
        continue

    sig = load_wav(files[0])
    L = min(len(sig), 3 * FS)
    sig = sig[:L]
    t = np.arange(L) / FS

    # — 时域波形 —
    ax = axes[row][0]
    ax.plot(t[:4000], sig[:4000], color=color, lw=0.2)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('幅度')
    ax.set_title(f'{label} — 时域波形')
    ax.set_xlim(0, 0.25)
    ax.grid(True, alpha=0.3)

    # — PSD —
    ax = axes[row][1]
    f, pxx = welch(sig, FS, nperseg=2048)
    mask = (f >= 0) & (f <= 4000)
    ax.plot(f[mask] / 1000, 10 * np.log10(pxx[mask] + 1e-10),
            color=color, lw=0.5)
    ax.set_xlabel('频率 (kHz)'); ax.set_ylabel('PSD (dB/Hz)')
    ax.set_title(f'{label} — 功率谱密度')
    ax.set_xlim(0, 4)
    ax.grid(True, alpha=0.3)

    # — 语谱图 —
    ax = axes[row][2]
    f_s, t_s, Sxx = spectrogram(sig, FS, nperseg=256, noverlap=200)
    s_mask = f_s <= 4000
    im = ax.pcolormesh(t_s, f_s[s_mask] / 1000, 10 * np.log10(Sxx[s_mask] + 1e-10),
                       shading='auto', cmap='plasma', rasterized=True)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('频率 (kHz)')
    ax.set_title(f'{label} — 语谱图')
    ax.set_ylim(0, 4)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
fig.savefig(FIG_DIR / 'ch2_noise_analysis.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'ch2_noise_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved to {FIG_DIR / "ch2_noise_analysis.pdf"}')
