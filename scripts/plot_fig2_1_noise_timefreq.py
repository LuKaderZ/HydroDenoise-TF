"""第二章 2.1 — 不同噪声类型的时域/频域/时频图（3×3）
风噪声 / 水流噪声 / 水库噪声，每行: 时域波形 | PSD | 时频图
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

    # — 时频图 —
    ax = axes[row][2]
    f_s, t_s, Sxx = spectrogram(sig, FS, nperseg=256, noverlap=200)
    s_mask = f_s <= 4000
    im = ax.pcolormesh(t_s, f_s[s_mask] / 1000, 10 * np.log10(Sxx[s_mask] + 1e-10),
                       shading='auto', cmap='plasma', rasterized=True)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('频率 (kHz)')
    ax.set_title(f'{label} — 时频图')
    ax.set_ylim(0, 4)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig2-1_noise_timefreq.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig2-1_noise_timefreq.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 终端数据输出（用于正文分析） ----
print('\n========== 图2-1 噪声信号特征分析 ==========')
bands = [(0, 500), (500, 2000), (2000, 4000)]

for label, folder, _ in sources:
    files = sorted(folder.glob('*.wav'))
    if not files:
        continue

    sig = load_wav(files[0])
    L = min(len(sig), 3 * FS)
    sig = sig[:L]
    rms = np.sqrt(np.mean(sig ** 2))

    f_full, pxx_full = welch(sig, FS, nperseg=2048)
    f_mask = f_full <= 4000
    f_khz = f_full[f_mask] / 1000
    pxx_db = 10 * np.log10(pxx_full[f_mask] + 1e-10)

    # 频段能量占比
    pxx_lin = pxx_full[f_mask]  # 线性功率
    total_power = np.sum(pxx_lin)
    band_info = []
    for lo, hi in bands:
        bm = (f_full[f_mask] >= lo) & (f_full[f_mask] <= hi)
        pct = np.sum(pxx_lin[bm]) / total_power * 100
        mean_db = np.mean(pxx_db[bm])
        band_info.append((lo, hi, pct, mean_db))

    # 谱峰
    peak_idx = np.argmax(pxx_db)
    peak_freq = f_khz[peak_idx] * 1000
    peak_db = pxx_db[peak_idx]

    # 谱质心
    centroid = np.sum(f_full[f_mask] * pxx_lin) / total_power

    print(f'\n--- {label} ---')
    print(f'  RMS 幅度: {rms:.4f}')
    print(f'  谱质心: {centroid:.0f} Hz')
    print(f'  峰值频率: {peak_freq:.0f} Hz ({peak_db:.1f} dB/Hz)')
    for lo, hi, pct, mean_db in band_info:
        print(f'  {lo:4d}–{hi:4d} Hz: 能量占比 {pct:5.1f}%  均值 {mean_db:.1f} dB/Hz')
    print(f'  PSD 整体范围: {pxx_db.min():.1f} ~ {pxx_db.max():.1f} dB/Hz')

print('\n============================================')
print(f'Saved to {FIG_DIR / "fig2-1_noise_timefreq.pdf"}')
