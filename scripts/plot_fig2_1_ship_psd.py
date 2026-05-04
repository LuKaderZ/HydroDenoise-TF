"""图2-1: ShipsEar船舶辐射噪声功率谱密度（含线谱检测）
1×3: 客船 PSD | 滚装船 PSD | 摩托艇 PSD
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path
from plot_utils import (setup_style, FIG_DIR, FS, load_wav, PROJECT_ROOT,
                        find_line_spectra)

setup_style()
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7})

SHIPS = [
    ('客船',   PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'passenger', '#2471A3'),
    ('滚装船', PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'roro',      '#2E86C1'),
    ('摩托艇', PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'motorboat', '#5DADE2'),
]

# ---- 第一遍：算平均 PSD + 统一 Y 轴 ----
all_pxx_db = []; all_f_khz = None
for _, folder, _ in SHIPS:
    files = sorted(folder.glob('*.wav'))
    avg_pxx = None
    for f in files:
        sig = load_wav(f)
        f_arr, pxx = welch(sig, FS, nperseg=2048)
        avg_pxx = pxx if avg_pxx is None else avg_pxx + pxx
    avg_pxx /= len(files)
    mask = (f_arr >= 0) & (f_arr <= 4000)
    all_f_khz = f_arr[mask] / 1000
    all_pxx_db.append(10 * np.log10(avg_pxx[mask] + 1e-10))

y_min = min(p.min() for p in all_pxx_db)
y_max = max(p.max() for p in all_pxx_db)
y_margin = (y_max - y_min) * 0.08

# ---- 第二遍：出图 + 线谱标记 ----
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))

for idx, (label, folder, color) in enumerate(SHIPS):
    ax = axes[idx]
    files = sorted(folder.glob('*.wav'))
    if not files:
        ax.text(0.5, 0.5, '无数据', transform=ax.transAxes, ha='center', va='center')
        continue

    pxx_db = all_pxx_db[idx]

    ax.plot(all_f_khz, pxx_db, color=color, lw=0.8)
    ax.set_xlabel('频率 (kHz)')
    if idx == 0:
        ax.set_ylabel('PSD (dB/Hz)')
    ax.set_xlim(0, 4)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.grid(True, alpha=0.3)

    # 谱质心
    avg_pxx_lin = 10 ** (pxx_db / 10)
    centroid = np.sum(f_arr[mask] * avg_pxx_lin) / (np.sum(avg_pxx_lin) + 1e-10)
    ax.axvline(centroid / 1000, color=color, linestyle='--', lw=0.8, alpha=0.4)
    ax.text(centroid / 1000 + 0.05, y_min + y_margin + 2,
            f'SC={centroid:.0f} Hz', color=color, fontsize=6.5, alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, lw=0))

    ax.set_title(f'({"abc"[idx]}) {label}', fontsize=9, fontweight='bold', pad=4)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig2-1_ship_psd.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig2-1_ship_psd.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 终端数据输出（用于论文写作） ----
print('\n========== 图2-1 船舶辐射噪声 PSD 与线谱分析 ==========')
for label, folder, color in SHIPS:
    files = sorted(folder.glob('*.wav'))
    print(f'\n--- {label} ({len(files)}个文件) ---')

    # RMS
    sigs = [load_wav(f) for f in files]
    rms_vals = [np.sqrt(np.mean(s**2)) for s in sigs]
    print(f'  RMS: 均值={np.mean(rms_vals):.4f}, 标准差={np.std(rms_vals):.4f}')

    # 平均 PSD 统计
    avg_pxx = None
    for f in files:
        sig = load_wav(f)
        _, pxx = welch(sig, FS, nperseg=2048)
        avg_pxx = pxx if avg_pxx is None else avg_pxx + pxx
    avg_pxx /= len(files)
    mask = (f_arr >= 0) & (f_arr <= 4000)
    pxx_db = 10 * np.log10(avg_pxx[mask] + 1e-10)

    # 整体统计
    print(f'  PSD 范围: {pxx_db.min():.1f} ~ {pxx_db.max():.1f} dB/Hz')

    # 谱峰
    peak_idx = np.argmax(pxx_db)
    print(f'  最高谱峰: {f_arr[mask][peak_idx]:.0f} Hz ({pxx_db[peak_idx]:.1f} dB/Hz)')

    # 频段能量占比
    pxx_lin = avg_pxx[mask]
    total = np.sum(pxx_lin)
    for lo, hi in [(0, 500), (500, 2000), (2000, 4000)]:
        bm = (f_arr[mask] >= lo) & (f_arr[mask] <= hi)
        print(f'  {lo:4d}–{hi:4d} Hz: 能量占比 {np.sum(pxx_lin[bm])/total*100:5.1f}%')

    # 线谱检测
    centroid = np.sum(f_arr[mask] * pxx_lin) / (total + 1e-10)
    print(f'  谱质心: {centroid:.0f} Hz')
    line_freqs = find_line_spectra(folder, n_peaks=5, freq_range=(100, 4000))
    print(f'  检测线谱: {np.round(line_freqs).astype(int).tolist()} Hz')

    # 线谱谐波关系检查
    if len(line_freqs) >= 2:
        ratios = []
        for i in range(len(line_freqs)):
            for j in range(i + 1, len(line_freqs)):
                r = line_freqs[j] / line_freqs[i]
                k = round(r)
                if abs(r - k) < 0.06 and k >= 2:
                    ratios.append(f'{line_freqs[j]:.0f}/{line_freqs[i]:.0f}≈{k}:1')
        if ratios:
            print(f'  谐波关系: {", ".join(ratios)}')

print('\n==========================================')
