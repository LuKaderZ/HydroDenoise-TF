"""图2-3: 船舶信号与不同噪声在-15dB下混合PSD对比（3×3）
行=船型(客船/滚装船/摩托艇), 列=噪声(风噪/水流噪/水库噪)
每张叠加: 干净信号 PSD | -15dB混合信号 PSD
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path
from plot_utils import setup_style, FIG_DIR, FS, load_wav, PROJECT_ROOT

setup_style()
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6})

SHIPS = [
    ('客船',   PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'passenger'),
    ('滚装船', PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'roro'),
    ('摩托艇', PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'motorboat'),
]
NOISES = [
    ('风噪声',   PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'wind'),
    ('水流噪声', PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'flow'),
    ('水库噪声', PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'reservoir'),
]
SNR_DB = -15


def load_average_signal(folder):
    """加载文件夹下所有音频，返回平均后的单通道信号和采样率"""
    files = sorted(folder.glob('*.wav'))
    sigs = [load_wav(f) for f in files]
    # 截到最短长度
    min_len = min(len(s) for s in sigs)
    sigs = [s[:min_len] for s in sigs]
    return np.mean(sigs, axis=0)


def mix_at_snr(clean, noise, snr_db):
    """按指定SNR混合，噪声不够长时循环拼接"""
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]
    p_s = np.mean(clean ** 2) + 1e-10
    p_n = np.mean(noise ** 2) + 1e-10
    alpha = np.sqrt(p_s / (p_n * (10 ** (snr_db / 10))))
    return clean + alpha * noise


# ---- 先算所有 PSD 确定统一 Y 轴 ----
all_psd_db = []
for ship_name, ship_dir in SHIPS:
    clean = load_average_signal(ship_dir)
    for noise_name, noise_dir in NOISES:
        noise = load_average_signal(noise_dir)
        noisy = mix_at_snr(clean, noise, SNR_DB)
        f_arr, _ = welch(clean, FS, nperseg=2048)
        _, pxx_c = welch(clean, FS, nperseg=2048)
        _, pxx_n = welch(noisy, FS, nperseg=2048)
        mask = (f_arr >= 0) & (f_arr <= 4000)
        all_psd_db.append(10 * np.log10(pxx_c[mask] + 1e-10))
        all_psd_db.append(10 * np.log10(pxx_n[mask] + 1e-10))

y_min = min(p.min() for p in all_psd_db)
y_max = max(p.max() for p in all_psd_db)
y_margin = (y_max - y_min) * 0.06
f_khz = f_arr[mask] / 1000

# ---- 出图 ----
fig, axes = plt.subplots(3, 3, figsize=(6.5, 5.5))

for row, (ship_name, ship_dir) in enumerate(SHIPS):
    clean = load_average_signal(ship_dir)

    for col, (noise_name, noise_dir) in enumerate(NOISES):
        ax = axes[row][col]
        noise = load_average_signal(noise_dir)
        noisy = mix_at_snr(clean, noise, SNR_DB)

        _, pxx_c = welch(clean, FS, nperseg=2048)
        _, pxx_n = welch(noisy, FS, nperseg=2048)
        pxx_c_db = 10 * np.log10(pxx_c[mask] + 1e-10)
        pxx_n_db = 10 * np.log10(pxx_n[mask] + 1e-10)

        ax.plot(f_khz, pxx_c_db, color='#000000', lw=0.8, label='干净')
        ax.plot(f_khz, pxx_n_db, color='#D95319', lw=0.5, alpha=0.7, label='混合')
        ax.set_xlim(0, 4)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.grid(True, alpha=0.25, lw=0.4)

        # 行标签（仅第一列）
        if col == 0:
            ax.set_ylabel(f'{ship_name}\nPSD (dB/Hz)', fontsize=7)
        # 底部 X 标签
        if row == 2:
            ax.set_xlabel('频率 (kHz)', fontsize=7)

        # SNR 标注
        ax.text(0.97, 0.12, f'{SNR_DB}dB', transform=ax.transAxes, fontsize=6,
                color='#D95319', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, lw=0))

        # 图例（仅第一张）
        if row == 0 and col == 0:
            ax.legend(fontsize=6, loc='upper right', framealpha=0.8)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig2-3_noisy_psd.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig2-3_noisy_psd.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- 终端输出 ----
print(f'\n========== 图2-3 -15dB混合PSD分析 ==========')
for ship_name, ship_dir in SHIPS:
    clean = load_average_signal(ship_dir)
    for noise_name, noise_dir in NOISES:
        noise = load_average_signal(noise_dir)
        noisy = mix_at_snr(clean, noise, SNR_DB)
        _, pxx_c = welch(clean, FS, nperseg=2048)
        _, pxx_n = welch(noisy, FS, nperseg=2048)
        mask = f_arr <= 4000

        # 频段内干净vs混合 PSD 差值
        diff_db = 10 * np.log10(pxx_n[mask] + 1e-10) - 10 * np.log10(pxx_c[mask] + 1e-10)
        print(f'\n{ship_name} + {noise_name}:')
        print(f'  0-500Hz   ΔPSD = {np.mean(diff_db[(f_arr[mask]>=0)&(f_arr[mask]<=500)]):.1f} dB')
        print(f'  500-2000Hz ΔPSD = {np.mean(diff_db[(f_arr[mask]>=500)&(f_arr[mask]<=2000)]):.1f} dB')
        print(f'  2000-4000Hz ΔPSD = {np.mean(diff_db[(f_arr[mask]>=2000)&(f_arr[mask]<=4000)]):.1f} dB')
print('==========================================')
