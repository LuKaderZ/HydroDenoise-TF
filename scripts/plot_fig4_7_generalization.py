"""图4-7: DCAMF-Net 泛化性能评估 (ShipsEar 三个测试集, SI-SNRi + SDRi)"""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import *

setup_style()

target_snrs = [-15, -10, -5]
test_sets = {
    'test1': (DATA_T1_CLEAN, DATA_T1_NOISY, DC_DIR),
    'test2': (DATA_T2_CLEAN, DATA_T2_NOISY, DC_DIR_T2),
    'test3': (DATA_T3_CLEAN, DATA_T3_NOISY, DC_DIR_T3),
}
names = ['测试集一 (已知船型+已知噪声)', '测试集二 (未知船型+已知噪声)', '测试集三 (已知船型+未知噪声)']

# ---- Compute SI-SNRi/SDRi per sample per test set ----
all_results = []
for s, (key, (clean_d, noisy_d, dc_d)) in enumerate(test_sets.items()):
    cfiles = sorted(clean_d.glob('*.wav'))
    for f in cfiles:
        dp = dc_d / f.name
        if not dp.exists(): continue
        clean = load_wav(f)
        noisy = load_wav(noisy_d / f.name)
        denoised = load_wav(dp)
        L = min(len(clean), len(noisy), len(denoised))
        clean, noisy, denoised = clean[:L], noisy[:L], denoised[:L]
        noise_actual = noisy - clean
        actual_snr = 10 * np.log10(np.mean(clean**2) / (np.mean(noise_actual**2) + 1e-10))
        snr_idx = np.argmin(np.abs(np.array(target_snrs) - actual_snr))
        sii = compute_sisnr(denoised, clean) - compute_sisnr(noisy, clean)
        sdi = compute_sdr(denoised, clean) - compute_sdr(noisy, clean)
        all_results.append({'set': s, 'snr': target_snrs[snr_idx], 'sii': sii, 'sdi': sdi})

# ---- Aggregate ----
sisnri = np.full((3, 3), np.nan)
sdri   = np.full((3, 3), np.nan)
for s in range(3):
    for t, snr in enumerate(target_snrs):
        mask = [(r['set'] == s and r['snr'] == snr) for r in all_results]
        if any(mask):
            sisnri[s, t] = np.mean([r['sii'] for r, m in zip(all_results, mask) if m])
            sdri[s, t]   = np.mean([r['sdi'] for r, m in zip(all_results, mask) if m])

bar_colors = [COLORS['ConvTasNet'], COLORS['CRN'], '#7E2F8E']  # blue, orange, purple

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))

for ax, data, ylabel, title in [
    (axes[0], sisnri, 'SI-SNRi (dB)', '(a) SI-SNR 提升量'),
    (axes[1], sdri,   'SDRi (dB)',   '(b) SDR 提升量'),
]:
    x = np.arange(3)
    w = 0.25
    for i in range(3):
        bars = ax.bar(x + i*w, data[i], w, label=names[i],
                      color=bar_colors[i], edgecolor='black', linewidth=0.5)
        for j, (bx, by) in enumerate(zip(x + i*w, data[i])):
            if not np.isnan(by):
                va = 'top' if by < 0 else 'bottom'
                offset = -0.5 if by < 0 else 0
                ax.text(bx, by + offset, f'{by:.2f}', ha='center', va=va, fontsize=7)

    ax.set_xticks(x + w)
    ax.set_xticklabels([f'{s} dB' for s in target_snrs])
    ax.set_xlabel('输入信噪比 (dB)'); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6, handlelength=1.0, borderpad=0.3, labelspacing=0.3); ax.grid(axis='y', alpha=0.3)

    all_vals = data[~np.isnan(data)]
    if len(all_vals) > 0:
        yb = np.floor(np.min(all_vals)) - 1 if np.min(all_vals) < 0 else 0
        ax.set_ylim(yb, np.ceil(np.max(all_vals)) + 2)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='-')

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig4-7_DCAMF_Net_ShipsEar.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig4-7_DCAMF_Net_ShipsEar.png', dpi=300, bbox_inches='tight')
plt.show()
print('图4-7 已保存')
