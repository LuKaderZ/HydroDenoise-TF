"""图4-4：各模型降噪后时频谱图对比 (plasma配色)"""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import spectrogram

from plot_utils import setup_style
setup_style()

project_root = Path(r'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF')

clean_dir = project_root / 'data' / 'ShipsEar' / 'test1' / 'clean'
noisy_dir = project_root / 'data' / 'ShipsEar' / 'test1' / 'noisy'
crn_dir   = project_root / 'baselines' / 'CRN-causal' / 'data' / 'data' / 'datasets' / 'tt' / 'tt_test1'
ct_dir    = project_root / 'experiments' / 'conv_tasnet' / 'estimates' / 'tt_test1'
dp_dir    = project_root / 'experiments' / 'dprnn' / 'estimates' / 'tt_test1'
dc_dir    = project_root / 'experiments' / 'dcamf_net' / 'denoised' / 'ShipsEar_test1'

fs = 16000
nperseg, noverlap, nfft = 256, 200, 512
freq_lim = 4000

# ------- 选取与图4-2一致的样本 (瞬态最强段 DCAMF-Net SI-SNRi 最高) -------
def compute_sisnr(est, ref):
    est = est - est.mean(); ref = ref - ref.mean()
    dot = np.dot(est, ref)
    target = dot * ref / (np.dot(ref, ref) + 1e-8)
    noise = est - target
    return 10 * np.log10(np.sum(target**2) / (np.sum(noise**2) + 1e-8) + 1e-8)

clean_files = sorted(list(clean_dir.glob('*.wav')))
best_sii, best_idx = -np.inf, 0
win_len = int(0.05 * fs)

for k, f in enumerate(clean_files):
    dcamf_file = dc_dir / f'{k+1:06d}.wav'
    if not dcamf_file.exists(): continue
    clean, _ = sf.read(clean_dir / f.name); clean = clean.mean(axis=1) if clean.ndim > 1 else clean
    noisy, _ = sf.read(noisy_dir / f.name); noisy = noisy.mean(axis=1) if noisy.ndim > 1 else noisy
    dcamf, _ = sf.read(dcamf_file); dcamf = dcamf.mean(axis=1) if dcamf.ndim > 1 else dcamf
    L = min(len(clean), len(noisy), len(dcamf))
    clean, noisy, dcamf = clean[:L], noisy[:L], dcamf[:L]
    if L < win_len: continue
    energies = np.array([np.sum(noisy[s:s+win_len]**2) for s in range(0, L-win_len, win_len)])
    start = np.argmax(energies) * win_len
    end = min(start + win_len, L)
    si = compute_sisnr(dcamf[start:end], clean[start:end]) - compute_sisnr(noisy[start:end], clean[start:end])
    if si > best_sii: best_sii, best_idx = si, k

print(f'选定样本: {best_idx}')

# ------- 加载该样本的全部模型输出 -------
def load_est(est_dir, idx, est_type):
    if est_type == 1: f = est_dir / f'{idx}_sph_est.wav'
    elif est_type == 2: f = est_dir / f'{idx:06d}_sph_est.wav'
    else: f = est_dir / f'{idx+1:06d}.wav'
    s, _ = sf.read(f); return s.mean(axis=1) if s.ndim > 1 else s

fname = clean_files[best_idx].name
clean, _ = sf.read(clean_dir / fname); clean = clean.mean(axis=1) if clean.ndim > 1 else clean
noisy, _ = sf.read(noisy_dir / fname); noisy = noisy.mean(axis=1) if noisy.ndim > 1 else noisy
crn = load_est(crn_dir, best_idx, 1)
ct  = load_est(ct_dir, best_idx, 2)
dp  = load_est(dp_dir, best_idx, 2)
dc  = load_est(dc_dir, best_idx, 3)

min_len = min(len(clean), len(noisy), len(crn), len(ct), len(dp), len(dc))
clean, noisy = clean[:min_len], noisy[:min_len]
crn, ct, dp, dc = crn[:min_len], ct[:min_len], dp[:min_len], dc[:min_len]

# ------- 计算语谱图 -------
signals = [clean, noisy, crn, ct, dp, dc]
titles  = ['干净信号', '带噪信号', 'CRN', 'Conv-TasNet', 'DPRNN', 'DCAMF-Net']
all_psd = []

for sig in signals:
    f, t, Sxx = spectrogram(sig, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    mask = f <= freq_lim
    all_psd.append(10 * np.log10(Sxx[mask] + 1e-10))

vmin = np.floor(min(np.concatenate([p.ravel() for p in all_psd])) / 5) * 5
vmax = np.ceil(max(np.concatenate([p.ravel() for p in all_psd])) / 5) * 5
f_plot = f[f <= freq_lim] / 1000

# ------- 绘图 -------
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for i, ax in enumerate(axes.flat):
    im = ax.pcolormesh(t, f_plot, all_psd[i], shading='auto', cmap='plasma',
                        vmin=vmin, vmax=vmax)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('频率 (kHz)')
    ax.set_title(titles[i], fontweight='bold')
    fig.colorbar(im, ax=ax)

plt.subplots_adjust(hspace=0.45, wspace=0.25)

save_dir = project_root / 'figures'
save_dir.mkdir(exist_ok=True)
fig.savefig(save_dir / 'fig4-4_Spectrogram_Comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(save_dir / 'fig4-4_Spectrogram_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print(f'图4-4 已保存至 {save_dir}')
