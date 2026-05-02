"""Shared utilities for all plot scripts."""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch, find_peaks

def setup_style():
    """论文统一风格：字号适中，PDF矢量输出保证插入Word后清晰可读。

    论文 A4 页边距左2.5右2.0cm，正文宽约 16.5 cm (6.5 in)，每张图宽设为此值，
    保存时 bbox_inches='tight' 自动裁白边，插图时缩放到列宽即可。
    """
    plt.rcParams.update({
        'font.sans-serif': ['SimHei'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.unicode_minus': False,
    })

PROJECT_ROOT = Path(r'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF')
FS = 16000

# ---- Paths ----
DATA_T1_CLEAN = PROJECT_ROOT / 'data' / 'ShipsEar' / 'test1' / 'clean'
DATA_T1_NOISY = PROJECT_ROOT / 'data' / 'ShipsEar' / 'test1' / 'noisy'
DATA_T2_CLEAN = PROJECT_ROOT / 'data' / 'ShipsEar' / 'test2' / 'clean'
DATA_T2_NOISY = PROJECT_ROOT / 'data' / 'ShipsEar' / 'test2' / 'noisy'
DATA_T3_CLEAN = PROJECT_ROOT / 'data' / 'ShipsEar' / 'test3' / 'clean'
DATA_T3_NOISY = PROJECT_ROOT / 'data' / 'ShipsEar' / 'test3' / 'noisy'
RAW_PASSENGER = PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'passenger'
RAW_RORO      = PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'roro'
RAW_WIND      = PROJECT_ROOT / 'raw_data' / 'ShipsEar' / 'wind'

CRN_DIR  = PROJECT_ROOT / 'baselines' / 'CRN-causal' / 'data' / 'data' / 'datasets' / 'tt' / 'tt_test1'
CT_DIR   = PROJECT_ROOT / 'experiments' / 'conv_tasnet' / 'estimates' / 'tt_test1'
DP_DIR   = PROJECT_ROOT / 'experiments' / 'dprnn' / 'estimates' / 'tt_test1'
DC_DIR   = PROJECT_ROOT / 'experiments' / 'dcamf_net' / 'denoised' / 'ShipsEar_test1'
DC_DIR_T2 = PROJECT_ROOT / 'experiments' / 'dcamf_net' / 'denoised' / 'ShipsEar_test2'
DC_DIR_T3 = PROJECT_ROOT / 'experiments' / 'dcamf_net' / 'denoised' / 'ShipsEar_test3'
FUSION_LOG_DIR = PROJECT_ROOT / 'experiments' / 'mask_fusion_weights'

FIG_DIR = PROJECT_ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# ---- Colors (IEEE style) ----
COLORS = {
    'clean':  '#000000',
    'noisy':  '#808080',
    'CRN':    '#D95319',
    'ConvTasNet': '#0072BD',
    'DPRNN':  '#EDB120',
    'DCAMF':  '#A2142F',
}

# Test set metadata
TEST_SETS = {
    'test1': (DATA_T1_CLEAN, DATA_T1_NOISY, DC_DIR, 'Test-1 (Known + Known)'),
    'test2': (DATA_T2_CLEAN, DATA_T2_NOISY, DC_DIR_T2, 'Test-2 (Unseen Vessel)'),
    'test3': (DATA_T3_CLEAN, DATA_T3_NOISY, DC_DIR_T3, 'Test-3 (Unseen Noise)'),
}

# ---- Audio loading ----
def load_wav(path):
    s, _ = sf.read(str(path))
    return s.mean(axis=1) if s.ndim > 1 else s

def load_est(est_dir, idx, est_type):
    """Load model estimate. est_type: 1=CRN, 2=ConvTasNet/DPRNN, 3=DCAMF-Net."""
    if est_type == 1:
        f = Path(est_dir) / f'{idx-1}_sph_est.wav'
    elif est_type == 2:
        f = Path(est_dir) / f'{idx-1:06d}_sph_est.wav'
    elif est_type == 3:
        f = Path(est_dir) / f'{idx:06d}.wav'
    else:
        raise ValueError(f'Unknown est_type: {est_type}')
    if not f.exists():
        raise FileNotFoundError(f'Estimate not found: {f}')
    return load_wav(f)

# ---- Metrics ----
def compute_sisnr(est, ref):
    est = est.reshape(-1); ref = ref.reshape(-1)
    est = est - est.mean(); ref = ref - ref.mean()
    dot = np.dot(est, ref)
    scale = dot / (np.dot(ref, ref) + 1e-8)
    target = scale * ref
    noise = est - target
    return 10 * np.log10(np.sum(target**2) / (np.sum(noise**2) + 1e-8) + 1e-8)

def compute_sdr(est, ref):
    """原版MATLAB公式: SDR = 10*log10(||ref||^2 / ||ref-est||^2)"""
    est = est.reshape(-1); ref = ref.reshape(-1)
    noise = ref - est
    return 10 * np.log10(np.sum(ref**2) / (np.sum(noise**2) + 1e-8) + 1e-8)

def psd_db(sig, nperseg=1024, noverlap=None):
    if noverlap is None: noverlap = nperseg // 2
    f, pxx = welch(sig, FS, nperseg=nperseg, noverlap=noverlap, nfft=nperseg)
    return f, 10 * np.log10(pxx + 1e-10)

# ---- Line spectrum detection ----
def find_line_spectra(audio_dir=RAW_PASSENGER, n_peaks=5, freq_range=(0, 4000)):
    """Auto-detect prominent line frequencies from average PSD of ship audio."""
    files = sorted(Path(audio_dir).glob('*.wav'))
    avg_pxx = None
    for f in files:
        sig = load_wav(f)
        _, pxx = welch(sig, FS, nperseg=2048, noverlap=1024, nfft=4096)
        if avg_pxx is None: avg_pxx = pxx
        else: avg_pxx += pxx
    avg_pxx /= len(files)

    f = np.fft.rfftfreq(4096, 1/FS)
    mask = (f >= freq_range[0]) & (f <= freq_range[1])
    pxx_db = 10 * np.log10(avg_pxx[mask] + 1e-10)
    f_sub = f[mask]

    peaks, props = find_peaks(pxx_db, prominence=3, distance=3)  # ~12 Hz, close to MATLAB's 10 Hz
    if len(peaks) > n_peaks:
        top = np.argsort(props['prominences'])[-n_peaks:]
        peaks = peaks[top]
    return np.sort(f_sub[peaks])

# ---- Sample selection ----
def select_best_transient():
    """Select sample with highest DCAMF SI-SNRi in transient window (like Fig 4.2)."""
    clean_files = sorted(DATA_T1_CLEAN.glob('*.wav'))
    best_sii, best_idx = -np.inf, 0
    win_len = int(0.05 * FS)

    for k, f in enumerate(clean_files):
        dcamf_file = DC_DIR / f'{k+1:06d}.wav'
        if not dcamf_file.exists(): continue
        clean = load_wav(DATA_T1_CLEAN / f.name)
        noisy = load_wav(DATA_T1_NOISY / f.name)
        dcamf = load_wav(dcamf_file)
        L = min(len(clean), len(noisy), len(dcamf))
        clean, noisy, dcamf = clean[:L], noisy[:L], dcamf[:L]
        if L < win_len: continue

        energies = np.array([np.sum(noisy[s:s+win_len]**2) for s in range(0, L-win_len, win_len)])
        start = np.argmax(energies) * win_len
        end = min(start + win_len, L)
        si = compute_sisnr(dcamf[start:end], clean[start:end]) - compute_sisnr(noisy[start:end], clean[start:end])
        if si > best_sii: best_sii, best_idx = si, k

    return best_idx + 1  # 1-indexed

def select_best_linespectra(line_freqs):
    """Select sample maximizing DCAMF-Net advantage at line frequencies."""
    clean_files = sorted(DATA_T1_CLEAN.glob('*.wav'))
    n = len(clean_files)
    scores = np.zeros(n)

    for idx in range(1, n + 1):
        fname = clean_files[idx - 1].name
        clean = load_wav(DATA_T1_CLEAN / fname)
        try:
            crn = load_est(CRN_DIR, idx, 1)
            ct  = load_est(CT_DIR, idx, 2)
            dp  = load_est(DP_DIR, idx, 2)
            dc  = load_est(DC_DIR, idx, 3)
        except FileNotFoundError:
            continue

        L = min(len(clean), len(crn), len(ct), len(dp), len(dc))
        clean = clean[:L]; crn = crn[:L]; ct = ct[:L]; dp = dp[:L]; dc = dc[:L]

        _, pxx_clean = welch(clean, FS, nperseg=2048, noverlap=1024, nfft=4096)
        _, pxx_crn   = welch(crn,   FS, nperseg=2048, noverlap=1024, nfft=4096)
        _, pxx_ct    = welch(ct,    FS, nperseg=2048, noverlap=1024, nfft=4096)
        _, pxx_dp    = welch(dp,    FS, nperseg=2048, noverlap=1024, nfft=4096)
        _, pxx_dc    = welch(dc,    FS, nperseg=2048, noverlap=1024, nfft=4096)

        f = np.fft.rfftfreq(4096, 1/FS)
        score = 0
        for fq in line_freqs:
            i0 = np.argmin(np.abs(f - fq))
            w = slice(max(0, i0-3), min(len(f), i0+4))
            ref = np.mean(10 * np.log10(pxx_clean[w] + 1e-10))
            e_crn = abs(np.mean(10 * np.log10(pxx_crn[w] + 1e-10)) - ref)
            e_ct  = abs(np.mean(10 * np.log10(pxx_ct[w] + 1e-10)) - ref)
            e_dp  = abs(np.mean(10 * np.log10(pxx_dp[w] + 1e-10)) - ref)
            e_dc  = abs(np.mean(10 * np.log10(pxx_dc[w] + 1e-10)) - ref)
            other_best = min(e_crn, e_ct, e_dp)
            score += (other_best - e_dc)
        scores[idx - 1] = score

    return int(np.argmax(scores)) + 1

# ---- Fusion weight parsing ----
def extract_fusion_weights(log_path):
    """Extract the last MaskFusion Weights line from a training log."""
    with open(log_path) as f:
        lines = f.readlines()
    for line in reversed(lines):
        if 'MaskFusion Weights (softmax):' in line:
            import re
            m = re.search(r'\[(.+?)\]', line)
            if m:
                return np.array([float(x) for x in m.group(1).split(',')])
    raise ValueError(f'Fusion weights not found in {log_path}')

# ---- Evaluation on a test set ----
def eval_on_testset(model, test_dir, device='cuda'):
    """Run a PyTorch model on a test set directory, return si_snri, sdri."""
    import torch
    from tqdm import tqdm

    clean_dir = Path(test_dir) / 'clean'
    noisy_dir = Path(test_dir) / 'noisy'
    files = sorted([f.name for f in clean_dir.glob('*.wav')])
    sisnr_list, sdr_list = [], []

    model.eval()
    with torch.no_grad():
        for fname in tqdm(files, desc='Eval'):
            clean = load_wav(clean_dir / fname)
            noisy = load_wav(noisy_dir / fname)
            L = min(len(clean), len(noisy))
            clean, noisy = clean[:L], noisy[:L]

            noisy_t = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)
            est = model(noisy_t).squeeze().cpu().numpy()
            if len(est) > L: est = est[:L]
            elif len(est) < L: est = np.pad(est, (0, L - len(est)))

            sisnr_list.append(compute_sisnr(est, clean) - compute_sisnr(noisy, clean))
            sdr_list.append(compute_sdr(est, clean) - compute_sdr(noisy, clean))

    return {'si_snri': np.mean(sisnr_list), 'sdri': np.mean(sdr_list), 'n': len(files)}
