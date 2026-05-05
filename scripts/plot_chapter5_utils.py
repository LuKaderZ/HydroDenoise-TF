"""
第五章可解释性分析 — 公共工具模块
==============================
模型加载、信号生成、前向传播、定量指标、绘图辅助。
"""
import numpy as np
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.signal import welch, chirp as scipy_chirp
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'dcamf_net'))

from dcamf_net.model import DCAMFNet, overlap_add
from plot_utils import setup_style, FIG_DIR, COLORS

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 样式
# ==============================================================================
setup_style()
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Noto Sans SC', 'DejaVu Sans'],
    'font.size': 9,
    'axes.unicode_minus': False,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
})

# ==============================================================================
# 常量
# ==============================================================================
SR = 16000
DUR = 3.0
L = 48000
N_FFT = 1024
FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)

AF = 0.45
WF = 0.30
TP = 0.93
BM = 0.10
LM = 0.12
RM = 0.96

# ==============================================================================
# 数据输出
# ==============================================================================
DATA_PATH = PROJECT / 'tmp' / 'chapter5_data.txt'
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

def open_log():
    """每次运行追加到数据文件"""
    f = open(DATA_PATH, 'a', encoding='utf-8')
    f.write(f"\n{'='*60}\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    return f

def log(f, msg):
    print(msg)
    if f is not None:
        f.write(msg + '\n')

# ==============================================================================
# 模型加载 (单例)
# ==============================================================================
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model = None
_fw = None

def get_model():
    global _model, _fw
    if _model is None:
        _model = DCAMFNet(n_blocks=10).to(_device)
        ckpt_path = PROJECT / 'experiments' / 'dcamf_net' / 'checkpoints' / 'best_SISNR.pth'
        ckpt = torch.load(str(ckpt_path), map_location=_device, weights_only=False)
        sd = {k: v for k, v in ckpt.items()
              if 'total_ops' not in k and 'total_params' not in k
              and k not in ('total_ops', 'total_params')}
        _model.load_state_dict(sd, strict=False)
        _model.eval()
        _fw = F.softmax(_model.mask_fusion_weights, dim=0).detach().cpu().numpy()
    return _model, _fw, _device

# ==============================================================================
# 编码器/解码器滤波器分析 (延迟计算)
# ==============================================================================
_centroid_enc = None
_ch_order = None
_centroid_dec = None
_peak_enc = None

def get_filter_info():
    global _centroid_enc, _ch_order, _centroid_dec, _peak_enc
    if _centroid_enc is None:
        model, _, _ = get_model()
        W_enc = model.encoder.conv.weight.detach().cpu().squeeze(1).numpy()  # (256, 80)
        W_dec = model.decoder.deconv.weight.detach().cpu().squeeze(1).numpy()  # (256, 80)

        H_enc = np.abs(np.fft.rfft(W_enc, n=N_FFT, axis=1))
        H_dec = np.abs(np.fft.rfft(W_dec, n=N_FFT, axis=1))

        _peak_enc = FREQS[np.argmax(H_enc, axis=1)]
        _centroid_enc = np.sum(FREQS[None, :] * H_enc, axis=1) / (np.sum(H_enc, axis=1) + 1e-8)
        _ch_order = np.argsort(_peak_enc)  # 与旧代码一致: 按峰值频率排序

        _centroid_dec = np.sum(FREQS[None, :] * H_dec, axis=1) / (np.sum(H_dec, axis=1) + 1e-8)

    return _centroid_enc, _ch_order, _centroid_dec, _peak_enc

# 快速访问
def ch_x(): return np.arange(256)
def xticks(): return np.arange(0, 256, 32)
def xlabels():
    _, co, _, peak = get_filter_info()
    return [f'{peak[co[p]]:.0f}' for p in xticks()]

def ch500_pos():
    _, co, _, peak = get_filter_info()
    ch500 = np.argmin(np.abs(peak - 500))
    return np.where(co == ch500)[0][0]

# ==============================================================================
# 前向传播
# ==============================================================================
def to_t(x):
    return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(_device)

def ola(x_chunk, T_enc):
    model, _, _ = get_model()
    return overlap_add(x_chunk, model.chunk_size, model.hop_size, T_enc)

def run_full(x_np):
    """完整前向"""
    model, fw, _ = get_model()
    x_t = to_t(x_np)
    with torch.no_grad():
        W_e_raw = model.encoder.conv(x_t)[0].cpu().numpy()  # (256, T') 原始Conv1d
        W_cur, W_e, T_enc = model.encoder(x_t)

        masks_t = []
        for blk in model.dcam_blocks:
            W_cur, mc = blk(W_cur)
            masks_t.append(ola(mc, T_enc)[0].cpu().numpy())

        ms = torch.stack([torch.from_numpy(m) for m in masks_t], dim=0).unsqueeze(0).to(_device)
        wt = F.softmax(model.mask_fusion_weights, dim=0)
        fm = torch.einsum('b n f t, n -> b f t', ms, wt)[0].detach().cpu().numpy()

        nest = model.decoder(torch.from_numpy(fm).unsqueeze(0).to(_device) * W_e)
        nest = nest[0, 0, :L].detach().cpu().numpy()
        s_hat = x_np - nest

    return {
        'W_e_raw': W_e_raw, 'W_e': W_e[0].cpu().numpy(), 'T_enc': T_enc,
        'masks': masks_t, 'final_mask': fm, 'nest': nest, 's_hat': s_hat,
    }

def run_attention(x_np, block_idx=0):
    """提取注意力矩阵"""
    model, _, _ = get_model()
    x_t = to_t(x_np)
    with torch.no_grad():
        W_cur, W_e, T_enc = model.encoder(x_t)
        for i in range(block_idx):
            W_cur, _ = model.dcam_blocks[i](W_cur)
        blk = model.dcam_blocks[block_idx]
        W_cur, mask, attn_dict = blk(W_cur, return_attention=True)

    al = attn_dict['local'].detach().cpu().numpy()
    ag = attn_dict['global'].detach().cpu().numpy()
    a_local = al[0]
    a_global = ag[0]
    if a_local.ndim == 3:
        a_local = a_local[0]
    if a_global.ndim == 3:
        a_global = a_global[0]

    return {'attn_local': a_local, 'attn_global': a_global,
            'mask': ola(mask, T_enc)[0].cpu().numpy(), 'T_enc': T_enc}

# ==============================================================================
# 信号生成
# ==============================================================================
rng = np.random.RandomState(42)

def _fade(y, fade_len=None):
    if fade_len is None:
        fade_len = int(0.02 * SR)
    y = y.copy()
    y[:fade_len] *= np.linspace(0, 1, fade_len)
    y[-fade_len:] *= np.linspace(1, 0, fade_len)
    return y

def make_tone(f, amp=0.5):
    t = np.arange(L) / SR
    return _fade(amp * np.sin(2 * np.pi * f * t)).astype(np.float32)

def make_white():
    y = rng.randn(L).astype(np.float32)
    y /= np.max(np.abs(y)); y *= 0.5
    return y

def make_chirp(f0=100, f1=8000, t_chirp=None):
    """生成扫频信号，默认填满3秒。前后各留0.1秒静音做淡入淡出"""
    if t_chirp is None:
        t_chirp = DUR
    L_chirp = int(t_chirp * SR)
    # 生成稍长一点的chirp以便裁切
    t = np.arange(L) / SR
    y_c = 0.5 * scipy_chirp(t, f0=f0, f1=f1, t1=DUR, method='linear')
    # 前0.05s和后0.05s做淡入淡出
    y_c = _fade(y_c, fade_len=int(0.05 * SR))
    return y_c[:L].astype(np.float32)

def make_pulse_train(pulse_interval=0.2, pulse_width=0.005, amp=0.5):
    t = np.arange(L) / SR
    y = np.zeros(L, dtype=np.float32)
    n_pulses = int(DUR / pulse_interval)
    for i in range(n_pulses):
        y[int(i * pulse_interval * SR)] = amp
    sigma = int(pulse_width * SR)
    kt = np.arange(-4 * sigma, 4 * sigma + 1)
    kernel = np.exp(-0.5 * (kt / sigma)**2)
    kernel /= kernel.max()
    y = np.convolve(y, kernel, mode='same')
    y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1
    return _fade(y * amp).astype(np.float32)

def mix_snr(clean, noise, snr_db):
    sp = np.mean(clean**2)
    npow = np.mean(noise**2)
    scale = np.sqrt(sp / (npow * 10**(snr_db / 10) + 1e-8))
    mixed = clean + scale * noise
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed *= 0.95 / peak
    return mixed.astype(np.float32), scale

def get_tones():
    return {f: make_tone(f) for f in [200, 500, 1000, 2000, 4000]}

def get_mixes():
    tones = get_tones()
    mixes = {}
    for snr in [-5, -10]:
        wn = rng.randn(L).astype(np.float32); wn /= np.max(np.abs(wn)); wn *= 0.5
        mx, sc = mix_snr(tones[500], wn, snr)
        clean_scaled = tones[500]
        if np.max(np.abs(mx)) > 0.5:
            clean_scaled = tones[500] * 0.5 / np.max(np.abs(mx))
        mixes[snr] = {'mixed': mx, 'clean': clean_scaled, 'noise': mx - clean_scaled}
    return mixes

# ==============================================================================
# 定量指标
# ==============================================================================
def attention_distance(attn_mat):
    N = attn_mat.shape[0]
    i_idx, j_idx = np.arange(N)[:, None], np.arange(N)[None, :]
    return float(np.sum(attn_mat * np.abs(i_idx - j_idx)) / np.sum(attn_mat))

def compute_tv(mask_2d):
    tv_f = float(np.sum(np.abs(np.diff(mask_2d, axis=0))))
    tv_t = float(np.sum(np.abs(np.diff(mask_2d, axis=1))))
    return tv_f, tv_t

def compute_tpr_nsr(signal, ref_clean, ref_noisy, f0=500, delta_f=15):
    f_w, p_sig = welch(signal, SR, nperseg=1024)
    _, p_clean = welch(ref_clean, SR, nperseg=1024)
    _, p_noisy = welch(ref_noisy, SR, nperseg=1024)
    mask_target = (f_w >= f0 - delta_f) & (f_w <= f0 + delta_f)
    mask_nontarget = (f_w >= 50) & ~mask_target
    tpr = np.sum(p_sig[mask_target]) / (np.sum(p_clean[mask_target]) + 1e-8)
    nsr = 10 * np.log10(np.sum(p_noisy[mask_nontarget]) / (np.sum(p_sig[mask_nontarget]) + 1e-8) + 1e-8)
    peak_freq = f_w[mask_target][np.argmax(p_sig[mask_target])]
    return float(tpr), float(nsr), float(abs(peak_freq - f0))

def compute_sisnr(est, ref):
    est = est.reshape(-1).astype(np.float64); ref = ref.reshape(-1).astype(np.float64)
    est -= est.mean(); ref -= ref.mean()
    scale = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    target = scale * ref
    return float(10 * np.log10(np.sum(target**2) / (np.sum(est - target)**2 + 1e-8) + 1e-8))

# ==============================================================================
# 绘图辅助
# ==============================================================================
def sublabel(ax, text):
    ax.set_title(text, loc='center', fontsize=10, fontweight='bold', pad=3)

def save_fig(fig, name, wspace=None):
    """保存 PDF + PNG，统一间距。wspace可覆盖默认子图水平间距"""
    ws = wspace if wspace is not None else WF
    fig.subplots_adjust(hspace=AF, wspace=ws, top=TP, bottom=BM, left=LM, right=RM)
    fig.savefig(FIG_DIR / f'{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {FIG_DIR / name}.pdf/png")
