"""图5.1: 编解码器滤波器组分析 (2×2)
用法: python scripts/plot_fig5_1_encoder_analysis.py
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_chapter5_utils import (
    get_filter_info, get_model, to_t, make_chirp,
    open_log, log, sublabel, save_fig, SR, FREQS, N_FFT,
)


def main():
    f = open_log()
    model, _, _ = get_model()
    cent_enc, ch_order, cent_dec, peak_enc = get_filter_info()

    # ---- 编码器权重FFT ----
    W_enc = model.encoder.conv.weight.detach().cpu().squeeze(1).numpy()  # (256, 80)
    H_enc = np.abs(np.fft.rfft(W_enc, n=N_FFT, axis=1))
    H_enc_db = 20 * np.log10(H_enc + 1e-8)

    # ---- 解码器权重FFT ----
    W_dec = model.decoder.deconv.weight.detach().cpu().squeeze(1).numpy()  # (256, 80)
    H_dec = np.abs(np.fft.rfft(W_dec, n=N_FFT, axis=1))
    H_dec_db = 20 * np.log10(H_dec + 1e-8)

    # ---- Chirp响应 ----
    chirp_sig = make_chirp()
    x_t = to_t(chirp_sig)
    with torch.no_grad():
        _, W_e, T_enc = model.encoder(x_t)
        W_e_np = W_e[0].cpu().numpy()
    W_chirp = W_e_np[ch_order, :]
    E = np.abs(W_chirp)
    E_norm = (E - E.mean(axis=1, keepdims=True)) / (E.std(axis=1, keepdims=True) + 1e-8)

    # ---- 画图 ----
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5))

    # (a) 编码器卷积核幅频响应
    ax = axes[0, 0]
    sublabel(ax, '(a)')
    im = ax.pcolormesh(FREQS, np.arange(256), H_enc_db[ch_order, :],
                       shading='auto', cmap='inferno', vmin=-40, vmax=5, rasterized=True)
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('通道 (低频→高频)')
    ax.set_xlim(0, 8000)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('幅度 (dB)', fontsize=8)

    # (b) 编码器 Chirp 实际激活
    ax = axes[0, 1]
    sublabel(ax, '(b)')
    t_ms = np.arange(T_enc) * 40 / SR * 1000  # stride=40 → 时间轴
    im2 = ax.pcolormesh(t_ms, np.arange(256), E_norm,
                        shading='auto', cmap='inferno', vmin=-1, vmax=4, rasterized=True)
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('通道 (低频→高频)')
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('z-score |W_e|', fontsize=8)

    # (c) 解码器合成滤波器幅频响应
    ax = axes[1, 0]
    sublabel(ax, '(c)')
    im3 = ax.pcolormesh(FREQS, np.arange(256), H_dec_db[ch_order, :],
                        shading='auto', cmap='inferno', vmin=-60, vmax=10, rasterized=True)
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('通道 (低频→高频)')
    ax.set_xlim(0, 8000)
    cbar3 = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cbar3.set_label('幅度 (dB)', fontsize=8)

    # (d) 编码器 vs 解码器质心频率
    ax = axes[1, 1]
    sublabel(ax, '(d)')
    ax.scatter(cent_enc[ch_order], cent_dec[ch_order], s=4, alpha=0.4,
               color='#4A90D9', edgecolors='none')
    ax.plot([0, 5000], [0, 5000], '--', color='gray', lw=0.8, label='y=x')
    ax.set_xlabel('编码器质心频率 (Hz)')
    ax.set_ylabel('解码器质心频率 (Hz)')
    ax.set_xlim(0, 5000); ax.set_ylim(0, 5000)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # ---- 数据 ----
    r_ed, _ = pearsonr(cent_enc, cent_dec)
    n_low_enc = np.sum(peak_enc < 1000)
    n_low_dec = np.sum(cent_dec < 1000)
    log(f, "===== 图5.1: 编解码器滤波器组 =====")
    log(f, f"编码器: 峰值频率 {peak_enc.min():.0f}~{peak_enc.max():.0f} Hz, "
        f"<1kHz通道 {n_low_enc}/256 ({n_low_enc/256*100:.0f}%)")
    log(f, f"解码器: 质心频率 {cent_dec.min():.0f}~{cent_dec.max():.0f} Hz, "
        f"<1kHz通道 {n_low_dec}/256 ({n_low_dec/256*100:.0f}%)")
    log(f, f"编解码器质心频率 Pearson r = {r_ed:.4f}")

    save_fig(fig, 'fig5-1_encoder_decoder_filterbank')
    f.close()


if __name__ == '__main__':
    main()
