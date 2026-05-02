"""IEEE Journal Figures (English, color) — Fig 2-5 in one script."""
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import *

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

ieee_dir = FIG_DIR / 'ieee'
ieee_dir.mkdir(exist_ok=True)

BAR_COLORS = [COLORS['CRN'], COLORS['ConvTasNet'], COLORS['DPRNN'], COLORS['DCAMF']]
MODEL_NAMES = ['CRN', 'Conv-TasNet', 'DPRNN', 'DCAMF-Net']

# ====== Shared: detect line spectra + load sample ======
line_freqs = find_line_spectra(n_peaks=5)
print(f'Line freqs: {np.round(line_freqs).astype(int).tolist()} Hz')

# Sample for Fig 2 (transient SI-SNRi max)
idx_psd = select_best_transient()
# Sample for Fig 3 (linespectra advantage max)
idx_line = select_best_linespectra(line_freqs)
print(f'Fig 2 sample: {idx_psd}, Fig 3 sample: {idx_line}')

def load_sample(idx):
    cfiles = sorted(DATA_T1_CLEAN.glob('*.wav'))
    fn = cfiles[idx - 1].name
    clean = load_wav(DATA_T1_CLEAN / fn)
    noisy = load_wav(DATA_T1_NOISY / fn)
    crn = load_est(CRN_DIR, idx, 1)
    ct  = load_est(CT_DIR, idx, 2)
    dp  = load_est(DP_DIR, idx, 2)
    dc  = load_est(DC_DIR, idx, 3)
    L = min(len(clean), len(noisy), len(crn), len(ct), len(dp), len(dc))
    return clean[:L], noisy[:L], crn[:L], ct[:L], dp[:L], dc[:L]

# ====== FIG 2: PSD comparison ======
print('Fig 2: PSD Comparison')
clean, noisy, crn, ct, dp, dc = load_sample(idx_psd)
signals_psd = [crn, ct, dp, dc]
f, p_clean = psd_db(clean); _, p_noisy = psd_db(noisy)
psds = [psd_db(s)[1] for s in signals_psd]

mask = (f >= 0) & (f <= 4000); fk = f[mask] / 1000
all_psd = np.concatenate([p_clean[mask], p_noisy[mask]] + [p[mask] for p in psds])
ymin, ymax = np.floor(all_psd.min()/10)*10, np.ceil(all_psd.max()/10)*10

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, name, pxx, color in zip(axes.flat, MODEL_NAMES, psds, BAR_COLORS):
    ax.plot(fk, p_noisy[mask], color=COLORS['noisy'], lw=0.5, alpha=0.6)
    ax.plot(fk, p_clean[mask], 'k-', lw=1.5)
    ax.plot(fk, pxx[mask], '--', color=color, lw=1.2)
    for fq in line_freqs: ax.axvline(fq/1000, color='gray', ls=':', lw=0.4)
    ax.set(xlabel='Frequency (kHz)', ylabel='PSD (dB/Hz)', title=name, xlim=(0,4), ylim=(ymin, ymax))
    ax.legend(['Noisy', 'Clean', 'Enhanced'], fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(ieee_dir / 'Fig2_PSD_Comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(ieee_dir / 'Fig2_PSD_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ====== FIG 3: Line spectrum bars ======
print('Fig 3: Line Spectra')
clean, noisy, crn, ct, dp, dc = load_sample(idx_line)
models_sig = {'CRN': crn, 'Conv-TasNet': ct, 'DPRNN': dp, 'DCAMF-Net': dc}
f = np.fft.rfftfreq(4096, 1/FS)
dev = np.zeros((len(line_freqs), 4))
for i, fq in enumerate(line_freqs):
    i0 = np.argmin(np.abs(f - fq)); w = slice(max(0,i0-4), min(len(f), i0+5))
    _, p_cl = welch(clean, FS, nperseg=2048, noverlap=1024, nfft=4096)
    ref = np.mean(10*np.log10(p_cl[w]+1e-10))
    for j, (name, sig) in enumerate(models_sig.items()):
        _, pxx = welch(sig, FS, nperseg=2048, noverlap=1024, nfft=4096)
        dev[i, j] = np.mean(10*np.log10(pxx[w]+1e-10)) - ref

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(line_freqs)); w = 0.2
for i in range(4):
    ax.bar(x + (i-1.5)*w, dev[:, i], w, color=BAR_COLORS[i], edgecolor='k', lw=0.5)
    for j, v in enumerate(dev[:, i]):
        if abs(v) > 0.5:
            ax.text(x[j]+(i-1.5)*w, v, f'{v:.1f}', ha='center', va='bottom' if v>=0 else 'top', fontsize=7)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f'{int(round(fq))} Hz' for fq in line_freqs])
ax.set(xlabel='Key Line Frequency', ylabel='Power Deviation from Clean (dB)',
       title='Line Spectrum Power Recovery')
ax.legend(MODEL_NAMES, fontsize=8); ax.grid(axis='y', alpha=0.3)
dv = dev.ravel(); ax.set_ylim(min(dv.min()*1.15, -2), max(dv.max()*1.15, 2))
plt.tight_layout()
fig.savefig(ieee_dir / 'Fig3_Line_Spectra.pdf', dpi=300, bbox_inches='tight')
fig.savefig(ieee_dir / 'Fig3_Line_Spectra.png', dpi=300, bbox_inches='tight')
plt.close()

# ====== FIG 4: Generalization ======
print('Fig 4: Generalization')
target_snrs = [-15, -10, -5]
test_sets = [(DATA_T1_CLEAN, DATA_T1_NOISY, DC_DIR),
             (DATA_T2_CLEAN, DATA_T2_NOISY, DC_DIR_T2),
             (DATA_T3_CLEAN, DATA_T3_NOISY, DC_DIR_T3)]
test_names = ['Test-1 (Known + Known)', 'Test-2 (Unseen Vessel)', 'Test-3 (Unseen Noise)']
test_colors = [COLORS['ConvTasNet'], COLORS['CRN'], '#7E2F8E']

all_r = []
for s, (cd, nd, dcd) in enumerate(test_sets):
    for f in sorted(cd.glob('*.wav')):
        dp = dcd / f.name
        if not dp.exists(): continue
        clean = load_wav(f); noisy = load_wav(nd / f.name); denoised = load_wav(dp)
        L = min(len(clean), len(noisy), len(denoised))
        clean, noisy, denoised = clean[:L], noisy[:L], denoised[:L]
        snr = 10*np.log10(np.mean(clean**2)/(np.mean((noisy-clean)**2)+1e-10))
        snr_i = np.argmin(np.abs(np.array(target_snrs)-snr))
        all_r.append({'set': s, 'snr': target_snrs[snr_i],
                       'sii': compute_sisnr(denoised,clean)-compute_sisnr(noisy,clean),
                       'sdi': compute_sdr(denoised,clean)-compute_sdr(noisy,clean)})

sisnri = np.full((3,3), np.nan); sdri = np.full((3,3), np.nan)
for s in range(3):
    for t, snr in enumerate(target_snrs):
        vals = [r for r in all_r if r['set']==s and r['snr']==snr]
        if vals: sisnri[s,t]=np.mean([r['sii'] for r in vals]); sdri[s,t]=np.mean([r['sdi'] for r in vals])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, data, yl, ti in [(axes[0], sisnri, 'SI-SNRi (dB)', '(a) SI-SNR Improvement'),
                           (axes[1], sdri, 'SDRi (dB)', '(b) SDR Improvement')]:
    x = np.arange(3); w = 0.25
    for i in range(3):
        ax.bar(x+i*w, data[i], w, color=test_colors[i], edgecolor='k', lw=0.5, label=test_names[i])
        for j in range(3):
            if not np.isnan(data[i,j]): ax.text(x[j]+i*w, data[i,j], f'{data[i,j]:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x+w); ax.set_xticklabels([f'{s} dB' for s in target_snrs])
    ax.set(xlabel='Input SNR', ylabel=yl, title=ti); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    d = data[~np.isnan(data)]
    if len(d): ax.set_ylim(min(0, np.floor(d.min())-1), np.ceil(d.max())+1)
plt.tight_layout()
fig.savefig(ieee_dir / 'Fig4_Generalization.pdf', dpi=300, bbox_inches='tight')
fig.savefig(ieee_dir / 'Fig4_Generalization.png', dpi=300, bbox_inches='tight')
plt.close()

# ====== FIG 5: Fusion weights ======
print('Fig 5: Fusion Weights')
groups = {'Average SNR': 'train_avg.log', 'Low SNR': 'train_low.log', 'High SNR': 'train_high.log'}
wt_colors = [COLORS['ConvTasNet'], '#77AC30', COLORS['DCAMF']]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(1, 11); w = 0.25
for i, (name, fn) in enumerate(groups.items()):
    p = FUSION_LOG_DIR / fn
    if not p.exists(): continue
    wt = extract_fusion_weights(p)
    ax.bar(x+i*w, wt, w, color=wt_colors[i], edgecolor='k', lw=0.5, label=name)
    for j, v in enumerate(wt):
        if v > 0.01: ax.text(x[j]+i*w, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x+w); ax.set_xticklabels([str(i) for i in range(1,11)])
ax.set(xlabel='DCAM Block Layer', ylabel='Fusion Weight (softmax)',
       title='Multi-Layer Mask Fusion Weights')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max([extract_fusion_weights(FUSION_LOG_DIR/f)
                     for f in groups.values() if (FUSION_LOG_DIR/f).exists()])*1.2)
plt.tight_layout()
fig.savefig(ieee_dir / 'Fig5_Fusion_Weights.pdf', dpi=300, bbox_inches='tight')
fig.savefig(ieee_dir / 'Fig5_Fusion_Weights.png', dpi=300, bbox_inches='tight')
plt.close()

print('All IEEE figures saved to', ieee_dir)
