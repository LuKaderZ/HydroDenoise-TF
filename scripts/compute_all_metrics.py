"""Unified evaluation: all models on all test sets, consistent output format."""
import numpy as np
from pathlib import Path
from plot_utils import *

PROJECT_ROOT = Path(r'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF')
target_snrs = [-15, -10, -5]

MODELS = {
    'CRN': {
        'ShipsEar': (CRN_DIR, DATA_T1_CLEAN, DATA_T1_NOISY, 1),
        'DeepShip': (PROJECT_ROOT/'baselines/CRN-causal/data/data/datasets/tt/tt_DeepShip',
                      PROJECT_ROOT/'data/DeepShip/test/clean',
                      PROJECT_ROOT/'data/DeepShip/test/noisy', 1),
    },
    'Conv-TasNet': {
        'ShipsEar': (PROJECT_ROOT/'experiments/conv_tasnet/estimates/tt_test1',
                      DATA_T1_CLEAN, DATA_T1_NOISY, 2),
        'DeepShip': (PROJECT_ROOT/'experiments/conv_tasnet/estimates/tt_DeepShip',
                      PROJECT_ROOT/'data/DeepShip/test/clean',
                      PROJECT_ROOT/'data/DeepShip/test/noisy', 2),
    },
    'DPRNN': {
        'ShipsEar': (PROJECT_ROOT/'experiments/dprnn/estimates/tt_test1',
                      DATA_T1_CLEAN, DATA_T1_NOISY, 2),
        'DeepShip': (PROJECT_ROOT/'experiments/dprnn/estimates/tt_DeepShip',
                      PROJECT_ROOT/'data/DeepShip/test/clean',
                      PROJECT_ROOT/'data/DeepShip/test/noisy', 2),
    },
    'DCAMF-Net': {
        'ShipsEar': (DC_DIR, DATA_T1_CLEAN, DATA_T1_NOISY, 3),
        'DeepShip': (PROJECT_ROOT/'experiments/dcamf_net/denoised/DeepShip_test',
                      PROJECT_ROOT/'data/DeepShip/test/clean',
                      PROJECT_ROOT/'data/DeepShip/test/noisy', 3),
    },
}

ABLATION = {
    'Full DCAMF-Net': PROJECT_ROOT/'experiments/dcamf_net/denoised/ShipsEar_test1',
    'No Global Branch': PROJECT_ROOT/'experiments/ablation/ablation1/denoised',
    'No Local Branch': PROJECT_ROOT/'experiments/ablation/ablation2/denoised',
    'No Conv Enhance': PROJECT_ROOT/'experiments/ablation/ablation3/denoised',
}

print('=' * 60)
print('  UNIFIED EVALUATION REPORT')
print('=' * 60)

# ---- Main models ----
for model_name, datasets in MODELS.items():
    print(f'\n--- {model_name} ---')
    for ds_name, (est_dir, clean_dir, noisy_dir, est_type) in datasets.items():
        if not Path(est_dir).exists():
            print(f'  {ds_name}: SKIP (estimates not found)')
            continue
        cfiles = sorted(Path(clean_dir).glob('*.wav'))
        all_sii, all_sdi = [], []

        for idx, cf in enumerate(cfiles):
            try:
                enhanced = load_est(est_dir, idx + 1, est_type)
            except FileNotFoundError:
                continue
            clean = load_wav(cf)
            noisy = load_wav(Path(noisy_dir) / cf.name)
            L = min(len(clean), len(noisy), len(enhanced))
            clean, noisy, enhanced = clean[:L], noisy[:L], enhanced[:L]

            all_sii.append(compute_sisnr(enhanced, clean) - compute_sisnr(noisy, clean))
            all_sdi.append(compute_sdr(enhanced, clean) - compute_sdr(noisy, clean))

        if all_sii:
            sii, sdi = np.mean(all_sii), np.mean(all_sdi)
            print(f'  {ds_name:12s}: SI-SNRi = {sii:7.2f} dB, SDRi = {sdi:7.2f} dB  (n={len(all_sii)})')
    print(f'  >> Copy: SI-SNRi={sii:.2f} / SDRi={sdi:.2f}')

# ---- Ablation ----
print('\n--- Ablation (ShipsEar test1) ---')
for name, est_dir in ABLATION.items():
    if not Path(est_dir).exists():
        print(f'  {name}: SKIP')
        continue
    cfiles = sorted(DATA_T1_CLEAN.glob('*.wav'))
    all_sii, all_sdi = [], []
    for idx, cf in enumerate(cfiles):
        enhanced_file = Path(est_dir) / cf.name
        if not enhanced_file.exists(): continue
        clean = load_wav(cf)
        noisy = load_wav(DATA_T1_NOISY / cf.name)
        enhanced = load_wav(enhanced_file)
        L = min(len(clean), len(noisy), len(enhanced))
        clean, noisy, enhanced = clean[:L], noisy[:L], enhanced[:L]
        all_sii.append(compute_sisnr(enhanced, clean) - compute_sisnr(noisy, clean))
        all_sdi.append(compute_sdr(enhanced, clean) - compute_sdr(noisy, clean))
    if all_sii:
        sii, sdi = np.mean(all_sii), np.mean(all_sdi)
        print(f'  {name:25s}: SI-SNRi = {sii:7.2f} dB, SDRi = {sdi:7.2f} dB  (n={len(all_sii)})')

print('\n' + '=' * 60)
print('  Done. Copy values above to experiments_log.md / Table 4.1 & 4.2')
print('=' * 60)
