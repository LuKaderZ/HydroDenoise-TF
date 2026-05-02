"""Unified evaluation: all models on all test sets, consistent output format."""
import numpy as np
from pathlib import Path
from plot_utils import *

PROJECT_ROOT = Path(r'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF')
target_snrs = [-15, -10, -5]

# (est_dir, clean_dir, noisy_dir, naming_mode)
# naming_mode: 'name' = same filename as clean; 1=CRN; 2=ConvTas/DPRNN
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
        'ShipsEar': (DC_DIR, DATA_T1_CLEAN, DATA_T1_NOISY, 'name'),
        'DeepShip': (PROJECT_ROOT/'experiments/dcamf_net/denoised/DeepShip_test',
                      PROJECT_ROOT/'data/DeepShip/test/clean',
                      PROJECT_ROOT/'data/DeepShip/test/noisy', 'name'),
    },
}

ABLATION = {
    'Full DCAMF-Net': PROJECT_ROOT/'experiments/dcamf_net/denoised/ShipsEar_test1',
    'No Global Branch': PROJECT_ROOT/'experiments/ablation/ablation1/denoised',
    'No Local Branch': PROJECT_ROOT/'experiments/ablation/ablation2/denoised',
    'No Conv Enhance': PROJECT_ROOT/'experiments/ablation/ablation3/denoised',
}

# Store results for final table display
table_4_1 = {}  # {model: {'ShipsEar': (sii, sdi, n), 'DeepShip': (sii, sdi, n)}}
table_4_2 = []  # [(name, sii, sdi, n)]

# ---- Main models (Table 4.1) ----
for model_name, datasets in MODELS.items():
    table_4_1[model_name] = {}
    for ds_name, (est_dir, clean_dir, noisy_dir, naming) in datasets.items():
        if not Path(est_dir).exists():
            print(f'[SKIP] {model_name} {ds_name}: estimates not found')
            continue
        cfiles = sorted(Path(clean_dir).glob('*.wav'))
        all_sii, all_sdi = [], []

        for idx, cf in enumerate(cfiles):
            if naming == 'name':
                enhanced_file = Path(est_dir) / cf.name
                if not enhanced_file.exists():
                    continue
                enhanced = load_wav(enhanced_file)
            else:
                try:
                    enhanced = load_est(est_dir, idx + 1, naming)
                except FileNotFoundError:
                    continue
            clean = load_wav(cf)
            noisy = load_wav(Path(noisy_dir) / cf.name)
            L = min(len(clean), len(noisy), len(enhanced))
            clean, noisy, enhanced = clean[:L], noisy[:L], enhanced[:L]

            all_sii.append(compute_sisnr(enhanced, clean) - compute_sisnr(noisy, clean))
            all_sdi.append(compute_sdr(enhanced, clean) - compute_sdr(noisy, clean))

        if all_sii:
            table_4_1[model_name][ds_name] = (np.mean(all_sii), np.mean(all_sdi), len(all_sii))

# ---- Ablation (Table 4.2) ----
for name, est_dir in ABLATION.items():
    if not Path(est_dir).exists():
        print(f'[SKIP] Ablation {name}: estimates not found')
        continue
    cfiles = sorted(DATA_T1_CLEAN.glob('*.wav'))
    all_sii, all_sdi = [], []
    for cf in cfiles:
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
        table_4_2.append((name, np.mean(all_sii), np.mean(all_sdi), len(all_sii)))

# ---- Print Table 4.1 ----
print('\n' + '=' * 70)
print('  表4.1  不同模型在两个数据集上的性能对比')
print('=' * 70)
print(f'  {"模型":12s}  {"ShipsEar (SI-SNRi / SDRi)":28s}  {"DeepShip (SI-SNRi / SDRi)":28s}')
print(f'  {"-"*12}  {"-"*28}  {"-"*28}')
for model_name in ['CRN', 'Conv-TasNet', 'DPRNN', 'DCAMF-Net']:
    ship = table_4_1.get(model_name, {}).get('ShipsEar')
    deep = table_4_1.get(model_name, {}).get('DeepShip')
    s_ship = f'{ship[0]:.2f} / {ship[1]:.2f}' if ship else 'N/A'
    s_deep = f'{deep[0]:.2f} / {deep[1]:.2f}' if deep else 'N/A'
    print(f'  {model_name:12s}  {s_ship:28s}  {s_deep:28s}')
print('=' * 70)

# ---- Print Table 4.2 ----
print('\n' + '=' * 55)
print('  表4.2  消融实验结果')
print('=' * 55)
print(f'  {"模型变体":20s}  {"SI-SNRi (dB)":14s}  {"SDRi (dB)":14s}')
print(f'  {"-"*20}  {"-"*14}  {"-"*14}')
for name, sii, sdi, n in table_4_2:
    print(f'  {name:20s}  {sii:14.2f}  {sdi:14.2f}')
print('=' * 55)
print()
