# HydroDenoise-TF — DCAMF-Net: Underwater Acoustic Denoising with Dual-Branch Convolutional Enhanced Attention and Multi-Layer Mask Fusion

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

An end-to-end **time-domain underwater acoustic denoising neural network** that bridges deep learning with classical signal processing. DCAMF-Net achieves state-of-the-art signal fidelity (SDRi 14.35 dB) on the ShipsEar and DeepShip datasets, outperforming Conv-TasNet, DPRNN, and CRN baselines.

**Keywords**: underwater acoustic denoising, deep learning, time-domain speech enhancement, dual-branch self-attention, multi-layer mask fusion, overcomplete basis decomposition, wavelet shrinkage, Conv-TasNet, DPRNN, ShipsEar, DeepShip, PyTorch

**中文版**: [README.md](./README.md)

---

- **Framework**: PyTorch 2.1 + CUDA 12.1
- **Baselines**: CRN / Conv-TasNet / DPRNN
- **Datasets**: ShipsEar / DeepShip
- **Metrics**: SI-SNRi, SDRi

---

## Environment Setup

### Cloud Training (AutoDL)

Launch an instance: RTX 5090 (32 GB), image `PyTorch 2.1.0 + Python 3.10 (Ubuntu 22.04) + CUDA 12.1`.

Run the following in a JupyterLab terminal:

```bash
# Accelerate GitHub access
source /etc/network_turbo

# Create conda environment
conda create -n dcamf python=3.10 -y
conda init
# Restart terminal, then:
conda activate dcamf

# Install dependencies
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm numpy scipy matplotlib soundfile thop fast-bss-eval tensorboard h5py asteroid requests torchinfo

# Clone repository
cd /root/autodl-tmp
git clone https://github.com/LuKaderZ/HydroDenoise-TF.git
cd HydroDenoise-TF
```

### Local Environment (Inference & Plotting)

Requires Python for figure generation. Install the same Python dependencies (CUDA-enabled PyTorch not required locally).

---

## Data Preparation

### 1. Download Datasets

Download **ShipsEar** and **DeepShip** datasets and organize them under `raw_data/` as follows:

```
raw_data/
├── ShipsEar/
│   ├── passenger/      # Passenger vessel radiated noise
│   ├── roro/           # Roll-on/roll-off vessel radiated noise
│   ├── motorboat/      # Motorboat radiated noise
│   ├── wind/           # Wind noise
│   ├── flow/           # Water flow noise
│   └── reservoir/      # Reservoir surface noise
└── DeepShip/
    ├── Cargo/
    ├── Tug/
    ├── Passengership/
    └── Tanker/
```

### 2. Generate Training/Testing Data

```bash
cd scripts
python prepare_data.py
```

The `data/` directory will contain training and test sets with noisy/clean audio pairs.

---

## Training DCAMF-Net

```bash
cd dcamf_net
python train.py --train_dir ../data/ShipsEar/train --lr 5e-4 --save_dir ../experiments/dcamf_net/checkpoints --use_tensorboard
```

The best checkpoint is saved as `experiments/dcamf_net/checkpoints/best_SISNR.pth`.

---

## Baseline Models

### CRN

Adapted from [CRN-causal](https://github.com/JupiterEthan/CRN-causal). The original `.git` directory has been removed; `stft.py` (version compatibility) and `models.py` (`weights_only=False`) have been modified.

```bash
# 1. Run the three data adapter scripts under adapter/
cd baselines/CRN-causal/scripts

# 2. Training
python train.py --gpu_ids=0 --tr_list=../filelists/tr_list.txt --cv_file=../data/datasets/cv/cv.ex --ckpt_dir=exp --logging_period=5 --lr=0.0002 --time_log=./time.log --unit=utt --batch_size=16 --buffer_size=32 --max_n_epochs=150

# 3. Inference
python test.py --gpu_ids=0 --ckpt_dir=exp --model_file=exp/models/best.pt --tt_list=../filelists/tt_list.txt --est_path=../data/estimates
```

### Conv-TasNet / DPRNN

Implemented using `asteroid`'s `ConvTasNet` and `DPRNNTasNet`. Run `train.py` directly under the corresponding directory; hyperparameters are pre-configured in the scripts.

---

## Ablation Study

Three ablation variants remove the global branch, local branch, and convolutional enhancement submodule, respectively:

```bash
python train_ablation.py --model model_ablation1 --train_dir ../data/ShipsEar/train --save_dir ../experiments/ablation/ablation1/checkpoints
python train_ablation.py --model model_ablation2 --train_dir ../data/ShipsEar/train --save_dir ../experiments/ablation/ablation2/checkpoints
python train_ablation.py --model model_ablation3 --train_dir ../data/ShipsEar/train --save_dir ../experiments/ablation/ablation3/checkpoints
```

Batch testing after training:

```bash
python test_ablation_batch.py
```

---

## Multi-Layer Mask Fusion Weight Validation

Train models under different SNR distributions to compare convergence behavior of fusion weights across layers:

```bash
python prepare_data_high.py   # High SNR group ([-5, 0] dB)
python prepare_data_low.py    # Low SNR group ([-15, -10] dB)
```

After training, rename `train.log` from each run and place them under:

```
experiments/mask_fusion_weights/
├── train_avg.log     # Average SNR group (equal-probability mixture)
├── train_low.log     # Low SNR group
└── train_high.log    # High SNR group
```

---

## Inference & Plotting

### Python Inference

Download the best checkpoint from the server to the local project directory, then:

```bash
cd dcamf_net
python test.py
```

### Python Figures

Run the plotting scripts under `scripts/`. All figures are auto-saved to `figures/`.

| Figure | Script / Source | Content |
|--------|-----------------|---------|
| Fig 2.1 | `plot_fig2_1_ship_psd.py` | Ship radiated noise PSD |
| Fig 2.2 | `plot_fig2_2_noise_timefreq.py` | Noise type time/freq/time-freq analysis |
| Fig 2.3 | `plot_fig2_3_noisy_psd.py` | -15dB noisy mixture PSD comparison |
| Fig 3.1–3.3 | `drawio/fig3_*_*.drawio` | Network architecture diagrams (draw.io) |
| Fig 4.1 | `drawio/fig4_1_data_pipeline.drawio` | Dataset construction pipeline (draw.io) |
| Fig 4.2 | `plot_fig4_2_training_curves.py` | Training loss & SI-SNR curves |
| Fig 4.3 | `plot_fig4_3_sisnr_boxplot.py` | Per-sample SI-SNRi & SDRi box plot |
| Fig 4.4 | `plot_fig4_4_time_waveform.py` | Time-domain waveform comparison |
| Fig 4.5 | `plot_fig4_5_overall_spectrum.py` | Overall spectral comparison |
| Fig 4.6 | `plot_fig4_6_spectrogram.py` | Spectrogram comparison |
| Fig 4.7 | `plot_fig4_7_line_spectra.py` | Key line spectrum power recovery |
| Fig 4.8 | `plot_fig4_8_generalization.py` | Generalization performance |
| Fig 4.9 | `plot_fig4_9_fusion_weights.py` | Fusion weight distribution |
| Fig 4.10 | `plot_fig4_10_noise_estimation.py` | Noise estimation verification |

For IEEE journal paper figures (English, color):

```bash
python scripts/plot_ieee_figures.py
```

Outputs are saved to `figures/` and `figures/ieee/` respectively.

---

## Project Structure

```
HydroDenoise-TF/
├── dcamf_net/                  # DCAMF-Net model code
│   ├── model.py                #   Network architecture
│   ├── train.py                #   Training script
│   ├── test.py                 #   Inference & evaluation
│   ├── dataset.py              #   Data loader
│   ├── loss.py                 #   r-nSISNR loss function
│   ├── config.py               #   Evaluation utilities
│   ├── model_ablation[1-3].py  #   Ablation variants
│   ├── train_ablation.py       #   Ablation training
│   └── test_ablation_batch.py  #   Ablation testing
├── baselines/                  # Baseline models
│   ├── CRN-causal/
│   ├── conv_tasnet/
│   └── dprnn/
├── scripts/                    # Data preparation & figure generation
│   ├── prepare_data.py
│   ├── prepare_data_high.py
│   ├── prepare_data_low.py
│   ├── plot_utils.py
│   └── plot_fig*_*.py
├── drawio/                     # Architecture & pipeline diagrams
│   ├── fig3_1_architecture.drawio
│   ├── fig3_2_dcam_block.drawio
│   ├── fig3_3_cemhsa.drawio
│   └── fig4_1_data_pipeline.drawio
├── figures/                    # Generated figures
│   └── ieee/                   #   IEEE journal figures
├── experiments/                # Checkpoints, logs, denoised audio
├── raw_data/                   # Raw datasets
├── data/                       # Processed training/test data
└── experiments_log.md          # Quantitative results record
```
