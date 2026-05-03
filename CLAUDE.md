# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DCAMF-Net — 基于双分支卷积增强注意力与多层掩码融合的端到端时域水声降噪网络。PyTorch 2.1 + CUDA 12.1，在 ShipsEar/DeepShip 数据集上评测，指标为 SI-SNRi / SDRi。

## 常用命令

所有命令从项目根目录运行。

### 训练

```bash
# DCAMF-Net 主模型训练（150 epochs，batch_size=4）
python dcamf_net/train.py --data_dir data/ShipsEar/train --epochs 150 --batch_size 4

# 消融实验训练（100 epochs）
python dcamf_net/train_ablation.py --data_dir data/ShipsEar/train --ablation 1  # 去掉全局分支
python dcamf_net/train_ablation.py --data_dir data/ShipsEar/train --ablation 2  # 去掉局部分支
python dcamf_net/train_ablation.py --data_dir data/ShipsEar/train --ablation 3  # 纯MHSA（去卷积增强）

# 基线模型
python baselines/conv_tasnet/train.py
python baselines/dprnn/train.py
```

### 测试/评估

```bash
# DCAMF-Net 单测试集评估（checkpoint 必传）
python dcamf_net/test.py --checkpoint experiments/dcamf_net/checkpoints/best_SISNR.pth --test_dir data/ShipsEar/test1

# 多测试集评估
python dcamf_net/test.py --checkpoint experiments/dcamf_net/checkpoints/best_SISNR.pth \
  --test_dir data/ShipsEar/test1 data/ShipsEar/test2 data/ShipsEar/test3 data/DeepShip/test

# 消融模型批量评估
python dcamf_net/test_ablation_batch.py

# 统一评测所有模型并写入 experiments.txt
python scripts/compute_all_metrics.py
```

### 数据准备

```bash
# 标准 SNR 分布（[-15,-10], [-10,-5], [-5,0] dB 三区间）
python scripts/prepare_data.py

# 高/低 SNR 变体（用于融合权重分析）
python scripts/prepare_data_high.py
python scripts/prepare_data_low.py
```

### 出图

```bash
# 单张图片
python scripts/plot_fig4_1_shipsear_psd.py
python scripts/plot_fig4_2_time_waveform.py
# ... (fig4_3 到 fig4_8 类推)

# IEEE 英文组合图
python scripts/plot_ieee_figures.py
```

出图脚本依赖（本地）：MATLAB 已安装（`plot_fig4_2` 用 `soundfile`，其余用 `scipy`）。所有图同时输出 PDF + PNG 到 `figures/`。

## 架构

### 数据流

```
原始音频 (ShipsEar/DeepShip, 各船型/噪声类别)
  → prepare_data.py (16kHz重采样, 3秒片段, SNR混合)
  → data/{dataset}/{split}/noisy/ + clean/
  → AudioDenoisingDataset (重叠分段, 返回 (1, 48000))
  → DCAMFNet 前向 → 估计噪声 → 干净信号 = 输入 - 估计噪声
```

### 模型结构 (`dcamf_net/model.py`)

**Encoder-Masker-Decoder** 架构，核心创新为估计噪声（而非直接估计信号）：

1. **Encoder** (`ConvEncoder`): Conv1d(1→256, K=80, S=40) → PReLU → LayerNorm → 分块 segment(chunk_size=500, hop=250)
2. **DCAM Blocks** (×10): 每个 block 含两条分支 —
   - **Global branch**: 沿帧维度做 ConvEnhancedMHSA + GRU-FFN，捕获帧间上下文
   - **Local branch**: 沿局部段维度做 ConvEnhancedMHSA + GRU-FFN，捕获段内细节
   - 输出 = 恒等残差 + 0.5×local + 0.5×global
   - 每 block 生成一个 mask（gated tanh·sigmoid 机制）
3. **Multi-Layer Mask Fusion**: 10 个 mask 通过可学习 softmax 权重加权融合
4. **Decoder** (`ConvDecoder`): ConvTranspose1d(256→1, K=80, S=40)

### 其余关键模块

- **ConvEnhancedMHSA**: MultiheadAttention + 逐点Conv→GLU→深度Conv→BN→Swish→逐点Conv→Dropout 的卷积增强支路
- **ImprovedFFN**: GRU + LeakyReLU + Linear + Dropout（替代传统 MLP FFN）
- **Loss** (`loss.py`): r-nSISNR = `SISNR(n_est, n_true) + SISNR(s_est, s_true)`，取负值做最小化
- **消融变体**: Ablation1(去全局分支) / Ablation2(去局部分支) / Ablation3(纯MHSA无卷积增强) 在 `model_ablation*.py`

### 关键路径

| 用途 | 路径 |
|------|------|
| 模型定义 | `dcamf_net/model.py` |
| 训练入口 | `dcamf_net/train.py` |
| 推理评估 | `dcamf_net/test.py` + `dcamf_net/config.py` |
| 数据加载 | `dcamf_net/dataset.py` |
| 损失函数 | `dcamf_net/loss.py` |
| 出图公共工具 | `scripts/plot_utils.py` |
| 线谱检测算法 | `scripts/plot_utils.py` → `find_line_spectra()` |
| 掩码融合权重日志 | `experiments/mask_fusion_weights/` |
| 模型最佳权重 | `experiments/dcamf_net/checkpoints/best_SISNR.pth` |

## 论文出图约定

修改或新增 `scripts/plot_fig4_*.py` 时必须遵循（详见 `plot_utils.py` 的 `setup_style()`）：

- **画布宽度**: 6.5 inch（对应 A4 正文宽 16.5cm），高度按内容比例调整
- **字体**: SimHei（中文），字号 8–10pt
- **无图内标题** — 标题由 Word 排版添加，不要用 `plt.title()`
- **输出格式**: PDF（矢量）+ PNG（预览），保存到 `figures/`，命名 `fig4_X_*.pdf/png`
- **数据输出**: 涉及关键数据的图（fig4_3, fig4_5）需在终端打印详细数值用于论文写作
- **颜色**: IEEE 配色，`plot_utils.py` 中预定义
- 线谱检测、样本选择等共用逻辑在 `plot_utils.py`，各图脚本只写自己的差异部分
- `plot_utils.py` 包含硬编码的 Windows 绝对路径（`PROJECT_ROOT`），跨平台使用时需修改

## 环境依赖

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install tqdm numpy scipy matplotlib soundfile thop fast-bss-eval tensorboard h5py asteroid requests torchinfo
```

无 `requirements.txt` 或 `setup.py`，依赖直接安装。CRN 基线在 `baselines/CRN-causal/` 下有独立的 `requirements.txt`。

## 数据集结构

```
raw_data/
├── ShipsEar/{passenger,roro,motorboat,wind,flow,reservoir}/
└── DeepShip/{Cargo,Tug,Passengership,Tanker}/

data/
├── ShipsEar/{train,test1,test2,test3}/{noisy,clean}/
└── DeepShip/{test}/{noisy,clean}/
```

原始数据 gitignored，处理后数据 gitignored。
