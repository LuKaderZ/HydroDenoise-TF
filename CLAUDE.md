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
# 七张论文图（全部输出到 figures/，PDF+PNG）
# 图4.1 是 draw.io 流程图，见 drawio/fig4_1_data_pipeline.drawio
python scripts/plot_fig4_2_time_waveform.py          # 图4.2 时域波形对比
python scripts/plot_fig4_3_overall_spectrum.py       # 图4.3 总体频谱对比
python scripts/plot_fig4_4_spectrogram.py            # 图4.4 语谱图对比
python scripts/plot_fig4_5_line_spectra.py           # 图4.5 线谱功率恢复
python scripts/plot_fig4_6_generalization.py         # 图4.6 泛化性能
python scripts/plot_fig4_7_fusion_weights.py         # 图4.7 融合权重分布
python scripts/plot_fig4_8_noise_estimation.py       # 图4.8 噪声估计验证

# 第二章插图
python scripts/plot_fig2_1_ship_psd.py               # 图2.1 船舶辐射噪声功率谱密度
python scripts/plot_fig2_2_noise_timefreq.py          # 图2.2 不同噪声类型时域/频域/时频图

# 全量指标评估
python scripts/compute_all_metrics.py                # 输出到 experiments.txt
```

所有脚本依赖 Python（scipy, numpy, soundfile, matplotlib），无需 MATLAB。

## 第三章网络结构图

`drawio/` 目录下有四张 draw.io 源文件：

- `fig3_1_architecture.drawio` — DCAMF-Net 整体架构
- `fig3_2_dcam_block.drawio` — DCAM 模块双分支结构
- `fig3_3_cemhsa.drawio` — CE-MHSA 卷积增强自注意力
- `fig4_1_data_pipeline.drawio` — 数据集构建与预处理流程

用 draw.io 桌面版打开编辑，导出 PDF 插入 Word。编辑时自动生成的 `.bkp` 备份文件已 gitignored。

## 架构

### 数据流

```text
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

| 用途             | 路径                                               |
| ---------------- | -------------------------------------------------- |
| 模型定义         | `dcamf_net/model.py`                               |
| 训练入口         | `dcamf_net/train.py`                               |
| 推理评估         | `dcamf_net/test.py` + `dcamf_net/config.py`        |
| 数据加载         | `dcamf_net/dataset.py`                             |
| 损失函数         | `dcamf_net/loss.py`                                |
| 出图公共工具     | `scripts/plot_utils.py`                            |
| 线谱检测算法     | `scripts/plot_utils.py` → `find_line_spectra()`    |
| 掩码融合权重日志 | `experiments/mask_fusion_weights/`                 |
| 模型最佳权重     | `experiments/dcamf_net/checkpoints/best_SISNR.pth` |

## 论文出图约定

修改或新增 `scripts/plot_fig4_*.py` 时必须遵循（详见 `.claude/skills/thesis-plots/SKILL.md`）：

- **画布宽度**: 6.5 inch（A4 正文宽 16.5cm），高度按内容调整
- **字体**: SimHei（中文，不支持组合字符如 n̂ → 用 nest 代替）
- **无图内标题**: `fig.suptitle()` 和单面板 `ax.set_title()` 一律去掉（标题由 Word 图注负责）；多面板子图标签如 (a)(b) 可保留
- **中文标签**: 所有轴标签、图例、注释必须中文
- **输出**: `figures/fig4-X_描述.pdf` + `.png`（短横连接图号，下划线连描述），dpi=300
- **数据输出**: 涉及关键数值的图（fig4_3, fig4_5, fig4_8）终端打印详细数据供论文写作
- **颜色**: `plot_utils.COLORS`（IEEE 配色），不要硬编码 hex
- **路径**: 从 `plot_utils` 统一导入，不要自定义路径
- **savefig**: 默认用 `bbox_inches='tight'`；手动 `subplots_adjust` 时不加
- **子图间距**: 2×N 或 3×N 布局用 `hspace=0.40, wspace=0.25, top=0.93`
- **图例性能**: 大量 artist 时避免 `loc='best'`，硬编码如 `loc='upper right'`

## 线谱检测算法 (`plot_utils.py` → `find_line_spectra`)

算法流程：中值滤波去趋势(窗口=n_bins/20) → 对比度×二阶导数窄度评分 → 谐波检验(4%容差) → 综合排名取 top-N

| 参数         | 默认值        | 原因                                  |
| ------------ | ------------- | ------------------------------------- |
| `freq_range` | `(100, 4000)` | 低于 100Hz 在 16kHz 采样下不可靠      |
| `prominence` | `1.5`         | 去趋势后对比度曲线上的弱线谱也能抓到  |
| `distance`   | `3`           | ~12Hz 最小间距，匹配 Welch 频率分辨率 |

**修改线谱算法后必须重跑 fig4_3 和 fig4_5**，两个脚本共享此函数且样本选择依赖线谱频率。

## 论文写作

项目目录下有两个 skill 辅助论文工作：

- **thesis-plots**（`.claude/skills/thesis-plots/SKILL.md`）：画图脚本规范，修改画图代码时自动加载
- **thesis-review**（`.claude/skills/thesis-review/SKILL.md`）：审查论文文字，检查过度声称、信号处理类比不准确、描述与代码不一致等问题

论文审查的核心原则（来自多轮 GPT 反馈）：

- 弱化因果断言，用"有助于/可能/在本文实验条件下"替代"验证了/证明了/确实"
- 单样本可视化图必须加"该样本仅用于定性可视化分析，整体性能判断以全测试集统计结果为准"
- 避免将神经网络模块描述为显式信号处理操作（如"谱减""维纳滤波""子带平滑"），代码中没有对应实现
- PReLU 不叫"稀疏化"，掩码融合权重是全局参数不叫"自适应选择"
- SDR 公式使用简单能量比，不是 `fast_bss_eval`

## 环境依赖

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install tqdm numpy scipy matplotlib soundfile thop fast-bss-eval tensorboard h5py asteroid requests torchinfo
```

无 `requirements.txt` 或 `setup.py`，依赖直接安装。CRN 基线在 `baselines/CRN-causal/` 下有独立的 `requirements.txt`。

## 数据集结构

```text
raw_data/
├── ShipsEar/{passenger,roro,motorboat,wind,flow,reservoir}/
└── DeepShip/{Cargo,Tug,Passengership,Tanker}/

data/
├── ShipsEar/{train,test1,test2,test3}/{noisy,clean}/
└── DeepShip/{test}/{noisy,clean}/
```

原始数据 gitignored，处理后数据 gitignored。
