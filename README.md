# HydroDenoise-TF — DCAMF-Net：基于双分支卷积增强注意力与多层掩码融合的水声降噪网络

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

一种将深度学习与经典信号处理原理对应的端到端**时域水声降噪网络**。DCAMF-Net 在 ShipsEar 和 DeepShip 水声数据集上取得了最优的信号保真度（SDRi 14.35 dB），优于 Conv-TasNet、DPRNN、CRN 等基线模型。

**English version**: [README_EN.md](./README_EN.md)

**关键词**：水声降噪、深度学习、端到端时域去噪、双分支自注意力、多层掩码融合、过完备基分解、小波收缩、Conv-TasNet、DPRNN、ShipsEar、DeepShip、PyTorch

**水下声信号降噪** | **水声去噪** | **船舶辐射噪声** | **神经网络降噪** | **注意力机制** | **掩码估计**

---

- **Framework**: PyTorch 2.1 + CUDA 12.1
- **Baselines**: CRN / Conv-TasNet / DPRNN
- **Datasets**: ShipsEar / DeepShip
- **Metrics**: SI-SNRi, SDRi

---

## 环境配置

### 云端训练（AutoDL）

租用实例：RTX 5090（32 GB），镜像 `PyTorch 2.1.0 + Python 3.10（Ubuntu 22.04）+ CUDA 12.1`。

启动后在 JupyterLab 终端中执行：

```bash
# 加速 GitHub 访问
source /etc/network_turbo

# 创建 conda 环境
conda create -n dcamf python=3.10 -y
conda init
# 重启终端后继续
conda activate dcamf

# 安装依赖
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm numpy scipy matplotlib soundfile thop fast-bss-eval tensorboard h5py asteroid requests torchinfo

# 克隆仓库
cd /root/autodl-tmp
git clone https://github.com/LuKaderZ/HydroDenoise-TF.git
cd HydroDenoise-TF
```

### 本地环境（推理与绘图）

需要 Python 环境，安装依赖同上（无需 CUDA 版本的 PyTorch）。

---

## 数据准备

### 1. 下载数据集

下载 **ShipsEar** 和 **DeepShip** 数据集，按以下结构放入 `raw_data/`：

```text
raw_data/
├── ShipsEar/
│   ├── passenger/      # 客船辐射噪声
│   ├── roro/           # 滚装船辐射噪声
│   ├── motorboat/      # 摩托艇辐射噪声
│   ├── wind/           # 风噪声
│   ├── flow/           # 水流噪声
│   └── reservoir/      # 水库水面噪声
└── DeepShip/
    ├── Cargo/
    ├── Tug/
    ├── Passengership/
    └── Tanker/
```

### 2. 生成训练/测试数据

```bash
cd scripts
python prepare_data.py
```

完成后 `data/` 目录下将包含训练集和各测试集的 noisy / clean 音频对。

---

## 训练 DCAMF-Net

```bash
cd dcamf_net
python train.py --train_dir ../data/ShipsEar/train --lr 5e-4 --save_dir ../experiments/dcamf_net/checkpoints --use_tensorboard
```

训练完成后最佳权重保存为 `experiments/dcamf_net/checkpoints/best_SISNR.pth`。

---

## 基线模型

### CRN

基于 [CRN-causal](https://github.com/JupiterEthan/CRN-causal) 适配。仓库中已删除原 `.git` 目录，并对 `stft.py`（版本适配）和 `models.py`（`weights_only=False`）做了修改。

```bash
# 1. 依次运行 adapter/ 下的三个数据适配脚本
cd baselines/CRN-causal/scripts

# 2. 训练
python train.py --gpu_ids=0 --tr_list=../filelists/tr_list.txt --cv_file=../data/datasets/cv/cv.ex --ckpt_dir=exp --logging_period=5 --lr=0.0002 --time_log=./time.log --unit=utt --batch_size=16 --buffer_size=32 --max_n_epochs=150

# 3. 推理
python test.py --gpu_ids=0 --ckpt_dir=exp --model_file=exp/models/best.pt --tt_list=../filelists/tt_list.txt --est_path=../data/estimates
```

### Conv-TasNet / DPRNN

使用 `asteroid` 库的 `ConvTasNet` 和 `DPRNNTasNet` 实现。直接运行对应目录下的 `train.py` 即可，超参数已在脚本中预设。

---

## 消融实验

三种消融变体分别移除了全局支路、局部支路和卷积增强子模块：

```bash
python train_ablation.py --model model_ablation1 --train_dir ../data/ShipsEar/train --save_dir ../experiments/ablation/ablation1/checkpoints
python train_ablation.py --model model_ablation2 --train_dir ../data/ShipsEar/train --save_dir ../experiments/ablation/ablation2/checkpoints
python train_ablation.py --model model_ablation3 --train_dir ../data/ShipsEar/train --save_dir ../experiments/ablation/ablation3/checkpoints
```

训练完成后批量测试：

```bash
python test_ablation_batch.py
```

---

## 多层掩码融合权重验证

使用不同 SNR 分布的训练数据分别训练模型，对比各层融合权重的收敛行为：

```bash
python prepare_data_high.py   # 高 SNR 组（[-5, 0] dB）
python prepare_data_low.py    # 低 SNR 组（[-15, -10] dB）
```

训练完毕后，将各组的 `train.log` 重命名放入：

```text
experiments/mask_fusion_weights/
├── train_avg.log     # 平均 SNR 组（三组等概率混合）
├── train_low.log     # 低 SNR 组
└── train_high.log    # 高 SNR 组
```

---

## 推理与绘图

### Python 推理

将服务器上的最佳权重 `experiments/dcamf_net/checkpoints/best_SISNR.pth` 下载到本地对应路径，然后：

```bash
cd dcamf_net
python test.py
```

### Python 绘图

在 Python 环境中依次运行 `scripts/` 下的绘图脚本：

| 图号 | 脚本 / 来源 | 内容 |
| ------ | ------------ | ------------------- |
| 图 2.1 | `plot_fig2_1_ship_psd.py` | 船舶辐射噪声功率谱密度 |
| 图 2.2 | `plot_fig2_2_noise_timefreq.py` | 不同噪声类型时域/频域/时频图 |
| 图 2.3 | `plot_fig2_3_noisy_psd.py` | -15dB混合信号PSD对比 |
| 图 3.1–3.3 | `drawio/fig3_*_*.drawio` | 网络结构图（draw.io） |
| 图 4.1 | `drawio/fig4_1_data_pipeline.drawio` | 数据集构建与预处理流程（draw.io） |
| 图 4.2 | `plot_fig4_2_training_curves.py` | 训练损失与SI-SNR曲线 |
| 图 4.3 | `plot_fig4_3_time_waveform.py` | 时域波形对比 |
| 图 4.4 | `plot_fig4_4_overall_spectrum.py` | 总体频谱对比 |
| 图 4.5 | `plot_fig4_5_spectrogram.py` | 语谱图对比 |
| 图 4.6 | `plot_fig4_6_line_spectra.py` | 关键线谱功率恢复 |
| 图 4.7 | `plot_fig4_7_generalization.py` | 泛化性能评估 |
| 图 4.8 | `plot_fig4_8_fusion_weights.py` | 融合权重分布 |
| 图 4.9 | `plot_fig4_9_noise_estimation.py` | 噪声估计验证 |

脚本会自动读取实验产物并保存图像至 `figures/` 目录。

---

## 项目结构

```text
HydroDenoise-TF/
├── dcamf_net/                  # DCAMF-Net 模型代码
│   ├── model.py                #   网络结构定义
│   ├── train.py                #   训练脚本
│   ├── test.py                 #   推理评估脚本
│   ├── dataset.py              #   数据加载
│   ├── loss.py                 #   r-nSISNR 损失函数
│   ├── config.py               #   评估工具 / 模型构建
│   ├── model_ablation[1-3].py  #   消融变体
│   ├── train_ablation.py       #   消融训练
│   └── test_ablation_batch.py  #   消融测试
├── baselines/                  # 基线模型
│   ├── CRN-causal/
│   ├── conv_tasnet/
│   └── dprnn/
├── scripts/                    # 数据准备 + 论文图表绘制
│   ├── prepare_data.py
│   ├── prepare_data_high.py
│   ├── prepare_data_low.py
│   ├── plot_utils.py
│   └── plot_fig*_*.py
├── drawio/                     # 论文结构图源文件
│   ├── fig3_1_architecture.drawio
│   ├── fig3_2_dcam_block.drawio
│   ├── fig3_3_cemhsa.drawio
│   └── fig4_1_data_pipeline.drawio
├── figures/                    # 图表输出
├── experiments/                # 训练产物（检查点、日志、降噪音频）
├── raw_data/                   # 原始数据集
├── data/                       # 处理后的训练/测试数据
└── experiments.txt             # 实验指标记录
```
