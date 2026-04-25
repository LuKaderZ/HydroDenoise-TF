# HydroDenoise-TF

## 步骤1：[AutoDL](https://www.autodl.com/)租用实例

规格详情：
- **GPU**：RTX 5090（32 GB）× 1
- **镜像**：PyTorch 2.1.0 + Python 3.10（Ubuntu 22.04）+ CUDA 12.1

启动实例后，在 **JupyterLab终端** 中执行后续步骤。

## 步骤2：克隆仓库

在**JupyterLab终端**执行下面语句：

    source /etc/network_turbo

对仓库进行克隆：

    cd /root/autodl-tmp
    git clone https://github.com/LuKaderZ/HydroDenoise-TF.git
    cd HydroDenoise-TF

创建conda环境：

    conda create -n dcamf python=3.10 -y
    conda init

执行完后重启终端

    conda activate dcamf

    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip install tqdm numpy scipy matplotlib soundfile thop fast-bss-eval tensorboard

## 步骤3：准备原始数据

下载 **ShipsEar** 和 **DeepShip** 数据集，解压后按照以下结构放入raw_data/目录：

```
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

## 步骤4：生成训练/测试数据

    cd scripts
    python prepare_data.py

## 步骤5：开始训练

    cd ../dcamf_net
    python train.py --train_dir ../data/ShipsEar/train --lr 5e-4 --save_dir ../experiments/dcamf_net/checkpoints --use_tensorboard