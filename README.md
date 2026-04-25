# HydroDenoise-TF

## 步骤1：[AutoDL](https://www.autodl.com/)租用实例

规格详情：
- **GPU**：RTX 5090（32 GB）× 1
- **镜像**：PyTorch 2.1.0 + Python 3.10（Ubuntu 22.04）+ CUDA 12.1

启动实例后，在 **JupyterLab终端** 中执行后续步骤。

## 步骤2：配置环境，克隆仓库

在**JupyterLab终端**执行下面语句加速GitHub访问：

    source /etc/network_turbo

创建conda环境：

    conda create -n dcamf python=3.10 -y
    conda init

执行完后重启终端，安装环境依赖：

    conda activate dcamf
    
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip install tqdm numpy scipy matplotlib soundfile thop fast-bss-eval tensorboard
    
对仓库进行克隆：

    cd /root/autodl-tmp

    git clone https://github.com/LuKaderZ/HydroDenoise-TF.git

    cd HydroDenoise-TF


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

## 步骤6：本地推理与绘图

训练完成后，服务器上保存了最佳模型权重 `best_SISNR.pth`。后续推理和绘图在**本地**进行。

将以下文件从服务器下载到本地项目对应目录：
- 服务器路径：`/root/autodl-tmp/HydroDenoise-TF/experiments/dcamf_net/checkpoints/best_SISNR.pth`
- 本地路径：`experiments/dcamf_net/checkpoints/best_SISNR.pth`

在**本地**项目根目录下的 `dcamf_net/` 文件夹中执行：

    python test.py

在 MATLAB 中打开并运行：

    matlab/plot_dcamf_metrics.m

## 步骤7：基线模型

[CRN](https://github.com/JupiterEthan/CRN-causal)删除了`.git`文件夹，修改了`stft.py`文件进行版本适配,`models.py` 修改为`weights_only=False`以适配版本。

依次执行`adapter`下的三个数据适配脚本。

cd baselines\CRN-causal\scripts

python train.py --gpu_ids=0 --tr_list=../filelists/tr_list.txt --cv_file=../data/datasets/cv/cv.ex --ckpt_dir=exp --logging_period=5 --lr=0.0002 --time_log=./time.log --unit=utt --batch_size=16 --buffer_size=32 --max_n_epochs=150

python test.py --gpu_ids=0 --ckpt_dir=exp --model_file=exp/models/best.pt --tt_list=../filelists/tt_list.txt --est_path=../data/estimates