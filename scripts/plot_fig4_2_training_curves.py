"""图4-2: DCAMF-Net 训练损失与 SI-SNR 曲线
用法: python scripts/plot_fig4_2_training_curves.py [train.log路径]
兼容新旧两种日志格式
"""
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import setup_style

setup_style()
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'legend.fontsize': 7.5})


def parse_log(log_path):
    """解析训练日志，兼容新旧两种格式"""
    # 新格式: SI-SNR(s) + SI-SNR(n)
    pat_new = re.compile(
        r'Epoch\s*\[(\d+)\].*?'
        r'Train\s*\[Loss:\s*([-\d.]+),\s*SI-SNR\(s\):\s*([-\d.]+)dB,\s*SI-SNR\(n\):\s*([-\d.]+)dB\].*?'
        r'Val\s*\[Loss:\s*([-\d.]+),\s*SI-SNR\(s\):\s*([-\d.]+)dB,\s*SI-SNR\(n\):\s*([-\d.]+)dB\]'
    )
    # 旧格式: 单 SISNR
    pat_old = re.compile(
        r'Epoch\s*\[(\d+)\].*?'
        r'Train\s*\[Loss:\s*([-\d.]+),\s*SISNR:\s*([-\d.]+)dB\].*?'
        r'Val\s*\[Loss:\s*([-\d.]+),\s*SISNR:\s*([-\d.]+)dB\]'
    )

    data = {'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_sisnr': [], 'val_sisnr': [],
            'train_noise_sisnr': [], 'val_noise_sisnr': [],
            'format': None}

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pat_new.search(line)
            if m:
                data['format'] = 'new'
                data['epoch'].append(int(m.group(1)))
                data['train_loss'].append(float(m.group(2)))
                data['train_sisnr'].append(float(m.group(3)))
                data['train_noise_sisnr'].append(float(m.group(4)))
                data['val_loss'].append(float(m.group(5)))
                data['val_sisnr'].append(float(m.group(6)))
                data['val_noise_sisnr'].append(float(m.group(7)))
                continue
            m = pat_old.search(line)
            if m:
                data['format'] = 'old'
                data['epoch'].append(int(m.group(1)))
                data['train_loss'].append(float(m.group(2)))
                data['train_sisnr'].append(float(m.group(3)))
                data['val_loss'].append(float(m.group(4)))
                data['val_sisnr'].append(float(m.group(5)))

    return data


def plot_curves(data, save_path):
    epochs = data['epoch']
    is_new = data['format'] == 'new'
    n_panels = 3 if is_new else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(6.5, 3.2))
    if n_panels == 2:
        axes = [axes[0], axes[1], None]

    c_train, c_val = '#D95319', '#0072BD'

    # (a) Loss
    ax = axes[0]
    ax.plot(epochs, data['train_loss'], color=c_train, lw=1.0, label='训练')
    ax.plot(epochs, data['val_loss'], color=c_val, lw=1.0, label='验证')
    ax.set_xlabel('Epoch'); ax.set_ylabel('r-nSISNR Loss')
    ax.legend(loc='upper right', fontsize=7); ax.grid(True, alpha=0.25, lw=0.4)
    ax.set_title('(a) 训练损失', fontsize=10, fontweight='bold')

    # (b) SI-SNR(s)
    ax = axes[1]
    ax.plot(epochs, data['train_sisnr'], color=c_train, lw=1.0, label='训练 SI-SNR')
    ax.plot(epochs, data['val_sisnr'], color=c_val, lw=1.0, label='验证 SI-SNR')
    ax.set_xlabel('Epoch'); ax.set_ylabel('SI-SNR (dB)')
    ax.legend(loc='lower right', fontsize=7); ax.grid(True, alpha=0.25, lw=0.4)
    ax.set_title('(b) 信号恢复 SI-SNR', fontsize=10, fontweight='bold')

    # (c) SI-SNR(n) — 仅新格式
    if is_new and axes[2] is not None:
        ax = axes[2]
        ax.plot(epochs, data['train_noise_sisnr'], color=c_train, lw=1.0, label='训练 SI-SNR(n̂,n)')
        ax.plot(epochs, data['val_noise_sisnr'], color=c_val, lw=1.0, label='验证 SI-SNR(n̂,n)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('SI-SNR (dB)')
        ax.legend(loc='lower right', fontsize=7); ax.grid(True, alpha=0.25, lw=0.4)
        ax.set_title('(c) 噪声估计 SI-SNR', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(str(save_path) + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(str(save_path) + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'曲线已保存: {save_path}.pdf / .png')

    # 终端总结
    if len(epochs) > 0:
        last = len(epochs) - 1
        print(f'\n训练统计 ({len(epochs)} epochs, {data["format"]}格式):')
        print(f'  Train Loss: {data["train_loss"][0]:.2f} → {data["train_loss"][last]:.2f}')
        print(f'  Val   Loss: {data["val_loss"][0]:.2f} → {data["val_loss"][last]:.2f}')
        print(f'  Train SI-SNR: {data["train_sisnr"][0]:.1f} → {data["train_sisnr"][last]:.1f} dB')
        print(f'  Val   SI-SNR: {data["val_sisnr"][0]:.1f} → {data["val_sisnr"][last]:.1f} dB')
        if is_new:
            print(f'  Train SI-SNR(n): {data["train_noise_sisnr"][0]:.1f} → {data["train_noise_sisnr"][last]:.1f} dB')
            print(f'  Val   SI-SNR(n): {data["val_noise_sisnr"][0]:.1f} → {data["val_noise_sisnr"][last]:.1f} dB')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        log_path = Path(__file__).resolve().parent.parent / 'experiments' / 'dcamf_net' / 'checkpoints' / 'train.log'
    else:
        log_path = Path(sys.argv[1])

    if not log_path.exists():
        print(f'日志文件不存在: {log_path}')
        print(f'用法: python {Path(__file__).name} <train.log路径>')
        sys.exit(1)

    data = parse_log(str(log_path))
    print(f'解析到 {len(data["epoch"])} 个 epoch ({data["format"]}格式)')

    out_dir = Path(__file__).resolve().parent.parent / 'figures'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'fig4-2_training_curves'
    plot_curves(data, str(out_path))
