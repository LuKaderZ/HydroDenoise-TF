#!/usr/bin/env python3
"""
消融实验批量测试脚本
依次加载三个消融模型的权重，对 test1 进行降噪，结果保存到 experiments/ablation/ablationX/denoised/
"""

import os
import torch
import soundfile as sf
import importlib
from tqdm import tqdm

# ==================== 路径配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TEST_CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "ShipsEar", "test1", "clean")
TEST_NOISY_DIR = os.path.join(PROJECT_ROOT, "data", "ShipsEar", "test1", "noisy")

# 权重文件路径
CHECKPOINT_PATHS = {
    "ablation1": os.path.join(
        PROJECT_ROOT,
        "experiments",
        "ablation",
        "ablation1",
        "checkpoints",
        "best_SISNR.pth",
    ),
    "ablation2": os.path.join(
        PROJECT_ROOT,
        "experiments",
        "ablation",
        "ablation2",
        "checkpoints",
        "best_SISNR.pth",
    ),
    "ablation3": os.path.join(
        PROJECT_ROOT,
        "experiments",
        "ablation",
        "ablation3",
        "checkpoints",
        "best_SISNR.pth",
    ),
}

# 输出根目录
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "experiments", "ablation")

# 模型模块名
MODULE_NAMES = {
    "ablation1": "model_ablation1",
    "ablation2": "model_ablation2",
    "ablation3": "model_ablation3",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(module_name, checkpoint_path):
    model_module = importlib.import_module(module_name)
    DCAMFNet = model_module.DCAMFNet
    model = DCAMFNet(
        in_channels=1,
        enc_channels=256,
        enc_kernel_size=80,
        enc_stride=40,
        chunk_size=500,
        hop_size=250,
        n_blocks=6,
        n_heads=8,
        ffn_hidden=512,
        dw_kernel_size=31,
        dropout=0.1,
    ).to(DEVICE)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def process_one_file(model, input_path, output_path):
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        enhanced = model(waveform).squeeze(0).cpu().numpy()

    sf.write(output_path, enhanced.squeeze(0), sr)


def test_one_model(name, checkpoint_path, module_name):
    print(f"\n========== 测试 {name} ==========")
    if not os.path.exists(checkpoint_path):
        print(f"警告：权重文件不存在 {checkpoint_path}，跳过。")
        return

    model = load_model(module_name, checkpoint_path)
    output_dir = os.path.join(OUTPUT_BASE, name, "denoised")
    os.makedirs(output_dir, exist_ok=True)

    test_files = sorted([f for f in os.listdir(TEST_NOISY_DIR) if f.endswith(".wav")])
    for fname in tqdm(test_files, desc=f"处理 {name}"):
        noisy_path = os.path.join(TEST_NOISY_DIR, fname)
        out_path = os.path.join(output_dir, fname)
        process_one_file(model, noisy_path, out_path)

    print(f"{name} 测试完成，结果保存在 {output_dir}")


def main():
    for name in ["ablation1", "ablation2", "ablation3"]:
        test_one_model(name, CHECKPOINT_PATHS[name], MODULE_NAMES[name])
    print("\n所有消融模型测试完毕。")


if __name__ == "__main__":
    main()
