#!/usr/bin/env python3
"""
将 DCAMF-Net 测试集 WAV 转换为 CRN 格式的测试数据。
自动检测项目根目录，无需手动配置路径。
"""

import os
import h5py
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ==================== 自动计算项目根目录 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

TEST_SETS = [
    ("test1", os.path.join("data", "ShipsEar", "test1")),
    ("DeepShip", os.path.join("data", "DeepShip", "test")),
]

OUTPUT_BASE = os.path.join(
    PROJECT_ROOT, "baselines", "CRN-causal", "data", "datasets", "tt"
)
RMS_TARGET = 1.0
# ============================================================


def convert_test_set(name, rel_path):
    clean_dir = os.path.join(PROJECT_ROOT, rel_path, "clean")
    noisy_dir = os.path.join(PROJECT_ROOT, rel_path, "noisy")

    if not os.path.exists(clean_dir):
        print(f"警告：测试集 {name} 目录不存在 ({clean_dir})，跳过。")
        return

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    output_file = os.path.join(OUTPUT_BASE, f"tt_{name}.ex")

    files = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith(".wav")])
    if not files:
        print(f"警告：测试集 {name} 中没有 .wav 文件，跳过。")
        return

    print(f"转换测试集 {name}，共 {len(files)} 个样本 -> {output_file}")

    writer = h5py.File(output_file, "w")
    for idx, fname in enumerate(tqdm(files, desc=f"转换 {name}")):
        clean_path = os.path.join(clean_dir, fname)
        noisy_path = os.path.join(noisy_dir, fname)

        clean, _ = sf.read(clean_path)
        noisy, _ = sf.read(noisy_path)

        if clean.ndim > 1:
            clean = clean.mean(axis=1)
        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1)

        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]

        rms_noisy = np.sqrt(np.mean(noisy**2) + 1e-8)
        scale = RMS_TARGET / rms_noisy
        noisy *= scale
        clean *= scale

        grp = writer.create_group(str(idx))
        grp.create_dataset("mix", data=noisy.astype(np.float32), chunks=True)
        grp.create_dataset("sph", data=clean.astype(np.float32), chunks=True)
    writer.close()
    print(f"测试集 {name} 转换完成。")


def main():
    for name, rel_path in TEST_SETS:
        convert_test_set(name, rel_path)


if __name__ == "__main__":
    main()
