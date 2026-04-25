#!/usr/bin/env python3
"""
将 DCAMF-Net 训练集 WAV 转换为 CRN 格式，并自动划分训练/验证集。
适配重构后的项目结构：脚本位于 baselines/CRN-causal/adapter/ 下，
因而 PROJECT_ROOT 需向上三级到达 HydroDenoise-TF/。
"""

import os
import h5py
import soundfile as sf
import numpy as np
import random
from tqdm import tqdm

# ==================== 自动计算项目根目录 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # baselines/CRN-causal/adapter/
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(SCRIPT_DIR))
)  # HydroDenoise-TF/

# 原始训练数据路径
CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "ShipsEar", "train", "clean")
NOISY_DIR = os.path.join(PROJECT_ROOT, "data", "ShipsEar", "train", "noisy")

# CRN 数据输出目录
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "baselines", "CRN-causal", "data", "datasets")
TR_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "tr")
CV_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "cv")
CV_OUTPUT_FILE = os.path.join(CV_OUTPUT_DIR, "cv.ex")

RMS_TARGET = 1.0
VAL_SPLIT = 0.1
RANDOM_SEED = 42
# ============================================================


def load_and_preprocess(fname):
    clean_path = os.path.join(CLEAN_DIR, fname)
    noisy_path = os.path.join(NOISY_DIR, fname)

    clean, sr = sf.read(clean_path)
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

    return clean.astype(np.float32), noisy.astype(np.float32)


def main():
    os.makedirs(TR_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CV_OUTPUT_DIR, exist_ok=True)

    files = sorted([f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(".wav")])
    if not files:
        raise RuntimeError(f"在 {CLEAN_DIR} 中未找到 .wav 文件")

    print(f"找到 {len(files)} 个训练样本")
    print(f"原始数据目录: {CLEAN_DIR}")
    print(f"训练集输出: {TR_OUTPUT_DIR}")
    print(f"验证集输出: {CV_OUTPUT_FILE}")

    random.seed(RANDOM_SEED)
    random.shuffle(files)

    split_idx = int(len(files) * (1 - VAL_SPLIT))
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    print(f"训练集样本数: {len(train_files)}")
    print(f"验证集样本数: {len(val_files)}")

    for idx, fname in enumerate(tqdm(train_files, desc="转换训练集")):
        clean, noisy = load_and_preprocess(fname)
        out_path = os.path.join(TR_OUTPUT_DIR, f"tr_{idx}.ex")
        writer = h5py.File(out_path, "w")
        writer.create_dataset("mix", data=noisy, chunks=True)
        writer.create_dataset("sph", data=clean, chunks=True)
        writer.close()

    writer = h5py.File(CV_OUTPUT_FILE, "w")
    for idx, fname in enumerate(tqdm(val_files, desc="转换验证集")):
        clean, noisy = load_and_preprocess(fname)
        grp = writer.create_group(str(idx))
        grp.create_dataset("mix", data=noisy, chunks=True)
        grp.create_dataset("sph", data=clean, chunks=True)
        writer.flush()
    writer.close()

    print(f"\n转换完成！训练集保存至 {TR_OUTPUT_DIR}，验证集保存至 {CV_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
