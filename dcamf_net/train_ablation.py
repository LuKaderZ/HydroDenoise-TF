#!/usr/bin/env python3
"""
DCAMF-Net 消融实验训练脚本
用法:
    python train_ablation.py --model model_ablation1 --train_dir ../data/ShipsEar/train --save_dir ../experiments/ablation/ablation1/checkpoints
    python train_ablation.py --model model_ablation2 --train_dir ../data/ShipsEar/train --save_dir ../experiments/ablation/ablation2/checkpoints
    python train_ablation.py --model model_ablation3 --train_dir ../data/ShipsEar/train --save_dir ../experiments/ablation/ablation3/checkpoints
"""

import os
import argparse
import importlib
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import AudioDenoisingDataset
from loss import RnSISNR, sisnr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="DCAMF-Net 消融实验训练")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["model_ablation1", "model_ablation2", "model_ablation3"],
        help="消融模型文件名（不含.py）",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="训练数据根目录（包含clean/noisy子目录）",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints_ablation",
        help="模型保存目录（建议设为 experiments/ablation/ablationX/checkpoints）",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def calculate_sdr(est, target, eps=1e-8):
    noise = target - est
    return (
        10
        * torch.log10(
            torch.sum(target**2, -1) / (torch.sum(noise**2, -1) + eps) + eps
        ).mean()
    )


def train_one_epoch(model, dataloader, criterion, optimizer, max_grad_norm, device):
    model.train()
    total_loss = 0.0
    total_sisnr = 0.0
    total_sdr = 0.0
    n_batches = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        est = model(noisy)
        est_s, target_s = est.squeeze(1), clean.squeeze(1)
        loss = criterion(noisy.squeeze(1), target_s, est_s)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            total_sisnr += sisnr(est_s, target_s).mean().item()
            total_sdr += calculate_sdr(est_s, target_s).item()
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {
        "loss": total_loss / n_batches,
        "sisnr": total_sisnr / n_batches,
        "sdr": total_sdr / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_sisnr = 0.0
    total_sdr = 0.0
    n_batches = 0
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)
        est = model(noisy)
        est_s, target_s = est.squeeze(1), clean.squeeze(1)
        loss = criterion(noisy.squeeze(1), target_s, est_s)

        total_sisnr += sisnr(est_s, target_s).mean().item()
        total_sdr += calculate_sdr(est_s, target_s).item()
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    return {
        "loss": total_loss / n_batches,
        "sisnr": total_sisnr / n_batches,
        "sdr": total_sdr / n_batches,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"消融实验类型: {args.model}")
    print(f"设备: {device}")

    model_module = importlib.import_module(args.model)
    DCAMFNet = model_module.DCAMFNet

    model = DCAMFNet(
        in_channels=1,
        enc_channels=256,
        enc_kernel_size=80,
        enc_stride=40,
        chunk_size=500,
        hop_size=250,
        n_blocks=10,
        n_heads=8,
        ffn_hidden=512,
        dw_kernel_size=31,
        dropout=0.1,
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    full_dataset = AudioDenoisingDataset(
        root_dir=args.train_dir,
        sample_rate=args.sample_rate,
        segment_seconds=3.0,
        overlap_seconds=1.0,
    )
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    criterion = RnSISNR()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_val_sisnr = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, args.max_grad_norm, device
        )
        val_metrics = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f}, SISNR: {train_metrics['sisnr']:.2f}dB, SDR: {train_metrics['sdr']:.2f}dB | "
            f"Val Loss: {val_metrics['loss']:.4f}, SISNR: {val_metrics['sisnr']:.2f}dB, SDR: {val_metrics['sdr']:.2f}dB"
        )

        latest_path = os.path.join(args.save_dir, "latest.pth")
        torch.save(model.state_dict(), latest_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_loss.pth"))
            print(f"  -> 最佳损失模型已保存 (loss={best_val_loss:.4f})")

        if val_metrics["sisnr"] > best_val_sisnr:
            best_val_sisnr = val_metrics["sisnr"]
            torch.save(
                model.state_dict(), os.path.join(args.save_dir, "best_SISNR.pth")
            )
            print(f"  -> 最佳SISNR模型已保存 (SISNR={best_val_sisnr:.2f}dB)")

    print(
        f"训练完成。最佳验证损失: {best_val_loss:.4f}, 最佳验证SISNR: {best_val_sisnr:.2f}dB"
    )


if __name__ == "__main__":
    main()
