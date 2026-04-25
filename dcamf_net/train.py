#!/usr/bin/env python3
"""
DCAMF-Net 训练脚本
"""

import os
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import thop
    from thop import profile
except (ImportError, ModuleNotFoundError):
    thop = None
    profile = None

from model import DCAMFNet
from dataset import AudioDenoisingDataset
from loss import RnSISNR, sisnr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(save_dir):
    logger = logging.getLogger("DCAMF-Net")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def profile_model(model, device, sample_rate=16000, duration=3.0):
    model.eval()
    input_size = (1, 1, int(sample_rate * duration))
    dummy_input = torch.randn(input_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if profile is not None:
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2) / 1e9
        return total_params, trainable_params, gflops
    return total_params, trainable_params, None


def save_plot(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    plt.title("r-nSISNR Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_sisnr"], "b-", label="Train SISNR")
    plt.plot(epochs, history["val_sisnr"], "r-", label="Val SISNR")
    plt.title("SISNR (dB)")
    plt.xlabel("Epochs")
    plt.ylabel("dB")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def calculate_sdr(est, target, eps=1e-8):
    noise = target - est
    sdr = 10 * torch.log10(
        torch.sum(target**2, dim=-1) / (torch.sum(noise**2, dim=-1) + eps) + eps
    )
    return sdr.mean()


def train_one_epoch(model, dataloader, criterion, optimizer, max_grad_norm, device):
    model.train()
    metrics = {"loss": 0.0, "sisnr": 0.0, "sdr": 0.0}
    n_batches = 0
    pbar = tqdm(dataloader, desc="Training", unit="batch", leave=False)
    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        estimated_clean = model(noisy)
        est_s, target_s = estimated_clean.squeeze(1), clean.squeeze(1)
        loss = criterion(noisy.squeeze(1), target_s, est_s)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            cur_sisnr = sisnr(est_s, target_s).mean().item()
            cur_sdr = calculate_sdr(est_s, target_s).item()

        metrics["loss"] += loss.item()
        metrics["sisnr"] += cur_sisnr
        metrics["sdr"] += cur_sdr
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.2f}", "sisnr": f"{cur_sisnr:.1f}dB"})
    return {k: v / n_batches for k, v in metrics.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    metrics = {"loss": 0.0, "sisnr": 0.0, "sdr": 0.0}
    n_batches = 0
    pbar = tqdm(dataloader, desc="Validation", unit="batch", leave=False)
    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)
        estimated_clean = model(noisy)
        est_s, target_s = estimated_clean.squeeze(1), clean.squeeze(1)
        loss = criterion(noisy.squeeze(1), target_s, est_s)
        metrics["loss"] += loss.item()
        metrics["sisnr"] += sisnr(est_s, target_s).mean().item()
        metrics["sdr"] += calculate_sdr(est_s, target_s).item()
        n_batches += 1
        pbar.set_postfix({"val_loss": f"{loss.item():.2f}"})
    return {k: v / n_batches for k, v in metrics.items()}


def main(args):
    set_seed(args.seed)
    logger = setup_logger(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = DCAMFNet(
        enc_channels=args.enc_channels,
        ffn_hidden=args.ffn_hidden,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
    ).to(device)

    total_p, train_p, gflops = profile_model(model, device)
    logger.info(f"Model Profile (ffn_hidden={args.ffn_hidden}):")
    logger.info(f"  - Total Parameters: {total_p / 1e6:.2f} M")
    if gflops:
        logger.info(f"  - GFLOPs (3s input): {gflops:.2f} G")
    else:
        logger.warning("  - GFLOPs: 'thop' library not found.")

    full_dataset = AudioDenoisingDataset(
        root_dir=args.train_dir, sample_rate=args.sample_rate
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

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    criterion = RnSISNR()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 学习率衰减（余弦退火，第一个周期超过总epochs则仅单调下降）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min, last_epoch=-1
    )

    writer = None
    if args.use_tensorboard:
        log_dir = (
            args.log_dir if args.log_dir else os.path.join(args.save_dir, "tensorboard")
        )
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logging enabled. Logs will be saved to: {log_dir}")

    history = {"train_loss": [], "val_loss": [], "train_sisnr": [], "val_sisnr": []}
    best_val_loss = float("inf")
    best_val_sisnr = float("-inf")

    for epoch in range(args.epochs):
        train_res = train_one_epoch(
            model, train_loader, criterion, optimizer, args.max_grad_norm, device
        )
        val_res = validate(model, val_loader, criterion, device)

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["train_sisnr"].append(train_res["sisnr"])
        history["val_sisnr"].append(val_res["sisnr"])
        save_plot(history, os.path.join(args.save_dir, "learning_curves.png"))

        log_msg = (
            f"Epoch [{epoch + 1:03d}] | LR: {curr_lr:.2e} | "
            f"Train [Loss: {train_res['loss']:.4f}, SISNR: {train_res['sisnr']:.2f}dB] | "
            f"Val [Loss: {val_res['loss']:.4f}, SISNR: {val_res['sisnr']:.2f}dB]"
        )
        logger.info(log_msg)

        if writer is not None:
            writer.add_scalars(
                "Metrics/Loss",
                {"train": train_res["loss"], "val": val_res["loss"]},
                epoch,
            )
            writer.add_scalars(
                "Metrics/SISNR",
                {"train": train_res["sisnr"], "val": val_res["sisnr"]},
                epoch,
            )
            writer.add_scalars(
                "Metrics/SDR", {"train": train_res["sdr"], "val": val_res["sdr"]}, epoch
            )
            writer.add_scalar("Metrics/Learning_Rate", curr_lr, epoch)

        latest_path = os.path.join(args.save_dir, "latest.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "args": args,
            },
            latest_path,
        )

        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_loss.pth"))
            logger.info(
                f"  -> Best model (loss) saved with val_loss={best_val_loss:.4f}"
            )

        if val_res["sisnr"] > best_val_sisnr:
            best_val_sisnr = val_res["sisnr"]
            torch.save(
                model.state_dict(), os.path.join(args.save_dir, "best_SISNR.pth")
            )
            logger.info(
                f"  -> Best model (SISNR) saved with val_sisnr={best_val_sisnr:.2f}dB"
            )

    logger.info(
        f"Training complete. Best val loss: {best_val_loss:.4f}, Best val SISNR: {best_val_sisnr:.2f}dB"
    )

    if writer is not None:
        writer.close()
        logger.info("TensorBoard writer closed.")


def parse_args():
    parser = argparse.ArgumentParser(description="DCAMF-Net Training")
    parser.add_argument(
        "--train_dir", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    parser.add_argument("--enc_channels", type=int, default=256)
    parser.add_argument("--ffn_hidden", type=int, default=512)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)

    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for Adam"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)

    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--sample_rate", type=int, default=16000)

    parser.add_argument(
        "--T_0", type=int, default=200
    )  # 第一个周期长度，大于epochs则无重启
    parser.add_argument("--T_mult", type=int, default=20)
    parser.add_argument("--eta_min", type=float, default=1e-7)

    parser.add_argument(
        "--use_tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument("--log_dir", type=str, default="/root/tf-logs/")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
