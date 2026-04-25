#!/usr/bin/env python3
"""
Conv-TasNet 训练脚本（直接从 data/ShipsEar/train/ 加载数据，内部划分验证集）
"""

import os
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from asteroid.models import ConvTasNet
from asteroid.losses import pairwise_neg_sisdr
from tqdm import tqdm

# ==================== 路径配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "ShipsEar", "train")
EXP_DIR   = os.path.join(PROJECT_ROOT, "experiments", "conv_tasnet", "checkpoints")

BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 100
VAL_SPLIT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_KWARGS = {
    "n_src": 1,
    "sample_rate": 16000,
    "n_filters": 512,
    "kernel_size": 16,
    "stride": 8,
    "n_blocks": 8,
    "n_repeats": 3,
}


class WavDataset(Dataset):
    def __init__(self, root_dir):
        clean_dir = os.path.join(root_dir, "clean")
        noisy_dir = os.path.join(root_dir, "noisy")
        self.files = sorted([f for f in os.listdir(noisy_dir) if f.lower().endswith('.wav')])
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        noisy, _ = sf.read(os.path.join(self.noisy_dir, fname), dtype='float32')
        clean, _ = sf.read(os.path.join(self.clean_dir, fname), dtype='float32')

        if clean.ndim > 1: clean = clean.mean(axis=1)
        if noisy.ndim > 1: noisy = noisy.mean(axis=1)

        min_len = min(len(clean), len(noisy))
        clean, noisy = clean[:min_len], noisy[:min_len]

        mix = torch.from_numpy(noisy).unsqueeze(0)
        sph = torch.from_numpy(clean).unsqueeze(0)
        return mix, sph


def collate_fn(batch):
    mixes, sphs = zip(*batch)
    max_len = max(m.shape[1] for m in mixes)
    mix_padded = torch.stack([F.pad(m, (0, max_len - m.shape[1])) for m in mixes])
    sph_padded = torch.stack([F.pad(s, (0, max_len - s.shape[1])) for s in sphs])
    sph_padded = sph_padded.unsqueeze(1)
    return mix_padded, sph_padded


def main():
    os.makedirs(EXP_DIR, exist_ok=True)

    full_dataset = WavDataset(TRAIN_DIR)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model = ConvTasNet(**MODEL_KWARGS).to(DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = pairwise_neg_sisdr
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for mix, sph in pbar:
            mix, sph = mix.to(DEVICE), sph.to(DEVICE)
            optimizer.zero_grad()
            est_sources = model(mix)
            est_sources = est_sources.view(est_sources.size(0), est_sources.size(1), -1)
            sph = sph.view(sph.size(0), sph.size(1), -1)
            loss = criterion(est_sources, sph).mean()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for mix, sph in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                mix, sph = mix.to(DEVICE), sph.to(DEVICE)
                est_sources = model(mix)
                est_sources = est_sources.view(est_sources.size(0), est_sources.size(1), -1)
                sph = sph.view(sph.size(0), sph.size(1), -1)
                loss = criterion(est_sources, sph).mean()
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(EXP_DIR, "best_model.pth"))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(EXP_DIR, f"epoch_{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(EXP_DIR, "final_model.pth"))
    print("训练完成。")


if __name__ == "__main__":
    main()