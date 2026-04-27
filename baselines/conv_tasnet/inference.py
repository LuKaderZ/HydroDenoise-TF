#!/usr/bin/env python3
"""
Conv-TasNet 推理脚本（test1 + DeepShip）
"""

import os
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from asteroid.models import ConvTasNet
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "conv_tasnet", "checkpoints", "best_model.pth")

SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WavDataset(Dataset):
    def __init__(self, root_dir):
        noisy_dir = os.path.join(root_dir, "noisy")
        self.files = sorted([f for f in os.listdir(noisy_dir) if f.lower().endswith('.wav')])
        self.noisy_dir = noisy_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy, _ = sf.read(os.path.join(self.noisy_dir, self.files[idx]), dtype='float32')
        if noisy.ndim > 1: noisy = noisy.mean(axis=1)
        return torch.from_numpy(noisy).unsqueeze(0)


def run_inference(data_dir, est_dir, desc="推理"):
    """通用推理函数，对指定测试集生成增强音频"""
    os.makedirs(est_dir, exist_ok=True)

    model = ConvTasNet(n_src=1, sample_rate=SAMPLE_RATE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    loader = DataLoader(WavDataset(data_dir), batch_size=1, shuffle=False)
    for i, mix in enumerate(tqdm(loader, desc=desc)):
        mix = mix.to(DEVICE)
        with torch.no_grad():
            est_sph = model(mix).squeeze().cpu().numpy()
        sf.write(os.path.join(est_dir, f"{i:06d}_sph_est.wav"), est_sph, SAMPLE_RATE)
    print(f"{desc} 完成。")


def main():
    # ---- test1 ----
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ShipsEar", "test1")
    EST_DIR  = os.path.join(PROJECT_ROOT, "experiments", "conv_tasnet", "estimates", "tt_test1")
    run_inference(DATA_DIR, EST_DIR, desc="推理 test1")

    # ---- DeepShip ----
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "DeepShip", "test")
    EST_DIR  = os.path.join(PROJECT_ROOT, "experiments", "conv_tasnet", "estimates", "tt_DeepShip")
    run_inference(DATA_DIR, EST_DIR, desc="推理 DeepShip")


if __name__ == "__main__":
    main()