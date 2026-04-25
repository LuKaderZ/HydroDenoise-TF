#!/usr/bin/env python3
"""
DCAMF-Net 数据准备脚本
功能：从 raw_data/ 加载原始音频，按论文设置生成训练/测试数据到 data/ 目录。
路径说明：脚本位于 scripts/, 项目根目录为上级目录, raw_data/ 和 data/ 均在根目录下。
"""

import os
import random
import math
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
from tqdm import tqdm

# ------------------------ 路径配置 ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
OUTPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 音频参数
SAMPLE_RATE = 16000
SEGMENT_SEC = 3.0
OVERLAP_SEC = 1.0
TAIL_KEEP_THRESHOLD = 0.95  # 丢弃尾部不足95%的片段

# 训练信噪比范围（三组区间）
SNR_RANGES = [(-15.0, -10.0), (-10.0, -5.0), (-5.0, 0.0)]

# 验证集划分比例
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# 重采样质量
RESAMPLE_QUALITY = 5.0

# 支持的音频扩展名
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


# ------------------------ 工具函数 ------------------------
def collect_audio_files(directory):
    """递归收集目录下所有支持的音频文件"""
    files = []
    if not os.path.isdir(directory):
        return files
    for root, _, fnames in sorted(os.walk(directory)):
        for f in sorted(fnames):
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(root, f))
    return files


def load_and_resample(filepath, sr=16000, keep_stereo=False):
    """加载音频并重采样到目标采样率，转单声道（可选）"""
    orig_sr, data = wavfile.read(filepath)
    # 类型归一化
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 1.0:
            data = np.clip(data, -1.0, 1.0)
    else:
        raise ValueError(f"不支持的音频类型: {data.dtype}")
    # 立体声处理
    if data.ndim > 1:
        if not keep_stereo:
            data = data.mean(axis=1)
        else:
            # 保持立体声 (samples, channels)
            pass
    # 重采样
    if orig_sr != sr:
        gcd = math.gcd(sr, orig_sr)
        up = sr // gcd
        down = orig_sr // gcd
        if data.ndim == 1:
            data = scipy.signal.resample_poly(
                data, up, down, window=("kaiser", RESAMPLE_QUALITY)
            )
        else:
            resampled_channels = []
            for ch in range(data.shape[1]):
                ch_data = scipy.signal.resample_poly(
                    data[:, ch], up, down, window=("kaiser", RESAMPLE_QUALITY)
                )
                resampled_channels.append(ch_data)
            data = np.stack(resampled_channels, axis=1)
        data = data.astype(np.float32)
    return data


def segment_audio(
    waveform,
    sr,
    segment_sec=SEGMENT_SEC,
    overlap_sec=OVERLAP_SEC,
    keep_threshold=TAIL_KEEP_THRESHOLD,
):
    """切割音频为固定长度片段，丢弃尾部不足阈值的部分"""
    segment_len = int(sr * segment_sec)
    hop_len = int(sr * (segment_sec - overlap_sec))
    segments = []
    start = 0
    if waveform.ndim == 1:
        while start + segment_len <= len(waveform):
            segments.append(waveform[start : start + segment_len])
            start += hop_len
        remaining = len(waveform) - start
        if remaining >= segment_len * keep_threshold:
            segments.append(waveform[start : start + segment_len])
    else:
        while start + segment_len <= waveform.shape[0]:
            segments.append(waveform[start : start + segment_len, :])
            start += hop_len
        remaining = waveform.shape[0] - start
        if remaining >= segment_len * keep_threshold:
            segments.append(waveform[start : start + segment_len, :])
    return segments


def mix_at_snr(clean, noise, target_snr_db):
    """按目标 SNR 混合信号，同步缩放避免截幅"""
    T = clean.shape[0] if clean.ndim == 1 else clean.shape[0]
    # 噪声长度对齐
    if noise.ndim == 1:
        if len(noise) < T:
            reps = math.ceil(T / len(noise))
            noise = np.tile(noise, reps)
        if len(noise) > T:
            start = random.randint(0, len(noise) - T)
            noise = noise[start : start + T]
    else:
        if noise.shape[0] < T:
            reps = math.ceil(T / noise.shape[0])
            noise = np.tile(noise, (reps, 1))
        if noise.shape[0] > T:
            start = random.randint(0, noise.shape[0] - T)
            noise = noise[start : start + T, :]
    # 维度匹配
    if clean.ndim == 1 and noise.ndim == 2:
        clean = np.tile(clean[:, np.newaxis], (1, noise.shape[1]))
    elif clean.ndim == 2 and noise.ndim == 1:
        noise = np.tile(noise[:, np.newaxis], (1, clean.shape[1]))
    P_s = np.mean(clean**2) + 1e-10
    P_n = np.mean(noise**2) + 1e-10
    alpha = math.sqrt(P_s / (P_n * (10.0 ** (target_snr_db / 10.0))))
    noisy = clean + alpha * noise
    max_amp = np.max(np.abs(noisy))
    if max_amp > 1.0:
        scale = 1.0 / max_amp
        noisy = noisy * scale
        clean = clean * scale
    return clean.astype(np.float32), noisy.astype(np.float32)


def save_wav(filepath, waveform, sr=16000):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    waveform = np.clip(waveform, -1.0, 1.0)
    wavfile.write(filepath, sr, waveform.astype(np.float32))


# ------------------------ ShipsEar 数据生成 ------------------------
class ShipsEarPipeline:
    def __init__(self, raw_base, output_base):
        self.sr = SAMPLE_RATE
        raw_dir = os.path.join(raw_base, "ShipsEar")
        self.output_dir = os.path.join(output_base, "ShipsEar")
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        print("\n" + "=" * 60)
        print("  ShipsEar Pipeline — 加载原始音频")
        print("=" * 60)

        self.passenger_segs = self._load_category(raw_dir, "passenger")
        self.roro_segs = self._load_category(raw_dir, "roro")
        self.motorboat_segs = self._load_category(raw_dir, "motorboat")
        self.wind_segs = self._load_category(raw_dir, "wind")
        self.flow_segs = self._load_category(raw_dir, "flow")
        self.reservoir_segs = self._load_category(raw_dir, "reservoir")

        # 干净信号池（客船 + 滚装船）
        self.train_clean_pool = self.passenger_segs + self.roro_segs
        # 训练噪声池（仅风噪 + 水流噪，不含水库）
        self.train_noise_pool = self.wind_segs + self.flow_segs
        # 未知噪声池（水库噪）
        self.unseen_noise_pool = self.reservoir_segs

        self._check_pool(self.train_clean_pool, "干净信号 (passenger+roro)")
        self._check_pool(self.train_noise_pool, "训练噪声 (wind+flow)")
        self._check_pool(self.motorboat_segs, "摩托艇 (test2)")
        self._check_pool(self.unseen_noise_pool, "水库噪声 (test3)")

        print(f"  干净信号: {len(self.train_clean_pool)} 片段")
        print(f"  训练噪声: {len(self.train_noise_pool)} 片段")
        print(f"  未知噪声: {len(self.unseen_noise_pool)} 片段")
        print(f"  摩托艇:   {len(self.motorboat_segs)} 片段")

    def _load_category(self, base_dir, subdir):
        cat_dir = os.path.join(base_dir, subdir)
        files = collect_audio_files(cat_dir)
        print(f"  [{subdir:12s}] {len(files)} 文件")
        all_segs = []
        for f in files:
            wav = load_and_resample(f, self.sr, keep_stereo=False)
            segs = segment_audio(wav, self.sr)
            all_segs.extend(segs)
        print(f"  [{subdir:12s}] → {len(all_segs)} 片段")
        return all_segs

    def _check_pool(self, pool, name):
        if not pool:
            raise RuntimeError(f"[错误] 数据池为空: {name}")

    def generate(self):
        # 随机划分训练/测试
        n_total = len(self.train_clean_pool)
        indices = list(range(n_total))
        random.shuffle(indices)
        n_train = int(n_total * 0.8)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        test1_clean = [self.train_clean_pool[i] for i in test_idx]

        # 生成训练集（自动划分验证集在后续训练脚本中进行）
        train_dir = os.path.join(self.output_dir, "train")
        print(f"\n--- 生成训练集 ({n_train} 对, SNR 三组区间) ---")
        for i, idx in enumerate(tqdm(train_idx, desc="训练集")):
            # 从三个区间随机选择一个区间，再均匀采样
            chosen_range = random.choice(SNR_RANGES)
            snr = random.uniform(*chosen_range)
            noise_seg = random.choice(self.train_noise_pool)
            c, n = mix_at_snr(self.train_clean_pool[idx], noise_seg, snr)
            fname = f"{i:06d}.wav"
            save_wav(os.path.join(train_dir, "clean", fname), c, self.sr)
            save_wav(os.path.join(train_dir, "noisy", fname), n, self.sr)

        # 测试集1：已知船型 + 已知噪声
        self._gen_test("test1", test1_clean, self.train_noise_pool)

        # 测试集2：未知船型 (摩托艇) + 已知噪声
        self._gen_test("test2", self.motorboat_segs, self.train_noise_pool)

        # 测试集3：已知船型 + 未知噪声 (水库)
        self._gen_test("test3", test1_clean, self.unseen_noise_pool)

    def _gen_test(self, name, clean_pool, noise_pool):
        """生成一个测试集，固定三个 SNR: -15, -10, -5 dB"""
        print(f"\n--- 生成测试集 {name} ---")
        test_dir = os.path.join(self.output_dir, name)
        count = 0
        for target_snr in [-15, -10, -5]:
            for clean_seg in tqdm(clean_pool, desc=f"  {name} SNR={target_snr} dB"):
                noise_seg = random.choice(noise_pool)
                c, n = mix_at_snr(clean_seg, noise_seg, target_snr)
                fname = f"{count:06d}.wav"
                save_wav(os.path.join(test_dir, "clean", fname), c, self.sr)
                save_wav(os.path.join(test_dir, "noisy", fname), n, self.sr)
                count += 1
        print(f"  保存 {count} 对音频")


# ------------------------ DeepShip 数据生成 ------------------------
class DeepShipPipeline:
    def __init__(self, raw_base, output_base, noise_pool):
        self.sr = SAMPLE_RATE
        raw_dir = os.path.join(raw_base, "DeepShip")
        self.output_dir = os.path.join(output_base, "DeepShip", "test")
        self.noise_pool = noise_pool  # 使用传入的噪声池（所有 ShipsEar 噪声）
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        print("\n" + "=" * 60)
        print("  DeepShip Pipeline — 加载原始音频")
        print("=" * 60)

        self.all_ships = []
        ship_classes = ["Cargo", "Tug", "Passengership", "Tanker"]
        for cls in ship_classes:
            cls_dir = os.path.join(raw_dir, cls)
            files = collect_audio_files(cls_dir)
            print(f"  [{cls:14s}] {len(files)} 文件")
            for f in files:
                wav = load_and_resample(f, self.sr, keep_stereo=False)
                segs = segment_audio(wav, self.sr)
                self.all_ships.extend(segs)
            print(f"  [{cls:14s}] → 累计 {len(self.all_ships)} 片段")

        if not self.all_ships:
            raise RuntimeError("未加载到 DeepShip 片段！")

    def generate(self):
        target_minutes = 40.0
        target_segs = int(target_minutes * 60 / SEGMENT_SEC)
        if len(self.all_ships) < target_segs:
            print(f"  片段不足 ({len(self.all_ships)}), 有放回采样")
            selected = random.choices(self.all_ships, k=target_segs)
        else:
            selected = random.sample(self.all_ships, target_segs)

        print(f"\n--- 生成 DeepShip 测试集 ({len(selected)} 对, SNR 0~10 dB) ---")
        for i, clean_seg in enumerate(tqdm(selected, desc="DeepShip")):
            snr = random.uniform(0, 10)
            noise_seg = random.choice(self.noise_pool)
            c, n = mix_at_snr(clean_seg, noise_seg, snr)
            fname = f"{i:06d}.wav"
            save_wav(os.path.join(self.output_dir, "clean", fname), c, self.sr)
            save_wav(os.path.join(self.output_dir, "noisy", fname), n, self.sr)

        print(f"  保存 {len(selected)} 对音频")


# ------------------------ 主函数 ------------------------
def main():
    print("=" * 60)
    print("  DCAMF-Net 数据准备")
    print(f"  原始数据: {RAW_DATA_DIR}")
    print(f"  输出路径: {OUTPUT_DATA_DIR}")
    print("=" * 60)

    # 1. ShipsEar 数据集
    ship_pipe = ShipsEarPipeline(RAW_DATA_DIR, OUTPUT_DATA_DIR)
    ship_pipe.generate()

    # 收集所有噪声（风、流、水库）用于 DeepShip 混合
    all_noise = ship_pipe.wind_segs + ship_pipe.flow_segs + ship_pipe.reservoir_segs

    # 2. DeepShip 数据集
    deep_pipe = DeepShipPipeline(RAW_DATA_DIR, OUTPUT_DATA_DIR, all_noise)
    deep_pipe.generate()

    print("\n" + "=" * 60)
    print("  数据准备完成！")
    print(f"  输出目录: {os.path.abspath(OUTPUT_DATA_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
