import os
import math
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm
import soundfile as sf
import numpy as np

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


class AudioDenoisingDataset(Dataset):
    """
    针对 DCAMF-Net 模型优化的水下声学去噪数据集
    """

    def __init__(
        self, root_dir, sample_rate=16000, segment_seconds=3.0, overlap_seconds=1.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment_seconds)  # 48000 (3s)
        self.hop_len = int(
            sample_rate * (segment_seconds - overlap_seconds)
        )  # 32000 (2s)

        self.noisy_dir = os.path.join(root_dir, "noisy")
        self.clean_dir = os.path.join(root_dir, "clean")

        noisy_files = sorted(
            [
                f
                for f in os.listdir(self.noisy_dir)
                if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
            ]
        )

        self.pairs = []

        print(f"[Dataset] 正在扫描 {root_dir} 中的有效音频对...")
        for fname in noisy_files:
            n_path = os.path.join(self.noisy_dir, fname)
            c_path = os.path.join(self.clean_dir, fname)
            if os.path.isfile(c_path):
                self.pairs.append((n_path, c_path))

        if not self.pairs:
            raise RuntimeError(f"在 {root_dir} 中未找到匹配的 noisy/clean 音频对")

        self.segments = []
        for pair_idx, (n_path, _) in enumerate(tqdm(self.pairs, desc="预查音频长度")):
            info = sf.info(n_path)
            orig_sr = info.samplerate
            num_frames = info.frames
            n_samples_at_target = int(num_frames * self.sample_rate / orig_sr)

            if n_samples_at_target <= self.segment_len:
                n_segs = 1
            else:
                n_segs = (
                    math.ceil((n_samples_at_target - self.segment_len) / self.hop_len)
                    + 1
                )

            for i in range(n_segs):
                start_at_target = i * self.hop_len
                start_in_orig = int(start_at_target * orig_sr / self.sample_rate)
                self.segments.append((pair_idx, start_in_orig))

        print(
            f"[Dataset] 加载完成: {len(self.pairs)} 条原始音频 -> {len(self.segments)} 个 3s 片段"
        )

    def __len__(self):
        return len(self.segments)

    def _load_segment(self, path, start_frame):
        info = sf.info(path)
        orig_sr = info.samplerate
        num_frames_to_read = int(self.segment_len * orig_sr / self.sample_rate)

        try:
            audio, _ = sf.read(
                path, start=start_frame, frames=num_frames_to_read, dtype="float32"
            )
        except Exception as e:
            raise RuntimeError(f"读取音频失败: {path}，错误: {e}")

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        else:
            audio = audio.T

        waveform = torch.from_numpy(audio.copy())

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[-1] < self.segment_len:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_len - waveform.shape[-1])
            )
        else:
            waveform = waveform[:, : self.segment_len]

        return waveform

    def __getitem__(self, idx):
        pair_idx, start_frame = self.segments[idx]
        noisy_path, clean_path = self.pairs[pair_idx]

        noisy_seg = self._load_segment(noisy_path, start_frame)
        clean_seg = self._load_segment(clean_path, start_frame)

        return noisy_seg, clean_seg
