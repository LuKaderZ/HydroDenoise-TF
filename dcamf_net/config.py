import os
import torch
import torchaudio
import numpy as np
import random
from tqdm import tqdm
from thop import profile, clever_format
from fast_bss_eval import sdr as fast_bss_sdr
try:
    from model import DCAMFNet
except ImportError:
    from dcamf_net.model import DCAMFNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(args, device):
    model = DCAMFNet(
        in_channels=1,
        enc_channels=args.enc_channels,
        enc_kernel_size=args.enc_kernel_size,
        enc_stride=args.enc_stride,
        chunk_size=args.chunk_size,
        hop_size=args.hop_size,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        ffn_hidden=args.ffn_hidden,
        dw_kernel_size=args.dw_kernel_size,
    ).to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if "total_ops" not in k and "total_params" not in k
        }
        model.load_state_dict(filtered_state_dict)
        print(f"[INFO] 已加载模型权重: {args.checkpoint}")

    return model


def print_complexity(model, sample_rate, device):
    model.eval()
    dummy_input = torch.randn(1, 1, int(sample_rate * 3.0)).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops_str, params_str = clever_format([flops, params], "%.2f")
    print(f"\n{'='*40}\nParams: {params_str}\nFLOPs:  {flops_str}\n{'='*40}")


def _compute_sisnr(estimate, target):
    estimate = estimate - np.mean(estimate)
    target = target - np.mean(target)
    dot = np.sum(estimate * target)
    s_target = dot * target / (np.sum(target**2) + 1e-8)
    e_noise = estimate - s_target
    return 10.0 * np.log10(np.sum(s_target**2) / (np.sum(e_noise**2) + 1e-8) + 1e-8)


def _compute_sdr(estimate, target):
    est_t = torch.from_numpy(estimate).unsqueeze(0).float()
    ref_t = torch.from_numpy(target).unsqueeze(0).float()
    return float(fast_bss_sdr(ref_t, est_t).item())


def _load_audio(filepath, sample_rate):
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform


@torch.no_grad()
def run_evaluation(model, test_dir, sample_rate, device, output_dir=None):
    """
    核心评估函数：处理所有计算，返回指标字典
    """
    noisy_dir = os.path.join(test_dir, "noisy")
    clean_dir = os.path.join(test_dir, "clean")

    exts = {".wav", ".flac", ".ogg", ".mp3"}
    fnames = sorted(
        [f for f in os.listdir(noisy_dir) if os.path.splitext(f)[1].lower() in exts]
    )

    if not fnames:
        return None

    model.eval()
    results = {"sisnr_b": [], "sisnr_a": [], "sdr_b": [], "sdr_a": []}

    # 确定降噪音频保存路径
    denoised_root = output_dir if output_dir else os.path.join(test_dir, 'denoised')
    os.makedirs(denoised_root, exist_ok=True)

    for fname in tqdm(
        fnames, desc=f"Testing {os.path.basename(test_dir)}", unit="file"
    ):
        clean_path = os.path.join(clean_dir, fname)
        if not os.path.exists(clean_path):
            continue

        # 读取与对齐
        nw = _load_audio(os.path.join(noisy_dir, fname), sample_rate)
        cw = _load_audio(clean_path, sample_rate)
        min_len = min(nw.shape[-1], cw.shape[-1])
        nw, cw = nw[:, :min_len], cw[:, :min_len]

        # 模型推理
        est = model(nw.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        n_np, c_np = nw.squeeze(0).numpy(), cw.squeeze(0).numpy()
        e_np = est.squeeze(0)

        # 保存降噪
        torchaudio.save(
            os.path.join(denoised_root, fname), torch.from_numpy(e_np), sample_rate
        )

        # 指标存入列表
        results["sisnr_b"].append(_compute_sisnr(n_np, c_np))
        results["sisnr_a"].append(_compute_sisnr(e_np, c_np))
        results["sdr_b"].append(_compute_sdr(n_np, c_np))
        results["sdr_a"].append(_compute_sdr(e_np, c_np))

    # 聚合指标
    final_metrics = {
        "count": len(results["sisnr_b"]),
        "SISNR": np.mean(results["sisnr_a"]),
        "SISNRi": np.mean(np.array(results["sisnr_a"]) - np.array(results["sisnr_b"])),
        "SDR": np.mean(results["sdr_a"]),
        "SDRi": np.mean(np.array(results["sdr_a"]) - np.array(results["sdr_b"])),
    }
    return final_metrics
