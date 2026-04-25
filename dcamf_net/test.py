import argparse
import torch
from config import set_seed, build_model, print_complexity, run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="DCAMF-Net 水声降噪评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument(
        "--test_dir", type=str, nargs="+", required=True, help="一个或多个测试集路径"
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--hop_size", type=int, default=250)
    parser.add_argument("--enc_channels", type=int, default=256)
    parser.add_argument("--enc_kernel_size", type=int, default=80)
    parser.add_argument("--enc_stride", type=int, default=40)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ffn_hidden", type=int, default=512)
    parser.add_argument("--dw_kernel_size", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None, help='降噪音频保存目录')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args, device)
    print_complexity(model, args.sample_rate, device)

    for t_dir in args.test_dir:
        metrics = run_evaluation(model, t_dir, args.sample_rate, device, output_dir=args.output_dir)
        if metrics:
            print(f"\n>> 数据集: {t_dir}")
            print(f"   样本数: {metrics['count']}")
            print(
                f"   SISNR:  {metrics['SISNR']:.4f} dB (SISNRi: {metrics['SISNRi']:.4f})"
            )
            print(f"   SDR:    {metrics['SDR']:.4f} dB (SDRi: {metrics['SDRi']:.4f})")
        else:
            print(f"\n[!] 警告: 目录 {t_dir} 中未发现有效音频对。")


if __name__ == "__main__":
    import sys

    ckpt = "../experiments/dcamf_net/checkpoints/best_SISNR.pth"
    test_dirs = [
        ("../data/ShipsEar/train", "ShipsEar_train"),
        ("../data/ShipsEar/test1", "ShipsEar_test1"),
        ("../data/ShipsEar/test2", "ShipsEar_test2"),
        ("../data/ShipsEar/test3", "ShipsEar_test3"),
        ("../data/DeepShip/test",  "DeepShip_test"),
    ]
    for test_dir, subname in test_dirs:
        output_dir = f"../experiments/dcamf_net/denoised/{subname}"
        sys.argv = ["test.py", "--checkpoint", ckpt, "--test_dir", test_dir, "--output_dir", output_dir]
        main()