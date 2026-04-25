"""
r-nSISNR Loss Function for DCAMF-Net
=====================================
Dual-estimation loss that optimizes both noise estimation and clean signal recovery.

    r-nSISNR = SISNR(n_e, n) + SISNR(x - n_e, s)

Returned as *negative* so that minimizing the loss = maximizing SISNR.
"""

import torch
import torch.nn as nn


def sisnr(estimate, target, eps=1e-8):
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    s_target = dot * target / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    e_noise = estimate - s_target

    si_snr = 10 * torch.log10(
        torch.sum(s_target ** 2, dim=-1) /
        (torch.sum(e_noise ** 2, dim=-1) + eps) + eps
    )
    return si_snr


class RnSISNR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, noisy, clean, estimated_clean):
        true_noise = noisy - clean
        estimated_noise = noisy - estimated_clean

        sisnr_noise = sisnr(estimated_noise, true_noise)
        sisnr_signal = sisnr(estimated_clean, clean)

        r_nsisnr = sisnr_noise + sisnr_signal
        loss = -r_nsisnr.mean()
        return loss