"""
DCAMF-Net 消融变体1：移除全局支路
论文表4-2：移除全局支路
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def segment(x, chunk_size, hop_size):
    B, C, T = x.shape
    n_chunks = math.ceil((T - chunk_size) / hop_size) + 1
    pad_len = (n_chunks - 1) * hop_size + chunk_size - T
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))
    x = x.unfold(2, chunk_size, hop_size)
    x = x.permute(0, 1, 3, 2)
    return x


def overlap_add(x, chunk_size, hop_size, original_len):
    B, C, K, S = x.shape
    x = x.permute(0, 1, 3, 2).contiguous()
    T_out = (S - 1) * hop_size + chunk_size
    output = x.new_zeros(B, C, T_out)
    norm = x.new_zeros(1, 1, T_out)
    for i in range(S):
        start = i * hop_size
        output[:, :, start : start + K] += x[:, :, i, :]
        norm[:, :, start : start + K] += 1.0
    output = output / norm.clamp(min=1.0)
    return output[:, :, :original_len]


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=256,
        kernel_size=80,
        stride=40,
        chunk_size=500,
        hop_size=250,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.prelu = nn.PReLU()
        self.layer_norm = nn.LayerNorm(out_channels)
        self.chunk_size = chunk_size
        self.hop_size = hop_size

    def forward(self, x):
        W_e = self.conv(x)
        W_e = self.prelu(W_e)
        W_e = self.layer_norm(W_e.transpose(1, 2)).transpose(1, 2)
        T_enc = W_e.shape[-1]
        W = segment(W_e, self.chunk_size, self.hop_size)
        return W, W_e, T_enc


class ConvDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, kernel_size=80, stride=40):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(self, x):
        return self.deconv(x)


class ConvEnhancedMHSA(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dw_kernel_size=31, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ln_mhsa = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.pw_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1, bias=True)
        padding = (dw_kernel_size - 1) // 2
        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=dw_kernel_size,
            padding=padding,
            groups=d_model,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.swish_beta = nn.Parameter(torch.ones(1))
        self.pw_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        attn_out, _ = self.mhsa(x, x, x, attn_mask=attn_mask)
        attn_out = self.dropout(attn_out)
        x_res = self.ln_mhsa(attn_out + x)
        h = x_res.transpose(1, 2)
        h = self.pw_conv1(h)
        h = F.glu(h, dim=1)
        h = self.dw_conv(h)
        h = self.bn(h)
        h = h * torch.sigmoid(self.swish_beta * h)
        h = self.pw_conv2(h)
        h = self.dropout(h)
        W_conv = h.transpose(1, 2)
        return W_conv


class ImprovedFFN(nn.Module):
    def __init__(self, d_model=256, hidden_size=512, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(d_model, hidden_size, batch_first=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.linear = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.gru(x)
        h = self.leaky_relu(h)
        h = self.linear(h)
        h = self.dropout(h)
        return h


class DCAMBlock(nn.Module):
    """消融变体：只保留局部分支，移除全局支路"""

    def __init__(
        self, d_model=256, n_heads=4, ffn_hidden=512, dw_kernel_size=31, dropout=0.1
    ):
        super().__init__()
        self.local_cemhsa = ConvEnhancedMHSA(d_model, n_heads, dw_kernel_size, dropout)
        self.local_ffn = ImprovedFFN(d_model, ffn_hidden, dropout)

        self.mask_conv = nn.Sequential(
            nn.PReLU(), nn.Conv1d(d_model, d_model, kernel_size=1, bias=True), nn.Tanh()
        )
        self.mask_gate = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, W):
        B, F, K, S = W.shape

        W_local_in = W.permute(0, 3, 2, 1).contiguous().view(B * S, K, F)
        W_local = self.local_cemhsa(W_local_in)
        W_local = self.local_ffn(W_local)
        W_local = W_local.view(B, S, K, F).permute(0, 3, 2, 1)

        W_out = W + 0.5 * W_local

        W_for_mask = W_out.permute(0, 2, 3, 1).reshape(B, K * S, F)
        W_for_mask = W_for_mask.transpose(1, 2)
        mask_conv = self.mask_conv(W_for_mask)
        mask_gate = self.mask_gate(W_for_mask)
        mask_flat = mask_conv * mask_gate
        mask = mask_flat.view(B, F, K, S)

        return W_out, mask


class DCAMFNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        enc_channels=256,
        enc_kernel_size=80,
        enc_stride=40,
        chunk_size=500,
        hop_size=250,
        n_blocks=6,
        n_heads=8,
        ffn_hidden=512,
        dw_kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = ConvEncoder(
            in_channels, enc_channels, enc_kernel_size, enc_stride, chunk_size, hop_size
        )
        self.dcam_blocks = nn.ModuleList(
            [
                DCAMBlock(enc_channels, n_heads, ffn_hidden, dw_kernel_size, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.decoder = ConvDecoder(
            enc_channels, in_channels, enc_kernel_size, enc_stride
        )
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_blocks = n_blocks
        self.mask_fusion_weights = nn.Parameter(torch.ones(n_blocks) / n_blocks)

    def forward(self, x):
        T_orig = x.shape[-1]
        W, W_e, T_enc = self.encoder(x)
        masks_chunk = []
        for block in self.dcam_blocks:
            W, mask = block(W)
            masks_chunk.append(mask)
        masks_time = []
        for mask_chunk in masks_chunk:
            mask_t = overlap_add(mask_chunk, self.chunk_size, self.hop_size, T_enc)
            masks_time.append(mask_t)
        masks_stack = torch.stack(masks_time, dim=1)
        weights = F.softmax(self.mask_fusion_weights, dim=0)
        final_mask = torch.einsum("b n f t, n -> b f t", masks_stack, weights)
        W_noise = W_e * final_mask
        n_e = self.decoder(W_noise)
        T_dec = n_e.shape[-1]
        if T_dec > T_orig:
            n_e = n_e[:, :, :T_orig]
        elif T_dec < T_orig:
            n_e = F.pad(n_e, (0, T_orig - T_dec))
        s_e = x - n_e
        return s_e
