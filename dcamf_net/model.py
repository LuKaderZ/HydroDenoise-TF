"""
DCAMF-Net: Dual-branch Convolution-enhanced Attention and Multi-layer Mask Fusion Network
===========================================================================================
End-to-end time-domain denoising network for underwater acoustic signals.

Multi-layer mask fusion with learnable weights.
Each DCAM block outputs a mask, and all masks are fused via learned weights.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Helper: Segmentation & Overlap-Add
# ==============================================================================


def segment(x, chunk_size, hop_size):
    """
    Split a 1-D feature sequence into overlapping chunks.

    Args:
        x: (B, F, T)
        chunk_size: K
        hop_size: hop

    Returns:
        (B, F, K, S)
    """
    B, C, T = x.shape
    n_chunks = math.ceil((T - chunk_size) / hop_size) + 1
    pad_len = (n_chunks - 1) * hop_size + chunk_size - T
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))
    x = x.unfold(2, chunk_size, hop_size)  # (B, C, S, K)
    x = x.permute(0, 1, 3, 2)  # (B, C, K, S)
    return x


def overlap_add(x, chunk_size, hop_size, original_len):
    """
    Reconstruct a 1-D feature sequence from overlapping chunks via overlap-add.

    Args:
        x: (B, F, K, S)
        chunk_size: K
        hop_size: hop
        original_len: target output length along time axis

    Returns:
        (B, F, T)
    """
    B, C, K, S = x.shape
    x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, S, K)
    T_out = (S - 1) * hop_size + chunk_size
    output = x.new_zeros(B, C, T_out)
    norm = x.new_zeros(1, 1, T_out)
    for i in range(S):
        start = i * hop_size
        output[:, :, start : start + K] += x[:, :, i, :]
        norm[:, :, start : start + K] += 1.0
    output = output / norm.clamp(min=1.0)
    return output[:, :, :original_len]


# ==============================================================================
# 1. Convolutional Encoder
# ==============================================================================


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder:
        W_e = LayerNorm(PReLU(Conv1d(x)))

    Then segment W_e into overlapping chunks of length K with hop K//2.
    """

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
        """
        Args:
            x: (B, 1, T)
        Returns:
            W:  (B, F, K, S)  segmented features
            W_e: (B, F, T')   encoder features
            T_enc: int         T' length
        """
        W_e = self.conv(x)  # (B, F, T')
        W_e = self.prelu(W_e)
        W_e = self.layer_norm(W_e.transpose(1, 2)).transpose(1, 2)  # (B, F, T')
        T_enc = W_e.shape[-1]
        W = segment(W_e, self.chunk_size, self.hop_size)  # (B, F, K, S)
        return W, W_e, T_enc


# ==============================================================================
# 2. Convolution-Enhanced Multi-Head Self-Attention (CE-MHSA)
# ==============================================================================


class ConvEnhancedMHSA(nn.Module):
    """
    Convolution-Enhanced MHSA module.
    """

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

    def forward(self, x, attn_mask=None, return_attention=False):
        B, L, D = x.shape

        attn_out, attn_weights = self.mhsa(x, x, x, attn_mask=attn_mask)
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
        if return_attention:
            return W_conv, attn_weights
        return W_conv


# ==============================================================================
# 3. Improved FFN
# ==============================================================================


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


# ==============================================================================
# 4. DCAM Block (dual-branch with mask generation)
# ==============================================================================


class DCAMBlock(nn.Module):
    """
    DCAM Block with dual-branch architecture.
    Outputs:
        W_out: (B, F, K, S)  processed features
        mask:  (B, F, K, S)  mask generated from this block's output
    """

    def __init__(
        self, d_model=256, n_heads=4, ffn_hidden=512, dw_kernel_size=31, dropout=0.1
    ):
        super().__init__()

        self.global_cemhsa = ConvEnhancedMHSA(d_model, n_heads, dw_kernel_size, dropout)
        self.global_ffn = ImprovedFFN(d_model, ffn_hidden, dropout)

        self.local_cemhsa = ConvEnhancedMHSA(d_model, n_heads, dw_kernel_size, dropout)
        self.local_ffn = ImprovedFFN(d_model, ffn_hidden, dropout)

        # --- Mask generation branch (applied on W_out) ---
        self.mask_conv = nn.Sequential(
            nn.PReLU(), nn.Conv1d(d_model, d_model, kernel_size=1, bias=True), nn.Tanh()
        )
        self.mask_gate = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, W, return_attention=False):
        B, F, K, S = W.shape

        # Global branch
        W_global_in = W.permute(0, 2, 3, 1).contiguous().view(B * K, S, F)
        if return_attention:
            W_global, attn_global = self.global_cemhsa(W_global_in, return_attention=True)
        else:
            W_global = self.global_cemhsa(W_global_in)
        W_global = self.global_ffn(W_global)
        W_global = W_global.view(B, K, S, F).permute(0, 3, 1, 2)  # (B, F, K, S)

        # Local branch
        W_local_in = W.permute(0, 3, 2, 1).contiguous().view(B * S, K, F)
        if return_attention:
            W_local, attn_local = self.local_cemhsa(W_local_in, return_attention=True)
        else:
            W_local = self.local_cemhsa(W_local_in)
        W_local = self.local_ffn(W_local)
        W_local = W_local.view(B, S, K, F).permute(0, 3, 2, 1)  # (B, F, K, S)

        # Fusion
        W_out = W + 0.5 * W_local + 0.5 * W_global

        # Generate mask from W_out (in chunk domain)
        B, F, K, S = W_out.shape
        W_for_mask = W_out.permute(0, 2, 3, 1).reshape(B, K * S, F)  # (B, K*S, F)
        W_for_mask = W_for_mask.transpose(1, 2)  # (B, F, K*S)

        mask_conv = self.mask_conv(W_for_mask)  # (B, F, K*S)
        mask_gate = self.mask_gate(W_for_mask)  # (B, F, K*S)
        mask_flat = mask_conv * mask_gate  # (B, F, K*S)

        # Reshape back to (B, F, K, S)
        mask = mask_flat.view(B, F, K, S)

        if return_attention:
            return W_out, mask, {'global': attn_global, 'local': attn_local}
        return W_out, mask


# ==============================================================================
# 5. Transposed Convolutional Decoder
# ==============================================================================


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


# ==============================================================================
# 6. Top-level DCAMFNet Model (Multi-layer Mask Fusion)
# ==============================================================================


class DCAMFNet(nn.Module):
    """
    DCAMFNet with multi-layer mask fusion.

    Each DCAM block produces a mask. All masks are fused via learnable weights.
    """

    def __init__(
        self,
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

        # Learnable fusion weights for masks (initialized equally)
        self.mask_fusion_weights = nn.Parameter(torch.ones(n_blocks) / n_blocks)

    def forward(self, x):
        T_orig = x.shape[-1]

        # 1. Encode
        W, W_e, T_enc = self.encoder(x)  # W: (B,F,K,S), W_e: (B,F,T')

        # 2. Pass through DCAM blocks, collect features and masks
        masks_chunk = []  # list of (B, F, K, S)
        for block in self.dcam_blocks:
            W, mask = block(W)  # W updated, mask generated
            masks_chunk.append(mask)

        # 3. Convert each chunk mask to time-domain mask via overlap-add
        masks_time = []
        for mask_chunk in masks_chunk:
            mask_t = overlap_add(
                mask_chunk, self.chunk_size, self.hop_size, T_enc
            )  # (B, F, T')
            masks_time.append(mask_t)

        # 4. Learnable fusion of masks
        # Stack: (n_blocks, B, F, T') -> (B, n_blocks, F, T')
        masks_stack = torch.stack(masks_time, dim=1)
        weights = F.softmax(self.mask_fusion_weights, dim=0)  # (n_blocks,)
        # Weighted sum
        final_mask = torch.einsum("b n f t, n -> b f t", masks_stack, weights)

        # 5. Apply mask to encoder output (W_e) to get noise features
        W_noise = W_e * final_mask

        # 6. Decode to estimated noise
        n_e = self.decoder(W_noise)

        # 7. Residual for clean signal
        T_dec = n_e.shape[-1]
        if T_dec > T_orig:
            n_e = n_e[:, :, :T_orig]
        elif T_dec < T_orig:
            n_e = F.pad(n_e, (0, T_orig - T_dec))

        s_e = x - n_e
        return s_e


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    model = DCAMFNet(
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
    )

    x = torch.randn(2, 1, 16000)
    s_e = model(x)

    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {s_e.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(
        f"Learnable mask fusion weights (softmax): {F.softmax(model.mask_fusion_weights, dim=0).detach().cpu().numpy()}"
    )
    assert s_e.shape == x.shape, f"Shape mismatch! {s_e.shape} != {x.shape}"
    print("✓ Forward pass successful — shapes match.")
