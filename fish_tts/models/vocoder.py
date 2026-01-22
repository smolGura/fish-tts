"""DAC vocoder for Fish-TTS.

This module provides the DAC (Descript Audio Codec) based vocoder,
adapted from fish_speech for audio decoding from semantic codes.

Requires: dac, audiotools packages
"""

import math
import typing as tp
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools.ml import BaseModel
from dac.model.base import CodecMixin
from dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
from dac.nn.quantize import ResidualVectorQuantize
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class VQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    semantic_distill_z: torch.Tensor | None = None


@dataclass
class VocoderModelArgs:
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    dim: int = 512
    intermediate_size: int = 1536
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    channels_first: bool = True
    pos_embed_type: str = "rope"
    max_relative_position: int = 128

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return (
            k_out[:, :, : input_pos.max() + 1, :],
            v_out[:, :, : input_pos.max() + 1, :],
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-2,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class FeedForward(nn.Module):
    def __init__(self, config: VocoderModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class Attention(nn.Module):
    def __init__(self, config: VocoderModelArgs):
        super().__init__()
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.attn_dropout_rate = config.attn_dropout_rate
        self.pos_embed_type = config.pos_embed_type

        if self.pos_embed_type == "conformer":
            self.max_relative_position = config.max_relative_position
            num_pos_embeddings = 2 * config.max_relative_position + 1
            self.rel_pos_embeddings = nn.Parameter(
                torch.zeros(num_pos_embeddings, self.head_dim)
            )
            nn.init.normal_(self.rel_pos_embeddings, mean=0.0, std=0.02)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.pos_embed_type == "rope":
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_rate if self.training else 0.0,
            attn_mask=mask,
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)
        return self.wo(y)


class TransformerBlock(nn.Module):
    def __init__(self, config: VocoderModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_layer_scale = LayerScale(config.dim, inplace=True)
        self.ffn_layer_scale = LayerScale(config.dim, inplace=True)

    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
    ) -> Tensor:
        h = x + self.attention_layer_scale(
            self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        )
        out = h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: VocoderModelArgs) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if config.pos_embed_type == "rope":
            freqs_cis = precompute_freqs_cis(
                self.config.block_size, self.config.head_dim, self.config.rope_base
            )
            self.register_buffer("freqs_cis", freqs_cis)
        else:
            self.register_buffer("freqs_cis", None)

        causal_mask = torch.tril(
            torch.ones(self.config.block_size, self.config.block_size, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask)

        self.max_batch_size = -1
        self.max_seq_length = -1
        self.use_kv_cache = False

    def forward(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.config.pos_embed_type == "rope":
            freqs_cis = self.freqs_cis[input_pos]
        else:
            freqs_cis = None

        if mask is None:
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., : input_pos.max() + 1]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]

        for layer in self.layers:
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        return x


class WindowLimitedTransformer(Transformer):
    """Transformer with window limited attention."""

    def __init__(
        self,
        config: VocoderModelArgs,
        input_dim: int = 512,
        window_size: Optional[int] = None,
        causal: bool = True,
        look_ahead_conv: nn.Module = None,
    ):
        super().__init__(config)
        self.window_size = window_size
        self.causal = causal
        self.channels_first = config.channels_first
        self.look_ahead_conv = (
            look_ahead_conv if look_ahead_conv is not None else nn.Identity()
        )
        self.input_proj = (
            nn.Linear(input_dim, config.dim)
            if input_dim != config.dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(config.dim, input_dim)
            if input_dim != config.dim
            else nn.Identity()
        )

    def make_window_limited_mask(self, max_length: int) -> Tensor:
        mask = torch.tril(torch.ones(max_length, max_length))
        row_indices = torch.arange(max_length).view(-1, 1)
        window_size = self.window_size or max_length
        valid_range = (row_indices - window_size + 1).clamp(min=0)
        column_indices = torch.arange(max_length)
        mask = (column_indices >= valid_range) & mask.bool()
        return mask.bool()[None, None]

    def make_mask(self, max_length: int) -> Tensor:
        mask = torch.tril(torch.ones(max_length, max_length))
        return mask.bool()[None, None]

    def forward(self, x: Tensor, x_lens: Optional[Tensor] = None) -> Tensor:
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.look_ahead_conv(x)
        input_pos = torch.arange(x.shape[1], device=x.device)
        max_length = x.shape[1]
        if self.window_size is not None:
            mask = self.make_window_limited_mask(max_length)
        else:
            mask = self.make_mask(max_length)
        mask = mask.to(x.device)
        x = super().forward(x, input_pos, mask)
        x = self.output_proj(x)
        if self.channels_first:
            x = x.transpose(1, 2)
        return x


# Convolution utilities
def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    padding_left, padding_right = paddings
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zeros",
    value: float = 0.0,
):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class CausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        padding: int = None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def forward(self, x):
        pad = self.padding
        extra_padding = get_extra_padding_for_conv1d(
            x, self.kernel_size, self.stride, pad
        )
        x = pad1d(x, (pad, extra_padding), mode="constant", value=0)
        return self.conv(x).contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


class CausalTransConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        padding: int = None,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


def CausalWNConv1d(*args, **kwargs):
    return CausalConvNet(*args, **kwargs).weight_norm()


def CausalWNConvTranspose1d(*args, **kwargs):
    return CausalTransConvNet(*args, **kwargs).weight_norm()


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=1),
        )
        self.causal = causal

    def forward(self, x):
        y = self.block(x)
        pad = x.shape[-1] - y.shape[-1]
        if pad > 0:
            if self.causal:
                x = x[..., :-pad]
            else:
                x = x[..., pad // 2 : -pad // 2]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        transformer_module = (
            nn.Identity()
            if n_t_layer == 0
            else (
                WindowLimitedTransformer(
                    causal=causal,
                    input_dim=dim,
                    window_size=512,
                    config=transformer_general_config(
                        n_layer=n_t_layer,
                        n_head=dim // 64,
                        dim=dim,
                        intermediate_size=dim * 3,
                    ),
                )
            )
        )
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1, causal=causal),
            ResidualUnit(dim // 2, dilation=3, causal=causal),
            ResidualUnit(dim // 2, dilation=9, causal=causal),
            Snake1d(dim // 2),
            conv_class(
                dim // 2, dim,
                kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2),
            ),
            transformer_module,
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        n_transformer_layers: list = [0, 0, 4, 4],
        transformer_general_config: VocoderModelArgs = None,
        causal: bool = False,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        self.block = [conv_class(1, d_model, kernel_size=7, padding=3)]

        for stride, n_t_layer in zip(strides, n_transformer_layers):
            d_model *= 2
            self.block += [
                EncoderBlock(
                    d_model, stride=stride, causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                )
            ]

        self.block += [
            Snake1d(d_model),
            conv_class(d_model, d_latent, kernel_size=3, padding=1),
        ]

        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_trans_class = CausalWNConvTranspose1d if causal else WNConvTranspose1d
        self.block = nn.Sequential(
            Snake1d(input_dim),
            conv_trans_class(
                input_dim, output_dim,
                kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1, causal=causal),
            ResidualUnit(output_dim, dilation=3, causal=causal),
            ResidualUnit(output_dim, dilation=9, causal=causal),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: List[int],
        d_out: int = 1,
        causal: bool = False,
        n_transformer_layers: list = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        layers = [conv_class(input_channel, channels, kernel_size=7, padding=3)]

        for i, (stride, n_t_layer) in enumerate(zip(rates, n_transformer_layers)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [
                DecoderBlock(
                    input_dim, output_dim, stride, causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                )
            ]

        layers += [
            Snake1d(output_dim),
            conv_class(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# RVQ components
class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        self.dwconv = CausalConvNet(
            dim, dim, kernel_size=kernel_size, groups=dim, dilation=dilation,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, apply_residual: bool = True):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)
        if apply_residual:
            x = input + x
        return x


class DownsampleResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        n_codebooks: int = 9,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 4096,
        downsample_factor: tuple = (2, 2),
        downsample_dims: tuple = None,
        pre_module: nn.Module = None,
        post_module: nn.Module = None,
        semantic_predictor_module: nn.Module = None,
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        self.semantic_quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=1,
            codebook_size=semantic_codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=0.0,
        )

        self.quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    CausalConvNet(
                        all_dims[idx], all_dims[idx + 1],
                        kernel_size=factor, stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    CausalTransConvNet(
                        all_dims[idx + 1], all_dims[idx],
                        kernel_size=factor, stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )

        self.apply(self._init_weights)
        self.pre_module = pre_module if pre_module is not None else nn.Identity()
        self.post_module = post_module if post_module is not None else nn.Identity()
        self.semantic_predictor_module = (
            semantic_predictor_module
            if semantic_predictor_module is not None
            else nn.Identity()
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, z, n_quantizers: int = None, semantic_len: torch.Tensor = None, **kwargs
    ):
        original_shape = z.shape
        if semantic_len is None:
            semantic_len = torch.LongTensor([z.shape[-1]])
        z = self.downsample(z)
        z = self.pre_module(z)
        (
            semantic_z, semantic_codes, semantic_latents,
            semantic_commitment_loss, semantic_codebook_loss,
        ) = self.semantic_quantizer(z)
        residual_z = z - semantic_z
        residual_z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            residual_z, n_quantizers=n_quantizers
        )
        z = semantic_z + residual_z
        commitment_loss = commitment_loss + semantic_commitment_loss
        codebook_loss = codebook_loss + semantic_codebook_loss
        codes = torch.cat([semantic_codes, codes], dim=1)
        latents = torch.cat([semantic_latents, latents], dim=1)
        z = self.post_module(z)
        z = self.upsample(z)

        diff = original_shape[-1] - z.shape[-1]
        if diff > 0:
            z = F.pad(z, (0, diff))
        elif diff < 0:
            z = z[..., abs(diff):]

        return VQResult(
            z=z, codes=codes, latents=latents,
            commitment_loss=commitment_loss, codebook_loss=codebook_loss,
        )

    def decode(self, indices: torch.Tensor):
        new_indices = torch.zeros_like(indices)
        new_indices[:, 0] = torch.clamp(
            indices[:, 0], max=self.semantic_quantizer.codebook_size - 1
        )
        new_indices[:, 1:] = torch.clamp(
            indices[:, 1:], max=self.quantizer.codebook_size - 1
        )

        z_q_semantic = self.semantic_quantizer.from_codes(new_indices[:, :1])[0]
        z_q_residual = self.quantizer.from_codes(new_indices[:, 1:])[0]
        z_q = z_q_semantic + z_q_residual
        z_q = self.post_module(z_q)
        z_q = self.upsample(z_q)
        return z_q


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DAC(BaseModel, CodecMixin):
    """DAC vocoder model."""

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        quantizer: torch.nn.Module = None,
        sample_rate: int = 44100,
        causal: bool = True,
        encoder_transformer_layers: List[int] = [0, 0, 0, 0],
        decoder_transformer_layers: List[int] = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim, encoder_rates, latent_dim, causal=causal,
            n_transformer_layers=encoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )

        self.quantizer = quantizer

        self.decoder = Decoder(
            latent_dim, decoder_dim, decoder_rates, causal=causal,
            n_transformer_layers=decoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()
        self.frame_length = self.hop_length * 4

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        audio_lengths: torch.Tensor = None,
        n_quantizers: int = None,
        **kwargs,
    ):
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.frame_length) * self.frame_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        if audio_lengths is None:
            audio_lengths = torch.LongTensor([length + right_pad]).to(audio_data.device)

        z = self.encoder(audio_data)
        vq_results = self.quantizer(z, n_quantizers, **kwargs)
        indices = vq_results.codes
        indices_lens = torch.ceil(audio_lengths / self.frame_length).long()
        return indices, indices_lens

    def decode(self, indices: torch.Tensor, feature_lengths):
        if indices.ndim == 2:
            indices = indices[None]

        z = self.quantizer.decode(indices)
        audio_lengths = feature_lengths * self.frame_length
        return self.decoder(z), audio_lengths

    def forward(
        self,
        audio_data: torch.Tensor,
        template: torch.Tensor = None,
        mask: torch.Tensor = None,
        sample_rate: int = None,
        n_quantizers: int = None,
        **kwargs,
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        vq_results = self.encode(audio_data, n_quantizers, **kwargs)
        z = vq_results[0] if isinstance(vq_results, tuple) else vq_results.z
        x = self.decode(z)
        return x[..., :length], vq_results
