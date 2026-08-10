# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Faithful Inada transformer-family models for real-bogus classification."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from .types import InputMode


def convs(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    )


def convs_no_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


class DepthwiseConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=dim,
        )

    def forward(
        self, x: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class ImageProjection(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), height, width


class ModifiedMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DepthwiseConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(_init_transformer_weights)

    def forward(
        self, x: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(_init_transformer_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, channels = x.shape
        q = (
            self.q(x)
            .reshape(batch_size, token_count, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(x)
            .reshape(
                batch_size, token_count, 2, self.num_heads, self.head_dim
            )
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (
            torch.matmul(attn, v)
            .transpose(1, 2)
            .reshape(batch_size, token_count, channels)
        )
        x = self.proj(x)
        return self.proj_drop(x)


class RelativePositionAttention(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        dim: int,
        *,
        num_heads: int,
        pos_dim: int = 64,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.height = height
        self.width = width
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.pos_dim = pos_dim

        self.q = nn.Linear(dim + pos_dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim + pos_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.alpha = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.pos_embed = nn.Parameter(torch.zeros(1, height * width, pos_dim))

        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, height),
            torch.linspace(0.0, 1.0, width),
            indexing="ij",
        )
        coords = torch.stack((yy, xx), dim=-1).reshape(height * width, 2)
        distance = torch.cdist(
            coords.unsqueeze(0), coords.unsqueeze(0)
        ).unsqueeze(0)
        self.register_buffer("dist", distance, persistent=False)
        self.apply(_init_transformer_weights)

    def forward(
        self, x: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        del height, width
        batch_size, token_count, channels = x.shape
        x = torch.cat(
            (
                x,
                self.pos_embed.expand(batch_size, token_count, self.pos_dim),
            ),
            dim=-1,
        )
        q = (
            self.q(x)
            .reshape(batch_size, token_count, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(x)
            .reshape(
                batch_size, token_count, 2, self.num_heads, self.head_dim
            )
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = torch.exp(self.alpha * self.dist) * attn
        attn = self.attn_drop(attn)
        x = (
            torch.matmul(attn, v)
            .transpose(1, 2)
            .reshape(batch_size, token_count, channels)
        )
        x = self.proj(x)
        return self.proj_drop(x)


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        in_channels: int,
        *,
        n_heads: int = 4,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels % n_heads != 0:
            raise ValueError("in_channels must be divisible by n_heads")
        self.height = height
        self.width = width
        self.n_heads = n_heads
        self.head_dim = in_channels // n_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(in_channels, in_channels, bias=qkv_bias)
        self.kv = nn.Linear(2 * in_channels, 2 * in_channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.apply(_init_transformer_weights)

    def forward(self, src: torch.Tensor, tmp: torch.Tensor) -> torch.Tensor:
        batch_size = src.shape[0]
        kv_input = torch.cat((src, tmp), dim=2)
        q = (
            self.q(src)
            .reshape(
                batch_size,
                self.height * self.width,
                self.n_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(kv_input)
            .reshape(
                batch_size,
                self.height * self.width,
                2,
                self.n_heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).flatten(2)
        return self.drop(self.proj(x))


class TripletAttentionDecoder(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        in_channels: int,
        *,
        n_heads: int = 4,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels % n_heads != 0:
            raise ValueError("in_channels must be divisible by n_heads")
        self.height = height
        self.width = width
        self.n_heads = n_heads
        self.head_dim = in_channels // n_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(in_channels, in_channels, bias=qkv_bias)
        self.kv = nn.Linear(3 * in_channels, 2 * in_channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.apply(_init_transformer_weights)

    def forward(
        self,
        src: torch.Tensor,
        tmp: torch.Tensor,
        diff: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = src.shape[0]
        kv_input = torch.cat((src, tmp, diff), dim=2)
        q = (
            self.q(src)
            .reshape(
                batch_size,
                self.height * self.width,
                self.n_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(kv_input)
            .reshape(
                batch_size,
                self.height * self.width,
                2,
                self.n_heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).flatten(2)
        return self.drop(self.proj(x))


class AttentionDecoderBlock(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        in_channels: int,
        *,
        pos_dim: int = 16,
        n_heads: int = 4,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        output_dim = in_channels + pos_dim
        if output_dim % n_heads != 0:
            raise ValueError(
                "decoder output dim must be divisible by n_heads"
            )
        self.height = height
        self.width = width
        self.pos_dim = pos_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, height * width, pos_dim))
        self.src_norm_1 = nn.LayerNorm(output_dim)
        self.src_attn = SelfAttention(
            output_dim,
            num_heads=n_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_bias=qkv_bias,
        )
        self.src_norm_2 = nn.LayerNorm(output_dim)
        mlp_hidden_dim = int(output_dim * mlp_ratio)
        self.src_mlp = ModifiedMLP(
            output_dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )
        self.comb_src_norm_1 = nn.LayerNorm(output_dim)
        self.comb_tmp_norm_1 = nn.LayerNorm(output_dim)
        self.comb_attn = AttentionDecoder(
            height,
            width,
            output_dim,
            n_heads=n_heads,
            attn_drop=attn_drop,
            drop=drop,
            qkv_bias=qkv_bias,
        )
        self.comb_norm_2 = nn.LayerNorm(output_dim)
        self.comb_mlp = ModifiedMLP(
            output_dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )

    def forward(self, src: torch.Tensor, tmp: torch.Tensor) -> torch.Tensor:
        batch_size = src.shape[0]
        pos_embed = self.pos_embed.expand(
            batch_size, self.height * self.width, self.pos_dim
        )
        src = torch.cat((src, pos_embed), dim=2)
        tmp = torch.cat((tmp, pos_embed), dim=2)
        src = src + self.src_attn(self.src_norm_1(src))
        src = src + self.src_mlp(
            self.src_norm_2(src), self.height, self.width
        )
        x = src + self.comb_attn(
            self.comb_src_norm_1(src),
            self.comb_tmp_norm_1(tmp),
        )
        return x + self.comb_mlp(self.comb_norm_2(x), self.height, self.width)


class TripletAttentionDecoderBlock(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        in_channels: int,
        *,
        pos_dim: int = 16,
        n_heads: int = 4,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        output_dim = in_channels + pos_dim
        if output_dim % n_heads != 0:
            raise ValueError(
                "decoder output dim must be divisible by n_heads"
            )
        self.height = height
        self.width = width
        self.pos_dim = pos_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, height * width, pos_dim))
        self.src_norm_1 = nn.LayerNorm(output_dim)
        self.src_attn = SelfAttention(
            output_dim,
            num_heads=n_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_bias=qkv_bias,
        )
        self.src_norm_2 = nn.LayerNorm(output_dim)
        mlp_hidden_dim = int(output_dim * mlp_ratio)
        self.src_mlp = ModifiedMLP(
            output_dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )
        self.comb_src_norm_1 = nn.LayerNorm(output_dim)
        self.comb_tmp_norm_1 = nn.LayerNorm(output_dim)
        self.comb_diff_norm_1 = nn.LayerNorm(output_dim)
        self.comb_attn = TripletAttentionDecoder(
            height,
            width,
            output_dim,
            n_heads=n_heads,
            attn_drop=attn_drop,
            drop=drop,
            qkv_bias=qkv_bias,
        )
        self.comb_norm_2 = nn.LayerNorm(output_dim)
        self.comb_mlp = ModifiedMLP(
            output_dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )

    def forward(
        self,
        src: torch.Tensor,
        tmp: torch.Tensor,
        diff: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = src.shape[0]
        pos_embed = self.pos_embed.expand(
            batch_size, self.height * self.width, self.pos_dim
        )
        src = torch.cat((src, pos_embed), dim=2)
        tmp = torch.cat((tmp, pos_embed), dim=2)
        diff = torch.cat((diff, pos_embed), dim=2)
        src = src + self.src_attn(self.src_norm_1(src))
        src = src + self.src_mlp(
            self.src_norm_2(src), self.height, self.width
        )
        x = src + self.comb_attn(
            self.comb_src_norm_1(src),
            self.comb_tmp_norm_1(tmp),
            self.comb_diff_norm_1(diff),
        )
        return x + self.comb_mlp(self.comb_norm_2(x), self.height, self.width)


class EncoderTransformerBlock(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        dim: int,
        *,
        num_heads: int,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.norm_1 = nn.LayerNorm(dim)
        self.attn = RelativePositionAttention(
            height,
            width,
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm_2 = nn.LayerNorm(dim)
        self.mlp = ModifiedMLP(
            dim,
            hidden_features=int(dim * 4),
            drop=drop,
        )
        self.apply(_init_transformer_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), self.height, self.width)
        return x + self.mlp(self.norm_2(x), self.height, self.width)


class ExchangeBlock(nn.Module):
    def __init__(self, p_swap: int) -> None:
        super().__init__()
        if (
            isinstance(p_swap, bool)
            or not isinstance(p_swap, int)
            or p_swap <= 0
        ):
            raise ValueError("p_swap must be a positive integer")
        self.p_swap = p_swap

    def forward(self, *xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(xs) == 2:
            return self._forward_pair(xs[0], xs[1])
        if len(xs) == 3:
            return self._forward_triplet(xs[0], xs[1], xs[2])
        raise ValueError("ExchangeBlock supports only pair or triplet inputs")

    def _forward_pair(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, channels, _, _ = x1.shape
        exchange_mask = (
            torch.arange(channels, device=x1.device) % self.p_swap
        ) == 0
        exchange_mask = exchange_mask.view(1, channels, 1, 1)
        out_x1 = torch.where(exchange_mask, x2, x1)
        out_x2 = torch.where(exchange_mask, x1, x2)
        return out_x1, out_x2

    def _forward_triplet(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Faithful reproduction of the authors' reference ExchangeBlock
        # (reiwanol/nodiff triplet_nodiff_transformer.py @ 274da55): on
        # every p_swap-th channel it exchanges streams. This is intentionally
        # NOT a permutation -- mask_12 channels emit (x2, x1, x1) and mask_13
        # channels emit (x3, x3, x2), dropping one stream and duplicating
        # another, exactly as upstream. Do NOT "fix" this to a permutation
        # without confirming with the reference authors. test_exchange_block
        # asserts byte-parity with the reference formula.
        _, channels, _, _ = x1.shape
        idx = torch.arange(channels, device=x1.device)
        mask_12 = (
            ((idx % self.p_swap) == 0) & ((idx % (self.p_swap * 2)) != 0)
        ).view(1, channels, 1, 1)
        mask_13 = ((idx % (self.p_swap * 2)) == 0).view(1, channels, 1, 1)
        out_x1 = torch.where(mask_13, x3, torch.where(mask_12, x2, x1))
        out_x2 = torch.where(mask_13, x3, torch.where(mask_12, x1, x2))
        out_x3 = torch.where(mask_13, x2, torch.where(mask_12, x1, x3))
        return out_x1, out_x2, out_x3


class PairDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: list[int],
        hs: list[int],
        ws: list[int],
        decoder_embedding_dim: int,
        pos_dim: int,
        output_nc: int,
        attn_drop: float,
        drop: float,
    ) -> None:
        super().__init__()
        self.hs = hs
        self.ws = ws
        self.output_nc = output_nc
        self.attn_decoders = nn.ModuleList(
            [
                AttentionDecoderBlock(
                    hs[index],
                    ws[index],
                    in_channels[index],
                    pos_dim=pos_dim,
                    attn_drop=attn_drop,
                    drop=drop,
                )
                for index in range(len(in_channels))
            ]
        )
        self.intermediates = nn.ModuleList(
            [
                convs_no_relu(
                    in_channels[index] + pos_dim,
                    decoder_embedding_dim,
                )
                for index in range(len(in_channels))
            ]
        )
        self.conv_outs = nn.ModuleList(
            [
                convs(decoder_embedding_dim, output_nc)
                for _ in range(len(in_channels))
            ]
        )
        linear_in_features = output_nc * sum(
            height * width for height, width in zip(hs, ws)
        )
        self.lin_1 = nn.Linear(linear_in_features, 256)
        self.lin_2 = nn.Linear(256, 1)

    def forward(
        self,
        x1: list[torch.Tensor],
        x2: list[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = x1[-1].shape[0]
        flat_x1 = [x.flatten(2).transpose(1, 2) for x in x1]
        flat_x2 = [x.flatten(2).transpose(1, 2) for x in x2]
        outputs = []
        for index in range(len(flat_x1)):
            out = self.attn_decoders[index](flat_x1[index], flat_x2[index])
            out = out.permute(0, 2, 1).reshape(
                batch_size,
                -1,
                self.hs[index],
                self.ws[index],
            )
            out = self.intermediates[index](out)
            outputs.append(self.conv_outs[index](out))
        out = torch.cat(
            [torch.flatten(item, start_dim=1) for item in outputs],
            dim=1,
        )
        return self.lin_2(torch.relu(self.lin_1(out))).squeeze(1)


class TripletDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: list[int],
        hs: list[int],
        ws: list[int],
        decoder_embedding_dim: int,
        pos_dim: int,
        output_nc: int,
        attn_drop: float,
        drop: float,
    ) -> None:
        super().__init__()
        self.hs = hs
        self.ws = ws
        self.output_nc = output_nc
        self.attn_decoders = nn.ModuleList(
            [
                TripletAttentionDecoderBlock(
                    hs[index],
                    ws[index],
                    in_channels[index],
                    pos_dim=pos_dim,
                    attn_drop=attn_drop,
                    drop=drop,
                )
                for index in range(len(in_channels))
            ]
        )
        self.intermediates = nn.ModuleList(
            [
                convs_no_relu(
                    in_channels[index] + pos_dim,
                    decoder_embedding_dim,
                )
                for index in range(len(in_channels))
            ]
        )
        self.conv_outs = nn.ModuleList(
            [
                convs(decoder_embedding_dim, output_nc)
                for _ in range(len(in_channels))
            ]
        )
        linear_in_features = output_nc * sum(
            height * width for height, width in zip(hs, ws)
        )
        self.lin_1 = nn.Linear(linear_in_features, 256)
        self.lin_2 = nn.Linear(256, 1)

    def forward(
        self,
        x1: list[torch.Tensor],
        x2: list[torch.Tensor],
        x3: list[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = x1[-1].shape[0]
        flat_x1 = [x.flatten(2).transpose(1, 2) for x in x1]
        flat_x2 = [x.flatten(2).transpose(1, 2) for x in x2]
        flat_x3 = [x.flatten(2).transpose(1, 2) for x in x3]
        outputs = []
        for index in range(len(flat_x1)):
            out = self.attn_decoders[index](
                flat_x1[index],
                flat_x2[index],
                flat_x3[index],
            )
            out = out.permute(0, 2, 1).reshape(
                batch_size,
                -1,
                self.hs[index],
                self.ws[index],
            )
            out = self.intermediates[index](out)
            outputs.append(self.conv_outs[index](out))
        out = torch.cat(
            [torch.flatten(item, start_dim=1) for item in outputs],
            dim=1,
        )
        return self.lin_2(torch.relu(self.lin_1(out))).squeeze(1)


class PairEncoderTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        *,
        depths: list[int],
        embed_dims: list[int],
        num_heads: list[int],
        drop_rate: float,
        attn_drop: float,
        p_swap: int = 2,
    ) -> None:
        super().__init__()
        if not (len(depths) == len(embed_dims) == len(num_heads)):
            raise ValueError("depths, embed_dims, and num_heads must match")
        self.depths = depths
        self.embed_dims = [1] + list(embed_dims)
        self.stage_count = len(depths)
        kernel_sizes = [7] + [3 for _ in range(self.stage_count - 1)]
        strides = [2 for _ in range(self.stage_count // 2)] + [
            1 for _ in range(self.stage_count - self.stage_count // 2)
        ]
        self.img_projs = nn.ModuleList(
            [
                ImageProjection(
                    kernel_size=kernel_sizes[index],
                    stride=strides[index],
                    in_channels=self.embed_dims[index],
                    embed_dim=self.embed_dims[index + 1],
                )
                for index in range(self.stage_count)
            ]
        )
        height = width = image_size
        self.hs: list[int] = []
        self.ws: list[int] = []
        for index in range(self.stage_count):
            height = math.floor(
                (
                    height
                    + 2 * (kernel_sizes[index] // 2)
                    - kernel_sizes[index]
                )
                / strides[index]
                + 1
            )
            width = math.floor(
                (width + 2 * (kernel_sizes[index] // 2) - kernel_sizes[index])
                / strides[index]
                + 1
            )
            self.hs.append(height)
            self.ws.append(width)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        EncoderTransformerBlock(
                            self.hs[index],
                            self.ws[index],
                            self.embed_dims[index + 1],
                            num_heads=num_heads[index],
                            drop=drop_rate,
                            attn_drop=attn_drop,
                        )
                        for _ in range(depths[index])
                    ]
                )
                for index in range(self.stage_count)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.embed_dims[index + 1])
                for index in range(self.stage_count)
            ]
        )
        self.exchanges = nn.ModuleList(
            [ExchangeBlock(p_swap) for _ in range(self.stage_count)]
        )
        self.apply(_init_transformer_weights)

    def forward(
        self,
        src: torch.Tensor,
        tmp: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        src_outs: list[torch.Tensor] = []
        tmp_outs: list[torch.Tensor] = []
        x1 = src
        x2 = tmp
        batch_size = src.shape[0]
        for index in range(self.stage_count):
            x1, _, _ = self.img_projs[index](x1)
            x2, _, _ = self.img_projs[index](x2)
            for block in self.blocks[index]:
                x1 = block(x1)
                x2 = block(x2)
            x1 = self.norms[index](x1)
            x2 = self.norms[index](x2)
            x1 = (
                x1.reshape(
                    batch_size,
                    self.hs[index],
                    self.ws[index],
                    -1,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            x2 = (
                x2.reshape(
                    batch_size,
                    self.hs[index],
                    self.ws[index],
                    -1,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            x1, x2 = self.exchanges[index](x1, x2)
            src_outs.append(x1)
            tmp_outs.append(x2)
        return src_outs, tmp_outs


class TripletEncoderTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        *,
        depths: list[int],
        embed_dims: list[int],
        num_heads: list[int],
        drop_rate: float,
        attn_drop: float,
        p_swap: int = 2,
    ) -> None:
        super().__init__()
        if not (len(depths) == len(embed_dims) == len(num_heads)):
            raise ValueError("depths, embed_dims, and num_heads must match")
        self.depths = depths
        self.embed_dims = [1] + list(embed_dims)
        self.stage_count = len(depths)
        kernel_sizes = [7] + [3 for _ in range(self.stage_count - 1)]
        strides = [2 for _ in range(self.stage_count // 2)] + [
            1 for _ in range(self.stage_count - self.stage_count // 2)
        ]
        self.img_projs = nn.ModuleList(
            [
                ImageProjection(
                    kernel_size=kernel_sizes[index],
                    stride=strides[index],
                    in_channels=self.embed_dims[index],
                    embed_dim=self.embed_dims[index + 1],
                )
                for index in range(self.stage_count)
            ]
        )
        height = width = image_size
        self.hs: list[int] = []
        self.ws: list[int] = []
        for index in range(self.stage_count):
            height = math.floor(
                (
                    height
                    + 2 * (kernel_sizes[index] // 2)
                    - kernel_sizes[index]
                )
                / strides[index]
                + 1
            )
            width = math.floor(
                (width + 2 * (kernel_sizes[index] // 2) - kernel_sizes[index])
                / strides[index]
                + 1
            )
            self.hs.append(height)
            self.ws.append(width)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        EncoderTransformerBlock(
                            self.hs[index],
                            self.ws[index],
                            self.embed_dims[index + 1],
                            num_heads=num_heads[index],
                            drop=drop_rate,
                            attn_drop=attn_drop,
                        )
                        for _ in range(depths[index])
                    ]
                )
                for index in range(self.stage_count)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.embed_dims[index + 1])
                for index in range(self.stage_count)
            ]
        )
        self.exchanges = nn.ModuleList(
            [ExchangeBlock(p_swap) for _ in range(self.stage_count)]
        )
        self.apply(_init_transformer_weights)

    def forward(
        self,
        src: torch.Tensor,
        tmp: torch.Tensor,
        diff: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        src_outs: list[torch.Tensor] = []
        tmp_outs: list[torch.Tensor] = []
        diff_outs: list[torch.Tensor] = []
        x1 = src
        x2 = tmp
        x3 = diff
        batch_size = src.shape[0]
        for index in range(self.stage_count):
            x1, _, _ = self.img_projs[index](x1)
            x2, _, _ = self.img_projs[index](x2)
            x3, _, _ = self.img_projs[index](x3)
            for block in self.blocks[index]:
                x1 = block(x1)
                x2 = block(x2)
                x3 = block(x3)
            x1 = self.norms[index](x1)
            x2 = self.norms[index](x2)
            x3 = self.norms[index](x3)
            x1 = (
                x1.reshape(
                    batch_size,
                    self.hs[index],
                    self.ws[index],
                    -1,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            x2 = (
                x2.reshape(
                    batch_size,
                    self.hs[index],
                    self.ws[index],
                    -1,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            x3 = (
                x3.reshape(
                    batch_size,
                    self.hs[index],
                    self.ws[index],
                    -1,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            x1, x2, x3 = self.exchanges[index](x1, x2, x3)
            src_outs.append(x1)
            tmp_outs.append(x2)
            diff_outs.append(x3)
        return src_outs, tmp_outs, diff_outs


class NoDiffTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        *,
        depths: list[int],
        num_heads: list[int],
        embed_dims: list[int],
        decoder_embedding_dim: int,
        pos_dim: int,
        output_nc: int,
        drop_rate: float,
        attn_drop: float,
    ) -> None:
        super().__init__()
        self.encoder = PairEncoderTransformer(
            image_size,
            depths=depths,
            embed_dims=embed_dims,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop=attn_drop,
        )
        self.decoder = PairDecoder(
            in_channels=list(embed_dims),
            hs=self.encoder.hs,
            ws=self.encoder.ws,
            decoder_embedding_dim=decoder_embedding_dim,
            pos_dim=pos_dim,
            output_nc=output_nc,
            attn_drop=attn_drop,
            drop=drop_rate,
        )

    def forward(self, src: torch.Tensor, tmp: torch.Tensor) -> torch.Tensor:
        fx1, fx2 = self.encoder(src, tmp)
        return self.decoder(fx1, fx2)


class TripletNoDiffTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        *,
        depths: list[int],
        num_heads: list[int],
        embed_dims: list[int],
        decoder_embedding_dim: int,
        pos_dim: int,
        output_nc: int,
        drop_rate: float,
        attn_drop: float,
    ) -> None:
        super().__init__()
        self.encoder = TripletEncoderTransformer(
            image_size,
            depths=depths,
            embed_dims=embed_dims,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop=attn_drop,
        )
        self.decoder = TripletDecoder(
            in_channels=list(embed_dims),
            hs=self.encoder.hs,
            ws=self.encoder.ws,
            decoder_embedding_dim=decoder_embedding_dim,
            pos_dim=pos_dim,
            output_nc=output_nc,
            attn_drop=attn_drop,
            drop=drop_rate,
        )

    def forward(
        self,
        src: torch.Tensor,
        tmp: torch.Tensor,
        diff: torch.Tensor,
    ) -> torch.Tensor:
        fx1, fx2, fx3 = self.encoder(src, tmp, diff)
        return self.decoder(fx1, fx2, fx3)


class StackedInputRealBogusModel(nn.Module):
    def __init__(
        self,
        *,
        input_mode: InputMode,
        image_size: int,
        depths: list[int],
        num_heads: list[int],
        embed_dims: list[int],
        decoder_embedding_dim: int,
        pos_dim: int,
        output_nc: int,
        drop_rate: float,
        attn_drop: float,
        xfit_feature_names: list[str] | None = None,
        xfit_hidden_dim: int = 32,
        xfit_dropout: float = 0.0,
        xfit_modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_mode = input_mode
        self.xfit_feature_names = tuple(xfit_feature_names or ())
        if len(set(self.xfit_feature_names)) != len(self.xfit_feature_names):
            raise ValueError("xfit_feature_names must be unique")
        if self.xfit_feature_names and "fit_present" not in (
            self.xfit_feature_names
        ):
            raise ValueError(
                "xfit_feature_names must include the fit_present gate"
            )
        if xfit_hidden_dim <= 0:
            raise ValueError("xfit_hidden_dim must be positive")
        for name, value in (
            ("xfit_dropout", xfit_dropout),
            ("xfit_modality_dropout", xfit_modality_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must satisfy 0 <= value < 1")
        self.xfit_modality_dropout = float(xfit_modality_dropout)
        self._xfit_present_index = (
            self.xfit_feature_names.index("fit_present")
            if self.xfit_feature_names
            else None
        )
        if input_mode == "pair":
            self.inner = NoDiffTransformer(
                image_size,
                depths=depths,
                num_heads=num_heads,
                embed_dims=embed_dims,
                decoder_embedding_dim=decoder_embedding_dim,
                pos_dim=pos_dim,
                output_nc=output_nc,
                drop_rate=drop_rate,
                attn_drop=attn_drop,
            )
        elif input_mode == "triplet":
            self.inner = TripletNoDiffTransformer(
                image_size,
                depths=depths,
                num_heads=num_heads,
                embed_dims=embed_dims,
                decoder_embedding_dim=decoder_embedding_dim,
                pos_dim=pos_dim,
                output_nc=output_nc,
                drop_rate=drop_rate,
                attn_drop=attn_drop,
            )
        else:
            raise ValueError("input_mode must be 'pair' or 'triplet'")

        self.xfit_head: nn.Sequential | None = None
        if self.xfit_feature_names:
            self.xfit_head = nn.Sequential(
                nn.Linear(len(self.xfit_feature_names), xfit_hidden_dim),
                nn.GELU(),
                nn.Dropout(xfit_dropout),
                nn.Linear(xfit_hidden_dim, 1),
            )
            final = self.xfit_head[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        xfit_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("expected input tensor of shape [B, C, H, W]")
        if self.input_mode == "pair":
            if x.shape[1] != 2:
                raise ValueError("pair model expects 2-channel stacked input")
            image_logit = self.inner(x[:, 0:1], x[:, 1:2])
        else:
            if x.shape[1] != 3:
                raise ValueError(
                    "triplet model expects 3-channel stacked input"
                )
            image_logit = self.inner(x[:, 0:1], x[:, 1:2], x[:, 2:3])

        if self.xfit_head is None:
            if xfit_features is not None:
                raise ValueError(
                    "xFit features were supplied to an image-only model"
                )
            return image_logit
        if xfit_features is None:
            raise ValueError("this model requires xFit features")
        if xfit_features.ndim != 2:
            raise ValueError("xFit features must have shape [B, F]")
        expected_shape = (x.shape[0], len(self.xfit_feature_names))
        if tuple(xfit_features.shape) != expected_shape:
            raise ValueError(
                "xFit feature shape does not match the configured schema: "
                f"expected {expected_shape}, got "
                f"{tuple(xfit_features.shape)}"
            )
        assert self._xfit_present_index is not None
        gate = xfit_features[:, self._xfit_present_index]
        if self.training and self.xfit_modality_dropout > 0.0:
            keep = torch.rand_like(gate) >= self.xfit_modality_dropout
            gate = gate * keep.to(gate.dtype)
        fit_logit = self.xfit_head(xfit_features).squeeze(1)
        return image_logit + gate * fit_logit


def build_model(
    *,
    input_mode: InputMode,
    image_size: int,
    depths: list[int],
    num_heads: list[int],
    embed_dims: list[int],
    decoder_embedding_dim: int,
    pos_dim: int,
    output_nc: int,
    drop_rate: float,
    attn_drop: float,
    xfit_feature_names: list[str] | None = None,
    xfit_hidden_dim: int = 32,
    xfit_dropout: float = 0.0,
    xfit_modality_dropout: float = 0.0,
) -> StackedInputRealBogusModel:
    return StackedInputRealBogusModel(
        input_mode=input_mode,
        image_size=image_size,
        depths=list(depths),
        num_heads=list(num_heads),
        embed_dims=list(embed_dims),
        decoder_embedding_dim=decoder_embedding_dim,
        pos_dim=pos_dim,
        output_nc=output_nc,
        drop_rate=drop_rate,
        attn_drop=attn_drop,
        xfit_feature_names=xfit_feature_names,
        xfit_hidden_dim=xfit_hidden_dim,
        xfit_dropout=xfit_dropout,
        xfit_modality_dropout=xfit_modality_dropout,
    )


def _init_transformer_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv2d):
        fan_out = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.out_channels
        )
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()


__all__ = [
    "NoDiffTransformer",
    "TripletNoDiffTransformer",
    "StackedInputRealBogusModel",
    "build_model",
]
