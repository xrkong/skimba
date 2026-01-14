#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2023/05/15 12:55:11
@Author  :   Wenbo Li
@Desc    :   Util class for build model blocks
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from torch.nn import ModuleList
from ..modules.timestep_embedding import TimestepEmbedSequential
from ..modules.resnet2d import DownSample, ResBlock, UpSample

from ..modules.transformer import SpatialTransformer, CrossAttention,AttnBlock
import spconv.pytorch as spconv


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def build_conv_in(in_channels: int, out_channels: int):
    return nn.Conv3d(in_channels, out_channels, 3, padding=1)

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def build_input_blocks(
    in_channels: int,
    num_res_blocks: int,
    attention_resolutions: Optional[list] = None,
    n_heads: int = 1,
    n_layers: int = 1,
    dropout: float = 0.0,
    levels: int = 1,
    time_emb_dim: Optional[int] = None,
    channels_list: list = None,
    groups: int = 4,
):
    input_blocks = nn.ModuleList([])
    # num of channels at each block
    # will use reversly in output blocks
    input_block_channels = [in_channels]
    # * add levels input blocks(down sample layers)
    in_ch = in_channels
    # @ note: in labmlai, they use level in attn_resolutions but in origin stable diffusion paper, they use ds = [1,2,4,...],total num levels
    attn_mult = 1
    d_head = None
    for level in range(levels):
        for _ in range(num_res_blocks):
            # Residual block maps from previous number of channels to the number of
            # channels in the current level
            out_ch = channels_list[level]
            layers = [
                ResBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    time_emb_dim=time_emb_dim,
                    groups=groups,
                )
            ]
            in_ch = (
                out_ch  # this level's output channels is next level's input channels
            )
            # * add attention layers
            if attention_resolutions is not None and attn_mult in attention_resolutions:
                d_head = in_ch // n_heads
                layers.append(
                    SpatialTransformer(
                        in_ch,
                        n_heads=n_heads,
                        d_head=d_head,
                        n_layers=n_layers,
                        dropout=dropout,
                        groups=groups,
                    )
                )
            input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_channels.append(in_ch)
        # * add DownSample block except last level
        if level != levels - 1:
            input_blocks.append(TimestepEmbedSequential(DownSample(in_ch)))
            # add additional in_channels for downsample
            input_block_channels.append(in_ch)
            attn_mult *= (
                2  # double it for next itr, this won't be multiplied in the last layer
            )
    return input_blocks, input_block_channels, in_ch, d_head, attn_mult


def build_bottleneck(
    in_ch: int,
    time_emb_dim: Optional[int] = None,
    n_heads: int = 1,
    d_head: int = 1,
    n_layers: int = 1,
    dropout: float = 0.0,
    use_attn_only: bool = True,
    groups: int = 4,
):
    return TimestepEmbedSequential(
        ResBlock(in_channels=in_ch, time_emb_dim=time_emb_dim, dropout=dropout),
        (
            # CrossAttention(
            #     query_dim=in_ch, n_heads=n_heads, d_head=d_head, dropout=dropout
            # )
            CrossAttention(
                query_dim=in_ch, n_heads=n_heads, d_head=d_head, dropout=dropout
            )
            if use_attn_only
            else SpatialTransformer(
                in_ch,
                n_heads=n_heads,
                d_head=d_head,
                n_layers=n_layers,
                dropout=dropout,
                groups=groups,
            )
        ),
        ResBlock(
            in_channels=in_ch, time_emb_dim=time_emb_dim, dropout=dropout, groups=groups
        ),
    )


def build_output_blocks(
    num_res_blocks: int,
    attention_resolutions: Optional[list] = None,
    n_heads: int = 1,
    n_layers: int = 1,
    dropout: float = 0.0,
    input_block_channels: Optional[ModuleList] = None,
    levels: int = 1,
    time_emb_dim: Optional[int] = None,
    channels_list: list = None,
    in_ch: int = 1,
    attn_mult: Optional[int] = None,
    groups: int = 4,
):
    output_blocks = nn.ModuleList([])
    for level in reversed(range(levels)):
        # # * add resblocks
        # if input_block_channels:
        #     channel_select = input_block_channels.pop()
        #     input_block_channels.pop()
        #     input_block_channels.pop()
        # else:
        #     channel_select = 0
        for res_block in range(num_res_blocks+1): # num_res_blocks + 1
            # Residual block maps from previous number of channels to the number of
            # channels in the current level
            # * note: here should add input_block_channels.pop() because input blocks are used as skip connection
            out_ch = channels_list[level]
            layers = [
                ResBlock(
                    # in_channels=in_ch + channel_select,
                    in_channels=in_ch + input_block_channels.pop() if input_block_channels else in_ch,
                    out_channels=out_ch,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                    groups=groups,
                )
            ]
            in_ch = out_ch
            # * add attention layers
            if attention_resolutions is not None and attn_mult in attention_resolutions:
                d_head = in_ch // n_heads
                layers.append(
                    SpatialTransformer(
                        in_ch,
                        n_heads=n_heads,
                        d_head=d_head,
                        n_layers=n_layers,
                        dropout=dropout,
                        groups=groups,
                    )
                )
            # *add UpSample except the last one, note that in reversed order, level==0 is the last
            if level != 0 and res_block == num_res_blocks:
                # layers.append(TimestepEmbedSequential(UpSample(in_ch)))
                layers.append(UpSample(in_ch))
                if attn_mult:
                    attn_mult //= 2
            output_blocks.append(TimestepEmbedSequential(*layers))
    return output_blocks, in_ch


def build_final_output(out_ch: int, out_channels: int, groups: int = 4):
    # return nn.Sequential(
    #     nn.GroupNorm(groups, out_ch),
    #     nn.SiLU(),
    #     conv3x3(
    #         in_planes=out_ch, out_planes=out_channels
    #     ),
    # )
    return nn.Sequential(
        nn.GroupNorm(groups, out_ch),
        nn.SiLU(),
        nn.Conv3d(
            in_channels=out_ch, out_channels=out_channels, kernel_size=3, padding=1
        ),
    )


if __name__ == "__main__":
    batch = 10
    image_size = 64
    in_channels = 3
    channels = 128
    out_channels = 3
    tim_emb_dim = 512
    x = np.random.randn(batch, in_channels, image_size, image_size)
    x = torch.from_numpy(x).float()
    # test upsample
    upsample = UpSample(
        in_channels=channels,
    )
    up = upsample(x)
    print(up.shape)
    # test downsample
    downsample = DownSample(
        in_channels=channels,
    )
    down = downsample(x)
    print(down.shape)
    # test resblock
    resblock = ResBlock(
        in_channels=channels, out_channels=out_channels, time_emb_dim=tim_emb_dim
    )
    res = resblock(x, time_emb=torch.ones(size=(batch, tim_emb_dim)))
    print(res.shape)
