#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   timestep_embedding.py
@Time    :   2023/05/22 22:54:29
@Author  :   Wenbo Li
@Desc    :   Time step embedding implement
"""

from abc import abstractmethod
from typing import Optional
import torch
import math
from torch import nn
# from .resnet2d import ResBlock,UpSample,DownSample
# from .transformer import SpatialTransformer,CrossAttention
from ..modules.transformer import SpatialTransformer, CrossAttention,AttnBlock
from ..modules.resnet2d import DownSample, ResBlock, UpSample


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, time_emb):
        pass

pytorch_device = torch.device('cuda:0')
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None
    ):
        """
        forward pass of TimestepEmbeddingSequential

        passed in layers are basically:
            `ResBlock`, `Conv2d`, `UpSample`,`DownSample`, `SpatialTransformer`
        apply different forward pass depending on instance type

        Args:
            - x (Tensor):
                  input shape=[batch, in_channels, height, width]
            - time_emb (Tensor):
                  input shape=[batch, time_emb_dim]
            - context (Tensor, optional):
                  input shape=[batch, seq_len, context_dim]. Default: None.

        Returns:
            - Tensor:
                - output shape:
                    - `ResBlock`:           [batch, out_channels, height, width]
                    - `SpatialTransformer`: [batch, out_channels, height, width]
                    - `Conv2d`:             [batch, out_channels, height, width]
                    - `UpSample`:           [batch, out_channels, height*scale_factor, width*scale_factor]
                    - `DownSample`:         [batch, out_channels, height/scale_factor, width/scale_factor]
        """
        for module in self:
            # pass ResBlock
            if isinstance(module, ResBlock):
                x = module(x, time_emb)
            # pass spatial transformer
            elif isinstance(module, SpatialTransformer):
                x = module(x,context)
            else:
                x = module(x)
        return x


def sinusoidal_time_proj(
    time_steps: torch.Tensor, emb_dim: int, max_len: int = 10000
) -> torch.Tensor:
    """
    sinusoidal time step embedding implementation

    Args:
        - time_steps (torch.Tensor):
                time step of shape [batch_size,]
        - emb_dim (int):
                embed dimension
        - max_len (int, optional):
                max len of embedding. Default: 10000.

    Returns:
        - torch.Tensor:
                time embedding of shape [batch, emb_dim]
    """
    # half is sin, the other half is cos
    half = emb_dim // 2
    freq = torch.exp(
        math.log(max_len)
        / (half)
        * torch.arange(
            0, end=half, dtype=torch.float32
        )  # get the position of each time step
    ).to(time_steps.device)
    # shape=[batch, 1]*[1, half] = [batch, half]
    # freq[None]: shape=[half] -> [1, half]
    # time_steps[:, None]: shape=[batch] -> [batch, 1]
    args = time_steps[:, None].float() * freq[None]
    # shape=[batch, emb_dim]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
