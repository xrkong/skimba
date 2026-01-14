#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   attention.py
@Time    :   2023/05/14 16:03:43
@Author  :   Wenbo Li
@Desc    :   Transformer module for Stable Diffusion U-Net
"""

from typing import Optional
import torch
from torch import nn
from einops import rearrange
import spconv.pytorch as spconv

from ..models import utils
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=0, bias=False, indice_key=indice_key) # padding=1

class AttnBlock(nn.Module):
    """
    ## Attention block
    """

    def __init__(self, query_dim: int,
        n_heads: int = 1,
        d_head: int = 1,
        dropout: float = 0.0,
        groups: int=32):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        d_model = n_heads * d_head
        self.n_heads = n_heads
        # Group normalization
        # self.norm = nn.GroupNorm(query_dim, d_model, eps=1e-6)
        self.norm = nn.GroupNorm(groups, query_dim)
        # Query, key and value mappings
        self.q = nn.Conv3d(query_dim, d_model, 1)
        self.k = nn.Conv3d(query_dim, d_model, 1)
        self.v = nn.Conv3d(query_dim, d_model, 1)
        # Final $1 \times 1$ convolution layer
        self.proj_out = nn.Conv3d(d_model, query_dim, 1)
        # Attention scaling factor
        self.scale = query_dim ** -0.5

    def forward(self, x: torch.Tensor):
        """
        :param x: is the tensor of shape `[batch_size, channels, height, width]`
        """
        # Normalize `x`
        x_norm = self.norm(x)
        # Get query, key and vector embeddings
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # Reshape to query, key and vector embeedings from
        # `[batch_size, channels, height, width]` to
        # `[batch_size, channels, height * width]`
        b, c, h, w,d = q.shape
        q = q.view(b, c, h * w *d)
        k = k.view(b, c, h * w *d)
        v = v.view(b, c, h * w *d)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$
        out = torch.einsum('bij,bcj->bci', attn, v)

        # Reshape back to `[batch_size, channels, height, width]`
        out = out.view(b, c, h, w, d)
        # Final $1 \times 1$ convolution layer
        out = self.proj_out(out)

        # Add residual connection
        return x + out

class CrossAttention(nn.Module):
    """
    Cross attention module for transformer

    Architecture:
        - Q, K, V = Linear(x), Linear(context), Linear(context)
        - sim = Q * K^T / sqrt(dk)
        - attn = softmax(sim) * V
        - out = Linear(attn)

    Args:
        - query_dim (int):
                query dimension
        - context_dim (Optional[int]):
                context dimension, if not previded, equal to query_dim
        - n_heads (int):
                num of heads
        - d_head (int):
                dim of each head
        - dropout (float, optional):
                dropout rate. Default: `0.`.
    """

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 1,
        d_head: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_model = n_heads * d_head
        self.n_heads = n_heads
        self.scale = 1 / (d_head**0.5)  # 1 / sqrt(d_k) from paper
        self.to_q = nn.Linear(query_dim, d_model, bias=False)
        self.to_k = nn.Linear(query_dim, d_model, bias=False)
        self.to_v = nn.Linear(query_dim, d_model, bias=False)
        self.out = nn.Sequential(nn.Linear(d_model, query_dim), nn.Dropout(dropout))

    def forward(
        self,
        query: torch.Tensor,
        context_emb: torch.Tensor = None,
        mask: torch.Tensor = None
    ):
        """
        Cross attention forward pass
        - first calculate similarity between query and context embedding: sim = Q * K^T / sqrt(dk)
        - then calculate attention value: attn = softmax(sim) * V
        - finally, linear projection and dropout

        Args:
            - query (torch.Tensor):
                  feature map of shape `[batch, height*width, query_dim]`
                  height*width is equivalent to tgt_len in origin transformer
            - context_emb (torch.Tensor, optional):
                  conditional embeddings of shape `[batch, seq_len, context_dim]`. Default: `None`.
                  seq_len is equivalent to src_len in origin transformer
            - mask (torch.Tensor, optional):
                  mask of shape = `[batch, height*width, seq_len]`. Default: `None`.
                  actually never used...
        """
        # if no context_emb, equal to self-attention
        # when use cross attn without wrapped spatial transformer, convert to [batch, height*width, d_model] first
        convert = len(query.shape) == 5 # 4
        B, C, H, W, D = query.shape
        if convert:
            h = query.shape[2]
            query = rearrange(query, "b c h w d -> b (h w d) c")
        if context_emb is None:
            context_emb = query
        else:
            context_emb = rearrange(context_emb, "b c h w d -> b (h w d) c")
        Q, K, V = self.to_q(query), self.to_k(context_emb), self.to_v(context_emb)
        # q: [batch, h*w, d_model] -> [batch * n_head, h*w, d_head]
        # k,v: [batch, seq_len, d_model] -> [batch * n_head, seq_len, d_head]
        Q, K, V = map(
            lambda t: rearrange(
                t, "b n (n_heads d_head) -> (b n_heads) n d_head", n_heads=self.n_heads
            ),
            (Q, K, V),
        )
        # similarity = Q*K^T / sqrt(dk): [batch * n_head, h*w, d_head] * [batch * n_head, d_head, seq_len] -> [batch * n_head, h*w, seq_len]
        sim = torch.einsum("b n d,b m d->b n m", Q, K) * self.scale
        if mask is not None:
            # repeat for each head
            # mask: [batch, height*width, seq_len] -> [batch * n_head, height*width, seq_len]
            mask = mask.repeat("b n m, b h n m", h=self.n_heads)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(mask, max_neg_value)
        # softmax
        attn = sim.softmax(dim=-1)
        # attn value = attn*V: [batch * n_head, h*w, seq_len] * [batch * n_head, seq_len, d_head] -> [batch * n_head, h*w, d_head]
        attn_v = torch.einsum("b n m,b m d->b n d", attn, V)
        attn_v = rearrange(
            attn_v, "(b n_heads) n d_head -> b n (n_heads d_head)", n_heads=self.n_heads
        )
        out = self.out(attn_v)
        # convert it back to [batch, channels, height, width]
        if convert:
            out = rearrange(out, "b d c -> b c d")
            out = out.view(B,C,H,W,D)
        return out


class FeedForward(nn.Module):
    """
    origin paper use linear-relu-dropout-linear: `FFN(x) = max(0, xW1 + b1)W2 + b2`, equation(2) from Attention is all you need(https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

    here we use FFN(x) = `Dropout(GEGLU(x))*W + b`, where `GEGLU(x) = (xW + b) * GELU(xV + c)`

    Architecture:
        - GEGLU(x) = (xW + b) * GELU(xV + c)
        - Dropout(GEGLU(x))
        - Linear(Dropout(GEGLU(x)))

    Args:
        - d_model (int):
                d_model from transformer paper
        - dim_mult (int, optional):
                multiplicative factor for the hidden layer size. Default: `4`.
        - dropout (float, optional):
                dropout rate. Default: `0.`.
    Returns:
    TODO: x shape and output shape
        - Tensor:
                _description_
    """

    def __init__(self, d_model: int, dim_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            GEGLU(d_model, d_model * dim_mult),
            nn.Dropout(p=dropout),
            nn.Linear(d_model * dim_mult, d_model), #d_model * dim_mult
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W, D = x.shape
        x = self.net(x)
        x = rearrange(x, "b c h -> b h c")
        x = x.view(B,C,H,W,D)
        return x


class GEGLU(nn.Module):
    """
    GeGLU(x) = (xW + b) * GELU(xV + c) from paper: https://arxiv.org/abs/2002.05202

    Architecture:
        - xW + b, xV + c = Linear(x)
        - out = (xW + b) * GELU(xV + c)

    Args:
        - in_features (int):
                input feature dimension
        - out_features (int):
                output feature dimension
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # xW + b and xV + c all together
        self.proj = nn.Linear(in_features, out_features * 2)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        """
        forward pass of GEGLU

        Args:
            - x (torch.Tensor):
                  shape=`[]`

        Returns:
            - torch.Tensor:
                  shape=`[]`
        """
        B,C,H,W,D = x.shape
        x1 = rearrange(x, "b c h w d -> b (h w d) c")
        a = self.proj(x1)
        x, gate = self.proj(x1).chunk(2, dim=-1) # dim=-1
        return x * self.gelu(gate)


class BasicTransformerBlock(nn.Module):
    """
    Basic Transformer Block

    Architecture:
        - self attention = `CrossAttention(context=None)`
        - add norm
        - cross attention = `CrossAttention(context)`
        -add norm
        - feed forward = `FeedForward`
        - add norm

    Args:
        - d_model (int):
                dim of embedding
        - n_heads (int):
                num of heads
        - d_head (int):
                dim of each head
        - dropout (float, optional):
                dropout rate. Default: `0.`.
        - context_dim (int, optional):
                dim of conditional context. Default: `768`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attn = CrossAttention(
            d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.BatchNorm3d(d_model) # nn.LayerNorm()
        self.cross_attn = CrossAttention(
            d_model,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
        )
        self.norm2 = nn.BatchNorm3d(d_model) # nn.LayerNorm()
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.norm3 = nn.BatchNorm3d(d_model) # nn.LayerNorm()

    def forward(self, x: torch.Tensor=None, context_emb: torch.Tensor=None):
        """
        forward pass of BasicTransformerBlock

        Args:
            - x (torch.Tensor):
                   input embeddings of shape `[batch_size, height * width, d_model]`

            - context (torch.Tensor, optional):
                  conditional embeddings of shape `[batch_size,  seq_len, context_dim]`

        Returns:
            - torch.Tensor:
                  x with attention and skip connection and normalization, shape=`[batch_size, height * width, d_model]`
        """
        # check params

        assert (
            x.shape[1] == self.d_model
        ), f"input dim {x.shape[1]} should be equal to d_model {self.d_model}"
        # self attention
        x = self.norm1(x + self.self_attn(x, context_emb=None))
        # cross attention
        x = self.norm2(x + self.cross_attn(x, context_emb=context_emb)) # context_emb
        # feed forward
        x = self.norm3(x + self.ffn(x))
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.

    Architecture:
        - norm = GroupNorm
        - proj_in = Conv2d
        - transformer_blocks = [BasicTransformerBlock] * n_layers
        - proj_out = Conv2d

    Args:
        - in_channels (int):
            input num of channels in the feature map
        - n_heads (int):
            num of attention heads
        - d_head (int):
            dim of each head
        - n_layer (int, optional):
            num of transformer block. Default: `1`.
        - dropout (float, optional):
            dropout rate. Default: `0.`.
        - context_dim (int, optional):
            dim of context condition. Default: `None`.
        - groups (int, optional):
            num of groups for GroupNorm. Default: `2`.
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        groups: int = 2,
    ):
        super().__init__()
        # check params
        assert (
            n_heads > 0
        ), f"n_heads({n_heads}) should be greater than 0 for SpatialTransformer"
        assert (
            in_channels % groups == 0
        ), f"in_channels({in_channels}) should be divisible by num_groups({groups}) for GroupNorm"
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(groups, in_channels)
        self.proj_in = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        # Transformer layers
        # @ note: origin openai code use inner_dim = n_heads * d_head, but if legacy, d_head = in_channels // n_heads
        # => here we use in_channels for simiplicity
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    in_channels,
                    n_heads,
                    d_head,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.proj_out = utils.zero_module(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, context_emb: torch.Tensor = None) -> torch.Tensor:
        """
        forward pass

        Args:
            - x (torch.Tensor):
                  feature map of shape `[batch_size, channels, height, width]`
            - context_emb (torch.Tensor, optional):
                  conditional embeddings of shape `[batch_size,  seq_len, context_dim]`. Default: `None`.

        Returns:
            - torch.Tensor:
                  shape=`[batch_size, channels, height, width]`
        """
        # check params
        a = x.shape[1]
        b = x.shape[0]
        assert (
            x.shape[1] == self.in_channels
        ), f"input channels {x.shape[1]} should be equal to in_channels {self.in_channels}"
        # use for skip connection
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        # x = rearrange(x, "b c h w -> b (h w) c")
        for module in self.transformer_blocks:
            x = module(x, context_emb=context_emb)
        # x = rearrange(x, "b (h w) c -> b c h w", h=x_in.shape[2])
        x = self.proj_out(x)
        return x + x_in


if __name__ == "__main__":
    x = torch.randn(2, 128, 32, 32)
    context = torch.randn(2, 10, 768)
    model = SpatialTransformer(128, 4, 32, n_layers=2, context_dim=768)
    y = model(x, context)
    print(y.shape)
