import torch.nn as nn
import numpy as np
import torch
from typing import Optional
import torch.nn.functional as F
from ..models import utils
import spconv.pytorch as spconv


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)

def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=0, bias=False, indice_key=indice_key)  #padding=1

class SparseDropout(torch.nn.Module):
    def init(self, dropout=0.5):
        super(SparseDropout, self).init()
        self.p = dropout

    def forward(self, sparse_tensor):
        mask = torch.rand_like(sparse_tensor.values()) > self.p
        values = sparse_tensor.values() * mask.float()
        new_sparse_tensor = torch.sparse.FloatTensor(sparse_tensor.indices(), values, sparse_tensor.size() )
        return new_sparse_tensor

class UpSample(nn.Module):
    """
    constructor for UpSample layer

    Architecture:
        - Interpolate
        - Conv2d

    Args:
        - in_channels (int):
                input  num of channels
        - out_channels (int, optional):
                output  num of channels
        - scale_factor (int, optional):
                Up Sample by a factor of `scale factor`. Default: `2`.
        - padding (int, optional):
                padding for `Conv2d`. Default: `1`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        scale_factor: float = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        # if has out_channels passed in, use it or use default = in_channels
        self.out_channels = out_channels or in_channels
        self.scale_factor = scale_factor
        # * default we use Conv2d, use_conv and conv_nd are not implemented for simpilicity
        self.conv = nn.Conv3d(
            self.in_channels, self.out_channels, kernel_size=3, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward Up Sample layer which

        Args:
            - x (Tensor):
                  input shape=`[batch, in_channels, height, width]`

        Returns:
            - Tensor:
                  output shape=`[batch, out_channels, height*scale_factor, width*scale_factor]`
        """
        assert (
            x.shape[1] == self.in_channels
        ), f"input channel does not match: x.shape[1]({x.shape[1]}) != self.in_channels({self.in_channels}))"
        # * e.g. [1,3,256,256] -> [1,3,512,512]
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    """
    constructor for DownSample layer

    Architecture:
        - Conv2d
        - Interpolate

    Args:
        - in_channels (int):
                input  num of channels
        - out_channels (int, optional):
                output  num of channels
        - scale_factor (int, optional):
                Down Sample by a factor of `scale factor`. Default: 1/2.
        - padding (int, optional):
                padding for Conv2d. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        scale_factor: float = 1 / 2,
        padding: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        # if has out_channels passed in, use it or use default = in_channels
        self.out_channels = out_channels or in_channels
        # It turn out that origin openai unet model did not use UpSample and DownSample,
        # instead, they use ResNetBlock with parameters up, down
        # Here we use UpSample and DownSample to make it clean
        # Change: => add stride=2, to downsample by a factor of 2
        # or use scale factor = 1/2
        # TODO: compare difference between stride=2 and scale_factor=1/2
        # self.conv = conv3x3(self.in_channels, self.out_channels)
        self.conv = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=3, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward Down Sample layer which reduce height and width of x by a factor of 2

        Args:
            - x (Tensor):
                  input shape=[batch, in_channels, height, width]

        Returns:
            - Tensor:
                  output shape=[batch, out_channels, height/scale_factor, width/scale_factor]
        """
        assert (
            x.shape[1] == self.in_channels
        ), f"input channel does not match: x.shape[1]({x.shape[1]}) != self.in_channels({self.in_channels}))"
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return x


class ResBlock(nn.Module):
    """
    ResBlock used in U-Net and AutoEncoder

    Archetecture::
        - in_layers = [`GroupNorm`, `SiLU`, `Conv2d`]
        - time_emb = [`SiLU`, `Linear`]
        - out_layers = [`GroupNorm`, `SiLU`, `Dropout`]

    Args:
        - in_channels (int):
              input num of channels
        - out_channels (Optional[int], optional):
              output num of channels. Default: equal to `in_channels`.
        - time_emb_dim (int, optional):
              time embedding dim. Default: `512`.
        - dropout (int, optional):
              dropout rate. Default: `0.`
        - padding (int, optional):
              padding idx. Default: `1`.
        - groups (int, optional):
              num of groups for `GroupNorm`. Default: `2`.
    """

    def  __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        time_emb_dim: Optional[int] = None,
        dropout: Optional[int] = 0,
        padding: Optional[int] = 1,
        groups: int = 2,
    ) -> None:
        super().__init__()
        # check parameters
        assert (
            in_channels % groups == 0
        ), f"in_channels({in_channels}) must be divisible by num_groups({groups})"
        self.in_channels = in_channels
        # if has out_channels passed in, use it or use default = in_channels
        self.out_channels = out_channels or in_channels
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            # nn.BatchNorm3d(self.in_channels, momentum=0.1),
            nn.GroupNorm(num_groups=groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv3d(
                self.in_channels, self.out_channels, kernel_size=3, padding=padding
            ),
        )
        # Time embedding
        if self.time_emb_dim:
            self.time_embedding = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_emb_dim, self.out_channels),
            )
        else:
            self.time_embedding = nn.Identity()

        self.out_layers = nn.Sequential(
            # nn.BatchNorm3d(self.out_channels, momentum=0.1),
            nn.GroupNorm(num_groups=groups, num_channels=self.out_channels),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            # ? openai model used zero_module(conv_nd)
            # TODO: figure out why zero_module is used
            # [batch, in_channel, height, width] => [batch, out_channel, height, height]
            utils.zero_module(
                nn.Conv3d(
                    self.out_channels, self.out_channels, kernel_size=3, padding=padding
                )
            ),
        )
        # Map input to output channel
        if self.in_channels != self.out_channels:
            self.skip_connection = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(
        self, x: torch.Tensor=None, time_emb: torch.Tensor=None) -> torch.Tensor:
        """
        forward pass of ResBlock

        Args:
            - x (torch.Tensor):
                  input shape = `[batch, in_channels, height, width]`
            - time_emb (torch.Tensor):
                  input shape = `[batch, time_emb_dim]`

        Returns:
            - torch.Tensor:
                  output shape = `[batch, out_channels, height, width]`
        """
        if time_emb is not None:
            assert (
                x.shape[0] == time_emb.shape[0]
            ), f"batch size does not match: x.shape[0]({x.shape[0]}) != time_emb.shape[0]({time_emb.shape[0]})"
            assert (
                time_emb.shape[1] == self.time_emb_dim
            ), f"time_emb_dim does not match: time_emb.shape[1]({time_emb.shape[1]}) != self.time_emb_dim({self.time_emb_dim})"
        # h: [batch, out_channels, height, width]
        h = self.in_layers(x)
        # [batch, time_emb_dim] => [batch, out_channels] => [batch, out_channels, 1, 1]
        if time_emb is not None:
            time_emb = self.time_embedding(time_emb)
            h += time_emb[:, :, None, None, None]
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h
