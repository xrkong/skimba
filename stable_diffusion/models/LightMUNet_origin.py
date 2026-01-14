from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from timm.models.vision_transformer import PatchEmbed, Mlp

from mamba_ssm import Mamba
import math

def modulate(x, shift, scale):
    # x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x * (1 + scale) + shift
    return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def get_dwconv_layer(
        spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)


class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_slices, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        # self.mamba = Mamba(
        #     d_model=input_dim,  # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor
        #     bimamba_type = 'v2'
        # )

        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )



        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_mamba_layer(
        spatial_dims: int, in_channels: int, out_channels: int, num_slices: int, stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels, num_slices=num_slices)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims == 3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


class ResMambaBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            num_slices: int,
            norm: tuple | str,
            kernel_size: int = 3,
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, num_slices=num_slices
        )
        self.conv2 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, num_slices=num_slices
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x


class ResUpBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)
        return x

class MambaLayer_fusion(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2",
            nslices=num_slices,
        )

    def forward(self, x):
        B, N, C = x.shape
        assert C == self.dim
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        # out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return x_mamba


class LightMUNet(nn.Module):

    def __init__(
            self,
            spatial_dims: int = 3,
            init_filters: int = 64, #8
            in_channels: int = 32, #1
            out_channels: int = 8,
            dropout_prob: float | None = None,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            norm_name: str = "",
            num_groups: int = 8,
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 4), # (1, 2, 2, 4)
            blocks_up: tuple = (1, 1), #(1, 1, 1)
            upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_dwconv_layer(spatial_dims, in_channels, in_channels) # init_filters
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)
        in_chans = 16
        self.t_embedder = TimestepEmbedder(in_chans)
        self.conv_output = nn.Sequential(
            nn.Linear(8, in_chans),
            nn.ReLU()
        )
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(in_chans, 6 * in_chans, bias=True)
        # )
        # self.attn = MambaLayer_fusion(dim=in_chans, num_slices=64)
        # self.norm1 = nn.LayerNorm(in_chans, elementwise_affine=False, eps=1e-6)
        # self.norm2 = nn.LayerNorm(in_chans, elementwise_affine=False, eps=1e-6)
        # mlp_ratio = 4.0
        # mlp_hidden_dim = int(in_chans * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=in_chans, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # self.conv_output = nn.Sequential(
        #     nn.Linear(8, in_chans),
        #     nn.ReLU()
        # )

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            # layer_in_channels = filters * 4 ** i # 2
            layer_in_channels = [32, 32, 64, 256]
            num_slices_list = [8, 16, 32, 64]
            downsample_mamba = (
                get_mamba_layer(spatial_dims, layer_in_channels[i], layer_in_channels[i+1], num_slices=num_slices_list[i+1], stride=2) # layer_in_channels // 24,
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba,
                *[ResMambaBlock(spatial_dims, layer_in_channels[i+1], num_slices=num_slices_list[i+1], norm=norm, act=self.act) for _ in range(item)]

            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            # sample_in_channels = filters * 2 ** (n_up - i) #filters * 2 ** (n_up - i)
            sample_in_channels = [256, 64, 32]
            up_layers.append(
                nn.Sequential(
                    *[
                        ResUpBlock(spatial_dims, sample_in_channels[i+1], norm=norm, act=self.act) for _ in range(blocks_up[i]) #sample_in_channels // 2
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels[i], sample_in_channels[i+1], kernel_size=1), #sample_in_channels // 2
                        get_upsample_layer(spatial_dims, sample_in_channels[i+1], upsample_mode=upsample_mode), # sample_in_channels // 2
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.in_channels), #self.in_channels self.init_filters
            self.act_mod,
            get_dwconv_layer(self.spatial_dims, self.in_channels, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor, c_in_all: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        B, C, L, W, H = x.shape
        t = self.t_embedder(t_in)
        c_in_all = c_in_all.reshape(B, 16, L, W, H)
        c = c_in_all + t[:, :, None, None, None]
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        x = self.conv_output(x)
        x = x.permute(0, 2, 1).reshape(B, 16, L, W, H)
        x = torch.cat((x, c), dim=1)
        # x = torch.cat((x, c), dim=1)
        # c = c.flatten(start_dim=2).permute(0, 2, 1)
        # x = x.flatten(start_dim=2).permute(0, 2, 1)
        # x = self.conv_output(x)
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        # x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # x = x.permute(0, 2, 1).reshape(B, 16, L, W, H)

        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x