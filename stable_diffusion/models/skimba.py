# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
import torch.nn as nn
import torch
import math
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
from monai.networks.layers.utils import get_act_layer, get_norm_layer
import numbers
from stable_diffusion.models.mamba_simple_cross_modal import Mamba_fusion


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # l, w, h  = x.shape[2:]
        # return to_5d(self.body(to_3d(x)), l, w, h)
        x = self.body(x)
        return x

class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba_fusion(dim, nslices=8, bimamba_type="v3")
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, ms, pan):
        b, c, l, w, h = ms.shape
        ms = ms.flatten(start_dim=2).permute(0, 2, 1)
        pan = pan.flatten(start_dim=2).permute(0, 2, 1)
        ms = self.norm1(ms)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(ms, extra_emb=pan)
        global_f = global_f.reshape(b, c, l, w, h)
        global_f = self.dwconv(global_f)
        return global_f

class Bottleneck3D(nn.Module):

    def __init__(self, inplanes, planes, norm_layer, stride=1, dilation=[1, 1, 1], expansion=4, downsample=None,
                 fist_dilation=1, multi_grid=1,
                 bn_momentum=0.0003):
        super(Bottleneck3D, self).__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 1, 3), stride=(1, 1, stride),
                               dilation=(1, 1, dilation[0]), padding=(0, 0, dilation[0]), bias=False)
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 1), stride=(1, stride, 1),
                               dilation=(1, dilation[1], 1), padding=(0, dilation[1], 0), bias=False)
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                               dilation=(dilation[2], 1, 1), padding=(dilation[2], 0, 0), bias=False)
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu


class MambaLayer_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_slices, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
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

        self.norm_1 = nn.LayerNorm(input_dim)
        self.mamba_1 = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )
        self.proj_1 = nn.Linear(input_dim, output_dim)
        self.skip_scale_1 = nn.Parameter(torch.ones(1))

        self.norm_3 = nn.LayerNorm(input_dim)
        self.mamba_3 = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )
        self.proj_3 = nn.Linear(input_dim, output_dim)
        self.skip_scale_3 = nn.Parameter(torch.ones(1))

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
        # ---------------------dilation_1 -----------------------#
        selected_tensor_1 = x_flat[:, 0::2, :]
        x_norm_1 = self.norm_1(selected_tensor_1)
        x_mamba_1 = self.mamba_1(x_norm_1) + self.skip_scale_1 * selected_tensor_1
        x_mamba_1 = self.norm_1(x_mamba_1)
        x_mamba_1 = self.proj_1(x_mamba_1)
        restored_tensor_1 = torch.zeros_like(x_flat)
        restored_tensor_1[:, 0::2, :] = x_mamba_1
        ##---------------------dilation_3---------------------------#
        selected_tensor_3 = x_flat[:, 0::4, :]
        x_norm_3 = self.norm_3(selected_tensor_3)
        x_mamba_3 = self.mamba_3(x_norm_3) + self.skip_scale_3 * selected_tensor_3
        x_mamba_3 = self.norm_3(x_mamba_3)
        x_mamba_3 = self.proj_3(x_mamba_3)
        restored_tensor_3 = torch.zeros_like(x_flat)
        restored_tensor_3[:, 0::4, :] = x_mamba_3
        x_mamba = x_mamba + restored_tensor_1 + restored_tensor_3
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_mamba_layer(in_channels: int, out_channels: int, num_slices: int, stride: int = 1
                    ):
    mamba_layer = MambaLayer_Decoder(input_dim=in_channels, output_dim=out_channels, num_slices=num_slices)
    return mamba_layer


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


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )
        self.norm_1 = nn.LayerNorm(dim)
        self.mamba_1 = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )
        self.norm_3 = nn.LayerNorm(dim)
        self.mamba_3 = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )

    def forward(self, x):
        l, w, h = x.size(2), x.size(3), x.size(4)
        sparse_shape = [l, w, h]
        # -----------------dilation=0--------------------------#
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        # --------------dilation=1-------------------------#
        selected_tensor_1 = x_flat[:, 0::2, :]
        x_norm_1 = self.norm_1(selected_tensor_1)
        x_mamba_1 = self.mamba_1(x_norm_1)
        restored_tensor_1 = torch.zeros_like(x_flat)
        restored_tensor_1[:, 0::2, :] = x_mamba_1
        # ------------------dilation=3-----------------------------#
        selected_tensor_3 = x_flat[:, 0::4, :]
        x_norm_3 = self.norm_3(selected_tensor_3)
        x_mamba_3 = self.mamba_3(x_norm_3)
        restored_tensor_3 = torch.zeros_like(x_flat)
        restored_tensor_3[:, 0::4, :] = x_mamba_3
        out_final = x_mamba + restored_tensor_1 + restored_tensor_3
        out_final = out_final.transpose(-1, -2).reshape(B, C, *img_dims)
        out_final = out_final + x_skip
        return out_final


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.BatchNorm3d(in_channles)  # InstanceNorm3d
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.BatchNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.BatchNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.BatchNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class ResMambaBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            num_slices: int,
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
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
            in_channels=in_channels, out_channels=in_channels, num_slices=num_slices
        )
        self.conv2 = get_mamba_layer(
            in_channels=in_channels, out_channels=in_channels, num_slices=num_slices
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


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[64, 256, 256],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        bn_momentum = 0.1
        norm_layer = nn.InstanceNorm3d
        norm_name = "instance"
        spatial_dims = 3
        res_block = True

        self.downsample_layers = nn.ModuleList()

        semantic_layer_down_1 = nn.Sequential(
            UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=32,
                out_channels=dims[1],
                kernel_size=3,
                stride=2,
                norm_name=norm_name,
                res_block=res_block,
            ),
            Bottleneck3D(dims[1], dims[1] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(dims[1], dims[1] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(dims[1], dims[1] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        semantic_layer_down_2 = nn.Sequential(
            UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=dims[1],
                out_channels=dims[2],
                kernel_size=3,
                stride=(2, 2, 1),
                norm_name=norm_name,
                res_block=res_block,
            ),
            Bottleneck3D(dims[2], dims[2] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(dims[2], dims[2] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(dims[2], dims[2] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.downsample_layers.append(semantic_layer_down_1)
        self.downsample_layers.append(semantic_layer_down_2)

        self.stages = nn.ModuleList()

        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(2):
            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i + 1], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(2):
            layer = nn.InstanceNorm3d(dims[i_layer + 1])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer + 1], 2 * dims[i_layer + 1]))

    def forward_features(self, x):
        outs = []
        for i in range(2):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out_norm = norm_layer(x)
                x_out_mlp = self.mlps[i](x_out_norm)
                x_out = x + x_out_mlp
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class MambaDecoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[256, 256, 64],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        spatial_dims = 3
        norm_name = "instance"
        res_block = True
        bn_momentum = 0.1
        norm_layer = nn.InstanceNorm3d
        self.upsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        upsample_layer_1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[0],
            out_channels=dims[1],
            kernel_size=3,
            upsample_kernel_size=(2, 2, 1),
            norm_name=norm_name,
            res_block=res_block,
        )
        upsample_layer_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=dims[1],
            out_channels=dims[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.upsample_layers.append(upsample_layer_1)
        self.upsample_layers.append(upsample_layer_2)

        self.gscs = nn.ModuleList()

        semantic_layer_up_1 = nn.Sequential(
            Bottleneck3D(dims[1], dims[1] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(dims[1], dims[1] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(dims[1], dims[1] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        semantic_layer_up_2 = nn.Sequential(
            Bottleneck3D(dims[2], dims[2] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(dims[2], dims[2] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(dims[2], dims[2] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.gscs.append(semantic_layer_up_1)
        self.gscs.append(semantic_layer_up_2)

        self.stages = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        layer_in_channels = [256, 64, 32, 32]
        for i in range(2):
            stage = nn.Sequential(
                # *[MambaLayer(dim=dims[i+1], num_slices=num_slices_list[i]) for j in range(depths[i])]
                ResMambaBlock(spatial_dims, layer_in_channels[i], num_slices=num_slices_list[i])
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(2):
            layer = nn.InstanceNorm3d(dims[i_layer + 1])  # InstanceNorm3d
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer + 1], 2 * dims[i_layer + 1]))

    def forward_features(self, x, skip_features):
        outs = []
        for i in range(2):
            skip_features_partial = skip_features[i]
            x = self.upsample_layers[i](x, skip_features_partial)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out_norm = norm_layer(x)
                x_out_mlp = self.mlps[i](x_out_norm)
                x_out = x_out_mlp + x
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x, skip_features):
        x = self.forward_features(x, skip_features)
        return x


class SegMamba(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=8,
            depths=[2, 2, 2, 2],
            feat_size=[32, 64, 256],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",  # instance
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.t_embedder = TimestepEmbedder(16)
        chs = [16, 16]
        mybias = False
        self.a_conv1 = nn.Sequential(nn.Conv3d(chs[0], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.a_conv2 = nn.Sequential(nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.a_conv3 = nn.Sequential(
            nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU(),
            nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU()
        )
        self.a_conv4 = nn.Sequential(
            nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU(),
            nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU(),
            nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU()
        )
        self.a_conv5 = nn.Sequential(nn.Conv3d(chs[1] * 3, chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.a_conv6 = nn.Sequential(
            nn.Conv3d(chs[1] * 3, chs[1] * 2, 3, 1, padding=1, bias=mybias), nn.ReLU(),
            nn.Conv3d(chs[1] * 2, chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU(),
        )
        self.a_conv7 = nn.Sequential(
            nn.Conv3d(chs[1] * 3, chs[1] * 2, 3, 1, padding=1, bias=mybias), nn.ReLU(),
            nn.Conv3d(chs[1] * 2, chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU(),
            nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU(),
        )
        self.ch_conv1 = nn.Sequential(nn.Conv3d(chs[1] * 7, chs[1], kernel_size=1, stride=1, bias=mybias), nn.ReLU())
        self.res_1 = nn.Sequential(nn.Conv3d(chs[0], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.res_2 = nn.Sequential(
            nn.Conv3d(chs[0], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU(),
            nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU()
        )
        self.res_3 = nn.Sequential(
            nn.Conv3d(chs[0], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU(),
            nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU(),
            nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU()
        )
        ##-----------------------------fusion----------------------------------#
        self.fusion_mamba = CrossMamba(chs[1])

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans,
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=chs[1] * 2,
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],  # self.hidden_size
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.vit_1 = MambaDecoder(256,
                                  depths=depths,
                                  dims=[256, 256, 64],
                                  drop_path_rate=drop_path_rate,
                                  layer_scale_init_value=layer_scale_init_value,
                                  )

        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=self.out_chans)
        self.conv_output = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU()
        )

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x, c_in_all, t_in):
        B, C, L, W, H = x.shape
        t = self.t_embedder(t_in)
        c = c_in_all + t[:, :, None, None, None]

        x1 = self.a_conv1(c)
        x2 = self.a_conv2(x1)
        x3 = self.a_conv3(x1)
        x4 = self.a_conv4(x1)
        t1 = torch.cat((x2, x3, x4), 1)
        x5 = self.a_conv5(t1)
        x6 = self.a_conv6(t1)
        x7 = self.a_conv7(t1)
        x8 = torch.cat((x1, x2, x3, x4, x5, x6, x7), 1)
        y0 = self.ch_conv1(x8)
        y1 = self.res_1(c)
        y2 = self.res_2(c)
        y3 = self.res_3(c)
        c_all = c + y0 + y1 + y2 + y3

        x = x.flatten(start_dim=2).permute(0, 2, 1)
        x = self.conv_output(x)
        x = x.permute(0, 2, 1).reshape(B, 16, L, W, H)
        # -----------------------------fusion of condition features and noises -----------------------------------#
        feature_fusion_all = torch.cat((x, c_all), dim=1)

        skip_features = []
        outs = self.vit(feature_fusion_all)
        enc1 = self.encoder1(feature_fusion_all)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        skip_features.append(enc2)
        skip_features.append(enc1)

        x3 = outs[1]
        enc3 = self.encoder3(x3)

        decoder_output = self.vit_1(enc3, skip_features)
        decoder_output_final = decoder_output[1]
        decoder_output_final = self.decoder1(decoder_output_final)
        decoder_output_final = self.out(decoder_output_final)

        return decoder_output_final
