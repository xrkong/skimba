
from stable_diffusion.models.spconv_utils import replace_feature
from torch import nn, einsum
from stable_diffusion.dataclass import BaseDataclass
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

from stable_diffusion.modules.distributions import GaussianDistribution
import torch

from dataclasses import dataclass, field
from typing import List, Optional
from functools import partial
from stable_diffusion.models.spconv_backbone import post_act_block
import torch.nn.functional as F

class InitialConv(nn.Module):
    def __init__(self, in_channels=23, out_channels=16):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.bn   = nn.BatchNorm3d(out_channels)
        self.conv_down = nn.Conv3d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn_down = nn.BatchNorm3d(out_channels * 2)

    def forward(self, x):
        layer = F.relu(self.bn(self.conv(x)))
        layer = torch.add(layer, torch.cat([x[:,0:1]]*self.out_channels, 1))

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv

class DownConvBlock2b(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.out_channels = out_channels

        # self.conv_a = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn_a = nn.BatchNorm3d(out_channels)
        # self.conv_b = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn_b = nn.BatchNorm3d(out_channels)
        self.conv_a= nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1,padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )
        self.conv_b = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )
        self.conv_down = nn.Conv3d(in_channels=out_channels*2, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.bn_down = nn.BatchNorm3d(out_channels * 2)

    def forward(self, x):
        layer = F.relu(self.conv_a(x))
        layer = F.relu(self.conv_b(layer))
        layer = torch.cat((layer, x),dim=1)

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv

class UpConvBlock2b(nn.Module):
    def __init__(self, in_channels=32, out_channels=64, undersampling_factor=4):
        super().__init__()
        self.out_channels = out_channels
        self.undersampling_factor = undersampling_factor

        # self.conv_a = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn_a = nn.BatchNorm3d(out_channels)
        # self.conv_b = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn_b = nn.BatchNorm3d(out_channels)
        self.conv_a= nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1,padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )
        self.conv_b = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )
        self.conv_up = nn.ConvTranspose3d(in_channels=in_channels+out_channels, out_channels=out_channels // undersampling_factor, kernel_size=2, stride=2, padding=0)
        self.bn_up = nn.BatchNorm3d(out_channels // undersampling_factor)

    def forward(self, x):
        layer = F.relu(self.conv_a(x))
        layer = F.relu(self.conv_b(layer))
        # layer = torch.add(layer, x)
        layer = torch.cat((layer, x), dim=1)

        conv = F.relu(self.bn_up(self.conv_up(layer)))
        return layer, conv

class DownConvBlock3b(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.out_channels = out_channels

        # self.conv_a = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn_a = nn.BatchNorm3d(out_channels)
        # self.conv_b = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn_b = nn.BatchNorm3d(out_channels)
        # self.conv_c = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn_c = nn.BatchNorm3d(out_channels)
        self.conv_a= nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1,padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )
        self.conv_b = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )
        self.conv_c = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )
        self.conv_down = nn.Conv3d(in_channels=out_channels*2, out_channels=out_channels * 2, kernel_size=2, stride=2, padding=0)
        self.bn_down = nn.BatchNorm3d(out_channels * 2)

    def forward(self, x):
        layer = F.relu(self.conv_a(x))
        layer = F.relu(self.conv_b(layer))
        layer = F.relu(self.conv_c(layer))
        layer = torch.cat((layer, x), dim=1)

        conv = F.relu(self.bn_down(self.conv_down(layer)))
        return layer, conv

class UpConvBlock3b(nn.Module):
    def __init__(self,in_channels=8, out_channels=256, undersampling_factor=2):
        super().__init__()
        self.out_channels = out_channels

        # self.conv_a = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn_a = nn.BatchNorm3d(out_channels)
        # self.conv_b = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn_b = nn.BatchNorm3d(out_channels)
        # self.conv_c = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn_c = nn.BatchNorm3d(out_channels)

        self.conv_a= nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1,padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )
        self.conv_b = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )
        self.conv_c = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )

        self.conv_up = nn.ConvTranspose3d(in_channels=in_channels+out_channels, out_channels=out_channels // undersampling_factor, kernel_size=2, stride=2, padding=0)
        self.bn_up = nn.BatchNorm3d(out_channels // undersampling_factor)

    def forward(self, x):
        layer = F.relu(self.conv_a(x))
        layer = F.relu(self.conv_b(layer))
        layer = F.relu(self.conv_c(layer))
        layer = torch.cat((layer, x), dim=1)

        conv = F.relu(self.bn_up(self.conv_up(layer)))
        return layer, conv

class FinalConv(nn.Module):
    def __init__(self, num_outs=2, out_channels=32):
        super().__init__()
        # self.conv = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn = nn.BatchNorm3d(out_channels)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels)
        )
        self.conv_1x1 = nn.Conv3d(in_channels=out_channels*2, out_channels=num_outs, kernel_size=1, stride=1, padding=0)
        self.bn_1x1 = nn.BatchNorm3d(num_outs)
        # self.final = F.softmax
    def forward(self, x):
        layer = F.relu(self.conv(x))
        layer = torch.cat((layer, x), dim=1)
        layer = self.bn_1x1(self.conv_1x1(layer))
        # layer = self.final(layer, dim=1)
        return layer

class CatBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        cat = torch.cat((x1,x2), 1)
        return cat


def build_encoder(in_channels, latent_channels,autoencoder_channels_list, autoencoder_num_res_blocks, groups, dropout_rate):
    return Encoder( # 32, 64, [64, 128], 2, 32
        in_channels=in_channels,
        out_channels=latent_channels,
        channels_list=autoencoder_channels_list,
        num_res_blocks=autoencoder_num_res_blocks,
        groups=groups,
        dropout_rate = dropout_rate
    )


def build_decoder(latent_channels, out_channels,autoencoder_channels_list, autoencoder_num_res_blocks, groups, in_channels, dropout_rate):
    return Decoder( #
        in_channels=latent_channels,
        out_channels=out_channels or in_channels,
        channels_list=autoencoder_channels_list,
        num_res_blocks=autoencoder_num_res_blocks,
        groups=groups,
        dropout_rate = dropout_rate
    )

class AutoEncoderKL(nn.Module):
    def __init__(
        self,
        in_channels=8, #32
        out_channels = 8, #32
        latent_channels=8,
        autoencoder_num_res_blocks = 1, #2
        autoencoder_channels_list=[16, 32, 64, 128], #[32, 64, 128, 128]
        groups = 4, #32
        num_input_features = 3,
        init_size = 8,
        voxel_channel =1,
        dropout_rate=0.2
    ):

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.autoencoder_num_res_blocks = autoencoder_num_res_blocks
        self.autoencoder_channels_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.groups = groups
        # check params
        # assert (
        #     self.out_channels is None or self.out_channels == self.in_channels
        # ), f"input channels({self.input_channels}) of image should be equal to output channels({self.out_channels})"
        super(AutoEncoderKL, self).__init__()
        self.latent_channels = latent_channels = self.latent_channels
        self.encoder = build_encoder(self.in_channels, self.latent_channels , self.autoencoder_channels_list, self.autoencoder_num_res_blocks, self.groups, dropout_rate)
        self.decoder = build_decoder(self.latent_channels, self.out_channels, self.autoencoder_channels_list, self.autoencoder_num_res_blocks, self.groups, self.in_channels, dropout_rate)
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv3d(self.autoencoder_channels_list[1], self.autoencoder_channels_list[2], kernel_size=1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv3d(self.autoencoder_channels_list[1], self.autoencoder_channels_list[1], kernel_size=1)

    def encode(self, img: torch.Tensor) -> GaussianDistribution: #: torch.Tensor
        """
        Encode image into latent vector
        Args:
            - x (torch.Tensor):
                  image, shape = `[batch, channel, height, width, depth]`
        Returns:
            - gaussian distribution (torch.Tensor):

        """
        z = self.encoder(img)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        # Return the distribution(posterior)
        # moments = moments.dense()
        return moments, AutoEncoderKLOutput(GaussianDistribution(moments))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode images from latent representation

        Args:
            - z(torch.Tensor):
                  latent representation with shape `[batch_size, emb_channels, z_height, z_height]`
        """
        # check params
        z = self.post_quant_conv(latent)
        # assert (
        #     latent.shape[1] == self.latent_channels
        # ), f"Expected latent representation to have {self.latent_channels} channels, got {z.shape[1]}"
        output = self.decoder(z)
        return output



class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels_list: List[int],
            num_res_blocks: int,
            groups: int = 4,
            n_heads=4,
            dropout_rate=0.2
    ):
        super(Encoder, self).__init__()
        self.in_channels = 16
        self.out_channels = out_channels
        self.channels_list = [32, 64, 128, 256]
        self.num_res_blocks = num_res_blocks
        self.groups = groups
        self.n_heads = n_heads
        input_channels =23

        nclasses = 20
        channels = 16
        l_size = '882'
        attention = True

        # self.init_conv = InitialConv(in_channels=input_channels, out_channels=channels)

        self.conv_input = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU()
        )

        self.down_block_1 = DownConvBlock2b(out_channels=channels)
        self.down_block_2 = DownConvBlock3b(out_channels=channels * 2)
        # self.down_block_2 = DownConvBlock3b(out_channels=channels*2)
        # self.down_block_3 = DownConvBlock3b(out_channels=channels * 8)

        self.conv_output = nn.Sequential(
            nn.Linear(channels*4 , 8),
            nn.ReLU()
        )

    def forward(self, x):
        x = replace_feature(x, torch.cat([x.features, x.indices[:, 1:4]], dim=1))
        x = x.dense().float()

        b, c, w, l, h = x.shape
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        x = self.conv_input(x)
        x = x.permute(0, 2, 1).view(b, 16, w, l, h)

        # layer_down_1, conv_down_1 = self.init_conv(x)
        layer_down_2, conv_down_2 = self.down_block_1(x)
        layer_down_3, conv_down_3 = self.down_block_2(conv_down_2)

        b, c, w, l, h = conv_down_3.shape
        conv_down_3 = conv_down_3.flatten(start_dim=2).permute(0, 2, 1)
        conv_down_3 = self.conv_output(conv_down_3)
        conv_down_3 = conv_down_3.permute(0, 2, 1).view(b, 8, w, l, h)

        return conv_down_3


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_list: List[int],
        num_res_blocks: int,
        groups: int = 4,
        dropout_rate = 0.2
    ):
        super(Decoder, self).__init__()
        nclasses = 20
        init_size = 16
        l_size = '882'
        attention = True
        in_channels=16

        self.nclasses = nclasses
        self.l_size = l_size
        self.attention = attention
        channels = 16
        # self.up_block_2 = UpConvBlock3b(in_channels=8, out_channels=channels * 8, undersampling_factor=2)
        self.up_block_2 = UpConvBlock3b(in_channels=8, out_channels=channels * 8, undersampling_factor=2)
        self.up_block_1 = UpConvBlock2b(in_channels=channels*4, out_channels=channels * 4, undersampling_factor=2)

        self.out_conv = FinalConv(num_outs=2, out_channels=channels * 2)

    def forward(self, conv_down_3):

        layer_up_2, conv_up_2 = self.up_block_2(conv_down_3)
        layer_up_1, conv_up_1 = self.up_block_1(conv_up_2)

        layer_out = self.out_conv(conv_up_1)

        return layer_out


@dataclass
class AutoEncoderKLOutput:
    latent_dist: "GaussianDistribution"
