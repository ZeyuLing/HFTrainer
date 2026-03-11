"""Compact StyleGAN2-style generator and discriminator modules."""

import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hftrainer.registry import HF_MODELS


def normalize_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class EqualLinear(nn.Module):
    """Linear layer with equalized learning-rate style scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lr_mul: float = 1.0,
        activation: str = 'linear',
        bias_init: float = 0.0,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.scale = lr_mul / math.sqrt(in_features)
        self.lr_mul = lr_mul
        self.activation = activation
        if bias:
            self.bias = nn.Parameter(torch.full((out_features,), float(bias_init)))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.scale
        bias = self.bias * self.lr_mul if self.bias is not None else None
        x = F.linear(x, weight, bias)
        if self.activation == 'lrelu':
            x = F.leaky_relu(x, negative_slope=0.2) * math.sqrt(2.0)
        elif self.activation != 'linear':
            raise ValueError(f"Unsupported activation: {self.activation}")
        return x


class ConvLayer(nn.Module):
    """Equalized-conv block used in the discriminator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = 'lrelu',
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = kernel_size // 2
        self.scale = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            padding=self.padding,
        )
        if self.activation == 'lrelu':
            x = F.leaky_relu(x, negative_slope=0.2) * math.sqrt(2.0)
        elif self.activation != 'linear':
            raise ValueError(f"Unsupported activation: {self.activation}")
        return x


class MappingNetwork(nn.Module):
    """Style mapping network."""

    def __init__(
        self,
        z_dim: int,
        w_dim: int,
        num_ws: int,
        num_layers: int = 8,
        lr_mul: float = 0.01,
        w_avg_beta: float = 0.995,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.w_avg_beta = w_avg_beta

        layers = []
        in_features = z_dim
        for _ in range(num_layers):
            layers.append(
                EqualLinear(
                    in_features,
                    w_dim,
                    lr_mul=lr_mul,
                    activation='lrelu',
                )
            )
            in_features = w_dim
        self.layers = nn.ModuleList(layers)
        self.register_buffer('w_avg', torch.zeros(w_dim))

    def forward(
        self,
        z: torch.Tensor,
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
        skip_w_avg_update: bool = False,
    ) -> torch.Tensor:
        x = normalize_2nd_moment(z)
        for layer in self.layers:
            x = layer(x)

        if self.training and not skip_w_avg_update:
            self.w_avg.copy_(self.w_avg.lerp(x.detach().mean(dim=0), 1 - self.w_avg_beta))

        ws = x.unsqueeze(1).repeat(1, self.num_ws, 1)
        if truncation_psi != 1.0:
            cutoff = self.num_ws if truncation_cutoff is None else truncation_cutoff
            ws[:, :cutoff] = self.w_avg.lerp(ws[:, :cutoff], truncation_psi)
        return ws


class ModulatedConv2d(nn.Module):
    """Style-modulated convolution with demodulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
        demodulate: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.demodulate = demodulate
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.style = EqualLinear(style_dim, in_channels, bias_init=1.0)
        self.scale = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        style = self.style(w).view(batch_size, 1, self.in_channels, 1, 1)

        weight = self.weight.unsqueeze(0) * style
        weight = weight * self.scale

        if self.demodulate:
            demod = torch.rsqrt(weight.square().sum((2, 3, 4)) + 1e-8)
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1, 1)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            height, width = x.shape[2:]

        x = x.view(1, batch_size * self.in_channels, height, width)
        weight = weight.view(
            batch_size * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        x = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
        x = x.view(batch_size, self.out_channels, x.shape[2], x.shape[3])
        return x


class StyledConv(nn.Module):
    """One StyleGAN2 synthesis conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        kernel_size: int = 3,
        upsample: bool = False,
        use_noise: bool = True,
    ):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            upsample=upsample,
            demodulate=True,
        )
        self.use_noise = use_noise
        if use_noise:
            self.noise_strength = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, w)
        if self.use_noise:
            noise = torch.randn(
                x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
            )
            x = x + noise * self.noise_strength
        x = x + self.bias.view(1, -1, 1, 1)
        x = F.leaky_relu(x, negative_slope=0.2) * math.sqrt(2.0)
        return x


class ToRGB(nn.Module):
    """StyleGAN2 ToRGB layer."""

    def __init__(self, in_channels: int, style_dim: int, out_channels: int = 3):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            style_dim=style_dim,
            demodulate=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, w)
        return x + self.bias.view(1, -1, 1, 1)


class SynthesisBlock(nn.Module):
    """One StyleGAN2 resolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        resolution: int,
        is_first: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.is_first = is_first
        if is_first:
            self.const = nn.Parameter(torch.randn(1, out_channels, 4, 4))
            self.conv0 = None
            self.conv1 = StyledConv(out_channels, out_channels, style_dim)
        else:
            self.const = None
            self.conv0 = StyledConv(in_channels, out_channels, style_dim, upsample=True)
            self.conv1 = StyledConv(out_channels, out_channels, style_dim)
        self.torgb = ToRGB(out_channels, style_dim)

    @property
    def num_ws(self) -> int:
        return 2 if self.is_first else 3

    def forward(
        self,
        x: Optional[torch.Tensor],
        image: Optional[torch.Tensor],
        ws: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w_iter = iter(ws.unbind(dim=1))

        if self.is_first:
            batch_size = ws.shape[0]
            x = self.const.repeat(batch_size, 1, 1, 1)
            x = self.conv1(x, next(w_iter))
        else:
            x = self.conv0(x, next(w_iter))
            x = self.conv1(x, next(w_iter))
            if image is not None:
                image = F.interpolate(
                    image, scale_factor=2, mode='bilinear', align_corners=False
                )

        rgb = self.torgb(x, next(w_iter))
        image = rgb if image is None else image + rgb
        return x, image


class DiscriminatorBlock(nn.Module):
    """Residual discriminator block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv0 = ConvLayer(in_channels, in_channels, kernel_size=3)
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3)
        self.skip = ConvLayer(in_channels, out_channels, kernel_size=1, activation='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.avg_pool2d(self.skip(x), kernel_size=2)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.avg_pool2d(x, kernel_size=2)
        return (x + y) / math.sqrt(2.0)


class MinibatchStdDev(nn.Module):
    """Append minibatch stddev channel."""

    def __init__(self, group_size: int = 4, num_channels: int = 1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        group = min(self.group_size, batch)
        if batch % group != 0:
            group = batch

        y = x.view(group, -1, self.num_channels, channels // self.num_channels, height, width)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.square().mean(dim=0) + 1e-8)
        y = y.mean(dim=(2, 3, 4))
        y = y.view(-1, self.num_channels, 1, 1)
        y = y.repeat(group, 1, height, width)
        return torch.cat([x, y], dim=1)


@HF_MODELS.register_module()
class StyleGAN2Generator(nn.Module):
    """
    Compact StyleGAN2-style generator.

    This is a compact PyTorch reference implementation that follows the
    architecture principles from the official StyleGAN2 paper: mapping
    network, modulated convolutions, noise injection, skip ToRGB, and
    path-length regularization support via returned W-space latents.
    """

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        img_resolution: int = 64,
        img_channels: int = 3,
        channel_base: int = 16384,
        channel_max: int = 512,
        mapping_layers: int = 8,
        style_mixing_prob: float = 0.9,
    ):
        super().__init__()
        if img_resolution < 4 or img_resolution & (img_resolution - 1):
            raise ValueError("img_resolution must be a power of two >= 4")

        self.latent_dim = z_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.style_mixing_prob = style_mixing_prob
        self.resolutions = [2 ** i for i in range(2, int(math.log2(img_resolution)) + 1)]
        channels = {
            res: min(channel_base // res, channel_max)
            for res in self.resolutions
        }

        blocks = []
        num_ws = 0
        in_channels = 0
        for res in self.resolutions:
            out_channels = channels[res]
            block = SynthesisBlock(
                in_channels=in_channels if res > 4 else out_channels,
                out_channels=out_channels,
                style_dim=w_dim,
                resolution=res,
                is_first=(res == 4),
            )
            blocks.append(block)
            num_ws += block.num_ws
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.num_ws = num_ws
        self.mapping = MappingNetwork(
            z_dim=z_dim,
            w_dim=w_dim,
            num_ws=num_ws,
            num_layers=mapping_layers,
        )

    def _maybe_style_mix(self, ws: torch.Tensor) -> torch.Tensor:
        if not self.training or self.style_mixing_prob <= 0 or random.random() >= self.style_mixing_prob:
            return ws
        cutoff = random.randint(1, self.num_ws - 1)
        z_mix = torch.randn(ws.shape[0], self.z_dim, device=ws.device, dtype=ws.dtype)
        ws_mix = self.mapping(z_mix, skip_w_avg_update=True)
        ws[:, cutoff:] = ws_mix[:, cutoff:]
        return ws

    def synthesis(self, ws: torch.Tensor) -> torch.Tensor:
        x = None
        image = None
        index = 0
        for block in self.blocks:
            block_ws = ws[:, index:index + block.num_ws]
            index += block.num_ws
            x, image = block(x, image, block_ws)
        return image

    def forward(
        self,
        z: torch.Tensor,
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
        return_latents: bool = False,
    ):
        ws = self.mapping(
            z,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        ws = self._maybe_style_mix(ws)
        if return_latents:
            ws = ws.clone().requires_grad_(True)
        image = self.synthesis(ws)
        if return_latents:
            return image, ws
        return image


@HF_MODELS.register_module()
class StyleGAN2Discriminator(nn.Module):
    """Compact StyleGAN2-style discriminator."""

    def __init__(
        self,
        img_resolution: int = 64,
        img_channels: int = 3,
        channel_base: int = 16384,
        channel_max: int = 512,
        mbstd_group_size: int = 4,
    ):
        super().__init__()
        if img_resolution < 4 or img_resolution & (img_resolution - 1):
            raise ValueError("img_resolution must be a power of two >= 4")

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.resolutions = [2 ** i for i in range(int(math.log2(img_resolution)), 2, -1)]
        channels = {
            res: min(channel_base // res, channel_max)
            for res in [4] + self.resolutions
        }

        self.from_rgb = ConvLayer(img_channels, channels[img_resolution], kernel_size=1)
        blocks = []
        in_channels = channels[img_resolution]
        for res in self.resolutions:
            out_channels = channels[res // 2]
            blocks.append(DiscriminatorBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.mbstd = MinibatchStdDev(group_size=mbstd_group_size, num_channels=1)
        self.final_conv = ConvLayer(in_channels + 1, channels[4], kernel_size=3)
        self.final_fc = EqualLinear(channels[4] * 4 * 4, channels[4], activation='lrelu')
        self.final_out = EqualLinear(channels[4], 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.from_rgb(image)
        for block in self.blocks:
            x = block(x)
        x = self.mbstd(x)
        x = self.final_conv(x)
        x = self.final_fc(x.flatten(1))
        return self.final_out(x)
