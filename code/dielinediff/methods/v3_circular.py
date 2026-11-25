"""Stage-local horizontal circular padding for frozen SDXL convolutional modules."""

from contextlib import contextmanager
from types import MethodType

import torch
from diffusers.models.downsampling import Downsample2D
from torch import nn
from torch.nn import functional as F


def _padding(layer):
    if layer.padding == "same":
        total_y = (layer.kernel_size[0] - 1) * layer.dilation[0]
        total_x = (layer.kernel_size[1] - 1) * layer.dilation[1]
        return (total_x // 2, total_x - total_x // 2,
                total_y // 2, total_y - total_y // 2)
    if layer.padding == "valid":
        return 0, 0, 0, 0
    y, x = layer.padding
    return x, x, y, y


def _horizontal_conv2d(layer, inputs):
    left, right, top, bottom = _padding(layer)
    padded = F.pad(inputs, (left, right, 0, 0), mode="circular")
    if top != bottom:
        padded = F.pad(padded, (0, 0, top, bottom))
        top = 0
    return F.conv2d(padded, layer.weight, layer.bias, layer.stride, (top, 0),
                    layer.dilation, layer.groups)


def _wrap_downsample_input(_layer, args):
    # Diffusers adds a zero column and row before padding-zero downsampling.
    # Replace its right column with the first column, including the zero bottom row.
    inputs = args[0]
    return (torch.cat((inputs[..., :-1], inputs[..., :1]), dim=-1), *args[1:])


@contextmanager
def horizontal_circular(*models):
    """Wrap convolutional coordinates along X and retain the native zero padding along Y."""
    patched = []
    hooks = []
    absent = object()
    try:
        for model in models:
            for layer in model.modules():
                if isinstance(layer, nn.Conv2d) and any(_padding(layer)[:2]):
                    if layer.padding_mode != "zeros":
                        raise ValueError("Horizontal circular padding expects native zero padding")
                    patched.append((layer, layer.__dict__.get("forward", absent)))
                    layer.forward = MethodType(_horizontal_conv2d, layer)
                if isinstance(layer, Downsample2D) and layer.use_conv and layer.padding == 0:
                    hooks.append(layer.conv.register_forward_pre_hook(_wrap_downsample_input))
        yield
    finally:
        for hook in hooks:
            hook.remove()
        for layer, previous in reversed(patched):
            if previous is absent:
                del layer.forward
            else:
                layer.forward = previous


@contextmanager
def full_vae_canvas(vae):
    """Encode and decode a complete canvas, restoring the pipeline's tiling option."""
    tiled = vae.use_tiling
    vae.disable_tiling()
    try:
        yield
    finally:
        vae.enable_tiling(tiled)


@contextmanager
def circular_ring_codec(unet, vae):
    """Use one unpartitioned VAE canvas along the entire circular ring."""
    with full_vae_canvas(vae), horizontal_circular(unet, vae):
        yield
