"""Region-masked SDXL cross-attention and optional read-only ring K/V sharing."""

from contextlib import contextmanager

import numpy as np
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from torch.nn import functional as F


class RegionRouting:
    """Group spatial queries by their latent-layout region at each U-Net scale."""

    def __init__(self, labels):
        self.labels = np.asarray(labels)
        if self.labels.ndim != 2 or min(self.labels.shape) < 1:
            raise ValueError("Regional attention needs a 2D latent region map")
        self.indices = {}

    def groups(self, queries, device):
        lookup = (queries, device)
        if lookup not in self.indices:
            height, width = self.labels.shape
            stride = 1
            while height * width > queries and min(height, width) > 1:
                height //= 2
                width //= 2
                stride *= 2
            if height * width != queries or (height * stride, width * stride) != self.labels.shape:
                raise ValueError(f"Cannot route {queries} queries for latent layout {self.labels.shape}")
            # Sample the center of each downsampled cell in canvas coordinates.
            offset = stride // 2
            flat = self.labels[offset::stride, offset::stride].reshape(-1)
            self.indices[lookup] = tuple(
                (int(label), torch.as_tensor(np.flatnonzero(flat == label), device=device))
                for label in np.unique(flat)
            )
        return self.indices[lookup]


class RegionalAttention(AttnProcessor2_0):
    """Apply the same region-mask routing to U-Net and flat ControlNet blocks."""

    def __init__(self, routing, contexts):
        super().__init__()
        self.routing = routing
        self.contexts = contexts

    @staticmethod
    def _heads(tensor, count):
        batch, length, channels = tensor.shape
        return tensor.reshape(batch, length, count, channels // count).transpose(1, 2)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        if encoder_hidden_states is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        if attention_mask is not None:
            raise ValueError("Regional SDXL expects unmasked fixed-length text tokens")
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        spatial_shape = hidden_states.shape if hidden_states.ndim == 4 else None
        if spatial_shape is not None:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self._heads(attn.to_q(hidden_states), attn.heads)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        output = torch.empty_like(query)
        for region, index in self.routing.groups(query.shape[-2], query.device):
            context = self.contexts[region]
            if attn.norm_cross:
                context = attn.norm_encoder_hidden_states(context)
            key = self._heads(attn.to_k(context), attn.heads)
            value = self._heads(attn.to_v(context), attn.heads)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            attended = F.scaled_dot_product_attention(query.index_select(2, index), key, value)
            output.index_copy_(2, index, attended)
        output = output.transpose(1, 2).flatten(2).to(query.dtype)
        output = attn.to_out[1](attn.to_out[0](output))
        if spatial_shape is not None:
            output = output.transpose(1, 2).reshape(spatial_shape)
        if attn.residual_connection:
            output = output + residual
        return output / attn.rescale_output_factor


@contextmanager
def region_attention(models, routing, contexts):
    """Route cross-attention only, restoring the original processors on exit."""
    processor = RegionalAttention(routing, contexts)
    replaced = []
    try:
        for model in models:
            for layer in model.modules():
                if isinstance(layer, Attention) and layer.is_cross_attention:
                    replaced.append((layer, layer.processor))
                    layer.set_processor(processor)
        yield
    finally:
        for layer, original in reversed(replaced):
            layer.set_processor(original)


@contextmanager
def ring_style_attention(unet, bank, *, capture):
    """Capture the last ring step or expose its frozen K/V to cap self-attention."""
    def save(name):
        def hook(_layer, _arguments, result):
            bank[name] = result.detach()
        return hook

    def append(name):
        def hook(_layer, _arguments, result):
            reference = bank[name]
            if reference.shape[0] != result.shape[0] or reference.shape[-1] != result.shape[-1]:
                raise ValueError(f"Incompatible ring attention reference at {name}")
            return torch.cat((result, reference), dim=1)
        return hook

    hook_for = save if capture else append
    handles = []
    try:
        for name, layer in unet.named_modules():
            if isinstance(layer, Attention) and not layer.is_cross_attention:
                handles.append(layer.to_k.register_forward_hook(hook_for((name, "key"))))
                handles.append(layer.to_v.register_forward_hook(hook_for((name, "value"))))
        yield
    finally:
        for handle in handles:
            handle.remove()
