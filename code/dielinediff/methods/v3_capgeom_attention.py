"""Cache one ring self-attention block and reuse only exact DDIM timestep keys."""

from contextlib import ExitStack, contextmanager

import torch


def _time_key(args, kwargs):
    value = kwargs.get("timestep", args[1] if len(args) > 1 else None)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected one UNet timestep per CFG batch")
        value = value.item()
    if value is None or int(value) != value:
        raise ValueError("Cannot capture attention without an exact DDIM timestep")
    return int(value)


@contextmanager
def capture_aligned_bank(unet, bank, layer):
    """Copy the chosen projections to CPU without replacing the ring attention."""
    modules = dict(unet.named_modules())
    attention = modules[layer]
    position = {}

    def enter(_model, args, kwargs):
        timestep = _time_key(args, kwargs)
        if timestep in bank:
            raise ValueError(f"Duplicate ring timestep {timestep}")
        position["timestep"] = timestep
        bank[timestep] = {}

    def record(name):
        def callback(_module, _args, value):
            at = position["timestep"]
            if name in bank[at]:
                raise ValueError(f"Duplicate {name} projection at ring timestep {at}")
            bank[at][name] = value.detach().to(device="cpu", copy=True).contiguous()
        return callback

    with ExitStack() as clean:
        clean.callback(unet.register_forward_pre_hook(enter, with_kwargs=True).remove)
        for key in ("to_k", "to_v"):
            clean.callback(getattr(attention, key).register_forward_hook(record(key)).remove)
        yield
    if not bank or any(set(fields) != {"to_k", "to_v"} for fields in bank.values()):
        raise RuntimeError("Ring attention capture did not produce complete K/V pairs")


@contextmanager
def inject_aligned_bank(unet, bank, layer, used):
    """Append ring K/V to one cap layer at the identical original scheduler index."""
    modules = dict(unet.named_modules())
    attention = modules[layer]
    position = {}

    def enter(_model, args, kwargs):
        timestep = _time_key(args, kwargs)
        if timestep not in bank:
            raise ValueError(f"No ring K/V captured at requested DDIM timestep {timestep}")
        position["timestep"] = timestep
        used.append(timestep)

    def append(name):
        def callback(_module, _args, value):
            origin = bank[position["timestep"]][name]
            if origin.ndim != value.ndim or origin.shape[:-2] != value.shape[:-2] or origin.shape[-1] != value.shape[-1]:
                raise ValueError("Ring and cap K/V projections are not shape-compatible")
            return torch.cat((value, origin.to(device=value.device, dtype=value.dtype)), dim=-2)
        return callback

    with ExitStack() as clean:
        clean.callback(unet.register_forward_pre_hook(enter, with_kwargs=True).remove)
        for key in ("to_k", "to_v"):
            clean.callback(getattr(attention, key).register_forward_hook(append(key)).remove)
        yield
