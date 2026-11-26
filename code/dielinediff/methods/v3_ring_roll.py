"""Roll M3 spatial UNet predictions while preserving the atlas coordinate frame."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .v3_attention import RegionRouting


ROLL_STRIDE = 37


def ring_shift(step_index, latent_width):
    if step_index < 0 or latent_width < 1:
        raise ValueError("M3 requires a nonnegative step and a nonempty cyclic axis")
    return (ROLL_STRIDE * step_index) % latent_width


def predict_at_phase(predict, latent, shift, *args):
    """Apply one forward at the chosen horizontal phase, returning native coordinates."""
    shift %= latent.shape[-1]
    if not shift:
        return predict(latent, *args)
    rolled = torch.roll(latent, shifts=shift, dims=-1)
    prediction = predict(rolled, *args)
    return torch.roll(prediction, shifts=-shift, dims=-1)


class PhasedRegionRouting:
    """Roll the base grid before extracting spatial groups at each UNet scale."""

    def __init__(self, original):
        self.original = original
        self.shift = 0
        self.routes = {0: original}

    def set_phase(self, _step_index, shift):
        shift %= self.original.labels.shape[-1]
        if shift not in self.routes:
            self.routes[shift] = RegionRouting(np.roll(self.original.labels, shift, axis=-1))
        self.shift = shift

    def groups(self, queries, device):
        return self.routes[self.shift].groups(queries, device)


@dataclass
class _RingPhase:
    rolling: bool
    set_phase: Callable[[int, int], None] | None
    shifts: list[int] = field(default_factory=list)


class M3RingMixin:
    """Scope phase changes to the ring's UNet calls, outside scheduler and LUT operations."""

    @contextmanager
    def _ring_phase(self, *, rolling, set_phase=None):
        if getattr(self, "_active_ring_phase", None) is not None:
            raise RuntimeError("A ring phase context is already active")
        phase = _RingPhase(rolling, set_phase)
        self._active_ring_phase = phase
        try:
            yield phase
        finally:
            del self._active_ring_phase

    def _predict(self, latent, timestep, text, condition, control):
        predict = super()._predict
        phase = getattr(self, "_active_ring_phase", None)
        if phase is None:
            return predict(latent, timestep, text, condition, control)
        if phase.rolling and control is not None:
            raise ValueError("A rolling ring must not use a flat spatial ControlNet condition")
        index = len(phase.shifts)
        shift = ring_shift(index, latent.shape[-1]) if phase.rolling else 0
        phase.shifts.append(shift)
        if phase.set_phase is not None:
            phase.set_phase(index, shift)
        # Text and pooled SDXL time IDs are global. Self-attention operates on
        # the rolling spatial input; the caller only sees unrolled epsilon.
        return predict_at_phase(predict, latent, shift, timestep, text, condition, control)
