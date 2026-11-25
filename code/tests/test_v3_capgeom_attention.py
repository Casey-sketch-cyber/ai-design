"""Verify timestep-aware capture, lookup and hook restoration."""

import pytest
import torch
from torch import nn

from dielinediff.methods.v3_capgeom_attention import capture_aligned_bank, inject_aligned_bank


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_k = nn.Identity()
        self.to_v = nn.Identity()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Projector()

    def forward(self, sample, timestep):
        return self.attn.to_k(sample + timestep), self.attn.to_v(2 * sample + timestep)


def test_capture_preserves_ring_output_and_injects_exact_timestep_only():
    ring = Network()
    cap = Network()
    bank, used = {}, []
    source = torch.ones((2, 3, 4))
    target = torch.ones((2, 2, 4)) * 3
    with capture_aligned_bank(ring, bank, "attn"):
        pair_20 = ring(source, 20)
        pair_10 = ring(source, torch.tensor(10))
    assert torch.equal(pair_20[0], source + 20)
    assert torch.equal(pair_10[1], 2 * source + 10)
    assert set(bank) == {10, 20}
    assert bank[10]["to_k"].device.type == "cpu"
    with inject_aligned_bank(cap, bank, "attn", used):
        key, val = cap(target, torch.tensor(10))
        assert torch.equal(key, torch.cat((target + 10, source + 10), dim=1))
        assert torch.equal(val, torch.cat((2 * target + 10, 2 * source + 10), dim=1))
        with pytest.raises(ValueError, match="No ring K/V"):
            cap(target, 12)
    assert used == [10]
    assert cap(target, 10)[0].shape == target.shape
