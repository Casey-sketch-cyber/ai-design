"""Check that guided outpaint uses the real step index and the correct cap."""

from types import SimpleNamespace

import torch

from dielinediff.methods import v3_joint_cap_j5 as joint


def test_outpaint_routes_the_two_schedules_to_disjoint_caps(monkeypatch):
    callbacks = {}

    class Guide:
        def __init__(self, face):
            self.face = face
            self.steps = []

        def __call__(self, index, timestep, latent, epsilon):
            self.steps.append(index)
            return epsilon + (1 if self.face == "T" else 2)

        def evidence(self):
            return {"executed_steps": self.steps}

    def texture_guidance(_engine, view, _ring, *_args, **_kwargs):
        guide = Guide(view.face)
        callbacks[view.face] = guide
        return guide

    class Predict:
        def _predict(self, latent, timestep, text, condition, control):
            return torch.zeros_like(latent)

    class Probe(joint.GuidedJointRingMixin, Predict):
        pass

    monkeypatch.setattr(joint, "texture_guidance", texture_guidance)
    probe = Probe()
    times = torch.tensor([20, 10, 0])
    probe.pipe = SimpleNamespace(scheduler=SimpleNamespace(timesteps=times))
    latent = torch.zeros((1, 4, 8, 8))
    frames = [SimpleNamespace(face=face, latent_shape=(8, 8)) for face in joint.CAPS]
    reference = SimpleNamespace(frozen_latent=latent)

    def simulate():
        for face, expected in (("T", 1), ("Btm", 2)):
            assert joint.CAPS.index(face) == expected - 1
            for t in times:
                got = probe._predict(latent, t, None, None, None)
                assert torch.all(got == expected)
        return joint_result

    from dataclasses import make_dataclass
    joint_result = make_dataclass("Stage", [("proof", dict)])({})
    result = probe._guided_stage(simulate, frames, {"gamma": 0.2, "directional": False},
                                 reference, separate=True)
    assert result.proof["stage1_guidance"] == {face: {"executed_steps": [0, 1, 2]} for face in joint.CAPS}
    assert probe._joint_uplift is None
