import unittest
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import DDIMScheduler
from torch.nn import functional as F

from dielinediff.methods.v3_captexture_ilvr import ILVRCapMixin, latent_lowpass, refine_cap_latent


class TracedDDIM(DDIMScheduler):
    def __init__(self):
        super().__init__(steps_offset=1, clip_sample=False, set_alpha_to_one=False,
                         timestep_spacing="leading", beta_schedule="scaled_linear",
                         beta_start=0.00085, beta_end=0.012)
        self.noises = []
        self.proposals = []

    def add_noise(self, original_samples, noise, timesteps):
        response = super().add_noise(original_samples, noise, timesteps)
        self.noises.append((int(timesteps[0]), noise.clone(), response.clone()))
        return response

    def step(self, model_output, timestep, sample, **kwargs):
        response = super().step(model_output, timestep, sample, **kwargs)
        self.proposals.append(response[0].clone())
        return response


class StubEngine(ILVRCapMixin):
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.config = {"steps": 30, "projection_tail": 0.30, "projection_interval": 3,
                       "projection_strength": 0.20}
        self.pipe = SimpleNamespace(scheduler=TracedDDIM())
        self.inputs = []
        self.projections = []
        self.prior = None

    def encode(self, pixels):
        reduced = F.avg_pool2d(pixels, 8)
        self.prior = torch.cat((reduced, reduced[:, :1]), dim=1)
        return self.prior

    def _predict(self, latent, timestep, _text, _condition, _control):
        self.inputs.append(latent.clone())
        return torch.zeros_like(latent)

    def color_constraint(self, _latent, epsilon, timestep):
        self.projections.append(int(timestep))
        return epsilon


class ILVRTests(unittest.TestCase):
    def test_full_canvas_filter_and_core_only_write(self):
        proposal = torch.arange(96, dtype=torch.float32).reshape(1, 1, 8, 12) / 100
        reference = torch.full_like(proposal, 0.4)
        reference[..., :4, :4] = 0.9
        before = proposal.clone()
        core = (2, 1, 10, 7)
        refined = refine_cap_latent(proposal, reference, 4, core)
        outside = torch.ones((8, 12), dtype=torch.bool)
        outside[1:7, 2:10] = False
        desired = proposal + latent_lowpass(reference, 4) - latent_lowpass(proposal, 4)
        torch.testing.assert_close(refined[..., 1:7, 2:10], desired[..., 1:7, 2:10])
        self.assertTrue(torch.equal(refined[..., outside], proposal[..., outside]))
        self.assertTrue(torch.equal(proposal, before))
        self.assertFalse(torch.equal(refined, proposal))

    @torch.inference_mode()
    def test_ddim_next_step_clean_endpoint_same_draw_and_unchanged_lut(self):
        core = (8, 8, 56, 56)
        outside = torch.ones((64, 64), dtype=torch.bool)
        outside[8:56, 8:56] = False
        pixels = np.full((512, 512, 3), 0.25, dtype=np.float32)
        for factor, fraction, guided in ((4, 1.0, 30), (4, 0.70, 21),
                                         (8, 1.0, 30), (8, 0.70, 21)):
            with self.subTest(factor=factor, fraction=fraction):
                engine = StubEngine()
                control = torch.Generator().manual_seed(42)
                expected_noise = torch.randn((1, 4, 64, 64), generator=control)
                generator = torch.Generator().manual_seed(42)
                result, proof = engine._ilvr_denoise((512, 512), generator, None, None,
                                                    pixels, factor, fraction, core_rect=core)
                scheduler = engine.pipe.scheduler
                self.assertEqual(len(engine.inputs), 30)
                self.assertEqual(proof["guided_steps"], guided)
                self.assertEqual(proof["full_steps"], 30)
                target_times = scheduler.timesteps[1:guided + 1].tolist()
                self.assertEqual([n[0] for n in scheduler.noises], target_times)
                self.assertEqual(proof["guided_target_timesteps"],
                                 target_times + ([None] if guided == 30 else []))
                torch.testing.assert_close(engine.inputs[0], expected_noise)
                self.assertTrue(torch.equal(generator.get_state(), control.get_state()))
                self.assertEqual(proof["projection_steps"], [21, 24, 27])
                self.assertEqual(engine.projections, scheduler.timesteps[[21, 24, 27]].tolist())
                for index, (_, used_noise, q) in enumerate(scheduler.noises):
                    self.assertTrue(torch.equal(used_noise, expected_noise))
                    expected = refine_cap_latent(scheduler.proposals[index], q, factor, core)
                    torch.testing.assert_close(engine.inputs[index + 1], expected)
                    self.assertTrue(torch.equal(engine.inputs[index + 1][..., outside],
                                                scheduler.proposals[index][..., outside]))
                final_proposal = scheduler.proposals[-1]
                expected = (refine_cap_latent(final_proposal, engine.prior, factor, core)
                            if guided == 30 else final_proposal)
                torch.testing.assert_close(result, expected)
                self.assertTrue(torch.equal(result[..., outside], final_proposal[..., outside]))

    def test_factor_requires_a_registered_latent_grid(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            latent_lowpass(torch.zeros((1, 4, 11, 12)), 4)


if __name__ == "__main__":
    unittest.main()
