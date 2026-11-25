import unittest
from types import SimpleNamespace

from diffusers import DDIMScheduler
import numpy as np
import torch
from torch.nn import functional as F

from dielinediff.methods.v3_capgeom import CapGeomEngine
from dielinediff.methods.v3_capgeom_init import harmonic_fill, sdedit_latents, sdedit_schedule


class HarmonicTests(unittest.TestCase):
    def test_manufactured_bilinear_harmonic_solution(self):
        y, x = np.mgrid[0:19, 0:22] / 24
        expected = np.stack((0.1 + 0.3 * x + 0.4 * y, 0.4 * x * y, 0.2 + 0.1 * x), axis=-1)
        known = np.ones(x.shape, dtype=bool)
        known[1:-1, 1:-1] = False
        boundary = np.where(known[..., None], expected, np.nan)
        solution, report = harmonic_fill(boundary, known)
        np.testing.assert_allclose(solution, expected, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(solution[known], expected.astype(np.float32)[known])
        self.assertTrue(np.isnan(boundary[~known]).all())
        self.assertLessEqual(report["laplace_residual_max"], 3e-8)

    def test_known_strips_remain_exact_and_virtual_corners_are_harmonic(self):
        known = np.zeros((23, 29), dtype=bool)
        known[3:7, 7:20] = True
        known[15:19, 7:20] = True
        known[7:15, 3:7] = True
        known[7:15, 20:24] = True
        y, x = np.mgrid[:23, :29]
        rgb = np.stack((33 + 5 * x + 3 * y, 35 + 2 * y, 80 + 4 * x), axis=-1).astype(np.uint8)
        source = np.where(known[..., None], rgb, 0)
        saved = source.copy()
        filled, _ = harmonic_fill(source, known)
        np.testing.assert_array_equal(source, saved)
        np.testing.assert_array_equal(filled[known], rgb[known] / np.float32(255))
        exterior = np.pad(filled, ((1, 1), (1, 1), (0, 0)), mode="edge")
        residual = (4 * filled - exterior[:-2, 1:-1] - exterior[2:, 1:-1]
                    - exterior[1:-1, :-2] - exterior[1:-1, 2:])
        self.assertLess(np.abs(residual[~known]).max(), 1e-6)
        self.assertTrue((filled[0, 0] > 0).all())
        self.assertTrue((filled[-1, -1] > 0).all())
        self.assertTrue((filled[8:14, 8:19] > 0).all())


class SDEditTests(unittest.TestCase):
    def test_uses_suffix_of_original_ddim_schedule_and_exact_forward_noise(self):
        scheduler = DDIMScheduler(steps_offset=1, clip_sample=False)
        original = torch.ones(1, 4, 3, 5) * 0.25
        for strength, expected_start, length, first in ((0.75, 8, 22, 694), (0.85, 5, 25, 793)):
            with self.subTest(strength=strength):
                start, timesteps = sdedit_schedule(scheduler, 30, strength, "cpu")
                self.assertEqual(start, expected_start)
                self.assertEqual(len(timesteps), length)
                self.assertEqual(int(timesteps[0]), first)
                self.assertEqual(int(timesteps[-1]), 1)
                self.assertEqual(scheduler.num_inference_steps, 30)
                torch.testing.assert_close(timesteps, scheduler.timesteps[start:])
                generator = torch.Generator().manual_seed(42)
                noisy, noise = sdedit_latents(scheduler, original, timesteps[0], generator)
                control = torch.randn(original.shape, generator=torch.Generator().manual_seed(42))
                alpha = scheduler.alphas_cumprod[first]
                torch.testing.assert_close(noise, control)
                torch.testing.assert_close(noisy, alpha.sqrt() * original + (1-alpha).sqrt() * control)
                torch.testing.assert_close(original, torch.full_like(original, 0.25))

    @torch.inference_mode()
    def test_cap_denoise_preserves_absolute_lut_steps_after_truncation(self):
        engine = CapGeomEngine.__new__(CapGeomEngine)
        engine.device = torch.device("cpu")
        engine.dtype = torch.float32
        engine.config = {"steps": 30, "projection_strength": 0.2, "projection_tail": 0.3,
                         "projection_interval": 3}
        scheduler = DDIMScheduler(steps_offset=1, clip_sample=False)
        engine.pipe = SimpleNamespace(scheduler=scheduler)
        calls = []
        projected = []

        def encode(pixels):
            coarse = F.avg_pool2d(pixels, 8)
            return torch.cat((coarse, coarse[:, :1]), dim=1)

        def predict(latent, timestep, _text, _condition, _control):
            calls.append(int(timestep))
            return torch.zeros_like(latent)

        def projection(_latent, epsilon, timestep):
            projected.append(int(timestep))
            return epsilon

        engine.encode = encode
        engine._predict = predict
        engine.color_constraint = projection
        known = np.zeros((4, 4), dtype=bool)
        known[[0, -1], :] = True
        rgb = np.ones((32, 32, 3), dtype=np.uint8) * 128
        layout = SimpleNamespace(width_px=32, height_px=32, known_mask=known)
        result, proof = engine._sdedit(layout, rgb, 0.75, torch.Generator().manual_seed(42),
                                       None, None, None, None)
        self.assertEqual(result.shape, (1, 4, 4, 4))
        self.assertEqual(calls, scheduler.timesteps[8:].tolist())
        self.assertEqual(proof["projection_steps"], [21, 24, 27])
        self.assertEqual(projected, scheduler.timesteps[[21, 24, 27]].tolist())


if __name__ == "__main__":
    unittest.main()
