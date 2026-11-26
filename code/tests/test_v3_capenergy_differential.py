from types import SimpleNamespace
import unittest

from diffusers import DDIMScheduler
import numpy as np
import torch
from torch.nn.functional import avg_pool2d

from dielinediff.methods.v3_capenergy_differential import E4_SETTINGS, cap_strength_map, differential_sdedit
from dielinediff.methods.v3_captexture import CapTextureEngine


BOX = {"length": 70.0, "width": 45.0}


def example_layout():
    shape = (40, 52)
    rectangle = (5, 5, 47, 35)
    x0, y0, x1, y1 = rectangle
    frozen = np.ones(shape, dtype=bool)
    frozen[y0:y1, x0:x1] = False
    return SimpleNamespace(latent_shape=shape, width_px=shape[1] * 8, height_px=shape[0] * 8,
                           center_rect=rectangle, known_mask=frozen)


class RecordingEngine:
    def __init__(self):
        self.topology = {"box_mm": BOX}
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.pipe = SimpleNamespace(scheduler=DDIMScheduler(
            num_train_timesteps=1000, steps_offset=1, clip_sample=False,
        ))
        self.config = {"steps": 30, "projection_tail": 0.30, "projection_interval": 3,
                       "projection_strength": 0.20}
        self.predicted = []
        self.projected = []

    def encode(self, pixels):
        latent = avg_pool2d(pixels, 8)
        return torch.cat((latent, latent.mean(dim=1, keepdim=True)), dim=1)

    def _predict(self, latent, timestep, text, condition, control):
        self.predicted.append((int(timestep), latent.detach().clone(), condition["time_ids"].clone(), control))
        return latent * 0.125 + 0.03

    def color_constraint(self, latent, epsilon, timestep):
        self.projected.append(int(timestep))
        return epsilon + 0.01


class DifferentialSDEditTests(unittest.TestCase):
    def setUp(self):
        self.layout = example_layout()
        self.rgb = np.random.default_rng(45).random((self.layout.height_px, self.layout.width_px, 3),
                                                    dtype=np.float32)
        self.text = torch.zeros((2, 4, 8))
        self.pooled = torch.ones((2, 2))

    def test_distance_field_uses_physical_latent_centers_and_leaves_ring_context(self):
        scale = cap_strength_map(self.layout, BOX, E4_SETTINGS)
        x0, y0, x1, y1 = self.layout.center_rect
        core = scale[y0:y1, x0:x1]
        self.assertEqual(scale.shape, self.layout.latent_shape)
        self.assertTrue(np.all(scale[self.layout.known_mask] == 0.85))
        np.testing.assert_array_equal(core, core[::-1])
        np.testing.assert_array_equal(core, core[:, ::-1])
        center_column = (x1 - x0) // 2
        self.assertEqual(core[1, center_column], 0.85)
        self.assertLess(core[2, center_column], 0.85)
        self.assertGreater(core[2, center_column], 0.70)
        self.assertEqual(core[-1, center_column], 0.85)
        self.assertEqual(core[(y1 - y0) // 2, center_column], 0.70)
        self.assertTrue(np.all(np.diff(core[:15, center_column]) <= 0))

    def test_uniform_map_matches_existing_sdedit_and_the_generator_stream(self):
        one, two = RecordingEngine(), RecordingEngine()
        original_generator = torch.Generator().manual_seed(321)
        differential_generator = torch.Generator().manual_seed(321)
        standard, standard_proof = CapTextureEngine._injected_sdedit(
            one, self.layout, self.rgb, 0.85, original_generator, self.text, self.pooled,
        )
        e4, proof = differential_sdedit(
            two, self.layout, self.rgb, differential_generator, self.text, self.pooled,
            settings={**E4_SETTINGS, "interior_strength": 0.85},
        )
        self.assertTrue(torch.equal(e4, standard))
        self.assertTrue(torch.equal(differential_generator.get_state(), original_generator.get_state()))
        self.assertEqual(proof["restoration_steps"], [])
        for key, expected in standard_proof.items():
            self.assertEqual(proof[key], expected)

    def test_nested_restoration_uses_same_current_time_noise_up_to_the_first_eligible_step(self):
        engine = RecordingEngine()
        generator = torch.Generator().manual_seed(7)
        before = self.rgb.copy()
        final, proof = differential_sdedit(
            engine, self.layout, self.rgb, generator, self.text, self.pooled,
            settings=E4_SETTINGS,
        )
        self.assertEqual(final.shape[-2:], self.layout.latent_shape)
        np.testing.assert_array_equal(self.rgb, before)
        self.assertEqual(proof["start_index"], 5)
        self.assertEqual(proof["executed_steps"], 25)
        self.assertEqual(proof["restoration_steps"], [6, 7, 8, 9])
        self.assertEqual(proof["projection_steps"], [21, 24, 27])
        self.assertEqual(set(proof["core_latent_step_histogram"]), {"21", "22", "23", "24", "25"})
        self.assertEqual(proof["noise_draws"], 1)

        initial = engine.encode(torch.from_numpy(before).permute(2, 0, 1)[None])
        expected_generator = torch.Generator().manual_seed(7)
        noise = torch.randn(initial.shape, generator=expected_generator)
        self.assertTrue(torch.equal(generator.get_state(), expected_generator.get_state()))
        starts = 30 - np.floor(cap_strength_map(self.layout, BOX, E4_SETTINGS) * 30).astype(int)
        for offset, (timestep, latent, time_ids, control) in enumerate(engine.predicted):
            index = 5 + offset
            self.assertEqual(timestep, int(engine.pipe.scheduler.timesteps[index]))
            self.assertEqual(time_ids[0].tolist(),
                             [self.layout.height_px, self.layout.width_px, 0, 0,
                              self.layout.height_px, self.layout.width_px])
            self.assertIsNone(control)
            if index <= 10:
                reference = engine.pipe.scheduler.add_noise(initial, noise, torch.tensor([timestep]))
                locked = torch.as_tensor(starts.ravel() >= index)
                if locked.any():
                    self.assertTrue(torch.equal(latent.flatten(-2)[..., locked], reference.flatten(-2)[..., locked]))
                if (~locked).any():
                    self.assertFalse(torch.equal(latent.flatten(-2)[..., ~locked], reference.flatten(-2)[..., ~locked]))


if __name__ == "__main__":
    unittest.main()
