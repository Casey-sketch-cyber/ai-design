from types import SimpleNamespace
import unittest

from diffusers import DDIMScheduler
import numpy as np
import torch

from dielinediff.methods.v3_capgeom_init import sdedit_latents, sdedit_schedule
from dielinediff.methods.v3_captexture import CapTextureEngine
from dielinediff.methods.v3_joint_cap_blend import blended_patch, denoise_blended, feather_known_mask
from dielinediff.methods.v3_joint_cap_guidance import cap_interior, texture_guidance, uplift_clean_latent
from dielinediff.methods.v3_joint_cap_prompts import cap_local_side, joint_cap_negative, joint_cap_prompt


BOX = {"length": 70.0, "width": 45.0, "height": 160.0}
GUIDANCE = {"first_step": 6, "last_step": 22, "stride": 2, "protected_mm": 3.0,
            "reference_mm": [9.0, 12.0], "gamma": [0.20, 0.30]}


class RecordingEngine:
    def __init__(self, steps=30):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.pipe = SimpleNamespace(scheduler=DDIMScheduler(steps_offset=1, clip_sample=False))
        self.config = {"steps": steps, "projection_strength": 0.2}
        self.topology = {"box_mm": BOX}
        self.predicted = []

    def encode(self, values):
        rgb = torch.nn.functional.avg_pool2d(values, 8)
        return torch.cat((rgb, rgb.mean(dim=1, keepdim=True)), dim=1)

    def _predict(self, latent, timestep, text, condition, control):
        self.predicted.append((int(timestep), latent.detach().clone()))
        return latent * 0.25 + 0.05

    def color_constraint(self, latent, epsilon, timestep):
        return epsilon + 0.01


def _example_layout():
    latent_shape = (72, 108)
    known = np.ones(latent_shape, dtype=bool)
    known[4:68, 5:103] = False
    ring = SimpleNamespace(
        latent_shape=(128, 160),
        body_rows=(8, 120),
        face_rects={"L": (0, 8, 31, 120), "F": (31, 8, 80, 120),
                    "R": (80, 8, 111, 120), "B": (111, 8, 160, 120)},
    )
    original = SimpleNamespace(edge_sources={
        "top": ("F", "top", False, 0), "left": ("L", "top", False, 0),
        "right": ("R", "top", False, 0), "bottom": ("B", "top", True, 0),
    })
    return SimpleNamespace(latent_shape=latent_shape, center_rect=(5, 4, 103, 68),
                           known_mask=known, ring=ring, original=original, face="T")


class JointCapTests(unittest.TestCase):
    def test_j1_keeps_local_motifs_and_uniform_brushwork(self):
        side = "continue a single panoramic composition of etched woodland and log rings around the package"
        self.assertEqual(cap_local_side(side), "log rings")
        prompt = joint_cap_prompt({"side_prompt": side, "main_color": "#487355", "continuous": True})
        self.assertIn("log rings", prompt)
        self.assertIn("same stroke direction", prompt)
        self.assertNotIn("woodland", prompt)
        self.assertNotIn("seamless background", prompt)
        self.assertTrue(joint_cap_negative().endswith("plain background, empty area, blank space, blurry"))

    def test_feather_is_one_cell_inside_the_frozen_mask_and_handles_circular_x(self):
        mask = torch.tensor([[True, True, True, False],
                             [True, True, True, False],
                             [True, True, True, False]])
        hard = feather_known_mask(mask)
        soft = feather_known_mask(mask, periodic_x=True)
        torch.testing.assert_close(hard[1], torch.tensor([1.0, 1.0, 0.5, 0.0]))
        torch.testing.assert_close(soft[1], torch.tensor([0.5, 1.0, 0.5, 0.0]))
        self.assertEqual(torch.count_nonzero(soft[~mask]), 0)

    def test_same_time_frozen_noise_before_prediction_and_last_three_feather(self):
        engine = RecordingEngine(steps=6)
        scheduler = engine.pipe.scheduler
        scheduler.set_timesteps(6)
        random = torch.Generator().manual_seed(42)
        shape = (1, 4, 5, 6)
        noise = torch.randn(shape, generator=random)
        frozen = torch.randn(shape, generator=random)
        mask = torch.zeros((1, 1, 5, 6), dtype=torch.bool)
        mask[..., :3, :] = True
        seen = []
        untouched = frozen.clone()

        def before_step(index, _, latent):
            seen.append(latent.detach().clone())
            return latent

        _, proof = denoise_blended(engine, noise, scheduler.timesteps, None, None, size=(48, 40),
                                   known_clean=frozen, noise=noise, known_mask=mask, before_step=before_step)
        self.assertEqual(proof["feather_steps"], [3, 4, 5])
        self.assertTrue(torch.equal(frozen, untouched))
        for position, (timestep, actual) in enumerate(engine.predicted):
            alpha = mask if position < 3 else feather_known_mask(mask)
            expected = blended_patch(seen[position], frozen, noise, alpha, scheduler, torch.tensor(timestep))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(actual[..., 3:, :], seen[position][..., 3:, :], atol=0, rtol=0)

    def test_unguided_inpaint_loop_is_exactly_the_original_sde_schedule(self):
        original, updated = RecordingEngine(), RecordingEngine()
        prior = np.random.default_rng(2).random((48, 64, 3), dtype=np.float32)
        size = SimpleNamespace(width_px=64, height_px=48)
        a = torch.Generator().manual_seed(8)
        b = torch.Generator().manual_seed(8)
        expected, old_proof = CapTextureEngine._injected_sdedit(original, size, prior, 0.85, a, None, None)
        schedule = updated.pipe.scheduler
        start, times = sdedit_schedule(schedule, 30, 0.85, updated.device)
        hard = updated.encode(torch.as_tensor(prior).permute(2, 0, 1)[None])
        initial, _ = sdedit_latents(schedule, hard, times[0], b)
        actual, proof = denoise_blended(updated, initial, times, None, None, size=(64, 48), start_index=start)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(a.get_state(), b.get_state()))
        self.assertEqual(proof["projection_steps"], old_proof["projection_steps"])

    def test_guidance_preserves_means_protected_zone_and_high_energy_channel(self):
        xs = torch.arange(30, dtype=torch.float32)
        rows = ((xs % 2) * 2 - 1)[None, :].expand(24, -1)
        clean = torch.stack((0.1 * rows, 1.0 * rows))[None]
        target = torch.full((1, 2, 1, 1), 0.3)
        mask = torch.zeros((1, 1, 24, 30), dtype=torch.bool)
        mask[..., 4:-4, 4:-4] = True
        delta, active = uplift_clean_latent(clean, target, mask, gamma=0.2,
                                            highpass_radius=(1, 1), local_radius=(2, 2), direction="isotropic")
        self.assertGreater(float(delta[:, 0].abs().max()), 0.005)
        self.assertFalse(active[:, 1].any())
        self.assertFalse(torch.any(delta[~mask.expand_as(delta)]))
        torch.testing.assert_close(delta.mean(dim=(-2, -1)), torch.zeros((1, 2)), atol=1e-7, rtol=0)
        self.assertGreater((clean + delta).std(dim=(-2, -1))[0, 0], clean.std(dim=(-2, -1))[0, 0])

    def test_directional_mode_restricts_to_native_side_and_preserves_row_mean(self):
        layout = _example_layout()
        engine = RecordingEngine()
        frozen = torch.Generator().manual_seed(18)
        ring = torch.randn((1, 4, 128, 160), generator=frozen)
        manual_stop = np.ones(layout.latent_shape, dtype=bool)
        manual_stop[-25:] = False
        guide = texture_guidance(engine, layout, ring, 0.20, GUIDANCE,
                                 editable_mask=manual_stop, direction="tangent")
        mask, _ = cap_interior(layout, BOX, 3.0, manual_stop)
        self.assertFalse(mask[-25:].any())
        self.assertTrue(mask.any())
        self.assertFalse(mask[layout.known_mask].any())
        steps = engine.pipe.scheduler
        steps.set_timesteps(30)
        initial = torch.randn((1, 4, *layout.latent_shape), generator=frozen) * 0.05
        epsilon = torch.randn(initial.shape, generator=frozen) * 0.05
        self.assertIs(guide(7, steps.timesteps[7], initial, epsilon), epsilon)
        actual = guide(8, steps.timesteps[8], initial, epsilon)
        self.assertFalse(torch.equal(actual, epsilon))
        protected = ~torch.as_tensor(mask.ravel())
        torch.testing.assert_close(actual.flatten(-2)[..., protected],
                                   epsilon.flatten(-2)[..., protected], atol=0, rtol=0)
        torch.testing.assert_close((actual - epsilon).mean(dim=-1),
                                   torch.zeros_like(actual[..., 0]), atol=1e-7, rtol=0)
        proof = guide.evidence()
        self.assertEqual(proof["executed_steps"], [8])
        self.assertEqual(proof["reference_sides"], ["F"])
        self.assertEqual(proof["mean_preservation"], "per_row_and_channel")


if __name__ == "__main__":
    unittest.main()
