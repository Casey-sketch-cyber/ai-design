import unittest
from types import SimpleNamespace
from unittest.mock import patch

from diffusers import DDIMScheduler
from PIL import Image
import torch
from torch import nn

from dielinediff.dieline.geometry import build_geometry
from dielinediff.methods.v3_geometry import build_ring_layout
from dielinediff.methods.v3_panel_fix_outpaint import (
    REPAIR, PanelFixOutpaintMixin, panel_noise_seed, registered_outpaint_noise,
)


class TestPanelNoise(unittest.TestCase):
    def test_seed_and_draw_are_stable_and_order_independent(self):
        self.assertEqual(panel_noise_seed(42, "E01-003", "T"), 12092639793112403083)
        keys = [(seed, f"E01-{n:03d}", face) for seed in (42, 1234)
                for n in range(1, 11) for face in ("T", "Btm")]
        derived = [panel_noise_seed(*key) for key in keys]
        self.assertEqual(len(set(derived)), len(derived))
        engine = SimpleNamespace(device="cpu", dtype=torch.float32)
        a, pa = registered_outpaint_noise(engine, 42, "E01-003", "T", (1280, 640))
        b, pb = registered_outpaint_noise(engine, 42, "E01-003", "Btm", (1280, 640))
        again, proof = registered_outpaint_noise(engine, 42, "E01-003", "T", (1280, 640))
        self.assertTrue(torch.equal(a, again))
        self.assertFalse(torch.equal(a, b))
        self.assertEqual(pa, proof)
        self.assertNotEqual(pa["noise_sha256"], pb["noise_sha256"])
        self.assertEqual(pa["draw_count"], 1)

    def test_does_not_allow_a_fallback_noise_domain(self):
        with self.assertRaises(ValueError):
            panel_noise_seed(42, "E01-003", "T", domain="find-a-passing-sample")


class TestPanelSampler(unittest.TestCase):
    def test_registered_known_patch_and_geometry_remain_native(self):
        topo = build_geometry(pixels_per_mm=6.4).topology
        ring = build_ring_layout(topo, 1280, 1024)
        source = torch.arange(128).reshape(1, 1, 128, 1).expand(1, 4, 128, 160).float().contiguous()
        pixels = Image.new("RGB", (1280, 1024))
        r0 = SimpleNamespace(layout=ring, frozen_latent=source, image=pixels,
                             rng_state=torch.Generator().manual_seed(42).get_state())

        class Canvas(nn.Module):
            use_tiling = False

            def disable_tiling(self):
                self.use_tiling = False

            def enable_tiling(self, value):
                self.use_tiling = value

        class Engine(PanelFixOutpaintMixin):
            device, dtype = "cpu", torch.float32
            config = {"steps": 30}
            topology = topo
            pipe = SimpleNamespace(scheduler=DDIMScheduler(), vae=Canvas())

            def _base_unet(self):
                return nn.Identity()

            def _panel_outpaint_conditions(self, brief):
                return {0: object()}, object(), {}

            def _rgb(self, latent):
                return Image.new("RGB", (1280, 640), "#404040")

        known_sets = []
        test = self

        def sampler(engine, latent, timesteps, text, pooled, **kwargs):
            known = kwargs["known_mask"]
            indexes = torch.nonzero(known[:, 0]).ravel()
            reference = kwargs["known_clean"][0, 0, :, 0][indexes]
            known_sets.append((indexes.tolist(), reference.tolist()))
            test.assertEqual(kwargs["feather_last"], 3)
            test.assertTrue(kwargs["periodic_x"])
            test.assertEqual(len(timesteps), 22)
            test.assertEqual(tuple(kwargs["noise"].shape), (1, 4, 80, 160))
            return latent, {"feather_steps": [19, 20, 21]}

        with patch("dielinediff.methods.v3_panel_fix_outpaint.denoise_blended", sampler):
            result = Engine().generate_panel_outpaint({"id": "E01-003"}, 42, r0, options=REPAIR)
        self.assertEqual(known_sets[0], (list(range(40, 80)), list(range(8, 48))))
        self.assertEqual(known_sets[1], (list(range(40)), list(range(80, 120))))
        self.assertIs(result.ring_image, pixels)
        self.assertIs(result.ring_latent, source)
        self.assertEqual(result.caps["T"].size, (392, 248))
        self.assertEqual(result.caps["Btm"].size, (392, 248))
        self.assertFalse(result.proof["vertical_transform_used"])
        self.assertNotEqual(result.proof["outpaint"]["T"]["noise"]["noise_sha256"],
                            result.proof["outpaint"]["Btm"]["noise"]["noise_sha256"])
