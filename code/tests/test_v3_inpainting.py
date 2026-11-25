import unittest

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from torch.nn import functional as F

from dielinediff.methods.v3_attention import region_attention
from dielinediff.methods.v3_inpainting import (
    ORIGINAL_VAE, UPSTREAM, prepare_inpaint_condition, predict_inpaint,
)
from dielinediff.methods.v3 import regional_routing


class InpaintingTests(unittest.TestCase):
    def test_pinned_digests_are_sha256(self):
        for value in (*[checksum for _, checksum in UPSTREAM.values()], *ORIGINAL_VAE.values()):
            self.assertRegex(value, r"^[a-f0-9]{64}$")

    def test_pixel_condition_masks_before_encoding_and_preserves_known_rgb(self):
        pixels = torch.arange(1 * 3 * 16 * 24, dtype=torch.float32).reshape(1, 3, 16, 24) / 1151
        original = pixels.clone()
        known = np.array([[True, False, False], [False, True, True]])
        before = known.copy()
        saw = []

        def encode(image):
            normalized = image * 2 - 1
            saw.append(normalized)
            pooled = F.avg_pool2d(normalized, 8)
            return torch.cat((pooled, pooled.mean(dim=1, keepdim=True)), dim=1)

        mask, context = prepare_inpaint_condition(pixels, known, encode)
        expected = torch.tensor([[[[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]]])
        torch.testing.assert_close(mask, expected)
        unknown_pixels = expected.bool().repeat_interleave(8, -2).repeat_interleave(8, -1).expand_as(pixels)
        self.assertEqual(torch.count_nonzero(saw[0][unknown_pixels]).item(), 0)
        torch.testing.assert_close(saw[0][~unknown_pixels], (original * 2 - 1)[~unknown_pixels])
        self.assertEqual(tuple(context.shape), (1, 4, 2, 3))
        torch.testing.assert_close(pixels, original)
        np.testing.assert_array_equal(known, before)

    @torch.inference_mode()
    def test_tiny_nine_channel_diffusers_forward_with_regional_conditioning(self):
        torch.manual_seed(44)
        model = UNet2DConditionModel(
            in_channels=9, out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(8, 16), layers_per_block=1,
            cross_attention_dim=12, attention_head_dim=(2, 4), norm_num_groups=4,
            addition_embed_type="text_time", addition_time_embed_dim=4,
            projection_class_embeddings_input_dim=32,
        ).eval()
        latent = torch.randn(2, 4, 8, 12)
        paint = torch.ones(1, 1, 8, 12)
        paint[:, :, :, :3] = 0
        reference = torch.randn(1, 4, 8, 12)
        conditions = {0: torch.randn(2, 5, 12), 1: torch.randn(2, 5, 12)}
        kwargs = {"text_embeds": torch.randn(2, 8),
                  "time_ids": torch.tensor([[64, 96, 0, 0, 64, 96]] * 2)}
        labels = np.asarray(paint[0, 0].numpy(), dtype=np.uint8)
        captured = []
        with model.register_forward_pre_hook(lambda _module, args: captured.append(args[0].clone())):
            with region_attention((model,), regional_routing(labels), conditions):
                sample = predict_inpaint(model, latent, torch.tensor(350), conditions[0], kwargs,
                                         paint, reference)
        self.assertEqual(tuple(sample.shape), (2, 4, 8, 12))
        self.assertTrue(torch.isfinite(sample).all())
        torch.testing.assert_close(captured[0][:, :4], latent)
        torch.testing.assert_close(captured[0][:, 4:5], paint.expand(2, -1, -1, -1))
        torch.testing.assert_close(captured[0][:, 5:], reference.expand(2, -1, -1, -1))


if __name__ == "__main__":
    unittest.main()
