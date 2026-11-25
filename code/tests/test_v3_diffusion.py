import unittest
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from torch import nn

from dielinediff.methods.v3 import DieLineDiffV3, regional_routing
from dielinediff.methods.v3_attention import region_attention, ring_style_attention
from dielinediff.methods.v3_circular import circular_ring_codec, horizontal_circular


def tiny_unet():
    return UNet2DConditionModel(
        in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(8, 16), layers_per_block=1,
        cross_attention_dim=12, attention_head_dim=(2, 4), norm_num_groups=4,
    ).eval()


class CircularDiffusionTests(unittest.TestCase):
    @torch.inference_mode()
    def test_frozen_unet_and_vae_are_equivariant_under_aligned_horizontal_rolls(self):
        torch.manual_seed(41)
        unet = tiny_unet()
        vae = AutoencoderKL(
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(8, 16), norm_num_groups=4, layers_per_block=1,
            sample_size=16,
        ).eval()
        vae.enable_tiling()
        latent = torch.randn(1, 4, 8, 12)
        pixels = torch.randn(1, 3, 16, 24)
        tokens = torch.randn(1, 5, 12)
        with circular_ring_codec(unet, vae):
            self.assertFalse(vae.use_tiling)
            # The shift is divisible by the downsampling factor.
            reference = unet(latent, 70, encoder_hidden_states=tokens).sample
            shifted = unet(latent.roll(4, -1), 70, encoder_hidden_states=tokens).sample
            torch.testing.assert_close(shifted, reference.roll(4, -1), rtol=2e-5, atol=3e-6)
            encoded = vae.encode(pixels).latent_dist.mode()
            shifted_encoded = vae.encode(pixels.roll(4, -1)).latent_dist.mode()
            torch.testing.assert_close(shifted_encoded, encoded.roll(2, -1), rtol=2e-5, atol=3e-6)
            decoded = vae.decode(latent).sample
            shifted_decoded = vae.decode(latent.roll(2, -1)).sample
            torch.testing.assert_close(shifted_decoded, decoded.roll(4, -1), rtol=2e-5, atol=3e-6)
        self.assertTrue(vae.use_tiling)

    def test_horizontal_patch_preserves_vertical_zero_padding_and_restores_native_mode(self):
        conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        conv.weight.data.fill_(1)
        ones = torch.ones(1, 1, 3, 5)
        native = conv(ones)
        with horizontal_circular(conv):
            wrapped = conv(ones)
        torch.testing.assert_close(wrapped[0, 0], torch.tensor([[6.0] * 5, [9.0] * 5, [6.0] * 5]))
        torch.testing.assert_close(conv(ones), native)
        self.assertEqual(native[0, 0, 0, 0].item(), 4)


class RegionalDiffusionTests(unittest.TestCase):
    @torch.inference_mode()
    def test_cross_attention_routes_queries_and_restores_native_processor(self):
        torch.manual_seed(42)
        attention = Attention(query_dim=6, cross_attention_dim=10, heads=2, dim_head=4)
        independent = attention.processor
        labels = np.array([[0, 1, 3, 4], [0, 2, 3, 4]], dtype=np.uint8)
        routing = regional_routing(labels)
        queries = torch.randn(2, labels.size, 6)
        contexts = {label: torch.randn(2, 5, 10) for label in (0, 1, 2, 3)}
        separate = {label: attention(queries, encoder_hidden_states=tokens)
                    for label, tokens in contexts.items()}
        with region_attention((attention,), routing, contexts):
            regional = attention(queries, encoder_hidden_states=contexts[0])
        for label, index in routing.groups(labels.size, queries.device):
            torch.testing.assert_close(regional[:, index], separate[label][:, index])
        self.assertIs(attention.processor, independent)
        self.assertEqual({label: index.tolist() for label, index in routing.groups(2, queries.device)},
                         {2: [0], 3: [1]})

    @torch.inference_mode()
    def test_optional_self_attention_reads_and_preserves_ring_reference(self):
        torch.manual_seed(43)
        attention = Attention(query_dim=8, heads=2, dim_head=4)
        bank = {}
        ring = torch.randn(2, 9, 8)
        cap = torch.randn(2, 4, 8)
        native = attention(cap)
        with ring_style_attention(attention, bank, capture=True):
            attention(ring)
        before = {name: tensor.clone() for name, tensor in bank.items()}
        with ring_style_attention(attention, bank, capture=False):
            styled = attention(cap)
        self.assertFalse(torch.allclose(styled, native))
        self.assertEqual(len(bank), 2)
        for name, stored in bank.items():
            self.assertTrue(torch.equal(stored, before[name]))
        torch.testing.assert_close(attention(cap), native)

    @torch.inference_mode()
    def test_flat_prediction_routes_controlnet_and_unet_together(self):
        torch.manual_seed(44)
        unet = tiny_unet()
        network = ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(4, 8, 8, 8))
        engine = DieLineDiffV3.__new__(DieLineDiffV3)
        engine.pipe = SimpleNamespace(unet=unet, scheduler=DDIMScheduler())
        engine.config = {"guidance": 5, "controlnet_scale": 0.65}
        labels = np.zeros((8, 12), dtype=np.uint8)
        labels[:, :6] = 1
        routing = regional_routing(labels)
        contexts = {label: torch.randn(2, 5, 12) for label in (0, 1)}
        image = torch.rand(2, 3, 64, 96)
        latent = torch.randn(1, 4, 8, 12)
        with region_attention((unet, network), routing, contexts):
            prediction = engine._predict(latent, torch.tensor(400), contexts[0], {}, (network, image))
        self.assertEqual(prediction.shape, latent.shape)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_cap_blending_anchors_each_denoising_step_without_writing_the_ring(self):
        class Scheduler:
            init_noise_sigma = 1

            def set_timesteps(self, steps, device):
                self.timesteps = torch.arange(steps, 0, -1, device=device)

            def add_noise(self, frozen, _noise, timestep):
                return frozen + timestep

            def step(self, _prediction, _timestep, latent, return_dict):
                return (latent + 10,)

        engine = DieLineDiffV3.__new__(DieLineDiffV3)
        engine.device = torch.device("cpu")
        engine.dtype = torch.float32
        engine.config = {"steps": 3, "guidance": 5}
        engine.pipe = SimpleNamespace(scheduler=Scheduler())
        frozen = torch.zeros(1, 4, 64, 64)
        frozen[..., :2] = 42
        before = frozen.clone()
        mask = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
        mask[..., :2] = True
        calls = []

        def predict(latent, timestep, _text, _condition, _control):
            value = latent[..., :2]
            torch.testing.assert_close(value, frozen[..., :2] + timestep)
            calls.append(timestep.item())
            return torch.zeros_like(latent)

        engine._predict = predict
        result = engine._denoise((512, 512), torch.Generator().manual_seed(1), None, None,
                                 frozen=frozen, mask=mask)
        self.assertEqual(calls, [3, 2, 1])
        self.assertTrue(torch.equal(frozen, before))
        self.assertTrue(torch.equal(result[..., :2], frozen[..., :2]))


if __name__ == "__main__":
    unittest.main()
