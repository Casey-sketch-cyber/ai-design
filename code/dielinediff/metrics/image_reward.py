"""Official ImageReward v1.0 scoring on the same upright views as E01."""

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

from .learned import _square

IMAGE_REWARD_REPO = "zai-org/ImageReward"
IMAGE_REWARD_REVISION = "5736be0"
IMAGE_REWARD_CODE_REVISION = "78f8051059b34fef8e0032ccb40561c23a2ddbbb"
CLIP_CODE_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
BERT_REPO = "google-bert/bert-base-uncased"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
CHECKPOINT_SHA256 = "d71dc8d4912bc633c06ab8ef1ef6a81e8199d4369f14b47b29baf1d19de561f6"
CONFIG_SHA256 = "d8b5029cc3c0124313d17009d043f0701bbe497f3721fa4ceb818720fe665326"
BERT_FILES = {
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
}


def cache_directory():
    workspace_cache = Path(__file__).resolve().parents[3] / "outputs" / ".cache"
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", workspace_cache))
    return Path(os.environ.get("E01B_IMAGE_REWARD_HOME", cache_root / "image-reward"))


def verify_asset(path, sha256):
    if not path.is_file():
        raise FileNotFoundError(f"Missing ImageReward asset {path}. Run dielinediff.metrics.prepare_image_reward.")
    with path.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != sha256:
            raise ValueError(f"ImageReward asset failed SHA-256 verification: {path}")
    return path


class ImageRewardScorer:
    """Batch the upstream ImageReward head without changing its model or normalization."""

    def __init__(self, device="cpu", cache_dir=None):
        import torch
        from ImageReward.ImageReward import ImageReward
        from ImageReward.models.BLIP import blip_pretrain
        from transformers import BertTokenizer

        cache = Path(cache_dir) if cache_dir is not None else cache_directory()
        checkpoint = verify_asset(cache / "ImageReward.pt", CHECKPOINT_SHA256)
        config = verify_asset(cache / "med_config.json", CONFIG_SHA256)
        tokenizer_dir = cache / "bert-base-uncased"
        for name, digest in BERT_FILES.items():
            verify_asset(tokenizer_dir / name, digest)

        def local_tokenizer():
            tokenizer = BertTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
            tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
            return tokenizer

        self.device = str(device)
        # The upstream tokenizer hook otherwise requests an unpinned implicit download.
        with patch.object(blip_pretrain, "init_tokenizer", local_tokenizer):
            model = ImageReward(str(config), device=self.device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

    def score(self, images, prompts):
        import torch

        pixels = torch.stack([self.model.preprocess(_square(image)) for image in images])
        texts = self.model.blip.tokenizer(
            prompts, padding="max_length", truncation=True, max_length=35, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            rewards = self.model.score_gard(texts.input_ids, texts.attention_mask, pixels.to(self.device))
        return float(rewards.mean().item())
