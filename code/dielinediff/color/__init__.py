"""CMYK profile round-trip and differentiable projection on the sRGB cube."""

import hashlib
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageCms


ECI_PROFILE_NAME = "ISOcoated_v2_eci.icc"
ECI_PROFILE_URL = (
    "https://sources.debian.org/data/non-free/i/icc-profiles/2.1-2/ISOcoated_v2_eci.icc"
)
ECI_PROFILE_SHA256 = "128dc02f7246cc3807af0323695379f64151a8f27a587736acc59f8b6ce894b8"


def fetch_eci_profile(cache_dir):
    """Fetch the E01 print profile from the Debian mirror of the ECI release."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    profile = cache_dir / ECI_PROFILE_NAME
    if profile.exists() and hashlib.sha256(profile.read_bytes()).hexdigest() == ECI_PROFILE_SHA256:
        return profile
    with TemporaryDirectory(dir=cache_dir) as temporary:
        pending = Path(temporary) / ECI_PROFILE_NAME
        with urlopen(ECI_PROFILE_URL, timeout=60) as response, pending.open("wb") as file:
            copyfileobj(response, file)
        if hashlib.sha256(pending.read_bytes()).hexdigest() != ECI_PROFILE_SHA256:
            raise ValueError("Downloaded ISO Coated v2 profile failed SHA-256 verification.")
        pending.replace(profile)
    return profile


class ICCRoundtrip:
    """Measure color loss using the ICC transforms, separately from the LUT."""

    def __init__(self, cmyk_path):
        cmyk = ImageCms.getOpenProfile(str(cmyk_path))
        srgb = ImageCms.createProfile("sRGB")
        flags = ImageCms.Flags.BLACKPOINTCOMPENSATION
        intent = ImageCms.Intent.RELATIVE_COLORIMETRIC
        self.forward = ImageCms.buildTransform(srgb, cmyk, "RGB", "CMYK", intent, flags)
        self.reverse = ImageCms.buildTransform(cmyk, srgb, "CMYK", "RGB", intent, flags)

    def __call__(self, rgb):
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim < 2 or rgb.shape[-1] != 3:
            raise ValueError("Expected RGB values in an array ending in 3 channels.")
        # Keep the Pillow image within its dimension and memory limits.
        flattened = rgb.reshape(-1, 3)
        projected = np.empty_like(flattened)
        for start in range(0, len(flattened), 32768):
            chunk = flattened[start:start + 32768]
            image = Image.fromarray(chunk.reshape(1, len(chunk), 3))
            cmyk = ImageCms.applyTransform(image, self.forward)
            projected[start:start + len(chunk)] = np.asarray(
                ImageCms.applyTransform(cmyk, self.reverse)
            )[0]
        return projected.reshape(rgb.shape)


def make_lut(profile_path, size=33):
    if size < 2:
        raise ValueError("LUT size must be at least 2.")
    levels = np.rint(np.linspace(0, 255, size)).astype(np.uint8)
    values = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1)
    return ICCRoundtrip(profile_path)(values).astype(np.float32) / 255


def save_lut(profile_path, output, size=33):
    output = Path(output)
    profile_path = Path(profile_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        np.save(file, make_lut(profile_path, size))
    metadata = {
        "profile_path": str(profile_path),
        "profile_name": ImageCms.getProfileDescription(str(profile_path)).strip(),
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "grid_size": size,
        "intent": "relative colorimetric",
        "black_point_compensation": True,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def project_rgb(rgb, lut):
    """Project a channel-last [0, 1] torch tensor with trilinear interpolation."""
    import torch
    from torch.nn import functional as F

    if rgb.ndim < 2 or rgb.shape[-1] != 3 or not rgb.is_floating_point():
        raise ValueError("Expected a floating-point RGB tensor ending in 3 channels.")
    table = torch.as_tensor(lut, device=rgb.device, dtype=torch.float32)
    if table.ndim != 4 or len(set(table.shape[:3])) != 1 or table.shape[-1] != 3:
        raise ValueError("LUT must have shape (size, size, size, 3).")
    table = table.permute(3, 0, 1, 2).unsqueeze(0)
    # grid_sample uses W, H, D coordinates, whereas the ICC table axes are R, G, B.
    grid = rgb.float().flip(-1).reshape(1, 1, 1, -1, 3).mul(2).sub(1)
    samples = F.grid_sample(table, grid, mode="bilinear", padding_mode="border",
                            align_corners=True)
    return samples.reshape(3, -1).T.reshape(rgb.shape).to(rgb.dtype)
