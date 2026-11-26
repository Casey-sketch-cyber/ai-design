"""Trace the J4b upper-cap artifact before changing the registered sampler."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .e01b import file_hash
from .e02_capgeom import ring as frozen_ring
from .e02_joint_cap import Experiment
from .e02_joint_cap_generate import _joint_engine
from .methods.v3_circular import circular_ring_codec
from .methods.v3_geometry import CAPS, SIDES, build_ring_layout
from .methods.v3_joint_cap_blend import denoise_blended, feather_known_mask
from .methods.v3_joint_cap_geometry import NATIVE, build_outpaint_layout
from .methods.v3_joint_cap_ring import J4B_OPTIONS, _outpaint_weights, _pure_noise, _set_timesteps, _temporary_config
from .methods.v3_ring_m2 import M2Routing, m2_attention


def _save_json(path, data):
    intermediate = path.with_suffix(".tmp")
    intermediate.write_text(json.dumps(data, indent=2) + "\n")
    intermediate.replace(path)


def _registered_view(topology, ring, face, options, flip=False):
    view = build_outpaint_layout(topology, ring, face, int(options["strip_height_px"]),
                                 int(options["cap_buffer_px"]))
    height, width = view.size_px[1] // 8, view.size_px[0] // 8
    known = np.zeros((height, width), dtype=bool)
    low, high = view.known_rows_latent
    known[low:high] = True
    source = np.full(height, -1, dtype=np.int64)
    source[low:high] = np.arange(*view.source_rows_latent)
    weights = _outpaint_weights(ring, face, height)
    rect = view.cap_rect_px
    if flip:
        known, source, weights = known[::-1].copy(), source[::-1].copy(), weights[:, ::-1].copy()
        left, top, right, bottom = rect
        rect = left, view.size_px[1] - bottom, right, view.size_px[1] - top
    return view, known, source, weights, rect


def geometry_report(topology, ring, options):
    proof = {}
    for face, flip, name in (("T", False, "native_T"), ("Btm", False, "native_Btm"),
                             ("T", True, "flipped_T")):
        view, known, source, weights, rect = _registered_view(topology, ring, face, options, flip)
        height, width = known.shape
        np.testing.assert_allclose(weights.sum(axis=0), 1)
        assert all(np.unique(row).size == 1 for row in known)
        assert known.sum() == (view.known_rows_latent[1] - view.known_rows_latent[0]) * width
        l, t, r, b = (v // 8 for v in rect)
        assert not known[t:b, l:r].any()
        used = source[source >= 0]
        assert len(np.unique(used)) == len(used)
        assert all(ring.body_rows[0] <= i < ring.body_rows[1] for i in used)
        feather = feather_known_mask(torch.from_numpy(known), periodic_x=True).numpy()
        proof[name] = {
            "cap_face": face,
            "size_px": list(view.size_px),
            "native_side": NATIVE[face][0],
            "cap_crop_in_model_coordinates_px": list(rect),
            "cap_crop_in_atlas_orientation_px": list(view.cap_rect_px),
            "vertical_flip": flip,
            "first_known_row": int(np.flatnonzero(known.any(axis=-1))[0]),
            "last_known_row": int(np.flatnonzero(known.any(axis=-1))[-1]),
            "rows": [
                {
                    "row": i,
                    "input_from_r0_latent_row": int(source[i]) if source[i] >= 0 else None,
                    "known_width": int(known[i].sum()),
                    "hard_alpha_unique": np.unique(known[i].astype(float)).tolist(),
                    "last_three_alpha_unique": np.unique(feather[i]).tolist(),
                    "routed_regions": {
                        side: weights[:, i, (ring.face_rects[side][0] + ring.face_rects[side][2]) // 2].tolist()
                        for side in SIDES
                    },
                } for i in range(height)
            ],
        }
    t, b, flip = proof["native_T"], proof["native_Btm"], proof["flipped_T"]
    assert (t["first_known_row"], t["last_known_row"]) == (40, 79)
    assert (b["first_known_row"], b["last_known_row"]) == (0, 39)
    assert (flip["first_known_row"], flip["last_known_row"]) == (0, 39)
    assert [row["input_from_r0_latent_row"] for row in flip["rows"]][::-1] == [
        row["input_from_r0_latent_row"] for row in t["rows"]
    ]
    return proof


def _orient(image, flip):
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM) if flip else image


def dark_track(image, *, bottom, width_mm=45):
    """Flag original rows with >=80 percent pixels at uint8 RGB < 55."""
    rgb = np.asarray(image)
    depth = (np.arange(len(rgb)) + 0.5) * width_mm / len(rgb)
    if bottom:
        depth = depth[::-1]
    cover = np.all(rgb < 55, axis=-1).mean(axis=1)
    marker = (cover >= 0.80) & (depth < 9)
    marked = depth[marker]
    return {"near_dark_width_max": float(cover[depth < 9].max()),
            "track_rows": int(marker.sum()),
            "track_depth_mm": [float(marked.min()), float(marked.max())] if len(marked) else None}


def _finite_stats(array):
    finite = np.isfinite(array)
    values = array[finite].astype(np.float64, copy=False)
    return {
        "finite_fraction": float(finite.mean()),
        "minimum": float(values.min()) if len(values) else None,
        "maximum": float(values.max()) if len(values) else None,
        "mean": float(values.mean()) if len(values) else None,
        "std": float(values.std()) if len(values) else None,
    }


def _probe_bands(array, width_mm, near_bottom, *, latent):
    """Record channel distributions at the physically aligned dark rows."""
    h = array.shape[-2]
    depth = (np.arange(h) + 0.5) * width_mm / h
    if near_bottom:
        depth = depth[::-1]
    window = (depth >= 2.8) & (depth <= 6)
    section = array[..., window, :]
    detail = {
        "all_channels": [_finite_stats(c) for c in array],
        "dark_interval_channels": [_finite_stats(c) for c in section],
        "per_row_minimum": np.min(array, axis=(0, 2)).tolist(),
        "per_row_maximum": np.max(array, axis=(0, 2)).tolist(),
        "per_row_mean": np.mean(array, axis=(0, 2)).tolist(),
    }
    if not latent:
        detail["dark_pixels_preclamp_all_channels_below_zero_fraction"] = float(np.all(section < 0, axis=0).mean())
        detail["dark_pixels_preclamp_all_channels_below_55_fraction"] = float(np.all(section < 55 / 255, axis=0).mean())
    return detail


def _decode_probe(engine, latent, view, face, flip, width_mm):
    """Replay the original VAE normalization and retain float/preclamp statistics."""
    layers = engine.pipe.vae
    decoded = layers.decode(latent / layers.config.scaling_factor, return_dict=False)[0].float().div(2).add(0.5)
    latent_view = latent.flip(-2) if flip else latent
    decoded_view = decoded.flip(-2) if flip else decoded
    l, t, r, b = view.cap_rect_px
    raw_latent = latent_view[0, :, t // 8:b // 8, l // 8:r // 8].float().cpu().numpy()
    raw_rgb = decoded_view[0, :, t:b, l:r].contiguous().cpu().numpy()
    original = decoded.clamp(0, 1)[0].permute(1, 2, 0)
    if not torch.isfinite(original).all():
        raise RuntimeError("Nonfinite decoded pixels before uint8 conversion")
    image = _orient(Image.fromarray(original.mul(255).round().to(torch.uint8).cpu().numpy()), flip)
    return image, {
        "latent_full_finite_fraction": float(torch.isfinite(latent).float().mean()),
        "linear_rgb_full_finite_fraction": float(torch.isfinite(decoded).float().mean()),
        "latent": _probe_bands(raw_latent, width_mm, face == "T", latent=True),
        "linear_rgb": _probe_bands(raw_rgb, width_mm, face == "T", latent=False),
    }


@torch.inference_mode()
def trace_branch(engine, prepared, face, noise_native, contexts, pooled, options, *, flip=False,
                 directory, width_mm, no_lut=False, numeric=False):
    ring = prepared.layout
    view, known_np, source, weights, _ = _registered_view(engine.topology, ring, face, options, flip)
    network = engine._base_unet()
    noise = noise_native.flip(-2) if flip else noise_native
    noise = noise.contiguous()
    reference = torch.zeros_like(noise)
    index = torch.as_tensor(source[source >= 0], device=engine.device)
    frozen = prepared.frozen_latent.to(engine.device, engine.dtype)
    if tuple(frozen.shape[-2:]) != ring.latent_shape:
        raise ValueError("Archived R0 latents and ring geometry disagree")
    row = torch.as_tensor(np.flatnonzero(source >= 0), device=engine.device)
    reference.index_copy_(-2, row, frozen.index_select(-2, index))
    known = torch.as_tensor(known_np, device=engine.device)
    routes = M2Routing(weights)
    out = directory
    out.mkdir(parents=True, exist_ok=True)
    steps = int(options["steps"])
    report = {}
    samples = []
    updates = {"steps": steps, "guidance": options["guidance"]}
    if no_lut:
        updates["projection_strength"] = 0
    with _temporary_config(engine, **updates), circular_ring_codec(network, engine.pipe.vae):
        scheduler = _set_timesteps(engine, steps)
        if scheduler.config.prediction_type != "epsilon":
            raise ValueError("x0 diagnostic expects the frozen epsilon-prediction DDIM schedule")

        def capture(index, timestep, latent, epsilon):
            if index % 3:
                return epsilon
            a = scheduler.alphas_cumprod[timestep].float()
            x0 = ((latent.float() - (1 - a).sqrt() * epsilon.float()) / a.sqrt()).to(engine.dtype)
            float_probe = {}
            if numeric:
                floating, float_probe = _decode_probe(engine, x0, view, face, flip, width_mm)
                rgb = floating.crop(view.cap_rect_px)
            else:
                rgb = _orient(engine._rgb(x0), flip).crop(view.cap_rect_px)
            file = out / f"x0-{index:02d}.png"
            rgb.save(file)
            stats = dark_track(rgb, bottom=face == "T", width_mm=width_mm)
            samples.append({"index_zero_based": index, "timestep": int(timestep),
                            "crop_png": file.name, **stats, **({"float_probe": float_probe} if numeric else {})})
            return epsilon

        with m2_attention((network,), routes, contexts, (1,)):
            latent, report = denoise_blended(
                engine, noise * scheduler.init_noise_sigma, scheduler.timesteps, contexts[0], pooled,
                size=view.size_px, known_clean=reference, noise=noise, known_mask=known, periodic_x=True,
                feather_last=int(options["feather_last"]), epsilon_transform=capture,
            )
        final_float = {}
        if numeric:
            image, final_float = _decode_probe(engine, latent, view, face, flip, width_mm)
        else:
            image = _orient(engine._rgb(latent), flip)
        alternate_vae = _orient(engine._rgb(latent.flip(-2).contiguous()), not flip)
    crop = image.crop(view.cap_rect_px)
    vae_only_flip_crop = alternate_vae.crop(view.cap_rect_px)
    image.save(out / "final-canvas.png")
    crop.save(out / "final-cap.png")
    vae_only_flip_crop.save(out / "same-latent-vae-flip-cap.png")
    with (out / "final-latents.npy").open("wb") as handle:
        np.save(handle, latent.detach().cpu().numpy(), allow_pickle=False)
    return crop, {"denoise": report, "saved_pre_lut_x0": samples, "no_lut": no_lut,
                  "final": {**dark_track(crop, bottom=face == "T", width_mm=width_mm),
                            **({"float_probe": final_float} if numeric else {})},
                  "same_latent_vae_flip": dark_track(vae_only_flip_crop, bottom=face == "T",
                                                     width_mm=width_mm)}


def _compare_archive(top, bottom, archived):
    pairs = (("T", top), ("Btm", bottom))
    proof = {}
    for name, image in pairs:
        path = archived.with_suffix(f".cap-{name}.png")
        with Image.open(path) as reference:
            pixels = np.asarray(reference.convert("RGB"))
        delta = np.abs(pixels.astype(np.int16) - np.asarray(image).astype(np.int16))
        proof[name] = {"pixel_identical": not bool(np.any(delta)),
                       "differing_pixels": int(np.any(delta != 0, axis=-1).sum()),
                       "maximum_abs_channel_difference": int(delta.max()),
                       "archived_sha256": file_hash(path)}
    return proof


def _contact(folder, report):
    headers = [
        ("J4b top", "native_T"), ("J4b bottom", "native_Btm"), ("Top flip, then restore", "flipped_T"),
    ]
    frames = ["x0-00.png", "x0-06.png", "x0-12.png", "x0-15.png", "x0-18.png",
              "x0-21.png", "final-cap.png", "same-latent-vae-flip-cap.png"]
    step_label = {f: f"{int(f[3:5])+1:02d} / x0 before LUT" for f in frames if f.startswith("x0")}
    step_label.update({"final-cap.png": "Final crop", "same-latent-vae-flip-cap.png": "Same latent, VAE flipped"})
    font = ImageFont.load_default(size=15)
    with Image.open(folder / "native_T/final-cap.png") as image:
        w, h = image.size
    for page, names in enumerate((frames[:4], frames[4:]), 1):
        board = Image.new("RGB", (3 * (w + 16) + 4, len(names) * (h + 25) + 41), "white")
        draw = ImageDraw.Draw(board)
        for i, (title, key) in enumerate(headers):
            x = 8 + i * (w + 16)
            draw.text((x, 8), title, font=font, fill="#1a2832")
            for j, frame in enumerate(names):
                y = 34 + j * (h + 25)
                with Image.open(folder / key / frame) as art:
                    board.paste(art, (x, y + 17))
                draw.text((x, y), step_label[frame], font=font, fill="#1a2832")
        board.save(folder / f"x0-contact-{page}.png", optimize=True)
    if "native_T_second_noise" in report:
        variants = [("Original T", "native_T"), ("Same noise and vertically flipped", "flipped_T"),
                    ("Original orientation, second noise draw", "native_T_second_noise")]
        sources = ("x0-00.png", "x0-03.png", "x0-12.png", "final-cap.png")
        crop_h = min(h, 112)
        tile = Image.new("RGB", (3 * (w + 18) + 8, 42 + len(sources) * (crop_h + 27)), "white")
        labels = ImageDraw.Draw(tile)
        for i, (title, key) in enumerate(variants):
            x = 8 + i * (w + 18)
            labels.text((x, 7), title, font=font, fill="#1a2832")
            for j, name in enumerate(sources):
                y = 34 + j * (crop_h + 27)
                labels.text((x, y), step_label.get(name, "04 / x0 before LUT"), font=font, fill="#1a2832")
                with Image.open(folder / key / name) as im:
                    tile.paste(im.crop((0, h - crop_h, w, h)), (x, y + 17))
        tile.save(folder / "x0-T-control-strips.png", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="E01-003")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--with-no-lut", action="store_true")
    parser.add_argument("--with-noise-swap", action="store_true")
    parser.add_argument("--float-statistics", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    topology = json.loads((args.root / "outputs/e01b/geometry/topology.json").read_text())
    ring = build_ring_layout(topology, 1280, 1024)
    _save_json(args.output / "registration-rows.json", geometry_report(topology, ring, J4B_OPTIONS))
    if not args.gpu:
        return
    experiment = Experiment(args.root)
    brief = next(brief for brief in experiment.briefs if brief["id"] == args.case)
    engine = _joint_engine(experiment, args.device)
    prepared, ring_proof, _, _ = frozen_ring(experiment, engine, brief, 42)
    options = {**J4B_OPTIONS, **experiment.options["joint_outpaint"]}
    rng = torch.Generator(device=engine.device)
    rng.set_state(prepared.rng_state.cpu())
    contexts, pooled, prompt_proof = engine._joint_conditions(brief, outpaint=True)
    shapes = [build_outpaint_layout(topology, prepared.layout, face).size_px for face in CAPS]
    noise_by_face = {face: _pure_noise(engine, rng, shape) for face, shape in zip(CAPS, shapes, strict=True)}
    folder = args.output / brief["id"]
    outputs, details = {}, {}
    variants = [("native_T", "T", False, False), ("native_Btm", "Btm", False, False),
                ("flipped_T", "T", True, False)]
    if args.with_no_lut:
        variants.append(("native_T_no_lut", "T", False, True))
    if args.with_noise_swap:
        variants.append(("native_T_second_noise", "T", False, False))
    width_mm = topology["box_mm"]["width"]
    for name, face, flip, no_lut in variants:
        start = time.perf_counter()
        outputs[name], details[name] = trace_branch(
            engine, prepared, face, noise_by_face["Btm" if name.endswith("_second_noise") else face],
            contexts, pooled, options,
            flip=flip, directory=folder / name, width_mm=width_mm, no_lut=no_lut, numeric=args.float_statistics,
        )
        details[name]["seconds"] = time.perf_counter() - start
        _save_json(folder / "trace-partial.json", details)
        print(name, {k: v for k, v in details[name]["final"].items() if k != "float_probe"},
              details[name]["same_latent_vae_flip"], flush=True)
    previous_path = (args.root / "outputs/e02/joint-cap/runs/3411a058f0e6/stage1/j4b/seed-42"
                     / f"{brief['id']}-42.png")
    compare = _compare_archive(outputs["native_T"], outputs["native_Btm"], previous_path)
    record = {"brief": brief["id"], "seed": 42, "ring_sha256": ring_proof["cache_sha256"],
              "options": options, "prompts": prompt_proof, "effects": details,
              "historical_native_replay": compare}
    _save_json(folder / "trace.json", record)
    _contact(folder, details)
    print("Historical replay:", compare, flush=True)


if __name__ == "__main__":
    main()
