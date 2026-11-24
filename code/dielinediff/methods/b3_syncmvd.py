"""Run the unmodified SyncMVD pipeline on the measured box UV atlas."""
from argparse import ArgumentParser
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


REVISION = "faa95621d220d7327847a994cd380608c04af5ab"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e"
DEPTH_MODEL = "lllyasviel/control_v11f1p_sd15_depth"
DEPTH_REVISION = "539f99181d33db39cf1af2e517cd8056785f0a87"


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(
        os.environ.get("E01_WORKSPACE", Path(__file__).resolve().parents[4])
    ))
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--rgb-render-size", type=int, default=1536)
    parser.add_argument("--max-batch-size", type=int, default=48)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path)
    mode.add_argument("--prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.manifest is None and (args.seed is None or args.output is None):
        parser.error("--prompt requires --seed and --output")
    if min(args.width, args.height, args.steps, args.rgb_render_size, args.max_batch_size) < 1:
        parser.error("sizes, steps and batch limit must be positive")
    return args


def environment(workspace):
    env = os.environ.copy()
    cache = workspace / "cache"
    temp = workspace / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    env.update({
        "TMPDIR": str(temp), "TMP": str(temp), "TEMP": str(temp),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "XDG_CACHE_HOME": str(cache),
        "TORCH_HOME": str(cache / "torch"),
        "CUPY_CACHE_DIR": str(cache / "cupy"), "CUDA_CACHE_PATH": str(cache / "cuda"),
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
    })
    return env


def upstream_options(parse_config, config_file, mesh, prompt, seed, args):
    """Use the upstream parser so all unspecified algorithm options keep official defaults."""
    old_argv = sys.argv
    sys.argv = [str(config_file), "--config", str(config_file), "--mesh", str(mesh),
                "--prompt=" + prompt, "--seed", str(seed), "--keep_mesh_uv",
                "--steps", str(args.steps), "--rgb_tex_size", str(min(args.width, args.height)),
                "--rgb_view_size", str(args.rgb_render_size)]
    try:
        return parse_config()
    finally:
        sys.argv = old_argv


def jobs(args):
    if args.manifest is None:
        yield args.prompt, args.seed, args.output
        return
    with args.manifest.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                job = json.loads(line)
                yield job["prompt"], int(job["seed"]), Path(job["output"])


def run(args):
    workspace = args.workspace.resolve()
    source = workspace / "third_party" / "SyncMVD"
    envs = workspace / "third_party" / "envs"
    env_dir = envs / "syncmvd"
    compat_env = envs / "syncmvd-compat"
    if not (env_dir / ".ready").is_file() and (compat_env / ".ready").is_file():
        env_dir = compat_env
    python = Path(os.environ.get("E01_SYNCMVD_ENV", env_dir)).resolve() / "bin" / "python"
    if not python.is_file():
        raise SystemExit("Missing SyncMVD environment. Run bash code/scripts/setup_3d_baselines.sh syncmvd.")
    if Path(sys.executable).resolve() != python.resolve():
        os.execve(python, [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], environment(workspace))
    os.environ.update(environment(workspace))
    rev = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise SystemExit(f"SyncMVD checkout {rev} differs from pinned {REVISION}")
    mesh = args.mesh.resolve(strict=True)
    queue = list(jobs(args))
    if not queue:
        raise SystemExit("The SyncMVD manifest contains no jobs.")

    # Delay environment-specific imports until running inside the independent conda prefix.
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from diffusers import ControlNetModel, DDPMScheduler, StableDiffusionControlNetPipeline
    from PIL import Image
    from pytorch3d.io import load_objs_as_meshes
    from src.configs import parse_config
    from src.pipeline import StableSyncMVDPipeline

    # PyTorch3D drops UVs without an input OBJ material texture. Reject implicit XAtlas fallback.
    mesh_data = load_objs_as_meshes([str(mesh)], device="cpu")
    if mesh_data.textures is None:
        raise ValueError("SyncMVD needs the provided OBJ with its MTL and placeholder atlas to preserve the dieline UVs.")
    del mesh_data
    controlnet = ControlNetModel.from_pretrained(
        DEPTH_MODEL, revision=DEPTH_REVISION, variant="fp16",
        torch_dtype=torch.float16, use_safetensors=True, local_files_only=True
    )
    # The default upstream repository is gated. The public SD 1.5 snapshot has matching
    # SHA-256s for the UNet, VAE and text encoder. SyncMVD overrides the inference
    # call and never invokes the parent pipeline's safety checker or feature extractor.
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, controlnet=controlnet,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        torch_dtype=torch.float16, use_safetensors=True, local_files_only=True
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    syncmvd = StableSyncMVDPipeline(**pipe.components)
    del pipe

    for prompt, seed, output in queue:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        logs = output.parent / (output.stem + ".syncmvd")
        logs.mkdir(exist_ok=True)
        config = logs / "config.yaml"
        config.write_text("", encoding="utf-8")
        opt = upstream_options(parse_config, config, mesh, prompt, seed, args)
        random.seed(seed)
        np.random.seed(seed % 2**32)
        generator = torch.manual_seed(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with (logs / "run.log").open("w", encoding="utf-8") as file, redirect_stdout(file), redirect_stderr(file):
            try:
                result = syncmvd(
                    prompt=opt.prompt,
                    height=opt.latent_view_size * 8, width=opt.latent_view_size * 8,
                    num_inference_steps=opt.steps, guidance_scale=opt.guidance_scale,
                    negative_prompt=opt.negative_prompt, generator=generator,
                    max_batch_size=args.max_batch_size, controlnet_guess_mode=opt.guess_mode,
                    controlnet_conditioning_scale=opt.conditioning_scale,
                    controlnet_conditioning_end_scale=opt.conditioning_scale_end,
                    control_guidance_start=opt.control_guidance_start,
                    control_guidance_end=opt.control_guidance_end,
                    guidance_rescale=opt.guidance_rescale, use_directional_prompt=True,
                    mesh_path=opt.mesh, mesh_transform={"scale": opt.mesh_scale}, mesh_autouv=False,
                    camera_azims=opt.camera_azims, top_cameras=not opt.no_top_cameras,
                    texture_size=opt.latent_tex_size, render_rgb_size=opt.rgb_view_size,
                    texture_rgb_size=opt.rgb_tex_size, multiview_diffusion_end=opt.mvd_end,
                    exp_start=opt.mvd_exp_start, exp_end=opt.mvd_exp_end,
                    ref_attention_end=opt.ref_attention_end,
                    shuffle_background_change=opt.shuffle_bg_change, shuffle_background_end=opt.shuffle_bg_end,
                    logging_config={"output_dir": str(logs), "log_interval": opt.log_interval,
                                    "view_fast_preview": opt.view_fast_preview, "tex_fast_preview": opt.tex_fast_preview},
                    cond_type=opt.cond_type,
                )
            except Exception:
                import traceback
                traceback.print_exc(file=file)
                raise
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated() / 1024**2
        if result is None:
            raise RuntimeError(f"SyncMVD ended before producing {output}")
        del result
        # PyTorch3D saves the UV atlas in standard OBJ PNG orientation, including its own background.
        # Resample to the common raster size without changing baseline pixel semantics.
        with Image.open(logs / "results" / "textured.png") as atlas:
            image = atlas.convert("RGB").resize((args.width, args.height), Image.Resampling.BILINEAR)
            temporary = output.with_suffix(".tmp.png")
            image.save(temporary)
            temporary.replace(output)
        data = {"elapsed_seconds": elapsed, "peak_memory_mib": peak}
        output.with_suffix(".run.json").write_text(json.dumps(data) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), **data}), flush=True)


if __name__ == "__main__":
    run(parse_args())
