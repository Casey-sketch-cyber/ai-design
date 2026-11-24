"""Run both official Paint3D stages without altering the input atlas or background."""
from argparse import ArgumentParser
from contextlib import ExitStack
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch


REVISION = "d5545d38db6aaf78efc563269f70177688a4218e"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISIONS = {
    BASE_MODEL: "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    "lllyasviel/control_v11f1p_sd15_depth": "539f99181d33db39cf1af2e517cd8056785f0a87",
    "lllyasviel/control_v11p_sd15_inpaint": "c96e03a807e64135568ba8aecb66b3a306ec73bd",
    "GeorgeQi/Paint3d_UVPos_Control": "1d904225f9b4ff1cd2ff414082b9b211e4df6be5",
    "GeorgeQi/realisticVisionV13_v13": "e33f3418cbc91b3de3dd8705b3b6e07e956343aa",
    "lllyasviel/control_v11f1e_sd15_tile": "3f877705c37010b7221c3d10743307d6b5b6efac",
}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(
        os.environ.get("E01_WORKSPACE", Path(__file__).resolve().parents[4])
    ))
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.width, args.height) < 64 or args.steps < 1:
        parser.error("atlas dimensions must be at least 64 and sampling steps must be positive")
    return args


def launch_stage(stage, config, render_config, mesh, outdir, texture=None):
    argv = sys.argv
    sys.argv = [stage.__name__, "--sd_config", str(config), "--render_config", str(render_config),
                "--mesh_path", str(mesh), "--outdir", str(outdir)]
    if texture is not None:
        sys.argv.extend(["--texture_path", str(texture)])
    try:
        stage.main()
    finally:
        sys.argv = argv


def model_snapshot(cache, identifier):
    path = (cache / "huggingface" / "hub" / ("models--" + identifier.replace("/", "--"))
            / "snapshots" / MODEL_REVISIONS[identifier])
    if not path.is_dir():
        raise FileNotFoundError(f"Missing the pinned Paint3D model {identifier} at {path}")
    return path


def corner_uvs(mesh):
    """Return texture coordinates keyed by the geometric corners of each OBJ face."""
    vertex_count = 0
    coordinates = []
    faces = []
    for line in mesh.read_text(encoding="utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if fields[0] == "v":
            vertex_count += 1
        elif fields[0] == "vt":
            coordinates.append(tuple(map(float, fields[1:3])))
        elif fields[0] == "f":
            corners = []
            for field in fields[1:]:
                vertex, uv, *_ = field.split("/")
                vertex = int(vertex)
                vertex = vertex_count + vertex + 1 if vertex < 0 else vertex
                uv = int(uv)
                corners.append((vertex, *coordinates[uv - 1 if uv > 0 else uv]))
            faces.append(sorted(corners))
    if not faces:
        raise ValueError(f"No textured OBJ faces in {mesh}")
    return sorted(faces, key=lambda face: tuple(corner[0] for corner in face))


def atlas_registration(mesh, native_mesh):
    original = corner_uvs(mesh)
    native = corner_uvs(native_mesh)
    original_corners = [tuple(corner[0] for corner in face) for face in original]
    native_corners = [tuple(corner[0] for corner in face) for face in native]
    if original_corners != native_corners:
        raise ValueError("Paint3D changed the geometry of the input OBJ faces.")
    aligned = [
        (source, target)
        for source_face, target_face in zip(original, native)
        for source, target in zip(source_face, target_face)
    ]
    # Kaolin versions may import OBJ V as 1-V. Determine the actual coordinate
    # convention from Paint3D's stage-one mesh instead of assuming a backend version.
    for flip_v in (False, True):
        error = max(
            max(abs(source[1] - target[1]), abs(source[2] - (1 - target[2] if flip_v else target[2])))
            for source, target in aligned
        )
        if error <= 1e-6:
            return flip_v, error
    raise ValueError("Paint3D's atlas cannot be registered to the supplied dieline UVs.")


def save_registered_atlas(raw_atlas, mesh, native_mesh, width, height, output):
    from PIL import Image

    flip_v, error = atlas_registration(mesh, native_mesh)
    with Image.open(raw_atlas) as atlas:
        image = atlas.convert("RGB")
        if flip_v:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        temporary = output.with_suffix(".tmp.png")
        resized.save(temporary)
        temporary.replace(output)
    return {"transformation": "flip_vertical" if flip_v else "identity", "max_uv_difference": error}


def run(args):
    workspace = args.workspace.resolve()
    source = workspace / "third_party" / "Paint3D"
    envs = workspace / "third_party" / "envs"
    env_dir = envs / "paint3d"
    compat_env = envs / "paint3d-compat"
    if not (env_dir / ".ready").is_file() and (compat_env / ".ready").is_file():
        env_dir = compat_env
    python = Path(os.environ.get("E01_PAINT3D_ENV", env_dir)).resolve() / "bin" / "python"
    if not python.is_file():
        raise SystemExit("Missing Paint3D environment. Run bash code/scripts/setup_3d_baselines.sh paint3d.")
    cache = workspace / "cache"
    temp = workspace / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    env = {
        "TMPDIR": str(temp), "TMP": str(temp), "TEMP": str(temp),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "XDG_CACHE_HOME": str(cache), "TORCH_HOME": str(cache / "torch"),
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"), "CUDA_CACHE_PATH": str(cache / "cuda"),
    }
    if Path(sys.executable).resolve() != python.resolve():
        os.execve(python, [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], {**os.environ, **env})
    os.environ.update(env)
    rev = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise SystemExit(f"Paint3D checkout {rev} differs from pinned {REVISION}")
    mesh = args.mesh.resolve(strict=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / (output.stem + ".paint3d")
    logs.mkdir(exist_ok=True)
    os.chdir(source)
    sys.path.insert(0, str(source))

    import torch
    from diffusers import (StableDiffusionControlNetImg2ImgPipeline,
                           StableDiffusionControlNetInpaintPipeline,
                           StableDiffusionControlNetPipeline)
    from omegaconf import OmegaConf
    import pipeline_paint3d_stage1 as stage1
    import pipeline_paint3d_stage2 as stage2

    templates = source / "controlnet" / "config"
    coarse = OmegaConf.load(templates / "depth_based_inpaint_template.yaml")
    refined = OmegaConf.load(templates / "UV_based_inpaint_template.yaml")
    resolution = min(args.width, args.height) // 64 * 64
    for config in (coarse, refined):
        for section in config.values():
            section.prompt = args.prompt
            section.seed = args.seed
            section.num_inference_steps = args.steps
            # The upstream SD 1.5 Hub ID now redirects to the same public model.
            model = BASE_MODEL if section.sd_model_key == "runwayml/stable-diffusion-v1-5" else section.sd_model_key
            section.sd_model_key = str(model_snapshot(cache, model))
            for unit in section.controlnet_units:
                unit.controlnet_key = str(model_snapshot(cache, unit.controlnet_key))
    for section in refined.values():
        section.width = resolution
        section.height = resolution
    coarse_path = logs / "stage1.yaml"
    refined_path = logs / "stage2.yaml"
    OmegaConf.save(coarse, coarse_path)
    OmegaConf.save(refined, refined_path)
    # Use the official renderer configuration, changing only the common seed, atlas
    # dimensions and the path for Paint3D's own projection cache.
    render_config = logs / "e01_paint3d_render.py"
    render_config.write_text(
        "from paint3d.config.train_config_paint3d import TrainConfig as OfficialConfig\n"
        "def TrainConfig():\n"
        "    config = OfficialConfig()\n"
        f"    config.optim.seed = {args.seed}\n"
        f"    config.guide.texture_resolution = [{resolution}, {resolution}]\n"
        f"    config.log.cache_path = {str(logs / 'cache')!r}\n"
        "    return config\n",
        encoding="utf-8",
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    coarse_out = logs / "stage1"
    refined_out = logs / "stage2"
    # Each upstream stage sets pipe.safety_checker to None before any denoising.
    # Avoid loading the unused safety/feature weights into the 24 GB device.
    with ExitStack() as stack:
        for pipeline in (StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline,
                         StableDiffusionControlNetImg2ImgPipeline):
            loader = partial(pipeline.from_pretrained, safety_checker=None, feature_extractor=None,
                             requires_safety_checker=False, local_files_only=True)
            stack.enter_context(patch.object(pipeline, "from_pretrained", loader))
        launch_stage(stage1, coarse_path, render_config, mesh, coarse_out)
        torch.cuda.empty_cache()
        launch_stage(stage2, refined_path, render_config, mesh, refined_out,
                     coarse_out / "res-0" / "albedo.png")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 1024**2
    final_atlas = refined_out / "tile_res_0" / "albedo.png"
    native_mesh = coarse_out / "res-0" / "mesh.obj"
    registration = save_registered_atlas(final_atlas, mesh, native_mesh, args.width, args.height, output)
    (logs / "atlas_registration.json").write_text(json.dumps(registration) + "\n", encoding="utf-8")
    data = {"elapsed_seconds": elapsed, "peak_memory_mib": peak}
    output.with_suffix(".run.json").write_text(json.dumps(data) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **data}), flush=True)


if __name__ == "__main__":
    run(parse_args())
