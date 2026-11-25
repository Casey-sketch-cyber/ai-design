"""Resumable E01 generation and evaluation on one GPU."""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

from .methods.sdxl import prompt_for

METHODS = ("b2", "b3", "b4", "ours", "no_sync", "no_projection")
EXTERNAL = {"b3": "b3_syncmvd", "b4": "b4_paint3d"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_briefs(rows):
    required = ("id", "category", "region_style", "main_color", "product_name_zh",
                "front_prompt", "side_prompt", "top_prompt")
    if len(rows) != 50 or len({row["id"] for row in rows}) != 50:
        raise ValueError("E01 expects 50 uniquely identified briefs")
    if any(not row.get(key) for row in rows for key in required):
        raise ValueError("Missing brief fields")
    if sum(row["continuous"] for row in rows) < 20:
        raise ValueError("E01 requires at least 20 cross-panel briefs")


def jobs_for(root, methods, rows, seeds):
    jobs = []
    for method in methods:
        for brief in rows:
            for seed in seeds:
                output = root / method / f'{brief["id"]}-{seed}.png'
                jobs.append((method, brief, seed, output))
    return jobs


def generate_native(method, jobs, config, geometry, args):
    models = args.root / "cache" / "models"
    if method == "b2":
        from .methods.b2_controlnet import ControlNetBaseline

        engine = ControlNetBaseline(models, geometry, config, args.device)
    else:
        from .methods.ours_tf import DieLineDiff

        engine = DieLineDiff(models, geometry, config, args.device, args.lut)
    cuda = torch.device(args.device).type == "cuda"
    for _, brief, seed, output in jobs:
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if method == "b2":
            image = engine.generate(brief, seed)
        else:
            image = engine.generate(brief, seed, synchronized=method != "no_sync",
                                    project=method != "no_projection")
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        memory = float(torch.cuda.max_memory_allocated()) / 2**20 if cuda else None
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".tmp.png")
        image.save(temp)
        temp.replace(output)
        write_json(output.with_suffix(".run.json"), {
            "elapsed_seconds": elapsed, "peak_memory_mib": memory,
        })
        print(f"{method} {brief['id']} {seed} {elapsed:.2f}s {memory} MiB", flush=True)


def generate_external(method, jobs, config, geometry, args):
    method_dir = jobs[0][-1].parent
    method_dir.mkdir(parents=True, exist_ok=True)
    image = geometry["topology"]["image"]
    common = [
        sys.executable, "-m", f"dielinediff.methods.{EXTERNAL[method]}",
        "--mesh", str(geometry["directory"] / "box.obj"),
        "--width", str(image["width"]), "--height", str(image["height"]),
        "--steps", str(config["steps"]),
    ]
    env = os.environ | {"E01_WORKSPACE": str(args.root)}
    if method == "b3":
        manifest = method_dir / "jobs.jsonl"
        manifest.write_text("".join(json.dumps({
            "prompt": prompt_for(brief), "seed": seed, "output": str(output)
        }) + "\n" for _, brief, seed, output in jobs), encoding="utf-8")
        subprocess.run(common + ["--manifest", str(manifest)], env=env, check=True)
    else:
        for _, brief, seed, output in jobs:
            subprocess.run(common + ["--prompt", prompt_for(brief), "--seed", str(seed),
                                     "--output", str(output)], env=env, check=True)


def summarize_outputs(jobs, root):
    rows = []
    for method, brief, seed, output in jobs:
        path = output.with_suffix(".metrics.json")
        if not path.is_file():
            continue
        rows.append({
            "method": method, "brief_id": brief["id"], "seed": seed,
            "split": "tune" if int(brief["id"].split("-")[-1]) <= 10 else "holdout",
            "continuous": brief["continuous"], **read_json(path),
        })
    path = root / "results.csv"
    meta = ("method", "brief_id", "seed", "split", "continuous")
    known_metrics = ("seam_sdr", "seam_lpips", "upright_rate", "glue_ink_pct",
                     "flap_edge_pct", "flap_ocr_count", "bleed_fill_pct",
                     "lighting_residual", "oog_pct", "roundtrip_delta_e00",
                     "clip_score", "hps_v2", "elapsed_seconds", "peak_memory_mib")
    extra = sorted({key for row in rows for key in row} - set(meta + known_metrics))
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*meta, *known_metrics, *extra), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    print(f"{len(rows)}/{len(jobs)} measured rows in {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "measure", "collect"))
    parser.add_argument("--phase", choices=("pilot", "tune", "full"), default="pilot")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get(
        "DIELINEDIFF_ROOT", "/root/autodl-tmp/ai-design"
    )))
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--config", type=Path, default=Path("code/configs/E01.yaml"))
    parser.add_argument("--briefs", type=Path, default=Path("data/briefs/E01_briefs.jsonl"))
    parser.add_argument("--lut", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learned", nargs="*", default=["lpips", "clip", "hps", "ocr"])
    parser.add_argument("--follow-until", type=Path, help="Wait for generation to write this marker")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if args.phase == "full" and not config["frozen"]:
        parser.error("Freeze the pilot/tuning settings before starting the holdout")
    args.geometry = args.geometry or args.root / "outputs" / "geometry"
    args.lut = args.lut or args.root / "cache" / "icc" / "iso-coated-v2-lut.npy"
    rows = [json.loads(line) for line in args.briefs.read_text(encoding="utf-8").splitlines()]
    validate_briefs(rows)
    seeds = config["seeds"]
    if args.phase == "pilot":
        rows, seeds = rows[:5], seeds[:1]
    elif args.phase == "tune":
        rows, seeds = rows[:10], seeds[:1]
    methods = args.methods or (METHODS if args.phase == "full" else METHODS[:4])
    geometry_files = ("topology.json", "region_map.png", "bleed_owner.png", "box.obj")
    digest_input = [config_bytes, args.briefs.read_bytes(), args.lut.read_bytes()]
    digest_input.extend((args.geometry / name).read_bytes() for name in geometry_files)
    digest = sha256(b"".join(digest_input)).hexdigest()[:12]
    output_root = args.root / "outputs" / "e01" / f"{args.phase}-{digest}"
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "inputs.json", {"config": config, "briefs_sha256":
                                            sha256(args.briefs.read_bytes()).hexdigest()})
    jobs = jobs_for(output_root, methods, rows, seeds)

    if args.command == "collect":
        summarize_outputs(jobs, output_root)
        return
    if args.command == "generate":
        geometry = {"directory": args.geometry, "topology": read_json(args.geometry / "topology.json")}
        for method in methods:
            pending = [job for job in jobs if job[0] == method and not (
                job[-1].exists() and job[-1].with_suffix(".run.json").exists()
            )]
            if not pending:
                continue
            if method in EXTERNAL:
                generate_external(method, pending, config, geometry, args)
            else:
                generate_native(method, pending, config, geometry, args)
        return

    from .metrics import Evaluator

    with Evaluator(args.geometry, args.lut, device=args.device, learned=tuple(args.learned)) as evaluator:
        pending = list(jobs)
        while pending:
            generation_finished = args.follow_until is None or args.follow_until.is_file()
            waiting = []
            for method, brief, seed, output in pending:
                if not output.is_file() or not output.with_suffix(".run.json").is_file():
                    waiting.append((method, brief, seed, output))
                    continue
                target = output.with_suffix(".metrics.json")
                if target.is_file():
                    existing = read_json(target)
                    if existing.get("learned_backends") == sorted(args.learned):
                        continue
                run = read_json(output.with_suffix(".run.json"))
                metrics = evaluator.measure(
                    output, prompt=prompt_for(brief), elapsed_seconds=run["elapsed_seconds"],
                    peak_memory_mib=run["peak_memory_mib"],
                )
                write_json(target, {**metrics, "learned_backends": sorted(args.learned)})
                print(f"Measured {method} {brief['id']} {seed}", flush=True)
            if len(waiting) != len(pending):
                summarize_outputs(jobs, output_root)
            if generation_finished:
                break
            pending = waiting
            if pending:
                time.sleep(5)
    summarize_outputs(jobs, output_root)


if __name__ == "__main__":
    main()
