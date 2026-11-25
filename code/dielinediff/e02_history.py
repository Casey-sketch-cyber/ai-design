"""Trial the predeclared E02 metric definitions on archived images only."""

import argparse
import csv
import importlib
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

from .e01 import read_json, validate_briefs, write_json
from .e01b import file_hash, stable_hash
from .e02_sources import sources
from .metrics.e02_repetition import DinoRepetition, DEFAULT_M4


DINO_WEIGHTS = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
DINO_MODEL_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
BRIEF_PATH = Path("data/briefs/E01_briefs.jsonl")


def read_briefs(path=BRIEF_PATH):
    briefs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_briefs(briefs)
    if [row["id"] for row in briefs] != [f"E01-{n:03d}" for n in range(1, 51)]:
        raise ValueError("E02 requires the registered brief order")
    return briefs


def dino_files(root):
    directory = root / "cache/models/dinov2-small"
    hashes = {name: file_hash(directory / name) for name in DINO_MODEL_FILES}
    if hashes["model.safetensors"] != DINO_WEIGHTS:
        raise ValueError("E02 requires the published, pinned DINOv2-S safetensors")
    return directory, hashes


def sanity_repetition(scorer, geometry, output):
    topology = read_json(geometry / "topology.json")
    page_size = topology["image"]["width"], topology["image"]["height"]
    motif = Image.new("RGB", (128, 128), "#c7daab")
    draw = ImageDraw.Draw(motif)
    draw.rectangle((21, 21, 107, 107), outline="#224a43", width=6)
    draw.ellipse((30, 28, 98, 99), fill="#385c47")
    draw.line((33, 89, 87, 38), fill="#f4eeb2", width=7)
    repeated = Image.new("RGB", page_size, "white")
    differentiated = Image.new("RGB", page_size, "white")
    distinct = ImageDraw.Draw(differentiated)
    colors = {
        "L": ("#765338", "#d7bf7b"), "F": ("#f2dac8", "#ba476b"),
        "R": ("#243f55", "#4ac8d2"), "B": ("#6e8a63", "#d4e6d0"),
    }
    for face, (bg, fg) in colors.items():
        x0, y0, x1, y1 = topology["faces"][face]["rect"]
        width, height = x1 - x0, y1 - y0
        if face == "F":
            middle = x0 + width // 2
            repeated.paste(motif.resize((middle - x0, height)), (x0, y0))
            repeated.paste(motif.resize((x1 - middle, height)), (middle, y0))
        else:
            repeated.paste(motif.resize((width, height)), (x0, y0))
        distinct.rectangle((x0, y0, x1 - 1, y1 - 1), fill=bg)
        if face == "F":
            distinct.ellipse((x0 + width // 12, y0 + height // 5,
                              x0 + width * 5 // 12, y0 + height * 3 // 5), fill=fg)
            distinct.rectangle((x0 + width * 8 // 12, y0 + height * 2 // 5,
                                x0 + width * 11 // 12, y0 + height * 4 // 5), fill="#332b51")
        elif face == "L":
            distinct.polygon(((x0 + width // 10, y0 + height * 3 // 4),
                              (x0 + width // 2, y0 + height // 8),
                              (x1 - width // 10, y0 + height * 3 // 4)), fill=fg)
        elif face == "R":
            for k in range(1, 5):
                yy = y0 + k * height // 6
                distinct.line((x0 + width // 8, yy, x1 - width // 8, yy),
                              fill=fg, width=max(height // 50, 1))
    ample = scorer.measure_batch([(repeated, topology), (differentiated, topology)])
    delta = ample[0]["m4_repeat_score"] - ample[1]["m4_repeat_score"]
    write_json(output / "m4-sanity.json", {"repeated_wallpaper": ample[0], "distinct_panels": ample[1],
                                            "delta": delta, "passed": delta > 0.05})
    if delta <= 0.05:
        raise RuntimeError("DINOv2 M4 failed the repeated-wallpaper sanity check.")


def trial(args):
    root = args.root
    jobs = list(sources(root, read_briefs()))
    jobs = jobs[args.part::args.parts]
    config = yaml.safe_load(args.config.read_text()) if args.config.is_file() else {}
    evidence = {"kind": args.kind, "device": args.device, "part": args.part, "parts": args.parts}
    if args.kind == "m4":
        model, hashes = dino_files(root)
        thresholds = config.get("m4", DEFAULT_M4)
        scorer = DinoRepetition(model, thresholds, device=args.device)
        evidence["model"] = hashes
        module = importlib.import_module("dielinediff.metrics.e02_repetition")
    else:
        module = importlib.import_module("dielinediff.metrics.e02_physical")
        thresholds = config.get("physical", module.DEFAULT_CONFIG)
        physical = {
            geometry: module.E02PhysicalEvaluator(geometry, device=args.device, config=thresholds)
            for geometry in dict.fromkeys(job.geometry for job in jobs)
        }
    evidence.update(thresholds=thresholds, implementation=file_hash(Path(module.__file__)))
    output = root / "outputs/e02/step0/history" / f"{args.kind}-{stable_hash(evidence)[:12]}"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "inputs.json", evidence)
    if args.kind == "m4":
        topologies = {geo: read_json(geo / "topology.json") for geo in {j.geometry for j in jobs}}
        sanity_repetition(scorer, jobs[0].geometry, output)
    fields = None
    temp = output / f"results.{os.getpid()}.tmp"
    with temp.open("w", newline="", encoding="utf-8") as stream:
        for i in range(0, len(jobs), args.batch):
            batch = jobs[i:i + args.batch]
            images = []
            for job in batch:
                with Image.open(job.path) as source:
                    images.append(source.convert("RGB"))
            if args.kind == "m4":
                values = scorer.measure_batch([(image, topologies[job.geometry])
                                                for job, image in zip(batch, images, strict=True)])
            else:
                values = [physical[job.geometry].measure(image)
                          for job, image in zip(batch, images, strict=True)]
            for job, result in zip(batch, values, strict=True):
                row = {"method": job.method, "brief_id": job.brief["id"], "seed": job.seed,
                       "atlas_sha256": file_hash(job.path), **result}
                if fields is None:
                    fields = list(row)
                    writer = csv.DictWriter(stream, fields, lineterminator="\n")
                    writer.writeheader()
                writer.writerow(row)
            if i == 0 or (i + args.batch) % 80 == 0 or i + args.batch >= len(jobs):
                print(f"{args.kind} {min(i + args.batch, len(jobs))}/{len(jobs)} {output}", flush=True)
            stream.flush()
    temp.replace(output / "results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("physical", "m4"))
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--config", type=Path, default=Path("code/configs/E02_metrics.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.root is None or not 0 <= args.part < args.parts or args.batch <= 0:
        parser.error("invalid root, partition or batch")
    trial(args)
