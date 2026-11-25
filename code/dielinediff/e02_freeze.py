"""Bind the E02 metrics to their pre-generation Git revision and resource hashes."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import yaml

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_history import dino_files


METRICS_CONFIG = Path("code/configs/E02_metrics.yaml")
EXPERIMENT_DOCS = Path("docs/experiments/E02")
MANIFEST = EXPERIMENT_DOCS / "metrics-freeze.json"
FROZEN_FILES = (
    METRICS_CONFIG,
    Path("code/dielinediff/metrics/e02_physical.py"),
    Path("code/dielinediff/metrics/e02_repetition.py"),
    Path("code/dielinediff/metrics/physical.py"),
    Path("code/dielinediff/metrics/__init__.py"),
    Path("code/dielinediff/metrics/e01b.py"),
)


def git(*arguments):
    return subprocess.check_output(["git", *arguments], text=True, encoding="utf-8").strip()


def committed_files(revision):
    hashes = {}
    for path in FROZEN_FILES:
        blob = subprocess.check_output(["git", "show", f"{revision}:{path.as_posix()}"])
        hashes[path.as_posix()] = sha256(blob).hexdigest()
        if file_hash(path) != hashes[path.as_posix()]:
            raise ValueError(f"Metric file {path} does not match the recorded Git revision")
    return hashes


def verify(root):
    record = read_json(MANIFEST)
    revision = record["metrics_commit"]
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"], check=True)
    if committed_files(revision) != record["files"]:
        raise ValueError("The E02 metric definitions or thresholds do not match the frozen manifest.")
    model, hashes = dino_files(root)
    if hashes != record["dino_files"]:
        raise ValueError("The local DINOv2 setup differs from the metrics freeze.")
    return yaml.safe_load(METRICS_CONFIG.read_text(encoding="utf-8")), model, record


def freeze(root, history_evidence):
    if MANIFEST.exists():
        raise FileExistsError("E02 metrics were already frozen.")
    if any((root / "outputs/e02/step1").glob("*/v3*/*.png")):
        raise RuntimeError("A v3 atlas already exists. The E02 freeze must precede any v3 generation.")
    revision = git("rev-parse", "HEAD")
    _, hashes = dino_files(root)
    evidence = read_json(history_evidence)
    if (evidence["physical_count"] != 755 or evidence["dino_count"] != 755
            or not evidence["sanity_passed"]):
        raise ValueError("Complete the historical trials and sanity checks before freezing.")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST, {
        "metrics_commit": revision,
        "parameter_sha256": file_hash(METRICS_CONFIG),
        "files": committed_files(revision),
        "dino_files": hashes,
        "history_evidence": evidence,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "freeze"))
    parser.add_argument("--root", type=Path, default=os.environ.get("DIELINEDIFF_ROOT"))
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.root is None or args.action == "freeze" and args.evidence is None:
        parser.error("root is required; freeze also requires historical evidence")
    if args.action == "freeze":
        freeze(args.root, args.evidence)
    else:
        _, _, manifest = verify(args.root)
        print(json.dumps({"metrics_commit": manifest["metrics_commit"],
                          "parameter_sha256": manifest["parameter_sha256"]}))
