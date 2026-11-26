"""Produce the preregistered overlays and denoising traces on the data disk."""

import subprocess
import sys


def run_diagnostics(experiment, device):
    root = str(experiment.root)
    output = experiment.output / "diagnostics"
    commands = (
        ("dielinediff.e02_panel_fix_overlay", "--output", str(output / "panel-overlay")),
        ("dielinediff.e02_panel_x0", "--output", str(output / "panel-x0"), "--device", device),
        ("dielinediff.e02_panel_fix_dark_diagnostic", "--output", str(output / "dark-band"),
         "--case", "E01-003", "--gpu", "--with-noise-swap", "--float-statistics", "--device", device),
    )
    for program, *args in commands:
        subprocess.run([sys.executable, "-u", "-m", program, "--root", root, *args], check=True)
    from .e02_panel_x0_scan import scan

    scan(output / "panel-x0", output / "panel-x0")
