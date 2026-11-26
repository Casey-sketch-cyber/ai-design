"""Bind Task 20 visual observations to the exact generated original pixels."""

from .e01 import read_json
from .e02_cap_energy_visual import verify_content
from .e02_captexture import DOCS
from .e02_joint_cap import CONTROL


def content_path(experiment, explicit=None):
    for path in (explicit, experiment.output / "cap-reviews.json", DOCS / "joint-cap-content-review.json"):
        if path is not None and path.is_file():
            return path
    return None


def content_reviews(experiment, recipes, explicit=None):
    archived = read_json(DOCS / "cap-energy-cap-content-review.json")["e0"]
    path = content_path(experiment, explicit)
    provided = read_json(path) if path else {}
    available = {CONTROL: archived, **provided}
    missing = [method for method in recipes if method not in available]
    if missing:
        print(f"Original-pixel lid review required for {', '.join(missing)} in {experiment.output / 'cap-content'}",
              flush=True)
        return None, None
    scope = {method: available[method] for method in recipes}
    return verify_content(experiment, scope, methods=recipes)
