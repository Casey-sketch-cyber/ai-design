"""Continue the same cap illustration without packaging-layout text."""

import re

import torch

from .v3_attention import RegionRouting
from .v3_capfix import cap_region_ids
from .v3_joint_cap_engine import JointCapEngine
from .v3_joint_cap_prompts import (
    J1_POSITIVE,
    cap_local_side,
    joint_cap_negative,
    joint_token_proof,
)


_LAYOUT = re.compile(
    r"\b(?:packag(?:e|ing)|graphic design|layout|information panel|panels?|labels?|"
    r"frames?|borders?|cards?|collage|patchwork|inset picture|rectangles?)\b",
    re.IGNORECASE,
)
_FALLBACK_MOTIF = "local motifs from the adjacent artwork"


def panel_cap_prompt(brief, settings):
    """Keep the registered J1 continuation and swap its layout-bearing style."""
    style = str(settings["style"]).strip()
    if not style or _LAYOUT.search(style):
        raise ValueError("The panel-fix cap style must describe illustration only")
    side = brief["side_prompt"]
    if brief.get("product_name_zh"):
        side = side.replace(brief["product_name_zh"], "")
    local = cap_local_side(side)
    motifs = [part for part in local.split(", ") if not _LAYOUT.search(part)]
    return J1_POSITIVE.format(style=style, side_prompt=", ".join(motifs) or _FALLBACK_MOTIF,
                              main_color=brief["main_color"]).strip()


def panel_cap_negative(settings):
    """Keep the complete J1 negative context and add the registered layout terms."""
    additions = settings["negative_additions"].strip()
    if not additions:
        raise ValueError("The panel-fix cap needs explicit negative layout terms")
    return f"{joint_cap_negative()}, {additions}"


def panel_cap_conditions(engine, brief):
    """Encode the same cap text for isolated caps and the frozen-ring outpaint."""
    settings = engine.config["panel_fix"]["cap_prompt"]
    positive_text = panel_cap_prompt(brief, settings)
    negative_text = panel_cap_negative(settings)
    proof = joint_token_proof((engine.pipe.tokenizer, engine.pipe.tokenizer_2), positive_text, negative_text)
    positive, negative, pooled, negative_pooled = engine.pipe.encode_prompt(
        prompt=[positive_text], negative_prompt=[negative_text], device=engine.device,
        do_classifier_free_guidance=True,
    )
    return (positive_text, negative_text, {0: torch.stack((negative[0], positive[0]))},
            torch.stack((negative_pooled[0], pooled[0])), proof)


class PanelFixEngine(JointCapEngine):
    """Use J3's initialization, SDXL inpainting and guidance with the cap-only text."""

    def __init__(self, models_dir, geometry_dir, config, device="cuda", lut_path=None):
        options = config["panel_fix"]
        original = config["joint_cap"]
        strengths = tuple(dict.fromkeys((*original["guidance"]["gamma"],
                                         options["guidance_gamma"]["k1"], options["guidance_gamma"]["k2"])))
        guidance = {**original["guidance"], "gamma": strengths}
        configured = {**config, "joint_cap": {**original, "guidance": guidance}}
        super().__init__(models_dir, geometry_dir, configured, device=device, lut_path=lut_path)

    def _cap_conditions(self, brief, layout, prompt_mode):
        if self._joint_candidate is None or not self._joint_candidate.get("panel_prompt"):
            return super()._cap_conditions(brief, layout, prompt_mode)
        if prompt_mode != "plain":
            raise ValueError("The isolated panel-fix caps require the original single-region route")
        positive, negative, contexts, pooled, proof = panel_cap_conditions(self, brief)
        self._joint_prompt_record = {"positive": positive, "negative": negative, "tokens": proof,
                                     "protocol": "panel_fix"}
        outer = float(self.config["capfix_prompts"].get("outer_fraction", 0.25))
        routing = RegionRouting(cap_region_ids(layout, prompt_mode, outer))
        return contexts, pooled, routing, [positive]

