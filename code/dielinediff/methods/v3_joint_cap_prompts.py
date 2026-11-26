"""Task 20 cap continuation text shared by the isolated and joint generators."""

import re

from ..e01b import stable_hash
from .sdxl import NEGATIVE, STYLE


J1_POSITIVE = (
    "{style}, {side_prompt}, {main_color}, continue the same pattern and brushwork evenly over "
    "the whole surface, uniform detail density, same stroke direction as the adjacent artwork"
)
J1_NEGATIVE_SUFFIX = "plain background, empty area, blank space, blurry"
_SIDE_PREFIX = re.compile(r"^(?:continue a single panoramic composition of |restrained patterns based on )",
                          re.IGNORECASE)
_SIDE_SUFFIX = re.compile(r"(?: around the package|,\s*sparse product motifs)$", re.IGNORECASE)
_FAR_SCENE = re.compile(
    r"\b(?:mountains?|hills?(?:ides)?|ridges?|terraces?|plateau|valleys?|groves?|woodland|"
    r"forest silhouettes?|paddies|paddy|distant|horizons?|upland|orchard rows|panoramic)\b",
    re.IGNORECASE,
)
_FALLBACK_SIDE = "local motifs from the adjacent artwork"


def cap_local_side(text):
    """Keep adjacent artwork motifs while removing composition and landscape instructions."""
    unwrapped = _SIDE_SUFFIX.sub("", _SIDE_PREFIX.sub("", text.strip()))
    phrases = re.split(r",\s*|\s+and\s+", unwrapped)
    details = [phrase.strip() for phrase in phrases if phrase.strip() and not _FAR_SCENE.search(phrase)]
    return ", ".join(details) if details else _FALLBACK_SIDE


def joint_cap_prompt(brief, *, template=J1_POSITIVE):
    style = STYLE + (", wraparound artwork" if brief.get("continuous", False) else "")
    side = brief["side_prompt"]
    if brief.get("product_name_zh"):
        side = side.replace(brief["product_name_zh"], "")
    return template.format(style=style, side_prompt=cap_local_side(side),
                           main_color=brief["main_color"]).strip()


def joint_cap_negative(suffix=J1_NEGATIVE_SUFFIX):
    return f"{NEGATIVE}, {suffix}"


def joint_token_proof(tokenizers, positive, negative):
    """Record exact input IDs in both SDXL CLIP streams before model conditioning."""
    proof = {}
    for index, tokenizer in enumerate(tokenizers, 1):
        ids = tokenizer([positive, negative], add_special_tokens=True, truncation=False)["input_ids"]
        capacity = int(tokenizer.model_max_length)
        counts = [len(group) for group in ids]
        if max(counts) > capacity:
            raise ValueError(f"Cap text is truncated in CLIP stream {index}: {counts}/{capacity}")
        proof[f"tokenizer_{index}"] = {"counts": counts, "capacity": capacity,
                                       "input_ids_sha256": stable_hash(ids)}
    if len(proof) != 2:
        raise ValueError("Task 20 needs both SDXL CLIP streams")
    return proof
