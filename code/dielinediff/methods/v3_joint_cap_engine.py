"""Run Task 20 refinements on the original Task 17 SDEdit cap canvas."""

import torch

from .v3_attention import RegionRouting
from .v3_capenergy_engine import EnergyCapEngine
from .v3_capfix import cap_region_ids
from .v3_joint_cap_prompts import joint_cap_negative, joint_cap_prompt, joint_token_proof


class JointCapEngine(EnergyCapEngine):
    """J1 changes text, J2 swaps the denoiser, and J3 adds within-loop texture guidance."""

    _joint_candidate = None
    _joint_prompt_record = None

    def generate_caps(self, brief, prepared, candidate):
        backend = candidate.get("cap_backend", "inpaint")
        gamma = candidate.get("guidance_gamma", 0)
        settings = self.config["joint_cap"]["guidance"]
        if (backend not in ("inpaint", "base_blend") or candidate["inpaint"] != (backend == "inpaint")
                or gamma not in (0, *settings["gamma"])):
            raise ValueError("Invalid registered joint-cap backend or guidance strength")
        if candidate.get("match_energy") or candidate.get("differential"):
            raise ValueError("Task 20 must start from the unmodified J0 cap prior")
        self._joint_candidate = candidate
        self._joint_prompt_record = None
        try:
            image, details = super().generate_caps(
                brief, prepared, {**candidate, "match_energy": False, "differential": False}
            )
            if self._joint_prompt_record is not None:
                details["joint_prompt"] = self._joint_prompt_record
            return image, details
        finally:
            self._joint_candidate = None

    def _cap_conditions(self, brief, layout, prompt_mode):
        if self._joint_candidate is None or not self._joint_candidate.get("clean_prompt", False):
            return super()._cap_conditions(brief, layout, prompt_mode)
        if prompt_mode != "plain":
            raise ValueError("J1 must use the same single-region cap route as J0")
        tuning = self.config["joint_cap"]["cap_prompt"]
        positive_text = joint_cap_prompt(brief, template=tuning["positive"])
        negative_text = joint_cap_negative(tuning["negative"])
        proof = joint_token_proof((self.pipe.tokenizer, self.pipe.tokenizer_2), positive_text, negative_text)
        positive, negative, pooled, negative_pooled = self.pipe.encode_prompt(
            prompt=[positive_text], negative_prompt=[negative_text], device=self.device,
            do_classifier_free_guidance=True,
        )
        self._joint_prompt_record = {"positive": positive_text, "negative": negative_text, "tokens": proof}
        outer = float(self.config["capfix_prompts"].get("outer_fraction", 0.25))
        routing = RegionRouting(cap_region_ids(layout, prompt_mode, outer))
        return {0: torch.stack((negative[0], positive[0]))}, torch.stack((negative_pooled[0], pooled[0])), routing, [positive_text]

    def _candidate_cap(self, layout, prepared, generator, network, text, pooled, routing, candidate):
        if candidate.get("cap_backend", "inpaint") == "inpaint" and not candidate.get("guidance_gamma", 0):
            return super()._candidate_cap(layout, prepared, generator, network, text, pooled, routing, candidate)

        from .v3_joint_cap_refine import generate_refined_cap

        return generate_refined_cap(self, layout, prepared, generator, network, text, pooled, routing, candidate)
