"""Apply J5 inside Stage 1 so the native 9–12 mm band can respond."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from .v3_geometry import CAPS, build_cap_layout
from .v3_joint_cap_geometry import build_joint_ring_layout, build_outpaint_layout
from .v3_joint_cap_guidance import texture_guidance
from .v3_joint_cap_stage2 import _preserve_native
from .v3_joint_cap_ring import downsample_energy_retention


class GuidedJointRingMixin:
    """Attach clean-latent guidance to the registered Stage 1 prediction path."""

    _joint_uplift = None

    def _guided_stage(self, generator, frames, settings, reference, *, separate):
        if self._joint_uplift is not None:
            raise RuntimeError("A guided Stage 1 is already active")
        trace = {"frames": frames, "settings": settings, "reference": reference,
                 "separate": separate, "current": 0, "last_t": None, "callbacks": {}}
        self._joint_uplift = trace
        try:
            result = generator()
        finally:
            self._joint_uplift = None
        evidence = {face: callback.evidence() for face, callback in trace["callbacks"].items()}
        if set(evidence) != set(CAPS) or any(not row["executed_steps"] for row in evidence.values()):
            raise RuntimeError("Stage 1 did not guide both registered cap interiors")
        return replace(result, proof={**result.proof, "stage1_guidance": evidence})

    def _predict(self, latent, timestep, text, condition, control):
        epsilon = super()._predict(latent, timestep, text, condition, control)
        trace = self._joint_uplift
        if trace is None:
            return epsilon
        shape = tuple(latent.shape[-2:])
        frames = trace["frames"]
        if shape not in {view.latent_shape for view in frames}:
            return epsilon
        timestep_value = int(timestep)
        times = self.pipe.scheduler.timesteps
        index = int(torch.nonzero(times == timestep, as_tuple=False)[0, 0])
        if trace["separate"]:
            if trace["last_t"] is not None and timestep_value > trace["last_t"]:
                trace["current"] += 1
            trace["last_t"] = timestep_value
            frames = frames[trace["current"]:trace["current"] + 1]
            if not frames or frames[0].latent_shape != shape:
                raise RuntimeError("The guided outpaint order is inconsistent")
        settings = trace["settings"]
        for view in frames:
            if view.face not in trace["callbacks"]:
                trace["callbacks"][view.face] = texture_guidance(
                    self, view, trace["reference"].frozen_latent, settings["gamma"], settings,
                    direction="tangent" if settings["directional"] else "isotropic",
                )
            epsilon = trace["callbacks"][view.face](index, timestep, latent, epsilon)
        return epsilon

    def _guidance_view(self, face, size, rect, registered_ring, *, known_rows=None):
        if any(value % 8 for value in (*size, *rect)):
            raise ValueError("J5 cap registration needs whole latent cells")
        shape = size[1] // 8, size[0] // 8
        mask = np.zeros(shape, dtype=bool)
        if known_rows is not None:
            start, end = known_rows
            mask[start:end] = True
        band = int(self.config.get("cap_band", 16))
        original = build_cap_layout(self.topology, registered_ring, face, band_latent=band)
        return SimpleNamespace(face=face, latent_shape=shape, center_rect=tuple(value // 8 for value in rect),
                               known_mask=mask, original=original, ring=registered_ring)

    @torch.inference_mode()
    def guided_outpaint(self, brief, seed, prepared, *, options, settings):
        frames = []
        for face in CAPS:
            canvas = build_outpaint_layout(self.topology, prepared.layout, face, int(options["strip_height_px"]),
                                           int(options["cap_buffer_px"]))
            frames.append(self._guidance_view(face, canvas.size_px, canvas.cap_rect_px, prepared.layout,
                                              known_rows=canvas.known_rows_latent))
        return self._guided_stage(lambda: self.generate_frozen_outpaint(brief, seed, prepared, options=options),
                                  frames, settings, prepared, separate=True)

    @torch.inference_mode()
    def guided_joint(self, brief, seed, parent, *, options, settings):
        full = build_joint_ring_layout(self.topology, int(options["target_width_px"]), int(options["cap_buffer_px"]))
        size = full.ring.width_px, full.ring.height_px
        reference = SimpleNamespace(layout=parent.ring, frozen_latent=parent.ring_latent)
        if reference.frozen_latent is None:
            raise ValueError("Joint J5 needs the exact selected parent ring latent")
        frames = [self._guidance_view(face, size, full.cap_rects_px[face], reference.layout) for face in CAPS]
        result = self._guided_stage(lambda: self.generate_joint_stage1(brief, seed, options=options),
                                    frames, settings, reference, separate=False)
        if result.ring.face_rects != parent.ring.face_rects or result.ring_image.size != parent.ring_image.size:
            raise ValueError("The guided joint stage must share the locked parent geometry")
        caps = {}
        lock_settings = {"native_lock_mm": 3.0, "native_unlock_mm": 3.0}
        for face in CAPS:
            caps[face], _ = _preserve_native(parent.caps[face], result.caps[face], face=face,
                                              box_mm=self.topology["box_mm"], settings=lock_settings)
        retention = {"ring_body": parent.proof["downsample_energy_2x"]["ring_body"],
                     **{face: downsample_energy_retention(caps[face]) for face in CAPS}}
        return replace(parent, caps=caps, proof={**result.proof, "cap_rect_source": "guided_crops",
                                                 "ring_locked_to_parent": True,
                                                 "native_parent_protected_mm": 3.0,
                                                 "downsample_energy_2x": retention})
