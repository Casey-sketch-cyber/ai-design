"""Coarse-to-fine regional SDXL conditioning for the ring and flat atlas."""

from contextlib import contextmanager

import numpy as np
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from torch.nn import functional as F

from ..dieline.geometry import REGION_LABELS
from ..e01b import stable_hash
from .sdxl import NEGATIVE, STYLE
from .v3 import DieLineDiffV3, REGIONAL_FACES, regional_routing
from .v3_attention import RegionalAttention
from .v3_circular import circular_ring_codec
from .v3_geometry import LATENT_SCALE, SIDES, atlas_region_ids


M2_REGIONAL_SETTINGS = {
    "global_only_steps": 9,
    "final_regional_weight": 0.6,
    "feather_mm": 7.0,
    "global_prompt": "{style}, seamless wraparound packaging illustration, {theme}, color accent {main_color}",
    "regional_template": "continuous edge-to-edge packaging illustration with balanced visual density",
    "back_suffix": "slightly lower visual density, softer details",
}


def m2_prompt_plan(brief, settings):
    """Keep the front subject and locale globally, and keep layout instructions local."""
    front = brief["front_prompt"]
    theme = front.removeprefix("prominent illustration of ").removesuffix(", clear space for product name")
    global_prompt = settings["global_prompt"].format(style=STYLE, theme=theme, main_color=brief["main_color"])
    continuous = ", wraparound artwork" if brief["continuous"] else ""
    template = f"{STYLE}{continuous}, {settings['regional_template']}"
    color = brief["main_color"]
    descriptions = {
        "F": front,
        "B": f"{brief['side_prompt']}, {settings['back_suffix']}",
        "L": brief["side_prompt"],
        "T": brief["top_prompt"],
        "Btm": f"{brief['top_prompt']}, simple ornamental treatment",
    }
    return {0: global_prompt, **{label: f"{template}, {descriptions[face]}, color accent {color}"
                                for label, face in REGIONAL_FACES.items()}}


def m2_schedule(steps, settings):
    """Keep nine global steps, then sample both endpoints of the twenty-one-step ramp."""
    global_steps = settings["global_only_steps"]
    limit = float(settings["final_regional_weight"])
    if not isinstance(global_steps, int) or not 0 < global_steps < steps - 1 or not 0 < limit <= 1:
        raise ValueError("The M2 schedule needs a global phase and a nonzero ramp with two endpoints")
    return (0.0,) * global_steps + tuple(np.linspace(0.0, limit, steps - global_steps).tolist())


def _hard_weights(labels):
    labels = regional_routing(labels).labels
    return np.stack([(labels == label).astype(np.float32) for label in range(7)])


def _assign_blend(weights, left, right, rows, columns, signed_mm, feather_mm):
    right_weight = np.clip((1 + signed_mm / feather_mm) / 2, 0, 1)
    weights[:, rows, columns] = 0
    weights[left][rows, columns] = 1 - right_weight
    weights[right][rows, columns] = right_weight


def _face_label(name):
    return REGION_LABELS["L" if name == "R" else name]


def m2_ring_weights(topology, layout, feather_mm):
    """Blend both sides of each physical upright edge, including the circular L–B edge."""
    weights = _hard_weights(layout.region_ids)
    dims = topology["box_mm"]
    widths = {"L": dims["width"], "F": dims["length"], "R": dims["width"], "B": dims["length"]}
    if not 0 < feather_mm < min(widths.values()) / 2:
        raise ValueError("Ring feather bands must stay disjoint within each face")
    for name, neighbor in zip(SIDES, SIDES[1:] + SIDES[:1], strict=True):
        left0, _, left1, _ = layout.face_rects[name]
        right0, _, right1, _ = layout.face_rects[neighbor]
        left_centers = np.arange(left0, left1, dtype=np.float64) + 0.5
        right_centers = np.arange(right0, right1, dtype=np.float64) + 0.5
        a = (left_centers - left1) * widths[name] / (left1 - left0)
        b = (right_centers - right0) * widths[neighbor] / (right1 - right0)
        columns = np.r_[np.arange(left0, left1), np.arange(right0, right1)]
        distance = np.r_[a, b]
        within = np.abs(distance) < feather_mm
        _assign_blend(weights, _face_label(name), _face_label(neighbor), slice(None),
                      columns[within], distance[within], feather_mm)
    return weights


def m2_atlas_weights(topology, region_map, feather_mm):
    """Feather only contiguous side-face folds in the nonperiodic ControlNet canvas."""
    if feather_mm <= 0:
        raise ValueError("Atlas feather band must have positive physical width")
    labels = atlas_region_ids(region_map)
    weights = _hard_weights(labels)
    rectangles = {face: topology["faces"][face]["rect"] for face in SIDES}
    ppm = float(topology["image"]["pixels_per_mm"])
    centers_x = np.arange(labels.shape[1]) * LATENT_SCALE + LATENT_SCALE // 2 + 0.5
    centers_y = np.arange(labels.shape[0]) * LATENT_SCALE + LATENT_SCALE // 2 + 0.5
    for name, neighbor in zip(SIDES[:-1], SIDES[1:], strict=True):
        left0, y0, edge, y1 = rectangles[name]
        right0, right_y0, right1, right_y1 = rectangles[neighbor]
        if (edge, y0, y1) != (right0, right_y0, right_y1):
            raise ValueError(f"The atlas does not register the expected vertical fold {name}/{neighbor}")
        if feather_mm * ppm >= min(edge - left0, right1 - edge) / 2:
            raise ValueError("Atlas feather bands must stay disjoint within each face")
        distance = (centers_x - edge) / ppm
        columns = np.flatnonzero(np.abs(distance) < feather_mm)
        rows = np.flatnonzero((centers_y >= y0) & (centers_y < y1))
        _assign_blend(weights, _face_label(name), _face_label(neighbor), rows[:, None], columns,
                      distance[columns], feather_mm)
    return weights


class M2Routing:
    """Convert soft latent labels to sparse query groups at each attention resolution."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float32)
        if self.weights.ndim != 3 or min(self.weights.shape) < 1:
            raise ValueError("A soft routing map needs regional weights over a latent canvas")
        if not np.isfinite(self.weights).all() or self.weights.min() < 0 or not np.allclose(
                self.weights.sum(axis=0), 1, atol=1e-7):
            raise ValueError("M2 routing expects a nonnegative, normalized regional mixture")
        self.shift = 0
        self.indices = {}

    def set_shift(self, shift):
        shift %= self.weights.shape[-1]
        if shift != self.shift:
            self.indices.clear()
            self.shift = shift

    def weights_at_stride(self, stride):
        """Roll in full latent coordinates, then sample the downsampled cell centers."""
        offset = stride // 2
        width = self.weights.shape[-1]
        columns = (np.arange(offset, width, stride) - self.shift) % width
        return self.weights[:, offset::stride, :][..., columns]

    def groups(self, queries, device, dtype):
        lookup = (queries, device, dtype)
        if lookup not in self.indices:
            latent_h, latent_w = self.weights.shape[1:]
            stride = 1
            while latent_h * latent_w // (stride * stride) > queries:
                stride *= 2
            if latent_h % stride or latent_w % stride or latent_h * latent_w != queries * stride * stride:
                raise ValueError(f"Cannot route {queries} queries for M2 latent shape {(latent_h, latent_w)}")
            flat = self.weights_at_stride(stride).reshape(self.weights.shape[0], queries)
            groups = []
            for label, row in enumerate(flat):
                index = np.flatnonzero(row > 0)
                if len(index):
                    values = row[index]
                    scale = None if (values == 1).all() else torch.as_tensor(values, device=device, dtype=dtype)
                    groups.append((label, torch.as_tensor(index, device=device), scale))
            self.indices[lookup] = tuple(groups)
        return self.indices[lookup]


class M2Attention(AttnProcessor2_0):
    """Interpolate regional text over time, then blend attended features in space."""

    def __init__(self, routing, contexts, schedule):
        super().__init__()
        self.routing = routing
        self.contexts = contexts
        self.schedule = schedule
        self.weight = None
        self.conditioned = {}
        self.set_phase(0)

    def set_phase(self, index, shift=0):
        self.routing.set_shift(shift)
        weight = self.schedule[index]
        if weight == self.weight:
            return
        self.weight = weight
        global_context = self.contexts[0]
        self.conditioned = {0: global_context}
        if weight:
            self.conditioned.update({key: torch.stack((global_context[0],
                                                       global_context[1].lerp(context[1], weight)))
                                     for key, context in self.contexts.items() if key != 0})

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        if encoder_hidden_states is None or not self.weight:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        if attention_mask is not None:
            raise ValueError("M2 uses the original unmasked fixed-length text tokens")
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        spatial_shape = hidden_states.shape if hidden_states.ndim == 4 else None
        if spatial_shape is not None:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        heads = RegionalAttention._heads
        query = heads(attn.to_q(hidden_states), attn.heads)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        output = torch.zeros_like(query)
        for label, index, weight in self.routing.groups(query.shape[-2], query.device, query.dtype):
            context = self.conditioned[label]
            if attn.norm_cross:
                context = attn.norm_encoder_hidden_states(context)
            key = heads(attn.to_k(context), attn.heads)
            value = heads(attn.to_v(context), attn.heads)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            attended = F.scaled_dot_product_attention(query.index_select(2, index), key, value)
            if weight is not None:
                attended = attended * weight[None, None, :, None]
            output.index_add_(2, index, attended)
        output = output.transpose(1, 2).flatten(2).to(query.dtype)
        output = attn.to_out[1](attn.to_out[0](output))
        if spatial_shape is not None:
            output = output.transpose(1, 2).reshape(spatial_shape)
        if attn.residual_connection:
            output = output + residual
        return output / attn.rescale_output_factor


@contextmanager
def m2_attention(models, routing, contexts, schedule):
    """Scope the same temporal and spatial attention to all participating model blocks."""
    processor = M2Attention(routing, contexts, schedule)
    replaced = []
    try:
        for model in models:
            for layer in model.modules():
                if isinstance(layer, Attention) and layer.is_cross_attention:
                    replaced.append((layer, layer.processor))
                    layer.set_processor(processor)
        yield processor
    finally:
        for layer, original in reversed(replaced):
            layer.set_processor(original)


def m2_tokenizer_proof(tokenizers, prompts):
    """Check both SDXL streams before encoding, including the longest regional prompt."""
    evidence = {}
    for stream, tokenizer in enumerate(tokenizers, start=1):
        texts = [*prompts.values(), NEGATIVE]
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
        counts = [len(ids) for ids in encoded]
        maximum = int(tokenizer.model_max_length)
        if max(counts) > maximum:
            raise ValueError(f"M2 prompt would be truncated by CLIP stream {stream}: {counts}/{maximum}")
        evidence[f"tokenizer_{stream}"] = {"capacity": maximum, "counts": counts,
                                           "input_ids_sha256": stable_hash(encoded)}
    if len(evidence) != 2:
        raise ValueError("SDXL M2 expects two CLIP streams")
    return evidence


class M2RingMixin:
    """Expose the same prompt plan, schedule and registered routing to both renderers."""

    def _m2_conditions(self, brief):
        prompts = m2_prompt_plan(brief, self.config["m2"])
        tokenizers = m2_tokenizer_proof((self.pipe.tokenizer, self.pipe.tokenizer_2), prompts)
        positive, negative, pooled, negative_pooled = self.pipe.encode_prompt(
            prompt=list(prompts.values()), negative_prompt=[NEGATIVE] * len(prompts), device=self.device,
            do_classifier_free_guidance=True,
        )
        contexts = {label: torch.stack((negative[0], positive[index])) for index, label in enumerate(prompts)}
        self.m2_prompt_proof = {"positive_sha256": stable_hash(list(prompts.values())), "prompts": prompts,
                                "negative_prompt": NEGATIVE, "tokenizers": tokenizers}
        return contexts, torch.stack((negative_pooled[0], pooled[0]))

    def _m2_ring_routing(self, layout):
        return M2Routing(m2_ring_weights(self.topology, layout, self.config["m2"]["feather_mm"]))

    def _m2_schedule(self):
        return m2_schedule(self.config["steps"], self.config["m2"])

    def _conditions(self, brief):
        return self._m2_conditions(brief)

    def _prepare_ring(self, contexts, pooled, generator, size):
        if self._controlnet is not None:
            self._controlnet.to("cpu")
        ring, _, layouts = self._ring_layout(size)
        bank = {} if self.config.get("share_attention", False) else None
        unet = self.pipe.unet
        routing = self._m2_ring_routing(ring)
        with circular_ring_codec(unet, self.pipe.vae):
            with m2_attention((unet,), routing, contexts, self._m2_schedule()) as processor:
                with self._ring_phase(rolling=self.config.get("ring_roll", False),
                                      set_phase=processor.set_phase):
                    frozen = self._denoise((ring.width_px, ring.height_px), generator,
                                           contexts[0], pooled, capture_style=bank)
            image = self._rgb(frozen)
        return ring, layouts, bank, frozen, image


class M2RegionalFlat(M2RingMixin, DieLineDiffV3):
    """Use M2 in B2-regional while keeping its original sampler, ControlNet and atlas."""

    _flat_m2 = None

    def _predict(self, latent, timestep, text, condition, control):
        if self._flat_m2 is not None:
            processor, indices = self._flat_m2
            processor.set_phase(next(indices))
        return super()._predict(latent, timestep, text, condition, control)

    def _generate_flat(self, contexts, pooled, generator):
        control = self._control()
        weights = m2_atlas_weights(self.topology, self.region_map, self.config["m2"]["feather_mm"])
        schedule = self._m2_schedule()
        with m2_attention((self.pipe.unet, control[0]), M2Routing(weights), contexts, schedule) as processor:
            self._flat_m2 = processor, iter(range(len(schedule)))
            try:
                latent = self._denoise(self.atlas_size, generator, contexts[0], pooled, control=control)
            finally:
                self._flat_m2 = None
        height, width = self.region_map.shape
        self.last_run.update(flat_size_px=list(self.atlas_size), m2=self.m2_prompt_proof,
                             regional_weight=max(schedule), m2_weights=list(schedule),
                             m2_feather_mm=self.config["m2"]["feather_mm"])
        return self._rgb(latent).crop((0, 0, width, height))
