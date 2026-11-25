"""Render the OBJ's actual UVs orthographically without OpenGL or lighting."""

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


# Direction is from object to viewer; up is in the resulting image plane.
VIEWS = {
    "F": ((0, 0, 1), (0, 1, 0)),
    "B": ((0, 0, -1), (0, 1, 0)),
    "L": ((-1, 0, 0), (0, 1, 0)),
    "R": ((1, 0, 0), (0, 1, 0)),
    "T": ((0, 1, 0), (0, 0, -1)),
    "Btm": ((0, -1, 0), (0, 0, 1)),
    "iso": ((1.35, 0.90, 1.8), (0, 1, 0)),
}
BACKGROUND = (232, 236, 239)


def load_mesh(path):
    """Load the written OBJ without merging vertices across UV seams."""
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.visual.uv is None:
        raise ValueError(f"Expected one UV-mapped OBJ mesh: {path}")
    return mesh


def _project(vertices, direction, up, size):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    upward = np.cross(direction, right)
    projected = vertices @ np.array([right, -upward, direction]).T
    low, high = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
    scale = (size * 0.84) / (high - low).max()
    projected[:, :2] = (projected[:, :2] - (high + low) / 2) * scale + size / 2
    return projected


def _draw_triangle(canvas, depth_buffer, triangle, uv, texture):
    x0 = max(0, int(np.floor(triangle[:, 0].min())))
    x1 = min(canvas.shape[1], int(np.ceil(triangle[:, 0].max())))
    y0 = max(0, int(np.floor(triangle[:, 1].min())))
    y1 = min(canvas.shape[0], int(np.ceil(triangle[:, 1].max())))
    origin = triangle[0, :2]
    vectors = (triangle[1:, :2] - origin).T
    if abs(np.linalg.det(vectors)) < 1e-8 or x0 == x1 or y0 == y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    points = np.stack((xs + 0.5, ys + 0.5), axis=-1)
    other = (points - origin) @ np.linalg.inv(vectors).T
    weights = np.concatenate((1 - other.sum(axis=-1, keepdims=True), other), axis=-1)
    depth = weights @ triangle[:, 2]
    existing = depth_buffer[y0:y1, x0:x1]
    visible = (weights >= -1e-8).all(axis=-1) & (depth > existing)
    if not np.any(visible):
        return
    interpolated_uv = weights[visible] @ uv
    height, width = texture.shape[:2]
    ix = np.clip((interpolated_uv[:, 0] * width).astype(int), 0, width - 1)
    iy = np.clip(((1 - interpolated_uv[:, 1]) * height).astype(int), 0, height - 1)
    canvas[y0:y1, x0:x1][visible] = texture[iy, ix]
    existing[visible] = depth[visible]


def render_texture(mesh, texture, direction, up, size=480):
    """Draw a single flat-color view with depth occlusion."""
    if size < 32:
        raise ValueError("At least 32 pixels are needed for a render")
    texture = np.asarray(texture.convert("RGB"))
    depth_buffer = np.full((size, size), -np.inf)
    canvas = np.full((size, size, 3), BACKGROUND, dtype=np.uint8)
    projected = _project(mesh.vertices, direction, up, size)
    for indices in mesh.faces:
        _draw_triangle(canvas, depth_buffer, projected[indices], mesh.visual.uv[indices], texture)
    return Image.fromarray(canvas)


def render_views(geometry, texture, output, size=480, views=tuple(VIEWS)):
    """Render named orthographic views from an atlas image and save one PNG per view."""
    geometry, output = Path(geometry), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mesh = load_mesh(geometry / "box.obj")
    images = {}
    with Image.open(texture) as atlas:
        for name in views:
            images[name] = render_texture(mesh, atlas, *VIEWS[name], size=size)
            images[name].save(output / f"folded-{name}.png", optimize=True)
    return images
