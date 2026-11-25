"""Render registered comparison examples and unresampled native crease views."""

from PIL import Image, ImageDraw, ImageFont

from .e01 import read_json
from .metrics.physical import _edges_on_sheet
from .render import render_views


LABELS = {"b2": "B2", "b2_regional": "B2-regional"}


def _preview(image, size):
    tile = image.copy()
    tile.thumbnail(size, Image.Resampling.LANCZOS)
    return tile


def render_examples(experiment, chosen):
    records = {(job.atlas.brief["id"], job.atlas.method): job for job in
               experiment.samples(methods=("b2", "b2_regional", chosen))}
    destination = experiment.output / "examples"
    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    columns = ("b2", "b2_regional", chosen)
    tile_width, tile_height = 320, 340
    for brief in experiment.briefs[:3]:
        brief_id = brief["id"]
        canvas = Image.new("RGB", (3 * tile_width + 36, 2 * tile_height + 61), "white")
        draw = ImageDraw.Draw(canvas)
        ink = (28, 42, 55)
        draw.text((18, 10), f"{brief_id} / seed 42 / Preflight atlas and folded view", fill=ink, font=font)
        for i, method in enumerate(columns):
            job = records[brief_id, method]
            fold_dir = destination / brief_id / method
            render_views(job.atlas.geometry, job.preflight, fold_dir, size=350, views=["iso"])
            x0 = 18 + i * tile_width
            draw.text((x0, 40), LABELS.get(method, "v3"), fill=ink, font=small)
            for j, source in enumerate((job.preflight, fold_dir / "folded-iso.png")):
                with Image.open(source) as image:
                    tile = _preview(image.convert("RGB"), (tile_width - 13, tile_height - 13))
                canvas.paste(tile, (x0 + (tile_width - tile.width) // 2,
                                    64 + j * tile_height + (tile_height - tile.height) // 2))
        canvas.save(destination / f"{brief_id}-comparison.jpg", quality=89, optimize=True)


def _native_creases(topology):
    faces = topology["faces"]
    for seam in topology["adjacencies"]:
        a, b = seam["a"], seam["b"]
        first = _edges_on_sheet(faces[a["face"]]["rect"], a["edge"])
        second = _edges_on_sheet(faces[b["face"]]["rect"], b["edge"])
        if frozenset(first) == frozenset(second):
            yield a["face"], b["face"], first


def render_native_creases(experiment, chosen):
    """Inspect five physical folds in all ten samples, staying at native pixel scale."""
    topology = read_json(experiment.geometry / "topology.json")
    creases = tuple(_native_creases(topology))
    if len(creases) != 5:
        raise ValueError("Expected five registered native physical folds")
    destination = experiment.output / "native-folds"
    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=14)
    radius = 31
    long_edge = max(max(x1 - x0, y1 - y0) for _, _, ((x0, y0), (x1, y1)) in creases)
    crop_refs = []
    for job in experiment.samples(methods=(chosen,)):
        with Image.open(job.atlas.path) as source:
            atlas = source.convert("RGB")
        canvas = Image.new("RGB", (long_edge + 39, 46 + (radius * 2 + 26) * len(creases)), "white")
        draw = ImageDraw.Draw(canvas)
        dark, tick = (29, 43, 56), (25, 149, 162)
        draw.text((18, 10), f"{job.atlas.brief['id']} / v3 / native pixels", font=font, fill=dark)
        for i, (face, neighbor, ((x0, y0), (x1, y1))) in enumerate(creases):
            y = 50 + i * (2 * radius + 26)
            label = f"{face} / {neighbor}"
            if y0 == y1:
                crop = atlas.crop((x0, y0 - radius, x1, y0 + radius))
            else:
                crop = atlas.crop((x0 - radius, y0, x0 + radius, y1)).transpose(Image.Transpose.ROTATE_90)
                label += " (90-degree rotation)"
            draw.text((18, y - 5), label, fill=dark, font=font, anchor="lb")
            canvas.paste(crop, (18, y))
            for x in (5, 23 + crop.width):
                draw.line((x, y + radius, x + 8, y + radius), fill=tick, width=2)
        path = destination / f"{job.atlas.brief['id']}-folds.png"
        canvas.save(path, optimize=True)
        crop_refs.append(path)
    return crop_refs
