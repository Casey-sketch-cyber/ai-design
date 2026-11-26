"""Trace reviewed cap patch boundaries into the exact pre-print cap geometry."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab

from .e01 import read_json, write_json
from .e01b import file_hash
from .e02_cap_energy_curves import CapEnergyCurves
from .e02_capfix_ring import bytes_digest
from .methods.v3_capgeom_geometry import build_capgeom_layout
from .methods.v3_geometry import LATENT_SCALE
from .methods.v3_uniform_texture import UNIFORM_TEXTURE_PARAMETERS


# Native cap-local locations were registered from the archived Task 20 review.
# The two endpoints locate the visible straight part, not the entire model frame.
@dataclass(frozen=True)
class Edge:
    name: str
    axis: str
    endpoints: tuple
    search_px: int = 4


EDGES_004 = (Edge("left_patch_side", "x", ((48, 24), (48, 77))),
             Edge("left_patch_foot", "y", ((5, 81), (43, 81)), 5))
EDGES_005 = (Edge("left_wood_block", "x", ((15, 65), (15, 220))),
             Edge("right_wood_block", "x", ((368, 35), (368, 169)), 5))
EDGES_007 = (Edge("upper_panel_border", "y", ((29, 20), (397, 18)), 4),
             Edge("left_panel_border", "x", ((16, 44), (8, 239)), 5),
             Edge("lower_panel_border", "y", ((24, 248), (423, 236)), 5))
EDGES_009 = (Edge("upper_flower_panel", "y", ((180, 34), (405, 34))),
             Edge("right_flower_panel", "x", ((412, 47), (412, 165))))
EDGES_010 = (Edge("right_leaf_block", "x", ((383, 80), (383, 205)), 6),)
REGISTERED = (
    ("j0", "E01-004", "Btm", EDGES_004), ("j0", "E01-009", "Btm", EDGES_009),
    ("j1", "E01-004", "Btm", EDGES_004), ("j1", "E01-005", "T", EDGES_005),
    ("j1", "E01-007", "T", EDGES_007), ("j1", "E01-009", "Btm", EDGES_009),
    ("j1", "E01-010", "T", EDGES_010),
    ("j3_g030", "E01-004", "Btm", EDGES_004), ("j3_g030", "E01-005", "T", EDGES_005),
    ("j3_g030", "E01-009", "Btm", EDGES_009), ("j3_g030", "E01-010", "T", EDGES_010),
)
PALETTE = {
    "highpass_nearest_coarse_start": (244, 208, 105),
    "known_mask": (232, 112, 88), "known_mask_corners": (232, 112, 88),
    "nearest_edge_partition": (92, 179, 195), "highpass_support": (124, 192, 116),
    "highpass_phase_lock_intersections": (241, 184, 89), "highpass_coarse_full": (183, 139, 222),
    "print_crop": (144, 243, 244), "preflight_glue": (196, 96, 142),
    "preflight_tuck": (244, 140, 37), "preflight_dust": (236, 216, 83),
    "latent_grid": (37, 51, 76), "reviewed_edge": (255, 49, 116),
}


def native_to_model(points, native_size, core):
    """Map cap-local pixel-corner coordinates through the production UV affine."""
    x0, y0, x1, y1 = core
    return np.asarray(points, dtype=np.float64) * ((x1 - x0) / native_size[0],
                                                 (y1 - y0) / native_size[1]) + (x0, y0)


def _runs(values):
    padded = np.pad(values.astype(np.int8), 1)
    return np.flatnonzero(np.diff(padded)).reshape(-1, 2)


def mask_segments(mask):
    """Trace exact pixel-corner edges around a binary raster, merging runs."""
    mask = np.asarray(mask, dtype=bool)
    edges = []
    for normal, array in enumerate((mask.T, mask)):
        outer = np.pad(array, ((1, 1), (0, 0)))
        for loc, row in enumerate(outer[1:] != outer[:-1]):
            for a, b in _runs(row):
                edges.append(((loc, a), (loc, b)) if normal == 0 else ((a, loc), (b, loc)))
    return np.asarray(edges, dtype=np.float64).reshape(-1, 2, 2)


def _rect(rect):
    x0, y0, x1, y1 = rect
    values = np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return np.stack((values, np.roll(values, -1, axis=0)), axis=1)


def _inset(core, amount):
    a, b, c, d = core
    return a + amount, b + amount, c - amount, d - amount


def _bisectors(core):
    a, b, c, d = core
    h, w = d - b, c - a
    length = min(h, w) / 2
    ul, ur, ll, lr = ((a + length, b + length), (c - length, b + length),
                      (a + length, d - length), (c - length, d - length))
    center = (ul, ur) if w >= h else (ul, ll)
    return np.asarray([((a, b), ul), ((c, b), ur), ((a, d), ll), ((c, d), lr), center])


def _boundary_layers(layout, labels, atlas_rect):
    core = tuple(LATENT_SCALE * t for t in layout.center_rect)
    x0, y0, x1, y1 = atlas_rect
    painter = {"known_mask": mask_segments(layout.known_mask) * LATENT_SCALE,
               "nearest_edge_partition": _bisectors(core),
               "highpass_support": _rect(core),
               "print_crop": _rect(core)}
    lock = int(UNIFORM_TEXTURE_PARAMETERS["phase_lock_px_at_scale_1"] * layout.scale)
    start = lock - 0.5
    a, b, c, d = core
    painter["highpass_phase_lock_intersections"] = np.asarray([
        ((a + start, b), (a + start, d)), ((c - start, b), (c - start, d)),
        ((a, b + start), (c, b + start)), ((a, d - start), (c, d - start)),
    ])
    painter["highpass_nearest_coarse_start"] = _rect(_inset(core, start))
    painter["highpass_coarse_full"] = _rect(_inset(core, 2 * lock - 0.5))
    for name, tag in (("glue", 7), ("tuck", 8), ("dust", 9)):
        projected = native_to_model(mask_segments(labels == tag) - (x0, y0), (x1-x0, y1-y0), core)
        painter[f"preflight_{name}"] = projected
    return core, painter


def _points_to_segments(points, segments):
    if not len(segments):
        return None
    start, stop = segments[:, 0], segments[:, 1]
    delta = stop - start
    length2 = np.square(delta).sum(axis=-1)
    difference = points[:, None, :] - start
    fractions = (difference * delta).sum(axis=-1) / np.maximum(length2, 1e-20)
    outgoing = difference - np.clip(fractions, 0, 1)[..., None] * delta
    return np.sqrt(np.square(outgoing).sum(axis=-1).min(axis=1))


def _distances(values, tolerance):
    if values is None:
        return None
    return {"min_px": float(np.min(values)), "median_px": float(np.median(values)),
            "max_px": float(np.max(values)), "within_tolerance_fraction": float(np.mean(values <= tolerance))}


def _stripe_edge(image, annotation):
    """Refine a human-reviewed edge by a short 2px-vs-2px Lab jump search."""
    lab = rgb2lab(np.asarray(image).astype(np.float32)/255)
    normal = 0 if annotation.axis == "x" else 1
    view = lab.transpose(1, 0, 2) if normal == 0 else lab
    a, b = np.asarray(annotation.endpoints, dtype=np.float64)
    tangent = np.arange(int(a[1-normal]), int(b[1-normal]))
    normal_coords = np.linspace(a[normal], b[normal], len(tangent), endpoint=False)
    candidates = []
    for offset in range(-annotation.search_px, annotation.search_px + 1):
        center = np.rint(normal_coords + offset).astype(int)
        if min(center) < 2 or max(center) > len(view) - 2:
            continue
        before = (view[center-2, tangent] + view[center-1, tangent]) / 2
        after = (view[center, tangent] + view[center+1, tangent]) / 2
        jump = np.linalg.norm(after - before, axis=1)
        candidates.append((float(np.median(jump)), offset))
    if not candidates:
        raise ValueError(f"No valid native pixels at {annotation}")
    strongest, offset = max(candidates, key=lambda pair: (pair[0], -abs(pair[1])))
    result = np.stack((a, b))
    result[:, normal] += offset
    return result, strongest, offset


def _grid_evidence(points, normal, native_size, core, tolerance):
    center = points[:, normal]
    cell = LATENT_SCALE * round(float(np.median(center)) / LATENT_SCALE)
    distances = np.abs(center - cell)
    increments = np.asarray(core[2:]) - core[:2]
    span = np.arange(1, native_size[normal])
    plain_grid = core[normal] + span * increments[normal] / native_size[normal]
    offset = np.abs(plain_grid - LATENT_SCALE * np.rint(plain_grid / LATENT_SCALE))
    return {**_distances(distances,tolerance), "nearest_parallel_grid_coordinate_px": cell,
            "native_integer_null_1px": float(np.mean(offset <= 1)),
            "native_integer_null_2px": float(np.mean(offset <= 2)),
            "whole_edge_within_1px": bool(max(distances) <= 1),
            "whole_edge_within_2px": bool(max(distances) <= 2)}


def _measure_edge(image, annotation, native_size, core, layers):
    detected, contrast, shift = _stripe_edge(image, annotation)
    transformed = native_to_model(detected, native_size, core)
    points = np.linspace(*transformed, 97)
    normal = 0 if annotation.axis == "x" else 1
    tolerance = (core[normal+2] - core[normal])/native_size[normal]
    corners = np.unique(layers["known_mask"].reshape(-1, 2), axis=0)
    evidence = {name:_distances(_points_to_segments(points, segments), tolerance)
                for name, segments in layers.items()}
    evidence["known_mask_corners"] = _distances(_points_to_segments(points, np.stack((corners,corners), axis=1)),
                                                  tolerance)
    evidence["latent_grid"] = _grid_evidence(points, normal, native_size, core, tolerance)
    return {"edge": annotation.name, "axis": annotation.axis, "native_endpoints": detected.tolist(),
            "native_search_shift": shift, "median_native_2px_delta_e76": contrast,
            "model_endpoints": transformed.tolist(), "localization_1_native_px_as_model_px": tolerance,
            "distances": evidence}



def _draw_geometry(path, layout, source_rgb, native_face, core, layers, edges, label):
    known = layout.known_mask.repeat(LATENT_SCALE, 0).repeat(LATENT_SCALE, 1)
    context = np.where(known[..., None], source_rgb, 231).astype(np.uint8)
    sheet = Image.new("RGB", (layout.width_px + 300, layout.height_px), (245, 247, 250))
    sheet.paste(Image.fromarray(context), (0, 0))
    a, b, c, d = core
    sheet.paste(native_face.resize((c-a, d-b), Image.Resampling.BILINEAR), (a,b))
    draw = ImageDraw.Draw(sheet, "RGBA")
    for x in range(0,layout.width_px, LATENT_SCALE):
        draw.line((x,b,x,d), fill=(*PALETTE["latent_grid"],45), width=1)
    for y in range(0,layout.height_px, LATENT_SCALE):
        draw.line((a,y,c,y), fill=(*PALETTE["latent_grid"],45), width=1)
    for name, segments in layers.items():
        if name == "print_crop":
            continue
        for p, q in segments:
            draw.line((*p, *q), fill=(*PALETTE[name], 224), width=3)
    for x, y in np.unique(layers["known_mask"].reshape(-1,2),axis=0):
        draw.ellipse((x-4,y-4,x+4,y+4), fill=(*PALETTE["known_mask_corners"],255))
    for p, q in layers["print_crop"]:
        delta = q - p
        for u in np.arange(0.0, 1.0, 0.025):
            z = p + u*delta
            w = p + min(u+0.012,1.0)*delta
            draw.line((*z,*w), fill=(*PALETTE["print_crop"],244), width=3)
    for index, edge in enumerate(edges,start=1):
        p,q=edge["model_endpoints"]
        draw.line((*p,*q),fill=(*PALETTE["reviewed_edge"],244),width=4)
        draw.text((p[0]+4,p[1]+2),str(index),fill=(255,255,255,255),
                  font=ImageFont.load_default(size=21),stroke_width=2,stroke_fill=(22,20,24))
    x=layout.width_px+16
    title=ImageFont.load_default(size=21)
    font=ImageFont.load_default(size=16)
    draw.text((x,18),label,font=title,fill=(27,39,55))
    draw.text((x,53),"2x source geometry / px",font=font,fill=(36,46,59))
    draw.text((x,77),"Raw face projected for reference",font=font,fill=(36,46,59))
    legend_names = ("Known strips and corners", "Nearest-side ties", "Residual support",
                    "Per-edge phase lock ends", "Nearest coarse gate starts", "Coarse gate reaches full",
                    "Print crop (dashes)",
                    "Glue reset outline", "Tuck reset outline", "Dust reset outline",
                    "8 px latent grid", "Reviewed native edge")
    palette_names = ("known_mask","nearest_edge_partition","highpass_support",
                     "highpass_phase_lock_intersections", "highpass_nearest_coarse_start",
                     "highpass_coarse_full", "print_crop", "preflight_glue",
                     "preflight_tuck","preflight_dust","latent_grid","reviewed_edge")
    for row,(name,color) in enumerate(zip(legend_names,palette_names,strict=True)):
        y=128+29*row
        draw.line((x,y+8,x+20,y+8),fill=(*PALETTE[color],255),width=4)
        draw.text((x+27,y),name,font=font,fill=(34,49,69))
    y=514
    draw.text((x,y),"Native source crop",font=font,fill=(34,49,69))
    display=native_face.copy()
    display.thumbnail((266,266),Image.Resampling.LANCZOS)
    sheet.paste(display,(x,y+25))
    sheet.save(path,optimize=True)


def _jsonl_csv(path, rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _match_summary(rows):
    grid=[row["distances"]["latent_grid"] for row in rows]
    families={}
    for row in rows:
        key=row["brief_id"],row["face"],row["edge"]
        families.setdefault(key,[]).append(row)
    matches={name:sum(r["distances"][name] is not None and
                       r["distances"][name]["within_tolerance_fraction"]>=.8 for r in rows)
             for name in rows[0]["distances"]}
    return {"straight_edges":len(rows),"distinct_brief_face_edge_families":len(families),
            "within_measured_1_native_pixel_for_at_least_80pct_by_boundary":matches,
            "grid_whole_edge_within_1px":sum(g["whole_edge_within_1px"] for g in grid),
            "grid_whole_edge_within_2px":sum(g["whole_edge_within_2px"] for g in grid),
            "grid_uniform_phase_expected_at_1px":0.25,
            "grid_uniform_phase_expected_at_2px":0.5,
            "grid_native_integer_expected_at_1px_mean":mean(g["native_integer_null_1px"] for g in grid),
            "grid_native_integer_expected_at_2px_mean":mean(g["native_integer_null_2px"] for g in grid),
            "grid_families_with_any_variant_within_2px":sum(any(v["distances"]["latent_grid"]["whole_edge_within_2px"]
                                                             for v in family) for family in families.values()),
            "referenced_versions_are_paired_variants_not_independent_trials":True}


def diagnose(root, output, coherence=None):
    """Localize the eleven reviewed cases and audit geometry without changing any gates."""
    root,output=Path(root),Path(output)
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Keep panel evidence on the project data disk")
    output.mkdir(parents=True,exist_ok=True)
    e02=root/"outputs/e02"
    archive=Path(coherence) if coherence is not None else e02/"joint-cap/runs/3411a058f0e6"
    published=Path("docs/experiments/E02/joint-cap-content-review.json")
    reviews=read_json(published)
    geometry=root/"outputs/e01b/geometry"
    settings=read_json(e02/"step1/c6bb770094d3/inputs.json")["generation_config"]["common"]
    analyzer=CapEnergyCurves(geometry,cap_band=int(settings.get("cap_band",16)))
    layouts={}
    for face in ("T","Btm"):
        layout=build_capgeom_layout(analyzer.caps[face],analyzer.ring,analyzer.cap_band,2)
        rect=tuple(analyzer.topology["faces"][face]["rect"])
        core,layers=_boundary_layers(layout,analyzer.labels,rect)
        layouts[face]=layout,rect,core,layers
    case_evidence, rows = {}, []
    source_files=set()
    curves={}
    for method,brief,face,edge_regions in REGISTERED:
        curve_source=archive/"cap-energy-curves"/f"{method}-42.json"
        if method not in curves:
            curves[method]=read_json(curve_source)["cases"]
        stored=curves[method][brief]["source"]
        path=Path(stored["raw_path"])
        source_files.add(curve_source)
        registered=reviews[method]["samples"][brief]
        if not registered["collage"] or registered["raw_sha256"] != file_hash(path) or stored["raw_sha256"] != file_hash(path):
            raise ValueError(f"Unbound Task 20 review image for {method}/{brief}")
        record=read_json(path.with_suffix(".run.json"))
        if file_hash(path.with_suffix(".run.json")) != stored["run_sha256"]:
            raise ValueError(f"Archived generation metadata differs: {path}")
        layout,rect,core,layers=layouts[face]
        cap_info=record["details"]["caps"][face]
        if ([layout.width_px,layout.height_px]!=cap_info["size_px"]
                or [core[2]-core[0],core[3]-core[1]]!=cap_info["core_size_px"]
                or int(layout.known_mask.sum())!=cap_info["known_latents"]):
            raise ValueError(f"Rebuilt cap layout differs for {path}")
        preflight=path.parent.parent/"evaluation"/path.parent.name/(path.stem+".preflight.png")
        with Image.open(path) as png, Image.open(preflight) as print_ready:
            native=png.crop(rect).convert("RGB")
            restored=print_ready.crop(rect).convert("RGB")
        equal=np.array_equal(native,restored)
        a,b,c,d=rect
        reset_is_clear=not np.isin(analyzer.labels[b:d,a:c], [0,7,8,9]).any()
        if not equal or not reset_is_clear:
            raise ValueError("The checked print cap must be untouched by preflight")
        archive_ring=Path(stored["ring_path"])
        if file_hash(archive_ring)!=stored["ring_sha256"]:
            raise ValueError("The paired Task 20 R0 ring cache has changed")
        with np.load(archive_ring,allow_pickle=False) as payload:
            canvas=payload["rgb"]
        if bytes_digest(canvas)!=record["ring"]["ring_rgb_sha256"]:
            raise ValueError("R0 archived ring pixels differ from the candidate input")
        source_rgb=np.asarray(layout.read_frozen_ring_rgb(Image.fromarray(canvas)))
        detected=[_measure_edge(native,e,native.size,core,layers) for e in edge_regions]
        ident=f"{method}-{brief}-{face}"
        native_file=output/f"{ident}-native.png"
        native.save(native_file,optimize=True)
        x0, y0, x1, y1 = layout.center_rect
        inside = int(layout.known_mask[y0:y1, x0:x1].sum()) * LATENT_SCALE ** 2
        if inside:
            raise ValueError("An exact known stripe overlaps the print core.")
        overlay=output/f"{ident}-geometry-2x.png"
        _draw_geometry(overlay,layout,source_rgb,native,core,layers,detected,ident)
        case_evidence[ident]={"native_path":str(path),"native_sha256":file_hash(path),
                              "review_notes":registered["notes"],"registered_print_crop":list(rect),
                              "model_canvas_size":[layout.width_px,layout.height_px],"model_core":list(core),
                              "run_sha256":file_hash(path.with_suffix(".run.json")),
                              "preflight_sha256":file_hash(preflight),"raw_equals_preflight_print_cap":equal,
                              "r0_cache_sha256":file_hash(archive_ring),"known_mask_in_core":inside,
                              "overlay_sha256":file_hash(overlay),"native_crop_sha256":file_hash(native_file)}
        rows.extend({"method": method,"brief_id":brief,"face":face,**line} for line in detected)
        print(f"Panel geometry {ident}: {len(detected)} edges",flush=True)
    cells=[]
    for row in rows:
        one={k:v for k,v in row.items() if k not in ("distances","native_endpoints","model_endpoints")}
        for key in ("native_endpoints","model_endpoints"):
            for i, (x,y) in enumerate(row[key],start=1):
                one[f"{key}_{i}_x"],one[f"{key}_{i}_y"]=x,y
        for family, gaps in row["distances"].items():
            if gaps is not None:
                one.update({f"{family}_{name}":value for name,value in gaps.items()})
        cells.append(one)
    _jsonl_csv(output/"panel-edges.csv",cells)
    proof={"seed":42,"cases":case_evidence,"edges":rows,"summary":_match_summary(rows),
           "geometry_sha256":{p.name:file_hash(p) for p in (geometry/"topology.json",geometry/"region_map.png")},
           "content_review_sha256":file_hash(published),
           "source_curves_sha256":{p.name:file_hash(p) for p in sorted(source_files)},
           "implementation_sha256":file_hash(Path(__file__)),
           "edge_csv_sha256":file_hash(output/"panel-edges.csv")}
    write_json(output/"panel-geometry.json",proof)
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--archive",type=Path)
    args=parser.parse_args()
    diagnose(args.root,args.output,args.archive)


if __name__ == "__main__":
    main()
