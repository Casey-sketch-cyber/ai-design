"""Harmonic RGB initialization and the exact truncated DDIM schedule for cap SDEdit."""

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
import torch


def _laplace(image, degree):
    result = image * degree[..., None]
    result[:-1] -= image[1:]
    result[1:] -= image[:-1]
    result[:, :-1] -= image[:, 1:]
    result[:, 1:] -= image[:, :-1]
    return result


@lru_cache(maxsize=4)
def _dirichlet_system(shape, encoded_mask):
    from pyamg import ruge_stuben_solver

    known = np.frombuffer(encoded_mask, dtype=bool).reshape(shape)
    unknown = np.nonzero(~known)
    count = len(unknown[0])
    index = np.full(shape, -1, dtype=np.int32)
    order = np.arange(count, dtype=np.int32)
    index[unknown] = order
    degree = np.full(shape, 4.0)
    degree[[0, -1], :] -= 1
    degree[:, [0, -1]] -= 1
    rows, columns = [order], [order]
    for left, right in ((index[1:], index[:-1]), (index[:, 1:], index[:, :-1])):
        live = (left >= 0) & (right >= 0)
        a, b = left[live], right[live]
        rows.extend((a, b))
        columns.extend((b, a))
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    weights = np.full(len(rows), -1.0)
    weights[:count] = degree[unknown]
    matrix = sparse.csr_matrix((weights, (rows, columns)), shape=(count, count))
    preconditioner = ruge_stuben_solver(matrix, max_coarse=100).aspreconditioner(cycle="V")
    return unknown, degree, matrix, preconditioner


def harmonic_fill(rgb, known_mask, *, tolerance=3e-8, max_iterations=80):
    """Solve the RGB Dirichlet problem on every unknown pixel, including virtual corners.

    Pixels on the frozen strips are exactly fixed and the outer canvas has zero
    normal flux. Algebraic multigrid preconditions CG on the original full-resolution
    grid. The hierarchy depends only on the geometric mask and is reused across briefs.
    """
    source = np.asarray(rgb)
    mask = np.asarray(known_mask, dtype=bool)
    if source.ndim != 3 or source.shape[2] != 3 or source.shape[:2] != mask.shape or min(mask.shape) < 2:
        raise ValueError("Harmonic initialization expects H×W×3 RGB and a matching 2D mask")
    if not mask.any():
        raise ValueError("Harmonic initialization requires frozen Dirichlet pixels")
    values = source.astype(np.float64)
    if source.dtype == np.uint8:
        values /= 255
    boundary = values[mask]
    if np.any(~np.isfinite(boundary)) or np.any((boundary < 0) | (boundary > 1)):
        raise ValueError("Harmonic RGB boundary values must lie in [0,1]")
    if mask.all():
        return values.astype(np.float32), {"iterations": 0, "laplace_residual_max": 0.0}
    unknown, degree, matrix, preconditioner = _dirichlet_system(mask.shape, mask.tobytes())
    image = np.where(mask[..., None], values, 0.0)
    rhs = -_laplace(image, degree)[unknown]
    means = boundary.mean(axis=0)
    counts = []
    for channel in range(3):
        cycles = 0

        def count_cycle(_):
            nonlocal cycles
            cycles += 1

        solved, status = cg(
            matrix, rhs[:, channel], x0=np.full(len(rhs), means[channel]), M=preconditioner,
            rtol=0, atol=tolerance / 20, maxiter=max_iterations, callback=count_cycle,
        )
        if status != 0:
            raise RuntimeError(f"Harmonic cap solve did not converge for RGB channel {channel}: {status}")
        image[unknown[0], unknown[1], channel] = solved
        counts.append(cycles)
    residual = float(np.max(np.abs(_laplace(image, degree)[unknown])))
    if residual > tolerance:
        raise RuntimeError(f"Harmonic RGB solve exceeded residual tolerance: {residual:.3g}")
    return image.clip(0, 1).astype(np.float32), {
        "iterations": max(counts), "channel_iterations": counts, "laplace_residual_max": residual,
    }


def sdedit_schedule(scheduler, steps, strength, device):
    """Keep the complete DDIM schedule, executing its last floor(steps*strength) entries."""
    if not 0 < strength <= 1:
        raise ValueError("SDEdit strength must be in (0, 1]")
    scheduler.set_timesteps(steps, device=device)
    count = int(steps * strength)
    if not count or scheduler.order != 1 or len(scheduler.timesteps) != steps:
        raise ValueError("SDEdit requires at least one active step and a first-order DDIM schedule")
    start = steps - count
    return start, scheduler.timesteps[start:]


def sdedit_latents(scheduler, initial, first_timestep, generator):
    """Add the scheduler's forward noise at the first timestep using one original draw."""
    noise = torch.randn(initial.shape, dtype=initial.dtype, device=initial.device, generator=generator)
    noised = scheduler.add_noise(initial, noise, first_timestep.reshape(1))
    return noised, noise
