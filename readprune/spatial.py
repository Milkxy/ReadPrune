"""Spatial transfer building blocks for the auxiliary-CLIP Qwen adaptation.

These functions map an auxiliary signed response field onto a native token
grid. They do not implement Qwen decoding, positional indices, token packing,
or a native Qwen deletion response. Apply the absolute-value quality transform
only after interpolation, and form diversity from the native model features.
"""

import math

import numpy as np


def retain_budget(n, retain_ratio):
    """Floor a fraction of native tokens, retaining at least one token."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    ratio = float(retain_ratio)
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("retain_ratio must be in (0,1]")
    return max(1, min(n, int(ratio * n)))


def _grid(grid_hw):
    if len(grid_hw) != 2 or any(isinstance(v, bool) or int(v) != v or v < 1 for v in grid_hw):
        raise ValueError("grid_hw must contain two positive integers")
    return tuple(map(int, grid_hw))


def map_letterbox_features(values, original_wh, grid_hw, source_hw=(24, 24)):
    """Bilinear sample source patch centers at Qwen token centers.

    PIL places the original image at integer (left,top) in a square canvas.
    Sampling uses original-image coordinates, then the exact letterbox offset.
    Border replication applies only outside the outermost CLIP patch centers.
    Signed response vectors are interpolated BEFORE the abs/deadzone transform.
    """
    h, w = _grid(grid_hw)
    sh, sw = _grid(source_hw)
    ow, oh = _grid(original_wh)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) != sh * sw or not np.isfinite(values).all():
        raise ValueError("invalid source-grid features")
    side = max(ow, oh)
    left, top = (side - ow) // 2, (side - oh) // 2
    x = (left + (np.arange(w, dtype=np.float64) + .5) * ow / w) / side * sw - .5
    y = (top + (np.arange(h, dtype=np.float64) + .5) * oh / h) / side * sh - .5
    x, y = np.clip(x, 0., sw - 1.), np.clip(y, 0., sh - 1.)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, sw - 1), np.minimum(y0 + 1, sh - 1)
    wx, wy = (x - x0)[None, :, None], (y - y0)[:, None, None]
    field = values.reshape(sh, sw, -1)
    lo = (1 - wx) * field[y0[:, None], x0[None, :]] + wx * field[y0[:, None], x1[None, :]]
    hi = (1 - wx) * field[y1[:, None], x0[None, :]] + wx * field[y1[:, None], x1[None, :]]
    mapped = ((1 - wy) * lo + wy * hi).reshape(h * w, -1)
    return np.ascontiguousarray(mapped), {
        "original_wh": [ow, oh], "letterbox_side": side,
        "letterbox_offset_xy": [left, top], "source_grid_hw": [sh, sw],
        "target_grid_hw": [h, w], "coordinate_convention": "patch centers; original-image to padded-square",
        "interpolation": "bilinear signed vectors; replicate outer patch-center border",
        "response_semantics": "transferred auxiliary CLIP response; not native Qwen deletion response",
    }
