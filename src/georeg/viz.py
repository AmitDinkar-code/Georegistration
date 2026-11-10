"""Drawing helpers. Uses the matplotlib Figure API directly (no pyplot), so it
never needs a display / Tk backend - important on Windows servers and CI."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

from .io_utils import ensure_dir, imwrite_unicode


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def save_figure(fig: Figure, path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(str(p), dpi=120, bbox_inches="tight")
    return p


def draw_correspondences(gray_s, gray_r, layers, path=None, max_lines: int = 250,
                         seed: int = 0, max_width: int = 2000) -> np.ndarray:
    """Draw lines between two images side by side.

    layers: list of dict(pts_s, pts_r, color=(B,G,R), label=str, thickness=int).
    Layers are drawn in order (later on top). Lines per layer are sub-sampled.
    """
    hs, ws = gray_s.shape[:2]
    hr, wr = gray_r.shape[:2]
    scale = min(1.0, max_width / float(ws + wr))
    a = to_bgr(cv2.resize(gray_s, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)) if scale < 1 else to_bgr(gray_s)
    b = to_bgr(cv2.resize(gray_r, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)) if scale < 1 else to_bgr(gray_r)
    H = max(a.shape[0], b.shape[0])
    canvas = np.zeros((H + 28, a.shape[1] + b.shape[1], 3), np.uint8)
    canvas[28:28 + a.shape[0], :a.shape[1]] = a
    canvas[28:28 + b.shape[0], a.shape[1]:] = b
    rng = np.random.default_rng(seed)
    x_off = a.shape[1]
    legend_x = 6
    for layer in layers:
        ps = np.asarray(layer["pts_s"], float).reshape(-1, 2) * scale
        pr = np.asarray(layer["pts_r"], float).reshape(-1, 2) * scale
        n = len(ps)
        if n == 0:
            continue
        idx = rng.permutation(n)[: max(1, max_lines // max(1, len(layers)))]
        col = tuple(int(c) for c in layer.get("color", (0, 255, 0)))
        th = int(layer.get("thickness", 1))
        for i in idx:
            p1 = (int(round(ps[i, 0])), int(round(ps[i, 1])) + 28)
            p2 = (int(round(pr[i, 0])) + x_off, int(round(pr[i, 1])) + 28)
            cv2.line(canvas, p1, p2, col, th, cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, col, -1)
            cv2.circle(canvas, p2, 2, col, -1)
        label = f'{layer.get("label", "")} ({n})'
        cv2.putText(canvas, label, (legend_x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        legend_x += 14 * len(label) // 2 + 30
    if path is not None:
        imwrite_unicode(path, canvas)
    return canvas
