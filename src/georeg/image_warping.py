"""Phase 5: resample the source onto the reference grid.

Direction matters. cv2.warpPerspective(src, T, (W,H)) *without* WARP_INVERSE_MAP
treats T as source->destination and internally samples the source at T^-1(x,y)
for every destination pixel (backward mapping, so no holes). T is exactly what
transformation_estimator returns (source px -> reference px), so the output has
the reference's width, height and pixel grid.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

from .dataset_loader import to_gray_float
from .viz import save_figure

INTERP = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
_ERODE = {"nearest": 0, "bilinear": 1, "cubic": 2}   # interpolation kernels bleed in nodata at edges
_NATIVE = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


@dataclass
class WarpResult:
    data: np.ndarray       # (C,H,W), dtype of the source
    valid: np.ndarray      # (H,W) bool
    interpolation: str
    seconds: float


def warp_to_reference(data: np.ndarray, T: np.ndarray, ref_hw, interpolation: str = "bilinear",
                      src_valid: np.ndarray | None = None, nodata=0) -> WarpResult:
    if interpolation not in INTERP:
        raise ValueError(f"interpolation must be one of {list(INTERP)}")
    t0 = time.perf_counter()
    H, W = ref_hw
    flag = INTERP[interpolation]
    out_bands = []
    for b in data:
        work = b if b.dtype.type in _NATIVE else b.astype(np.float32)
        w = cv2.warpPerspective(work, T, (W, H), flags=flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out_bands.append(w.astype(b.dtype, copy=False))
    out = np.stack(out_bands)

    sv = np.ones(data.shape[1:], np.uint8) * 255 if src_valid is None else src_valid.astype(np.uint8) * 255
    valid = cv2.warpPerspective(sv, T, (W, H), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
    n = _ERODE[interpolation]
    if n:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=n).astype(bool)
    out[:, ~valid] = nodata
    return WarpResult(out, valid, interpolation, time.perf_counter() - t0)


def psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def roundtrip_psnr(data: np.ndarray, T: np.ndarray, ref_hw, interpolation: str, src_valid=None) -> float:
    """Warp to the reference grid and back to the source grid; PSNR vs the original on
    pixels away from the border. Isolates resampling loss from registration error."""
    fwd = warp_to_reference(data, T, ref_hw, interpolation, src_valid)
    back = warp_to_reference(fwd.data, np.linalg.inv(T), data.shape[1:], interpolation, fwd.valid)
    m = back.valid & (np.ones(data.shape[1:], bool) if src_valid is None else src_valid)
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3).astype(bool)
    if not m.any():
        return float("nan")
    peak = float(np.iinfo(data.dtype).max) if data.dtype.kind in "ui" else float(np.nanmax(data))
    return psnr(data[:, m], back.data[:, m], peak)


def _pixel_keys(data: np.ndarray) -> np.ndarray:
    """One integer per pixel encoding ALL bands (<=4 bands of 8 bit): a pixel keeps its
    key only if its whole colour vector was copied, not blended."""
    if data.dtype == np.uint8 and data.shape[0] <= 4:
        k = np.zeros(data.shape[1:], np.int64)
        for b in data:
            k = (k << 8) | b.astype(np.int64)
        return k
    return data[0].astype(np.float64)


def compare_interpolations(data, T, ref_hw, ref_gray, src_valid=None, nodata=0) -> tuple[list[dict], dict]:
    """Nearest vs bilinear vs cubic, measured (not asserted):
    * ncc_vs_reference  - similarity to the reference on the common valid area
    * sharpness         - variance of the Laplacian (higher = crisper, but noise raises it too)
    * value_preserved   - share of output pixels whose full colour vector exists in the source
                         (1.0 only for nearest: blending invents new values)
    * roundtrip_psnr_db - resampling loss, forward and back
    """
    results = {k: warp_to_reference(data, T, ref_hw, k, src_valid, nodata) for k in INTERP}
    common = np.logical_and.reduce([r.valid for r in results.values()])
    src_vals = _pixel_keys(data)
    rows = []
    for k, r in results.items():
        g = to_gray_float(r.data)
        lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
        a, b = g[common], ref_gray.astype(np.float32)[common]
        ncc = float(np.corrcoef(a, b)[0, 1]) if common.sum() > 10 else float("nan")
        rows.append(dict(interpolation=k, seconds=r.seconds, ncc_vs_reference=ncc,
                         sharpness=float(lap[common].var()) if common.any() else float("nan"),
                         value_preserved=float(np.isin(_pixel_keys(r.data)[common], src_vals).mean()) if common.any() else float("nan"),
                         roundtrip_psnr_db=roundtrip_psnr(data, T, ref_hw, k, src_valid),
                         valid_pixels=int(r.valid.sum())))
    return rows, results


def save_interpolation_figure(results: dict, rows: list[dict], path, zoom: int = 40) -> Path:
    """Zoomed crop around the centre of the footprint for the three kernels."""
    common = np.logical_and.reduce([r.valid for r in results.values()])
    ys, xs = np.nonzero(common)
    cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (0, 0)
    fig = Figure(figsize=(12, 4.4))
    axes = fig.subplots(1, 3)
    for ax, row in zip(axes, rows):
        g = to_gray_float(results[row["interpolation"]].data)
        y0, x0 = max(0, cy - zoom // 2), max(0, cx - zoom // 2)
        ax.imshow(g[y0:y0 + zoom, x0:x0 + zoom], cmap="gray", interpolation="nearest")
        ax.set_title(f'{row["interpolation"]}\nsharp {row["sharpness"]:.0f}  rt-PSNR {row["roundtrip_psnr_db"]:.1f} dB',
                     fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Interpolation comparison ({zoom}x{zoom} px crop, each pixel enlarged)", fontsize=10)
    return save_figure(fig, path)
