"""Phase 1: dataset loading, inspection and preparation for feature extraction."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.errors import NotGeoreferencedWarning

from .config import PreprocessCfg
from .errors import InputError
from .io_utils import imread_unicode

_GEOTIFF_EXT = {".tif", ".tiff"}


@dataclass
class Raster:
    data: np.ndarray            # (bands, H, W), native dtype
    valid: np.ndarray           # (H, W) bool, False on nodata / alpha=0
    transform: Affine | None    # None when the file carries no georeferencing
    crs: object | None
    nodata: float | None
    path: Path | None = None

    @property
    def count(self) -> int:
        return self.data.shape[0]

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def is_georeferenced(self) -> bool:
        return self.crs is not None and self.transform is not None


def load_raster(path) -> Raster:
    """Read a GeoTIFF (with georeferencing if present) or a plain image (PNG/JPG/...)."""
    p = Path(path)
    if not p.is_file():
        raise InputError(f"file not found: {p}")
    if p.suffix.lower() in _GEOTIFF_EXT:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(str(p)) as ds:
                data = ds.read()
                valid = ds.dataset_mask() > 0
                crs = ds.crs
                tr = ds.transform
                nodata = ds.nodata
        if crs is None or tr == Affine.identity():
            tr = None if crs is None else tr
        return Raster(data, valid, tr, crs, nodata, p)

    img = imread_unicode(p, cv2.IMREAD_UNCHANGED)
    valid = np.ones(img.shape[:2], bool)
    if img.ndim == 2:
        data = img[None]
    else:
        if img.shape[2] == 4:
            valid = img[..., 3] > 0
            img = img[..., :3]
        data = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))  # BGR -> RGB
    return Raster(data, valid, None, None, None, p)


def to_gray_float(data: np.ndarray) -> np.ndarray:
    """Collapse (bands,H,W) to one float32 plane. >=3 bands: luminance of the
    first three (assumed R,G,B); otherwise the first band."""
    d = data.astype(np.float32)
    if d.shape[0] >= 3:
        return 0.299 * d[0] + 0.587 * d[1] + 0.114 * d[2]
    return d[0]


def prepare_gray(data: np.ndarray, valid: np.ndarray | None, pre: PreprocessCfg) -> np.ndarray:
    """Percentile-stretch to 8 bit (stats from valid pixels only), optional CLAHE.
    Works for any input dtype, so 12/16-bit imagery is handled correctly."""
    g = to_gray_float(data)
    v = valid if valid is not None and valid.any() else np.ones(g.shape, bool)
    lo, hi = np.percentile(g[v], [pre.lower_pct, pre.upper_pct])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    out = np.clip((g - lo) / (hi - lo), 0, 1) * 255.0
    out = out.astype(np.uint8)
    if pre.clahe:
        out = cv2.createCLAHE(clipLimit=pre.clahe_clip,
                              tileGridSize=(pre.clahe_tile, pre.clahe_tile)).apply(out)
    if valid is not None:
        out = np.where(valid, out, 0).astype(np.uint8)
    return out


def detection_mask(valid: np.ndarray | None, erode_px: int = 3) -> np.ndarray | None:
    """uint8 mask for the detectors; eroded so border/nodata edges give no keypoints."""
    if valid is None or valid.all():
        return None
    m = valid.astype(np.uint8) * 255
    if erode_px > 0:
        k = 2 * erode_px + 1
        m = cv2.erode(m, np.ones((k, k), np.uint8))
    return m


def describe_raster(r: Raster) -> dict:
    d = r.data
    info = dict(
        path=str(r.path) if r.path else None,
        width=r.width, height=r.height, bands=r.count, dtype=str(r.dtype),
        georeferenced=r.is_georeferenced,
        crs=r.crs.to_string() if r.crs is not None else None,
        valid_fraction=float(r.valid.mean()),
        nodata=r.nodata,
    )
    if r.transform is not None:
        t = r.transform
        info["pixel_size_x"] = abs(t.a)
        info["pixel_size_y"] = abs(t.e)
        info["rotation_terms_bd"] = (t.b, t.d)
        info["bounds"] = (t.c, t.f + t.e * r.height, t.c + t.a * r.width, t.f)
    info["band_stats"] = [dict(min=float(d[i][r.valid].min()) if r.valid.any() else None,
                               max=float(d[i][r.valid].max()) if r.valid.any() else None,
                               mean=float(d[i][r.valid].mean()) if r.valid.any() else None)
                          for i in range(min(r.count, 4))]
    return info


def compare_inputs(src: Raster, ref: Raster) -> dict:
    """Resolution / geometry differences that matter for feature matching."""
    out = dict(size_ratio_w=src.width / ref.width, size_ratio_h=src.height / ref.height,
               bands_source=src.count, bands_reference=ref.count,
               dtype_source=str(src.dtype), dtype_reference=str(ref.dtype))
    if src.transform is not None and ref.transform is not None:
        out["gsd_ratio_source_over_reference"] = abs(src.transform.a) / abs(ref.transform.a)
    notes = []
    if src.count != ref.count:
        notes.append("band count differs: features use a luminance plane, so spectral "
                     "mismatch (e.g. NIR vs RGB) can still break matching")
    if max(out["size_ratio_w"], 1 / out["size_ratio_w"]) > 4:
        notes.append("pixel-count ratio > 4: SIFT scale invariance helps, ORB may struggle")
    out["notes"] = notes
    return out
