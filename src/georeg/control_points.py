"""Phase 4b: from pixel correspondences to Ground Control Points.

A feature match alone is two PIXEL coordinates. It becomes a GCP only after the
REFERENCE pixel is converted to a MAP coordinate with the reference geotransform:

    map = transform * (col + 0.5, row + 0.5)         (rasterio/GDAL, pixel-corner origin)

OpenCV puts the centre of pixel (0,0) at (0,0), so every OpenCV coordinate needs
+0.5 before it meets an Affine geotransform. Skipping this is a silent half-pixel
shift in every output coordinate.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from affine import Affine

from .io_utils import ensure_dir


def pixel_to_map(transform: Affine, x, y):
    """OpenCV pixel coords (centre-at-integer) -> map coords of the pixel centre."""
    c, r = np.asarray(x, float) + 0.5, np.asarray(y, float) + 0.5
    t = transform   # explicit arithmetic: independent of affine's operator-overload changes
    return t.a * c + t.b * r + t.c, t.d * c + t.e * r + t.f


def map_to_pixel(transform: Affine, X, Y):
    t = ~transform
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    return t.a * X + t.b * Y + t.c - 0.5, t.d * X + t.e * Y + t.f - 0.5


def composite_pixel_to_map(ref_transform: Affine, T: np.ndarray) -> np.ndarray:
    """3x3 matrix taking SOURCE OpenCV pixels straight to map coordinates:
    A_ref . shift(+0.5) . T. For a homography T the result is projective."""
    A = np.array([[ref_transform.a, ref_transform.b, ref_transform.c],
                  [ref_transform.d, ref_transform.e, ref_transform.f], [0, 0, 1]], float)
    half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], float)
    return A @ half @ T


@dataclass(frozen=True)
class GCP:
    id: int
    src_x: float          # OpenCV px in the unreferenced image
    src_y: float
    ref_x: float          # OpenCV px in the reference image
    ref_y: float
    map_x: float          # reference CRS coordinates
    map_y: float
    residual_px: float    # forward reprojection error under the final transform
    role: str             # inlier | outlier | checkpoint_ok | checkpoint_bad


def build_gcps(pts_src, pts_ref, errors, roles, ref_transform: Affine, only=("inlier", "checkpoint_ok")) -> list[GCP]:
    mx, my = pixel_to_map(ref_transform, pts_ref[:, 0], pts_ref[:, 1])
    out = []
    for i, (s, r, e, role) in enumerate(zip(pts_src, pts_ref, errors, roles)):
        if only is None or role in only:
            out.append(GCP(i, float(s[0]), float(s[1]), float(r[0]), float(r[1]),
                           float(mx[i]), float(my[i]), float(e), str(role)))
    return out


def gcps_to_csv(gcps: list[GCP], path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(gcps[0]).keys()) if gcps else ["id"])
        w.writeheader()
        for g in gcps:
            w.writerow(asdict(g))
    return p


def rasterio_gcps(gcps: list[GCP]):
    """rasterio GCP objects (row/col are pixel-corner based, hence +0.5)."""
    from rasterio.control import GroundControlPoint
    return [GroundControlPoint(row=g.src_y + 0.5, col=g.src_x + 0.5, x=g.map_x, y=g.map_y, id=str(g.id))
            for g in gcps]


def gdal_commands(gcps: list[GCP], src: str, out: str, epsg: str = "EPSG:32643", max_gcps: int = 40) -> str:
    """Equivalent GDAL command lines (for people who want to reproduce the result
    with gdal_translate/gdalwarp). Uses the best `max_gcps` by residual. Windows: run
    from an OSGeo4W / conda shell; quote paths."""
    best = sorted(gcps, key=lambda g: g.residual_px)[:max_gcps]
    args = " ".join(f"-gcp {g.src_x + 0.5:.3f} {g.src_y + 0.5:.3f} {g.map_x:.3f} {g.map_y:.3f}" for g in best)
    return (f'gdal_translate -of GTiff {args} "{src}" "gcp_tmp.tif"\n'
            f'gdalwarp -order 1 -r bilinear -t_srs {epsg} "gcp_tmp.tif" "{out}"\n'
            "# -order 1 = affine; use -order 2 (polynomial) or -tps only with many well-spread GCPs.")
