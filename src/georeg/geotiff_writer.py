"""GeoTIFF output.

Terminology used throughout this package (they are NOT interchangeable):

* assigning spatial metadata - stamping a CRS/transform on pixels that were
  never moved. Only correct if the pixels already sit on that grid.
* image registration        - estimating the geometric transform between two
  images (this package's core job).
* georeferencing            - using a registration (or GCPs) to resample the
  image onto a known map grid so each pixel has a map coordinate.
* orthorectification        - removing terrain-relief and sensor-geometry
  displacement using a DEM + sensor model. NOT done here (see docs/THEORY.md).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from .io_utils import ensure_dir

log = logging.getLogger("georeg")


def choose_nodata(dtype, preferred=None):
    """Pick a nodata value that the dtype can hold."""
    dt = np.dtype(dtype)
    if preferred is not None:
        if dt.kind == "f" or (np.iinfo(dt).min <= preferred <= np.iinfo(dt).max):
            return preferred
    if dt.kind == "f":
        return float("nan")
    return 0


def write_geotiff(path, data: np.ndarray, transform: Affine | None, crs,
                  valid: np.ndarray | None = None, nodata=None, tags: dict | None = None,
                  descriptions: list[str] | None = None, compress: str = "deflate") -> dict:
    """Write (bands, H, W) data. Pixels where valid is False are set to nodata AND
    flagged in an internal GDAL mask band, so consumers that ignore nodata still
    see the footprint. Radiometric dtype is preserved (no rescaling)."""
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    count, H, W = data.shape
    nodata = choose_nodata(data.dtype, nodata)
    out = data.copy()
    n_collide = 0
    if valid is not None:
        valid = np.asarray(valid, bool)
        if not (isinstance(nodata, float) and np.isnan(nodata)):
            n_collide = int(np.count_nonzero((out[:, valid] == nodata).any(axis=0)))
        out[:, ~valid] = nodata

    p = Path(path)
    ensure_dir(p.parent)
    profile = dict(driver="GTiff", height=H, width=W, count=count, dtype=out.dtype.name,
                   transform=transform if transform is not None else Affine.identity(),
                   compress=compress, nodata=nodata)
    if crs is not None:
        profile["crs"] = crs
    mask_written = False
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(str(p), "w", **profile) as dst:
            dst.write(out)
            if valid is not None:
                try:
                    dst.write_mask((valid.astype(np.uint8) * 255))
                    mask_written = True
                except Exception as exc:  # driver/build without internal-mask support
                    log.warning("internal mask band not written (%s); nodata is still set", exc)
            all_tags = dict(tags or {})
            if n_collide:
                all_tags["VALID_PIXELS_EQUAL_TO_NODATA"] = str(n_collide)
            if all_tags:
                dst.update_tags(**{k: str(v) for k, v in all_tags.items()})
            if descriptions:
                for i, d in enumerate(descriptions[:count], start=1):
                    dst.set_band_description(i, d)
    return dict(path=str(p), shape=(count, H, W), dtype=out.dtype.name, nodata=nodata,
                mask_written=mask_written, valid_pixels_equal_to_nodata=n_collide)
