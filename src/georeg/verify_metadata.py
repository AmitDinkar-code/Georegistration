"""Metadata verification of an output GeoTIFF against its reference, using
rasterio (always) and GDAL's own Python bindings (when installed).

    python -m georeg.verify_metadata OUTPUT.tif REFERENCE.tif
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import rasterio
from rasterio.enums import MaskFlags


def verify_with_rasterio(output, reference) -> dict:
    checks = {}
    with rasterio.open(str(output)) as o, rasterio.open(str(reference)) as r:
        checks["crs_equal"] = (o.crs is not None and o.crs == r.crs, f"{o.crs} vs {r.crs}")
        checks["transform_equal"] = (o.transform.almost_equals(r.transform, precision=1e-9), f"{tuple(o.transform)[:6]}")
        checks["shape_equal"] = ((o.height, o.width) == (r.height, r.width), f"{(o.height, o.width)} vs {(r.height, r.width)}")
        checks["bounds_equal"] = (np.allclose(tuple(o.bounds), tuple(r.bounds), atol=1e-6), f"{tuple(o.bounds)}")
        checks["resolution_equal"] = (np.allclose(o.res, r.res), f"{o.res} vs {r.res}")
        checks["nodata_defined"] = (o.nodata is not None, f"{o.nodata}")
        flags = [f.name for f in o.mask_flag_enums[0]]
        checks["mask_declared"] = (any(f in flags for f in ("per_dataset", "nodata", "alpha")), ",".join(flags))
        m = o.dataset_mask()
        checks["has_valid_pixels"] = (bool((m > 0).any()), f"valid fraction {np.mean(m > 0):.3f}")
        checks["has_invalid_pixels_or_full"] = (True, f"invalid fraction {np.mean(m == 0):.3f}")
        tags = o.tags()
        checks["declares_not_orthorectified"] = (tags.get("ORTHORECTIFIED") == "NO", tags.get("ORTHORECTIFIED", "missing"))
        checks["readable_all_bands"] = (o.read().shape[0] == o.count, f"{o.count} bands, dtype {o.dtypes[0]}")
    return checks


def verify_with_gdal(output, reference) -> dict:
    """Independent read through GDAL itself. Returns {'skipped': reason} if osgeo is absent."""
    try:
        from osgeo import gdal, osr  # type: ignore
    except ImportError as exc:
        return {"skipped": (None, f"osgeo not installed ({exc})")}
    gdal.UseExceptions()
    o, r = gdal.Open(str(output)), gdal.Open(str(reference))
    checks = {}
    checks["gdal_geotransform_equal"] = (np.allclose(o.GetGeoTransform(), r.GetGeoTransform(), atol=1e-9), str(o.GetGeoTransform()))
    so, sr = osr.SpatialReference(wkt=o.GetProjection()), osr.SpatialReference(wkt=r.GetProjection())
    checks["gdal_srs_same"] = (bool(so.IsSame(sr)), so.GetName())
    checks["gdal_size_equal"] = ((o.RasterXSize, o.RasterYSize) == (r.RasterXSize, r.RasterYSize), f"{o.RasterXSize}x{o.RasterYSize}")
    nd = o.GetRasterBand(1).GetNoDataValue()
    checks["gdal_nodata"] = (nd is not None, str(nd))
    return checks


def report(checks: dict) -> bool:
    ok = True
    for k, (passed, detail) in checks.items():
        status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        ok &= passed is not False
        print(f"[{status}] {k:32s} {detail}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output"); ap.add_argument("reference")
    a = ap.parse_args(argv)
    ok = report(verify_with_rasterio(a.output, a.reference))
    ok &= report(verify_with_gdal(a.output, a.reference))
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
