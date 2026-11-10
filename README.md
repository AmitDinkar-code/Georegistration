# georeg - automated image georegistration

Takes an **unreferenced** optical image and a **georeferenced** reference image, finds the geometric
transform between them with feature matching + RANSAC, resamples the image onto the reference grid and
writes a **GeoTIFF** - then measures the accuracy independently instead of trusting its own fit.

```
source.tif (no CRS) ─┐                                             ┌─ registered GeoTIFF (reference CRS + grid, nodata + mask)
                     ├─ SIFT/ORB → match → RANSAC → gates → warp ──┼─ GCPs (map coordinates) + GDAL commands
reference.tif (CRS) ─┘        └─ fallback chain ─┘                 └─ report (json/md) + overlays, checkerboard, error maps
```

**Honest scope:** this is *image-to-image registration onto a reference grid*. It is **not** orthorectification
(no DEM, no sensor model). Output accuracy is bounded by the reference's own accuracy. See `docs/THEORY.md` §8.

---------------------------------------------------------------------------------------------------

## Quick start

### Windows (PowerShell) - recommended path, no GDAL install needed
```powershell
cd Georegistration
Set-ExecutionPolicy -Scope Process Bypass      # only if scripts are blocked
.\setup.ps1                                    # creates .venv, installs everything from pip wheels
.\run_tests.ps1                                # 54 tests
.\.venv\Scripts\python.exe -m georeg demo      # end-to-end synthetic demo -> outputs\
```
Manual equivalent: `py -3.12 -m venv .venv`, `.venv\Scripts\python -m pip install -r requirements.txt`, `.venv\Scripts\python -m pip install -e . --no-deps`.

### macOS / Linux
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest -q
.venv/bin/python -m georeg demo
```

### Your own imagery
```
python -m georeg register --source my_image.tif --reference reference.tif --out outputs
#   --detector auto|sift|orb   --model auto|homography|affine|similarity   --matcher bf|flann
#   --interp nearest|bilinear|cubic   --threshold 3.0 (RANSAC px)
python -m georeg verify outputs/geotiff/<name>_registered.tif reference.tif
```
The source may be a GeoTIFF without georeferencing, or PNG/JPG. The reference **must** be a GeoTIFF with CRS.
Other commands: `make-synthetic`, `demo`, `robustness`, `verify` (`python -m georeg --help`).

---------------------------------------------------------------------------------------------------

## Windows notes (what was designed for, and what was NOT tested)

* **GDAL is optional.** `rasterio`'s pip wheels bundle GDAL, so reading/writing GeoTIFFs and the whole pipeline work with plain `pip`.
  The `osgeo.gdal` Python bindings are used only for an *extra* independent check in `verify_metadata`; the test is skipped if absent.
* To get the bindings on Windows use **conda-forge** (`conda create -n georeg -c conda-forge python=3.12 gdal rasterio opencv scipy matplotlib pytest`)
  or a prebuilt GDAL wheel matching your Python. Do not mix pip-`rasterio` and a separately installed GDAL in one environment
  (they ship different `PROJ`/`GDAL_DATA` and can fight; if you see "PROJ: proj_create_from_database", unset `PROJ_LIB`/`PROJ_DATA`).
* Paths: everything uses `pathlib`; images are read via `numpy.fromfile` + `imdecode`, so **non-ASCII user folders work**
  (plain `cv2.imread` silently returns `None` there). Text files are written as UTF-8; console output is forced to UTF-8.
* Plots use matplotlib's `Figure` API directly (no Tk/display backend), so headless runs work.
* Tests use `tmp_path` and close all files, avoiding Windows file-lock errors.
* **Not verified:** I developed and ran everything on macOS (Python 3.12, OpenCV 5.0.0, rasterio 1.5.2/GDAL 3.12). The
  Windows scripts (`setup.ps1`, `run_tests.ps1`, `run_demo.ps1`) have **not been executed on Windows**. If `setup.ps1` trips, the manual commands above are the fallback.
  macOS quirk, irrelevant on Windows: editable-install `.pth` files may be flagged hidden - `chflags nohidden .venv/lib/*/site-packages/*.pth`.

---------------------------------------------------------------------------------------------------

## Pipeline

| stage | module | what it does |
|---|---|---|
| 1 data | `dataset_loader`, `synthetic_transform` | read GeoTIFF/PNG, valid mask, percentile-stretch to 8-bit; synthetic scene + exact-truth test transforms |
| 2 features | `feature_detector` | SIFT (primary) / ORB: location, scale, orientation, descriptor |
| 3 matching | `feature_matcher` | BF or FLANN; Lowe ratio (SIFT), Hamming (ORB); optional mutual check; uniqueness |
| 4 estimation | `transformation_estimator`, `control_points` | RANSAC similarity / affine / homography, parsimony model selection, held-out checkpoints, GCPs in map coordinates |
| 5 warp | `image_warping` | backward-mapped warp onto the reference grid, nearest/bilinear/cubic, valid mask |
| 6 georeference | `geotiff_writer`, `verify_metadata` | reference CRS + transform, nodata + internal mask, rasterio and (optional) GDAL verification |
| 7 validation | `registration_validation` | checkpoint error, ground-truth error, bootstrap consistency, overlays, checkerboard, difference, error maps, report |
| 8 robustness | `robustness`, `pipeline` | quality gates, fallback chain, SIFT-vs-ORB study, logging |

**Fallback chain** (`PipelineCfg.attempts`): SIFT {affine, homography} → ORB {affine, homography} → phase-correlation (translation-only, flagged `LOW_CONFIDENCE`).
A result is accepted only if it passes the **quality gates**: inliers ≥ 15, inlier ratio ≥ 0.15, inlier RMSE ≤ 2 px, inlier convex-hull coverage ≥ 5 % of the source,
no reflection, sane scale, convex warped footprint, bounded perspective. If everything fails, **no GeoTIFF is written** and the report says why.

Output layout: `outputs/{keypoints,matches,registered,geotiff,reports}/` (all produced by `demo`).

---------------------------------------------------------------------------------------------------

## Results (measured on this machine, synthetic data - read the caveats)

Synthetic reference: 1024×1024, 10 m pixels, EPSG:32643. Source derived by a known transform (`demo`: 20° rotation, 0.8× resolution, shear, 75 % footprint,
gain 1.1, bias −8, noise σ=4). Numbers copied from `outputs/reports/*_report.md`.

| run | model chosen | matches / RANSAC inliers (fit set) | inlier RMSE | held-out checkpoints (median) | **ground-truth RMSE** | naive "stamp the CRS" error |
|---|---|---|---|---|---|---|
| demo, SIFT | affine | 1515 / 1161 (96 %) | 0.64 px | 0.39 px (303 pts) | **0.16 px = 1.6 m** | 322 px = 3,218 m |
| demo, ORB | affine | 2178 / 1536 (88 %) | 1.41 px | 1.22 px (436 pts) | **0.12 px = 1.2 m** | 322 px |
| perspective, SIFT | **homography** | 1998 / 1561 (98 %) | 0.43 px | 0.21 px (400 pts) | **0.07 px = 0.7 m** | 242 px |

* Image similarity to the reference (NCC over the valid area): 0.04 before → 0.98 after. Bootstrap refits (70 % subsets) move probe points by ~0.02 px RMS (SIFT) / 0.17 px (ORB).
* The SIFT-vs-ORB ground-truth difference (0.16 vs 0.12 px) is far below what 1 trial can resolve; do not read it as ORB being more accurate.
  The inlier RMSE even ranks them the other way - a concrete reason not to use reprojection error as the accuracy measure.
* The GeoTIFF's CRS, transform, shape, bounds, resolution and nodata equal the reference's; an internal mask band is present (`verify` all PASS).

### Robustness study (`python -m georeg robustness`, 9 factors × ≥4 levels × 3 trials × 2 detectors = 264 runs)

| | SIFT | ORB |
|---|---|---|
| correct registrations (GT RMSE ≤ 3 px) | 119 / 132 | 114 / 132 |
| rejected by the gates (safe failure) | 13 | 18 |
| **accepted but wrong (false accept)** | **0** | **0** |
| median ground-truth RMSE of correct runs | 0.12 px | 0.37 px |
| mean time per run | 0.06 s | 0.06 s |

Held at 100 % for both detectors: rotation 0-180°, scale 0.3×-3×, brightness gain 0.3-1.4, noise σ up to 100, contrast/blur down to 8 % contrast, perspective up to 8e-4, 60 % scene change.
Where they broke: **scene change ≥ 80 %** (both 0 %), **source resolution ≤ 0.12× reference** (SIFT 33 % at 0.12×, 0 % at 0.08×; ORB already 0 % at 0.2×),
**occlusion 85 %** (SIFT 33 %, ORB 0 %). Full table: `outputs/reports/robustness/robustness_summary.md`.

### Read this before quoting any number
1. **Everything above is synthetic and the easy case**: the source is a resampled copy of the *same rendering* as the reference (same sensor, same date, same radiometry
   apart from the injected degradations). Real imagery adds seasonal/illumination change, different sensors and bands, relief and parallax. Expect **lower success rates
   and larger errors on real data**; none of these figures transfer to it.
2. "0 false accepts in 264 runs" describes these gates on this synthetic distribution (3 trials per cell). It is not a guarantee; on real imagery tune the gates and look at the overlays.
3. Ground-truth error here is *registration* error against a known transform. **Absolute geographic accuracy was not measured** and cannot be from synthetic data.
4. No real satellite/aerial pair was run. Phase 1 asked for real imagery too; the code accepts it (any GeoTIFF reference), but I did not have a pair, and I did not download one without your say-so.

---------------------------------------------------------------------------------------------------

## Design decisions worth knowing

* **Direction & conventions** are fixed once (`M`: ref→src, `T`: src→ref = what is estimated and what `warpPerspective` consumes) and unit-tested. OpenCV pixel centres sit at integers; GDAL/rasterio geotransforms reference pixel *corners* → `+0.5` in `control_points.pixel_to_map`. A test pins this.
* **Independent checkpoints:** 20 % of matches are withheld *before* RANSAC and never vote. Reported with the caveat that they come from the same matcher.
* **Parsimony:** the lowest-DoF model (similarity < affine < homography) whose inlier count is within 2 % of the best wins - so an affine scene is not over-fit with 8 parameters, and a perspective scene correctly selects the homography (tested).
* **Never silently ship a bad result:** gates + fallback chain + failure report; tested with a blank image and an unrelated scene.
* **Radiometry preserved:** dtype unchanged, no rescaling; nearest-neighbour keeps exact source values. Feature extraction works on a *separate* 8-bit stretched copy.
* **GeoTIFF is tagged** `ORTHORECTIFIED=NO`, method, model, inliers, RMSE, interpolation, low-confidence flag.

## Tests
`pytest` → **54 passed, 1 skipped** (the skip is the optional GDAL-bindings cross-check, `osgeo` is not installed here).
Coverage: detector shapes/validation, matcher precision vs ground truth (SIFT/ORB × BF/FLANN), ratio-test monotonicity, RANSAC recovery with 40 % outliers, DoF table,
RANSAC iteration formula, model selection, plausibility gates, holdout split, warp direction/exactness/dtype/nodata, interpolation ordering, half-pixel convention,
composite matrix, GeoTIFF round-trip, end-to-end accuracy vs ground truth, "better than naive stamping", grid/CRS equality, rejection of blank and unrelated inputs.

## Extending to real imagery - what to expect
* Same sensor and season: SIFT + homography/affine should work; check `*_overlay_after.png` and the checkerboard by eye.
* Different sensors/bands (optical vs SAR, RGB vs NIR) or big seasonal change: raw-intensity SIFT/ORB often fails; try `preprocess.clahe=True`, matching gradient/edge images, or a learned matcher. The gates will reject rather than guess.
* Large area / relief / off-nadir: a single planar transform leaves residual displacement that is invisible to inlier RMSE. Use tiles + local transforms, or proper orthorectification (DEM + RPC/sensor model, e.g. `gdalwarp -rpc -to RPC_DEM=...`). Details in `docs/THEORY.md` §8.

## Files
`docs/DATASET.md` (Phase 1 documentation) · `docs/THEORY.md` (the maths and engineering reasoning for every stage) · `notebooks/registration_experiments.ipynb`.
