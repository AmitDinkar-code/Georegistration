# Dataset documentation (Phase 1)

## Inputs the pipeline accepts
| | reference | source (to be registered) |
|---|---|---|
| format | GeoTIFF **with CRS + geotransform** (required) | GeoTIFF without georeferencing, or PNG/JPG/BMP |
| bands | any; features use luminance of bands 1-3 (assumed R,G,B) else band 1 | same |
| dtype | any (8/16-bit, float) - percentile-stretched to 8-bit *only for feature extraction* | same; **output keeps this dtype** |
| validity | nodata / internal mask / alpha → `valid` mask | same (nodata=0 recommended for rotated scenes) |

`dataset_loader.describe_raster()` reports size, bands, dtype, CRS, pixel size, bounds, valid fraction, band statistics.
`compare_inputs()` flags resolution ratio (pixel size ratio if both are georeferenced), band-count mismatch and size ratio > 4.

## Normalisation
Percentile stretch (2-98 %, computed on **valid pixels only**) to 8 bit, optional CLAHE (`PreprocessCfg.clahe`). Reason: SIFT/ORB need 8-bit
input, and 12/16-bit or low-contrast imagery would otherwise be crushed or saturated. A global gain/offset between acquisitions is absorbed by the
stretch (and SIFT's gradient-based descriptors) - see the brightness rows of the robustness study.
Detection masks are eroded by 3 px so nodata borders do not create fake corners.

## Synthetic data (`synthetic_transform.py`)
* `procedural_scene(size, seed)`: deterministic aerial-like RGB (multi-scale terrain texture, field patches, roads, buildings, tanks; no zero pixels so 0 can be nodata).
* `make_reference_geotiff`: writes it as a UTM 43N GeoTIFF, 10 m pixels (origin 480000, 2570000).
* `make_case(...)`: derives the "unreferenced" image through a known matrix `M` (reference px → source px) built from rotation, scale (source px per reference px), shear, translation, perspective,
  footprint crop, then degrades it: gain/bias, Gaussian noise, opaque occlusion blobs. Downscaling is anti-aliased. **`T_true = inv(M)` is returned as exact ground truth.**
* Presets: `rotation, translation, scale, affine, perspective, demo`.
* Everything is seeded; the same call gives the same bytes.

## Resolution differences
The source can be coarser or finer than the reference (`scale`); the pipeline never assumes equal sizes. Registration estimates the scale as part of `T`, and the output is always resampled to the
**reference** grid (so a coarser source is upsampled, a finer one downsampled by the interpolator - which, for strong decimation, aliases; pre-filter first if that matters).

## Real data (not included)
No real pair is bundled. Good first pairs: Sentinel-2 L2A RGB (B04/B03/B02) vs a Landsat or Sentinel-2 scene from another date over a low-relief area (same season), or an aerial orthophoto vs a
satellite tile. Put the reference GeoTIFF anywhere and run `python -m georeg register ...`. Prefer low relief and similar acquisition dates; see README for what to expect otherwise.
