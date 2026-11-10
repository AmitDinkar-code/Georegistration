# Registration report - demo_orb

**Outcome:** ACCEPTED  |  method `orb` / model `affine`

## Attempts

| detector | models | status | keypoints (src/ref) | matches | detail |
|---|---|---|---|---|---|
| orb | affine,homography | accepted | 8000/8000 | 2178 | affine, 1536/1742 inliers |

## Relative alignment (self-consistency)

| metric | value |
|---|---|
| keypoints_source | 8000 |
| keypoints_reference | 8000 |
| matches_accepted | 2178 |
| ransac_inliers_fit_set | 1536 |
| inlier_ratio | 0.882 |
| inlier_rmse_px | 1.414 |
| inlier_median_px | 1.128 |
| model_dof | 6 |
| ransac_iterations_needed_est | 5.971 |

## Independent checkpoints (withheld matches)

- checkpoints: 436, consistent within 2x threshold: 427 (0.98)
- median error: 1.224 px; RMSE over consistent: 1.719 px

## Registration consistency (bootstrap)

- 30 refits on 70 % subsets: probe-point spread RMS 0.172 px, max 0.753 px

## Ground-truth error (synthetic test only)

- pixel error (reference px): RMSE 0.116, median 0.098, p95 0.190, max 0.210
- map error: RMSE 1.164, max 2.103 (CRS units (metres for projected CRS))
- for comparison, simply stamping the CRS on the unwarped image: RMSE 321.804 px = 3218.0 map units

## Image similarity to the reference (valid area)

| | NCC | gradient-NCC |
|---|---|---|
| before (stamped, no warp) | 0.039 | -0.007 |
| after (registered) | 0.981 | 0.955 |

## Interpolation comparison

| method | NCC | sharpness | values preserved | round-trip PSNR dB | ms |
|---|---|---|---|---|---|
| nearest | 0.9707 | 574.5 | 1.000 | 57.00 | 7.2 |
| bilinear | 0.9800 | 102.8 | 0.676 | 38.91 | 10.1 |
| cubic | 0.9843 | 316.0 | 0.659 | 46.96 | 8.4 |

## Output

- GeoTIFF: `outputs/geotiff/demo_orb_registered.tif`
- footprint covered: 0.557 of reference grid
- interpolation: bilinear
- GCPs written: 1963 (`outputs/reports/demo_orb_gcps.csv`)

## What these numbers do and do not show

- Reprojection error (inlier RMSE) measures how well the matches agree with the fitted model - it is computed on the very points that voted for the model, so it is optimistic and says nothing about errors shared by ALL matches (a systematic offset, a wrong CRS, terrain relief).
- Held-out checkpoint error is more honest but still relative: the checkpoints come from the same matcher and the same two images, so they cannot reveal an error that is common to both.
- Absolute geographic accuracy requires independent surveyed points (GNSS / cadastral corners) or a reference whose own absolute accuracy is known. The output can be no more accurate than the reference grid it is registered to.
- A planar transform (affine/homography) cannot remove terrain-relief displacement or sensor geometry; registration is not orthorectification.

## Timing (s)

load+prepare: 0.04, register: 0.14, warp: 0.01, bootstrap: 0.01, total: 1.57
