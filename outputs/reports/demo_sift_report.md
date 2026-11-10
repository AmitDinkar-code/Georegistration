# Registration report - demo_sift

**Outcome:** ACCEPTED  |  method `sift` / model `affine`

## Attempts

| detector | models | status | keypoints (src/ref) | matches | detail |
|---|---|---|---|---|---|
| sift | affine,homography | accepted | 4403/8001 | 1515 | affine, 1161/1212 inliers |

## Relative alignment (self-consistency)

| metric | value |
|---|---|
| keypoints_source | 4403 |
| keypoints_reference | 8001 |
| matches_accepted | 1515 |
| ransac_inliers_fit_set | 1161 |
| inlier_ratio | 0.958 |
| inlier_rmse_px | 0.636 |
| inlier_median_px | 0.402 |
| model_dof | 6 |
| ransac_iterations_needed_est | 3.271 |

## Independent checkpoints (withheld matches)

- checkpoints: 303, consistent within 2x threshold: 294 (0.97)
- median error: 0.387 px; RMSE over consistent: 0.615 px

## Registration consistency (bootstrap)

- 30 refits on 70 % subsets: probe-point spread RMS 0.024 px, max 0.071 px

## Ground-truth error (synthetic test only)

- pixel error (reference px): RMSE 0.161, median 0.158, p95 0.201, max 0.207
- map error: RMSE 1.610, max 2.071 (CRS units (metres for projected CRS))
- for comparison, simply stamping the CRS on the unwarped image: RMSE 321.804 px = 3218.0 map units

## Image similarity to the reference (valid area)

| | NCC | gradient-NCC |
|---|---|---|
| before (stamped, no warp) | 0.039 | -0.007 |
| after (registered) | 0.981 | 0.952 |

## Interpolation comparison

| method | NCC | sharpness | values preserved | round-trip PSNR dB | ms |
|---|---|---|---|---|---|
| nearest | 0.9701 | 568.6 | 1.000 | 55.94 | 11.5 |
| bilinear | 0.9794 | 102.4 | 0.675 | 38.89 | 14.2 |
| cubic | 0.9836 | 315.4 | 0.657 | 46.96 | 11.5 |

## Output

- GeoTIFF: `outputs/geotiff/demo_sift_registered.tif`
- footprint covered: 0.558 of reference grid
- interpolation: bilinear
- GCPs written: 1455 (`outputs/reports/demo_sift_gcps.csv`)

## What these numbers do and do not show

- Reprojection error (inlier RMSE) measures how well the matches agree with the fitted model - it is computed on the very points that voted for the model, so it is optimistic and says nothing about errors shared by ALL matches (a systematic offset, a wrong CRS, terrain relief).
- Held-out checkpoint error is more honest but still relative: the checkpoints come from the same matcher and the same two images, so they cannot reveal an error that is common to both.
- Absolute geographic accuracy requires independent surveyed points (GNSS / cadastral corners) or a reference whose own absolute accuracy is known. The output can be no more accurate than the reference grid it is registered to.
- A planar transform (affine/homography) cannot remove terrain-relief displacement or sensor geometry; registration is not orthorectification.

## Timing (s)

load+prepare: 0.04, register: 0.20, warp: 0.01, bootstrap: 0.01, total: 1.65
