# Registration report - perspective_sift

**Outcome:** ACCEPTED  |  method `sift` / model `homography`

## Attempts

| detector | models | status | keypoints (src/ref) | matches | detail |
|---|---|---|---|---|---|
| sift | affine,homography | accepted | 3713/8001 | 1998 | homography, 1561/1598 inliers |

## Relative alignment (self-consistency)

| metric | value |
|---|---|
| keypoints_source | 3713 |
| keypoints_reference | 8001 |
| matches_accepted | 1998 |
| ransac_inliers_fit_set | 1561 |
| inlier_ratio | 0.977 |
| inlier_rmse_px | 0.428 |
| inlier_median_px | 0.228 |
| model_dof | 8 |
| ransac_iterations_needed_est | 2.861 |

## Independent checkpoints (withheld matches)

- checkpoints: 400, consistent within 2x threshold: 396 (0.99)
- median error: 0.211 px; RMSE over consistent: 0.435 px

## Registration consistency (bootstrap)

- 30 refits on 70 % subsets: probe-point spread RMS 0.018 px, max 0.058 px

## Ground-truth error (synthetic test only)

- pixel error (reference px): RMSE 0.067, median 0.057, p95 0.109, max 0.135
- map error: RMSE 0.675, max 1.351 (CRS units (metres for projected CRS))
- for comparison, simply stamping the CRS on the unwarped image: RMSE 242.205 px = 2422.0 map units

## Image similarity to the reference (valid area)

| | NCC | gradient-NCC |
|---|---|---|
| before (stamped, no warp) | 0.016 | 0.010 |
| after (registered) | 0.988 | 0.972 |

## Interpolation comparison

| method | NCC | sharpness | values preserved | round-trip PSNR dB | ms |
|---|---|---|---|---|---|
| nearest | 0.9819 | 198.7 | 1.000 | 50.31 | 11.8 |
| bilinear | 0.9872 | 68.6 | 0.876 | 43.23 | 13.8 |
| cubic | 0.9913 | 206.0 | 0.867 | 52.63 | 12.8 |

## Output

- GeoTIFF: `outputs/geotiff/perspective_sift_registered.tif`
- footprint covered: 0.562 of reference grid
- interpolation: bilinear
- GCPs written: 1957 (`outputs/reports/perspective_sift_gcps.csv`)

## What these numbers do and do not show

- Reprojection error (inlier RMSE) measures how well the matches agree with the fitted model - it is computed on the very points that voted for the model, so it is optimistic and says nothing about errors shared by ALL matches (a systematic offset, a wrong CRS, terrain relief).
- Held-out checkpoint error is more honest but still relative: the checkpoints come from the same matcher and the same two images, so they cannot reveal an error that is common to both.
- Absolute geographic accuracy requires independent surveyed points (GNSS / cadastral corners) or a reference whose own absolute accuracy is known. The output can be no more accurate than the reference grid it is registered to.
- A planar transform (affine/homography) cannot remove terrain-relief displacement or sensor geometry; registration is not orthorectification.

## Timing (s)

load+prepare: 0.05, register: 0.18, warp: 0.01, bootstrap: 0.02, total: 1.60
