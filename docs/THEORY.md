# Theory and engineering decisions

Notation: `p = (x, y)` source pixel, `q = (u, v)` reference pixel, `T` maps `p → q` (3×3, homogeneous). OpenCV convention: the centre of pixel (0,0) is (0,0).

## 1. Terminology (not interchangeable)
| term | meaning | done here? |
|---|---|---|
| assigning spatial metadata | stamping a CRS/transform on pixels that never moved | **No** - explicitly avoided. The demo measures it: 322 px (3.2 km) error |
| image registration | estimating the geometric transform between two images | yes |
| georeferencing | using a registration (or GCPs) to put pixels on a map grid | yes: resample onto the reference grid, inherit its CRS+transform |
| orthorectification | removing relief + sensor-geometry displacement (DEM + sensor model) | **No** (§8) |

## 2. Keypoints vs descriptors; invariance
* A **keypoint** is a *where*: location, plus the scale and orientation at which it was found. A **descriptor** is a *what*: a vector summarising the neighbourhood, normalised by that scale and orientation so two views of the same corner give nearby vectors. Detect first, describe second; matching only uses descriptors, geometry only uses locations.
* **Scale invariance:** SIFT builds a Gaussian scale-space, takes difference-of-Gaussians extrema across space *and* scale, so a blob is found at the scale matching its size (`keypoint.size`). ORB uses a 8-level image pyramid (factor 1.2) with FAST corners per level - invariance only within the covered scale range (1.2⁷ ≈ 3.6×).
* **Rotation invariance:** each keypoint gets a dominant gradient orientation (SIFT: histogram peak; ORB: intensity-centroid angle) and the descriptor is computed in that rotated frame. This is why the study handles 0-180° identically.
* **SIFT:** 128-float descriptor of gradient histograms (4×4 cells × 8 bins), L2 distance, high quality, slower, strongly illumination-tolerant. **ORB:** oriented FAST + rotated BRIEF = 256 binary intensity comparisons, Hamming distance (XOR+popcount), very fast, less distinctive and less scale-tolerant (measured: ORB failed at 0.2× resolution, SIFT survived to 0.12×).
* **Limits for this task:** both use *raw intensity structure*. Multispectral pairs (RGB vs NIR), optical vs SAR, strong seasonal change, low texture (water, desert, uniform crops) and repetitive texture (rows, grids) all reduce distinctive, repeatable corners. Mitigations: CLAHE, matching gradient/edge maps, band selection that correlates across sensors, area-based methods (mutual information) or learned matchers.

## 3. Matching
Brute force computes all N×M descriptor distances (exact). FLANN uses randomised KD-trees (float) or LSH (binary) for approximate nearest neighbours - faster for large sets, can miss true neighbours (tests include both).
**Lowe ratio test:** with nearest distance `d₁` and second `d₂`, keep a match only if `d₁ < r·d₂` (r = 0.75 SIFT, 0.8 ORB). A correct match usually has one clearly closest neighbour; a wrong/repetitive one has several near-equals. Tests show precision is non-decreasing as `r` tightens and that the count is monotone in `r`.
Additional filters: optional mutual (reciprocal) nearest neighbour; uniqueness (several source points claiming one reference point keep the closest). If fewer than `min_matches` survive → `InsufficientMatchesError` → next fallback.

## 4. Transformation models and RANSAC
| model | DoF | min. pairs | preserves | use |
|---|---|---|---|---|
| similarity | 4 | 2 | angles, ratios | same sensor, nadir, flat |
| affine | 6 | 3 | parallelism | adds anisotropic scale/shear (e.g. scan geometry); weak perspective |
| homography | 8 | 4 | straight lines | plane-to-plane perspective; the exact model for a flat scene from any viewpoint |

Each correspondence gives two equations; a homography `h` (9 entries, defined up to scale ⇒ 8 DoF) is `[x' y' w]ᵀ = H [x y 1]ᵀ`, `(u,v) = (x'/w, y'/w)`.
**RANSAC:** repeat { draw the minimal sample, fit, count pairs whose forward reprojection error `e = ‖T(p) − q‖ ≤ τ` }; keep the largest consensus set; then least-squares refit on its inliers and re-score. Required iterations for confidence `P`, inlier ratio `w`, sample size `s`: `N = ln(1−P) / ln(1−wˢ)` (e.g. w = 0.5, s = 4, P = 0.99 → 71; a test pins this). With the 96 % inlier ratio of the demo it is ~3 - far below the 5000 cap. Inlier threshold `τ` (default 3 ref px) is a *configurable assumption about match noise*, not a measured accuracy.
**Model selection** by parsimony: the lowest-DoF model within 2 % of the best inlier count. Extra DoF fit noise and extrapolate badly outside the matched area; `cond(L)` of the linear part and the *coverage* gate (convex hull of inliers / image area) guard against that.

## 5. GCPs
A match is two *pixel* coordinates. It becomes a Ground Control Point only after the **reference** pixel is mapped through the reference geotransform: `X = a·(u+0.5) + b·(v+0.5) + c`, `Y = d·(u+0.5) + e·(v+0.5) + f` (the +0.5: OpenCV pixel centres are at integers, GDAL/rasterio transforms refer to pixel corners; omitting it silently shifts everything by half a pixel). Each `GCP` carries source px, reference px, map XY, residual and role. `gdal_commands()` emits the equivalent `gdal_translate -gcp … / gdalwarp -order 1` lines (GDAL GCP pixel coordinates are corner-based, so +0.5 on the source pixel too).

## 6. Warping and interpolation
`cv2.warpPerspective(src, T, (W_ref, H_ref))` is backward mapping: for every *output* pixel it samples the source at `T⁻¹(x, y)`, so there are no holes and the output **is** the reference grid. Giving `T` in the source→reference direction is correct *without* `WARP_INVERSE_MAP` (a test places a single bright pixel and asserts it lands at `T(p)`).
The valid mask is the warped source footprint (nearest), eroded by 0/1/2 px for nearest/bilinear/cubic because their kernels reach 0/1/2 px past the edge and would blend nodata in.
| kernel | support | effect |
|---|---|---|
| nearest | 1 px | keeps exact source values (right for classified/DN data), but blocky edges and up to ±0.5 px positional jitter |
| bilinear | 2×2 | smooth, no overshoot, but low-pass blurs (loses sharpness) and invents intermediate values |
| cubic | 4×4 | sharper than bilinear, can ring/overshoot at edges (clipped to dtype) |
Measured in the demo (`demo_sift_report.md`, source coarser than reference so it is *up*-sampled): sharpness (Laplacian variance) nearest 569 (inflated by staircase edges), bilinear 102, cubic 315; NCC to the reference 0.970 / 0.979 / 0.984; share of pixels whose full colour vector exists in the source: 1.00 / 0.68 / 0.66; round-trip PSNR 55.9 / 38.9 / 47.0 dB. Note the last: for *up*-sampling, nearest round-trips almost perfectly because it loses nothing; for a pure rotation the order flips (bilinear/cubic > nearest, pinned by a test). There is no universally best kernel: pick by data type and purpose.

## 7. Validation: what each number can and cannot say
* **Inlier ratio / inlier RMSE** - agreement of matches with *their own* fitted model. Optimistic; blind to anything all matches share.
* **Held-out checkpoint error** - same, on matches that did not vote. Better, still relative and from the same matcher.
* **Bootstrap consistency** - refit on 70 % subsets with different seeds; how far five probe points move. Small = well-determined by the data, *not* correct.
* **Ground-truth error** - only with a known transform (synthetic). Computed over a dense 15×15 grid of source pixels: `‖T_est(p) − T_true(p)‖` in reference px and map units (displacement via the linear part of the geotransform).
* **Absolute accuracy** needs independent surveyed points or a reference of known absolute accuracy. Not measurable from this software alone. The demo shows the tension: ORB had the *worse* inlier RMSE (1.41 vs 0.64 px) yet no worse ground-truth error.
* Image metrics (NCC, gradient-NCC, overlay, checkerboard, difference) show alignment visually/statistically; the difference image removes global gain/offset first so it shows geometry, not radiometry.

## 8. Why a planar transform is not orthorectification
A homography/affine is exact only when the scene is **a plane** or the viewpoint is a **pure rotation** about the camera centre. Reality breaks that:
* **Terrain relief:** a point at height `h` is displaced in the image by roughly `h · tan(θ)` (θ = off-nadir angle) relative to its ground position. At 10° off-nadir a 100 m hill shifts ≈ 17.6 m - several pixels at 5 m GSD - and the shift varies with the terrain, so no single matrix removes it.
* **Large areas:** Earth curvature and map-projection scale variation violate the flat-plane assumption; a scene-wide matrix averages local distortions.
* **Parallax / different viewing angles:** nearby objects (buildings, trees) shift relative to the ground between views; matches on rooftops and on ground disagree, RANSAC keeps one population and the rest become "outliers" that are actually correct but inconsistent.
* **Complex sensor geometry:** pushbroom satellites have a different projection centre per scan line (line-by-line attitude/orbit), whiskbroom/rolling-shutter effects; the image is not a single perspective frame at all.
**What fixes it:** orthorectification with a **DEM** and a **sensor model** (rigorous physical model, or RPC/RPB rational polynomial coefficients shipped with the product). For each output ground cell `(X, Y)` take its elevation `Z` from the DEM, project `(X, Y, Z)` through the sensor model into the source image, and sample there - the relief term is then computed explicitly instead of ignored. Ground control points (like the ones this tool produces) are then used to *refine* the sensor model (RPC bias correction), not replace it. In GDAL: `gdalwarp -rpc -to RPC_DEM=dem.tif …`. A practical stepping stone with this code: tile the scene and fit local transforms, accepting relief only if it is small relative to the tile's error budget.

## 9. Engineering decisions
Quality gates + ordered fallback (§README); everything configurable via frozen dataclasses (`config.py`) and logged; exceptions typed (`errors.py`) so each failure maps to "try next attempt" or "abort"; failure produces a report but **no** GeoTIFF; the phase-correlation fallback is translation-only, assumes the same footprint and is always flagged low-confidence; deterministic seeds everywhere (RANSAC via `cv2.setRNGSeed`, splits and synthetic scenes via NumPy generators).
