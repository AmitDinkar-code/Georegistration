"""Phase 4: robust geometric estimation (RANSAC) and model comparison.

Maths in one place
------------------
Homography (8 DoF, plane-to-plane projective):
    [x' y' w]^T = H [x y 1]^T,  (u,v) = (x'/w, y'/w);  H defined up to scale, h33 = 1.
    Needs 4 correspondences (2 equations each).
Affine (6 DoF): last row fixed to [0 0 1]; keeps parallel lines parallel.
    Needs 3 correspondences.
Similarity (4 DoF): rotation + isotropic scale + translation. Needs 2.

RANSAC: draw a minimal sample, fit, count correspondences whose forward
reprojection error  e_i = || T(p_i) - q_i ||  (reference pixels) is below a
threshold. Repeat; keep the best consensus set. With inlier ratio w and sample
size s, the iterations needed for confidence p are  N = ln(1-p) / ln(1-w^s).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import QualityGates, RansacCfg
from .errors import EstimationError

log = logging.getLogger("georeg")

DOF = {"homography": 8, "affine": 6, "similarity": 4, "translation": 2}
MIN_SAMPLES = {"homography": 4, "affine": 3, "similarity": 2}


# ------------------------------------------------------------ primitives ----
def apply_transform(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(M, np.float64)).reshape(-1, 2)


def reprojection_errors(M, pts_src, pts_ref) -> np.ndarray:
    return np.linalg.norm(apply_transform(M, pts_src) - np.asarray(pts_ref, np.float64), axis=1)


def est_ransac_iterations(inlier_ratio: float, sample: int, confidence: float) -> float:
    w = min(max(inlier_ratio, 1e-9), 1 - 1e-12)
    return math.log(1 - confidence) / math.log(1 - w ** sample)


def decompose(M: np.ndarray) -> dict:
    """Interpretable parameters of the linear part (exact for affine, local for homography)."""
    M = M / M[2, 2]
    L = M[:2, :2]
    sx = float(np.hypot(L[0, 0], L[1, 0]))
    det = float(np.linalg.det(L))
    sy = det / sx if sx > 0 else float("nan")
    return dict(translation=(float(M[0, 2]), float(M[1, 2])),
                rotation_deg=float(np.degrees(np.arctan2(L[1, 0], L[0, 0]))),
                scale_x=sx, scale_y=float(sy),
                shear=float((L[0, 0] * L[0, 1] + L[1, 0] * L[1, 1]) / det) if det else float("nan"),
                determinant=det, perspective=(float(M[2, 0]), float(M[2, 1])))


def _to_3x3(A: np.ndarray) -> np.ndarray:
    M = np.eye(3)
    M[:2, :] = A
    return M


def _fit_affine_lstsq(s: np.ndarray, r: np.ndarray) -> np.ndarray:
    X = np.c_[s, np.ones(len(s))]
    sol, *_ = np.linalg.lstsq(X, r, rcond=None)
    return _to_3x3(sol.T)


def _fit_similarity_umeyama(s: np.ndarray, r: np.ndarray) -> np.ndarray:
    ms, mr = s.mean(0), r.mean(0)
    S0, R0 = s - ms, r - mr
    U, D, Vt = np.linalg.svd(R0.T @ S0 / len(s))
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    Rm = U @ np.diag([1, d]) @ Vt
    scale = (D * np.array([1, d])).sum() / ((S0 ** 2).sum() / len(s))
    t = mr - scale * Rm @ ms
    M = np.eye(3)
    M[:2, :2] = scale * Rm
    M[:2, 2] = t
    return M


# ------------------------------------------------------------- estimate -----
@dataclass
class TransformEstimate:
    model: str
    matrix: np.ndarray                  # 3x3, source px -> reference px
    inlier_mask: np.ndarray             # (N,) bool over the points passed in
    errors: np.ndarray                  # (N,) forward reprojection error, ref px
    n_points: int
    n_inliers: int
    inlier_ratio: float
    rmse_inliers: float
    median_inliers: float
    max_inliers: float
    dof: int
    reproj_thresh: float
    est_iterations: float
    cond_linear: float
    decomposition: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return dict(model=self.model, dof=self.dof, n_points=self.n_points, n_inliers=self.n_inliers,
                    inlier_ratio=self.inlier_ratio, rmse_inliers_px=self.rmse_inliers,
                    median_inliers_px=self.median_inliers, max_inliers_px=self.max_inliers,
                    ransac_iterations_needed=self.est_iterations, cond_linear=self.cond_linear,
                    matrix=self.matrix.tolist(), decomposition=self.decomposition)


def _package(model, M, pts_s, pts_r, thresh, conf) -> TransformEstimate:
    err = reprojection_errors(M, pts_s, pts_r)
    inl = err <= thresh
    n_in = int(inl.sum())
    ein = err[inl] if n_in else np.array([np.nan])
    ratio = n_in / max(1, len(err))
    L = (M / M[2, 2])[:2, :2]
    return TransformEstimate(
        model, M / M[2, 2], inl, err, len(err), n_in, ratio,
        float(np.sqrt(np.mean(ein ** 2))), float(np.median(ein)), float(ein.max()),
        DOF[model], thresh,
        est_ransac_iterations(ratio, MIN_SAMPLES.get(model, 4), conf) if 0 < ratio < 1 else 1.0,
        float(np.linalg.cond(L)), decompose(M))


def estimate_transform(pts_src, pts_ref, cfg: RansacCfg, model: str) -> TransformEstimate:
    s = np.ascontiguousarray(pts_src, np.float64).reshape(-1, 2)
    r = np.ascontiguousarray(pts_ref, np.float64).reshape(-1, 2)
    if model not in MIN_SAMPLES:
        raise ValueError(f"unknown model '{model}'")
    if len(s) < MIN_SAMPLES[model] + 1:
        raise EstimationError(f"{len(s)} points is too few for a {model}")
    cv2.setRNGSeed(cfg.seed)
    sc, rc = s.reshape(-1, 1, 2), r.reshape(-1, 1, 2)
    if model == "homography":
        meth = cv2.USAC_MAGSAC if cfg.method == "magsac" else cv2.RANSAC
        M, _ = cv2.findHomography(sc, rc, meth, cfg.reproj_thresh, maxIters=cfg.max_iters,
                                  confidence=cfg.confidence)
    else:
        fn = cv2.estimateAffine2D if model == "affine" else cv2.estimateAffinePartial2D
        A, _ = fn(sc, rc, method=cv2.RANSAC, ransacReprojThreshold=cfg.reproj_thresh,
                  maxIters=cfg.max_iters, confidence=cfg.confidence)
        M = None if A is None else _to_3x3(A)
    if M is None or not np.all(np.isfinite(M)):
        raise EstimationError(f"RANSAC {model} failed to find a model")
    est = _package(model, M, s, r, cfg.reproj_thresh, cfg.confidence)

    if cfg.refine and est.n_inliers >= max(6, MIN_SAMPLES[model] + 2):
        si, ri = s[est.inlier_mask], r[est.inlier_mask]
        try:
            if model == "homography":
                Mr, _ = cv2.findHomography(si.reshape(-1, 1, 2), ri.reshape(-1, 1, 2), 0)
            elif model == "affine":
                Mr = _fit_affine_lstsq(si, ri)
            else:
                Mr = _fit_similarity_umeyama(si, ri)
        except (np.linalg.LinAlgError, cv2.error):
            Mr = None
        if Mr is not None and np.all(np.isfinite(Mr)):
            est2 = _package(model, Mr, s, r, cfg.reproj_thresh, cfg.confidence)
            if est2.n_inliers >= est.n_inliers:
                est = est2
    log.debug("%s: %d/%d inliers, rmse %.3f px", model, est.n_inliers, est.n_points, est.rmse_inliers)
    return est


def compare_models(pts_src, pts_ref, cfg: RansacCfg, models) -> dict:
    out = {}
    for m in models:
        try:
            out[m] = estimate_transform(pts_src, pts_ref, cfg, m)
        except EstimationError as exc:
            log.info("model %s unavailable: %s", m, exc)
            out[m] = None
    return out


def select_model(estimates: dict, tol: float) -> TransformEstimate:
    """Parsimony: the lowest-DoF model whose consensus is within `tol` of the best."""
    ok = {k: v for k, v in estimates.items() if v is not None}
    if not ok:
        raise EstimationError("no model could be estimated")
    best_n = max(v.n_inliers for v in ok.values())
    for v in sorted(ok.values(), key=lambda e: e.dof):
        if v.n_inliers >= (1 - tol) * best_n:
            return v
    return max(ok.values(), key=lambda e: e.n_inliers)  # unreachable, kept for safety


# ------------------------------------------------------------- QA gates -----
def spatial_coverage(pts_src, src_hw) -> float:
    """Convex-hull area of the points / source area. Low = inliers clustered, the
    model is being extrapolated over most of the image."""
    p = np.asarray(pts_src, np.float32).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    hull = cv2.convexHull(p)
    return float(cv2.contourArea(hull) / (src_hw[0] * src_hw[1]))


def check_plausibility(M: np.ndarray, src_hw, gates: QualityGates) -> list[str]:
    fails = []
    if not np.all(np.isfinite(M)) or abs(M[2, 2]) < 1e-12:
        return ["non-finite or degenerate matrix"]
    M = M / M[2, 2]
    det = float(np.linalg.det(M[:2, :2]))
    if abs(det) < 1e-9:
        return ["linear part is singular"]
    if det < 0 and not gates.allow_reflection:
        fails.append("reflection (negative determinant)")
    scale = math.sqrt(abs(det))
    if scale > gates.max_scale or scale < 1 / gates.max_scale:
        fails.append(f"implausible scale factor {scale:.3g}")
    h, w = src_hw
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    hom = (M @ np.c_[corners, np.ones(4)].T).T
    if np.any(hom[:, 2] <= 0):
        fails.append("corner maps beyond the horizon (w <= 0)")
        return fails
    if np.max(np.abs(hom[:, 2] - 1)) > gates.max_perspective:
        fails.append(f"perspective strength {np.max(np.abs(hom[:, 2] - 1)):.2f} > {gates.max_perspective}")
    q = hom[:, :2] / hom[:, 2:3]
    cr = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cr.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(x > 0 for x in cr) or all(x < 0 for x in cr)):
        fails.append("warped footprint is not a convex quadrilateral (folded)")
    return fails


def evaluate_gates(est: TransformEstimate, pts_src_inliers, src_hw, gates: QualityGates) -> list[str]:
    f = []
    if est.n_inliers < gates.min_inliers:
        f.append(f"inliers {est.n_inliers} < {gates.min_inliers}")
    if est.inlier_ratio < gates.min_inlier_ratio:
        f.append(f"inlier ratio {est.inlier_ratio:.2f} < {gates.min_inlier_ratio}")
    if est.rmse_inliers > gates.max_inlier_rmse_px:
        f.append(f"inlier RMSE {est.rmse_inliers:.2f}px > {gates.max_inlier_rmse_px}")
    cov = spatial_coverage(pts_src_inliers, src_hw)
    if cov < gates.min_coverage:
        f.append(f"inlier coverage {cov:.3f} < {gates.min_coverage}")
    f += check_plausibility(est.matrix, src_hw, gates)
    return f


# ------------------------------------------------------- independent QA -----
def split_holdout(n: int, frac: float, seed: int = 0, min_fit: int = 20, min_hold: int = 8):
    """Fit/checkpoint split performed BEFORE estimation, so checkpoints never vote."""
    n_hold = int(round(n * frac))
    if frac <= 0 or n_hold < min_hold or n - n_hold < min_fit:
        return np.arange(n), np.zeros(0, int)
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_hold:]), np.sort(perm[:n_hold])


def holdout_report(M, pts_src, pts_ref, thresh: float) -> dict:
    """Error of withheld matches. CAVEAT: these are matcher output, so some are
    wrong matches; we report the median over all and the RMSE over the ones
    consistent within 2*thresh. That filter makes RMSE optimistic - it is a
    repeatability check, not an absolute-accuracy measurement."""
    e = reprojection_errors(M, pts_src, pts_ref)
    cons = e <= 2 * thresh
    return dict(n_checkpoints=int(len(e)), n_consistent=int(cons.sum()),
                fraction_consistent=float(cons.mean()) if len(e) else float("nan"),
                median_error_px=float(np.median(e)) if len(e) else float("nan"),
                rmse_consistent_px=float(np.sqrt(np.mean(e[cons] ** 2))) if cons.any() else float("nan"),
                errors=e)


def ground_truth_error(T_est, T_true, src_hw, ref_transform=None, n_grid: int = 15, margin: float = 0.02) -> dict:
    """Error of the estimated transform against known truth over a dense grid of
    source pixels: ||T_est(p) - T_true(p)||. In reference pixels and, when the
    reference geotransform is given, in map units (e.g. metres)."""
    h, w = src_hw
    xs = np.linspace(margin * w, (1 - margin) * w, n_grid)
    ys = np.linspace(margin * h, (1 - margin) * h, n_grid)
    P = np.array([[x, y] for y in ys for x in xs])
    d = apply_transform(T_est, P) - apply_transform(T_true, P)
    px = np.linalg.norm(d, axis=1)
    out = dict(grid_points=P, error_vectors_px=d, errors_px=px,
               rmse_px=float(np.sqrt(np.mean(px ** 2))), mean_px=float(px.mean()),
               median_px=float(np.median(px)), max_px=float(px.max()),
               p95_px=float(np.percentile(px, 95)))
    if ref_transform is not None:
        A = np.array([[ref_transform.a, ref_transform.b], [ref_transform.d, ref_transform.e]])
        dm = d @ A.T                       # linear part only: a displacement, not a position
        mm = np.linalg.norm(dm, axis=1)
        out.update(errors_map=mm, rmse_map=float(np.sqrt(np.mean(mm ** 2))), max_map=float(mm.max()),
                   map_unit="CRS units (metres for projected CRS)")
    return out


def bootstrap_consistency(pts_src, pts_ref, cfg: RansacCfg, model: str, src_hw,
                          n_boot: int = 30, frac: float = 0.7, seed: int = 0) -> dict:
    """Registration consistency: refit on random 70 % subsets with different RANSAC
    seeds and measure how far five probe points move in the reference. Small spread =
    the solution is well-determined by the data, NOT that it is geographically right."""
    h, w = src_hw
    probes = np.array([[w * 0.1, h * 0.1], [w * 0.9, h * 0.1], [w * 0.9, h * 0.9],
                       [w * 0.1, h * 0.9], [w * 0.5, h * 0.5]])
    rng = np.random.default_rng(seed)
    s, r = np.asarray(pts_src), np.asarray(pts_ref)
    results = []
    for b in range(n_boot):
        idx = rng.choice(len(s), size=max(MIN_SAMPLES[model] + 4, int(frac * len(s))), replace=False)
        try:
            est = estimate_transform(s[idx], r[idx], RansacCfgSeed(cfg, seed + b), model)
        except EstimationError:
            continue
        results.append(apply_transform(est.matrix, probes))
    if len(results) < 3:
        return dict(n_fits=len(results), spread_rms_px=float("nan"), spread_max_px=float("nan"))
    R = np.stack(results)                       # (B,5,2)
    dev = np.linalg.norm(R - R.mean(0, keepdims=True), axis=2)
    return dict(n_fits=len(results), spread_rms_px=float(np.sqrt((dev ** 2).mean())),
                spread_max_px=float(dev.max()))


def RansacCfgSeed(cfg: RansacCfg, seed: int) -> RansacCfg:
    from dataclasses import replace
    return replace(cfg, seed=seed, refine=True)
