"""Orchestration: detect -> match -> estimate -> gate -> (fallback) -> warp -> write -> validate."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from . import control_points as cp
from . import registration_validation as rv
from .config import PipelineCfg, QualityGates
from .dataset_loader import (Raster, compare_inputs, describe_raster, detection_mask, load_raster,
                             prepare_gray)
from .errors import (EstimationError, GeoregError, InputError, InsufficientFeaturesError,
                     InsufficientMatchesError)
from .feature_detector import FeatureSet, detect_features, draw_keypoints, keypoint_statistics
from .feature_matcher import MatchResult, draw_match_figures, match_features
from .geotiff_writer import write_geotiff
from .image_warping import compare_interpolations, save_interpolation_figure, warp_to_reference
from .io_utils import ensure_dir
from .transformation_estimator import (DOF, TransformEstimate, apply_transform, bootstrap_consistency,
                                       compare_models, evaluate_gates, ground_truth_error, holdout_report,
                                       reprojection_errors, select_model, split_holdout)

log = logging.getLogger("georeg")


@dataclass
class AttemptLog:
    detector: str
    models: tuple
    status: str = "not_run"
    reason: str = ""
    n_kp_src: int = 0
    n_kp_ref: int = 0
    n_matches: int = 0
    chosen_model: str | None = None
    seconds: float = 0.0


@dataclass
class CoreResult:
    success: bool
    method: str | None = None
    low_confidence: bool = False
    T: np.ndarray | None = None
    estimate: TransformEstimate | None = None
    all_estimates: dict = field(default_factory=dict)
    match: MatchResult | None = None
    fs_src: FeatureSet | None = None
    fs_ref: FeatureSet | None = None
    roles: np.ndarray | None = None
    errors_all: np.ndarray | None = None
    fit_idx: np.ndarray | None = None
    hold_idx: np.ndarray | None = None
    holdout: dict | None = None
    attempts: list = field(default_factory=list)
    failure_reason: str = ""


def _ncc_overlap(a, b, va, vb):
    m = va & vb
    return rv.ncc(a, b, m) if m.sum() > 50 else float("nan")


def phase_correlation_fallback(src_gray, ref_gray, gates: QualityGates):
    """LAST RESORT: translation-only. ASSUMES the two images cover the same footprint
    (it just rescales source to reference size) with ~no rotation. Flagged low-confidence."""
    Hr, Wr = ref_gray.shape
    Hs, Ws = src_gray.shape
    S = np.diag([Wr / Ws, Hr / Hs, 1.0])
    s_rs = cv2.resize(src_gray, (Wr, Hr), interpolation=cv2.INTER_AREA if Wr < Ws else cv2.INTER_LINEAR)
    win = cv2.createHanningWindow((Wr, Hr), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref_gray.astype(np.float32), s_rs.astype(np.float32), win)
    best = None
    for sign in (1, -1):  # resolve the sign convention empirically instead of trusting it
        Tt = np.array([[1, 0, sign * dx], [0, 1, sign * dy], [0, 0, 1.0]])
        w = cv2.warpPerspective(s_rs, Tt, (Wr, Hr))
        v = cv2.warpPerspective(np.full_like(s_rs, 255), Tt, (Wr, Hr)) > 127
        n = _ncc_overlap(w, ref_gray, v, np.ones_like(v))
        if best is None or (np.isfinite(n) and n > best[0]):
            best = (n, Tt)
    ncc_v, Tt = best
    if resp < gates.min_phase_response or not np.isfinite(ncc_v) or ncc_v < gates.min_fallback_ncc:
        return None, f"phase correlation response {resp:.2f}, NCC {ncc_v:.2f} below gates"
    T = Tt @ S
    est = TransformEstimate("translation", T / T[2, 2], np.zeros(0, bool), np.zeros(0), 0, 0, float("nan"),
                            float("nan"), float("nan"), float("nan"), DOF["translation"], float("nan"),
                            float("nan"), float(np.linalg.cond(T[:2, :2])), {})
    return est, f"response {resp:.2f}, NCC {ncc_v:.2f}"


def estimate_registration(src_gray, ref_gray, src_mask, ref_mask, cfg: PipelineCfg) -> CoreResult:
    """Image-array level engine (no file I/O): used by the pipeline AND the robustness study."""
    res = CoreResult(success=False)
    src_hw = src_gray.shape
    for det, models in cfg.attempts:
        a = AttemptLog(det, tuple(models))
        res.attempts.append(a)
        t0 = time.perf_counter()
        try:
            dcfg = replace(cfg.detector, method=det)
            fs_s = detect_features(src_gray, dcfg, src_mask)
            fs_r = detect_features(ref_gray, dcfg, ref_mask)
            a.n_kp_src, a.n_kp_ref = len(fs_s), len(fs_r)
            mr = match_features(fs_s, fs_r, cfg.matcher)
            a.n_matches = len(mr)
            fit, hold = split_holdout(len(mr), cfg.ransac.holdout_fraction, cfg.ransac.seed)
            ests = compare_models(mr.pts_src[fit], mr.pts_ref[fit], cfg.ransac, models)
            chosen = select_model(ests, cfg.ransac.model_select_tol)
            a.chosen_model = chosen.model
            fails = evaluate_gates(chosen, mr.pts_src[fit][chosen.inlier_mask], src_hw, cfg.gates)
            if fails:
                a.status, a.reason = "rejected", "; ".join(fails)
                log.warning("attempt %s rejected: %s", det, a.reason)
                continue
            a.status, a.reason = "accepted", f"{chosen.model}, {chosen.n_inliers}/{chosen.n_points} inliers"
            res.success, res.method, res.T = True, det, chosen.matrix
            res.estimate, res.all_estimates = chosen, ests
            res.match, res.fs_src, res.fs_ref = mr, fs_s, fs_r
            res.fit_idx, res.hold_idx = fit, hold
            err = reprojection_errors(chosen.matrix, mr.pts_src, mr.pts_ref)
            res.errors_all = err
            roles = np.empty(len(mr), dtype=object)
            roles[fit] = np.where(err[fit] <= cfg.ransac.reproj_thresh, "inlier", "outlier")
            if len(hold):
                roles[hold] = np.where(err[hold] <= 2 * cfg.ransac.reproj_thresh, "checkpoint_ok", "checkpoint_bad")
                res.holdout = holdout_report(chosen.matrix, mr.pts_src[hold], mr.pts_ref[hold], cfg.ransac.reproj_thresh)
            res.roles = roles
            a.seconds = time.perf_counter() - t0
            return res
        except (InsufficientFeaturesError, InsufficientMatchesError, EstimationError) as exc:
            a.status, a.reason = "failed", str(exc).split("; stats=")[0]
            log.warning("attempt %s failed: %s", det, a.reason)
        finally:
            a.seconds = time.perf_counter() - t0

    if cfg.phase_fallback:
        a = AttemptLog("phase_correlation", ("translation",))
        res.attempts.append(a)
        est, why = phase_correlation_fallback(src_gray, ref_gray, cfg.gates)
        if est is not None:
            a.status, a.reason, a.chosen_model = "accepted", why, "translation"
            res.success, res.method, res.low_confidence = True, "phase_correlation", True
            res.T, res.estimate, res.all_estimates = est.matrix, est, {"translation": est}
            return res
        a.status, a.reason = "failed", why
    res.failure_reason = "; ".join(f"{x.detector}: {x.reason}" for x in res.attempts)
    return res


@dataclass
class RegistrationResult:
    success: bool
    run_name: str
    core: CoreResult
    report: dict
    outputs: dict
    warp: object | None = None

    @property
    def T(self):
        return self.core.T


def run_registration(source_path, reference_path, out_dir, cfg: PipelineCfg | None = None,
                     run_name: str | None = None, T_true: np.ndarray | None = None,
                     save_artifacts: bool = True) -> RegistrationResult:
    cfg = cfg or PipelineCfg()
    timing = {}
    t_all = time.perf_counter()
    name = run_name or f"{Path(source_path).stem}_to_{Path(reference_path).stem}"
    out = Path(out_dir)
    dirs = {k: ensure_dir(out / k) for k in ("keypoints", "matches", "registered", "geotiff", "reports")}

    t0 = time.perf_counter()
    src, ref = load_raster(source_path), load_raster(reference_path)
    if not ref.is_georeferenced:
        raise InputError("reference image has no CRS/geotransform; it must be a georeferenced GeoTIFF")
    inputs = dict(source=describe_raster(src), reference=describe_raster(ref), comparison=compare_inputs(src, ref))
    src_gray = prepare_gray(src.data, src.valid, cfg.preprocess)
    ref_gray = prepare_gray(ref.data, ref.valid, cfg.preprocess)
    smask = detection_mask(src.valid, cfg.preprocess.mask_erode_px)
    rmask = detection_mask(ref.valid, cfg.preprocess.mask_erode_px)
    timing["load+prepare"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    core = estimate_registration(src_gray, ref_gray, smask, rmask, cfg)
    timing["register"] = time.perf_counter() - t0

    report = dict(run_name=name, success=core.success, method=core.method, low_confidence=core.low_confidence,
                  model=core.estimate.model if core.estimate else None, failure_reason=core.failure_reason,
                  inputs=inputs, config=cfg.to_dict(), attempts=[a.__dict__ for a in core.attempts],
                  outputs={}, metrics={}, timing=timing)
    outputs = {}

    if not core.success:
        log.error("registration REJECTED: %s", core.failure_reason)
        outputs.update(rv.write_reports(report, dirs["reports"], name))
        timing["total"] = time.perf_counter() - t_all
        return RegistrationResult(False, name, core, report, outputs)

    T, est, mr = core.T, core.estimate, core.match
    ref_hw = (ref.height, ref.width)

    if save_artifacts and mr is not None:
        outputs["keypoints_source"] = dirs["keypoints"] / f"{name}_source.png"
        outputs["keypoints_reference"] = dirs["keypoints"] / f"{name}_reference.png"
        draw_keypoints(src_gray, core.fs_src, outputs["keypoints_source"])
        draw_keypoints(ref_gray, core.fs_ref, outputs["keypoints_reference"])
        outputs.update({f"matches_{k}": v for k, v in draw_match_figures(src_gray, ref_gray, mr, dirs["matches"], name).items()})
        outputs["correspondences"] = dirs["matches"] / f"{name}_ransac_inliers_outliers.png"
        rv.correspondence_plot(src_gray, ref_gray, mr.pts_src, mr.pts_ref, core.roles, outputs["correspondences"])
        report["keypoint_stats"] = dict(source=keypoint_statistics(core.fs_src), reference=keypoint_statistics(core.fs_ref))
        report["match_stats"] = mr.stats

    t0 = time.perf_counter()
    w = warp_to_reference(src.data, T, ref_hw, cfg.warp.interpolation, src.valid,
                          nodata=0 if src.nodata is None else src.nodata)
    timing["warp"] = time.perf_counter() - t0

    # naive baseline: pixels left where they are, reference georeferencing stamped on them
    naive = warp_to_reference(src.data, np.eye(3), ref_hw, "nearest", src.valid)
    naive_gray = prepare_gray(naive.data, naive.valid, cfg.preprocess)
    reg_gray = prepare_gray(w.data, w.valid, cfg.preprocess)
    interp_rows = None
    if cfg.warp.compare_interpolations:
        interp_rows, interp_res = compare_interpolations(src.data, T, ref_hw, ref_gray, src.valid)
        if save_artifacts:
            outputs["interpolation_figure"] = save_interpolation_figure(interp_res, interp_rows,
                                                                        dirs["registered"] / f"{name}_interpolation.png")

    # ---- GeoTIFF: reference CRS + reference transform, because the pixels now LIVE on that grid
    tags = dict(REGISTRATION_METHOD=core.method, REGISTRATION_MODEL=est.model,
                INTERPOLATION=cfg.warp.interpolation, RANSAC_INLIERS=est.n_inliers,
                INLIER_RMSE_PX=f"{est.rmse_inliers:.4f}", LOW_CONFIDENCE=int(core.low_confidence),
                ORTHORECTIFIED="NO", NOTE="relative image-to-image registration; terrain relief not removed",
                SOURCE_FILE=Path(source_path).name, REFERENCE_FILE=Path(reference_path).name)
    gt_path = dirs["geotiff"] / f"{name}_registered.tif"
    wr = write_geotiff(gt_path, w.data, ref.transform, ref.crs, valid=w.valid,
                       nodata=src.nodata if src.nodata is not None else 0, tags=tags,
                       descriptions=[f"source band {i + 1}" for i in range(src.count)])
    outputs["geotiff"] = gt_path

    # ---- validation
    metrics = dict(valid_fraction=float(w.valid.mean()), interpolation=cfg.warp.interpolation,
                   model_dof=est.dof, geotiff_nodata=wr["nodata"], geotiff_mask_band=wr["mask_written"])
    if mr is not None:
        metrics.update(keypoints_source=len(core.fs_src), keypoints_reference=len(core.fs_ref),
                       matches_accepted=len(mr), ransac_inliers_fit_set=est.n_inliers,
                       inlier_ratio=est.inlier_ratio, inlier_rmse_px=est.rmse_inliers,
                       inlier_median_px=est.median_inliers, ransac_iterations_needed_est=est.est_iterations)
    report["metrics"] = metrics
    report["holdout"] = core.holdout
    report["transform"] = est.summary()
    report["all_models"] = {k: (v.summary() if v is not None else None) for k, v in core.all_estimates.items()}
    report["image_similarity"] = dict(
        ncc_before=_ncc_overlap(naive_gray, ref_gray, naive.valid, np.ones_like(naive.valid)),
        gradient_ncc_before=rv.gradient_ncc(naive_gray, ref_gray, naive.valid) if naive.valid.any() else float("nan"),
        ncc_after=_ncc_overlap(reg_gray, ref_gray, w.valid, np.ones_like(w.valid)),
        gradient_ncc_after=rv.gradient_ncc(reg_gray, ref_gray, w.valid))
    report["interpolation"] = interp_rows

    gt = None
    if T_true is not None:
        gt = ground_truth_error(T, T_true, (src.height, src.width), ref.transform)
        report["ground_truth"] = {k: v for k, v in gt.items() if not isinstance(v, np.ndarray)}
        report["naive_stamp_ground_truth"] = {k: v for k, v in ground_truth_error(
            np.eye(3), T_true, (src.height, src.width), ref.transform).items() if not isinstance(v, np.ndarray)}

    if mr is not None and cfg.bootstrap_runs > 0:
        t0 = time.perf_counter()
        report["consistency"] = bootstrap_consistency(mr.pts_src, mr.pts_ref, cfg.ransac, est.model,
                                                      (src.height, src.width), cfg.bootstrap_runs)
        timing["bootstrap"] = time.perf_counter() - t0

    if mr is not None:
        gcps = cp.build_gcps(mr.pts_src, mr.pts_ref, core.errors_all, core.roles, ref.transform)
        if gcps:
            gp = cp.gcps_to_csv(gcps, dirs["reports"] / f"{name}_gcps.csv")
            (dirs["reports"] / f"{name}_gdal_commands.txt").write_text(
                cp.gdal_commands(gcps, Path(source_path).name, gt_path.name, ref.crs.to_string()), encoding="utf-8")
            report["gcps"] = dict(n=len(gcps), csv=str(gp))
        report["composite_source_pixel_to_map"] = cp.composite_pixel_to_map(ref.transform, T).tolist()

    if save_artifacts:
        outputs.update(rv.save_comparison_images(ref_gray, reg_gray, w.valid, dirs["registered"], name, naive_gray, naive.valid))
        if mr is not None:
            outputs["spatial_error"] = rv.spatial_error_figure(dirs["registered"] / f"{name}_spatial_error.png", ref_gray,
                                                               mr.pts_src, mr.pts_ref, T, core.roles, gt, cfg.ransac.reproj_thresh)
            outputs["model_comparison"] = rv.model_comparison_figure(dirs["registered"] / f"{name}_models.png",
                                                                     core.all_estimates, est.model)
    timing["total"] = time.perf_counter() - t_all
    report["outputs"] = {k: str(v) for k, v in outputs.items()}
    outputs.update(rv.write_reports(report, dirs["reports"], name))
    return RegistrationResult(True, name, core, report, outputs, w)
