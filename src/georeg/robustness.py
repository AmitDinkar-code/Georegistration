"""Phase 8: controlled robustness study, SIFT vs ORB.

Every case is synthetic with an exact ground-truth transform. Outcomes:
  accepted_correct : passed the quality gates AND ground-truth corner-grid RMSE <= tol
  accepted_wrong   : passed the gates but is actually wrong  (the dangerous one: FALSE ACCEPT)
  rejected         : gates / fallback refused it (safe failure)
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

from .config import PipelineCfg
from .dataset_loader import detection_mask, prepare_gray
from .io_utils import ensure_dir
from .pipeline import estimate_registration
from .synthetic_transform import make_case, procedural_scene
from .transformation_estimator import ground_truth_error
from .viz import save_figure

log = logging.getLogger("georeg")

BASE = dict(rotation_deg=10.0, scale=0.9, translation=(15.0, -10.0), crop=0.8)

FACTORS = {
    "rotation_deg": [0, 15, 45, 90, 180],
    "scale (src px per ref px)": [0.3, 0.5, 1.0, 2.0, 3.0],
    "brightness (gain,bias)": [(1.0, 0), (0.6, 0), (1.4, 20), (0.3, -30)],
    "noise sigma": [0, 20, 40, 70, 100],
    "occlusion fraction": [0.0, 0.3, 0.5, 0.7, 0.85],
    "low texture (blur px, contrast)": [(0, 1.0), (4, 0.4), (8, 0.25), (12, 0.15), (16, 0.08)],
    "source resolution (x ref)": [1.0, 0.35, 0.2, 0.12, 0.08],
    "perspective strength": [0.0, 2e-4, 4e-4, 6e-4, 8e-4],
    "scene change (area replaced)": [0.0, 0.3, 0.6, 0.8, 0.9],
}


def _case_kwargs(factor: str, level) -> dict:
    kw = dict(BASE)
    if factor == "rotation_deg":
        kw["rotation_deg"] = float(level)
    elif factor.startswith("scale"):
        kw["scale"] = float(level)
    elif factor.startswith("brightness"):
        kw["gain"], kw["bias"] = level
    elif factor.startswith("noise"):
        kw["noise_sigma"] = float(level)
    elif factor.startswith("occlusion"):
        kw["occlusion"] = float(level)
    elif factor.startswith("source resolution"):
        kw["scale"] = float(level)
    elif factor.startswith("scene change"):
        pass  # handled when the source scene is built
    elif factor.startswith("perspective"):
        kw["perspective"] = (float(level), -0.6 * float(level))
    return kw


def _reference_for(factor: str, level, size: int, seed: int) -> np.ndarray:
    ref = procedural_scene(size, seed)
    if factor.startswith("low texture") and level != (0, 1.0):
        blur, contrast = level
        f = ref.astype(np.float32)
        if blur:
            f = np.stack([gaussian_filter(b, blur) for b in f])
        mean = f.mean(axis=(1, 2), keepdims=True)
        ref = np.clip(mean + contrast * (f - mean), 1, 255).astype(np.uint8)
    return ref


def _changed_scene(ref: np.ndarray, fraction: float, size: int, seed: int) -> np.ndarray:
    """Simulate change between acquisition dates: replace `fraction` of the area with
    unrelated content (new fields / construction) before the source is derived."""
    if fraction <= 0:
        return ref
    other = procedural_scene(size, seed + 5000)
    mask = np.zeros(ref.shape[1:], np.uint8)
    rng = np.random.default_rng(seed)
    import cv2
    while mask.mean() / 255.0 < fraction:
        w, h = rng.integers(size // 8, size // 3, 2)
        x, y = rng.integers(0, size - w), rng.integers(0, size - h)
        mask[y:y + h, x:x + w] = 255
    out = ref.copy()
    out[:, mask > 0] = other[:, mask > 0]
    return out


def run_robustness(out_dir, size: int = 640, n_trials: int = 3, detectors=("sift", "orb"),
                   seed: int = 11, gt_tol_px: float = 3.0, factors: dict | None = None) -> dict:
    out = ensure_dir(out_dir)
    factors = factors or FACTORS
    rows = []
    t_start = time.perf_counter()
    for factor, levels in factors.items():
        for level in levels:
            for trial in range(n_trials):
                ref = _reference_for(factor, level, size, seed + trial)
                kw = _case_kwargs(factor, level)
                kw["rotation_deg"] += 3.0 * trial          # small per-trial jitter
                kw["translation"] = (kw["translation"][0] + 7 * trial, kw["translation"][1] - 5 * trial)
                src_scene = _changed_scene(ref, level, size, seed + trial) if factor.startswith("scene change") else ref
                case = make_case(src_scene, seed=seed + 100 * trial, **kw)
                for det in detectors:
                    cfg = PipelineCfg(attempts=((det, ("affine", "homography")),), phase_fallback=False, bootstrap_runs=0)
                    sg = prepare_gray(case.data, case.valid, cfg.preprocess)
                    rg = prepare_gray(ref, None, cfg.preprocess)
                    t0 = time.perf_counter()
                    core = estimate_registration(sg, rg, detection_mask(case.valid, 3), None, cfg)
                    dt = time.perf_counter() - t0
                    row = dict(factor=factor, level=str(level), trial=trial, detector=det, seconds=round(dt, 3),
                               n_keypoints=len(core.fs_src) if core.fs_src else 0,
                               n_matches=len(core.match) if core.match else 0,
                               n_inliers=core.estimate.n_inliers if core.estimate else 0,
                               model=core.estimate.model if core.estimate else "", gt_rmse_px="")
                    if core.success:
                        gt = ground_truth_error(core.T, case.T_true, sg.shape)
                        row["gt_rmse_px"] = round(gt["rmse_px"], 3)
                        row["outcome"] = "accepted_correct" if gt["rmse_px"] <= gt_tol_px else "accepted_wrong"
                    else:
                        row["outcome"] = "rejected"
                    rows.append(row)
        log.info("robustness: finished factor '%s' (%.0fs elapsed)", factor, time.perf_counter() - t_start)

    raw = out / "robustness_trials.csv"
    with open(raw, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    summary = []
    for factor, levels in factors.items():
        for level in levels:
            for det in detectors:
                sel = [r for r in rows if r["factor"] == factor and r["level"] == str(level) and r["detector"] == det]
                n = len(sel)
                good = [r for r in sel if r["outcome"] == "accepted_correct"]
                summary.append(dict(
                    factor=factor, level=str(level), detector=det, trials=n,
                    success_rate=len(good) / n,
                    false_accept_rate=sum(r["outcome"] == "accepted_wrong" for r in sel) / n,
                    rejected_rate=sum(r["outcome"] == "rejected" for r in sel) / n,
                    median_gt_rmse_px=float(np.median([r["gt_rmse_px"] for r in good])) if good else float("nan"),
                    median_inliers=float(np.median([r["n_inliers"] for r in sel])),
                    mean_seconds=float(np.mean([r["seconds"] for r in sel]))))
    sp = out / "robustness_summary.csv"
    with open(sp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    md = ["# Robustness study", "",
          f"Synthetic scene {size}x{size}, {n_trials} trials per cell, correct = ground-truth RMSE <= {gt_tol_px} px. "
          "Success counts only correct registrations; `false` = accepted by the gates but wrong.", "",
          "| factor | level | detector | success | false-accept | rejected | median GT RMSE px | median inliers | s |", "|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        md.append(f"| {s['factor']} | {s['level']} | {s['detector']} | {s['success_rate']:.0%} | {s['false_accept_rate']:.0%} | "
                  f"{s['rejected_rate']:.0%} | {s['median_gt_rmse_px']:.2f} | {s['median_inliers']:.0f} | {s['mean_seconds']:.2f} |")
    (out / "robustness_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig = Figure(figsize=(15, 8))
    axes = fig.subplots(2, 4).ravel()
    for ax, (factor, levels) in zip(axes, factors.items()):
        x = np.arange(len(levels)); wd = 0.38
        for j, det in enumerate(detectors):
            vals = [next(s["success_rate"] for s in summary if s["factor"] == factor and s["level"] == str(l) and s["detector"] == det) for l in levels]
            ax.bar(x + (j - (len(detectors) - 1) / 2) * wd, vals, wd, label=det.upper())
        ax.set_xticks(x); ax.set_xticklabels([str(l) for l in levels], fontsize=7, rotation=20)
        ax.set_ylim(0, 1.05); ax.set_title(factor, fontsize=9)
    axes[0].legend(); axes[0].set_ylabel("correct-registration rate")
    fig.suptitle("SIFT vs ORB robustness (synthetic, exact ground truth)")
    fp = save_figure(fig, out / "robustness_summary.png")
    return dict(rows=rows, summary=summary, csv=raw, summary_csv=sp, markdown=out / "robustness_summary.md", figure=fp)
