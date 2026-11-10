"""Phase 7: quantitative + visual accuracy assessment and report writing."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

from .io_utils import ensure_dir, imwrite_unicode
from .viz import draw_correspondences, save_figure, to_bgr

LIMITATIONS = [
    "Reprojection error (inlier RMSE) measures how well the matches agree with the fitted model - "
    "it is computed on the very points that voted for the model, so it is optimistic and says "
    "nothing about errors shared by ALL matches (a systematic offset, a wrong CRS, terrain relief).",
    "Held-out checkpoint error is more honest but still relative: the checkpoints come from the same "
    "matcher and the same two images, so they cannot reveal an error that is common to both.",
    "Absolute geographic accuracy requires independent surveyed points (GNSS / cadastral corners) "
    "or a reference whose own absolute accuracy is known. The output can be no more accurate than the "
    "reference grid it is registered to.",
    "A planar transform (affine/homography) cannot remove terrain-relief displacement or sensor "
    "geometry; registration is not orthorectification.",
]


def ncc(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    if valid is not None:
        a, b = a[valid], b[valid]
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) / d) if d > 0 else float("nan")


def gradient_ncc(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    """NCC of Sobel magnitudes - tolerant to radiometric differences (brightness, sensor)."""
    def mag(x):
        x = x.astype(np.float32)
        return np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))
    v = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return ncc(mag(a), mag(b), v) if v.any() else float("nan")


def overlay(ref_gray, reg_gray, valid) -> np.ndarray:
    """Reference in R+B, registered in G: aligned detail is grey, misalignment shows magenta/green fringes."""
    r = ref_gray.copy(); g = np.where(valid, reg_gray, 0).astype(np.uint8)
    return np.dstack([r, g, r])  # BGR order -> B=ref, G=reg, R=ref


def difference_image(ref_gray, reg_gray, valid) -> np.ndarray:
    ref = ref_gray.astype(np.float32); reg = reg_gray.astype(np.float32)
    if valid.sum() > 10:  # remove global gain/offset so we see GEOMETRIC differences
        reg = (reg - reg[valid].mean()) / (reg[valid].std() + 1e-6) * ref[valid].std() + ref[valid].mean()
    d = np.abs(ref - reg)
    d[~valid] = 0
    d8 = np.clip(d / max(1.0, np.percentile(d[valid], 99) if valid.any() else 1.0) * 255, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    out[~valid] = 0
    return out


def checkerboard(ref_gray, reg_gray, valid, tile: int = 64) -> np.ndarray:
    H, W = ref_gray.shape
    yy, xx = np.mgrid[0:H, 0:W]
    pick = (((yy // tile) + (xx // tile)) % 2 == 0) & valid
    out = np.where(pick, reg_gray, ref_gray).astype(np.uint8)
    out = to_bgr(out)
    edge = np.zeros((H, W), bool)
    edge[::tile, :] = True; edge[:, ::tile] = True
    out[edge & valid] = (0, 200, 255)
    return out


def save_comparison_images(ref_gray, reg_gray, valid, out_dir, name, naive_gray=None, naive_valid=None) -> dict:
    out_dir = ensure_dir(out_dir)
    paths = {}
    paths["overlay_after"] = imwrite_unicode(out_dir / f"{name}_overlay_after.png", overlay(ref_gray, reg_gray, valid))
    paths["difference"] = imwrite_unicode(out_dir / f"{name}_difference.png", difference_image(ref_gray, reg_gray, valid))
    paths["checkerboard"] = imwrite_unicode(out_dir / f"{name}_checkerboard.png", checkerboard(ref_gray, reg_gray, valid))
    if naive_gray is not None:
        paths["overlay_before"] = imwrite_unicode(out_dir / f"{name}_overlay_before.png",
                                                  overlay(ref_gray, naive_gray, naive_valid))
        fig = Figure(figsize=(13, 6.4))
        a, b = fig.subplots(1, 2)
        a.imshow(overlay(ref_gray, naive_gray, naive_valid)[..., ::-1]); a.set_title("BEFORE: source stamped on the grid, no transform", fontsize=10)
        b.imshow(overlay(ref_gray, reg_gray, valid)[..., ::-1]); b.set_title("AFTER: registered (grey = aligned, colour fringe = offset)", fontsize=10)
        a.axis("off"); b.axis("off")
        paths["before_after"] = save_figure(fig, out_dir / f"{name}_before_after.png")
    return paths


def correspondence_plot(src_gray, ref_gray, pts_src, pts_ref, roles, path) -> np.ndarray:
    roles = np.asarray(roles)
    layers = []
    for role, col, label in [("outlier", (60, 60, 230), "RANSAC outliers"),
                             ("checkpoint_bad", (30, 140, 255), "checkpoints off"),
                             ("checkpoint_ok", (230, 160, 40), "checkpoints ok"),
                             ("inlier", (60, 200, 60), "RANSAC inliers")]:
        m = roles == role
        if m.any():
            layers.append(dict(pts_s=pts_src[m], pts_r=pts_ref[m], color=col, label=label))
    return draw_correspondences(src_gray, ref_gray, layers, path, max_lines=300)


def spatial_error_figure(path, ref_gray, pts_src, pts_ref, T, roles, gt: dict | None, thresh: float) -> Path:
    from .transformation_estimator import apply_transform
    pred = apply_transform(T, pts_src)
    res = pred - pts_ref
    err = np.linalg.norm(res, axis=1)
    roles = np.asarray(roles)
    fig = Figure(figsize=(16, 5.2))
    a, b, c = fig.subplots(1, 3)
    a.imshow(ref_gray, cmap="gray")
    show = (roles == "inlier") | (roles == "checkpoint_ok")
    k = 15.0
    col = np.where(roles[show] == "inlier", 0.0, 1.0)
    a.quiver(pts_ref[show, 0], pts_ref[show, 1], -res[show, 0] * k, -res[show, 1] * k, col, cmap="winter",
             angles="xy", scale_units="xy", scale=1, width=0.003)
    a.set_title(f"Residual vectors x{k:.0f} (green=fit, blue=checkpoint)", fontsize=10); a.axis("off")
    if gt is not None:
        P = gt["grid_points"]
        sc = b.scatter(P[:, 0], P[:, 1], c=gt["errors_px"], cmap="viridis", s=45)
        b.invert_yaxis(); b.set_aspect("equal")
        fig.colorbar(sc, ax=b, label="error vs ground truth (ref px)")
        b.set_title(f"Ground-truth error over source grid\nRMSE {gt['rmse_px']:.3f} px, max {gt['max_px']:.3f}", fontsize=10)
    else:
        b.text(0.5, 0.5, "no ground truth available", ha="center"); b.axis("off")
    bins = np.linspace(0, max(thresh * 3, float(np.percentile(err, 95)) if len(err) else thresh), 40)
    for r, colr in [("inlier", "tab:green"), ("outlier", "tab:red"), ("checkpoint_ok", "tab:blue"), ("checkpoint_bad", "tab:orange")]:
        m = roles == r
        if m.any():
            c.hist(np.clip(err[m], 0, bins[-1]), bins=bins, alpha=0.6, label=f"{r} ({m.sum()})", color=colr)
    c.axvline(thresh, ls="--", color="k"); c.set_xlabel("forward reprojection error (ref px)"); c.legend(fontsize=8)
    c.set_title("Error distribution (clipped at right edge)", fontsize=10)
    return save_figure(fig, path)


def model_comparison_figure(path, estimates: dict, chosen: str) -> Path:
    fig = Figure(figsize=(14, 4.6))
    a, b = fig.subplots(1, 2)
    a.axis("off")
    txt = ""
    for k, e in estimates.items():
        if e is None:
            txt += f"{k}: failed\n\n"; continue
        m = e.matrix
        txt += (f"{k}{'  <== selected' if k == chosen else ''}  ({e.dof} DoF)\n"
                + "\n".join("  [" + "  ".join(f"{v: 11.5f}" for v in row) + " ]" for row in m)
                + f"\n  inliers {e.n_inliers}/{e.n_points}  RMSE {e.rmse_inliers:.3f}px  cond(L) {e.cond_linear:.2f}\n\n")
    a.text(0, 1, txt, family="monospace", fontsize=8, va="top")
    names = [k for k, e in estimates.items() if e is not None]
    inl = [estimates[k].n_inliers for k in names]
    bb = b.bar(names, inl, color=["tab:green" if k == chosen else "tab:gray" for k in names])
    b.set_ylabel("RANSAC inliers"); b.set_title("Model comparison (fewer DoF preferred if inliers within tolerance)", fontsize=10)
    for rect, e in zip(bb, [estimates[k] for k in names]):
        b.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"RMSE {e.rmse_inliers:.2f}px", ha="center", va="bottom", fontsize=8)
    return save_figure(fig, path)


# ------------------------------------------------------------- reports ------
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items() if not isinstance(v, np.ndarray) or v.size < 5000}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def _f(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_markdown(r: dict) -> str:
    L = [f"# Registration report - {r['run_name']}", ""]
    L.append(f"**Outcome:** {'ACCEPTED' if r['success'] else 'REJECTED - no GeoTIFF written'}"
             + (f"  |  method `{r.get('method')}` / model `{r.get('model')}`" if r['success'] else f"  |  reason: {r.get('failure_reason')}"))
    if r.get("low_confidence"):
        L.append("\n> **LOW CONFIDENCE:** produced by the translation-only phase-correlation fallback.")
    L += ["", "## Attempts", "", "| detector | models | status | keypoints (src/ref) | matches | detail |", "|---|---|---|---|---|---|"]
    for a in r["attempts"]:
        L.append(f"| {a['detector']} | {','.join(a['models'])} | {a['status']} | {a['n_kp_src']}/{a['n_kp_ref']} | {a['n_matches']} | {a['reason']} |")
    if r["success"]:
        m = r["metrics"]
        L += ["", "## Relative alignment (self-consistency)", "", "| metric | value |", "|---|---|"]
        for k in ["keypoints_source", "keypoints_reference", "matches_accepted", "ransac_inliers_fit_set", "inlier_ratio",
                  "inlier_rmse_px", "inlier_median_px", "model_dof", "ransac_iterations_needed_est"]:
            if k in m:
                L.append(f"| {k} | {_f(m[k])} |")
        h = r.get("holdout")
        if h:
            L += ["", "## Independent checkpoints (withheld matches)", "",
                  f"- checkpoints: {h['n_checkpoints']}, consistent within 2x threshold: {h['n_consistent']} ({_f(h['fraction_consistent'], 2)})",
                  f"- median error: {_f(h['median_error_px'])} px; RMSE over consistent: {_f(h['rmse_consistent_px'])} px"]
        else:
            L += ["", "## Independent checkpoints", "", "- not available (too few matches to withhold a set)"]
        c = r.get("consistency")
        if c:
            L += ["", "## Registration consistency (bootstrap)", "",
                  f"- {c['n_fits']} refits on 70 % subsets: probe-point spread RMS {_f(c['spread_rms_px'])} px, max {_f(c['spread_max_px'])} px"]
        g = r.get("ground_truth")
        if g:
            L += ["", "## Ground-truth error (synthetic test only)", "",
                  f"- pixel error (reference px): RMSE {_f(g['rmse_px'])}, median {_f(g['median_px'])}, p95 {_f(g['p95_px'])}, max {_f(g['max_px'])}"]
            if "rmse_map" in g:
                L.append(f"- map error: RMSE {_f(g['rmse_map'])}, max {_f(g['max_map'])} ({g['map_unit']})")
            if r.get("naive_stamp_ground_truth"):
                n = r["naive_stamp_ground_truth"]
                L.append(f"- for comparison, simply stamping the CRS on the unwarped image: RMSE {_f(n['rmse_px'])} px"
                         + (f" = {_f(n['rmse_map'], 1)} map units" if 'rmse_map' in n else ""))
        im = r.get("image_similarity")
        if im:
            L += ["", "## Image similarity to the reference (valid area)", "",
                  "| | NCC | gradient-NCC |", "|---|---|---|",
                  f"| before (stamped, no warp) | {_f(im['ncc_before'])} | {_f(im['gradient_ncc_before'])} |",
                  f"| after (registered) | {_f(im['ncc_after'])} | {_f(im['gradient_ncc_after'])} |"]
        ip = r.get("interpolation")
        if ip:
            L += ["", "## Interpolation comparison", "", "| method | NCC | sharpness | values preserved | round-trip PSNR dB | ms |", "|---|---|---|---|---|---|"]
            for x in ip:
                L.append(f"| {x['interpolation']} | {_f(x['ncc_vs_reference'], 4)} | {_f(x['sharpness'], 1)} | {_f(x['value_preserved'], 3)} | {_f(x['roundtrip_psnr_db'], 2)} | {x['seconds']*1000:.1f} |")
        L += ["", "## Output", "", f"- GeoTIFF: `{r['outputs'].get('geotiff')}`", f"- footprint covered: {_f(r['metrics'].get('valid_fraction'), 3)} of reference grid",
              f"- interpolation: {r['metrics'].get('interpolation')}"]
        if r.get("gcps"):
            L.append(f"- GCPs written: {r['gcps']['n']} (`{r['gcps']['csv']}`)")
    L += ["", "## What these numbers do and do not show", ""] + [f"- {x}" for x in LIMITATIONS]
    L += ["", "## Timing (s)", "", ", ".join(f"{k}: {v:.2f}" for k, v in r.get("timing", {}).items()), ""]
    return "\n".join(L)


def write_reports(report: dict, out_dir, name: str) -> dict:
    out_dir = ensure_dir(out_dir)
    jp = out_dir / f"{name}_report.json"
    mp = out_dir / f"{name}_report.md"
    jp.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    mp.write_text(render_markdown(to_jsonable(report)), encoding="utf-8")
    return dict(json=jp, markdown=mp)
