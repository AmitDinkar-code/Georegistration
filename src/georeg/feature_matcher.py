"""Phase 3: descriptor matching with ambiguity rejection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import MatcherCfg
from .errors import InsufficientMatchesError
from .feature_detector import FeatureSet
from .viz import draw_correspondences

log = logging.getLogger("georeg")


@dataclass
class MatchResult:
    method: str
    src_idx: np.ndarray
    ref_idx: np.ndarray
    pts_src: np.ndarray          # accepted matches, source px (N,2)
    pts_ref: np.ndarray          # accepted matches, reference px (N,2)
    distances: np.ndarray
    rejected_src: np.ndarray     # best candidate of matches thrown out by ratio / mutual / uniqueness
    rejected_ref: np.ndarray
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.src_idx)


def _make_matcher(method: str, cfg: MatcherCfg):
    if cfg.matcher == "flann":
        if method == "sift":   # float descriptors -> randomised KD-trees
            return cv2.FlannBasedMatcher(dict(algorithm=1, trees=cfg.flann_trees),
                                         dict(checks=cfg.flann_checks))
        # binary descriptors -> locality-sensitive hashing
        return cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
                                     dict(checks=cfg.flann_checks))
    if cfg.matcher != "bf":
        raise ValueError("matcher must be 'bf' or 'flann'")
    return cv2.BFMatcher(cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING)


def match_features(fs_src: FeatureSet, fs_ref: FeatureSet, cfg: MatcherCfg) -> MatchResult:
    """Query = source, train = reference.

    1. k=2 nearest neighbours in descriptor space.
    2. Lowe ratio test: keep if d1 < ratio * d2 (best clearly beats second best,
       otherwise the match is ambiguous - typical for repetitive texture).
    3. Optional mutual check (reference's best match points back to the same source kp).
    4. Uniqueness: if several source kps pick one reference kp, keep the closest.
    """
    if fs_src.method != fs_ref.method:
        raise ValueError("descriptor types differ; cannot match SIFT against ORB")
    method = fs_src.method
    ratio = cfg.ratio_for(method)
    matcher = _make_matcher(method, cfg)
    knn = matcher.knnMatch(fs_src.descriptors, fs_ref.descriptors, k=2)

    best = [p[0] for p in knn if len(p) >= 1]
    passed, failed = [], []
    for p in knn:
        if len(p) < 2:                    # cannot apply ratio test -> ambiguous
            if p:
                failed.append(p[0])
            continue
        (passed if p[0].distance < ratio * p[1].distance else failed).append(p[0])
    n_ratio = len(passed)

    if cfg.mutual and passed:
        rev = cv2.BFMatcher(cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING) \
            .match(fs_ref.descriptors, fs_src.descriptors)
        back = {m.queryIdx: m.trainIdx for m in rev}
        keep = []
        for m in passed:
            (keep if back.get(m.trainIdx) == m.queryIdx else failed).append(m)
        passed = keep
    n_mutual = len(passed)

    uniq = {}
    for m in passed:
        if m.trainIdx not in uniq or m.distance < uniq[m.trainIdx].distance:
            uniq[m.trainIdx] = m
    dup = [m for m in passed if uniq[m.trainIdx] is not m]
    failed.extend(dup)
    acc = sorted(uniq.values(), key=lambda m: m.queryIdx)

    sxy, rxy = fs_src.xy, fs_ref.xy
    stats = dict(method=method, matcher=cfg.matcher, ratio=ratio, mutual=cfg.mutual,
                 n_keypoints_src=len(fs_src), n_keypoints_ref=len(fs_ref),
                 n_candidates=len(best), n_after_ratio=n_ratio, n_after_mutual=n_mutual,
                 n_accepted=len(acc), n_rejected=len(failed),
                 accepted_fraction=len(acc) / max(1, len(best)))
    log.info("match[%s/%s]: %d candidates -> %d accepted", method, cfg.matcher, len(best), len(acc))
    if len(acc) < cfg.min_matches:
        raise InsufficientMatchesError(
            f"only {len(acc)} matches survived (need >= {cfg.min_matches}); stats={stats}", stats)

    q = np.array([m.queryIdx for m in acc]); t = np.array([m.trainIdx for m in acc])
    return MatchResult(
        method, q, t, sxy[q], rxy[t], np.array([m.distance for m in acc], np.float32),
        sxy[[m.queryIdx for m in failed]] if failed else np.zeros((0, 2)),
        rxy[[m.trainIdx for m in failed]] if failed else np.zeros((0, 2)), stats)


def draw_match_figures(gray_s, gray_r, mr: MatchResult, out_dir, name: str, max_lines: int = 250) -> dict:
    """Two images: all initial candidates (grey), and accepted (green) vs rejected (red)."""
    out_dir = Path(out_dir)
    paths = {}
    allp_s = np.vstack([mr.pts_src, mr.rejected_src])
    allp_r = np.vstack([mr.pts_ref, mr.rejected_ref])
    paths["initial"] = out_dir / f"{name}_matches_initial.png"
    draw_correspondences(gray_s, gray_r, [dict(pts_s=allp_s, pts_r=allp_r, color=(190, 190, 190),
                                              label="initial candidates")], paths["initial"], max_lines)
    paths["accepted_vs_rejected"] = out_dir / f"{name}_matches_accepted_vs_rejected.png"
    draw_correspondences(gray_s, gray_r, [
        dict(pts_s=mr.rejected_src, pts_r=mr.rejected_ref, color=(60, 60, 230), label="rejected"),
        dict(pts_s=mr.pts_src, pts_r=mr.pts_ref, color=(60, 200, 60), label="accepted")],
        paths["accepted_vs_rejected"], max_lines)
    return paths
