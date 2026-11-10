"""Phase 2: keypoint detection + descriptor extraction (SIFT primary, ORB alternative)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorCfg
from .errors import InsufficientFeaturesError
from .io_utils import imwrite_unicode

log = logging.getLogger("georeg")


@dataclass
class FeatureSet:
    method: str
    keypoints: tuple
    descriptors: np.ndarray
    image_shape: tuple

    def __len__(self) -> int:
        return len(self.keypoints)

    @property
    def xy(self) -> np.ndarray:
        return np.array([k.pt for k in self.keypoints], np.float64).reshape(-1, 2)

    @property
    def size(self) -> np.ndarray:        # diameter of the meaningful neighbourhood = scale
        return np.array([k.size for k in self.keypoints], np.float32)

    @property
    def angle(self) -> np.ndarray:       # dominant orientation, degrees
        return np.array([k.angle for k in self.keypoints], np.float32)

    @property
    def response(self) -> np.ndarray:
        return np.array([k.response for k in self.keypoints], np.float32)


def _make_detector(cfg: DetectorCfg):
    if cfg.method == "sift":
        return cv2.SIFT_create(nfeatures=cfg.max_features, contrastThreshold=cfg.sift_contrast,
                               edgeThreshold=cfg.sift_edge, sigma=cfg.sift_sigma)
    if cfg.method == "orb":
        return cv2.ORB_create(nfeatures=cfg.max_features, scaleFactor=cfg.orb_scale_factor,
                              nlevels=cfg.orb_levels, edgeThreshold=cfg.orb_edge_threshold,
                              patchSize=cfg.orb_patch_size, fastThreshold=cfg.orb_fast_threshold)
    raise ValueError(f"unknown detector '{cfg.method}' (use 'sift' or 'orb')")


def detect_features(gray: np.ndarray, cfg: DetectorCfg, mask: np.ndarray | None = None) -> FeatureSet:
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("detect_features expects a 2-D uint8 image (use dataset_loader.prepare_gray)")
    det = _make_detector(cfg)
    kps, desc = det.detectAndCompute(gray, mask)
    if desc is None or len(kps) < cfg.min_features:
        raise InsufficientFeaturesError(
            f"{cfg.method.upper()} found {0 if desc is None else len(kps)} keypoints "
            f"(< {cfg.min_features}); image is blank / extremely low-texture")
    log.info("%s: %d keypoints", cfg.method.upper(), len(kps))
    return FeatureSet(cfg.method, tuple(kps), desc, gray.shape)


def keypoint_statistics(fs: FeatureSet, grid: int = 8) -> dict:
    """Count, scale/orientation spread and spatial uniformity (occupied grid cells)."""
    xy = fs.xy
    h, w = fs.image_shape
    gx = np.clip((xy[:, 0] / w * grid).astype(int), 0, grid - 1)
    gy = np.clip((xy[:, 1] / h * grid).astype(int), 0, grid - 1)
    occupied = len(set(zip(gx.tolist(), gy.tolist())))
    s = fs.size
    return dict(method=fs.method, count=len(fs), descriptor_shape=tuple(fs.descriptors.shape),
                descriptor_dtype=str(fs.descriptors.dtype),
                scale_min=float(s.min()), scale_median=float(np.median(s)), scale_max=float(s.max()),
                grid_occupancy=occupied / float(grid * grid))


def draw_keypoints(gray: np.ndarray, fs: FeatureSet, path, max_draw: int = 1500):
    """Circles = scale, radius line = orientation (DRAW_RICH_KEYPOINTS)."""
    order = np.argsort(-fs.response)[:max_draw]
    kps = [fs.keypoints[i] for i in order]
    vis = cv2.drawKeypoints(gray, kps, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(vis, f"{fs.method.upper()}: {len(fs)} keypoints (top {len(kps)} drawn)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    imwrite_unicode(path, vis)
    return vis
