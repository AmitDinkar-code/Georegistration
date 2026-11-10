"""Phase 1: controlled, reproducible test cases with exact ground truth.

Convention (used everywhere):
    M  : reference pixel -> source pixel   (how the test image was made)
    T  : source pixel -> reference pixel   (what registration must recover)
    T_true = inv(M)
Pixel coordinates follow OpenCV: the centre of pixel (0,0) is (0,0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from affine import Affine
from scipy.ndimage import gaussian_filter

from .geotiff_writer import write_geotiff


# ---------------------------------------------------------------- scene -----
def procedural_scene(hw, seed: int = 7) -> np.ndarray:
    """(3,H,W) uint8 aerial-like scene: multi-scale terrain texture, fields,
    roads, buildings and circular tanks. Rich in corners, no zeros (0 = nodata)."""
    H, W = (hw, hw) if isinstance(hw, int) else hw
    rng = np.random.default_rng(seed)

    def field_(sigma):
        n = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma)
        return n / (n.std() + 1e-6)

    low = 0.5 * field_(40) + 0.4 * field_(14) + 0.3 * field_(5) + 0.15 * field_(1.5)
    img = np.stack([118 + 30 * (0.8 * low + 0.5 * field_(30)),
                    126 + 28 * (0.8 * low + 0.5 * field_(30)),
                    96 + 24 * (0.7 * low + 0.5 * field_(30))], axis=-1)
    img = np.clip(img, 0, 255).astype(np.uint8)

    palette = [(70, 120, 60), (150, 135, 90), (95, 150, 80), (170, 150, 110),
               (60, 95, 50), (185, 170, 130), (120, 110, 80)]
    # fields
    overlay = img.copy()
    for _ in range(max(10, H * W // 18000)):
        w, h = rng.integers(50, 220, 2)
        x, y = rng.integers(0, max(1, W - 10)), rng.integers(0, max(1, H - 10))
        cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)),
                      palette[rng.integers(len(palette))], -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    # roads
    for _ in range(max(6, H // 90)):
        pts = np.stack([rng.integers(0, W, 3), rng.integers(0, H, 3)], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, (95, 92, 88), int(rng.integers(4, 7)), cv2.LINE_AA)
        cv2.polylines(img, [pts], False, (205, 200, 190), int(rng.integers(2, 4)), cv2.LINE_AA)
    # buildings
    for _ in range(max(80, H * W // 7000)):
        x, y = rng.integers(0, W - 25), rng.integers(0, H - 25)
        w, h = rng.integers(6, 22, 2)
        col = [(70, 70, 150), (200, 200, 205), (90, 90, 95), (120, 140, 190)][rng.integers(4)]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, -1)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (30, 30, 30), 1)
    # tanks
    for _ in range(max(12, H * W // 40000)):
        c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        r = int(rng.integers(5, 26))
        cv2.circle(img, c, r, tuple(int(v) for v in rng.integers(90, 240, 3)), -1, cv2.LINE_AA)
        cv2.circle(img, c, r, (25, 25, 25), 1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.6).astype(np.float32)
    img += rng.normal(0, 2.5, img.shape)
    img = np.clip(img, 1, 255).astype(np.uint8)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def make_reference_geotiff(path, size: int = 1024, seed: int = 7, crs: str = "EPSG:32643",
                           origin=(480000.0, 2570000.0), pixel_size: float = 10.0) -> Affine:
    """Write a synthetic georeferenced reference (UTM 43N, 10 m px by default)."""
    scene = procedural_scene(size, seed)
    tr = Affine(pixel_size, 0, origin[0], 0, -pixel_size, origin[1])
    write_geotiff(path, scene, tr, crs, valid=None, nodata=0,
                  tags=dict(SYNTHETIC="1", SEED=seed))
    return tr


# ------------------------------------------------------------- matrices -----
def _translate(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)


def build_matrix(ref_hw, out_hw, rotation_deg=0.0, scale=1.0, shear=(0.0, 0.0),
                 translation=(0.0, 0.0), perspective=(0.0, 0.0)) -> np.ndarray:
    """M: reference px -> source px.
    M = T_out . P . Shear . R . S . T_in  (all about the image centres).
    scale > 1 means the source has FINER pixels (more px per ground distance)."""
    (Hr, Wr), (Ho, Wo) = ref_hw, out_hw
    th = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    S = np.diag([scale, scale, 1.0])
    Sh = np.array([[1, shear[0], 0], [shear[1], 1, 0], [0, 0, 1]], float)
    P = np.array([[1, 0, 0], [0, 1, 0], [perspective[0], perspective[1], 1]], float)
    Tin = _translate(-(Wr - 1) / 2, -(Hr - 1) / 2)
    Tout = _translate((Wo - 1) / 2 + translation[0], (Ho - 1) / 2 + translation[1])
    M = Tout @ P @ Sh @ R @ S @ Tin
    return M / M[2, 2]


def invert(M: np.ndarray) -> np.ndarray:
    T = np.linalg.inv(M)
    return T / T[2, 2]


@dataclass
class SyntheticCase:
    data: np.ndarray            # (C,H,W) source image, same dtype as the reference
    valid: np.ndarray           # (H,W) bool
    M_ref_to_src: np.ndarray
    T_true: np.ndarray          # src -> ref  (ground truth)
    params: dict = field(default_factory=dict)


def make_source(ref_data: np.ndarray, M: np.ndarray, out_hw, gain: float = 1.0, bias: float = 0.0,
                noise_sigma: float = 0.0, occlusion: float = 0.0, interpolation=cv2.INTER_LINEAR,
                seed: int = 0, params: dict | None = None) -> SyntheticCase:
    """Resample the reference through M, then degrade radiometrically."""
    C, H, W = ref_data.shape
    Ho, Wo = out_hw
    rng = np.random.default_rng(seed)
    s = float(np.sqrt(abs(np.linalg.det(M[:2, :2] / M[2, 2]))))
    bands = []
    for c in range(C):
        b = ref_data[c]
        if s < 1.0:  # anti-alias before decimating, like a real coarser sensor
            b = cv2.GaussianBlur(b, (0, 0), 0.5 / s)
        bands.append(cv2.warpPerspective(b, M, (Wo, Ho), flags=interpolation,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    arr = np.stack(bands).astype(np.float32)
    valid = cv2.warpPerspective(np.full((H, W), 255, np.uint8), M, (Wo, Ho),
                                flags=cv2.INTER_NEAREST) > 127
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    arr = arr * gain + bias
    if noise_sigma > 0:
        arr += rng.normal(0, noise_sigma, arr.shape).astype(np.float32)
    if occlusion > 0:  # opaque 'cloud' blobs until the requested footprint fraction is covered
        occ = np.zeros((Ho, Wo), np.uint8)
        target = occlusion * valid.sum()
        guard = 0
        while occ[valid > 0].sum() / 255.0 < target and guard < 200:
            cx, cy = int(rng.integers(0, Wo)), int(rng.integers(0, Ho))
            ax, ay = int(rng.integers(Wo // 20 + 5, Wo // 4 + 6)), int(rng.integers(Ho // 20 + 5, Ho // 4 + 6))
            cv2.ellipse(occ, (cx, cy), (ax, ay), float(rng.integers(0, 180)), 0, 360, 255, -1)
            guard += 1
        arr[:, occ > 0] = 235.0
    info = np.iinfo(ref_data.dtype) if ref_data.dtype.kind in "ui" else None
    if info is not None:
        arr = np.clip(arr, max(info.min, 1), info.max)
    out = arr.astype(ref_data.dtype)
    out[:, ~valid] = 0
    return SyntheticCase(out, valid, M, invert(M), params or {})


def make_case(ref_data: np.ndarray, rotation_deg=0.0, scale=1.0, shear=(0.0, 0.0),
              translation=(0.0, 0.0), perspective=(0.0, 0.0), crop=1.0,
              gain=1.0, bias=0.0, noise_sigma=0.0, occlusion=0.0, seed=0) -> SyntheticCase:
    """Convenience: footprint = crop * (scale * reference size), centred + translated."""
    _, H, W = ref_data.shape
    out_hw = (max(64, int(round(H * scale * crop))), max(64, int(round(W * scale * crop))))
    M = build_matrix((H, W), out_hw, rotation_deg, scale, shear, translation, perspective)
    params = dict(rotation_deg=rotation_deg, scale=scale, shear=shear, translation=translation,
                  perspective=perspective, crop=crop, gain=gain, bias=bias,
                  noise_sigma=noise_sigma, occlusion=occlusion, seed=seed, out_hw=out_hw)
    return make_source(ref_data, M, out_hw, gain, bias, noise_sigma, occlusion, seed=seed, params=params)


PRESETS = {
    "rotation": dict(rotation_deg=25.0, crop=0.8),
    "translation": dict(translation=(60.0, -40.0), crop=0.7),
    "scale": dict(scale=0.6, crop=0.9),
    "affine": dict(rotation_deg=12.0, scale=0.85, shear=(0.06, -0.03), translation=(20.0, 10.0), crop=0.75),
    "perspective": dict(rotation_deg=8.0, scale=0.9, perspective=(1.2e-4, -0.8e-4), crop=0.75),
    "demo": dict(rotation_deg=20.0, scale=0.8, shear=(0.02, 0.0), translation=(30.0, -20.0),
                 gain=1.1, bias=-8.0, noise_sigma=4.0, crop=0.75),
}
