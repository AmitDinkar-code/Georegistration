"""Configuration objects. Everything tunable lives here (frozen dataclasses, so
a config can be hashed, logged and compared)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class PreprocessCfg:
    lower_pct: float = 2.0      # percentile stretch (computed on valid pixels only)
    upper_pct: float = 98.0
    clahe: bool = False         # local contrast equalisation; helps low-contrast scenes
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    mask_erode_px: int = 3      # keep keypoints away from nodata borders


@dataclass(frozen=True)
class DetectorCfg:
    method: str = "sift"        # "sift" | "orb"
    max_features: int = 8000
    min_features: int = 20
    sift_contrast: float = 0.02  # OpenCV default 0.04; lower = more features in flat areas
    sift_edge: float = 10.0
    sift_sigma: float = 1.6
    orb_scale_factor: float = 1.2
    orb_levels: int = 8
    orb_fast_threshold: int = 10
    orb_edge_threshold: int = 31
    orb_patch_size: int = 31


@dataclass(frozen=True)
class MatcherCfg:
    matcher: str = "bf"         # "bf" | "flann"
    ratio_sift: float = 0.75    # Lowe ratio test
    ratio_orb: float = 0.80
    mutual: bool = False        # additionally require reciprocal nearest neighbour
    min_matches: int = 12
    flann_trees: int = 5
    flann_checks: int = 64

    def ratio_for(self, method: str) -> float:
        return self.ratio_sift if method == "sift" else self.ratio_orb


@dataclass(frozen=True)
class RansacCfg:
    method: str = "ransac"      # "ransac" | "magsac" (homography only)
    reproj_thresh: float = 3.0  # px in the REFERENCE grid
    confidence: float = 0.999
    max_iters: int = 5000
    holdout_fraction: float = 0.2   # matches withheld from the fit as checkpoints
    refine: bool = True         # least-squares refit on the inliers
    seed: int = 0
    model_select_tol: float = 0.02  # prefer fewer DoF unless >2 % more inliers


@dataclass(frozen=True)
class WarpCfg:
    interpolation: str = "bilinear"   # "nearest" | "bilinear" | "cubic"
    compare_interpolations: bool = True


@dataclass(frozen=True)
class QualityGates:
    min_inliers: int = 15
    min_inlier_ratio: float = 0.15
    max_inlier_rmse_px: float = 2.0
    min_coverage: float = 0.05        # convex hull of inliers / source area
    max_scale: float = 20.0           # accepted scale factor between 1/x and x
    max_perspective: float = 0.5      # max |w-1| over the source corners
    allow_reflection: bool = False
    min_phase_response: float = 0.20  # phase-correlation fallback
    min_fallback_ncc: float = 0.35


@dataclass(frozen=True)
class PipelineCfg:
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    matcher: MatcherCfg = field(default_factory=MatcherCfg)
    ransac: RansacCfg = field(default_factory=RansacCfg)
    warp: WarpCfg = field(default_factory=WarpCfg)
    gates: QualityGates = field(default_factory=QualityGates)
    # ordered fallback chain: (detector, candidate models)
    attempts: tuple = (
        ("sift", ("affine", "homography")),
        ("orb", ("affine", "homography")),
    )
    phase_fallback: bool = True
    bootstrap_runs: int = 30
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
