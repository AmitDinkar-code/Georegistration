import math
from dataclasses import replace

import numpy as np
import pytest
from affine import Affine

from georeg.config import QualityGates, RansacCfg
from georeg.synthetic_transform import build_matrix, invert
from georeg.transformation_estimator import (apply_transform, check_plausibility, compare_models, decompose,
                                             est_ransac_iterations, estimate_transform, ground_truth_error,
                                             holdout_report, select_model, split_holdout)


def _points(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 800, (n, 2))


def _with_outliers(H, frac=0.4, noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    s = _points(400, seed)
    r = apply_transform(H, s) + rng.normal(0, noise, s.shape)
    bad = rng.random(len(s)) < frac
    r[bad] = rng.uniform(0, 800, (bad.sum(), 2))
    return s, r, ~bad


def test_ransac_recovers_homography_with_40pct_outliers():
    H = build_matrix((800, 800), (800, 800), rotation_deg=12, scale=0.9, perspective=(1e-4, -5e-5), translation=(20, 5))
    s, r, good = _with_outliers(H)
    est = estimate_transform(s, r, RansacCfg(reproj_thresh=2.0), "homography")
    grid = _points(500, 9)
    assert np.abs(apply_transform(est.matrix, grid) - apply_transform(H, grid)).max() < 1.5
    flagged = est.inlier_mask
    assert (flagged & good).sum() / good.sum() > 0.9          # recall
    assert (flagged & good).sum() / flagged.sum() > 0.95      # precision


def test_affine_exact_data_has_near_zero_residual():
    A = build_matrix((800, 800), (800, 800), rotation_deg=30, scale=1.2, shear=(0.05, 0.0))
    s = _points()
    est = estimate_transform(s, apply_transform(A, s), RansacCfg(), "affine")
    assert est.rmse_inliers < 1e-6 and est.n_inliers == len(s)


def test_similarity_dof_and_scale():
    S = build_matrix((800, 800), (800, 800), rotation_deg=20, scale=0.7)
    s = _points()
    est = estimate_transform(s, apply_transform(S, s), RansacCfg(), "similarity")
    d = est.decomposition
    assert est.dof == 4 and math.isclose(d["scale_x"], 0.7, rel_tol=1e-6) and math.isclose(d["rotation_deg"], 20, abs_tol=1e-6)


def test_degrees_of_freedom_table():
    s = _points(50)
    ests = compare_models(s, s, RansacCfg(), ("similarity", "affine", "homography"))
    assert [ests[k].dof for k in ("similarity", "affine", "homography")] == [4, 6, 8]


def test_model_selection_prefers_fewer_dof_when_equal():
    A = build_matrix((800, 800), (800, 800), rotation_deg=10, scale=1.1)
    s = _points()
    ests = compare_models(s, apply_transform(A, s), RansacCfg(), ("affine", "homography", "similarity"))
    assert select_model(ests, 0.02).model == "similarity"      # data is exactly a similarity


def test_model_selection_picks_homography_for_perspective():
    H = build_matrix((800, 800), (800, 800), rotation_deg=5, perspective=(4e-4, 2e-4))
    s = _points()
    ests = compare_models(s, apply_transform(H, s), RansacCfg(reproj_thresh=1.0), ("affine", "homography"))
    assert select_model(ests, 0.02).model == "homography"


def test_direction_convention():
    """M: ref->src, T_true: src->ref. Applying M then T_true returns the start."""
    M = build_matrix((600, 700), (400, 500), rotation_deg=33, scale=0.6, translation=(12, -7))
    T = invert(M)
    p = _points(50)
    assert np.allclose(apply_transform(T, apply_transform(M, p)), p, atol=1e-8)


def test_ransac_iteration_formula():
    # w=0.5, s=4, p=0.99  ->  ln(0.01)/ln(1-0.0625) = 71.2
    assert est_ransac_iterations(0.5, 4, 0.99) == pytest.approx(71.2, abs=0.2)
    assert est_ransac_iterations(0.5, 3, 0.99) < est_ransac_iterations(0.5, 4, 0.99)


def test_plausibility_rejects_bad_models():
    g = QualityGates()
    assert check_plausibility(np.eye(3), (500, 500), g) == []
    flip = np.diag([-1.0, 1, 1]); flip[0, 2] = 500
    assert any("reflection" in f for f in check_plausibility(flip, (500, 500), g))
    assert check_plausibility(np.diag([1e-4, 1e-4, 1.0]), (500, 500), g)
    fold = np.eye(3); fold[2, 0] = 0.01     # corner crosses the horizon
    assert check_plausibility(fold, (500, 500), g)
    assert check_plausibility(np.zeros((3, 3)), (500, 500), g)


def test_ground_truth_error_zero_and_map_units():
    T = build_matrix((500, 500), (500, 500), rotation_deg=5)
    assert ground_truth_error(T, T, (500, 500))["rmse_px"] == 0
    shifted = T.copy(); shifted[0, 2] += 2.0           # 2 reference px east
    tr = Affine(10, 0, 5e5, 0, -10, 2.5e6)
    g = ground_truth_error(shifted, T, (500, 500), tr)
    assert g["rmse_px"] == pytest.approx(2.0, abs=1e-6)
    assert g["rmse_map"] == pytest.approx(20.0, abs=1e-6)  # 2 px * 10 m


def test_holdout_split_is_disjoint_and_deterministic():
    fit, hold = split_holdout(200, 0.2, seed=3)
    assert len(hold) == 40 and not set(fit) & set(hold)
    assert np.array_equal(hold, split_holdout(200, 0.2, seed=3)[1])
    fit2, hold2 = split_holdout(30, 0.2)           # too few matches -> no checkpoints
    assert len(hold2) == 0 and len(fit2) == 30


def test_holdout_report_flags_inconsistent_points():
    s = _points(60); r = s.copy(); r[:10] += 50
    rep = holdout_report(np.eye(3), s, r, 3.0)
    assert rep["n_consistent"] == 50 and rep["rmse_consistent_px"] == 0


def test_decompose_identity():
    d = decompose(np.eye(3))
    assert d["scale_x"] == 1 and d["rotation_deg"] == 0 and d["determinant"] == 1
