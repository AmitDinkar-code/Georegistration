import numpy as np
import pytest
from dataclasses import replace

from georeg.config import DetectorCfg, MatcherCfg
from georeg.errors import InsufficientFeaturesError, InsufficientMatchesError
from georeg.feature_detector import detect_features, keypoint_statistics
from georeg.feature_matcher import match_features
from georeg.transformation_estimator import apply_transform, reprojection_errors


def test_sift_descriptor_shapes(grays):
    src, ref, mask = grays
    fs = detect_features(ref, DetectorCfg("sift"))
    assert len(fs) > 200
    assert fs.descriptors.shape == (len(fs), 128) and fs.descriptors.dtype == np.float32
    assert fs.xy.shape == (len(fs), 2) and (fs.size > 0).all()
    assert ((fs.angle >= 0) & (fs.angle < 360)).all()


def test_orb_descriptor_shapes(grays):
    _, ref, _ = grays
    fs = detect_features(ref, DetectorCfg("orb"))
    assert fs.descriptors.shape[1] == 32 and fs.descriptors.dtype == np.uint8
    assert keypoint_statistics(fs)["grid_occupancy"] > 0.3


def test_blank_image_raises():
    with pytest.raises(InsufficientFeaturesError):
        detect_features(np.full((200, 200), 128, np.uint8), DetectorCfg("sift"))


def test_self_match_is_identity(grays):
    _, ref, _ = grays
    fs = detect_features(ref, DetectorCfg("sift"))
    mr = match_features(fs, fs, MatcherCfg())
    assert len(mr) > 100
    assert np.median(np.linalg.norm(mr.pts_src - mr.pts_ref, axis=1)) == 0


@pytest.mark.parametrize("method", ["sift", "orb"])
@pytest.mark.parametrize("matcher", ["bf", "flann"])
def test_matches_agree_with_ground_truth(grays, case, method, matcher):
    src, ref, mask = grays
    d = DetectorCfg(method)
    mr = match_features(detect_features(src, d, mask), detect_features(ref, d), MatcherCfg(matcher=matcher))
    err = reprojection_errors(case.T_true, mr.pts_src, mr.pts_ref)
    assert len(mr) >= 30
    assert (err < 3.0).mean() > 0.6, "most accepted matches must be geometrically right"


def test_ratio_test_is_monotonic(grays):
    src, ref, mask = grays
    d = DetectorCfg("sift")
    fs, fr = detect_features(src, d, mask), detect_features(ref, d)
    n_strict = len(match_features(fs, fr, MatcherCfg(ratio_sift=0.6)))
    n_loose = len(match_features(fs, fr, MatcherCfg(ratio_sift=0.9)))
    assert n_strict < n_loose


def test_ratio_test_improves_precision(grays, case):
    src, ref, mask = grays
    d = DetectorCfg("sift")
    fs, fr = detect_features(src, d, mask), detect_features(ref, d)
    def prec(r):
        mr = match_features(fs, fr, MatcherCfg(ratio_sift=r, min_matches=5))
        return (reprojection_errors(case.T_true, mr.pts_src, mr.pts_ref) < 3).mean()
    assert prec(0.7) >= prec(0.99)


def test_insufficient_matches_raises(grays):
    src, ref, mask = grays
    d = DetectorCfg("sift")
    fs, fr = detect_features(src, d, mask), detect_features(ref, d)
    with pytest.raises(InsufficientMatchesError):
        match_features(fs, fr, MatcherCfg(min_matches=10 ** 6))


def test_mixed_descriptor_types_rejected(grays):
    _, ref, _ = grays
    with pytest.raises(ValueError):
        match_features(detect_features(ref, DetectorCfg("sift")), detect_features(ref, DetectorCfg("orb")), MatcherCfg())
