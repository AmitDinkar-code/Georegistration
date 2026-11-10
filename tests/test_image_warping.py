import cv2
import numpy as np
import pytest

from georeg.image_warping import compare_interpolations, roundtrip_psnr, warp_to_reference


def _smooth_image(h=200, w=240, seed=0):
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 4)
    n = (n - n.min()) / (n.max() - n.min())
    return (1 + n * 250).astype(np.uint8)[None]


def test_integer_translation_is_exact_with_nearest():
    img = _smooth_image()
    T = np.array([[1, 0, 5], [0, 1, 3], [0, 0, 1.0]])
    w = warp_to_reference(img, T, img.shape[1:], "nearest")
    assert np.array_equal(w.data[0, 3:, 5:], img[0, :-3, :-5])
    assert not w.valid[:3].any() and not w.valid[:, :5].any()
    assert (w.data[0][~w.valid] == 0).all()


def test_direction_point_lands_at_T_of_p():
    """A bright pixel at p in the source must appear at T(p) in the output."""
    img = np.full((1, 120, 160), 20, np.uint8)
    img[0, 40, 30] = 255
    T = np.array([[1.5, 0.2, 10], [-0.1, 1.3, 5], [0, 0, 1.0]])
    w = warp_to_reference(img, T, (200, 260), "nearest")
    ey, ex = np.unravel_index(np.argmax(w.data[0]), w.data[0].shape)
    px = T @ np.array([30, 40, 1.0])
    assert abs(ex - px[0]) <= 1.0 and abs(ey - px[1]) <= 1.0


def test_output_matches_reference_grid_size():
    img = _smooth_image()
    w = warp_to_reference(img, np.eye(3), (333, 444), "bilinear")
    assert w.data.shape == (1, 333, 444) and w.valid.shape == (333, 444)


@pytest.mark.parametrize("interp", ["nearest", "bilinear", "cubic"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_dtype_preserved_and_cubic_does_not_wrap(interp, dtype):
    img = (_smooth_image().astype(np.float32) * (200 if dtype == np.uint16 else 1)).astype(dtype)
    T = cv2.getRotationMatrix2D((100, 100), 20, 1.1); T = np.vstack([T, [0, 0, 1]])
    w = warp_to_reference(img, T, img.shape[1:], interp)
    assert w.data.dtype == dtype
    if dtype != np.float32:
        assert w.data[0][w.valid].max() <= np.iinfo(dtype).max


def test_multiband_independent_bands():
    img = np.concatenate([_smooth_image(seed=1), _smooth_image(seed=2), _smooth_image(seed=3),
                          _smooth_image(seed=4), _smooth_image(seed=5)])   # 5 bands (cv2 caps at 4 channels)
    w = warp_to_reference(img, np.eye(3), img.shape[1:], "bilinear")
    assert w.data.shape[0] == 5
    assert np.abs(w.data[4].astype(int) - img[4].astype(int))[w.valid].max() == 0


def test_source_nodata_propagates_to_mask():
    img = _smooth_image(); valid = np.ones(img.shape[1:], bool); valid[50:80, 60:90] = False
    w = warp_to_reference(img, np.eye(3), img.shape[1:], "nearest", valid)
    assert not w.valid[60, 70] and w.valid[10, 10]


def test_roundtrip_psnr_ordering_for_rotation():
    img = _smooth_image()
    T = np.vstack([cv2.getRotationMatrix2D((120, 100), 17, 1.0), [0, 0, 1]])
    nn = roundtrip_psnr(img, T, img.shape[1:], "nearest")
    bl = roundtrip_psnr(img, T, img.shape[1:], "bilinear")
    cu = roundtrip_psnr(img, T, img.shape[1:], "cubic")
    assert bl > nn and cu > nn


def test_compare_interpolations_nearest_preserves_values():
    img = _smooth_image()
    T = np.vstack([cv2.getRotationMatrix2D((120, 100), 9, 1.0), [0, 0, 1]])
    rows, _ = compare_interpolations(img, T, img.shape[1:], img[0])
    by = {r["interpolation"]: r for r in rows}
    assert by["nearest"]["value_preserved"] == 1.0
    assert by["bilinear"]["value_preserved"] < 1.0
