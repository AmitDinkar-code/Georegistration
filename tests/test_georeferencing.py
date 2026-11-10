import numpy as np
import pytest
import rasterio
from affine import Affine

from georeg.config import PipelineCfg
from georeg.control_points import (build_gcps, composite_pixel_to_map, map_to_pixel, pixel_to_map)
from georeg.dataset_loader import load_raster
from georeg.errors import InputError
from georeg.geotiff_writer import write_geotiff
from georeg.pipeline import run_registration
from georeg.synthetic_transform import make_case, make_reference_geotiff
from georeg.verify_metadata import verify_with_gdal, verify_with_rasterio

TR = Affine(10.0, 0, 480000.0, 0, -10.0, 2570000.0)


def test_pixel_centre_convention():
    # centre of OpenCV pixel (0,0) is half a pixel inside the upper-left corner
    x, y = pixel_to_map(TR, 0, 0)
    assert (x, y) == (480005.0, 2569995.0)
    c, r = map_to_pixel(TR, x, y)
    assert abs(c) < 1e-9 and abs(r) < 1e-9


def test_composite_matrix_equals_two_step():
    T = np.array([[1.1, 0.1, 5], [-0.05, 0.9, 7], [1e-5, 0, 1]])
    p = np.array([120.0, 80.0])
    q = T @ [*p, 1]; q = q[:2] / q[2]
    mx, my = pixel_to_map(TR, *q)
    C = composite_pixel_to_map(TR, T) @ [*p, 1]
    assert np.allclose(C[:2] / C[2], [mx, my])


def test_gcp_map_coordinates():
    s = np.array([[1.0, 2.0], [3.0, 4.0]]); r = np.array([[10.0, 20.0], [30.0, 40.0]])
    g = build_gcps(s, r, [0.1, 0.2], ["inlier", "outlier"], TR)
    assert len(g) == 1 and (g[0].map_x, g[0].map_y) == (480105.0, 2569795.0)


def test_geotiff_roundtrip_preserves_values_nodata_and_mask(tmp_path):
    data = np.random.default_rng(0).integers(1, 255, (3, 40, 50), dtype=np.uint8)
    valid = np.ones((40, 50), bool); valid[:10] = False
    p = tmp_path / "t.tif"
    info = write_geotiff(p, data, TR, "EPSG:32643", valid=valid, nodata=0, tags={"K": "V"})
    with rasterio.open(p) as ds:
        assert ds.crs.to_epsg() == 32643 and ds.transform == TR and ds.nodata == 0
        assert ds.dtypes[0] == "uint8" and ds.tags()["K"] == "V"
        back = ds.read()
        assert np.array_equal(back[:, valid], data[:, valid]) and (back[:, ~valid] == 0).all()
        assert np.array_equal(ds.dataset_mask() > 0, valid)
    assert info["mask_written"]


def test_float_nodata_defaults_to_nan(tmp_path):
    p = tmp_path / "f.tif"
    write_geotiff(p, np.ones((1, 8, 8), np.float32), TR, "EPSG:32643", valid=np.eye(8, dtype=bool))
    with rasterio.open(p) as ds:
        assert np.isnan(ds.nodata) and np.isnan(ds.read()[0][0, 1])


def test_missing_reference_georeferencing_is_rejected(tmp_path, scene):
    ref, src = tmp_path / "ref.tif", tmp_path / "src.tif"
    write_geotiff(ref, scene, None, None)                  # NOT georeferenced
    write_geotiff(src, scene, None, None)
    with pytest.raises(InputError):
        run_registration(src, ref, tmp_path / "o")


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    ref_path = d / "ref.tif"
    make_reference_geotiff(ref_path, 512, seed=7)
    ref = load_raster(ref_path)
    case = make_case(ref.data, rotation_deg=15, scale=0.85, translation=(8, -6), crop=0.8, gain=1.1, bias=-5,
                     noise_sigma=3, seed=2)
    src_path = d / "src.tif"
    write_geotiff(src_path, case.data, None, None, valid=case.valid, nodata=0)
    cfg = PipelineCfg(attempts=(("sift", ("affine", "homography")),), phase_fallback=False, bootstrap_runs=8)
    res = run_registration(src_path, ref_path, d / "out", cfg, run_name="t", T_true=case.T_true)
    return d, ref_path, src_path, case, res


def test_end_to_end_accuracy_against_ground_truth(e2e):
    *_, res = e2e
    assert res.success
    gt = res.report["ground_truth"]
    assert gt["rmse_px"] < 1.5, gt["rmse_px"]
    assert gt["rmse_map"] == pytest.approx(gt["rmse_px"] * 10.0, rel=1e-6)   # 10 m pixels


def test_registration_beats_naive_stamping(e2e):
    *_, res = e2e
    assert res.report["naive_stamp_ground_truth"]["rmse_px"] > 20 * res.report["ground_truth"]["rmse_px"]
    s = res.report["image_similarity"]
    assert s["gradient_ncc_after"] > s["gradient_ncc_before"] + 0.3


def test_output_geotiff_lives_on_reference_grid(e2e):
    d, ref_path, src_path, case, res = e2e
    checks = verify_with_rasterio(res.outputs["geotiff"], ref_path)
    failed = {k: v for k, v in checks.items() if v[0] is False}
    assert not failed, failed
    with rasterio.open(src_path) as s, rasterio.open(res.outputs["geotiff"]) as o:
        assert o.shape != s.shape                     # resampled to the reference grid, not copied
        assert o.dtypes == s.dtypes                   # radiometric type preserved
        assert o.tags()["ORTHORECTIFIED"] == "NO"


def test_holdout_and_consistency_reported(e2e):
    *_, res = e2e
    h = res.report["holdout"]
    assert h["n_checkpoints"] > 0 and h["median_error_px"] < 3
    assert res.report["consistency"]["spread_rms_px"] < 1.0


def test_gcps_use_reference_geotransform(e2e):
    d, ref_path, *_rest, res = e2e
    import csv
    ref = load_raster(ref_path)
    rows = list(csv.DictReader(open(res.report["gcps"]["csv"], encoding="utf-8")))
    r = rows[0]
    mx, my = pixel_to_map(ref.transform, float(r["ref_x"]), float(r["ref_y"]))
    assert float(r["map_x"]) == pytest.approx(mx) and float(r["map_y"]) == pytest.approx(my)


def test_poor_input_is_rejected_without_writing_geotiff(tmp_path, scene):
    ref_path = tmp_path / "ref.tif"
    make_reference_geotiff(ref_path, 512, seed=7)
    blank = np.full((3, 300, 300), 128, np.uint8)
    sp = tmp_path / "blank.tif"
    write_geotiff(sp, blank, None, None)
    res = run_registration(sp, ref_path, tmp_path / "o", PipelineCfg(bootstrap_runs=0), run_name="blank")
    assert not res.success and "geotiff" not in res.outputs
    assert not list((tmp_path / "o" / "geotiff").glob("*.tif"))
    assert (tmp_path / "o" / "reports" / "blank_report.md").exists()


def test_unrelated_image_is_not_accepted(tmp_path, scene):
    """Safety: a scene that does NOT overlap must be rejected, not forced onto the grid."""
    from georeg.synthetic_transform import procedural_scene
    ref_path = tmp_path / "ref.tif"
    make_reference_geotiff(ref_path, 512, seed=7)
    sp = tmp_path / "other.tif"
    write_geotiff(sp, procedural_scene(400, seed=999), None, None)
    res = run_registration(sp, ref_path, tmp_path / "o", PipelineCfg(bootstrap_runs=0, phase_fallback=False))
    assert not res.success


def test_gdal_bindings_agree_if_installed(e2e):
    pytest.importorskip("osgeo")
    _d, ref_path, _s, _c, res = e2e
    checks = verify_with_gdal(res.outputs["geotiff"], ref_path)
    assert all(v[0] for v in checks.values()), checks
