"""Command line:  python -m georeg <command> ..."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import PipelineCfg
from .geotiff_writer import write_geotiff
from .io_utils import ensure_dir
from .pipeline import run_registration
from .synthetic_transform import PRESETS, make_case, make_reference_geotiff
from .dataset_loader import load_raster


def _setup_logging(verbose: bool, logfile: Path | None = None):
    handlers = [logging.StreamHandler()]
    if logfile is not None:
        ensure_dir(logfile.parent)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def _cfg_from_args(a) -> PipelineCfg:
    cfg = PipelineCfg()
    models = ("affine", "homography") if a.model == "auto" else (a.model,)
    dets = ("sift", "orb") if a.detector == "auto" else (a.detector,)
    cfg = replace(cfg, attempts=tuple((d, models) for d in dets),
                  matcher=replace(cfg.matcher, matcher=a.matcher),
                  ransac=replace(cfg.ransac, reproj_thresh=a.threshold),
                  warp=replace(cfg.warp, interpolation=a.interp))
    return cfg


def cmd_make_synthetic(a) -> int:
    d = ensure_dir(a.out)
    ref_path, src_path = d / "reference.tif", d / "source_unreferenced.tif"
    ref_tr = make_reference_geotiff(ref_path, a.size, a.seed)
    ref = load_raster(ref_path)
    case = make_case(ref.data, seed=a.seed, **PRESETS[a.preset])
    write_geotiff(src_path, case.data, None, None, valid=case.valid, nodata=0, tags=dict(SYNTHETIC="1", PRESET=a.preset))
    (d / "truth.json").write_text(json.dumps(dict(
        preset=a.preset, params={k: list(v) if isinstance(v, tuple) else v for k, v in case.params.items()},
        M_ref_to_src=case.M_ref_to_src.tolist(), T_true_src_to_ref=case.T_true.tolist(),
        reference_transform=list(ref_tr)[:6]), indent=2), encoding="utf-8")
    print(f"wrote {ref_path}\nwrote {src_path}\nwrote {d / 'truth.json'}")
    return 0


def cmd_register(a) -> int:
    out = Path(a.out)
    _setup_logging(a.verbose, out / "reports" / "run.log")
    T_true = None
    if a.truth:
        T_true = np.array(json.loads(Path(a.truth).read_text(encoding="utf-8"))["T_true_src_to_ref"])
    res = run_registration(a.source, a.reference, out, _cfg_from_args(a), run_name=a.name, T_true=T_true)
    print(("ACCEPTED" if res.success else "REJECTED"), "-> report:", res.outputs.get("markdown"))
    return 0 if res.success else 2


def cmd_demo(a) -> int:
    out = Path(a.out)
    _setup_logging(a.verbose, out / "reports" / "run.log")
    data = ensure_dir(Path(a.data))
    ref_path = data / "reference.tif"
    ref_tr = make_reference_geotiff(ref_path, a.size, 7)
    ref = load_raster(ref_path)
    summary = []
    scenarios = [("demo_sift", "demo", "sift"), ("demo_orb", "demo", "orb"),
                 ("perspective_sift", "perspective", "sift")]
    for name, preset, det in scenarios:
        case = make_case(ref.data, seed=3, **PRESETS[preset])
        sp = data / f"{name}_source_unreferenced.tif"
        write_geotiff(sp, case.data, None, None, valid=case.valid, nodata=0, tags=dict(SYNTHETIC="1", PRESET=preset))
        cfg = PipelineCfg(attempts=((det, ("affine", "homography")),), phase_fallback=False)
        res = run_registration(sp, ref_path, out, cfg, run_name=name, T_true=case.T_true)
        gt = res.report.get("ground_truth", {})
        row = dict(run=name, accepted=res.success, model=res.report.get("model"),
                   gt_rmse_px=gt.get("rmse_px"), gt_rmse_map=gt.get("rmse_map"))
        summary.append(row)
        print(row)
        if res.success:
            from .verify_metadata import report as vreport, verify_with_gdal, verify_with_rasterio
            ok = vreport(verify_with_rasterio(res.outputs["geotiff"], ref_path))
            vreport(verify_with_gdal(res.outputs["geotiff"], ref_path))
            print("metadata check:", "PASS" if ok else "FAIL")
    (out / "reports" / "demo_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


def cmd_robustness(a) -> int:
    from .robustness import run_robustness
    _setup_logging(a.verbose)
    r = run_robustness(Path(a.out) / "reports" / "robustness", size=a.size, n_trials=a.trials)
    print("wrote", r["markdown"], r["figure"], r["csv"])
    return 0


def cmd_verify(a) -> int:
    from .verify_metadata import main
    return main([a.output, a.reference])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="georeg", description="Automated image georegistration")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-synthetic", help="write a synthetic reference + unreferenced source + truth.json")
    s.add_argument("--out", default="data/synthetic"); s.add_argument("--size", type=int, default=1024)
    s.add_argument("--seed", type=int, default=7); s.add_argument("--preset", choices=list(PRESETS), default="demo")
    s.set_defaults(fn=cmd_make_synthetic)

    s = sub.add_parser("register", help="register SOURCE onto REFERENCE and write a GeoTIFF")
    s.add_argument("--source", required=True); s.add_argument("--reference", required=True)
    s.add_argument("--out", default="outputs"); s.add_argument("--name")
    s.add_argument("--detector", choices=["auto", "sift", "orb"], default="auto")
    s.add_argument("--model", choices=["auto", "homography", "affine", "similarity"], default="auto")
    s.add_argument("--matcher", choices=["bf", "flann"], default="bf")
    s.add_argument("--interp", choices=["nearest", "bilinear", "cubic"], default="bilinear")
    s.add_argument("--threshold", type=float, default=3.0, help="RANSAC reprojection threshold, reference px")
    s.add_argument("--truth", help="truth.json from make-synthetic (enables ground-truth error)")
    s.set_defaults(fn=cmd_register)

    s = sub.add_parser("demo", help="end-to-end run on synthetic data (3 scenarios)")
    s.add_argument("--out", default="outputs"); s.add_argument("--data", default="data/synthetic")
    s.add_argument("--size", type=int, default=1024)
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("robustness", help="SIFT vs ORB robustness study")
    s.add_argument("--out", default="outputs"); s.add_argument("--size", type=int, default=640)
    s.add_argument("--trials", type=int, default=3)
    s.set_defaults(fn=cmd_robustness)

    s = sub.add_parser("verify", help="verify GeoTIFF metadata against its reference")
    s.add_argument("output"); s.add_argument("reference")
    s.set_defaults(fn=cmd_verify)
    return p


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    a = build_parser().parse_args(argv)
    if a.cmd in ("make-synthetic", "verify"):
        _setup_logging(a.verbose)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
