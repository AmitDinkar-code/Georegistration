import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georeg.config import PreprocessCfg  # noqa: E402
from georeg.dataset_loader import detection_mask, prepare_gray  # noqa: E402
from georeg.synthetic_transform import make_case, procedural_scene  # noqa: E402


@pytest.fixture(scope="session")
def scene():
    return procedural_scene(512, seed=7)


@pytest.fixture(scope="session")
def case(scene):
    return make_case(scene, rotation_deg=18.0, scale=0.85, translation=(10.0, -8.0), crop=0.8, seed=1)


@pytest.fixture(scope="session")
def grays(scene, case):
    pre = PreprocessCfg()
    return (prepare_gray(case.data, case.valid, pre), prepare_gray(scene, None, pre),
            detection_mask(case.valid, 3))
