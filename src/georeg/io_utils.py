"""Windows-safe image I/O.

cv2.imread / cv2.imwrite fail silently on non-ASCII Windows paths (e.g. a user
folder named with accents). Going through numpy + imdecode/imencode avoids it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .errors import InputError


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def imread_unicode(path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"file not found: {p}")
    buf = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise InputError(f"cannot decode image: {p}")
    return img


def imwrite_unicode(path, img: np.ndarray) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    ext = p.suffix or ".png"
    ok, enc = cv2.imencode(ext, img)
    if not ok:
        raise InputError(f"cannot encode image as {ext}: {p}")
    enc.tofile(str(p))
    return p
