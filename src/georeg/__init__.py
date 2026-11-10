"""georeg - automated optical image georegistration (feature matching -> RANSAC -> GeoTIFF)."""
__version__ = "1.0.0"

from .config import PipelineCfg  # noqa: F401
from .pipeline import run_registration, estimate_registration  # noqa: F401
