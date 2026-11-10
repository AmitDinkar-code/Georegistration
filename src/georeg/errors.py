"""Exception hierarchy. Every expected failure mode has its own type so the
pipeline can decide between 'try the fallback' and 'abort'."""


class GeoregError(Exception):
    """Base class for all errors raised on purpose by this package."""


class InputError(GeoregError):
    """Unreadable file, wrong shape, missing georeferencing on the reference."""


class InsufficientFeaturesError(GeoregError):
    """Too few keypoints (blank / extremely low-texture image)."""


class InsufficientMatchesError(GeoregError):
    """Too few descriptor matches survived filtering."""

    def __init__(self, message: str, stats: dict | None = None):
        super().__init__(message)
        self.stats = stats or {}


class EstimationError(GeoregError):
    """RANSAC could not produce a transformation."""


class RegistrationFailed(GeoregError):
    """Every attempt (including fallbacks) failed the quality gates."""
