"""Pipeline configuration.

Preprocessing params are fixed (calibrated in the thesis). Feature thresholds
sit on the Config dataclass so the UI can override them per request.
"""
from dataclasses import dataclass, fields

# Fixed preprocessing parameters, calibrated in the thesis.
PREPROCESS_PARAMS = {
    "lightning": 218,
    "dilation": 5,
    "radius": 18,
    "open_w": 5,
    "open_h": 17,
    "gaussian": 1,
    "blackhat": 26,
    "frangi_min": 4,
    "frangi_max": 15,
    "small_objects": 19,
}
IMG_SIZE = 512, 256  # (width, height)


@dataclass
class Config:
    """Feature thresholds. Defaults are the calibrated values from the thesis."""

    oblique_max_angle: float = 15.0
    oblique_min_size_factor: float = 0.5
    large_distance_factor: float = 1.5
    general_bite_cv_min: float = 0.36
    general_bite_cv_max: float = 0.4
    disruption_ratio_min: float = 0.28
    disruption_ratio_max: float = 0.3
    wide_large_bite_pct: float = 0.65
    excessive_tightening_pct: float = 0.3
    partial_thickness_pct: float = 0.75

    @classmethod
    def from_overrides(cls, overrides) -> "Config":
        """Build a Config from a dict of overrides, ignoring unknown/blank keys."""
        cfg = cls()
        if overrides:
            valid = {f.name for f in fields(cls)}
            for key, value in overrides.items():
                if key in valid and value not in (None, ""):
                    try:
                        setattr(cfg, key, float(value))
                    except (TypeError, ValueError):
                        pass
        return cfg
