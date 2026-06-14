"""Pipeline orchestration for the web app: decode the upload, run preprocessing
and extraction, score the selected features, and render an annotated overlay
(base64 PNG) for each.
"""
import base64
from io import BytesIO

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .config import Config
from .plotting import render
from .preprocessing import preprocess_image, extract_anast_line_and_stitches
from .features import (
    extract_oblique_stitch,
    extract_large_distance_between_two_knots,
    extract_general_bite_size,
    extract_disruption_of_anastomosis_line,
    extract_wide_large_bite,
    extract_excessive_tightening,
    extract_partial_thickness,
)


class PipelineError(Exception):
    """Raised when an image cannot be scored (e.g. no suture line detected)."""


# The seven features in canonical order; meta() hands these to the UI.
FEATURES = [
    {
        "key": "oblique_stitch",
        "label": "Oblique stitch",
        "kind": "count",
        "fn": extract_oblique_stitch,
        "help": "Stitches set at an angle rather than perpendicular to the suture line.",
    },
    {
        "key": "large_distance",
        "label": "Gaps between knots",
        "kind": "count",
        "fn": extract_large_distance_between_two_knots,
        "help": "Excessive spacing between adjacent stitches.",
    },
    {
        "key": "general_bite_size",
        "label": "General bite size",
        "kind": "category",
        "fn": extract_general_bite_size,
        "categories": {"constant": "ok", "unequal": "error", "not_sure": "uncertain"},
        "help": "Whether bite size stays consistent around the circumference.",
    },
    {
        "key": "disruption",
        "label": "Disruption of anastomosis line",
        "kind": "category",
        "fn": extract_disruption_of_anastomosis_line,
        "categories": {"no": "ok", "yes": "error", "not_sure": "uncertain"},
        "help": "Whether the stitches stray from a smooth, continuous suture line.",
    },
    {
        "key": "wide_large_bite",
        "label": "Wide / large bite",
        "kind": "count",
        "fn": extract_wide_large_bite,
        "help": "Stitches taking an excessively large tissue margin.",
    },
    {
        "key": "excessive_tightening",
        "label": "Excessive tightening",
        "kind": "count",
        "fn": extract_excessive_tightening,
        "help": "Stitches pulled tight enough to risk strangulating the tissue.",
    },
    {
        "key": "partial_thickness",
        "label": "Partial thickness",
        "kind": "count",
        "fn": extract_partial_thickness,
        "help": "Shallow, one-sided bites that miss the full wall depth.",
    },
]

FEATURE_BY_KEY = {f["key"]: f for f in FEATURES}

# Adjustable thresholds exposed in the advanced panel.
THRESHOLDS = [
    {"key": "oblique_max_angle", "label": "Oblique max angle (°)", "default": 15, "min": 0, "max": 45, "step": 1, "feature": "oblique_stitch"},
    {"key": "oblique_min_size_factor", "label": "Oblique min size factor", "default": 0.5, "min": 0, "max": 1, "step": 0.05, "feature": "oblique_stitch"},
    {"key": "large_distance_factor", "label": "Gap distance factor", "default": 1.5, "min": 1, "max": 3, "step": 0.1, "feature": "large_distance"},
    {"key": "general_bite_cv_min", "label": "Bite-size CV lower", "default": 0.36, "min": 0, "max": 1, "step": 0.01, "feature": "general_bite_size"},
    {"key": "general_bite_cv_max", "label": "Bite-size CV upper", "default": 0.40, "min": 0, "max": 1, "step": 0.01, "feature": "general_bite_size"},
    {"key": "disruption_ratio_min", "label": "Disruption ratio lower", "default": 0.28, "min": 0, "max": 1, "step": 0.01, "feature": "disruption"},
    {"key": "disruption_ratio_max", "label": "Disruption ratio upper", "default": 0.30, "min": 0, "max": 1, "step": 0.01, "feature": "disruption"},
    {"key": "wide_large_bite_pct", "label": "Wide bite percentage", "default": 0.65, "min": 0, "max": 1.5, "step": 0.05, "feature": "wide_large_bite"},
    {"key": "excessive_tightening_pct", "label": "Tightening percentage", "default": 0.30, "min": 0, "max": 1, "step": 0.05, "feature": "excessive_tightening"},
    {"key": "partial_thickness_pct", "label": "Partial-thickness fraction", "default": 0.75, "min": 0, "max": 1.5, "step": 0.05, "feature": "partial_thickness"},
]


def meta() -> dict:
    """Metadata for the GUI (features + threshold spec)."""
    feats = [{k: v for k, v in f.items() if k != "fn"} for f in FEATURES]
    return {"features": feats, "thresholds": THRESHOLDS}


def decode_image(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise PipelineError("Could not read that file as an image. Please upload a JPG or PNG.")
    return img


def _new_canvas(width_in=8.0, height_in=4.0, dpi=185):
    # Figure + Agg directly, not pyplot — avoids pyplot's global figure state.
    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.axis("off")
    return fig, ax


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02, transparent=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _render_overview(img_orig, a, b, stitches) -> str:
    fig, ax = _new_canvas()
    ax.imshow(img_orig, cmap="gray")
    pts = np.vstack(list(stitches.values()))
    ax.scatter(pts[:, 0], pts[:, 1], s=3, c="red")
    ax.plot([a[0], b[0]], [a[1], b[1]], "blue", linewidth=2.5)
    return _fig_to_b64(fig)


def _classify(feature, value):
    """Return (display_string, status) where status is ok|error|uncertain."""
    if feature["kind"] == "count":
        return str(int(value)), ("error" if value and value > 0 else "ok")
    status = feature.get("categories", {}).get(value, "uncertain")
    label = {"not_sure": "uncertain"}.get(value, value)
    return label, status


def analyze(image_bytes: bytes, selected_keys, overrides=None) -> dict:
    """Score the selected features for an uploaded image.

    Returns a dict with num_stitches, an overview image, and a list of
    per-feature results (value, display, status, annotated image).
    """
    cfg = Config.from_overrides(overrides)
    img_color = decode_image(image_bytes)

    try:
        img_orig, mask = preprocess_image(img_color)
        a, b, stitches = extract_anast_line_and_stitches(mask)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the UI
        raise PipelineError(
            "Could not detect a suture line in this image. Please upload a clear "
            "intraluminal photo of a completed anastomosis."
        ) from exc

    if len(stitches) < 2:
        raise PipelineError(
            "Fewer than two stitches were detected — not enough to score. Try a clearer or "
            "higher-contrast intraluminal photo."
        )

    # Preserve the canonical feature order regardless of request order.
    selected = [f for f in FEATURES if f["key"] in set(selected_keys)]

    results = []
    for feature in selected:
        try:
            fig, ax = _new_canvas()
            value, spec = feature["fn"](a, b, stitches, img_orig, cfg)
            render(spec, ax)
            display, status = _classify(feature, value)
            image = _fig_to_b64(fig)
            results.append({
                "key": feature["key"],
                "label": feature["label"],
                "kind": feature["kind"],
                "value": int(value) if feature["kind"] == "count" else value,
                "display": display,
                "status": status,
                "image": image,
            })
        except Exception as exc:  # noqa: BLE001 - one bad feature shouldn't drop the rest
            results.append({
                "key": feature["key"],
                "label": feature["label"],
                "kind": feature["kind"],
                "value": None,
                "display": "n/a",
                "status": "uncertain",
                "image": None,
                "error": str(exc),
            })

    return {
        "num_stitches": len(stitches),
        "overview": _render_overview(img_orig, a, b, stitches),
        "results": results,
    }
