# Anastomosis Assessment

A small Flask web app for scoring microsurgical end-to-end anastomoses against
an error taxonomy rubric. Upload an intraluminal photo, pick which criteria to evaluate,
and the pipeline returns a value and an annotated diagnostic overlay for each.

The scoring core (`scoring_core/`) is forked from the upstream detection pipeline and
kept deterministic — same preprocessing and feature logic as the thesis, wrapped
so it can run on uploaded images and accept per-request threshold overrides.

## Features

The seven criteria, scored independently:

- **Oblique stitch** — stitches angled rather than perpendicular to the line
- **Gaps between knots** — excessive spacing between adjacent stitches
- **General bite size** — whether bite size stays consistent around the circumference
- **Disruption of anastomosis line** — stitches straying from a smooth, continuous line
- **Wide / large bite** — stitches taking too large a tissue margin
- **Excessive tightening** — stitches pulled tight enough to risk strangulating tissue
- **Partial thickness** — shallow, one-sided bites that miss the full wall depth

Thresholds default to the calibrated values from the thesis but can be tweaked
per run from the **Threshold overrides** panel.

## Requirements

- [uv](https://docs.astral.sh/uv/) (handles Python and dependencies)
- Python 3.12 — uv installs it automatically if it's missing

## Running

`uv run` reads `pyproject.toml`/`uv.lock`, sets up the environment on first run,
and starts the app:

```bash
uv run python app.py
```

## Usage

1. Drop in an intraluminal photo of a completed anastomosis, or click one of the
   bundled reference specimens.
2. Choose the criteria to evaluate (all are selected by default).
3. Hit **Run assessment**. Check the overview image first to confirm the stitches
   and anastomotic line were detected correctly, then review the per-criterion
   overlays and scores.

Each run is kept in a session **History** drawer (stored in the browser) and can
be exported to CSV.

## Project layout

```
app.py                  Flask routes (/, /api/meta, /api/samples, /api/analyze)
scoring_core/
  config.py             fixed preprocessing params + the Config threshold dataclass
  preprocessing.py      image preprocessing and suture-line extraction
  features.py           the seven feature extractors + overlay drawing
  pipeline.py           ties it together: decode -> extract -> score -> render
  utils.py              geometry helpers
templates/index.html    single-page UI
static/                 app.js, style.css, and sample images
```
