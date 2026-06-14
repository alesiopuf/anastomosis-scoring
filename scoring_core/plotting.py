"""Rendering of the per-feature diagnostic overlays.

This module owns *how* an overlay is drawn. Each feature extractor in
features.py stays pure computation and hands back a :class:`PlotSpec` describing
*what* to draw (background image, scatters, lines, labels); :func:`render` turns
that spec into matplotlib calls on a given axis. Keeping the two apart means the
extractors carry no matplotlib dependency and the drawing can be tested,
restyled, or swapped out independently.
"""
import matplotlib
matplotlib.use("Agg")  # headless rendering; set before pyplot is imported anywhere
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Optional, Sequence  # noqa: E402


# Shared overlay palette / styling.
ANAST_COLOR = "blue"
ANAST_WIDTH = 2.5
HIGHLIGHT_COLOR = "yellow"
POINT_COLOR = "red"
LABEL_COLOR = "lime"


@dataclass
class Scatter:
    """A cloud of marker points (e.g. all detected stitch pixels)."""
    points: np.ndarray
    color: str = POINT_COLOR
    size: float = 3


@dataclass
class Line:
    """A straight segment between two points."""
    p1: Sequence[float]
    p2: Sequence[float]
    color: str = ANAST_COLOR
    linewidth: float = ANAST_WIDTH
    linestyle: str = "-"


@dataclass
class Polyline:
    """A connected path through an ordered sequence of points."""
    points: np.ndarray
    color: str = HIGHLIGHT_COLOR
    linewidth: float = 1.5
    linestyle: str = "-"


@dataclass
class Label:
    """A text label, optionally with leader line(s) drawn in the label colour.

    ``line``/``lines`` (each a (p1, p2) pair) are drawn as thin leader segments
    and also used to nudge the text clear of the segment. Set ``skip_line`` to
    keep a leader purely for positioning without drawing it.
    """
    pos: Sequence[float]
    text: str
    line: Optional[Sequence] = None
    lines: Optional[Sequence] = None
    skip_line: bool = False


@dataclass
class PlotSpec:
    """A self-contained description of one feature's diagnostic overlay."""
    img: np.ndarray
    scatters: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    polylines: list = field(default_factory=list)
    labels: list = field(default_factory=list)


def _draw_labels(ax, labels):
    for lbl in labels:
        if lbl.lines:
            for seg in lbl.lines:
                ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                        color=LABEL_COLOR, linestyle='-', linewidth=1.6)
        if lbl.line is not None and not lbl.skip_line:
            seg = lbl.line
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                    color=LABEL_COLOR, linestyle='-', linewidth=1.6)

        # Nudge the text off the leader segment along its normal.
        offset_x, offset_y = 5, 5
        seg = lbl.line if lbl.line is not None else (lbl.lines[0] if lbl.lines else None)
        if seg is not None:
            p1, p2 = seg
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                offset_x, offset_y = (-dy / dist) * 7, (dx / dist) * 7

        ax.text(lbl.pos[0] + offset_x, lbl.pos[1] + offset_y,
                lbl.text, color=LABEL_COLOR, fontsize=11, fontweight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='#222222', alpha=0.85, edgecolor='none', pad=0.5))


def render(spec: PlotSpec, ax=None):
    """Draw ``spec`` onto ``ax`` (or the current pyplot axis if ``ax`` is None)."""
    plot_ctx = ax if ax is not None else plt

    plot_ctx.imshow(spec.img, cmap='gray')
    for s in spec.scatters:
        pts = np.asarray(s.points)
        if pts.size > 0:
            plot_ctx.scatter(pts[:, 0], pts[:, 1], s=s.size, c=s.color)
    for ln in spec.lines:
        plot_ctx.plot([ln.p1[0], ln.p2[0]], [ln.p1[1], ln.p2[1]],
                      ln.color, linewidth=ln.linewidth, linestyle=ln.linestyle)
    for pl in spec.polylines:
        pts = np.asarray(pl.points)
        if pts.size > 0:
            plot_ctx.plot(pts[:, 0], pts[:, 1], pl.color,
                          linewidth=pl.linewidth, linestyle=pl.linestyle)
    _draw_labels(plot_ctx, spec.labels)

    if ax is not None:
        ax.axis('off')
    else:
        plt.axis('off')
        plt.show()
