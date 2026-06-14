"""The seven feature extractors (forked from the upstream detection pipeline).

Same logic as the original, with two adaptations: each extractor takes a Config
instead of module-level threshold constants (so the UI can override per request),
and each returns ``(value, PlotSpec)`` where the PlotSpec is a matplotlib-free
description of its diagnostic overlay. Rendering lives in plotting.py; see
:func:`scoring_core.plotting.render`.
"""
import numpy as np

from .plotting import PlotSpec, Scatter, Line, Polyline, Label
from .utils import (
    angle_between_vectors,
    calculate_stitch_lengths,
    calculate_string_length,
    threshold_by_percentage,
)


def _anast_line(a, b):
    """The blue anastomosis baseline segment shared by every overlay."""
    return Line(a, b)


def extract_oblique_stitch(a, b, stitches, img, cfg):
    anastomosis_vec = np.array(b) - np.array(a)
    anastomosis_len = np.linalg.norm(anastomosis_vec)
    u = anastomosis_vec / (anastomosis_len + 1e-6)
    n_vec = np.array([-u[1], u[0]])

    oblique_points = []
    count = 0
    median = np.median(calculate_stitch_lengths(stitches))
    labels = []

    for label, group in stitches.items():
        if calculate_string_length(group) < cfg.oblique_min_size_factor * median:
            continue

        mean = group.mean(axis=0)
        centered = group - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        stitch_dir = eigvecs[:, np.argmax(eigvals)]

        angle = angle_between_vectors(stitch_dir, n_vec)
        if angle > 90:
            angle = 180 - angle

        base_angle = angle_between_vectors(stitch_dir, u)
        if not (90 - cfg.oblique_max_angle <= base_angle <= 90 + cfg.oblique_max_angle):
            oblique_points.extend(group.tolist())
            count += 1

        vec_am = mean - np.array(a)
        proj_dist = np.dot(vec_am, u)
        p_anast = np.array(a) + proj_dist * u
        dist_to_line = np.linalg.norm(mean - p_anast)
        line_len = max(dist_to_line, 12)
        p_perp = p_anast + (mean - p_anast) / (dist_to_line + 1e-6) * line_len
        s_dot_n = np.dot(stitch_dir, (mean - p_anast))
        s_plot = stitch_dir if s_dot_n >= 0 else -stitch_dir
        p_orient = p_anast + s_plot * line_len

        labels.append(Label(
            pos=(p_anast[0], p_anast[1]),
            text=f'{angle:.0f}°',
            lines=[(p_anast, p_perp), (p_anast, p_orient)],
        ))

    oblique_points = np.array(oblique_points)
    points = np.vstack(list(stitches.values()))

    spec = PlotSpec(
        img=img,
        scatters=[Scatter(points), Scatter(oblique_points, color="yellow", size=12)],
        lines=[_anast_line(a, b)],
        labels=labels,
    )
    return count, spec


def extract_large_distance_between_two_knots(a, b, stitches, img, cfg):
    centroids = []
    for group in stitches.values():
        if group.size > 0:
            centroids.append(group.mean(axis=0))
    centroids = np.array(centroids)
    if len(centroids) < 2:
        return 0, PlotSpec(img=img, lines=[_anast_line(a, b)])

    line_vec = np.array(b) - np.array(a)
    line_unit = line_vec / np.linalg.norm(line_vec)
    projections = np.dot(centroids - a, line_unit)
    sorted_indices = np.argsort(projections)
    sorted_centroids = centroids[sorted_indices]

    distances = np.linalg.norm(np.diff(sorted_centroids, axis=0), axis=1)
    avg_distance = np.mean(distances)
    large_indices = np.where(distances > cfg.large_distance_factor * avg_distance)[0]
    count = len(large_indices)
    large_set = set(large_indices.tolist())

    # Yellow segment for each large gap; a green leader line for the rest (the
    # large gaps keep only the yellow line so it stays visible — skip_line).
    gap_lines = []
    labels = []
    for idx in range(len(sorted_centroids) - 1):
        p1, p2 = sorted_centroids[idx], sorted_centroids[idx + 1]
        is_large = idx in large_set
        if is_large:
            gap_lines.append(Line(p1, p2, color="yellow", linewidth=2))
        labels.append(Label(
            pos=((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            text=f'{distances[idx]:.1f}',
            line=(p1, p2),
            skip_line=is_large,
        ))

    spec = PlotSpec(
        img=img,
        scatters=[Scatter(np.vstack(list(stitches.values())))],
        lines=gap_lines + [_anast_line(a, b)],
        labels=labels,
    )
    return count, spec


def extract_general_bite_size(a, b, stitches, img, cfg):
    bite_sizes = []
    labels = []
    for group in stitches.values():
        if len(group) < 2:
            continue
        pts = group.astype(float)
        diff = pts[:, None, :] - pts[None, :, :]
        dmat = np.linalg.norm(diff, axis=-1)
        i, j = np.unravel_index(dmat.argmax(), dmat.shape)
        p1, p2 = pts[i], pts[j]
        bite_size = float(dmat.max())
        bite_sizes.append(bite_size)
        labels.append(Label(
            pos=((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            text=f'{bite_size:.1f}',
            line=(p1, p2),
        ))

    cv = np.std(bite_sizes) / np.mean(bite_sizes)

    spec = PlotSpec(
        img=img,
        scatters=[Scatter(np.vstack(list(stitches.values())))],
        lines=[_anast_line(a, b)],
        labels=labels,
    )

    if cfg.general_bite_cv_min <= cv <= cfg.general_bite_cv_max:
        return 'not_sure', spec
    return ('unequal' if cv > cfg.general_bite_cv_max else 'constant'), spec


def extract_disruption_of_anastomosis_line(a, b, stitches, img, cfg):
    a = np.array(a)
    b = np.array(b)
    line_vec = b - a
    norm_len = np.linalg.norm(line_vec)
    if norm_len == 0 or not stitches:
        return 'no', PlotSpec(img=img, lines=[_anast_line(a, b)])
    unit_line_vec = line_vec / norm_len

    centroids, distances, projections, stitch_sizes, labels = [], [], [], [], []
    for group in stitches.values():
        group_np = np.array(group)
        stitch_size = np.linalg.norm(group_np.max(axis=0) - group_np.min(axis=0))
        stitch_sizes.append(stitch_size)
        centroid = np.mean(group_np, axis=0)
        vec = centroid - a
        proj_scalar = np.dot(vec, unit_line_vec)
        perp = vec - (proj_scalar * unit_line_vec)
        distance = np.linalg.norm(perp)
        proj_point = a + proj_scalar * unit_line_vec
        centroids.append(centroid)
        distances.append(distance)
        projections.append(proj_scalar)
        labels.append(Label(
            pos=((centroid[0] + proj_point[0]) / 2, (centroid[1] + proj_point[1]) / 2),
            text=f'{distance:.1f}',
            line=(centroid, proj_point),
        ))

    mean_distance = np.mean(distances)
    avg_stitch_size = np.mean(stitch_sizes)
    disruption_ratio = mean_distance / avg_stitch_size if avg_stitch_size > 0 else 0

    centroids_np = np.array(centroids)
    sorted_centroids = centroids_np[np.argsort(projections)]
    spec = PlotSpec(
        img=img,
        lines=[_anast_line(a, b)],
        polylines=[Polyline(sorted_centroids, color="yellow", linestyle='--')],
        labels=labels,
    )

    if cfg.disruption_ratio_min <= disruption_ratio <= cfg.disruption_ratio_max:
        return 'not_sure', spec
    return ('yes' if disruption_ratio > cfg.disruption_ratio_max else 'no'), spec


def extract_wide_large_bite(a, b, stitches, img, cfg):
    count = 0
    min_threshold, max_threshold = threshold_by_percentage(stitches, cfg.wide_large_bite_pct)
    bite_points = []
    labels = []
    for label, group in stitches.items():
        length = calculate_string_length(group)
        if length > max_threshold:
            bite_points.extend(group.tolist())
            count += 1
        pts = group.astype(float)
        diff = pts[:, None, :] - pts[None, :, :]
        dmat = np.linalg.norm(diff, axis=-1)
        i, j = np.unravel_index(dmat.argmax(), dmat.shape)
        p1, p2 = pts[i], pts[j]
        labels.append(Label(
            pos=((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            text=f'{length:.1f}',
            line=(p1, p2),
        ))

    spec = PlotSpec(
        img=img,
        scatters=[
            Scatter(np.vstack(list(stitches.values()))),
            Scatter(np.array(bite_points), color="yellow", size=12),
        ],
        lines=[_anast_line(a, b)],
        labels=labels,
    )
    return count, spec


def extract_excessive_tightening(a, b, stitches, img, cfg):
    min_threshold, max_threshold = threshold_by_percentage(stitches, cfg.excessive_tightening_pct)
    partial_points = []
    labels = []
    count = 0
    for label, group in stitches.items():
        length = calculate_string_length(group)
        if length < min_threshold:
            partial_points.extend(group.tolist())
            count += 1
        pts = group.astype(float)
        diff = pts[:, None, :] - pts[None, :, :]
        dmat = np.linalg.norm(diff, axis=-1)
        i, j = np.unravel_index(dmat.argmax(), dmat.shape)
        p1, p2 = pts[i], pts[j]
        labels.append(Label(
            pos=((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            text=f'{length:.1f}',
            line=(p1, p2),
        ))

    spec = PlotSpec(
        img=img,
        scatters=[
            Scatter(np.vstack(list(stitches.values()))),
            Scatter(np.array(partial_points), color="yellow", size=12),
        ],
        lines=[_anast_line(a, b)],
        labels=labels,
    )
    return count, spec


def extract_partial_thickness(a, b, stitches, img, cfg):
    count = 0
    vector_ab = np.array([b[0] - a[0], b[1] - a[1]])
    stitches_mean_length = np.mean(calculate_stitch_lengths(stitches))
    partial_points = []
    labels = []
    for group in stitches.values():
        has_above = has_below = False
        for point in group:
            vector_ac = (point[0] - a[0], point[1] - a[1])
            cross_product = (vector_ab[0] * vector_ac[1]) - (vector_ab[1] * vector_ac[0])
            if cross_product > 0:
                has_above = True
            elif cross_product < 0:
                has_below = True
        group_length = calculate_string_length(group)
        if (has_above and not has_below and group_length <= cfg.partial_thickness_pct * stitches_mean_length) or \
           (not has_above and has_below and group_length <= cfg.partial_thickness_pct * stitches_mean_length):
            partial_points.extend(group.tolist())
            count += 1
        pos = "above" if has_above and not has_below else ("below" if not has_above and has_below else "both")
        pts = group.astype(float)
        diff = pts[:, None, :] - pts[None, :, :]
        dmat = np.linalg.norm(diff, axis=-1)
        i, j = np.unravel_index(dmat.argmax(), dmat.shape)
        p1, p2 = pts[i], pts[j]
        labels.append(Label(
            pos=((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            text=f'{group_length:.1f}\n{pos}',
            line=(p1, p2),
        ))

    spec = PlotSpec(
        img=img,
        scatters=[
            Scatter(np.vstack(list(stitches.values()))),
            Scatter(np.array(partial_points), color="yellow", size=12),
        ],
        lines=[_anast_line(a, b)],
        labels=labels,
    )
    return count, spec
