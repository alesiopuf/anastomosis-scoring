"""The seven feature extractors (forked from the upstream detection pipeline).

Same logic as the original, with two adaptations: each extractor takes a Config
instead of module-level threshold constants (so the UI can override per request),
and each draws its diagnostic overlay onto a matplotlib ax passed in by the caller.
"""
import matplotlib
matplotlib.use("Agg")  # headless rendering; set before pyplot is imported anywhere
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .utils import (  # noqa: E402
    angle_between_vectors,
    calculate_stitch_lengths,
    calculate_string_length,
    threshold_by_percentage,
)


def annotate_plot(plot_ctx, annotations):
    for ann in annotations:
        if 'lines' in ann:
            for l in ann['lines']:
                plot_ctx.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], color='lime', linestyle='-', linewidth=1.6)
        if 'line' in ann:
            l = ann['line']
            plot_ctx.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], color='lime', linestyle='-', linewidth=1.6)

        offset_x, offset_y = 5, 5
        if 'line' in ann:
            p1, p2 = ann['line']
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                offset_x, offset_y = (-dy / dist) * 7, (dx / dist) * 7
        elif 'lines' in ann and len(ann['lines']) > 0:
            p1, p2 = ann['lines'][0]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                offset_x, offset_y = (-dy / dist) * 7, (dx / dist) * 7

        plot_ctx.text(ann['pos'][0] + offset_x, ann['pos'][1] + offset_y,
                      ann['text'], color='lime', fontsize=11, fontweight='bold', ha='center', va='center',
                      bbox=dict(facecolor='#222222', alpha=0.85, edgecolor='none', pad=0.5))


def extract_oblique_stitch(a, b, stitches, img, cfg, verbose=False, ax=None):
    anastomosis_vec = np.array(b) - np.array(a)
    anastomosis_len = np.linalg.norm(anastomosis_vec)
    u = anastomosis_vec / (anastomosis_len + 1e-6)
    n_vec = np.array([-u[1], u[0]])

    oblique_points = []
    count = 0
    median = np.median(calculate_stitch_lengths(stitches))
    annotations = []

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

        annotations.append({
            'pos': (p_anast[0], p_anast[1]),
            'text': f'{angle:.0f}°',
            'lines': [(p_anast, p_perp), (p_anast, p_orient)],
        })

    oblique_points = np.array(oblique_points)
    points = np.vstack(list(stitches.values()))

    if verbose:
        plot_ctx = ax if ax is not None else plt
        plot_ctx.imshow(img, cmap='gray')
        plot_ctx.scatter(points[:, 0], points[:, 1], s=3, c='red')
        if oblique_points.size > 0:
            plot_ctx.scatter(oblique_points[:, 0], oblique_points[:, 1], s=12, c='yellow')
        plot_ctx.plot([a[0], b[0]], [a[1], b[1]], 'blue', linewidth=2.5)
        annotate_plot(plot_ctx, annotations)
        if ax is not None:
            ax.axis('off')
        else:
            plt.axis('off')
            plt.show()

    return count


def extract_large_distance_between_two_knots(a, b, stitches, img, cfg, verbose=False, ax=None):
    centroids = []
    for group in stitches.values():
        if group.size > 0:
            centroids.append(group.mean(axis=0))
    centroids = np.array(centroids)
    if len(centroids) < 2:
        return 0

    line_vec = np.array(b) - np.array(a)
    line_unit = line_vec / np.linalg.norm(line_vec)
    projections = np.dot(centroids - a, line_unit)
    sorted_indices = np.argsort(projections)
    sorted_centroids = centroids[sorted_indices]

    distances = np.linalg.norm(np.diff(sorted_centroids, axis=0), axis=1)
    avg_distance = np.mean(distances)
    large_indices = np.where(distances > cfg.large_distance_factor * avg_distance)[0]
    count = len(large_indices)

    annotations = []
    for idx in range(len(sorted_centroids) - 1):
        p1, p2 = sorted_centroids[idx], sorted_centroids[idx + 1]
        annotations.append({
            'pos': ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            'text': f'{distances[idx]:.1f}',
            'line': (p1, p2),
        })

    if verbose:
        plot_ctx = ax if ax is not None else plt
        all_points = np.vstack(list(stitches.values()))
        plot_ctx.imshow(img, cmap='gray')
        plot_ctx.scatter(all_points[:, 0], all_points[:, 1], s=3, c='red')
        for idx in large_indices:
            p1, p2 = sorted_centroids[idx], sorted_centroids[idx + 1]
            plot_ctx.plot([p1[0], p2[0]], [p1[1], p2[1]], 'yellow', linewidth=2)
        annotate_plot(plot_ctx, annotations)
        plot_ctx.plot([a[0], b[0]], [a[1], b[1]], 'blue', linewidth=2.5)
        if ax is not None:
            ax.axis('off')
        else:
            plt.axis('off')
            plt.show()

    return count


def extract_general_bite_size(a, b, stitches, img, cfg, verbose=False, ax=None):
    bite_sizes = []
    annotations = []
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
        annotations.append({
            'pos': ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            'text': f'{bite_size:.1f}',
            'line': (p1, p2),
        })

    cv = np.std(bite_sizes) / np.mean(bite_sizes)

    if verbose:
        plot_ctx = ax if ax is not None else plt
        all_points = np.vstack(list(stitches.values()))
        plot_ctx.imshow(img, cmap='gray')
        plot_ctx.scatter(all_points[:, 0], all_points[:, 1], s=3, c='red')
        plot_ctx.plot([a[0], b[0]], [a[1], b[1]], 'blue', linewidth=2.5)
        annotate_plot(plot_ctx, annotations)
        if ax is not None:
            ax.axis('off')
        else:
            plt.axis('off')
            plt.show()

    if cfg.general_bite_cv_min <= cv <= cfg.general_bite_cv_max:
        return 'not_sure'
    return 'unequal' if cv > cfg.general_bite_cv_max else 'constant'


def extract_disruption_of_anastomosis_line(a, b, stitches, img, cfg, verbose=False, ax=None):
    a = np.array(a)
    b = np.array(b)
    line_vec = b - a
    norm_len = np.linalg.norm(line_vec)
    if norm_len == 0 or not stitches:
        return 'no'
    unit_line_vec = line_vec / norm_len

    centroids, distances, projections, stitch_sizes, annotations = [], [], [], [], []
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
        annotations.append({
            'pos': ((centroid[0] + proj_point[0]) / 2, (centroid[1] + proj_point[1]) / 2),
            'text': f'{distance:.1f}',
            'line': (centroid, proj_point),
        })

    mean_distance = np.mean(distances)
    avg_stitch_size = np.mean(stitch_sizes)
    disruption_ratio = mean_distance / avg_stitch_size if avg_stitch_size > 0 else 0

    if verbose:
        plot_ctx = ax if ax is not None else plt
        centroids_np = np.array(centroids)
        sorted_indices = np.argsort(projections)
        sorted_centroids = centroids_np[sorted_indices]
        plot_ctx.imshow(img, cmap='gray')
        plot_ctx.plot([a[0], b[0]], [a[1], b[1]], 'blue', linewidth=2.5)
        plot_ctx.plot(sorted_centroids[:, 0], sorted_centroids[:, 1], 'yellow', linestyle='--')
        annotate_plot(plot_ctx, annotations)
        if ax is not None:
            ax.axis('off')
        else:
            plt.axis('off')
            plt.show()

    if cfg.disruption_ratio_min <= disruption_ratio <= cfg.disruption_ratio_max:
        return 'not_sure'
    return 'yes' if disruption_ratio > cfg.disruption_ratio_max else 'no'


def extract_wide_large_bite(a, b, stitches, img, cfg, verbose=False, ax=None):
    count = 0
    min_threshold, max_threshold = threshold_by_percentage(stitches, cfg.wide_large_bite_pct)
    bite_points = []
    annotations = []
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
        annotations.append({
            'pos': ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            'text': f'{length:.1f}',
            'line': (p1, p2),
        })

    if verbose:
        plot_ctx = ax if ax is not None else plt
        plot_ctx.imshow(img, cmap='gray')
        points = np.vstack(list(stitches.values()))
        plot_ctx.scatter(points[:, 0], points[:, 1], s=3, c='red')
        if len(bite_points) > 0:
            bite_points = np.array(bite_points)
            plot_ctx.scatter(bite_points[:, 0], bite_points[:, 1], s=12, c='yellow')
        plot_ctx.plot([a[0], b[0]], [a[1], b[1]], 'blue', linewidth=2.5)
        annotate_plot(plot_ctx, annotations)
        if ax is not None:
            ax.axis('off')
        else:
            plt.axis('off')
            plt.show()
    return count


def extract_excessive_tightening(a, b, stitches, img, cfg, verbose=False, ax=None):
    min_threshold, max_threshold = threshold_by_percentage(stitches, cfg.excessive_tightening_pct)
    partial_points = []
    annotations = []
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
        annotations.append({
            'pos': ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            'text': f'{length:.1f}',
            'line': (p1, p2),
        })

    if verbose:
        plot_ctx = ax if ax is not None else plt
        plot_ctx.imshow(img, cmap='gray')
        points = np.vstack(list(stitches.values()))
        plot_ctx.scatter(points[:, 0], points[:, 1], s=3, c='red')
        if len(partial_points) > 0:
            partial_points = np.array(partial_points)
            plot_ctx.scatter(partial_points[:, 0], partial_points[:, 1], s=12, c='yellow')
        plot_ctx.plot([a[0], b[0]], [a[1], b[1]], 'blue', linewidth=2.5)
        annotate_plot(plot_ctx, annotations)
        if ax is not None:
            ax.axis('off')
        else:
            plt.axis('off')
            plt.show()
    return count


def extract_partial_thickness(a, b, stitches, img, cfg, verbose=False, ax=None):
    count = 0
    vector_ab = np.array([b[0] - a[0], b[1] - a[1]])
    stitches_mean_length = np.mean(calculate_stitch_lengths(stitches))
    partial_points = []
    annotations = []
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
        annotations.append({
            'pos': ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2),
            'text': f'{group_length:.1f}\n{pos}',
            'line': (p1, p2),
        })

    if verbose:
        plot_ctx = ax if ax is not None else plt
        plot_ctx.imshow(img, cmap='gray')
        points = np.vstack(list(stitches.values()))
        plot_ctx.scatter(points[:, 0], points[:, 1], s=3, c='red')
        if len(partial_points) > 0:
            partial_points = np.array(partial_points)
            plot_ctx.scatter(partial_points[:, 0], partial_points[:, 1], s=12, c='yellow')
        plot_ctx.plot([a[0], b[0]], [a[1], b[1]], 'blue', linewidth=2.5)
        annotate_plot(plot_ctx, annotations)
        if ax is not None:
            ax.axis('off')
        else:
            plt.axis('off')
            plt.show()
    return count
