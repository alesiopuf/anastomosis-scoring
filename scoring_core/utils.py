import numpy as np
import cv2
from math import degrees, acos


def angle_between_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return degrees(acos(dot_product))


def calculate_string_length(group):
    """Stitch length: ``(p1, p2, distance)`` for the two farthest-apart points."""
    pts = np.asarray(group, dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)
    i, j = np.unravel_index(dmat.argmax(), dmat.shape)
    return pts[i], pts[j], float(dmat.max())


def calculate_stitch_lengths(stitches):
    return [calculate_string_length(group)[2] for group in stitches.values()]


def threshold_by_percentage(stitches, percent):
    stitch_lengths = calculate_stitch_lengths(stitches)
    median = np.median(stitch_lengths)
    allowed_error_size = median * percent
    min_threshold = median - allowed_error_size
    max_threshold = median + allowed_error_size
    return min_threshold, max_threshold


def resize_with_white_pad(img, target_width, target_height):
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if len(img.shape) == 3:
        canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    else:
        canvas = np.full((target_height, target_width), 255, dtype=np.uint8)
    x_off = (target_width - new_w) // 2
    y_off = (target_height - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas
