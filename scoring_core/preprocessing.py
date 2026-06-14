"""Image preprocessing and suture-line extraction (forked from the upstream detection pipeline).

The pixel operations match the validated pipeline; the only change is that the
entry point also accepts an in-memory BGR image, not just a file path.
"""
import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import gaussian, frangi, threshold_otsu

from .config import PREPROCESS_PARAMS, IMG_SIZE
from .utils import resize_with_white_pad


def preprocess_image(img_color):
    """Run the deterministic preprocessing chain on a BGR uint8 image.

    Returns (img_orig, mask): the normalized grayscale image and the binary
    vesselness mask used for skeletonization.
    """
    if img_color is None:
        raise ValueError("Could not read the image.")
    if img_color.ndim == 2:
        img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
    elif img_color.shape[2] == 4:
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGRA2BGR)

    img_color = cv2.rotate(img_color, cv2.ROTATE_90_CLOCKWISE) if img_color.shape[0] > img_color.shape[1] else img_color
    img_color = resize_with_white_pad(img_color, *IMG_SIZE)

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_orig = img_gray.copy()

    p = dict(PREPROCESS_PARAMS)

    # Mask specular highlights (glare) and inpaint them away.
    img_lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
    lower_bound = np.array([p['lightning'], 0, 0])
    upper_bound = np.array([255, 255, 255])
    mask = cv2.inRange(img_lab, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p['dilation'], p['dilation']))
    mask = cv2.dilate(mask, kernel, iterations=1)
    img = cv2.inpaint(img_gray, mask, p['radius'], cv2.INPAINT_TELEA)

    # Morphological opening then smoothing.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p['open_w'], p['open_h']))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = gaussian(img, sigma=p['gaussian'])

    # Black-hat to bring out the dark suture threads, then vesselness.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p['blackhat'], p['blackhat']))
    img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    img = frangi(
        img,
        sigmas=range(p['frangi_min'], p['frangi_max']),
        alpha=0.5,
        beta=0.5,
        gamma=None,
        black_ridges=False,
    )

    thresh = threshold_otsu(img)
    mask = remove_small_objects(img > thresh, max_size=p["small_objects"])
    return img_orig, mask


def preprocess(img_path):
    """File-path entry point (handy for CLI / tests)."""
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return preprocess_image(img_color)


def extract_anast_line_and_stitches(mask):
    """Skeletonize the mask, fit the anastomotic line by PCA, and split the
    skeleton into individual stitches via connected components.

    Returns (ptA, ptB, stitches) where stitches maps an index -> Nx2 point array,
    sorted left-to-right by centroid x.
    """
    skel = skeletonize(mask != 0)
    ys, xs = np.nonzero(skel)
    if xs.size == 0:
        raise ValueError("No suture structure detected after preprocessing.")
    points = np.column_stack([xs, ys])

    # PCA -> dominant axis = anastomotic line.
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]

    projections = centered @ direction
    t_min, t_max = projections.min(), projections.max()
    ptA = mean + t_min * direction
    ptB = mean + t_max * direction

    # Connected components -> stitches.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        skel.astype(np.uint8) * 255, connectivity=8
    )
    stitches = []
    for label in range(1, num_labels):
        ys, xs = np.nonzero(labels == label)
        pts = np.column_stack([xs, ys])
        stitches.append({'pts': pts, 'centroid': centroids[label], 'area': stats[label, cv2.CC_STAT_AREA]})
    stitches.sort(key=lambda s: s['centroid'][0])
    final_stitches = {i: s['pts'] for i, s in enumerate(stitches)}

    return tuple(ptA), tuple(ptB), final_stitches
