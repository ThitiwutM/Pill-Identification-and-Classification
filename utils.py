import numpy as np

from PIL import Image

from skimage.color import rgb2gray

from skimage.feature import (
    local_binary_pattern,
    canny
)

# =========================
# COLOR FEATURES
# =========================

def extract_color_features(image):

    image = image.astype(np.float32) / 255.0

    mean_rgb = np.mean(
        image,
        axis=(0,1)
    )

    std_rgb = np.std(
        image,
        axis=(0,1)
    )

    feature = np.concatenate([
        mean_rgb,
        std_rgb
    ])

    return feature

# =========================
# TEXTURE FEATURES
# =========================

def extract_lbp_features(image):

    gray = rgb2gray(image)

    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0,11),
        range=(0,10)
    )

    hist = hist.astype("float")

    hist /= (hist.sum() + 1e-6)

    return hist

# =========================
# SHAPE FEATURES
# =========================

def extract_shape_features(image):

    gray = rgb2gray(image)

    edges = canny(gray)

    edge_density = np.mean(edges)

    h, w = gray.shape

    aspect_ratio = w / h

    return np.array([
        edge_density,
        aspect_ratio
    ])
