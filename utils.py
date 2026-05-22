import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_color_features(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hist_h = cv2.calcHist([hsv],[0],None,[32],[0,256])
    hist_s = cv2.calcHist([hsv],[1],None,[32],[0,256])
    hist_v = cv2.calcHist([hsv],[2],None,[32],[0,256])

    feature = np.concatenate([
        hist_h.flatten(),
        hist_s.flatten(),
        hist_v.flatten()
    ])

    feature = feature / np.sum(feature)

    return feature


def extract_lbp_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10)
    )

    hist = hist.astype('float')

    hist /= (hist.sum() + 1e-6)

    return hist


def extract_shape_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    edge_density = np.sum(edges > 0) / edges.size

    h, w = gray.shape

    aspect_ratio = w / h

    return np.array([
        edge_density,
        aspect_ratio
    ])
