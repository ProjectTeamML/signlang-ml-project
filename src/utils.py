# utils.py
import cv2
import numpy as np
from skimage.feature import hog
from collections import Counter

def compute_hog_features(image, hog_params):
    """
    Extract HOG features from a single grayscale image.
    """
    features = hog(
        image,
        orientations=hog_params.get('orientations', 12),
        pixels_per_cell=hog_params.get('pixels_per_cell', (4,4)),
        cells_per_block=hog_params.get('cells_per_block', (2,2)),
        block_norm='L2-Hys',
        visualize=False
    )
    return features.reshape(1, -1)

labels_map = {i: chr(65+i) for i in range(26)}  # A-Z

def decode_prediction(pred_raw):
    """Convert numeric prediction into alphabet."""
    if isinstance(pred_raw, (np.ndarray, list)):
        pred_raw = int(pred_raw[0])
    return labels_map.get(pred_raw, str(pred_raw))

def make_square_crop(x_min, y_min, x_max, y_max, frame_w, frame_h):
    """
    Convert a rectangle to square while keeping center.
    """
    box_w, box_h = x_max - x_min, y_max - y_min
    size = max(box_w, box_h)
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_min_new = max(x_center - size // 2, 0)
    y_min_new = max(y_center - size // 2, 0)
    x_max_new = min(x_center + size // 2, frame_w)
    y_max_new = min(y_center + size // 2, frame_h)
    return x_min_new, y_min_new, x_max_new, y_max_new

def most_common_prediction(buffer):
    """Return the most frequent element in a deque buffer."""
    if not buffer:
        return None
    return Counter(buffer).most_common(1)[0][0]