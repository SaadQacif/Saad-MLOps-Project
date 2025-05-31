"""
Utility functions for the MLOps project.
"""

import numpy as np
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops
from skimage import filters
from skimage.measure import shannon_entropy
from skimage.util import view_as_blocks


def dimensionality_reduction(X_train):
    """
    Reduce the dimensionality of the input data.
    
    Args:
        X_train: Training data
        
    Returns:
        Reduced dimensionality data
    """
    # Placeholder for dimensionality reduction logic
    return X_train


def calcul_dev(img):
    """
    Calculate the RGB standard deviation for an image.
    
    Args:
        img: Input image array
        
    Returns:
        Array of RGB standard deviations
    """
    r = np.std(img[:, :, 0])
    g = np.std(img[:, :, 1])
    b = np.std(img[:, :, 2])
    
    return np.array([r, g, b])


def calculate_ch(image):
    """
    Calculate the convex hull ratio for an image.
    
    Args:
        image: Input image array
        
    Returns:
        Mean convex hull ratio
    """
    threshold = filters.threshold_otsu(image)
    bin_image = image > threshold

    props = regionprops(bin_image.astype(int))

    convex_rat = []

    for prop in props:
        cov = convex_hull_image(prop.image)
        oba = prop.area
        h_a = np.sum(cov)

        if h_a > 0:
            convex_rat.append(oba / h_a)
    
    return np.mean(convex_rat) if convex_rat else 0


def calculate_entropy_variation_rhythm(image, window_size=8):
    """
    Calculate the entropy variation rhythm for an image.
    
    Args:
        image: Input image array
        window_size: Size of the window for entropy calculation
        
    Returns:
        Entropy map
    """
    blocks = view_as_blocks(image, block_shape=(window_size, window_size))
    entropy_map = np.zeros(blocks.shape[:2])

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            entropy_map[i, j] = shannon_entropy(blocks[i, j])
    
    return entropy_map
