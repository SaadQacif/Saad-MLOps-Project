"""
Image processing functions for the MLOps project.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_monochrome(x):
    """
    Create a histogram for monochrome data.
    
    Args:
        x: Input data
        
    Returns:
        Tuple of (labels, counts)
    """
    buckets = [0 for _ in range(256)]
    labels = np.array([i for i in range(256)])

    for i in x:
        buckets[int(i)] += 1

    return labels, np.array(buckets)


def contour(im_arr, threshold=70, gray=False):
    """
    Detect contours in an image.
    
    Args:
        im_arr: Image array
        threshold: Threshold for edge detection
        gray: Whether the input image is grayscale
        
    Returns:
        Binary image with contours
    """
    if gray:
        gr_im_arr = im_arr
    else:
        gr_im_arr = np.sum(im_arr, axis=2) / 3

    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    
    dst = cv2.dilate(gr_im_arr, kernel, iterations=1)
    dst = cv2.erode(gr_im_arr, kernel, iterations=1)
    
    kernel = np.array([
        [0., -1., 0.],
        [-1., 4., -1.],
        [0., -1., 0.],
    ])
    
    dst = cv2.filter2D(dst, -1, np.flip(kernel))
    dst = np.clip(np.floor(dst), 0, 255)

    dst = (dst >= threshold) * 255

    return dst


def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    
    Args:
        image: Input image
        
    Returns:
        Image with CLAHE applied
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to lightness channel
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
    
    # Convert back to RGB
    enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    
    return enhanced_image


def visualize_image(image, title="Image", save_path=None):
    """
    Visualize an image.
    
    Args:
        image: Input image
        title: Title for the plot
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
