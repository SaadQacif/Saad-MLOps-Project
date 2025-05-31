"""
Image segmentation module for MLOps project.
This module provides functionality for image segmentation and contour detection.
"""

import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class BoundingRect:
    """Represents a rectangular region in an image."""
    x: int
    y: int
    width: int
    height: int


def contour_detection(image_array, threshold=70):
    """
    Detect contours in an image.
    
    Args:
        image_array: numpy array of the image
        threshold: threshold value for edge detection
        
    Returns:
        binary image with contours
    """
    # Convert to grayscale if it's a color image
    if len(image_array.shape) == 3:
        gray_image = np.sum(image_array, axis=2) / 3
    else:
        gray_image = image_array
        
    # Apply morphological operations
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    
    dilated = cv2.dilate(gray_image, kernel, iterations=1)
    eroded = cv2.erode(gray_image, kernel, iterations=1)
    
    # Apply edge detection filter
    edge_kernel = np.array([
        [0., -1., 0.],
        [-1., 4., -1.],
        [0., -1., 0.],
    ])
    
    edges = cv2.filter2D(eroded, -1, np.flip(edge_kernel))
    edges = np.clip(np.floor(edges), 0, 255)
    
    # Apply threshold to get binary image
    binary = (edges >= threshold) * 255
    
    return binary.astype(np.uint8)


def find_bounding_boxes(binary_image):
    """
    Find bounding boxes around contours in a binary image.
    
    Args:
        binary_image: binary image with contours
        
    Returns:
        list of BoundingRect objects
    """
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create bounding rectangles
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append(BoundingRect(x, y, w, h))
    
    return bounding_boxes


def find_main_bounding_box(bounding_boxes):
    """
    Find the main bounding box that contains all other bounding boxes.
    
    Args:
        bounding_boxes: list of BoundingRect objects
        
    Returns:
        BoundingRect object representing the main bounding box
    """
    if not bounding_boxes:
        return None
    
    x_min = min(box.x for box in bounding_boxes)
    y_min = min(box.y for box in bounding_boxes)
    x_max = max(box.x + box.width for box in bounding_boxes)
    y_max = max(box.y + box.height for box in bounding_boxes)
    
    return BoundingRect(x_min, y_min, x_max - x_min, y_max - y_min)


def crop_image(image_path, output_path, bounding_box=None):
    """
    Crop an image based on contour detection or a provided bounding box.
    
    Args:
        image_path: path to the input image
        output_path: path to save the cropped image
        bounding_box: optional BoundingRect to use for cropping
        
    Returns:
        cropped PIL Image object
    """
    # Open the image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Get bounding box if not provided
    if bounding_box is None:
        contours = contour_detection(image_array)
        boxes = find_bounding_boxes(contours)
        bounding_box = find_main_bounding_box(boxes)
    
    if bounding_box is None:
        # If no bounding box was found, return the original image
        return image
    
    # Crop the image
    cropped = image.crop((
        bounding_box.x, 
        bounding_box.y, 
        bounding_box.x + bounding_box.width, 
        bounding_box.y + bounding_box.height
    ))
    
    # Save the cropped image
    if output_path:
        cropped.save(output_path)
    
    return cropped


def segment_image(input_path, output_path, original_path=None):
    """
    Main function to segment an image. This replaces the Rust segmenter functionality.
    
    Args:
        input_path: path to the filtered input image
        output_path: path to save the cropped image
        original_path: optional path to the original image to crop instead of the filtered one
    """
    # Process the filtered image to get the bounding box
    filtered_image = np.array(Image.open(input_path))
    if len(filtered_image.shape) > 2 and filtered_image.shape[2] > 1:
        filtered_image = filtered_image[:, :, 0]  # Use first channel if multi-channel
    
    boxes = find_bounding_boxes(filtered_image)
    main_box = find_main_bounding_box(boxes)
    
    if main_box:
        # If original_path is provided, crop that image instead of the filtered one
        if original_path:
            crop_image(original_path, output_path, main_box)
        else:
            crop_image(input_path, output_path, main_box)
    else:
        # If no bounding box was found, copy the original image
        if original_path:
            image = Image.open(original_path)
        else:
            image = Image.open(input_path)
        image.save(output_path)


if __name__ == "__main__":
    import sys
    
    # Handle command line arguments, similar to the Rust version
    if len(sys.argv) < 3:
        print("Usage: python segmentation.py <input_image> <output_image> [<original_image>]")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_image = sys.argv[2]
    original_image = sys.argv[3] if len(sys.argv) > 3 else None
    
    segment_image(input_image, output_image, original_image)
