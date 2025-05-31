"""
Preprocessing script for the potato disease dataset.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import logging
import shutil
import random

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.segmentation import contour_detection, segment_image
from src.image_processing import contour, apply_clahe, visualize_image
from src.utils import calcul_dev

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def process_image(image_path, output_path, tmp_dir, config):
    """
    Process a single image through the full pipeline.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the processed image
        tmp_dir: Directory for temporary files
        config: Configuration dictionary
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open image
        with Image.open(image_path) as img:
            im_arr = np.array(img)
        
        # Apply contour detection
        contour_threshold = config['image_processing']['contour']['threshold']
        contours = contour(im_arr, threshold=contour_threshold)
        
        # Save contours to temporary file
        os.makedirs(tmp_dir, exist_ok=True)
        filtered_path = os.path.join(tmp_dir, f"filtered_{os.path.basename(image_path)}")
        cv2.imwrite(filtered_path, contours)
        
        # Segment the image
        cropped_path = os.path.join(tmp_dir, f"cropped_{os.path.basename(image_path)}")
        segment_image(filtered_path, cropped_path, image_path)
        
        # Apply CLAHE to the cropped image
        try:
            with Image.open(cropped_path) as img:
                cropped_arr = np.array(img)
                
            enhanced_arr = apply_clahe(cropped_arr)
            
            # Save the processed image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(enhanced_arr).save(output_path)
            return True
        except Exception as e:
            logger.warning(f"Error in CLAHE processing for {image_path}: {e}")
            # If CLAHE fails, just copy the original image
            shutil.copy(image_path, output_path)
            return True
            
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return False

def split_dataset(input_dir, output_dir, test_split=0.2, val_split=0.1):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        input_dir: Directory containing processed images organized by class
        output_dir: Directory to save the split dataset
        test_split: Fraction of data to use for testing
        val_split: Fraction of data to use for validation
    """
    # Create train, val, test directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get class directories
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for class_name in class_dirs:
        # Create class directories in train, val, test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all images for this class
        class_dir = os.path.join(input_dir, class_name)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle images for random split
        random.seed(42)  # For reproducibility
        random.shuffle(images)
        
        # Calculate split indices
        num_images = len(images)
        num_test = int(num_images * test_split)
        num_val = int(num_images * val_split)
        
        # Split images
        test_images = images[:num_test]
        val_images = images[num_test:num_test+num_val]
        train_images = images[num_test+num_val:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy(
                os.path.join(class_dir, img),
                os.path.join(train_dir, class_name, img)
            )
        
        for img in val_images:
            shutil.copy(
                os.path.join(class_dir, img),
                os.path.join(val_dir, class_name, img)
            )
        
        for img in test_images:
            shutil.copy(
                os.path.join(class_dir, img),
                os.path.join(test_dir, class_name, img)
            )
        
        logger.info(f"Class {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    logger.info(f"Dataset split complete: {train_dir}, {val_dir}, {test_dir}")


def preprocess_dataset(input_dir, output_dir, tmp_dir, config, skip_existing=True):
    """
    Preprocess all images in the dataset.
    
    Args:
        input_dir: Directory containing the dataset organized by class
        output_dir: Directory to save the processed dataset
        tmp_dir: Directory for temporary files
        config: Configuration dictionary
        skip_existing: Skip processing if output image already exists
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Process each class
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    total_images = 0
    processed_images = 0
    
    for class_name in class_dirs:
        class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Get all images for this class
        image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        logger.info(f"Processing class '{class_name}' with {len(image_files)} images")
        total_images += len(image_files)
        
        # Process each image
        for image_file in image_files:
            input_path = os.path.join(class_dir, image_file)
            output_path = os.path.join(output_class_dir, image_file)
            
            # Skip if output already exists and skip_existing is True
            if skip_existing and os.path.exists(output_path):
                processed_images += 1
                continue
            
            success = process_image(input_path, output_path, tmp_dir, config)
            if success:
                processed_images += 1
    
    logger.info(f"Preprocessing complete: {processed_images}/{total_images} images processed successfully")


def main():
    parser = argparse.ArgumentParser(description="Preprocess the potato disease dataset")
    parser.add_argument("--input", type=str, help="Input directory containing the dataset")
    parser.add_argument("--output", type=str, help="Output directory to save the processed dataset")
    parser.add_argument("--processed", type=str, help="Directory to save the processed dataset (before splitting)")
    parser.add_argument("--tmp", type=str, help="Directory for temporary files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip processing if output image already exists")
    parser.add_argument("--split", action="store_true", help="Split the dataset after preprocessing")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Set default directories if not provided
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not args.input:
        args.input = os.path.join(base_dir, "data", "inputs")
    
    if not args.processed:
        args.processed = os.path.join(base_dir, "data", "processed")
    
    if not args.output:
        args.output = os.path.join(base_dir, "data", "split")
    
    if not args.tmp:
        args.tmp = os.path.join(base_dir, "tmp")
    
    # Preprocess the dataset
    logger.info(f"Starting preprocessing from {args.input} to {args.processed}")
    preprocess_dataset(args.input, args.processed, args.tmp, config, args.skip_existing)
    
    # Split the dataset if requested
    if args.split:
        logger.info(f"Splitting dataset from {args.processed} to {args.output}")
        split_dataset(args.processed, args.output)


if __name__ == "__main__":
    main()
