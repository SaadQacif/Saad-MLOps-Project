"""
Data processing script for the MLOps project.
This script handles data preparation, augmentation, and preprocessing.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from segmentation import contour_detection, segment_image
from image_processing import contour, apply_clahe


def load_config():
    """Load configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def preprocess_images(input_dir, output_dir, config):
    """
    Preprocess images from input directory and save to output directory.
    
    Args:
        input_dir: Path to input directory
        output_dir: Path to output directory
        config: Configuration dictionary
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_file in image_files:
        try:
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, f"processed_{image_file}")
            
            print(f"Processing {image_file}...")
            
            # Open image
            with Image.open(input_path) as img:
                im_arr = np.array(img)
            
            # Apply contour detection
            contour_threshold = config['image_processing']['contour']['threshold']
            contours = contour(im_arr, threshold=contour_threshold)
            
            # Create temporary directory for intermediate files
            tmp_dir = os.path.join(os.path.dirname(__file__), '..', config['paths']['data']['temp'])
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Save contours to temporary file
            filtered_path = os.path.join(tmp_dir, f"filtered_{image_file}")
            cv2.imwrite(filtered_path, contours)
            
            # Segment the image
            cropped_path = os.path.join(tmp_dir, f"cropped_{image_file}")
            segment_image(filtered_path, cropped_path, input_path)
            
            # Apply CLAHE to the cropped image
            with Image.open(cropped_path) as img:
                cropped_arr = np.array(img)
                
            enhanced_arr = apply_clahe(cropped_arr)
            
            # Save the processed image
            Image.fromarray(enhanced_arr).save(output_path)
            
            print(f"Saved processed image to {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
    
    print("Preprocessing complete!")


def augment_images(input_dir, output_dir, augmentation_factor=3):
    """
    Augment images from input directory and save to output directory.
    
    Args:
        input_dir: Path to input directory
        output_dir: Path to output directory
        augmentation_factor: Number of augmented images to create per original image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to augment")
    
    # Process each image
    for image_file in image_files:
        try:
            input_path = os.path.join(input_dir, image_file)
            
            # Open image
            with Image.open(input_path) as img:
                im_arr = np.array(img)
            
            # Generate augmentations
            for i in range(augmentation_factor):
                # Apply random transformations
                augmented = apply_augmentation(im_arr)
                
                # Generate output filename
                base_name, ext = os.path.splitext(image_file)
                output_path = os.path.join(output_dir, f"{base_name}_aug{i}{ext}")
                
                # Save augmented image
                Image.fromarray(augmented).save(output_path)
                
                print(f"Saved augmented image to {output_path}")
                
        except Exception as e:
            print(f"Error augmenting {image_file}: {str(e)}")
    
    print("Augmentation complete!")


def apply_augmentation(image):
    """
    Apply random augmentations to an image.
    
    Args:
        image: Input image array
        
    Returns:
        Augmented image array
    """
    # Create a copy of the image
    img = image.copy()
    
    # Randomly apply transformations
    np.random.seed()
    
    # Random rotation
    if np.random.random() > 0.5:
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Random brightness adjustment
    if np.random.random() > 0.5:
        brightness = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    
    # Random horizontal flip
    if np.random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    return img


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Data processing script")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess images")
    parser.add_argument("--augment", action="store_true", help="Augment images")
    parser.add_argument("--input", type=str, help="Input directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--factor", type=int, default=3, help="Augmentation factor")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Set default directories if not provided
    if not args.input:
        args.input = os.path.join(os.path.dirname(__file__), '..', config['paths']['data']['input'])
    if not args.output:
        args.output = os.path.join(os.path.dirname(__file__), '..', config['paths']['data']['output'])
    
    # Run selected operations
    if args.preprocess:
        preprocess_images(args.input, args.output, config)
    
    if args.augment:
        augment_images(args.input, args.output, args.factor)
    
    # If no operation selected, print help
    if not (args.preprocess or args.augment):
        parser.print_help()


if __name__ == "__main__":
    main()
