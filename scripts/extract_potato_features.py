"""
Extract features from potato leaf images for disease classification
based on the features identified in the notebooks.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from skimage.measure import shannon_entropy
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops, label

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import calcul_dev

def extract_rgb_features(image):
    """
    Extract RGB statistics from an image.
    
    Args:
        image: Image array in RGB format
        
    Returns:
        Dictionary of RGB features
    """
    # Extract standard deviations using calcul_dev
    std_red, std_green, std_blue = calcul_dev(image)
    
    # Calculate means for each channel
    mean_red = np.mean(image[:, :, 0])
    mean_green = np.mean(image[:, :, 1])
    mean_blue = np.mean(image[:, :, 2])
    
    return {
        'std_red': std_red,
        'std_green': std_green,
        'std_blue': std_blue,
        'mean_red': mean_red,
        'mean_green': mean_green,
        'mean_blue': mean_blue
    }

def extract_entropy_features(image):
    """
    Extract entropy-based features from an image.
    
    Args:
        image: Image array in RGB format
        
    Returns:
        Dictionary of entropy features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    # Calculate entropy
    entropy = shannon_entropy(gray_image)
    
    # Calculate histogram
    hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
    hist_prob = hist / hist.sum()
    
    # Calculate entropy variation rhythm (standard deviation of entropy gradient)
    entropy_variation_rhythm = np.std(np.gradient(hist_prob))
    
    return {
        'entropy': entropy,
        'entropy_variation_rhythm': entropy_variation_rhythm
    }

def extract_shape_features(image):
    """
    Extract shape-based features from an image.
    
    Args:
        image: Image array in RGB format
        
    Returns:
        Dictionary of shape features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    # Threshold the image to create a binary mask
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate convex hull
    try:
        hull = convex_hull_image(binary > 0)
        
        # Calculate convexity ratio
        if np.sum(hull) > 0:
            convexity_ratio = np.sum(binary > 0) / np.sum(hull)
        else:
            convexity_ratio = 0
    except Exception:
        convexity_ratio = 0
    
    # Extract more shape properties
    try:
        labeled_image = label(binary > 0)
        props = regionprops(labeled_image)
        
        if props:
            largest_region = max(props, key=lambda r: r.area)
            eccentricity = largest_region.eccentricity
            solidity = largest_region.solidity
            
            if largest_region.major_axis_length > 0:
                aspect_ratio = largest_region.minor_axis_length / largest_region.major_axis_length
            else:
                aspect_ratio = 0
        else:
            eccentricity = 0
            solidity = 0
            aspect_ratio = 0
    except Exception:
        eccentricity = 0
        solidity = 0
        aspect_ratio = 0
    
    return {
        'convexity_ratio': convexity_ratio,
        'eccentricity': eccentricity,
        'solidity': solidity,
        'aspect_ratio': aspect_ratio
    }

def extract_hog_features(image, reduced=True):
    """
    Extract HOG features from an image.
    If reduced=True, returns a simplified version with fewer features.
    
    Args:
        image: Image array in RGB format
        reduced: If True, reduce the dimensionality of HOG features
        
    Returns:
        Dictionary of HOG features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    # Resize for consistent HOG features
    resized = cv2.resize(gray_image, (128, 128))
    
    # Calculate HOG features
    try:
        from skimage.feature import hog
        
        hog_features = hog(
            resized,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        
        if reduced:
            # Apply dimensionality reduction as shown in the feature_selection notebook
            kernel = np.ones(3)
            pool_size = 2
            
            # Flatten if needed
            if len(hog_features.shape) > 1:
                hog_features = hog_features.flatten()
            
            # Apply convolution and max pooling
            for _ in range(4):
                hog_features = np.convolve(hog_features, kernel, mode='valid')
                hog_features = np.array([max(hog_features[i:i + pool_size]) 
                                        for i in range(0, len(hog_features), pool_size)])
            
            # Return dict with reduced HOG features
            return {f'hog_{i}': val for i, val in enumerate(hog_features)}
        
        else:
            # Return full HOG features
            return {f'hog_{i}': val for i, val in enumerate(hog_features)}
    
    except Exception:
        # Return empty dict if HOG extraction fails
        return {}

def extract_all_features(image_path):
    """
    Extract all features from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary of all features
    """
    try:
        # Load the image
        with Image.open(image_path) as img:
            image = np.array(img)
        
        return extract_all_features_from_array(image)
    
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def extract_all_features_from_array(image_array):
    """
    Extract all features from an image array.
    Compatible with the 30-feature training data format.
    
    Args:
        image_array: Image as numpy array (RGB format)
        
    Returns:
        Dictionary of all features (30 features total)
    """
    try:
        # Use the original calcul_dev function to get r_dev, g_dev, b_dev
        from src.utils import calcul_dev
        r_dev, g_dev, b_dev = calcul_dev(image_array)
        
        # Initialize features dictionary with the exact 30 features from training data
        features = {
            'r_dev': r_dev,
            'g_dev': g_dev, 
            'b_dev': b_dev,
            'red_mean': float(np.mean(image_array[:, :, 0])),
            'red_std': float(np.std(image_array[:, :, 0])),
            'red_kurtosis': 0.0,  # Placeholder for now
            'red_skew': 0.0,      # Placeholder for now
            'green_mean': float(np.mean(image_array[:, :, 1])),
            'green_std': float(np.std(image_array[:, :, 1])),
            'green_kurtosis': 0.0,  # Placeholder for now
            'green_skew': 0.0,      # Placeholder for now
            'blue_mean': float(np.mean(image_array[:, :, 2])),
            'blue_std': float(np.std(image_array[:, :, 2])),
            'blue_kurtosis': 0.0,   # Placeholder for now
            'blue_skew': 0.0,       # Placeholder for now
            'green_ratio': 0.0,     # Placeholder for now
            'diseased_ratio': 0.0,  # Placeholder for now
            'entropy_mean': 0.0,    # Placeholder for now
            'entropy_std': 0.0,     # Placeholder for now
            'entropy_variation_rhythm': 0.0,  # Placeholder for now
            'texture_contrast': 0.0,          # Placeholder for now
            'avg_convexity': 0.0,             # Placeholder for now
            'min_convexity': 0.0,             # Placeholder for now
            'contour_area': 0.0,              # Placeholder for now
            'contour_perimeter': 0.0,         # Placeholder for now
            'circularity': 0.0,               # Placeholder for now
            'eccentricity': 0.0,              # Placeholder for now
            'solidity': 0.0,                  # Placeholder for now
            'aspect_ratio': 1.0,              # Placeholder for now
            'area_bbox_ratio': 0.0            # Placeholder for now
        }
        
        # Try to compute actual values for some features
        try:
            # Entropy features
            entropy_features = extract_entropy_features(image_array)
            if entropy_features:
                features['entropy_variation_rhythm'] = entropy_features.get('entropy_variation_rhythm', 0.0)
            
            # Shape features
            shape_features = extract_shape_features(image_array)
            if shape_features:
                features['eccentricity'] = shape_features.get('eccentricity', 0.0)
                features['solidity'] = shape_features.get('solidity', 0.0)
                features['aspect_ratio'] = shape_features.get('aspect_ratio', 1.0)
                features['convexity_ratio'] = shape_features.get('convexity_ratio', 0.0)
                features['avg_convexity'] = features['convexity_ratio']  # Use same value
                
        except Exception as e:
            print(f"Warning: Could not compute some features: {e}")
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from image array: {e}")
        return None

def extract_features_from_directory(input_dir, output_file):
    """
    Extract features from all images in a directory structure.
    
    Args:
        input_dir: Directory containing class subdirectories with images
        output_file: Path to save the output CSV file
    """
    features_list = []
    
    # Get class directories
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for class_name in class_dirs:
        class_dir = os.path.join(input_dir, class_name)
        print(f"Processing class: {class_name}")
        
        # Process each image in the class directory
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for i, img_file in enumerate(image_files):
            if i % 100 == 0:
                print(f"  Processing image {i+1}/{len(image_files)}")
            
            img_path = os.path.join(class_dir, img_file)
            
            # Extract features
            features = extract_all_features(img_path)
            
            if features:
                # Add class and image path
                features['class'] = class_name
                features['image_path'] = img_path
                
                # Add to list
                features_list.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Features extracted and saved to {output_file}")
    print(f"Total samples: {len(df)}")

def extract_features(cv2_image):
    """
    Extract features from a CV2 image (for use with training scripts).
    Returns exactly 30 features to match the training data.
    
    Args:
        cv2_image: Image loaded with cv2.imread (BGR format)
        
    Returns:
        numpy array of 30 features or None if error
    """
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        # Extract all features using the existing function
        features_dict = extract_all_features_from_array(rgb_image)
        
        if features_dict is None:
            return None
            
        # Define the exact order of features as in training data
        feature_order = [
            'r_dev', 'g_dev', 'b_dev', 'red_mean', 'red_std', 'red_kurtosis', 'red_skew',
            'green_mean', 'green_std', 'green_kurtosis', 'green_skew', 'blue_mean', 'blue_std',
            'blue_kurtosis', 'blue_skew', 'green_ratio', 'diseased_ratio', 'entropy_mean',
            'entropy_std', 'entropy_variation_rhythm', 'texture_contrast', 'avg_convexity',
            'min_convexity', 'contour_area', 'contour_perimeter', 'circularity', 'eccentricity',
            'solidity', 'aspect_ratio', 'area_bbox_ratio'
        ]
        
        # Convert to numpy array in the correct order
        feature_values = [features_dict.get(name, 0.0) for name in feature_order]
        
        return np.array(feature_values, dtype=np.float32)
        
    except Exception as e:
        print(f"Error in extract_features: {e}")
        return None

def get_feature_names():
    """
    Get the names of all features in the same order as extract_features returns them.
    Returns the exact 30 feature names from the training data.
    
    Returns:
        List of 30 feature names
    """
    return [
        'r_dev', 'g_dev', 'b_dev', 'red_mean', 'red_std', 'red_kurtosis', 'red_skew',
        'green_mean', 'green_std', 'green_kurtosis', 'green_skew', 'blue_mean', 'blue_std',
        'blue_kurtosis', 'blue_skew', 'green_ratio', 'diseased_ratio', 'entropy_mean',
        'entropy_std', 'entropy_variation_rhythm', 'texture_contrast', 'avg_convexity',
        'min_convexity', 'contour_area', 'contour_perimeter', 'circularity', 'eccentricity',
        'solidity', 'aspect_ratio', 'area_bbox_ratio'
    ]

def main():
    parser = argparse.ArgumentParser(description="Extract features from potato leaf images")
    parser.add_argument("--input", type=str, required=True, help="Input directory with class subdirectories")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    
    args = parser.parse_args()
    
    extract_features_from_directory(args.input, args.output)

if __name__ == "__main__":
    main()
