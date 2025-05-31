"""
Main module for the MLOps project.
This module processes images and runs predictions using the trained model.
"""

import os
import sys
import shutil
import pickle
from PIL import Image
import numpy as np
import cv2

from segmentation import contour_detection, segment_image
from image_processing import contour, apply_clahe
from utils import calcul_dev


def process_image(image_path, output_dir="tmp"):
    """
    Process a single image through the full pipeline.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to store temporary files
        
    Returns:
        Processed image array and features
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the image
    with Image.open(image_path) as im:
        im_arr = np.asarray(im)
    
    # Filter the image to detect contours
    contours = contour(im_arr)
    filtered_path = os.path.join(output_dir, "filtered.png")
    cv2.imwrite(filtered_path, contours)
    
    # Segment the image
    cropped_path = os.path.join(output_dir, "cropped.png")
    segment_image(filtered_path, cropped_path, image_path)
    
    # Open the cropped image
    with Image.open(cropped_path) as im:
        im_arr = np.asarray(im)
    
    # Apply CLAHE for contrast enhancement
    enhanced_arr = apply_clahe(im_arr)
    
    # Extract features
    features = calcul_dev(enhanced_arr)
    
    return enhanced_arr, features


def predict(features, model_path, encoder_path):
    """
    Make predictions based on extracted features.
    
    Args:
        features: Extracted image features
        model_path: Path to the trained model
        encoder_path: Path to the label encoder
        
    Returns:
        Predicted class label
    """
    # Load the model and encoder
    with open(model_path, 'rb') as f:
        models_by_features = pickle.load(f)
    
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    
    # Make prediction
    prediction = models_by_features['rgb'].predict(features.reshape(1, -1))
    prediction_label = encoder.inverse_transform(prediction)[0]
    
    return prediction_label


def main():
    """Main function to process images and make predictions."""
    # Define default paths
    default_images_path = "data/inputs/"
    model_path = "models/bin/models_by_features.pkl"
    encoder_path = "models/bin/encoder.pkl"
    tmp_dir = "tmp"
    
    # Parse command line arguments
    images_path = default_images_path
    if len(sys.argv) > 1:
        images_path = sys.argv[1]
    
    # Create temporary directory
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Get list of image files
    files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    
    if not files:
        print(f"No files found in {images_path}")
        return
    
    # Process each image
    log = ""
    results = []
    
    for file_name in files:
        file_path = os.path.join(images_path, file_name)
        print(f"Processing {file_name}...")
        
        try:
            # Process the image
            _, features = process_image(file_path, tmp_dir)
            
            # Make prediction
            prediction = predict([features], model_path, encoder_path)
            
            # Log the result
            log_entry = f"Image: {file_name}, Prediction: {prediction}"
            print(log_entry)
            log += log_entry + "\n"
            
            results.append({
                'file_name': file_name,
                'prediction': prediction
            })
            
        except Exception as e:
            error_msg = f"Error processing {file_name}: {str(e)}"
            print(error_msg)
            log += error_msg + "\n"
    
    # Write log to file
    with open("data/outputs/log.txt", "w") as f:
        f.write(log)
    
    # Clean up temporary files
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
    return results


if __name__ == "__main__":
    main()
