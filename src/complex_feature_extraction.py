"""
Complex Feature Extraction Module for Potato Disease Classification

This module implements sophisticated feature extraction techniques including:
1. Advanced HOG features with convolution and max pooling
2. Multi-scale texture features
3. Wavelet-based features
4. Color space analysis
5. Morphological features
6. Statistical moments
7. Edge and contour features
"""

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy, regionprops, label
from skimage.morphology import convex_hull_image, opening, closing, erosion, dilation
from skimage.filters import gabor, sobel, roberts, prewitt
from skimage.segmentation import felzenszwalb, slic
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter
import pywt
from typing import Dict, List, Tuple, Optional
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class ComplexFeatureExtractor:
    """
    Advanced feature extractor for potato leaf images incorporating multiple
    sophisticated computer vision and signal processing techniques.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the complex feature extractor.
        
        Args:
            target_size: Target size for image resizing (width, height)
        """
        self.target_size = target_size
        self.gabor_frequencies = [0.1, 0.3, 0.5]
        self.gabor_angles = [0, 45, 90, 135]
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for feature extraction.
        
        Args:
            image: Input image (RGB or BGR)
            
        Returns:
            Tuple of (rgb_image, gray_image)
        """
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if BGR (OpenCV format) and convert to RGB
            if np.mean(image[:, :, 2]) < np.mean(image[:, :, 0]):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize image
        rgb_image = cv2.resize(rgb_image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1]
        rgb_image = rgb_image.astype(np.float32) / 255.0
        gray_image = gray_image.astype(np.float32) / 255.0
        
        return rgb_image, gray_image
    
    def extract_advanced_hog_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Extract advanced HOG features with convolution and max pooling
        as discovered in the feature selection notebook.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Reduced HOG feature vector
        """
        # Extract standard HOG features
        hog_features = hog(
            gray_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True
        )
        
        # Apply convolution and max pooling as in the notebook
        kernel = np.ones(3)
        pool_size = 2
        
        # Apply 4 iterations of convolution + max pooling
        for _ in range(4):
            # Convolution
            hog_features = np.convolve(hog_features, kernel, mode='valid')
            
            # Max pooling
            hog_features = np.array([
                max(hog_features[i:i + pool_size]) 
                for i in range(0, len(hog_features), pool_size)
            ])
        
        return hog_features
    
    def extract_wavelet_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet-based features using discrete wavelet transform.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary of wavelet features
        """
        features = {}
        
        try:
            # Multi-level wavelet decomposition
            wavelets = ['db4', 'haar', 'coif2']
            
            for wavelet_name in wavelets:
                coeffs = pywt.wavedec2(gray_image, wavelet_name, level=3)
                
                # Extract statistics from each decomposition level
                for i, coeff in enumerate(coeffs):
                    if i == 0:  # Approximation coefficients
                        prefix = f'{wavelet_name}_approx'
                        data = coeff
                    else:  # Detail coefficients (horizontal, vertical, diagonal)
                        for j, detail in enumerate(coeff):
                            prefix = f'{wavelet_name}_detail_{i}_{j}'
                            data = detail
                            
                            features.update({
                                f'{prefix}_mean': np.mean(data),
                                f'{prefix}_std': np.std(data),
                                f'{prefix}_energy': np.sum(data**2),
                                f'{prefix}_entropy': shannon_entropy(np.abs(data))
                            })
                
        except Exception as e:
            # Fill with zeros if wavelet decomposition fails
            for wavelet_name in ['db4', 'haar', 'coif2']:
                for i in range(1, 4):
                    for j in range(3):
                        prefix = f'{wavelet_name}_detail_{i}_{j}'
                        features.update({
                            f'{prefix}_mean': 0.0,
                            f'{prefix}_std': 0.0,
                            f'{prefix}_energy': 0.0,
                            f'{prefix}_entropy': 0.0
                        })
        
        return features
    
    def extract_texture_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Extract advanced texture features using multiple methods.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary of texture features
        """
        features = {}
        
        # Convert to uint8 for GLCM
        gray_uint8 = (gray_image * 255).astype(np.uint8)
        
        try:
            # Gray Level Co-occurrence Matrix (GLCM) features
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]
            
            for distance in distances:
                glcm = greycomatrix(
                    gray_uint8, [distance], 
                    np.radians(angles), 
                    levels=256, symmetric=True, normed=True
                )
                
                # Extract GLCM properties
                properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
                for prop in properties:
                    values = greycoprops(glcm, prop)
                    features[f'glcm_{prop}_d{distance}'] = np.mean(values)
            
            # Local Binary Pattern (LBP)
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
            
            # LBP histogram
            lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-8)
            
            # LBP statistics
            features.update({
                'lbp_uniformity': np.sum(lbp_hist**2),
                'lbp_entropy': shannon_entropy(lbp_hist),
                'lbp_mean': np.mean(lbp),
                'lbp_std': np.std(lbp)
            })
            
        except Exception as e:
            # Fill with default values if texture extraction fails
            for distance in [1, 2, 3]:
                for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy']:
                    features[f'glcm_{prop}_d{distance}'] = 0.0
            
            features.update({
                'lbp_uniformity': 0.0,
                'lbp_entropy': 0.0,
                'lbp_mean': 0.0,
                'lbp_std': 0.0
            })
        
        return features
    
    def extract_gabor_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Extract Gabor filter features for texture analysis.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary of Gabor features
        """
        features = {}
        
        try:
            for freq in self.gabor_frequencies:
                for angle in self.gabor_angles:
                    # Apply Gabor filter
                    filtered_real, filtered_imag = gabor(
                        gray_image, frequency=freq, 
                        theta=np.radians(angle)
                    )
                    
                    # Calculate magnitude
                    magnitude = np.sqrt(filtered_real**2 + filtered_imag**2)
                    
                    # Extract statistics
                    prefix = f'gabor_f{freq}_a{angle}'
                    features.update({
                        f'{prefix}_mean': np.mean(magnitude),
                        f'{prefix}_std': np.std(magnitude),
                        f'{prefix}_energy': np.sum(magnitude**2),
                        f'{prefix}_entropy': shannon_entropy(magnitude)
                    })
        
        except Exception as e:
            # Fill with zeros if Gabor filtering fails
            for freq in self.gabor_frequencies:
                for angle in self.gabor_angles:
                    prefix = f'gabor_f{freq}_a{angle}'
                    features.update({
                        f'{prefix}_mean': 0.0,
                        f'{prefix}_std': 0.0,
                        f'{prefix}_energy': 0.0,
                        f'{prefix}_entropy': 0.0
                    })
        
        return features
    
    def extract_edge_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Extract edge-based features using multiple edge detectors.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary of edge features
        """
        features = {}
        
        try:
            # Apply different edge detectors
            edges_sobel = sobel(gray_image)
            edges_roberts = roberts(gray_image)
            edges_prewitt = prewitt(gray_image)
            
            # Canny edge detection
            gray_uint8 = (gray_image * 255).astype(np.uint8)
            edges_canny = cv2.Canny(gray_uint8, 50, 150) / 255.0
            
            edge_methods = {
                'sobel': edges_sobel,
                'roberts': edges_roberts,
                'prewitt': edges_prewitt,
                'canny': edges_canny
            }
            
            for method_name, edges in edge_methods.items():
                # Edge density
                edge_density = np.sum(edges) / edges.size
                
                # Edge strength statistics
                edge_mean = np.mean(edges)
                edge_std = np.std(edges)
                edge_max = np.max(edges)
                
                features.update({
                    f'{method_name}_density': edge_density,
                    f'{method_name}_mean': edge_mean,
                    f'{method_name}_std': edge_std,
                    f'{method_name}_max': edge_max
                })
        
        except Exception as e:
            # Fill with zeros if edge detection fails
            for method in ['sobel', 'roberts', 'prewitt', 'canny']:
                features.update({
                    f'{method}_density': 0.0,
                    f'{method}_mean': 0.0,
                    f'{method}_std': 0.0,
                    f'{method}_max': 0.0
                })
        
        return features
    
    def extract_color_features(self, rgb_image: np.ndarray) -> Dict[str, float]:
        """
        Extract advanced color features from multiple color spaces.
        
        Args:
            rgb_image: RGB image
            
        Returns:
            Dictionary of color features
        """
        features = {}
        
        try:
            # RGB features
            for i, channel in enumerate(['red', 'green', 'blue']):
                channel_data = rgb_image[:, :, i]
                features.update({
                    f'{channel}_mean': np.mean(channel_data),
                    f'{channel}_std': np.std(channel_data),
                    f'{channel}_skew': skew(channel_data.flatten()),
                    f'{channel}_kurtosis': kurtosis(channel_data.flatten()),
                    f'{channel}_min': np.min(channel_data),
                    f'{channel}_max': np.max(channel_data),
                    f'{channel}_range': np.max(channel_data) - np.min(channel_data)
                })
            
            # Convert to other color spaces
            rgb_uint8 = (rgb_image * 255).astype(np.uint8)
            
            # HSV color space
            hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
            for i, channel in enumerate(['hue', 'saturation', 'value']):
                channel_data = hsv[:, :, i].astype(np.float32)
                features.update({
                    f'hsv_{channel}_mean': np.mean(channel_data),
                    f'hsv_{channel}_std': np.std(channel_data),
                    f'hsv_{channel}_entropy': shannon_entropy(channel_data.astype(np.uint8))
                })
            
            # LAB color space
            lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
            for i, channel in enumerate(['l', 'a', 'b']):
                channel_data = lab[:, :, i].astype(np.float32)
                features.update({
                    f'lab_{channel}_mean': np.mean(channel_data),
                    f'lab_{channel}_std': np.std(channel_data),
                    f'lab_{channel}_range': np.max(channel_data) - np.min(channel_data)
                })
        
        except Exception as e:
            # Fill with default values if color analysis fails
            for channel in ['red', 'green', 'blue']:
                features.update({
                    f'{channel}_mean': 0.0, f'{channel}_std': 0.0,
                    f'{channel}_skew': 0.0, f'{channel}_kurtosis': 0.0,
                    f'{channel}_min': 0.0, f'{channel}_max': 0.0,
                    f'{channel}_range': 0.0
                })
            
            for channel in ['hue', 'saturation', 'value']:
                features.update({
                    f'hsv_{channel}_mean': 0.0,
                    f'hsv_{channel}_std': 0.0,
                    f'hsv_{channel}_entropy': 0.0
                })
            
            for channel in ['l', 'a', 'b']:
                features.update({
                    f'lab_{channel}_mean': 0.0,
                    f'lab_{channel}_std': 0.0,
                    f'lab_{channel}_range': 0.0
                })
        
        return features
    
    def extract_morphological_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features using mathematical morphology.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        try:
            # Threshold image
            gray_uint8 = (gray_image * 255).astype(np.uint8)
            _, binary = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_normalized = binary / 255.0
            
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            
            opened = opening(binary_normalized, kernel)
            closed = closing(binary_normalized, kernel)
            eroded = erosion(binary_normalized, kernel)
            dilated = dilation(binary_normalized, kernel)
            
            # Calculate differences
            opening_diff = np.sum(np.abs(binary_normalized - opened))
            closing_diff = np.sum(np.abs(binary_normalized - closed))
            erosion_diff = np.sum(np.abs(binary_normalized - eroded))
            dilation_diff = np.sum(np.abs(binary_normalized - dilated))
            
            features.update({
                'morpho_opening_diff': opening_diff,
                'morpho_closing_diff': closing_diff,
                'morpho_erosion_diff': erosion_diff,
                'morpho_dilation_diff': dilation_diff
            })
            
            # Shape properties
            if np.sum(binary_normalized) > 0:
                labeled_image = label(binary_normalized)
                props = regionprops(labeled_image)
                
                if props:
                    largest_region = max(props, key=lambda r: r.area)
                    
                    features.update({
                        'shape_area': largest_region.area,
                        'shape_perimeter': largest_region.perimeter,
                        'shape_eccentricity': largest_region.eccentricity,
                        'shape_solidity': largest_region.solidity,
                        'shape_extent': largest_region.extent,
                        'shape_euler_number': largest_region.euler_number
                    })
                    
                    # Convexity
                    hull = convex_hull_image(binary_normalized > 0)
                    if np.sum(hull) > 0:
                        convexity = np.sum(binary_normalized) / np.sum(hull)
                    else:
                        convexity = 0
                    
                    features['shape_convexity'] = convexity
                else:
                    # Default shape features
                    shape_defaults = {
                        'shape_area': 0, 'shape_perimeter': 0,
                        'shape_eccentricity': 0, 'shape_solidity': 0,
                        'shape_extent': 0, 'shape_euler_number': 0,
                        'shape_convexity': 0
                    }
                    features.update(shape_defaults)
            else:
                # Default shape features
                shape_defaults = {
                    'shape_area': 0, 'shape_perimeter': 0,
                    'shape_eccentricity': 0, 'shape_solidity': 0,
                    'shape_extent': 0, 'shape_euler_number': 0,
                    'shape_convexity': 0
                }
                features.update(shape_defaults)
        
        except Exception as e:
            # Fill with zeros if morphological analysis fails
            morpho_defaults = {
                'morpho_opening_diff': 0.0, 'morpho_closing_diff': 0.0,
                'morpho_erosion_diff': 0.0, 'morpho_dilation_diff': 0.0,
                'shape_area': 0, 'shape_perimeter': 0,
                'shape_eccentricity': 0, 'shape_solidity': 0,
                'shape_extent': 0, 'shape_euler_number': 0,
                'shape_convexity': 0
            }
            features.update(morpho_defaults)
        
        return features
    
    def extract_statistical_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features including higher-order moments.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        try:
            # Flatten image for statistical analysis
            pixels = gray_image.flatten()
            
            # Basic statistics
            features.update({
                'stat_mean': np.mean(pixels),
                'stat_std': np.std(pixels),
                'stat_var': np.var(pixels),
                'stat_min': np.min(pixels),
                'stat_max': np.max(pixels),
                'stat_range': np.max(pixels) - np.min(pixels),
                'stat_median': np.median(pixels),
                'stat_skew': skew(pixels),
                'stat_kurtosis': kurtosis(pixels)
            })
            
            # Percentiles
            percentiles = [10, 25, 75, 90]
            for p in percentiles:
                features[f'stat_percentile_{p}'] = np.percentile(pixels, p)
            
            # Entropy
            features['stat_entropy'] = shannon_entropy(gray_image)
            
            # Gradient statistics
            grad_x = np.gradient(gray_image, axis=1)
            grad_y = np.gradient(gray_image, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.update({
                'grad_mean': np.mean(gradient_magnitude),
                'grad_std': np.std(gradient_magnitude),
                'grad_max': np.max(gradient_magnitude)
            })
        
        except Exception as e:
            # Fill with default values if statistical analysis fails
            stat_defaults = {
                'stat_mean': 0.0, 'stat_std': 0.0, 'stat_var': 0.0,
                'stat_min': 0.0, 'stat_max': 0.0, 'stat_range': 0.0,
                'stat_median': 0.0, 'stat_skew': 0.0, 'stat_kurtosis': 0.0,
                'stat_entropy': 0.0, 'grad_mean': 0.0, 'grad_std': 0.0,
                'grad_max': 0.0
            }
            
            for p in [10, 25, 75, 90]:
                stat_defaults[f'stat_percentile_{p}'] = 0.0
            
            features.update(stat_defaults)
        
        return features
    
    def extract_all_features(self, image: np.ndarray, max_features: int = None) -> np.ndarray:
        """
        Extract all complex features from an image.
        
        Args:
            image: Input image (RGB or BGR)
            max_features: Maximum number of features to return. If None, returns all features.
                          If set to 121, returns exactly 121 features (for model compatibility).
            
        Returns:
            Feature vector as numpy array with max_features if specified
        """
        # Preprocess image
        rgb_image, gray_image = self.preprocess_image(image)
        
        # Extract all feature types
        all_features = {}
        
        # 1. Advanced HOG features (most important based on notebook)
        hog_features = self.extract_advanced_hog_features(gray_image)
        for i, feat in enumerate(hog_features):
            all_features[f'hog_{i}'] = feat
        
        # 2. Wavelet features
        wavelet_features = self.extract_wavelet_features(gray_image)
        all_features.update(wavelet_features)
        
        # 3. Texture features
        texture_features = self.extract_texture_features(gray_image)
        all_features.update(texture_features)
        
        # 4. Gabor features
        gabor_features = self.extract_gabor_features(gray_image)
        all_features.update(gabor_features)
        
        # 5. Edge features
        edge_features = self.extract_edge_features(gray_image)
        all_features.update(edge_features)
        
        # 6. Color features
        color_features = self.extract_color_features(rgb_image)
        all_features.update(color_features)
        
        # 7. Morphological features
        morpho_features = self.extract_morphological_features(gray_image)
        all_features.update(morpho_features)
        
        # 8. Statistical features
        stat_features = self.extract_statistical_features(gray_image)
        all_features.update(stat_features)
          # Convert to numpy array
        feature_vector = np.array(list(all_features.values()), dtype=np.float32)
        
        # Handle any NaN or infinite values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
          # Limit to max_features if specified
        if max_features is not None:
            original_size = len(feature_vector)
            if original_size > max_features:
                # Keep the most important features (first ones)
                feature_vector = feature_vector[:max_features]
                # Don't log this every time - only if the reduction is large
                if original_size > max_features * 1.5:
                    print(f"Reduced feature vector from {original_size} to {max_features} features")
            elif original_size < max_features:
                # Pad with zeros if we don't have enough features
                padded_vector = np.zeros(max_features, dtype=np.float32)
                padded_vector[:original_size] = feature_vector
                feature_vector = padded_vector
                print(f"Padded feature vector from {original_size} to {max_features} features")
        
        return feature_vector


def create_feature_extractor(feature_count: int = 121) -> ComplexFeatureExtractor:
    """
    Create and return a complex feature extractor instance.
    
    Args:
        feature_count: Number of features to extract. Default is 121 for model compatibility.
                       Set to None to get all available features.
    
    Returns:
        ComplexFeatureExtractor instance
    """
    extractor = ComplexFeatureExtractor()
    
    # Test the extractor to verify feature count
    if feature_count is not None:
        # Create a small test image
        test_image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        features = extractor.extract_all_features(test_image, max_features=feature_count)
        if len(features) != feature_count:
            print(f"Warning: Feature extractor returned {len(features)} features, expected {feature_count}")
    
    return extractor


# Example usage and testing
if __name__ == "__main__":
    # Test the feature extractor
    extractor = ComplexFeatureExtractor()
    
    # Create a dummy image for testing
    test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Extract features
    features = extractor.extract_all_features(test_image)
    
    print(f"Extracted {len(features)} complex features")
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature statistics: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
