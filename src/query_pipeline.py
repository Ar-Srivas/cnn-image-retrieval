"""
Query Image Pipeline
Handles preprocessing and feature extraction for uploaded/query images.
"""

import numpy as np
from typing import Union, Tuple
from pathlib import Path

from src.preprocess import preprocess_image, get_histogram
from src.extract_features import get_cnn_features


def process_query_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Process a query image and extract combined features.
    
    Args:
        image_path: Path to the query image
        
    Returns:
        Combined feature vector (histogram + CNN features)
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image cannot be processed
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        # Preprocess the image
        img = preprocess_image(str(image_path))
        
        # Extract combined features
        hist = get_histogram(img)
        cnn = get_cnn_features(img)
        combined_features = np.concatenate([hist, cnn])
        
        return combined_features
    
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {str(e)}")


def batch_process_query_images(image_paths: list) -> dict:
    """
    Process multiple query images in batch.
    
    Args:
        image_paths: List of paths to query images
        
    Returns:
        Dictionary mapping filenames to feature vectors
    """
    features = {}
    
    for path in image_paths:
        path = Path(path)
        try:
            feat = process_query_image(path)
            features[path.name] = feat
        except Exception as e:
            print(f"Warning: Skipping {path.name}: {e}")
    
    return features


def get_query_features_from_array(img_array: np.ndarray) -> np.ndarray:
    """
    Extract features from an already-loaded image array (e.g., from web upload).
    
    Args:
        img_array: Numpy array representing the image (BGR format, any size)
        
    Returns:
        Combined feature vector
        
    Raises:
        ValueError: If image array is invalid
    """
    import cv2
    
    if not isinstance(img_array, np.ndarray):
        raise ValueError(f"Image must be numpy array, got {type(img_array).__name__}")
    
    if img_array.size == 0:
        raise ValueError("Image array is empty")
    
    # Ensure proper format
    if len(img_array.shape) != 3:
        raise ValueError("Image must be 3-dimensional (H, W, C)")
    
    if img_array.shape[2] not in [3, 4]:
        raise ValueError(f"Image must have 3 or 4 channels, got {img_array.shape[2]}")
    
    # Resize and blur (same as preprocess_image)
    img = cv2.resize(img_array, (224, 224))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Extract features
    hist = get_histogram(img)
    cnn = get_cnn_features(img)
    
    combined = np.concatenate([hist, cnn])
    
    if np.isnan(combined).any() or np.isinf(combined).any():
        raise ValueError("Feature extraction produced NaN or Inf values")
    
    return combined


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.query_pipeline <image_path>")
        sys.exit(1)
    
    query_path = sys.argv[1]
    
    print(f"Processing query image: {query_path}")
    features = process_query_image(query_path)
    print(f"Extracted features: shape={features.shape}, dtype={features.dtype}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
