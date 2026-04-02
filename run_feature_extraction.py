import os
import pickle
import numpy as np
import random
from tqdm import tqdm

from src.preprocess import preprocess_image, get_histogram
from src.extract_features import get_cnn_features

def combine_features(img):
    hist = get_histogram(img)
    cnn = get_cnn_features(img)
    return np.concatenate([hist, cnn])

DATASET_PATH = "data/myntradataset/images"
FEATURES_PATH = "features/image_features.pkl"

def main():
    features = {}

    # Get all valid image files
    all_files = os.listdir(DATASET_PATH)
    image_files = [f for f in all_files if f.endswith((".jpg", ".png", ".jpeg"))]
    
    # Randomly select up to 5000 images
    num_images = min(5000, len(image_files))
    selected_files = random.sample(image_files, num_images)
    
    print(f"Processing {num_images} randomly selected images...")

    for file in tqdm(selected_files):
        path = os.path.join(DATASET_PATH, file)

        try:
            img = preprocess_image(path)
            feat = combine_features(img)
            features[file] = feat
        except Exception as e:
            print(f"Skipping {file}: {e}")

    # Create features directory if it doesn't exist
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)

    print("Done. Features saved.")

if __name__ == "__main__":
    main()