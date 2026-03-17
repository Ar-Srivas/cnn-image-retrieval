import os
import pickle
import numpy as np
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

    files = os.listdir(DATASET_PATH)

    for file in tqdm(files):
        if not file.endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(DATASET_PATH, file)

        try:
            img = preprocess_image(path)
            feat = combine_features(img)
            features[file] = feat
        except Exception as e:
            print(f"Skipping {file}: {e}")

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)

    print("Done. Features saved.")

if __name__ == "__main__":
    main()