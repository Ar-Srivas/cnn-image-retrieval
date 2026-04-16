import cv2
import numpy as np

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}. File may not exist or be corrupted.")
    img = cv2.resize(img, (224, 224))
    img = cv2.GaussianBlur(img, (5,5), 0)

    
    return img




def get_histogram(img):
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256]*3)
    return hist.flatten()