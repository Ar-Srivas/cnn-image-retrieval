import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_cnn_features(img):
    if img is None or img.size == 0:
        raise ValueError("Invalid image: None or empty")
    
    if len(img.shape) != 3:
        raise ValueError(f"Image must be 3D (H,W,C), got shape {img.shape}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        features = model(img)

    features_np = features.numpy().flatten()
    
    if np.isnan(features_np).any() or np.isinf(features_np).any():
        raise ValueError("Feature extraction produced NaN or Inf values")
    
    return features_np