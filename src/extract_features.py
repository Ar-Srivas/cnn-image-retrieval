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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        features = model(img)

    return features.numpy().flatten()