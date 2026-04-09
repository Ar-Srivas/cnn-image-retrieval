"""
Compare Different CNN Feature Extractors
Implements multiple CNN architectures for feature extraction comparison.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import pickle
from pathlib import Path
from typing import Dict, Callable
from tqdm import tqdm
import os

from src.preprocess import get_histogram
from src.evaluate import evaluate_retrieval_system, print_evaluation_report
from src.similarity_search import search_cosine


class FeatureExtractor:
    """Base class for feature extractors."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extract CNN features from an image."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0)
        
        with torch.no_grad():
            features = self.model(img_tensor)
        
        return features.numpy().flatten()
    
    def get_combined_features(self, img: np.ndarray) -> np.ndarray:
        """Combine histogram and CNN features."""
        hist = get_histogram(img)
        cnn = self.extract_features(img)
        return np.concatenate([hist, cnn])


class ResNet50Extractor(FeatureExtractor):
    """ResNet50 feature extractor."""
    
    def __init__(self):
        super().__init__("ResNet50")
        print(f"Loading {self.model_name}...")
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()


class ResNet18Extractor(FeatureExtractor):
    """ResNet18 feature extractor (lighter version)."""
    
    def __init__(self):
        super().__init__("ResNet18")
        print(f"Loading {self.model_name}...")
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()


class VGG16Extractor(FeatureExtractor):
    """VGG16 feature extractor."""
    
    def __init__(self):
        super().__init__("VGG16")
        print(f"Loading {self.model_name}...")
        base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Extract features from the last fully connected layer
        self.model = torch.nn.Sequential(
            base_model.features,
            base_model.avgpool,
            torch.nn.Flatten(),
            *list(base_model.classifier.children())[:-1]
        )
        self.model.eval()


class EfficientNetB0Extractor(FeatureExtractor):
    """EfficientNet-B0 feature extractor (efficient and accurate)."""
    
    def __init__(self):
        super().__init__("EfficientNet-B0")
        print(f"Loading {self.model_name}...")
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Remove classifier to get feature vector
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()


class MobileNetV2Extractor(FeatureExtractor):
    """MobileNetV2 feature extractor (lightweight and fast)."""
    
    def __init__(self):
        super().__init__("MobileNetV2")
        print(f"Loading {self.model_name}...")
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()


def extract_features_with_model(
    extractor: FeatureExtractor,
    dataset_path: str = "data/myntradataset/images",
    max_images: int = 5000
) -> Dict[str, np.ndarray]:
    """
    Extract features from dataset using a specific model.
    
    Args:
        extractor: Feature extractor instance
        dataset_path: Path to image dataset
        max_images: Maximum number of images to process
        
    Returns:
        Dictionary mapping filenames to feature vectors
    """
    features = {}
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Get image files
    all_files = os.listdir(dataset_path)
    image_files = [f for f in all_files if f.endswith((".jpg", ".png", ".jpeg"))]
    image_files = image_files[:max_images]
    
    print(f"\nExtracting features using {extractor.model_name}...")
    print(f"Processing {len(image_files)} images...")
    
    from src.preprocess import preprocess_image
    
    for file in tqdm(image_files):
        path = os.path.join(dataset_path, file)
        
        try:
            img = preprocess_image(path)
            feat = extractor.get_combined_features(img)
            features[file] = feat
        except Exception as e:
            print(f"Skipping {file}: {e}")
    
    return features


def compare_models(
    dataset_path: str = "data/myntradataset/images",
    max_images: int = 1000,
    num_eval_queries: int = 50
) -> Dict[str, Dict]:
    """
    Compare different CNN models for feature extraction.
    
    Args:
        dataset_path: Path to image dataset
        max_images: Number of images to process
        num_eval_queries: Number of queries for evaluation
        
    Returns:
        Comparison results for all models
    """
    # Define models to compare
    extractors = [
        ResNet50Extractor(),
        ResNet18Extractor(),
        VGG16Extractor(),
        EfficientNetB0Extractor(),
        MobileNetV2Extractor(),
    ]
    
    results = {}
    
    for extractor in extractors:
        print(f"\n{'='*60}")
        print(f"Testing {extractor.model_name}")
        print(f"{'='*60}")
        
        # Extract features
        features = extract_features_with_model(
            extractor, dataset_path, max_images
        )
        
        # Save features
        output_path = Path(f"features/{extractor.model_name.lower()}_features.pkl")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved to {output_path}")
        
        # Evaluate
        print(f"\nEvaluating {extractor.model_name}...")
        metrics = evaluate_retrieval_system(
            features, search_cosine, num_queries=num_eval_queries
        )
        
        # Store results
        results[extractor.model_name] = {
            'metrics': metrics,
            'feature_dim': list(features.values())[0].shape[0] if features else 0,
            'num_images': len(features)
        }
    
    return results


def print_model_comparison_report(results: Dict[str, Dict]):
    """
    Print comprehensive comparison report.
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON REPORT")
    print("="*70)
    
    # Summary table
    print("\nSUMMARY TABLE:")
    print("-" * 70)
    print(f"{'Model':<20} {'mAP':<10} {'P@5':<10} {'R@5':<10} {'Feat Dim':<12}")
    print("-" * 70)
    
    for model_name, data in results.items():
        metrics = data['metrics']
        feat_dim = data['feature_dim']
        print(f"{model_name:<20} {metrics['mAP']:<10.4f} "
              f"{metrics.get('precision@5', 0):<10.4f} "
              f"{metrics.get('recall@5', 0):<10.4f} "
              f"{feat_dim:<12}")
    
    print("-" * 70)
    
    # Detailed metrics
    print("\nDETAILED METRICS:")
    for model_name, data in results.items():
        metrics = data['metrics']
        print(f"\n{model_name}:")
        print(f"  Feature Dimension: {data['feature_dim']}")
        print(f"  Images Processed:  {data['num_images']}")
        print(f"  mAP:              {metrics['mAP']:.4f}")
        print(f"  Avg Search Time:  {metrics['avg_search_time']*1000:.2f} ms")
        print(f"  Precision@5:      {metrics.get('precision@5', 0):.4f}")
        print(f"  Recall@5:         {metrics.get('recall@5', 0):.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['mAP'])
    fastest_model = min(results.items(), key=lambda x: x[1]['metrics']['avg_search_time'])
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print(f"  Best Accuracy:  {best_model[0]} (mAP: {best_model[1]['metrics']['mAP']:.4f})")
    print(f"  Fastest Search: {fastest_model[0]} ({fastest_model[1]['metrics']['avg_search_time']*1000:.2f} ms)")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    # Check if dataset exists
    dataset_path = "data/myntradataset/images"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the dataset is in the correct location.")
        sys.exit(1)
    
    # Run comparison
    print("Starting model comparison...")
    print("This may take several minutes depending on your hardware.\n")
    
    results = compare_models(
        dataset_path=dataset_path,
        max_images=1000,  # Adjust based on your needs
        num_eval_queries=50
    )
    
    # Print report
    print_model_comparison_report(results)
    
    # Save comparison results
    output_path = Path("features/model_comparison_results.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nComparison results saved to {output_path}")
