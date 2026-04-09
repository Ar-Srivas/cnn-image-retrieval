# Usage Guide: CNN Image Retrieval System

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
uv sync

# Verify installation
python -c "import torch; import torchvision; print('✓ Setup complete')"
```

### 2. Extract Features (Already Done)
```bash
# Your features are already extracted in features/image_features.pkl
# To re-extract (takes ~10-15 minutes for 5000 images):
python run_feature_extraction.py
```

### 3. Run System Evaluation
```bash
# Quick evaluation (recommended to start)
python generate_report.py --queries 50

# Full evaluation
python generate_report.py --queries 100

# With model comparison (WARNING: Takes 1-2 hours!)
python generate_report.py --compare-models --queries 50
```

---

## What You've Implemented

### ✅ Query Image Pipeline (`src/query_pipeline.py`)
- Process uploaded/query images
- Extract combined features (histogram + CNN)
- Support for file paths and numpy arrays
- Batch processing capability

**Example Usage:**
```python
from src.query_pipeline import process_query_image

# Process a query image
features = process_query_image("path/to/image.jpg")
print(f"Feature vector shape: {features.shape}")
```

### ✅ Evaluation Metrics (`src/evaluate.py`)
- **Precision@K**: How many of top K results are relevant
- **Recall@K**: What fraction of relevant images are in top K
- **Average Precision (AP)**: Quality of ranking for single query
- **Mean Average Precision (mAP)**: Overall system quality

**Example Usage:**
```python
from src.evaluate import precision_at_k, recall_at_k

retrieved = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]
relevant = ["img1.jpg", "img3.jpg", "img7.jpg"]

p5 = precision_at_k(retrieved, relevant, 5)  # 2/5 = 0.40
r5 = recall_at_k(retrieved, relevant, 5)     # 2/3 = 0.67
```

### ✅ Model Comparison (`src/model_comparison.py`)
Compare different CNN architectures:
- **ResNet50**: Deep, accurate (2048-dim features)
- **ResNet18**: Lighter version (512-dim)
- **VGG16**: Classic architecture (4096-dim)
- **EfficientNet-B0**: Efficient and modern (1280-dim)
- **MobileNetV2**: Fast, mobile-friendly (1280-dim)

**Example Usage:**
```bash
python -m src.model_comparison
```

### ✅ Evaluation Report Generator (`generate_report.py`)
Creates comprehensive reports:
- Text report: `features/evaluation_report.txt`
- JSON report: `features/evaluation_report.json`
- Raw pickle: `features/evaluation_results.pkl`

---

## Understanding Your Results

### Current Results (from evaluation_report.txt)

**Search Method Comparison:**
- **Euclidean Distance**: mAP = 0.0015 (slightly better)
- **Cosine Similarity**: mAP = 0.0013
- **Winner**: Euclidean Distance

**Performance:**
- Average search time: ~24-32 ms
- Precision@5: ~12% (1 in 8 results is relevant)
- Recall@5: ~0.12% (finding very few relevant items)

**Why Low Scores?**
The low scores are due to:
1. Using pseudo-categories based on image ID ranges
2. No actual semantic labels in your dataset
3. Generic feature extraction without fine-tuning

For **real evaluation**, you need:
- Actual category labels (e.g., "shirts", "pants", "dresses")
- Ground truth relevance annotations
- Or use a labeled dataset

---

## Demo: Search Similar Images

```bash
# Test with any image from your dataset
python demo_search.py data/myntradataset/images/56011.jpg 5 cosine
```

**Output:**
```
============================================================
RESULTS (Cosine Similarity):
============================================================
 1. 56011.jpg                              | Score: 1.0000
 2. 33895.jpg                              | Score: 0.9823
 3. 18292.jpg                              | Score: 0.9789
 4. 15831.jpg                              | Score: 0.9756
 5. 7730.jpg                               | Score: 0.9745
============================================================
```

---

## Next Steps for Your Report

### 1. Document Your Implementation ✅
You've implemented:
- Query image preprocessing pipeline
- Feature extraction for uploaded images
- Connection to similarity search system
- Comprehensive evaluation metrics (Precision@K, Recall@K, mAP)
- Model comparison framework

### 2. Run Full Model Comparison (Optional)
```bash
# WARNING: This takes 1-2 hours!
python -m src.model_comparison
```

This will:
- Extract features using 5 different CNN models
- Evaluate each model
- Generate comparison report

### 3. Create Your Report Document

**Include:**
- System overview and architecture
- Implementation details (query pipeline, evaluation)
- Evaluation results (include the reports in `features/`)
- Metrics explanation (Precision@K, Recall@K, mAP)
- Performance comparison (Cosine vs Euclidean)
- Model comparison results (if you ran it)
- Conclusions and recommendations

**Key Findings to Report:**
1. **Query Pipeline**: Successfully processes images and extracts 2560-dim features
2. **Evaluation Framework**: Implemented standard IR metrics (P@K, R@K, mAP)
3. **Search Methods**: Euclidean distance slightly outperforms cosine (mAP: 0.0015 vs 0.0013)
4. **Performance**: ~25ms average search time on 5000 images
5. **Scalability**: System can handle real-time queries

---

## Files Generated

```
features/
├── image_features.pkl              # Original ResNet50 features (51 MB)
├── evaluation_report.txt           # Human-readable report
├── evaluation_report.json          # Machine-readable results
├── evaluation_results.pkl          # Raw evaluation data
└── model_comparison_results.pkl    # (if you ran model comparison)

src/
├── query_pipeline.py              # Your query processing code
├── evaluate.py                     # Your evaluation metrics
└── model_comparison.py            # Your model comparison code
```

---

## Troubleshooting

**Issue**: Import errors
```bash
# Solution: Ensure you're in the project root
cd /Users/shubht/Desktop/cnn-image-retrieval
uv sync
```

**Issue**: Dataset not found
```bash
# Your dataset should be at:
data/myntradataset/images/

# Check if it exists:
ls data/myntradataset/images/ | head
```

**Issue**: Out of memory during model comparison
```bash
# Reduce the number of images:
# Edit src/model_comparison.py, line 220
# Change max_images=1000 to max_images=500
```

---

## API Reference

### Query Pipeline
```python
from src.query_pipeline import process_query_image, get_query_features_from_array

# From file
features = process_query_image("image.jpg")

# From numpy array (e.g., web upload)
import cv2
img = cv2.imread("image.jpg")
features = get_query_features_from_array(img)
```

### Evaluation
```python
from src.evaluate import (
    precision_at_k,
    recall_at_k,
    average_precision,
    evaluate_retrieval_system
)
```

### Model Comparison
```python
from src.model_comparison import (
    ResNet50Extractor,
    VGG16Extractor,
    EfficientNetB0Extractor,
    compare_models
)
```

---

## Summary

✅ **Query Pipeline**: Built and tested  
✅ **Performance Evaluation**: Implemented metrics (P@K, R@K, mAP)  
✅ **Search Comparison**: Evaluated Cosine vs Euclidean  
✅ **Model Framework**: Ready to compare different CNNs  
✅ **Reports**: Auto-generated comprehensive reports  

**Your system is complete and ready for demonstration!**
